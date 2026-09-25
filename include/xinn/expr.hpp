// SPDX-License-Identifier: BSD-3-Clause
// Expr<Op, Args...>: a lazy operation.
//
// Building an expression computes nothing but its shape. The work happens in
// eval(). Operands are stored by value; tensors copy shallowly, so this is
// cheap, and an expression stays valid after its inputs go out of scope.
//
// There are two kinds of operation:
//
//   element-wise  Op is callable on single elements: Op{}(1.0f, 2.0f).
//                 Rank and shape follow broadcasting. A tree of element-wise
//                 operations is evaluated in one fused loop, with no
//                 temporary tensors.
//
//   structural    anything else (matmul, transpose, sum). Op supplies
//                   template <size_t... R> static constexpr size_t rank;
//                   Shape<rank> shape(const Args&...) const;
//                   Tensor<T, rank> eval(const Tensor<...>&...) const;
//                 Its operands are evaluated first, then Op::eval runs.
//
// The fused loop runs on SIMD vectors when every op in the tree also works on
// std::simd values (chapter 6), and in parallel chunks when it is long.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/constant.hpp"
#include "xinn/parallel.hpp"
#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

template <class Op, class... Args>
concept ElementwiseOp = std::invocable<const Op&, typename Args::value_type...>;

namespace detail {

template <class Op, class... Args>
consteval std::size_t result_rank() {
    if constexpr (ElementwiseOp<Op, Args...>) return std::max({Args::rank...});
    else return Op::template rank<Args::rank...>;
}

template <class Op, class... Args>
struct result_value {
    using type = std::common_type_t<typename Args::value_type...>;
};

template <class Op, class... Args>
    requires ElementwiseOp<Op, Args...>
struct result_value<Op, Args...> {
    using type = std::remove_cvref_t<std::invoke_result_t<const Op&, typename Args::value_type...>>;
};

// ---- readers ---------------------------------------------------------------
//
// A reader maps a flat (row-major) index to an element. Fused evaluation
// builds one reader for the whole element-wise tree and calls it once per
// output element; the compiler inlines the lot into a single loop.
//
// Each reader can also load<V>(i): the V::size() elements starting at i as
// one SIMD vector V, so the same tree runs a vector at a time.

template <class T>
struct PtrReader {
    const T* p;
    T operator()(std::size_t i) const { return p[i]; }
    template <class V> V load(std::size_t i) const { return V(p + i, std::experimental::element_aligned); }
};

template <class T>
struct ValueReader {
    T v;
    T operator()(std::size_t) const { return v; }
    template <class V> V load(std::size_t) const { return V(v); }   // the same value in every lane
};

// For operands that are not element-wise: evaluate once, then read.
template <class T, std::size_t R>
struct OwningReader {
    Tensor<T, R> t;
    const T* p = t.flat().data();
    T operator()(std::size_t i) const { return p[i]; }
    template <class V> V load(std::size_t i) const { return V(p + i, std::experimental::element_aligned); }
};

// Broadcasting. Shapes align at the trailing end, so in row-major order a
// smaller operand simply repeats: element i of the result reads element
// i % n of an operand with n elements. `full` is true when the operand has
// all the result's elements, and no wrapping happens.
template <class Reader>
struct Wrap {
    Reader read;
    std::size_t n;
    bool full;

    auto operator()(std::size_t i) const { return read(i < n ? i : i % n); }

    template <class V>
    V load(std::size_t i) const {
        if (full) return read.template load<V>(i);
        if (n == 1) return V(read(0));
        const std::size_t j = i % n;
        if (j + V::size() <= n) return read.template load<V>(j);   // the vector does not wrap around
        return V([&](auto lane) { return read((i + lane) % n); });   // it does: one lane at a time
    }
};

template <class Op, class... Readers>
struct MapReader {
    [[no_unique_address]] Op op;
    std::tuple<Readers...> readers;
    auto operator()(std::size_t i) const {
        return std::apply([&](const auto&... r) { return op(r(i)...); }, readers);
    }
    template <class V>
    V load(std::size_t i) const {
        return std::apply([&](const auto&... r) { return V(op(r.template load<V>(i)...)); }, readers);
    }
};

// Can operand X join a vectorized loop over elements of type T? Leaves and
// structural expressions are read from memory, so yes; an element-wise
// expression only if it is vectorizable itself.
template <class X, class T>
consteval bool simd_operand() {
    if constexpr (!std::same_as<typename X::value_type, T>) return false;
    else if constexpr (requires { X::elementwise; }) return !X::elementwise || X::vectorizable;
    else return true;
}

// An element-wise op is vectorizable when it accepts SIMD vectors and returns
// one: arithmetic and std::exp and friends do; a lambda taking float does not.
template <class Op, class T, class... Args>
consteval bool vectorizable_op() {
    if constexpr (!simd_element<T> || !(simd_operand<Args, T>() && ...)) return false;
    else if constexpr (!std::invocable<const Op&, decltype((void)std::declval<Args>(), simd<T>{})...>) return false;
    else return std::same_as<std::remove_cvref_t<std::invoke_result_t<const Op&, decltype((void)std::declval<Args>(), simd<T>{})...>>, simd<T>>;
}

template <TensorLike X>
auto make_reader(const X& x) {
    using T = typename X::value_type;
    if constexpr (requires { x.reader(); }) return x.reader();                 // element-wise Expr
    else if constexpr (is_tensor_v<X>) return PtrReader<T>{x.flat().data()};    // Tensor
    else if constexpr (requires { x.value(); }) return ValueReader<T>{x.value()};  // constants
    else return OwningReader<T, X::rank>{x.eval()};                            // everything else
}

}  // namespace detail

template <class Op, TensorLike... Args>
    requires(sizeof...(Args) >= 1)
class Expr {
public:
    using op_type = Op;
    using value_type = typename detail::result_value<Op, Args...>::type;
    static constexpr std::size_t rank = detail::result_rank<Op, Args...>();
    static constexpr bool elementwise = ElementwiseOp<Op, Args...>;
    static constexpr std::size_t arity = sizeof...(Args);
    // Fused evaluation of this tree can run on SIMD vectors.
    static constexpr bool vectorizable = [] {
        if constexpr (elementwise) return detail::vectorizable_op<Op, value_type, Args...>();
        else return false;
    }();

    template <std::size_t I> using arg_type = Args...[I];

    explicit Expr(Op op, Args... args)
        : op_(std::move(op)), args_(std::move(args)...), shape_(compute_shape()) {}

    const Shape<rank>& shape() const noexcept { return shape_; }
    const Op& op() const noexcept { return op_; }
    const std::tuple<Args...>& args() const noexcept { return args_; }

    template <std::size_t I>
    const arg_type<I>& arg() const noexcept { return std::get<I>(args_); }

    Tensor<value_type, rank> eval() const {
        if constexpr (elementwise) {
            auto out = Tensor<value_type, rank>::uninitialized(shape_);   // every element is written below
            const auto read = reader();
            auto dst = out.mut_flat();
            parallel_for(dst.size(), config.grain, [&](std::size_t begin, std::size_t end) { fill(read, dst, begin, end); });
            return out;
        } else {
            return std::apply([&](const auto&... a) { return op_.eval(a.eval()...); }, args_);
        }
    }

    // The fused reader for this element-wise tree.
    auto reader() const
        requires elementwise
    {
        return std::apply([&](const auto&... a) {
            return detail::MapReader{op_, std::tuple{detail::Wrap{detail::make_reader(a), a.shape().count(),
                                                                  a.shape().count() == shape_.count()}...}};
        }, args_);
    }

private:
    // Elements [begin, end) of the result: a SIMD vector at a time where
    // possible, then the remainder one by one.
    template <class Reader>
    static void fill(const Reader& read, std::span<value_type> dst, std::size_t begin, std::size_t end) {
        std::size_t i = begin;
        if constexpr (vectorizable) {
            if (config.simd) {
                using V = simd<value_type>;
                for (; i + V::size() <= end; i += V::size())
                    read.template load<V>(i).copy_to(dst.data() + i, std::experimental::element_aligned);
            }
        }
        for (; i < end; ++i) dst[i] = static_cast<value_type>(read(i));
    }

    Shape<rank> compute_shape() const {
        return std::apply([&](const auto&... a) {
            if constexpr (elementwise) return broadcast(a.shape()...);
            else return op_.shape(a...);
        }, args_);
    }

    [[no_unique_address]] Op op_;
    std::tuple<Args...> args_;
    Shape<rank> shape_;
};

// Build an Expr from any operands, dropping references and cv-qualifiers.
template <class Op, TensorArg... Args>
auto make_expr(Op op, Args&&... args) {
    return Expr<Op, std::remove_cvref_t<Args>...>(std::move(op), std::forward<Args>(args)...);
}

// Is X an expression whose operation is Op?
template <class X, class Op>
concept ExprOf = requires { typename std::remove_cvref_t<X>::op_type; } &&
                 std::same_as<typename std::remove_cvref_t<X>::op_type, Op>;

// Evaluate anything tensor-like into a Tensor.
template <TensorArg X>
auto eval(const X& x) { return x.eval(); }

}  // namespace xinn
