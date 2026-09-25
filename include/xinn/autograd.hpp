// SPDX-License-Identifier: BSD-3-Clause
// Reverse-mode automatic differentiation over expression types.
//
//   Param W(...), b(...);
//   auto loss = mean(square(matmul(x, W) + b - y));
//   float value = backward(loss);   // W.grad() and b.grad() now hold dLoss/dW, dLoss/db
//
// backward() works in two passes:
//
//   1. record: evaluate the expression bottom-up and keep every intermediate
//      value that a gradient rule may need. The result is a *tape*: a tree of
//      Node and Leaf objects with the same shape as the expression type.
//   2. backprop: walk the tape top-down. At each node, the op's grad<I> rule
//      turns the gradient of the node's output into the gradient of operand
//      I. At a Param leaf, the gradient is accumulated.
//
// Which parts need gradients is known from the types alone. A subtree with
// no Param in it is evaluated fused, stored as one tensor, and never visited
// by backprop. The pruning happens at compile time.
//
// backward(y, seed) does the same for an expression y of any shape, seeded
// with a tensor of y's shape: a vector-Jacobian product.
#pragma once

#include <array>
#include <cstddef>
#include <meta>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xinn/constant.hpp"
#include "xinn/expr.hpp"
#include "xinn/param.hpp"
#include "xinn/reduce.hpp"

namespace xinn {

// ---- which expressions depend on parameters --------------------------------

template <class X>
struct depends_on_params : std::bool_constant<is_param_v<X>> {};

template <class Op, class... Args>
struct depends_on_params<Expr<Op, Args...>> : std::bool_constant<(depends_on_params<Args>::value || ...)> {};

template <class X>
concept Trainable = depends_on_params<std::remove_cvref_t<X>>::value;

namespace detail {

template <std::size_t N>
inline constexpr auto indices = [] {
    std::array<std::size_t, N> a{};
    for (std::size_t i = 0; i < N; ++i) a[i] = i;
    return a;
}();

// ---- the tape ------------------------------------------------------------------

// A leaf: a Param, a constant, a tensor, or a parameter-free subtree that has
// already been evaluated.
template <class L>
struct Leaf {
    static constexpr bool trainable = is_param_v<L>;
    L leaf;

    // The value the gradient rules see: for a Param, its current tensor.
    decltype(auto) value() const {
        if constexpr (is_param_v<L>) return leaf.tensor();
        else return (leaf);
    }
};

// An operation on the path to some Param: its op, its output, its operands.
template <class Op, class Out, class Kids>
struct Node {
    static constexpr bool trainable = true;
    [[no_unique_address]] Op op;
    Out out;
    Kids kids;   // std::tuple of Leaf / Node

    const Out& value() const { return out; }
};

template <TensorLike X>
auto record(const X& x) {
    if constexpr (is_param_v<X>) {
        return Leaf<X>{x};
    } else if constexpr (!Trainable<X>) {
        if constexpr (requires { typename X::op_type; })
            return Leaf<Tensor<typename X::value_type, X::rank>>{x.eval()};   // fused, then frozen
        else
            return Leaf<X>{x};                                                // tensor or constant
    } else {
        auto kids = std::apply([](const auto&... a) { return std::tuple{record(a)...}; }, x.args());
        auto out = std::apply([&](const auto&... k) { return make_expr(x.op(), k.value()...).eval(); }, kids);
        return Node<typename X::op_type, decltype(out), decltype(kids)>{x.op(), std::move(out), std::move(kids)};
    }
}

// ---- gradient rules --------------------------------------------------------------

template <class Op>
consteval std::string_view no_grad_message() {
    return std::define_static_string(
        std::string("xinn: operation `") + std::string(std::meta::display_string_of(^^Op)) +
        "` has no gradient rule, but a Param is among its operands. "
        "Give the op a grad<I>(g, y, operands...) member, or keep Params out of it.");
}

template <std::size_t I, class Op, class G, class Y, class... A>
auto grad_of(const Op& op, const G& g, const Y& y, const A&... a) {
    if constexpr (requires { op.template grad<I>(g, y, a...); })
        return op.template grad<I>(g, y, a...);
    else
        static_assert(false, no_grad_message<Op>());
}

// A broadcast operand of rank RK received a gradient of the output's larger
// rank. Broadcasting repeated the operand along the leading dimensions, so
// its gradient is the sum over them.
template <std::size_t RK, class G>
auto unbroadcast(G&& g) {
    constexpr std::size_t RG = std::remove_cvref_t<G>::rank;
    static_assert(RG >= RK);
    return sum_leading<RG - RK>(std::forward<G>(g));
}

// ---- backpropagation ----------------------------------------------------------------

template <class L, class G>
void backprop(const Leaf<L>& leaf, const G& g) {
    if constexpr (is_param_v<L>) leaf.leaf.accumulate(g);
}

template <class Op, class Out, class... Kids, class G>
void backprop(const Node<Op, Out, std::tuple<Kids...>>& node, const G& g_in) {
    const Out g = g_in;   // evaluate the incoming gradient once
    template for (constexpr std::size_t I : indices<sizeof...(Kids)>) {
        using Kid = Kids...[I];
        if constexpr (Kid::trainable) {
            const Kid& kid = std::get<I>(node.kids);
            auto gi = std::apply([&](const auto&... k) { return grad_of<I>(node.op, g, node.out, k.value()...); },
                                 node.kids);
            backprop(kid, unbroadcast<std::remove_cvref_t<decltype(kid.value())>::rank>(std::move(gi)));
        }
    }
}

}  // namespace detail

// Differentiate a scalar loss: add dLoss/dP to P.grad() for every Param P it
// depends on, and return the loss value.
template <TensorArg L>
    requires(std::remove_cvref_t<L>::rank == 0)
auto backward(const L& loss) {
    static_assert(Trainable<L>, "xinn: backward() on a loss that depends on no Param");
    using T = typename std::remove_cvref_t<L>::value_type;
    const auto tape = detail::record(loss);
    detail::backprop(tape, Tensor<T, 0>(Shape<0>{}, T{1}));   // dLoss/dLoss = 1
    return tape.value()();
}

// Vector-Jacobian product, for an expression y of any rank: given a seed s
// of y's shape, add  sum_i s_i dy_i/dP  to P.grad() for every Param P, and
// return y's value. With s = dLoss/dy this continues a backward pass that
// was cut at y (recurrent networks, chapter 9). backward(loss) is the case
// of a rank-0 y with seed 1.
template <TensorArg Y, TensorArg S>
    requires(std::remove_cvref_t<Y>::rank == std::remove_cvref_t<S>::rank) &&
            std::same_as<typename std::remove_cvref_t<Y>::value_type, typename std::remove_cvref_t<S>::value_type>
auto backward(const Y& y, const S& seed) {
    // Checked in the body: GCC 16.1 crashes (ICE in check_noexcept_r) on a
    // pre() of this template, whose return type is deduced.
    contract_assert(y.shape() == seed.shape());
    static_assert(Trainable<Y>, "xinn: backward() on an expression that depends on no Param");
    const auto tape = detail::record(y);
    detail::backprop(tape, seed);
    return Tensor<typename std::remove_cvref_t<Y>::value_type, std::remove_cvref_t<Y>::rank>(tape.value());
}

}  // namespace xinn
