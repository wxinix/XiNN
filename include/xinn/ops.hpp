// SPDX-License-Identifier: BSD-3-Clause
// Element-wise operations: + - * /, unary minus, and math functions.
//
// Each operation is a small function object. Anything callable on single
// elements works as an operation, so user lambdas plug in through map().
//
// Some operators rewrite themselves at compile time when an operand's type
// makes the answer known:  x + zeros -> x,  x * ones -> x,  -(-x) -> x.
#pragma once

#include <cmath>
#include <string_view>
#include <type_traits>
#include <utility>

#include "xinn/constant.hpp"
#include "xinn/expr.hpp"

namespace xinn {

namespace ops {

// Gradient rules (chapter 3). For y = op(a0, a1, ...), grad<I>(g, y, a...)
// returns dLoss/da_I as a lazy expression, given g = dLoss/dy. The result
// has y's shape; the caller sums it down to a_I's shape if a_I was
// broadcast.

struct Add {
    static constexpr std::string_view symbol = "+";
    constexpr auto operator()(auto a, auto b) const { return a + b; }
    template <std::size_t I> auto grad(const auto& g, const auto&, const auto&...) const { return g; }
};
struct Sub {
    static constexpr std::string_view symbol = "-";
    constexpr auto operator()(auto a, auto b) const { return a - b; }
    template <std::size_t I> auto grad(const auto& g, const auto&, const auto&...) const {
        if constexpr (I == 0) return g;
        else return -g;
    }
};
struct Mul {
    static constexpr std::string_view symbol = "*";
    constexpr auto operator()(auto a, auto b) const { return a * b; }
    template <std::size_t I> auto grad(const auto& g, const auto&, const auto&... a) const {
        return g * a...[1 - I];   // d(a0*a1)/da0 = a1, and the other way round
    }
};
struct Div {
    static constexpr std::string_view symbol = "/";
    constexpr auto operator()(auto a, auto b) const { return a / b; }
    template <std::size_t I> auto grad(const auto& g, const auto& y, const auto&... a) const {
        if constexpr (I == 0) return g / a...[1];   // d(a/b)/da = 1/b
        else return -(g * y) / a...[1];             // d(a/b)/db = -a/b^2 = -y/b
    }
};
struct Neg {
    static constexpr std::string_view symbol = "-";
    constexpr auto operator()(auto a) const { return -a; }
    template <std::size_t> auto grad(const auto& g, const auto&, const auto&) const { return -g; }
};

// The math functions are called unqualified after `using std::exp;` and so
// on: for a float that is std::exp, for a SIMD vector ADL finds the std::simd
// overload. The same op then serves both the scalar and the vector loop.

struct Exp {
    auto operator()(auto a) const { using std::exp; return exp(a); }
    template <std::size_t> auto grad(const auto& g, const auto& y, const auto&) const { return g * y; }
};
struct Log {
    auto operator()(auto a) const { using std::log; return log(a); }
    template <std::size_t> auto grad(const auto& g, const auto&, const auto& a) const { return g / a; }
};
struct Sqrt {
    auto operator()(auto a) const { using std::sqrt; return sqrt(a); }
    template <std::size_t> auto grad(const auto& g, const auto& y, const auto&) const { return g / (y * 2); }
};
struct Tanh {
    auto operator()(auto a) const { using std::tanh; return tanh(a); }
    template <std::size_t> auto grad(const auto& g, const auto& y, const auto&) const { return g * (1 - y * y); }
};
struct Square {
    constexpr auto operator()(auto a) const { return a * a; }
    template <std::size_t> auto grad(const auto& g, const auto&, const auto& a) const { return g * a * 2; }
};
struct Sigmoid {
    auto operator()(auto a) const { using std::exp; return decltype(a){1} / (decltype(a){1} + exp(-a)); }
    template <std::size_t> auto grad(const auto& g, const auto& y, const auto&) const { return g * y * (1 - y); }
};

// ReLU passes the gradient where its input was positive, and blocks it elsewhere.
struct ReluGrad {
    constexpr auto operator()(auto g, auto a) const {
        if constexpr (std::is_arithmetic_v<decltype(a)>) {
            return a > decltype(a){0} ? g : decltype(g){0};
        } else {   // SIMD: a lane-wise select
            decltype(g) out(0);
            where(a > decltype(a)(0), out) = g;
            return out;
        }
    }
};
struct Relu {
    constexpr auto operator()(auto a) const { using std::max; return max(a, decltype(a){0}); }
    template <std::size_t> auto grad(const auto& g, const auto&, const auto& a) const {
        return make_expr(ReluGrad{}, g, a);
    }
};

// expand(x, shape): x repeated to a larger shape by broadcasting. The second
// operand only supplies the shape.
struct Expand {
    constexpr auto operator()(auto a, auto) const { return a; }
};

}  // namespace ops

template <class X>
concept Number = std::is_arithmetic_v<std::remove_cvref_t<X>>;

namespace detail {

template <class X> using bare = std::remove_cvref_t<X>;

// Operands of a binary operator: two tensors, or a tensor and a number.
template <class A, class B>
concept BinaryArgs = (TensorArg<A> && (TensorArg<B> || Number<B>)) || (Number<A> && TensorArg<B>);

// A number becomes a rank-0 constant of the other operand's element type.
template <class Other, class X>
auto lift(X&& x) {
    if constexpr (Number<X>) {
        using T = typename bare<Other>::value_type;
        return scalar(static_cast<T>(x));
    } else {
        return bare<X>(std::forward<X>(x));
    }
}

template <class X> concept ZerosLike = requires { bare<X>::fill_value; } && (bare<X>::fill_value == 0);
template <class X> concept OnesLike  = requires { bare<X>::fill_value; } && (bare<X>::fill_value == 1);

// "a already has the shape of a (op) b": a's rank is not smaller than b's.
// Broadcasting then yields a's shape, provided the shapes fit at all.
template <class A, class B>
bool keeps_shape(const A& a, const B& b) {
    return A::rank >= B::rank && broadcastable(a.shape(), b.shape());
}

template <class A, class B>
auto add(A a, B b) {
    if constexpr (ZerosLike<B> && A::rank >= B::rank) {
        contract_assert(keeps_shape(a, b));
        return a;
    } else if constexpr (ZerosLike<A> && B::rank >= A::rank) {
        contract_assert(keeps_shape(b, a));
        return b;
    } else {
        return make_expr(ops::Add{}, std::move(a), std::move(b));
    }
}

template <class A, class B>
auto mul(A a, B b) {
    using T = std::common_type_t<typename A::value_type, typename B::value_type>;
    if constexpr (ZerosLike<A> || ZerosLike<B>) {
        return zeros<T>(broadcast(a.shape(), b.shape()));
    } else if constexpr (OnesLike<B> && A::rank >= B::rank) {
        contract_assert(keeps_shape(a, b));
        return a;
    } else if constexpr (OnesLike<A> && B::rank >= A::rank) {
        contract_assert(keeps_shape(b, a));
        return b;
    } else {
        return make_expr(ops::Mul{}, std::move(a), std::move(b));
    }
}

template <class A>
auto neg(A a) {
    if constexpr (ExprOf<A, ops::Neg>) return a.template arg<0>();   // -(-x) -> x
    else if constexpr (ZerosLike<A>) return a;                       // -0 -> 0
    else return make_expr(ops::Neg{}, std::move(a));
}

template <class A, class B>
auto sub(A a, B b) {
    if constexpr (ZerosLike<B> && A::rank >= B::rank) {
        contract_assert(keeps_shape(a, b));
        return a;
    } else if constexpr (ZerosLike<A> && B::rank >= A::rank) {
        contract_assert(keeps_shape(b, a));
        return neg(std::move(b));
    } else {
        return make_expr(ops::Sub{}, std::move(a), std::move(b));
    }
}

template <class A, class B>
auto div(A a, B b) {
    if constexpr (OnesLike<B> && A::rank >= B::rank) {
        contract_assert(keeps_shape(a, b));
        return a;
    } else {
        return make_expr(ops::Div{}, std::move(a), std::move(b));
    }
}

}  // namespace detail

// ---- operators ---------------------------------------------------------------

template <class A, class B> requires detail::BinaryArgs<A, B>
auto operator+(A&& a, B&& b) {
    return detail::add(detail::lift<B>(std::forward<A>(a)), detail::lift<A>(std::forward<B>(b)));
}

template <class A, class B> requires detail::BinaryArgs<A, B>
auto operator-(A&& a, B&& b) {
    return detail::sub(detail::lift<B>(std::forward<A>(a)), detail::lift<A>(std::forward<B>(b)));
}

template <class A, class B> requires detail::BinaryArgs<A, B>
auto operator*(A&& a, B&& b) {
    return detail::mul(detail::lift<B>(std::forward<A>(a)), detail::lift<A>(std::forward<B>(b)));
}

template <class A, class B> requires detail::BinaryArgs<A, B>
auto operator/(A&& a, B&& b) {
    return detail::div(detail::lift<B>(std::forward<A>(a)), detail::lift<A>(std::forward<B>(b)));
}

template <TensorArg A>
auto operator-(A&& a) { return detail::neg(detail::bare<A>(std::forward<A>(a))); }

// ---- functions ---------------------------------------------------------------

template <TensorArg X> auto exp(X&& x)     { return make_expr(ops::Exp{}, std::forward<X>(x)); }
template <TensorArg X> auto log(X&& x)     { return make_expr(ops::Log{}, std::forward<X>(x)); }
template <TensorArg X> auto sqrt(X&& x)    { return make_expr(ops::Sqrt{}, std::forward<X>(x)); }
template <TensorArg X> auto tanh(X&& x)    { return make_expr(ops::Tanh{}, std::forward<X>(x)); }
template <TensorArg X> auto square(X&& x)  { return make_expr(ops::Square{}, std::forward<X>(x)); }
template <TensorArg X> auto relu(X&& x)    { return make_expr(ops::Relu{}, std::forward<X>(x)); }
template <TensorArg X> auto sigmoid(X&& x) { return make_expr(ops::Sigmoid{}, std::forward<X>(x)); }

// expand(x, shape): x broadcast to `shape`, as a lazy expression.
template <TensorArg X, std::size_t R>
auto expand(X&& x, const Shape<R>& shape) {
    using T = typename detail::bare<X>::value_type;
    return make_expr(ops::Expand{}, std::forward<X>(x), zeros<T>(shape));
}

// map(f, xs...): apply any element-wise callable. The operands broadcast.
//   map([](float a, float b) { return std::max(a, b); }, x, y)
template <class F, TensorArg... Xs>
    requires std::invocable<const F&, typename detail::bare<Xs>::value_type...>
auto map(F f, Xs&&... xs) {
    return make_expr(std::move(f), std::forward<Xs>(xs)...);
}

}  // namespace xinn
