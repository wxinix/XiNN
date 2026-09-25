// SPDX-License-Identifier: BSD-3-Clause
// Reductions: sum and mean, over all elements or along one axis.
//
//   sum(x)       rank 0: the total
//   sum<1>(m)    rank R-1: for a matrix, one sum per row
//
// The axis is a template argument: it decides the result's shape, and the
// compiler rejects an axis that the tensor does not have.
#pragma once

#include <cstddef>
#include <limits>

#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/ops.hpp"

namespace xinn {

inline constexpr std::size_t all_axes = std::numeric_limits<std::size_t>::max();

namespace ops {

template <std::size_t Axis, std::size_t R> struct RepeatAxis;

template <std::size_t Axis = all_axes>
struct Sum {
    template <std::size_t... R>
    static constexpr std::size_t rank = Axis == all_axes ? 0 : (R + ...) - 1;

    template <class X>
    Shape<rank<X::rank>> shape(const X& x) const {
        Shape<rank<X::rank>> out;
        if constexpr (Axis != all_axes)
            for (std::size_t d = 0, o = 0; d < X::rank; ++d)
                if (d != Axis) out.dims[o++] = x.shape()[d];
        return out;
    }

    template <class T, std::size_t R>
    Tensor<T, rank<R>> eval(const Tensor<T, R>& x) const {
        if constexpr (Axis == all_axes) {
            T total{};
            for (T v : x.flat()) total += v;
            return Tensor<T, 0>(Shape<0>{}, total);
        } else {
            // View x as (outer, n, inner), with n the extent of Axis:
            // out[o, i] = sum over j of x[o, j, i].
            std::size_t outer = 1, inner = 1;
            for (std::size_t d = 0; d < Axis; ++d) outer *= x.shape()[d];
            for (std::size_t d = Axis + 1; d < R; ++d) inner *= x.shape()[d];
            const std::size_t n = x.shape()[Axis];

            Tensor<T, rank<R>> out(shape(x));
            auto src = x.flat();
            auto dst = out.mut_flat();
            for (std::size_t o = 0; o < outer; ++o)
                for (std::size_t j = 0; j < n; ++j)
                    for (std::size_t i = 0; i < inner; ++i) dst[o * inner + i] += src[(o * n + j) * inner + i];
            return out;
        }
    }

    // Every input element contributed once to its sum, so each receives the
    // gradient of that sum: g repeated back along the reduced axis.
    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& a) const {
        if constexpr (Axis == all_axes) return expand(g, a.shape());
        else return make_expr(RepeatAxis<Axis, std::remove_cvref_t<decltype(a)>::rank>{a.shape()}, g);
    }
};

// The inverse shape change of Sum<Axis>: repeat g along a new axis Axis,
// giving `target`, a shape of rank R.
template <std::size_t Axis, std::size_t R>
struct RepeatAxis {
    Shape<R> target;

    template <std::size_t...> static constexpr std::size_t rank = R;

    template <class G>
    Shape<R> shape(const G&) const { return target; }

    template <class T>
    Tensor<T, R> eval(const Tensor<T, R - 1>& g) const {
        std::size_t outer = 1, inner = 1;
        for (std::size_t d = 0; d < Axis; ++d) outer *= target[d];
        for (std::size_t d = Axis + 1; d < R; ++d) inner *= target[d];
        const std::size_t n = target[Axis];

        Tensor<T, R> out(target);
        auto src = g.flat();
        auto dst = out.mut_flat();
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t i = 0; i < inner; ++i) dst[(o * n + j) * inner + i] = src[o * inner + i];
        return out;
    }
};

// Sum over the first K dimensions: (a, b, c, d) with K = 2 gives (c, d).
// This undoes broadcasting, which repeats a tensor along leading dimensions.
template <std::size_t K>
struct SumLeading {
    template <std::size_t... R>
    static constexpr std::size_t rank = (R + ...) - K;

    template <class X>
    Shape<X::rank - K> shape(const X& x) const {
        Shape<X::rank - K> out;
        for (std::size_t d = K; d < X::rank; ++d) out.dims[d - K] = x.shape()[d];
        return out;
    }

    template <class T, std::size_t R>
    Tensor<T, R - K> eval(const Tensor<T, R>& x) const {
        Tensor<T, R - K> out(shape(x));
        auto src = x.flat();
        auto dst = out.mut_flat();
        for (std::size_t i = 0; i < src.size(); ++i) dst[i % dst.size()] += src[i];
        return out;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& a) const { return expand(g, a.shape()); }
};

}  // namespace ops

// sum_leading<K>(x): sum over the first K dimensions.
template <std::size_t K, TensorArg X>
    requires(K <= std::remove_cvref_t<X>::rank)
auto sum_leading(X&& x) {
    if constexpr (K == 0) return std::remove_cvref_t<X>(std::forward<X>(x));
    else return make_expr(ops::SumLeading<K>{}, std::forward<X>(x));
}

template <std::size_t Axis = all_axes, TensorArg X>
    requires(Axis == all_axes || Axis < std::remove_cvref_t<X>::rank)
auto sum(X&& x) {
    return make_expr(ops::Sum<Axis>{}, std::forward<X>(x));
}

template <std::size_t Axis = all_axes, TensorArg X>
    requires(Axis == all_axes || Axis < std::remove_cvref_t<X>::rank)
auto mean(X&& x) {
    using T = typename std::remove_cvref_t<X>::value_type;
    const std::size_t n = Axis == all_axes ? x.shape().count() : x.shape()[Axis];
    return sum<Axis>(std::forward<X>(x)) * (T{1} / static_cast<T>(n));
}

}  // namespace xinn
