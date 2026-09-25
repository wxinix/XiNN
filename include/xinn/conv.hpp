// SPDX-License-Identifier: BSD-3-Clause
// Convolutional layers: conv2d, max_pool2d, and reshape.
//
// Images are rank-4 tensors in NCHW order: (batch, channels, height, width).
//
//   conv2d(x, w, b, stride, padding)   x (N, C, H, W), w (F, C, KH, KW), b (F)
//                                      -> (N, F, OH, OW)
//   max_pool2d<2>(x)                   (N, C, H, W) -> (N, C, H/2, W/2)
//   reshape(x, shape)                  the same elements, another shape
//
// Convolution is computed with "im2col": the receptive fields of one image
// are unrolled into the columns of a matrix, so the whole convolution
// becomes one matrix product with the chapter 6 kernel. The gradients are
// matrix products too (with the transposed operands), plus "col2im", which
// folds the columns back into an image.
#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/gemm.hpp"
#include "xinn/parallel.hpp"

namespace xinn {

namespace ops {

// ---- reshape ---------------------------------------------------------------------------------

template <std::size_t R2>
struct Reshape {
    Shape<R2> to;

    template <std::size_t...> static constexpr std::size_t rank = R2;

    template <class X>
    Shape<R2> shape(const X& x) const {
        contract_assert(x.shape().count() == to.count());
        return to;
    }

    template <class T, std::size_t R>
    Tensor<T, R2> eval(const Tensor<T, R>& x) const { return x.reshaped(to); }   // shares the buffer

    // The gradient flows back unchanged, in the operand's shape.
    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& a) const {
        constexpr std::size_t R = std::remove_cvref_t<decltype(a)>::rank;
        return make_expr(Reshape<R>{a.shape()}, g);
    }
};

}  // namespace ops

template <std::size_t R2, TensorArg X>
auto reshape(X&& x, const Shape<R2>& shape) {
    return make_expr(ops::Reshape<R2>{shape}, std::forward<X>(x));
}

namespace detail {

struct ConvGeometry {
    std::size_t N, C, H, W, F, KH, KW, OH, OW, stride, pad;

    static ConvGeometry of(const Shape<4>& x, const Shape<4>& w, std::size_t stride, std::size_t pad) {
        contract_assert(x[1] == w[1]);   // input channels must agree
        contract_assert(x[2] + 2 * pad >= w[2] && x[3] + 2 * pad >= w[3]);
        return {x[0], x[1], x[2], x[3], w[0], w[2], w[3],
                (x[2] + 2 * pad - w[2]) / stride + 1, (x[3] + 2 * pad - w[3]) / stride + 1, stride, pad};
    }
    std::size_t col_rows() const { return C * KH * KW; }
    std::size_t col_cols() const { return OH * OW; }
};

// col[(c, kh, kw)][(oh, ow)] = x[c][oh*s - p + kh][ow*s - p + kw], or 0 outside the image.
template <class T>
void im2col(const ConvGeometry& g, const T* x, T* col) {
    for (std::size_t c = 0; c < g.C; ++c)
        for (std::size_t kh = 0; kh < g.KH; ++kh)
            for (std::size_t kw = 0; kw < g.KW; ++kw) {
                T* row = col + ((c * g.KH + kh) * g.KW + kw) * g.col_cols();
                for (std::size_t oh = 0; oh < g.OH; ++oh) {
                    const long ih = long(oh * g.stride + kh) - long(g.pad);
                    for (std::size_t ow = 0; ow < g.OW; ++ow) {
                        const long iw = long(ow * g.stride + kw) - long(g.pad);
                        row[oh * g.OW + ow] = (ih >= 0 && ih < long(g.H) && iw >= 0 && iw < long(g.W))
                                                  ? x[(c * g.H + std::size_t(ih)) * g.W + std::size_t(iw)]
                                                  : T{0};
                    }
                }
            }
}

// The adjoint of im2col: add each column entry back to the pixel it came from.
template <class T>
void col2im(const ConvGeometry& g, const T* col, T* x) {
    for (std::size_t c = 0; c < g.C; ++c)
        for (std::size_t kh = 0; kh < g.KH; ++kh)
            for (std::size_t kw = 0; kw < g.KW; ++kw) {
                const T* row = col + ((c * g.KH + kh) * g.KW + kw) * g.col_cols();
                for (std::size_t oh = 0; oh < g.OH; ++oh) {
                    const long ih = long(oh * g.stride + kh) - long(g.pad);
                    if (ih < 0 || ih >= long(g.H)) continue;
                    for (std::size_t ow = 0; ow < g.OW; ++ow) {
                        const long iw = long(ow * g.stride + kw) - long(g.pad);
                        if (iw >= 0 && iw < long(g.W)) x[(c * g.H + std::size_t(ih)) * g.W + std::size_t(iw)] += row[oh * g.OW + ow];
                    }
                }
            }
}

}  // namespace detail

namespace ops {

// ---- convolution ------------------------------------------------------------------------------

// dL/dx for y = conv2d(x, w, b): per image, dcol = W^T . g, then col2im.
struct Conv2dGradInput {
    std::size_t stride, pad;
    Shape<4> x_shape;

    template <std::size_t...> static constexpr std::size_t rank = 4;
    template <class G, class Wt>
    Shape<4> shape(const G&, const Wt&) const { return x_shape; }

    template <class T>
    Tensor<T, 4> eval(const Tensor<T, 4>& g, const Tensor<T, 4>& w) const {
        const auto geo = detail::ConvGeometry::of(x_shape, w.shape(), stride, pad);
        Tensor<T, 4> dx(x_shape);
        const T* pw = w.flat().data();
        const T* pg = g.flat().data();
        T* pdx = dx.mut_flat().data();
        parallel_for(geo.N, 1, [&](std::size_t n0, std::size_t n1) {
            std::vector<T> dcol(geo.col_rows() * geo.col_cols());
            for (std::size_t n = n0; n < n1; ++n) {
                std::ranges::fill(dcol, T{0});
                detail::gemm<true, false>(geo.col_rows(), geo.col_cols(), geo.F, pw, pg + n * geo.F * geo.col_cols(),
                                          dcol.data());
                detail::col2im(geo, dcol.data(), pdx + n * geo.C * geo.H * geo.W);
            }
        });
        return dx;
    }
};

// dL/dw: the sum over images of g_n . col_n^T.
struct Conv2dGradWeight {
    std::size_t stride, pad;
    Shape<4> w_shape;

    template <std::size_t...> static constexpr std::size_t rank = 4;
    template <class G, class X>
    Shape<4> shape(const G&, const X&) const { return w_shape; }

    template <class T>
    Tensor<T, 4> eval(const Tensor<T, 4>& g, const Tensor<T, 4>& x) const {
        const auto geo = detail::ConvGeometry::of(x.shape(), w_shape, stride, pad);
        const std::size_t chunks = std::min(geo.N, hardware_threads());
        std::vector<std::vector<T>> partial(chunks, std::vector<T>(w_shape.count(), T{0}));
        const T* px = x.flat().data();
        const T* pg = g.flat().data();
        parallel_for(chunks, 1, [&](std::size_t c0, std::size_t c1) {
            std::vector<T> col(geo.col_rows() * geo.col_cols());
            for (std::size_t c = c0; c < c1; ++c)
                for (std::size_t n = c * geo.N / chunks; n < (c + 1) * geo.N / chunks; ++n) {
                    detail::im2col(geo, px + n * geo.C * geo.H * geo.W, col.data());
                    detail::gemm<false, true>(geo.F, geo.col_rows(), geo.col_cols(), pg + n * geo.F * geo.col_cols(),
                                              col.data(), partial[c].data());
                }
        });
        Tensor<T, 4> dw(w_shape);
        auto dst = dw.mut_flat();
        for (const auto& p : partial)
            for (std::size_t i = 0; i < p.size(); ++i) dst[i] += p[i];
        return dw;
    }
};

// Sum over batch and positions: (N, F, H, W) -> (F). The gradient of a per-channel bias.
struct ChannelSum {
    template <std::size_t...> static constexpr std::size_t rank = 1;
    template <class G>
    Shape<1> shape(const G& g) const { return Shape{g.shape()[1]}; }

    template <class T>
    Tensor<T, 1> eval(const Tensor<T, 4>& g) const {
        const auto s = g.shape();
        Tensor<T, 1> out(Shape{s[1]});
        auto src = g.flat();
        auto dst = out.mut_flat();
        const std::size_t plane = s[2] * s[3];
        for (std::size_t n = 0; n < s[0]; ++n)
            for (std::size_t f = 0; f < s[1]; ++f) {
                T total{};
                for (std::size_t i = 0; i < plane; ++i) total += src[(n * s[1] + f) * plane + i];
                dst[f] += total;
            }
        return out;
    }
};

struct Conv2d {
    std::size_t stride = 1, pad = 0;

    template <std::size_t...> static constexpr std::size_t rank = 4;

    template <class X, class Wt, class B>
    Shape<4> shape(const X& x, const Wt& w, const B& b) const {
        const auto geo = detail::ConvGeometry::of(x.shape(), w.shape(), stride, pad);
        contract_assert(b.shape()[0] == geo.F);
        return Shape{geo.N, geo.F, geo.OH, geo.OW};
    }

    template <class T>
    Tensor<T, 4> eval(const Tensor<T, 4>& x, const Tensor<T, 4>& w, const Tensor<T, 1>& b) const {
        const auto geo = detail::ConvGeometry::of(x.shape(), w.shape(), stride, pad);
        Tensor<T, 4> y(Shape{geo.N, geo.F, geo.OH, geo.OW});
        const T* px = x.flat().data();
        const T* pw = w.flat().data();
        const T* pb = b.flat().data();
        T* py = y.mut_flat().data();
        parallel_for(geo.N, 1, [&](std::size_t n0, std::size_t n1) {   // images in parallel
            std::vector<T> col(geo.col_rows() * geo.col_cols());
            for (std::size_t n = n0; n < n1; ++n) {
                detail::im2col(geo, px + n * geo.C * geo.H * geo.W, col.data());
                T* out = py + n * geo.F * geo.col_cols();
                detail::gemm<false, false>(geo.F, geo.col_cols(), geo.col_rows(), pw, col.data(), out);
                for (std::size_t f = 0; f < geo.F; ++f)
                    for (std::size_t i = 0; i < geo.col_cols(); ++i) out[f * geo.col_cols() + i] += pb[f];
            }
        });
        return y;
    }

    template <std::size_t I>
    auto grad(const auto& g, const auto&, const auto& x, const auto& w, const auto&) const {
        if constexpr (I == 0) return make_expr(Conv2dGradInput{stride, pad, x.shape()}, g, w);
        else if constexpr (I == 1) return make_expr(Conv2dGradWeight{stride, pad, w.shape()}, g, x);
        else return make_expr(ChannelSum{}, g);
    }
};

// ---- max pooling ------------------------------------------------------------------------------

// dL/dx: each window's gradient goes to the position of its maximum.
template <std::size_t K>
struct MaxPoolGrad {
    template <std::size_t...> static constexpr std::size_t rank = 4;
    template <class G, class X>
    Shape<4> shape(const G&, const X& x) const { return x.shape(); }

    template <class T>
    Tensor<T, 4> eval(const Tensor<T, 4>& g, const Tensor<T, 4>& x) const {
        const auto s = x.shape();
        const std::size_t OH = s[2] / K, OW = s[3] / K;
        Tensor<T, 4> dx(s);
        auto X = x.view();
        auto G = g.view();
        auto D = dx.mut();
        for (std::size_t n = 0; n < s[0]; ++n)
            for (std::size_t c = 0; c < s[1]; ++c)
                for (std::size_t oh = 0; oh < OH; ++oh)
                    for (std::size_t ow = 0; ow < OW; ++ow) {
                        std::size_t bh = oh * K, bw = ow * K;
                        for (std::size_t i = 0; i < K; ++i)
                            for (std::size_t j = 0; j < K; ++j)
                                if (X[n, c, oh * K + i, ow * K + j] > X[n, c, bh, bw]) bh = oh * K + i, bw = ow * K + j;
                        D[n, c, bh, bw] += G[n, c, oh, ow];
                    }
        return dx;
    }
};

template <std::size_t K>
struct MaxPool2d {
    template <std::size_t...> static constexpr std::size_t rank = 4;

    template <class X>
    Shape<4> shape(const X& x) const {
        contract_assert(x.shape()[2] % K == 0 && x.shape()[3] % K == 0);
        return Shape{x.shape()[0], x.shape()[1], x.shape()[2] / K, x.shape()[3] / K};
    }

    template <class T>
    Tensor<T, 4> eval(const Tensor<T, 4>& x) const {
        const auto s = shape(x);
        auto y = Tensor<T, 4>::uninitialized(s);
        auto X = x.view();
        auto Y = y.mut();
        for (std::size_t n = 0; n < s[0]; ++n)
            for (std::size_t c = 0; c < s[1]; ++c)
                for (std::size_t oh = 0; oh < s[2]; ++oh)
                    for (std::size_t ow = 0; ow < s[3]; ++ow) {
                        T m = std::numeric_limits<T>::lowest();
                        for (std::size_t i = 0; i < K; ++i)
                            for (std::size_t j = 0; j < K; ++j) m = std::max(m, X[n, c, oh * K + i, ow * K + j]);
                        Y[n, c, oh, ow] = m;
                    }
        return y;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& x) const { return make_expr(MaxPoolGrad<K>{}, g, x); }
};

}  // namespace ops

template <TensorArg X, TensorArg Wt, TensorArg B>
    requires(std::remove_cvref_t<X>::rank == 4 && std::remove_cvref_t<Wt>::rank == 4 &&
             std::remove_cvref_t<B>::rank == 1)
auto conv2d(X&& x, Wt&& w, B&& b, std::size_t stride = 1, std::size_t padding = 0) {
    return make_expr(ops::Conv2d{stride, padding}, std::forward<X>(x), std::forward<Wt>(w), std::forward<B>(b));
}

template <std::size_t K = 2, TensorArg X>
    requires(std::remove_cvref_t<X>::rank == 4)
auto max_pool2d(X&& x) {
    return make_expr(ops::MaxPool2d<K>{}, std::forward<X>(x));
}

}  // namespace xinn
