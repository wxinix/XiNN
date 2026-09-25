// SPDX-License-Identifier: BSD-3-Clause
// Operations for attention and transformers.
//
//   bmm<TA, TB>(a, b)        batched matrix product: (..., M, K) · (..., K, N) -> (..., M, N)
//   permute<P...>(x)         reorder the axes: out.shape[i] = x.shape[P[i]]
//   layer_norm(x, g, b)      normalize each row of the last axis, then scale by g and shift by b
//   embedding(table, ids)    rows of a (vocab, d) table picked by integer ids
//   take<Axis>(x, i)         one index along one axis: (B, T, d) with Axis 1 -> (B, d)
//   causal_mask<T>(n)        (n, n): 0 on and below the diagonal, a large negative number above
//   sinusoidal_positions<T>(n, d)   the positional encoding of Vaswani et al. (2017)
//
// Each is a structural operation (chapter 2) with gradient rules (chapter 3).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/gemm.hpp"
#include "xinn/ops.hpp"
#include "xinn/parallel.hpp"

namespace xinn {

namespace detail {

// (rows, cols) of op(X) for the last two extents of s.
template <bool Trans, std::size_t R>
std::pair<std::size_t, std::size_t> matrix_dims(const Shape<R>& s) {
    return Trans ? std::pair{s[R - 1], s[R - 2]} : std::pair{s[R - 2], s[R - 1]};
}

}  // namespace detail

template <bool TransA = false, bool TransB = false, TensorArg A, TensorArg B>
auto bmm(A&& a, B&& b);

namespace ops {

// ---- batched matrix product -------------------------------------------------------------------

// C[i] = op(A[i]) · op(B[i]) for every index i of the leading (batch) axes.
template <bool TransA = false, bool TransB = false>
struct BatchedMatMul {
    template <std::size_t... R> static constexpr std::size_t rank = std::max({R...});

    template <class A, class B>
    Shape<A::rank> shape(const A& a, const B& b) const {
        static_assert(A::rank == B::rank && A::rank >= 3, "bmm: operands of equal rank, at least 3");
        constexpr std::size_t R = A::rank;
        const auto [m, k]  = detail::matrix_dims<TransA>(a.shape());
        const auto [k2, n] = detail::matrix_dims<TransB>(b.shape());
        contract_assert(k == k2);   // inner dimensions must agree
        Shape<R> out = a.shape();
        for (std::size_t d = 0; d + 2 < R; ++d) contract_assert(a.shape()[d] == b.shape()[d]);
        out.dims[R - 2] = m;
        out.dims[R - 1] = n;
        return out;
    }

    template <class T, std::size_t R>
    Tensor<T, R> eval(const Tensor<T, R>& a, const Tensor<T, R>& b) const {
        const auto s = shape(a, b);
        const auto [m, k] = detail::matrix_dims<TransA>(a.shape());
        const std::size_t n = s[R - 1];
        const std::size_t batch = s.count() / (m * n);
        Tensor<T, R> out(s);   // zeroed: the kernel adds into it
        const T* pa = a.flat().data();
        const T* pb = b.flat().data();
        T* pc = out.mut_flat().data();
        parallel_for(batch, 1, [&](std::size_t i0, std::size_t i1) {   // batch items in parallel
            for (std::size_t i = i0; i < i1; ++i) {
                const T* A = pa + i * m * k;
                const T* B = pb + i * k * n;
                T* C = pc + i * m * n;
                if constexpr (simd_element<T>) {
                    detail::gemm<TransA, TransB>(m, n, k, A, B, C);
                } else {
                    for (std::size_t r = 0; r < m; ++r)
                        for (std::size_t p = 0; p < k; ++p) {
                            const T arp = TransA ? A[p * m + r] : A[r * k + p];
                            for (std::size_t c = 0; c < n; ++c) C[r * n + c] += arp * (TransB ? B[c * k + p] : B[p * n + c]);
                        }
                }
            }
        });
        return out;
    }

    // As for MatMul: with A' = op(A), B' = op(B) and C = A'·B',
    //   dA' = G·B'^T,   dB' = A'^T·G,
    // and a transposed operand takes the transpose of its gradient. Each case
    // is again one batched product, with the flags set accordingly.
    template <std::size_t I>
    auto grad(const auto& g, const auto&, const auto& a, const auto& b) const {
        if constexpr (I == 0) {
            if constexpr (!TransA) return bmm<false, !TransB>(g, b);   // G·op(B)^T
            else return bmm<TransB, true>(b, g);                       // (G·op(B)^T)^T = op(B)·G^T
        } else {
            if constexpr (!TransB) return bmm<!TransA, false>(a, g);   // op(A)^T·G
            else return bmm<true, TransA>(g, a);                       // (op(A)^T·G)^T = G^T·op(A)
        }
    }
};

// ---- permutation of axes ----------------------------------------------------------------------

template <std::size_t... P>
struct Permute {
    static constexpr std::size_t R = sizeof...(P);
    static constexpr std::array<std::size_t, R> perm{P...};

    // The inverse permutation: inverse[perm[i]] = i.
    static constexpr auto inverse = [] {
        std::array<std::size_t, R> inv{};
        for (std::size_t i = 0; i < R; ++i) inv[perm[i]] = i;
        return inv;
    }();

    static constexpr bool valid = [] {
        std::array<bool, R> seen{};
        for (std::size_t p : perm) {
            if (p >= R || seen[p]) return false;
            seen[p] = true;
        }
        return true;
    }();
    static_assert(valid, "permute: the axes must be a permutation of 0 .. rank-1");

    template <std::size_t...> static constexpr std::size_t rank = R;

    template <class X>
    Shape<R> shape(const X& x) const {
        static_assert(X::rank == R, "permute: one axis index per axis");
        Shape<R> out;
        for (std::size_t i = 0; i < R; ++i) out.dims[i] = x.shape()[perm[i]];
        return out;
    }

    template <class T>
    Tensor<T, R> eval(const Tensor<T, R>& x) const {
        const Shape<R> s = shape(x);
        std::array<std::size_t, R> in_stride{}, stride{};   // stride[i]: input step for output axis i
        in_stride[R - 1] = 1;
        for (std::size_t d = R - 1; d > 0; --d) in_stride[d - 1] = in_stride[d] * x.shape()[d];
        for (std::size_t i = 0; i < R; ++i) stride[i] = in_stride[perm[i]];

        auto out = Tensor<T, R>::uninitialized(s);
        const T* src = x.flat().data();
        T* dst = out.mut_flat().data();
        const std::size_t last = s[R - 1], rows = s.count() / std::max<std::size_t>(last, 1);
        parallel_for(rows, 256, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t r = r0; r < r1; ++r) {
                // The input offset of output row r: its index along axes 0 .. R-2.
                std::size_t off = 0, rest = r;
                for (std::size_t d = R - 1; d-- > 0;) {
                    off += (rest % s[d]) * stride[d];
                    rest /= s[d];
                }
                T* row = dst + r * last;
                for (std::size_t j = 0; j < last; ++j) row[j] = src[off + j * stride[R - 1]];
            }
        });
        return out;
    }

    // The gradient goes back through the inverse permutation.
    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto&) const {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return make_expr(Permute<inverse[I]...>{}, g);
        }(std::make_index_sequence<R>{});
    }
};

// ---- layer normalization ----------------------------------------------------------------------

// dx = r * (g - mean(g) - y * mean(g * y)) per row, where y = (x - mean) * r is
// the normalized output and r = 1 / sqrt(var + eps).
struct NormalizeGrad {
    double eps;

    template <std::size_t... R> static constexpr std::size_t rank = std::max({R...});
    template <class G, class Y, class X>
    Shape<X::rank> shape(const G&, const Y&, const X& x) const { return x.shape(); }

    template <class T, std::size_t R>
    Tensor<T, R> eval(const Tensor<T, R>& g, const Tensor<T, R>& y, const Tensor<T, R>& x) const {
        const std::size_t c = x.shape()[R - 1], rows = x.size() / c;
        auto out = Tensor<T, R>::uninitialized(x.shape());
        const T* gs = g.flat().data();
        const T* ys = y.flat().data();
        const T* xs = x.flat().data();
        T* dst = out.mut_flat().data();
        parallel_for(rows, 64, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t row = r0; row < r1; ++row) {
                const std::size_t o = row * c;
                double mu = 0, var = 0, gm = 0, gy = 0;
                for (std::size_t j = 0; j < c; ++j) mu += xs[o + j];
                mu /= double(c);
                for (std::size_t j = 0; j < c; ++j) {
                    var += (xs[o + j] - mu) * (xs[o + j] - mu);
                    gm += gs[o + j];
                    gy += double(gs[o + j]) * ys[o + j];
                }
                const double r = 1 / std::sqrt(var / double(c) + eps);
                gm /= double(c);
                gy /= double(c);
                for (std::size_t j = 0; j < c; ++j) dst[o + j] = T(r * (gs[o + j] - gm - ys[o + j] * gy));
            }
        });
        return out;
    }
};

// y = (x - mean) / sqrt(var + eps), each row of the last axis on its own.
struct Normalize {
    double eps = 1e-5;

    template <std::size_t... R> static constexpr std::size_t rank = (R + ...);
    template <class X>
    Shape<X::rank> shape(const X& x) const { return x.shape(); }

    template <class T, std::size_t R>
        requires(R >= 1)
    Tensor<T, R> eval(const Tensor<T, R>& x) const {
        const std::size_t c = x.shape()[R - 1], rows = x.size() / c;
        auto out = Tensor<T, R>::uninitialized(x.shape());
        const T* xs = x.flat().data();
        T* dst = out.mut_flat().data();
        parallel_for(rows, 64, [&](std::size_t r0, std::size_t r1) {
            for (std::size_t row = r0; row < r1; ++row) {
                const std::size_t o = row * c;
                double mu = 0, var = 0;
                for (std::size_t j = 0; j < c; ++j) mu += xs[o + j];
                mu /= double(c);
                for (std::size_t j = 0; j < c; ++j) var += (xs[o + j] - mu) * (xs[o + j] - mu);
                const double r = 1 / std::sqrt(var / double(c) + eps);
                for (std::size_t j = 0; j < c; ++j) dst[o + j] = T((xs[o + j] - mu) * r);
            }
        });
        return out;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto& y, const auto& x) const { return make_expr(NormalizeGrad{eps}, g, y, x); }
};

// ---- embedding ----------------------------------------------------------------------------------

// dTable: row ids[i] receives g's row i; rows used several times add up.
template <std::size_t R>
struct ScatterRows {
    Tensor<std::size_t, R> ids;
    std::size_t vocab;

    template <std::size_t...> static constexpr std::size_t rank = 2;
    template <class G>
    Shape<2> shape(const G& g) const { return Shape{vocab, g.shape()[R]}; }

    template <class T>
    Tensor<T, 2> eval(const Tensor<T, R + 1>& g) const {
        const std::size_t d = g.shape()[R];
        Tensor<T, 2> out(Shape{vocab, d});
        const T* src = g.flat().data();
        T* dst = out.mut_flat().data();
        const auto id = ids.flat();
        for (std::size_t i = 0; i < id.size(); ++i)
            for (std::size_t j = 0; j < d; ++j) dst[id[i] * d + j] += src[i * d + j];
        return out;
    }
};

// out[i..., :] = table[ids[i...], :]. The ids have shape Shape<R>; the result has rank R + 1.
template <std::size_t R>
struct Embed {
    Tensor<std::size_t, R> ids;

    template <std::size_t...> static constexpr std::size_t rank = R + 1;

    template <class Table>
    Shape<R + 1> shape(const Table& table) const {
        static_assert(Table::rank == 2, "embedding: the table is (vocab, d)");
        Shape<R + 1> out;
        for (std::size_t d = 0; d < R; ++d) out.dims[d] = ids.shape()[d];
        out.dims[R] = table.shape()[1];
        return out;
    }

    template <class T>
    Tensor<T, R + 1> eval(const Tensor<T, 2>& table) const {
        const std::size_t vocab = table.shape()[0], d = table.shape()[1];
        auto out = Tensor<T, R + 1>::uninitialized(shape(table));
        const T* src = table.flat().data();
        T* dst = out.mut_flat().data();
        const auto id = ids.flat();
        for (std::size_t i = 0; i < id.size(); ++i) {
            contract_assert(id[i] < vocab);
            std::copy_n(src + id[i] * d, d, dst + i * d);
        }
        return out;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& table) const {
        return make_expr(ScatterRows<R>{ids, table.shape()[0]}, g);
    }
};

// ---- one index along an axis -------------------------------------------------------------------

// The adjoint of Take: g placed at index `at` of axis Axis, zeros elsewhere.
template <std::size_t Axis, std::size_t R>
struct Untake {
    Shape<R> target;
    std::size_t at;

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
        const T* src = g.flat().data();
        T* dst = out.mut_flat().data();
        for (std::size_t o = 0; o < outer; ++o) std::copy_n(src + o * inner, inner, dst + (o * n + at) * inner);
        return out;
    }
};

template <std::size_t Axis>
struct Take {
    std::size_t at;

    template <std::size_t... R> static constexpr std::size_t rank = (R + ...) - 1;

    template <class X>
    Shape<X::rank - 1> shape(const X& x) const {
        contract_assert(at < x.shape()[Axis]);
        Shape<X::rank - 1> out;
        for (std::size_t d = 0, o = 0; d < X::rank; ++d)
            if (d != Axis) out.dims[o++] = x.shape()[d];
        return out;
    }

    template <class T, std::size_t R>
    Tensor<T, R - 1> eval(const Tensor<T, R>& x) const {
        std::size_t outer = 1, inner = 1;
        for (std::size_t d = 0; d < Axis; ++d) outer *= x.shape()[d];
        for (std::size_t d = Axis + 1; d < R; ++d) inner *= x.shape()[d];
        const std::size_t n = x.shape()[Axis];
        auto out = Tensor<T, R - 1>::uninitialized(shape(x));
        const T* src = x.flat().data();
        T* dst = out.mut_flat().data();
        for (std::size_t o = 0; o < outer; ++o) std::copy_n(src + (o * n + at) * inner, inner, dst + o * inner);
        return out;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto& x) const {
        constexpr std::size_t R = std::remove_cvref_t<decltype(x)>::rank;
        return make_expr(Untake<Axis, R>{x.shape(), at}, g);
    }
};

}  // namespace ops

// ---- functions ----------------------------------------------------------------------------------

template <bool TransA, bool TransB, TensorArg A, TensorArg B>
auto bmm(A&& a, B&& b) {
    static_assert(std::remove_cvref_t<A>::rank == std::remove_cvref_t<B>::rank && std::remove_cvref_t<A>::rank >= 3,
                  "bmm: operands of equal rank, at least 3");
    return make_expr(ops::BatchedMatMul<TransA, TransB>{}, std::forward<A>(a), std::forward<B>(b));
}

template <std::size_t... P, TensorArg X>
    requires(sizeof...(P) == std::remove_cvref_t<X>::rank)
auto permute(X&& x) {
    return make_expr(ops::Permute<P...>{}, std::forward<X>(x));
}

// Normalize each row of the last axis to mean 0 and variance 1.
template <TensorArg X>
    requires(std::remove_cvref_t<X>::rank >= 1)
auto normalize(X&& x, double eps = 1e-5) {
    return make_expr(ops::Normalize{eps}, std::forward<X>(x));
}

// Layer normalization (Ba, Kiros, Hinton 2016): normalize, then a learned
// gain and bias per feature. The last two steps are element-wise, so they
// fuse, and their gradients come from the rules of chapter 3.
template <TensorArg X, TensorArg G, TensorArg B>
    requires(std::remove_cvref_t<G>::rank == 1 && std::remove_cvref_t<B>::rank == 1)
auto layer_norm(X&& x, G&& gain, B&& bias, double eps = 1e-5) {
    return normalize(std::forward<X>(x), eps) * std::forward<G>(gain) + std::forward<B>(bias);
}

template <TensorArg Table, std::size_t R>
    requires(std::remove_cvref_t<Table>::rank == 2)
auto embedding(Table&& table, const Tensor<std::size_t, R>& ids) {
    return make_expr(ops::Embed<R>{ids}, std::forward<Table>(table));
}

template <std::size_t Axis, TensorArg X>
    requires(Axis < std::remove_cvref_t<X>::rank)
auto take(X&& x, std::size_t index) {
    return make_expr(ops::Take<Axis>{index}, std::forward<X>(x));
}

// Added to attention scores, it hides the future: position i sees 0..i only.
// The large negative number becomes exactly 0 after softmax's exp().
template <class T = float>
Matrix<T> causal_mask(std::size_t n) {
    Matrix<T> m(Shape{n, n});
    auto v = m.mut();
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) v[i, j] = T(-1e9);
    return m;
}

// PE[pos, 2i] = sin(pos / 10000^(2i/d)),  PE[pos, 2i+1] = cos(pos / 10000^(2i/d)).
template <class T = float>
Matrix<T> sinusoidal_positions(std::size_t n, std::size_t d) {
    Matrix<T> m(Shape{n, d});
    auto v = m.mut();
    for (std::size_t pos = 0; pos < n; ++pos)
        for (std::size_t j = 0; j < d; ++j) {
            const double angle = double(pos) / std::pow(10000.0, double(j - j % 2) / double(d));
            v[pos, j] = T(j % 2 == 0 ? std::sin(angle) : std::cos(angle));
        }
    return m;
}

}  // namespace xinn
