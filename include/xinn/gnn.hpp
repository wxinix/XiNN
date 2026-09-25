// SPDX-License-Identifier: BSD-3-Clause
// Graph neural networks: sparse matrices, spmm, and graph convolution.
//
//   auto A = SparseMatrix<float>::from_edges(n, n, edges);   // CSR, constant
//   auto A_hat = gcn_normalize(symmetrize(A));              // D^-1/2 (A + I) D^-1/2
//   Tensor<float, 3> y = spmm(A_hat, x);                    // x (B, n, F) -> (B, n, F)
//   GraphConv<{.activation = Activation::relu}> gc(A_hat, in, out, rng);
//   NodeDense<> out(in, out, rng);                           // a Dense layer at every node
//
// A graph of n nodes is stored as its adjacency matrix in compressed sparse
// row (CSR) form: for row i, the entries columns[offsets[i] .. offsets[i+1])
// with their values. A road network has a handful of neighbours per node, so
// a product A·X costs O(nnz · F) instead of O(n² · F).
//
// A sparse matrix here is a constant, like a tensor that is not a Param: it
// is never trained, and its gradient is never needed. The gradient of
// spmm(A, X) with respect to X is Aᵀ·G, so each matrix keeps its transpose,
// built once when the matrix is.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include "xinn/conv.hpp"
#include "xinn/expr.hpp"
#include "xinn/init.hpp"
#include "xinn/layers.hpp"
#include "xinn/linalg.hpp"
#include "xinn/module.hpp"
#include "xinn/parallel.hpp"
#include "xinn/param.hpp"

namespace xinn {

// One entry of a sparse matrix, or one weighted edge of a graph: row `from`,
// column `to`.
struct Edge {
    std::size_t from, to;
    double weight = 1;
};

// ---- the CSR sparse matrix -----------------------------------------------------------------

template <class T = float>
class SparseMatrix {
public:
    using value_type = T;

    SparseMatrix() = default;

    // Entries (from, to) = weight. Repeated entries are added; within a row,
    // columns are sorted.
    static SparseMatrix from_edges(std::size_t rows, std::size_t cols, std::span<const Edge> edges) {
        auto csr = build(rows, cols, edges, false);
        auto tr = build(cols, rows, edges, true);
        return SparseMatrix(std::move(csr), std::move(tr));
    }
    static SparseMatrix from_edges(std::size_t rows, std::size_t cols, const std::vector<Edge>& edges) {
        return from_edges(rows, cols, std::span<const Edge>(edges));
    }

    static SparseMatrix identity(std::size_t n) {
        std::vector<Edge> e;
        for (std::size_t i = 0; i < n; ++i) e.push_back({i, i, 1.0});
        return from_edges(n, n, e);
    }

    std::size_t rows() const noexcept { return a_ ? a_->rows : 0; }
    std::size_t cols() const noexcept { return a_ ? a_->cols : 0; }
    std::size_t nnz() const noexcept { return a_ ? a_->columns.size() : 0; }
    bool empty() const noexcept { return a_ == nullptr; }

    std::span<const std::size_t> offsets() const { return a_->offsets; }   // rows() + 1 of them
    std::span<const std::size_t> columns() const { return a_->columns; }
    std::span<const T> values() const { return a_->values; }

    // Aᵀ, without copying: the two halves swap places.
    SparseMatrix transposed() const { return SparseMatrix(t_, a_); }

    // A(i, j); zero if the entry is not stored.
    T operator()(std::size_t i, std::size_t j) const {
        contract_assert(i < rows() && j < cols());
        const auto begin = a_->columns.begin() + std::ptrdiff_t(a_->offsets[i]);
        const auto end = a_->columns.begin() + std::ptrdiff_t(a_->offsets[i + 1]);
        const auto it = std::lower_bound(begin, end, j);
        return it != end && *it == j ? a_->values[std::size_t(it - a_->columns.begin())] : T{0};
    }

    // All stored entries, row by row.
    std::vector<Edge> entries() const {
        std::vector<Edge> out;
        out.reserve(nnz());
        for (std::size_t i = 0; i < rows(); ++i)
            for (std::size_t k = a_->offsets[i]; k < a_->offsets[i + 1]; ++k)
                out.push_back({i, a_->columns[k], double(a_->values[k])});
        return out;
    }

    // The dense matrix, for tests and small examples.
    Matrix<T> dense() const {
        Matrix<T> m(Shape{rows(), cols()});
        auto M = m.mut();
        for (const auto& e : entries()) M[e.from, e.to] = T(e.weight);
        return m;
    }

private:
    struct Csr {
        std::size_t rows = 0, cols = 0;
        std::vector<std::size_t> offsets, columns;
        std::vector<T> values;
    };

    SparseMatrix(std::shared_ptr<const Csr> a, std::shared_ptr<const Csr> t) : a_(std::move(a)), t_(std::move(t)) {}

    // Counting sort by row, then sort each row by column and merge repeats.
    static std::shared_ptr<const Csr> build(std::size_t rows, std::size_t cols, std::span<const Edge> edges,
                                            bool transpose) {
        Csr c{rows, cols, std::vector<std::size_t>(rows + 1, 0), {}, {}};
        for (const auto& e : edges) {
            const std::size_t r = transpose ? e.to : e.from, k = transpose ? e.from : e.to;
            contract_assert(r < rows && k < cols);
            ++c.offsets[r + 1];
        }
        std::partial_sum(c.offsets.begin(), c.offsets.end(), c.offsets.begin());
        std::vector<std::pair<std::size_t, T>> slots(edges.size());
        std::vector<std::size_t> next(c.offsets.begin(), c.offsets.end() - 1);
        for (const auto& e : edges) {
            const std::size_t r = transpose ? e.to : e.from, k = transpose ? e.from : e.to;
            slots[next[r]++] = {k, T(e.weight)};
        }
        std::vector<std::size_t> merged(rows + 1, 0);
        for (std::size_t r = 0; r < rows; ++r) {
            auto first = slots.begin() + std::ptrdiff_t(c.offsets[r]), last = slots.begin() + std::ptrdiff_t(c.offsets[r + 1]);
            std::sort(first, last, [](const auto& a, const auto& b) { return a.first < b.first; });
            for (auto it = first; it != last; ++it) {
                if (merged[r + 1] > 0 && c.columns.back() == it->first) c.values.back() += it->second;
                else c.columns.push_back(it->first), c.values.push_back(it->second), ++merged[r + 1];
            }
        }
        std::partial_sum(merged.begin(), merged.end(), merged.begin());
        c.offsets = std::move(merged);
        return std::make_shared<const Csr>(std::move(c));
    }

    std::shared_ptr<const Csr> a_, t_;   // A and Aᵀ
};

// ---- building graphs -------------------------------------------------------------------------

// max(A, Aᵀ) entry by entry: a directed graph made undirected.
template <class T>
SparseMatrix<T> symmetrize(const SparseMatrix<T>& a) {
    contract_assert(a.rows() == a.cols());
    std::vector<Edge> e;
    for (const auto& x : a.entries()) {
        const double back = double(a(x.to, x.from));
        if (x.from == x.to) e.push_back(x);
        else if (back == 0 || x.weight > back || (x.weight == back && x.from < x.to)) {
            e.push_back(x);   // keep one copy of each pair, the larger weight
            e.push_back({x.to, x.from, x.weight});
        }
    }
    return SparseMatrix<T>::from_edges(a.rows(), a.cols(), e);
}

// The GCN propagation matrix of Kipf and Welling (2017):
//   Â = D^-1/2 (A + I) D^-1/2,   D = diag of the row sums of A + I.
// Every node keeps itself in the average (the self-loop), and the symmetric
// scaling keeps the spectrum of Â within [-1, 1], so stacking layers neither
// blows up nor fades out the signal.
template <class T>
SparseMatrix<T> gcn_normalize(const SparseMatrix<T>& a) {
    const std::size_t n = a.rows();
    contract_assert(n == a.cols());
    auto e = a.entries();
    for (std::size_t i = 0; i < n; ++i) e.push_back({i, i, 1.0});
    std::vector<double> degree(n, 0.0);
    for (const auto& x : e) degree[x.from] += x.weight;
    for (auto& x : e) x.weight /= std::sqrt(degree[x.from] * degree[x.to]);
    return SparseMatrix<T>::from_edges(n, n, e);
}

// The Gaussian kernel of DCRNN (Li et al., 2018): an edge of length d gets
// weight exp(-(d / sigma)²), and edges below `threshold` are dropped. With
// sigma = 0 the standard deviation of the lengths is used.
inline std::vector<Edge> gaussian_kernel(std::vector<Edge> distances, double sigma = 0, double threshold = 0.1) {
    if (sigma == 0 && !distances.empty()) {
        double m = 0, s = 0;
        for (const auto& e : distances) m += e.weight;
        m /= double(distances.size());
        for (const auto& e : distances) s += (e.weight - m) * (e.weight - m);
        sigma = std::sqrt(s / double(distances.size()));
    }
    std::vector<Edge> out;
    for (auto e : distances) {
        e.weight = std::exp(-(e.weight / sigma) * (e.weight / sigma));
        if (e.weight >= threshold) out.push_back(e);
    }
    return out;
}

// ---- spmm ----------------------------------------------------------------------------------

namespace detail {

// For every batch item b and row i of A:
//   y[b, i, 0:F] = (self ? self[b, i, 0:F] : 0) + Σ_k A[i, k] · x[b, k, 0:F]
// Rows of x, y and self are xw, yw and sw elements wide (F or more): the
// product may read or write one block of the feature axis. Rows are
// independent, so they run in parallel, at least about 16k multiply-adds
// per task.
template <class T>
void sparse_rows(const SparseMatrix<T>& a, std::size_t batch, std::size_t F, const T* x, std::size_t xw, T* y,
                 std::size_t yw, const T* self = nullptr, std::size_t sw = 0) {
    const std::size_t rows = a.rows(), cols = a.cols();
    const auto off = a.offsets();
    const auto col = a.columns();
    const auto val = a.values();
    const std::size_t per_row = std::max<std::size_t>(1, F * (a.nnz() / std::max<std::size_t>(rows, 1)));
    parallel_for(batch * rows, std::max<std::size_t>(1, (1 << 14) / per_row), [&](std::size_t r0, std::size_t r1) {
        for (std::size_t r = r0; r < r1; ++r) {
            const std::size_t b = r / rows, i = r % rows;
            T* out = y + r * yw;
            if (self) std::copy_n(self + r * sw, F, out);
            else std::fill_n(out, F, T{0});
            const T* in = x + b * cols * xw;
            for (std::size_t k = off[i]; k < off[i + 1]; ++k) {
                const T v = val[k];
                const T* xr = in + col[k] * xw;
                for (std::size_t f = 0; f < F; ++f) out[f] += v * xr[f];
            }
        }
    });
}

}  // namespace detail

namespace ops {

// Y = A · X for each batch item: X (A.cols, F) -> (A.rows, F), or
// X (B, A.cols, F) -> (B, A.rows, F).
template <class T>
struct SpMM {
    SparseMatrix<T> a;

    template <std::size_t... R> static constexpr std::size_t rank = (R + ...);

    template <class X>
    auto shape(const X& x) const {
        auto s = x.shape();
        constexpr std::size_t R = std::remove_cvref_t<X>::rank;
        contract_assert(s[R - 2] == a.cols());
        s.dims[R - 2] = a.rows();
        return s;
    }

    template <class U, std::size_t R>
    Tensor<T, R> eval(const Tensor<U, R>& x) const {
        static_assert(std::same_as<U, T>, "xinn: spmm needs a sparse matrix of the tensor's element type");
        const auto s = shape(x);
        const std::size_t F = s[R - 1];
        auto y = Tensor<T, R>::uninitialized(s);
        detail::sparse_rows(a, s.count() / (a.rows() * F), F, x.flat().data(), F, y.mut_flat().data(), F);
        return y;
    }

    // dL/dX = Aᵀ · G, batch item by batch item.
    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto&) const { return make_expr(SpMM{a.transposed()}, g); }
};

// dL/dZ for Y = graph_mix(A, Z): [Aᵀ · G | G].
template <class T>
struct GraphMixGrad {
    SparseMatrix<T> at;   // Aᵀ

    template <std::size_t... R> static constexpr std::size_t rank = (R + ...);

    template <class G>
    auto shape(const G& g) const {
        auto s = g.shape();
        constexpr std::size_t R = std::remove_cvref_t<G>::rank;
        s.dims[R - 2] = at.rows();
        s.dims[R - 1] *= 2;
        return s;
    }

    template <std::size_t R>
    Tensor<T, R> eval(const Tensor<T, R>& g) const {
        contract_assert(at.rows() == at.cols());
        const auto s = shape(g);
        const std::size_t F = g.shape()[R - 1], rows = s.count() / (2 * F);
        auto dz = Tensor<T, R>::uninitialized(s);
        T* p = dz.mut_flat().data();
        const T* pg = g.flat().data();
        detail::sparse_rows(at, rows / at.rows(), F, pg, F, p, 2 * F);
        for (std::size_t r = 0; r < rows; ++r) std::copy_n(pg + r * F, F, p + r * 2 * F + F);
        return dz;
    }
};

// Y = A · Z[..., :F] + Z[..., F:], for Z of 2F features: the first half is
// mixed over the graph, the second half is each node's own term.
template <class T>
struct GraphMix {
    SparseMatrix<T> a;

    template <std::size_t... R> static constexpr std::size_t rank = (R + ...);

    template <class Z>
    auto shape(const Z& z) const {
        auto s = z.shape();
        constexpr std::size_t R = std::remove_cvref_t<Z>::rank;
        contract_assert(a.rows() == a.cols() && s[R - 2] == a.cols() && s[R - 1] % 2 == 0);
        s.dims[R - 1] /= 2;
        return s;
    }

    template <class U, std::size_t R>
    Tensor<T, R> eval(const Tensor<U, R>& z) const {
        static_assert(std::same_as<U, T>, "xinn: graph_mix needs a sparse matrix of the tensor's element type");
        const auto s = shape(z);
        const std::size_t F = s[R - 1];
        auto y = Tensor<T, R>::uninitialized(s);
        const T* pz = z.flat().data();
        detail::sparse_rows(a, s.count() / (a.rows() * F), F, pz, 2 * F, y.mut_flat().data(), F, pz + F, 2 * F);
        return y;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto&, const auto&) const { return make_expr(GraphMixGrad<T>{a.transposed()}, g); }
};

}  // namespace ops

template <class T, TensorArg X>
    requires(std::remove_cvref_t<X>::rank == 2 || std::remove_cvref_t<X>::rank == 3)
auto spmm(const SparseMatrix<T>& a, X&& x) {
    return make_expr(ops::SpMM<T>{a}, std::forward<X>(x));
}

// A · Z[..., :F] + Z[..., F:] for Z (N, 2F) or (B, N, 2F) -> (.., N, F).
template <class T, TensorArg Z>
    requires(std::remove_cvref_t<Z>::rank == 2 || std::remove_cvref_t<Z>::rank == 3)
auto graph_mix(const SparseMatrix<T>& a, Z&& z) {
    return make_expr(ops::GraphMix<T>{a}, std::forward<Z>(z));
}

// ---- layers ----------------------------------------------------------------------------------

// A dense layer on the last axis of (B, N, F): the same weights at every node.
// With no graph mixing, a stack of these is the "per-node MLP".
template <DenseOptions Opt = {}, class T = float>
struct NodeDense : Module {
    Dense<Opt, T> dense;

    NodeDense(std::size_t in, std::size_t out, Rng& rng) : dense(in, out, rng) {}

    template <TensorArg X>
    auto forward(X&& x) const {
        const auto s = x.shape();
        constexpr std::size_t R = std::remove_cvref_t<X>::rank;
        if constexpr (R == 2) return dense(std::forward<X>(x));
        else {
            const std::size_t out = dense.weight.shape()[1];
            return reshape(dense(reshape(std::forward<X>(x), Shape{s[0] * s[1], s[2]})), Shape{s[0], s[1], out});
        }
    }
};

struct GraphConvOptions {
    bool bias = true;
    Activation activation = Activation::none;
    // A separate weight for the node's own features ("root weight"):
    //   H' = activation(H · W_self + Â · H · W + b)
    // Kipf's layer averages a node with its neighbours through one W; with
    // this, the node's own signal is not diluted by the average.
    bool self_weight = false;
};

// Graph convolution (Kipf and Welling, 2017): H' = activation(Â · H · W + b).
// H is (N, F) or (B, N, F); W acts on the feature axis, Â mixes the nodes.
// Â is a constant member, not a parameter: the module walks skip it.
template <GraphConvOptions Opt = {}, class T = float>
struct GraphConv : Module {
    static constexpr GraphConvOptions options = Opt;

    SparseMatrix<T> adjacency;
    // (in, out), or with self_weight (in, 2 out): [W | W_self] side by side,
    // so that H is multiplied once, and used once in the expression.
    Param<T, 2> weight;
    [[no_unique_address]] std::conditional_t<Opt.bias, Param<T, 1>, Nothing> bias;

    GraphConv(SparseMatrix<T> a_hat, std::size_t in, std::size_t out, Rng& rng)
        : adjacency(std::move(a_hat)), weight(init(in, out, rng)), bias(make_bias(out)) {}

    template <TensorArg X>
    auto forward(X&& x) const {
        contract_assert(x.shape()[std::remove_cvref_t<X>::rank - 2] == adjacency.cols());
        auto z = on_features(std::forward<X>(x), weight);   // H W first: fewer features to mix
        if constexpr (Opt.self_weight) return finish(graph_mix(adjacency, std::move(z)));
        else return finish(spmm(adjacency, std::move(z)));
    }

private:
    template <class Y>
    auto finish(Y&& y) const {
        if constexpr (Opt.bias) return activate<Opt.activation>(std::forward<Y>(y) + bias);
        else return activate<Opt.activation>(std::forward<Y>(y));
    }

    // H · W on the last axis.
    template <class X>
    static auto on_features(X&& x, const Param<T, 2>& w) {
        constexpr std::size_t R = std::remove_cvref_t<X>::rank;
        if constexpr (R == 2) return matmul(std::forward<X>(x), w);
        else {
            const auto s = x.shape();
            return reshape(matmul(reshape(std::forward<X>(x), Shape{s[0] * s[1], s[2]}), w),
                           Shape{s[0], s[1], w.shape()[1]});
        }
    }

    static Matrix<T> init(std::size_t in, std::size_t out, Rng& rng) {
        auto one = [&] {
            return Opt.activation == Activation::relu ? he_normal<T>(in, out, rng) : glorot_uniform<T>(in, out, rng);
        };
        if constexpr (!Opt.self_weight) return one();
        else {   // two independent blocks, side by side
            const Matrix<T> w = one(), s = one();
            Matrix<T> both(Shape{in, 2 * out});
            auto B = both.mut();
            for (std::size_t i = 0; i < in; ++i)
                for (std::size_t j = 0; j < out; ++j) B[i, j] = w(i, j), B[i, out + j] = s(i, j);
            return both;
        }
    }

    static auto make_bias(std::size_t out) {
        if constexpr (Opt.bias) return Param<T, 1>(Vector<T>(Shape{out}));
        else return Nothing{};
    }
};

}  // namespace xinn
