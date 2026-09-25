// SPDX-License-Identifier: BSD-3-Clause
// Sparse matrices, spmm and graph convolution: values against dense
// products, gradients against finite differences.
#include "check.hpp"

#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{23};

template <std::size_t R>
Tensor<double, R> random(const Shape<R>& s) { return randn<double>(s, rng); }

template <class F, std::size_t R>
Tensor<double, R> numeric_grad(F&& loss, Param<double, R>& p, double h = 1e-6) {
    const Tensor<double, R> base = p.tensor().clone();
    Tensor<double, R> out(base.shape());
    for (std::size_t i = 0; i < base.size(); ++i) {
        Tensor<double, R> moved = base.clone();
        moved.mut_flat()[i] += h;
        p.assign(moved);
        const double up = Scalar<double>(loss())();
        moved = base.clone();
        moved.mut_flat()[i] -= h;
        p.assign(moved);
        const double down = Scalar<double>(loss())();
        out.mut_flat()[i] = (up - down) / (2 * h);
    }
    p.assign(base);
    return out;
}

template <class F, class... Ps>
void gradcheck(F&& loss, Ps&... params) {
    (params.zero_grad(), ...);
    backward(loss());
    template for (auto& p : std::tie(params...)) {
        const auto expected = numeric_grad(loss, p);
        for (std::size_t i = 0; i < expected.size(); ++i)
            check::near(p.grad().flat()[i], expected.flat()[i], 1e-5 * (1 + std::abs(expected.flat()[i])));
    }
}

// A 5-node path 0-1-2-3-4 with one extra directed edge 4 -> 0, weighted.
std::vector<Edge> sample_edges() {
    return {{0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 0.5}, {3, 4, 1.5}, {4, 0, 3.0}, {1, 0, 0.25}};
}

// A rectangular random matrix, with a repeated entry.
SparseMatrix<double> random_sparse(std::size_t rows, std::size_t cols) {
    std::vector<Edge> e;
    std::uniform_int_distribution<std::size_t> R(0, rows - 1), C(0, cols - 1);
    std::normal_distribution<double> V;
    for (std::size_t k = 0; k < 2 * rows; ++k) e.push_back({R(rng), C(rng), V(rng)});
    e.push_back(e.front());
    return SparseMatrix<double>::from_edges(rows, cols, e);
}

struct TinyGcn : Module {
    GraphConv<{.activation = Activation::tanh}, double> g1;
    GraphConv<{}, double> g2;
    TinyGcn(const SparseMatrix<double>& a, Rng& r) : g1(a, 3, 4, r), g2(a, 4, 2, r) {}
    auto forward(const auto& x) const { return g2(g1(x)); }
};

}  // namespace

namespace tests {

void csr_is_built_from_an_edge_list() {
    auto a = SparseMatrix<double>::from_edges(5, 5, sample_edges());
    check::equal(a.rows(), 5uz);
    check::equal(a.nnz(), 6uz);
    check::equal(a(1, 2), 2.0);
    check::equal(a(2, 1), 0.0);
    check::equal(a(1, 0), 0.25);
    // Row 1 holds columns 0 and 2, sorted.
    check::equal(a.offsets()[1], 1uz);
    check::equal(a.offsets()[2], 3uz);
    check::equal(a.columns()[1], 0uz);
    check::equal(a.columns()[2], 2uz);
    // Repeated entries are added.
    std::vector<Edge> twice{{0, 1, 1.0}, {0, 1, 2.0}};
    check::equal(SparseMatrix<double>::from_edges(2, 2, twice)(0, 1), 3.0);
    check::equal(SparseMatrix<double>::from_edges(2, 2, twice).nnz(), 1uz);
}

void transpose_is_the_transpose() {
    auto a = random_sparse(6, 4);
    auto t = a.transposed();
    check::equal(t.rows(), 4uz);
    check::equal(t.cols(), 6uz);
    check::equal(t.nnz(), a.nnz());
    for (std::size_t i = 0; i < 6; ++i)
        for (std::size_t j = 0; j < 4; ++j) check::equal(t(j, i), a(i, j));
}

void spmm_matches_the_dense_product() {
    auto a = random_sparse(7, 5);
    const Matrix<double> dense = a.dense();
    auto x = random(Shape(5, 3));
    const Matrix<double> y = spmm(a, x);
    const Matrix<double> want = matmul(dense, x);
    check::equal(y.shape(), Shape(7, 3));
    for (std::size_t i = 0; i < y.size(); ++i) check::near(y.flat()[i], want.flat()[i], 1e-12);

    auto xb = random(Shape(4, 5, 3));   // a batch: each item separately
    const Tensor<double, 3> yb = spmm(a, xb);
    check::equal(yb.shape(), Shape(4, 7, 3));
    for (std::size_t b = 0; b < 4; ++b) {
        const Matrix<double> wb = matmul(dense, xb[b]);
        for (std::size_t i = 0; i < wb.size(); ++i) check::near(yb[b].flat()[i], wb.flat()[i], 1e-12);
    }
}

void spmm_runs_in_parallel_on_large_batches() {
    auto a = random_sparse(200, 200);
    auto x = randn<float>(Shape(64, 200, 32), rng);
    const Tensor<float, 3> y = spmm(SparseMatrix<float>::from_edges(200, 200, a.entries()), x);
    const Matrix<double> dense = a.dense();
    const Matrix<float> x9 = x[9];
    for (std::size_t i = 0; i < 200; i += 37)
        for (std::size_t f = 0; f < 32; f += 5) {
            double want = 0;
            for (std::size_t j = 0; j < 200; ++j) want += dense(i, j) * x9(j, f);
            check::near(double(y(9, i, f)), want, 1e-3);
        }
}

void spmm_gradients() {
    auto a = random_sparse(6, 4);
    Param X(random(Shape(4, 3)));
    auto w = random(Shape(6, 3));
    gradcheck([&] { return sum(spmm(a, X) * w); }, X);
    Param XB(random(Shape(2, 4, 3)));
    auto wb = random(Shape(2, 6, 3));
    gradcheck([&] { return sum(square(spmm(a, XB) - wb)); }, XB);
}

void symmetrize_keeps_the_larger_weight() {
    auto s = symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges()));
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) check::equal(s(i, j), s(j, i));
    check::equal(s(0, 1), 1.0);   // max(1, 0.25)
    check::equal(s(0, 4), 3.0);
    check::equal(s.nnz(), 10uz);
}

void gcn_normalization_properties() {
    auto a = symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges()));
    auto h = gcn_normalize(a);
    // Â = D^-1/2 (A + I) D^-1/2 entry by entry.
    std::vector<double> d(5, 1.0);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) d[i] += a(i, j);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) {
            const double want = ((i == j) + a(i, j)) / std::sqrt(d[i] * d[j]);
            check::near(h(i, j), want, 1e-12);
            check::near(h(i, j), h(j, i), 1e-15);   // symmetric
        }
    // D^1/2 · 1 is an eigenvector with eigenvalue 1, the largest.
    Matrix<double> v(Shape(5, 1));
    for (std::size_t i = 0; i < 5; ++i) v.set(i, 0, std::sqrt(d[i]));
    const Matrix<double> hv = spmm(h, v);
    for (std::size_t i = 0; i < 5; ++i) check::near(hv(i, 0), v(i, 0), 1e-12);
    // Power iteration from a random start never grows: spectral radius 1.
    Matrix<double> x = random(Shape(5, 1));
    double norm0 = 0, norm = 0;
    for (double e : x.flat()) norm0 += e * e;
    for (int k = 0; k < 50; ++k) x = spmm(h, x);
    for (double e : x.flat()) norm += e * e;
    check::that(norm <= norm0 + 1e-12);
    // A graph without edges: Â = I.
    auto none = gcn_normalize(SparseMatrix<double>::from_edges(3, 3, std::vector<Edge>{}));
    check::equal(none.nnz(), 3uz);
    check::equal(none(1, 1), 1.0);
}

void gaussian_kernel_weights() {
    std::vector<Edge> d{{0, 1, 100}, {1, 2, 200}, {2, 3, 300}};   // std of lengths: 81.65
    auto w = gaussian_kernel(d);
    check::equal(w.size(), 1uz);   // exp(-(200/81.65)^2) = 0.0025 < 0.1: dropped, and so is 300
    check::near(w[0].weight, std::exp(-1.5), 1e-12);
    check::equal(gaussian_kernel(d, 1000.0).size(), 3uz);
}

void graph_conv_layer_gradients() {
    auto a = gcn_normalize(symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges())));
    GraphConv<{.activation = Activation::tanh}, double> g(a, 3, 2, rng);
    static_assert(param_tensor_count<decltype(g)> == 2);   // the adjacency is not a parameter
    auto x = random(Shape(2, 5, 3));
    auto t = random(Shape(2, 5, 2));
    gradcheck([&] { return sum(square(g(x) - t)); }, g.weight, g.bias);

    // Through the input as well, and through two stacked layers.
    TinyGcn net(a, rng);
    Param X(random(Shape(2, 5, 3)));
    auto t2 = random(Shape(2, 5, 2));
    gradcheck([&] { return sum(square(net(X) - t2)); }, X, net.g1.weight, net.g1.bias, net.g2.weight);
}

void graph_mix_values_and_gradients() {
    auto a = gcn_normalize(symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges())));
    const Matrix<double> dense = a.dense();
    auto z = random(Shape(2, 5, 6));   // F = 3: columns 0-2 mixed, 3-5 kept
    const Tensor<double, 3> y = graph_mix(a, z);
    check::equal(y.shape(), Shape(2, 5, 3));
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t i = 0; i < 5; ++i)
            for (std::size_t f = 0; f < 3; ++f) {
                double want = z(b, i, 3 + f);
                for (std::size_t j = 0; j < 5; ++j) want += dense(i, j) * z(b, j, f);
                check::near(y(b, i, f), want, 1e-12);
            }
    Param Z(random(Shape(2, 5, 6)));
    auto w = random(Shape(2, 5, 3));
    gradcheck([&] { return sum(square(graph_mix(a, Z) - w)); }, Z);
    Param Z2(random(Shape(5, 4)));
    auto w2 = random(Shape(5, 2));
    gradcheck([&] { return sum(graph_mix(a, Z2) * w2); }, Z2);
}

void graph_conv_with_a_self_weight() {
    auto a = gcn_normalize(symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges())));
    GraphConv<{.activation = Activation::tanh, .self_weight = true}, double> g(a, 3, 2, rng);
    check::equal(g.weight.shape(), Shape(3, 4));   // [W | W_self]
    auto x = random(Shape(2, 5, 3));
    // H' = tanh(Â H W + H W_self + b), written out densely.
    Matrix<double> W(Shape(3, 2)), S(Shape(3, 2));
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j) W.set(i, j, g.weight.tensor()(i, j)), S.set(i, j, g.weight.tensor()(i, 2 + j));
    const Tensor<double, 3> y = g(x);
    for (std::size_t b = 0; b < 2; ++b) {
        const Matrix<double> want = tanh(matmul(matmul(a.dense(), x[b]), W) + matmul(x[b], S) + g.bias);
        for (std::size_t i = 0; i < want.size(); ++i) check::near(y[b].flat()[i], want.flat()[i], 1e-12);
    }
    auto t = random(Shape(2, 5, 2));
    Param X(random(Shape(2, 5, 3)));
    gradcheck([&] { return sum(square(g(X) - t)); }, X, g.weight, g.bias);
}

void graph_conv_matches_its_definition() {
    auto a = gcn_normalize(symmetrize(SparseMatrix<double>::from_edges(5, 5, sample_edges())));
    GraphConv<{.activation = Activation::relu}, double> g(a, 3, 4, rng);
    auto x = random(Shape(5, 3));
    const Matrix<double> y = g(x);
    const Matrix<double> want = relu(matmul(matmul(a.dense(), x), g.weight) + g.bias);
    for (std::size_t i = 0; i < y.size(); ++i) check::near(y.flat()[i], want.flat()[i], 1e-12);
    // With Â = I it is a dense layer at every node.
    GraphConv<{}, double> id(SparseMatrix<double>::identity(5), 3, 4, rng);
    NodeDense<{}, double> nd(3, 4, rng);
    nd.dense.weight.assign(id.weight.tensor());
    auto xb = random(Shape(2, 5, 3));
    const Tensor<double, 3> y1 = id(xb), y2 = nd(xb);
    for (std::size_t i = 0; i < y1.size(); ++i) check::near(y1.flat()[i], y2.flat()[i], 1e-12);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
