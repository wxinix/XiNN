// SPDX-License-Identifier: BSD-3-Clause
// Attention and transformers: every new operation against direct loops, and
// every gradient against finite differences.
#include "check.hpp"

#include <cmath>
#include <vector>

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

// Element (i, j) of op(X[b]) for a rank-3 X.
template <bool Trans>
double at(const Tensor<double, 3>& x, std::size_t b, std::size_t i, std::size_t j) {
    return Trans ? x(b, j, i) : x(b, i, j);
}

template <bool TA, bool TB>
void check_bmm_values() {
    const std::size_t B = 3, M = 5, K = 4, N = 6;
    auto a = TA ? random(Shape(B, K, M)) : random(Shape(B, M, K));
    auto b = TB ? random(Shape(B, N, K)) : random(Shape(B, K, N));
    const Tensor<double, 3> c = bmm<TA, TB>(a, b);
    check::equal(c.shape(), Shape(B, M, N));
    for (std::size_t n = 0; n < B; ++n)
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t j = 0; j < N; ++j) {
                double want = 0;
                for (std::size_t p = 0; p < K; ++p) want += at<TA>(a, n, i, p) * at<TB>(b, n, p, j);
                check::near(c(n, i, j), want, 1e-12);
            }
}

template <bool TA, bool TB>
void check_bmm_grads() {
    Param A(TA ? random(Shape(2, 3, 4)) : random(Shape(2, 4, 3)));
    Param B(TB ? random(Shape(2, 5, 3)) : random(Shape(2, 3, 5)));
    auto w = random(Shape(2, 4, 5));
    gradcheck([&] { return sum(bmm<TA, TB>(A, B) * w); }, A, B);
}

// A tiny causal language model, for the training test.
struct TinyLm : Module {
    Embedding<double> embed;
    TransformerBlock<double> block;
    LayerNorm<double> norm;
    TokenDense<{}, double> head;
    TinyLm(std::size_t vocab, std::size_t d, Rng& r)
        : embed(vocab, d, r), block({.d_model = d, .heads = 2, .ffn = 2, .causal = true}, r), norm(d),
          head(d, vocab, r) {}
    auto forward(const Tensor<std::size_t, 2>& ids) const {
        const auto pe = sinusoidal_positions<double>(ids.shape()[1], embed.table.shape()[1]);
        return head(norm(block(embed(ids) + pe)));
    }
};

}  // namespace

namespace tests {

void bmm_matches_the_definition() {
    check_bmm_values<false, false>();
    check_bmm_values<false, true>();
    check_bmm_values<true, false>();
    check_bmm_values<true, true>();
    // Rank 4: two batch axes.
    auto a = random(Shape(2, 3, 4, 5)), b = random(Shape(2, 3, 5, 2));
    const Tensor<double, 4> c = bmm(a, b);
    double want = 0;
    for (std::size_t p = 0; p < 5; ++p) want += a(1, 2, 3, p) * b(1, 2, p, 1);
    check::near(c(1, 2, 3, 1), want, 1e-12);
}

void bmm_gradients() {
    check_bmm_grads<false, false>();
    check_bmm_grads<false, true>();
    check_bmm_grads<true, false>();
    check_bmm_grads<true, true>();
}

void permute_moves_axes() {
    auto x = random(Shape(2, 3, 4, 5));
    const Tensor<double, 4> y = permute<0, 2, 1, 3>(x);
    check::equal(y.shape(), Shape(2, 4, 3, 5));
    const Tensor<double, 4> z = permute<3, 0, 2, 1>(x);
    check::equal(z.shape(), Shape(5, 2, 4, 3));
    for (std::size_t a = 0; a < 2; ++a)
        for (std::size_t b = 0; b < 3; ++b)
            for (std::size_t c = 0; c < 4; ++c)
                for (std::size_t d = 0; d < 5; ++d) {
                    check::equal(y(a, c, b, d), x(a, b, c, d));
                    check::equal(z(d, a, c, b), x(a, b, c, d));
                }
    // A permutation followed by its inverse is the identity.
    const Tensor<double, 4> back = permute<1, 3, 2, 0>(z);
    for (std::size_t i = 0; i < x.size(); ++i) check::equal(back.flat()[i], x.flat()[i]);
}

void permute_gradient() {
    Param X(random(Shape(2, 3, 2, 4)));
    auto w = random(Shape(4, 2, 2, 3));
    gradcheck([&] { return sum(permute<3, 0, 2, 1>(X) * w); }, X);
    Param Y(random(Shape(3, 4)));
    auto v = random(Shape(4, 3));
    gradcheck([&] { return sum(square(permute<1, 0>(Y)) * v); }, Y);
}

void normalize_gives_mean_0_variance_1() {
    auto x = random(Shape(3, 4, 7)) * 5.0 + 2.0;
    const Tensor<double, 3> y = normalize(x, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t t = 0; t < 4; ++t) {
            double mu = 0, var = 0;
            for (std::size_t j = 0; j < 7; ++j) mu += y(i, t, j) / 7;
            for (std::size_t j = 0; j < 7; ++j) var += (y(i, t, j) - mu) * (y(i, t, j) - mu) / 7;
            check::near(mu, 0, 1e-12);
            check::near(var, 1, 1e-10);
        }
}

void layer_norm_gradients() {
    Param X(random(Shape(2, 3, 5)));
    Param g(random(Shape(5)));
    Param b(random(Shape(5)));
    auto w = random(Shape(2, 3, 5));
    gradcheck([&] { return sum(layer_norm(X, g, b) * w); }, X, g, b);
    gradcheck([&] { return sum(square(layer_norm(X, g, b, 0.1))); }, X, g, b);
}

void embedding_gathers_rows_and_scatters_gradients() {
    Param table(random(Shape(5, 3)));
    Tensor<std::size_t, 2> ids(Shape(2, 3), {4, 0, 4, 1, 1, 2});   // some rows used twice
    const Tensor<double, 3> e = embedding(table, ids);
    check::equal(e.shape(), Shape(2, 3, 3));
    for (std::size_t j = 0; j < 3; ++j) {
        check::equal(e(0, 2, j), table.tensor()(4, j));
        check::equal(e(1, 0, j), table.tensor()(1, j));
    }
    auto w = random(Shape(2, 3, 3));
    gradcheck([&] { return sum(embedding(table, ids) * w); }, table);
    check::near(table.grad()(3, 0), 0);   // an unused row gets no gradient
}

void take_selects_one_index() {
    auto x = random(Shape(2, 4, 3));
    const Tensor<double, 2> y = take<1>(x, 3);
    check::equal(y.shape(), Shape(2, 3));
    check::equal(y(1, 2), x(1, 3, 2));
    Param X(x);
    auto w = random(Shape(2, 4));
    gradcheck([&] { return sum(take<2>(X, 1) * w); }, X);
    gradcheck([&] { return sum(square(take<0>(X, 1))); }, X);
}

void softmax_of_rank_3_and_4() {
    Param X(random(Shape(2, 3, 4)));
    auto w = random(Shape(2, 3, 4));
    gradcheck([&] { return sum(softmax(X) * w); }, X);
    Param Y(random(Shape(2, 2, 3, 3)));
    auto v = random(Shape(2, 2, 3, 3));
    gradcheck([&] { return sum(softmax(Y) * v); }, Y);
}

void causal_mask_hides_the_future() {
    const auto m = causal_mask<double>(4);
    const Tensor<double, 3> p = softmax(random(Shape(2, 4, 4)) + m);
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t i = 0; i < 4; ++i) {
            double total = 0;
            for (std::size_t j = 0; j < 4; ++j) {
                if (j > i) check::equal(p(b, i, j), 0.0);
                total += p(b, i, j);
            }
            check::near(total, 1, 1e-12);
        }
    // Changing the last token changes no earlier output of causal attention.
    Rng r{5};
    MultiHeadAttention<double> mha(8, 2, r, true);
    auto x = random(Shape(1, 5, 8));
    const Tensor<double, 3> y0 = mha(x);
    auto x2 = x.clone();
    for (std::size_t j = 0; j < 8; ++j) x2.set(0, 4, j, 10.0);
    const Tensor<double, 3> y1 = mha(x2);
    for (std::size_t t = 0; t < 4; ++t)
        for (std::size_t j = 0; j < 8; ++j) check::near(y0(0, t, j), y1(0, t, j), 1e-12);
    check::that(std::abs(y0(0, 4, 0) - y1(0, 4, 0)) > 1e-6);
}

void sinusoidal_positions_follow_the_formula() {
    const auto pe = sinusoidal_positions<double>(10, 6);
    check::near(pe(0, 0), 0);
    check::near(pe(0, 1), 1);
    check::near(pe(3, 2), std::sin(3 / std::pow(10000.0, 2.0 / 6)), 1e-12);
    check::near(pe(7, 5), std::cos(7 / std::pow(10000.0, 4.0 / 6)), 1e-12);
}

void shared_node_gives_the_same_gradient() {
    Param W(random(Shape(3, 3)));
    auto x = random(Shape(4, 3));
    // Plain expressions: h is evaluated (and differentiated) twice.
    auto plain = [&] {
        auto h = tanh(matmul(x, W));
        return sum(square(h + matmul(h, W)));
    };
    W.zero_grad();
    const double l0 = backward(plain());
    const auto g0 = W.grad().clone();
    // Shared: once.
    auto shared = [&] {
        const auto h = share(tanh(matmul(x, W)));
        const auto h2 = share(h + matmul(h, W));
        return sum(square(h2));
    };
    W.zero_grad();
    const double l1 = backward(shared());
    check::near(l0, l1, 1e-12);
    for (std::size_t i = 0; i < g0.size(); ++i) check::near(W.grad().flat()[i], g0.flat()[i], 1e-12);
    gradcheck(shared, W);
}

void a_deep_chain_of_shared_nodes() {
    // Ten residual steps: plain expressions would evaluate the first 2^10 times.
    Param W(Tensor<double, 2>(random(Shape(3, 3)) * 0.3));
    auto x = random(Shape(2, 3));
    auto loss = [&] {
        auto h = share(x + matmul(x, W));
        for (int i = 0; i < 10; ++i) h = share(h + tanh(matmul(h, W)));
        return sum(square(h));
    };
    gradcheck(loss, W);
}

void multi_head_attention_gradients() {
    Rng r{9};
    for (bool causal : {false, true}) {
        MultiHeadAttention<double> mha(4, 2, r, causal);
        Param X(random(Shape(2, 3, 4)));
        auto w = random(Shape(2, 3, 4));
        auto loss = [&] { return sum(mha(X) * w); };
        gradcheck(loss, X, mha.query.weight, mha.key.weight, mha.value.bias, mha.output.weight);
    }
}

void transformer_block_gradients() {
    Rng r{4};
    TransformerBlock<double> block({.d_model = 4, .heads = 2, .ffn = 2, .causal = true}, r);
    static_assert(param_tensor_count<TransformerBlock<double>> == 16);
    Param X(random(Shape(2, 3, 4)));
    auto w = random(Shape(2, 3, 4));
    auto loss = [&] { return sum(block(X) * w); };
    gradcheck(loss, X, block.norm1.gain, block.attention.key.weight, block.norm2.bias, block.ffn1.weight,
              block.ffn2.bias);
}

void a_tiny_language_model_learns_a_pattern() {
    // Sequences "a b a b a b a b" for all 12 pairs a != b of 4 tokens. After
    // the first token, the next one is the token two places back, which only
    // attention can see. Only the first prediction stays uncertain (one of
    // 3), so the loss can fall to log(3) / 8 = 0.137.
    Rng r{1};
    TinyLm lm(4, 8, r);
    const std::size_t B = 12, T = 8;
    Tensor<std::size_t, 2> ids(Shape(B, T));
    Matrix<double> targets(Shape{B * T, 4uz});
    std::size_t row = 0;
    for (std::size_t a = 0; a < 4; ++a)
        for (std::size_t b = 0; b < 4; ++b) {
            if (a == b) continue;
            for (std::size_t t = 0; t < T; ++t) {
                ids.set(row, t, t % 2 ? b : a);
                targets.set(row * T + t, t % 2 ? a : b, 1.0);
            }
            ++row;
        }
    Adam opt(lm, {.lr = 0.02});
    double first = 0, last = 0;
    for (int i = 0; i < 150; ++i) {
        opt.zero_grad();
        last = backward(softmax_cross_entropy(reshape(lm(ids), Shape{B * T, 4uz}), targets));
        if (i == 0) first = last;
        opt.step();
    }
    std::println("    loss {:.3f} -> {:.3f}", first, last);
    check::that(first > 1.0);
    check::that(last < 0.2);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
