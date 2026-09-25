// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <cmath>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{7};

template <std::size_t R>
Tensor<double, R> random(const Shape<R>& s) { return randn<double>(s, rng); }

// Central differences of a function returning double, for each element of p.
template <class F, std::size_t R>
Tensor<double, R> numeric_grad(F&& f, Param<double, R>& p, double h = 1e-6) {
    const Tensor<double, R> base = p.tensor().clone();
    Tensor<double, R> out(base.shape());
    for (std::size_t i = 0; i < base.size(); ++i) {
        Tensor<double, R> moved = base.clone();
        moved.mut_flat()[i] = base.flat()[i] + h;
        p.assign(moved);
        const double up = f();
        moved = base.clone();
        moved.mut_flat()[i] = base.flat()[i] - h;
        p.assign(moved);
        const double down = f();
        out.mut_flat()[i] = (up - down) / (2 * h);
    }
    p.assign(base);
    return out;
}

template <std::size_t R>
void check_close(const Tensor<double, R>& got, const Tensor<double, R>& expected, double tol = 1e-6) {
    check::equal(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
        check::near(got.flat()[i], expected.flat()[i], tol * (1 + std::abs(expected.flat()[i])));
}

double sigmoid_of(double a) { return 1 / (1 + std::exp(-a)); }

// A GRU and a linear read-out of its last state.
struct Recall : Module {
    GRU<float> gru;
    Dense<> head;
    explicit Recall(Rng& r) : gru(1, 16, r), head(16, 1, r) {}
};

}  // namespace

namespace tests {

// backward(y, s) for a matrix y gives the same gradients as backward(sum(y * s)).
void vjp_equals_gradient_of_weighted_sum() {
    auto x = random(Shape(4, 3));
    auto s = random(Shape(4, 2));
    Param W(random(Shape(3, 2)));
    Param b(random(Shape(2)));

    const Matrix<double> y = backward(tanh(matmul(x, W) + b), s);
    const Matrix<double> dW = W.grad().clone();
    const Vector<double> db = b.grad().clone();
    check_close(y, tanh(matmul(x, W.tensor()) + b.tensor()).eval(), 1e-12);

    W.zero_grad();
    b.zero_grad();
    backward(sum(tanh(matmul(x, W) + b) * s));
    check_close(dW, W.grad(), 1e-12);
    check_close(db, b.grad(), 1e-12);

    auto f = [&] { return Scalar<double>(sum(tanh(matmul(x, W) + b) * s))(); };
    check_close(dW, numeric_grad(f, W));
    check_close(db, numeric_grad(f, b));
}

void vjp_of_a_param_is_the_seed() {
    Param W(random(Shape(2, 2)));
    auto s = random(Shape(2, 2));
    backward(W, s);
    check_close(W.grad(), s, 1e-15);
}

void blend_mixes_and_differentiates() {
    Param z(Matrix<double>(Shape(1, 3), {0.0, 0.5, 1.0}));
    Param n(random(Shape(1, 3)));
    Param h(random(Shape(1, 3)));
    const Matrix<double> y = blend(z, n, h);
    check::near(y(0, 0), n.tensor()(0, 0), 1e-15);                                  // z = 0: the candidate
    check::near(y(0, 1), 0.5 * (n.tensor()(0, 1) + h.tensor()(0, 1)), 1e-15);       // halfway
    check::near(y(0, 2), h.tensor()(0, 2), 1e-15);                                  // z = 1: the old state
    auto c = random(Shape(1, 3));
    backward(blend(z, n, h), c);
    auto f = [&] { return Scalar<double>(sum(blend(z, n, h) * c))(); };
    check_close(Matrix<double>(z.grad()), numeric_grad(f, z));
    check_close(Matrix<double>(n.grad()), numeric_grad(f, n));
    check_close(Matrix<double>(h.grad()), numeric_grad(f, h));
}

// One step against the formulas, element by element.
void gru_step_matches_the_equations() {
    GRU<double> gru(2, 3, rng);
    gru.bz.assign(random(Shape(3)));
    gru.br.assign(random(Shape(3)));
    gru.bn.assign(random(Shape(3)));
    auto x = random(Shape(2, 2));
    auto h = random(Shape(2, 3));
    const Matrix<double> out = gru.step(x, h);

    auto W = [](const Param<double, 2>& p, std::size_t i, std::size_t j) { return p.tensor()(i, j); };
    for (std::size_t b = 0; b < 2; ++b)
        for (std::size_t j = 0; j < 3; ++j) {
            double az = gru.bz.tensor()(j), ar = gru.br.tensor()(j), xn = gru.bn.tensor()(j), hn = 0;
            for (std::size_t k = 0; k < 2; ++k) {
                az += x(b, k) * W(gru.Wz, k, j);
                ar += x(b, k) * W(gru.Wr, k, j);
                xn += x(b, k) * W(gru.Wn, k, j);
            }
            for (std::size_t k = 0; k < 3; ++k) {
                az += h(b, k) * W(gru.Uz, k, j);
                ar += h(b, k) * W(gru.Ur, k, j);
                hn += h(b, k) * W(gru.Un, k, j);
            }
            const double z = sigmoid_of(az), r = sigmoid_of(ar), n = std::tanh(xn + r * hn);
            check::near(out(b, j), (1 - z) * n + z * h(b, j), 1e-12);
        }
}

void forward_keeps_every_state() {
    GRU<double> gru(2, 3, rng);
    Tensor<double, 3> xs = random(Shape(5, 4, 2));
    auto hs = gru(xs);
    check::equal(hs.size(), 6uz);
    check::near(Scalar<double>(sum(square(hs[0])))(), 0.0);   // h_0 = 0
    const Matrix<double> h1 = gru.step(xs[0], hs[0]);
    check_close(hs[1], h1, 1e-15);
}

// The whole-sequence loss  L = sum_t sum(h_t * c_t)  against finite
// differences, for every weight and for the initial state.
void bptt_matches_finite_differences() {
    GRU<double> gru(2, 3, rng);
    gru.bz.assign(random(Shape(3)) * 0.5);
    gru.bn.assign(random(Shape(3)) * 0.5);
    Tensor<double, 3> xs = random(Shape(6, 2, 2));
    Param h0(random(Shape(2, 3)));
    std::vector<Matrix<double>> c;
    for (std::size_t t = 0; t < 6; ++t) c.push_back(random(Shape(2, 3)));

    auto loss = [&] {
        auto hs = gru(xs, h0.tensor());
        double total = 0;
        for (std::size_t t = 0; t < 6; ++t) total += Scalar<double>(sum(hs[t + 1] * c[t]))();
        return total;
    };

    gru.zero_grad();
    const auto hs = gru(xs, h0.tensor());
    const Matrix<double> dh0 = gru.bptt(xs, hs, c);   // dLoss/dh_t = c_{t-1}

    gru.for_each_param([&](const std::string&, auto& p) { check_close(p.grad().clone(), numeric_grad(loss, p)); });
    check_close(dh0, numeric_grad(loss, h0));
}

// The same, with the loss on the last state only, through a Dense head.
void bptt_through_a_head() {
    GRU<double> gru(1, 4, rng);
    Dense<{}, double> head(4, 2, rng);
    Tensor<double, 3> xs = random(Shape(5, 3, 1));
    auto y = random(Shape(3, 2));
    auto loss = [&] {
        auto hs = gru(xs);
        return Scalar<double>(mse(head(hs.back()), y))();
    };
    const auto hs = gru(xs);
    Param last(hs.back());
    backward(mse(head(last), y));
    gru.bptt(xs, hs, last.grad());
    check_close(Matrix<double>(gru.Uz.grad()), numeric_grad(loss, gru.Uz));
    check_close(Matrix<double>(gru.Wn.grad()), numeric_grad(loss, gru.Wn));
    check_close(Vector<double>(gru.br.grad()), numeric_grad(loss, gru.br));
    check_close(Matrix<double>(head.weight.grad()), numeric_grad(loss, head.weight));
}

void time_major_reorders_rows() {
    Matrix<float> rows(Shape(2, 6), {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15});
    auto xs = time_major(rows, 2);   // 3 steps of 2 features
    check::equal(xs.shape()[0], 3uz);
    check::equal(xs(1, 0, 1), 3.0f);
    check::equal(xs(2, 1, 0), 14.0f);
}

// A task only a memory can solve: after 8 steps of noise, output the first value.
void gru_learns_to_recall_the_first_value() {
    Rng r{3};
    Recall net(r);
    Adam opt(net, {.lr = 1e-2});
    const std::size_t steps = 8, batch = 64;
    auto make = [&] {
        Matrix<float> x = uniform<float>(Shape(batch, steps), r, -1, 1);
        Matrix<float> y(Shape(batch, 1uz));
        for (std::size_t b = 0; b < batch; ++b) y.set(b, 0, x(b, 0));
        return std::pair{time_major(x), y};
    };
    double first = 0, last = 0;
    for (int it = 0; it < 400; ++it) {
        auto [xs, y] = make();
        opt.zero_grad();
        const auto hs = net.gru(xs);
        Param h(hs.back());
        const double l = backward(mse(net.head(h), y));
        net.gru.bptt(xs, hs, h.grad());
        opt.step();
        if (it < 20) first += l / 20;
        if (it >= 380) last += l / 20;
    }
    std::println("    recall loss: {:.4f} -> {:.4f} (predicting 0 gives 0.333)", first, last);
    check::that(first > 0.2);
    check::that(last < 0.02);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
