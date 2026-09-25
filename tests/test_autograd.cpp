// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{42};

template <std::size_t R>
Tensor<double, R> random(const Shape<R>& s) { return randn<double>(s, rng); }

// Central differences: dLoss/dp_i ~ (f(p_i + h) - f(p_i - h)) / 2h.
template <class F, std::size_t R>
Tensor<double, R> numeric_grad(F&& loss, Param<double, R>& p, double h = 1e-6) {
    const Tensor<double, R> base = p.tensor().clone();
    Tensor<double, R> out(base.shape());
    for (std::size_t i = 0; i < base.size(); ++i) {
        Tensor<double, R> moved = base.clone();
        moved.mut_flat()[i] = base.flat()[i] + h;
        p.assign(moved);
        const double up = Scalar<double>(loss())();
        moved = base.clone();
        moved.mut_flat()[i] = base.flat()[i] - h;
        p.assign(moved);
        const double down = Scalar<double>(loss())();
        out.mut_flat()[i] = (up - down) / (2 * h);
    }
    p.assign(base);
    return out;
}

// backward() must agree with numeric_grad for every parameter.
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

}  // namespace

namespace tests {

void gradient_of_sum_of_squares_is_2w() {
    Param w(Vector<double>(Shape(3), {1, -2, 3}));
    const double loss = backward(sum(square(w)));
    check::near(loss, 14);
    check::near(w.grad()(0), 2);
    check::near(w.grad()(1), -4);
    check::near(w.grad()(2), 6);
}

void linear_regression_loss() {
    auto x = random(Shape(4, 3));
    auto y = random(Shape(4, 2));
    Param W(random(Shape(3, 2)));
    Param b(random(Shape(2)));
    gradcheck([&] { return mse(matmul(x, W) + b, y); }, W, b);
}

void bias_gradient_sums_over_the_batch() {
    Matrix<double> x(Shape(3, 2), 1.0);
    Param b(Vector<double>(Shape(2), {0, 0}));
    backward(sum(x + b));
    check::near(b.grad()(0), 3);   // b was added to 3 rows
    check::near(b.grad()(1), 3);
}

void chain_of_activations() {
    auto x = random(Shape(5, 3));
    Param W(random(Shape(3, 4)));
    Param b(random(Shape(4)));
    gradcheck([&] { return mean(sigmoid(tanh(matmul(x, W)) * 2 + b)); }, W, b);
}

void exp_log_div_sqrt() {
    Param u(random(Shape(2, 3)));
    Param v(random(Shape(3)));
    gradcheck([&] { return sum(log(exp(u) + 1) / sqrt(square(v) + 1)); }, u, v);
}

void relu_away_from_the_kink() {
    // Chosen so that no element of x·W is 0, where ReLU has no derivative.
    Matrix<double> x(Shape(2, 2), {1.0, 2.0, -1.5, 0.7});
    Param W(Matrix<double>(Shape(2, 2), {0.3, -0.7, 0.9, 0.2}));
    auto y = random(Shape(2, 2));
    gradcheck([&] { return sum(relu(matmul(x, W)) * y); }, W);
}

void softmax_and_cross_entropy() {
    auto x = random(Shape(6, 4));
    std::vector<std::size_t> labels{0, 3, 1, 1, 2, 0};
    auto t = one_hot<double>(labels, 4);
    Param W(random(Shape(4, 4)));
    Param b(random(Shape(4)));
    gradcheck([&] { return softmax_cross_entropy(matmul(x, W) + b, t); }, W, b);

    auto y = random(Shape(6, 4));
    gradcheck([&] { return sum(softmax(matmul(x, W)) * y); }, W);
}

void weighted_cross_entropy() {
    auto x = random(Shape(5, 3));
    std::vector<std::size_t> labels{0, 2, 1, 2, 2};
    const std::vector<double> w{0.5, 3.0, 1.5};
    auto t = one_hot<double>(labels, 3, w);
    Param W(random(Shape(3, 3)));
    gradcheck([&] { return softmax_cross_entropy(matmul(x, W), t); }, W);

    // A row of weight w counts w times: the same as the plain loss of the rows, weighted.
    const Matrix<double> logits = matmul(x, W.tensor());
    const double weighted = softmax_cross_entropy(logits, t).eval()();
    const double by_hand = -Scalar<double>(sum(t * log(softmax(logits))))() / 5;
    check::near(weighted, by_hand, 1e-12);
}

void balanced_weights_equalize_classes() {
    std::vector<std::size_t> labels{0, 0, 0, 0, 0, 0, 1, 1, 2, 2};
    auto w = balanced_class_weights(labels, 3);
    check::near(w[0] * 6, w[1] * 2);            // each class has the same total weight
    check::near(w[1] * 2, w[2] * 2);
    check::near((6 * w[0] + 2 * w[1] + 2 * w[2]) / 10, 1.0);   // mean weight 1
}

void cross_entropy_matches_its_definition() {
    Matrix<double> logits(Shape(2, 3), {1, 2, 3, 0, 0, 5});
    Matrix<double> t(Shape(2, 3), {0, 0, 1, 1, 0, 0});
    const double fused = softmax_cross_entropy(logits, t).eval()();
    const double plain = -Scalar<double>(sum(t * log(softmax(logits))))() / 2;
    check::near(fused, plain, 1e-12);
}

void sums_along_axes() {
    auto x = random(Shape(3, 4));
    Param W(random(Shape(4, 5)));
    gradcheck([&] { return sum(square(sum<1>(matmul(x, W)))); }, W);
    gradcheck([&] { return sum(square(sum<0>(matmul(x, W)))); }, W);
    gradcheck([&] { return sum(square(mean<1>(matmul(x, W)))); }, W);

    Param T3(random(Shape(2, 3, 4)));
    gradcheck([&] { return sum(square(sum<1>(T3))); }, T3);
}

void transposed_products() {
    auto x = random(Shape(3, 4));
    Param A(random(Shape(3, 2)));
    Param B(random(Shape(5, 4)));
    gradcheck([&] { return sum(square(matmul(transpose(A), x))); }, A);
    gradcheck([&] { return sum(square(matmul(x, transpose(B)))); }, B);
    gradcheck([&] { return sum(square(transpose(matmul(transpose(A), x)))); }, A);
}

void scalar_param_broadcasts() {
    auto x = random(Shape(4, 3));
    auto y = random(Shape(4, 3));
    Param s(Scalar<double>(Shape<0>{}, 0.5));
    gradcheck([&] { return mean(square(x * s - y)); }, s);
}

void a_param_used_twice_sums_its_gradients() {
    auto x = random(Shape(3, 3));
    Param W(random(Shape(3, 3)));
    gradcheck([&] { return sum(matmul(x, W) * matmul(W, x) - W); }, W);
}

void negation_and_subtraction() {
    auto x = random(Shape(2, 2));
    Param w(random(Shape(2, 2)));
    gradcheck([&] { return sum(square(-(x - w)) - (w / 2)); }, w);
}

void gradients_accumulate_until_zeroed() {
    Param w(Vector<double>(Shape(2), {1, 2}));
    backward(sum(w * 3));
    backward(sum(w * 3));
    check::near(w.grad()(0), 6);
    w.zero_grad();
    check::near(w.grad()(0), 0);
}

// ---- what the types say ---------------------------------------------------------

void parameter_free_subtrees_are_pruned() {
    Vector<double> x(Shape(2), {1, 2});
    Param w(Vector<double>(Shape(2), {3, 4}));

    auto tape = detail::record(exp(x * 2) + w);
    using Tape = decltype(tape);
    // The exp(x * 2) branch has no Param: it is evaluated and frozen into a
    // Leaf holding a plain tensor. Only the path to w is kept as nodes.
    static_assert(std::same_as<std::tuple_element_t<0, decltype(tape.kids)>, detail::Leaf<Vector<double>>>);
    static_assert(std::same_as<std::tuple_element_t<1, decltype(tape.kids)>, detail::Leaf<Param<double, 1>>>);
    static_assert(Tape::trainable);
    check::near(tape.value()(0), std::exp(2.0) + 3);
}

void matmul_gradients_need_no_transposed_copies() {
    Matrix<double> a(Shape(2, 3)), b(Shape(3, 4)), g(Shape(2, 4));
    ops::MatMul<> op;
    using DA = decltype(op.grad<0>(g, g, a, b));
    using DB = decltype(op.grad<1>(g, g, a, b));
    static_assert(std::same_as<DA::op_type, ops::MatMul<false, true>>);   // dA = G·Bᵀ
    static_assert(std::same_as<DB::op_type, ops::MatMul<true, false>>);   // dB = Aᵀ·G
}

void trainable_is_decided_by_type() {
    static_assert(Trainable<Param<float, 2>>);
    static_assert(!Trainable<Matrix<float>>);
    static_assert(Trainable<decltype(Matrix<float>() * Param<float, 2>(Matrix<float>(Shape(1, 1))))>);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
