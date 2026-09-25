// SPDX-License-Identifier: BSD-3-Clause
// Convolution, pooling and reshape: values against direct loops, gradients
// against finite differences.
#include "check.hpp"

#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{17};

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

// The textbook definition, for comparison.
Tensor<double, 4> direct_conv(const Tensor<double, 4>& x, const Tensor<double, 4>& w, const Tensor<double, 1>& b,
                              std::size_t s, std::size_t p) {
    const auto X = x.shape(), K = w.shape();
    const std::size_t OH = (X[2] + 2 * p - K[2]) / s + 1, OW = (X[3] + 2 * p - K[3]) / s + 1;
    Tensor<double, 4> y(Shape{X[0], K[0], OH, OW});
    for (std::size_t n = 0; n < X[0]; ++n)
        for (std::size_t f = 0; f < K[0]; ++f)
            for (std::size_t oh = 0; oh < OH; ++oh)
                for (std::size_t ow = 0; ow < OW; ++ow) {
                    double v = b(f);
                    for (std::size_t c = 0; c < X[1]; ++c)
                        for (std::size_t kh = 0; kh < K[2]; ++kh)
                            for (std::size_t kw = 0; kw < K[3]; ++kw) {
                                const long ih = long(oh * s + kh) - long(p), iw = long(ow * s + kw) - long(p);
                                if (ih >= 0 && ih < long(X[2]) && iw >= 0 && iw < long(X[3]))
                                    v += x(n, c, std::size_t(ih), std::size_t(iw)) * w(f, c, kh, kw);
                            }
                    y.set(n, f, oh, ow, v);
                }
    return y;
}

// A tiny CNN: conv -> relu -> pool -> flatten -> dense. (Member templates
// are not allowed in local classes, so it lives here.)
struct TinyCnn : Module {
    Conv2D<{.kernel = 3, .padding = 1, .activation = Activation::relu}, double> c1;
    MaxPool2D<2> pool;
    Flatten flat;
    Dense<{}, double> out;
    explicit TinyCnn(Rng& r) : c1(1, 2, r), out(2 * 2 * 2, 2, r) {}
    auto forward(const auto& x) const { return out(flat(pool(c1(x)))); }
};

}  // namespace

namespace tests {

void conv2d_matches_the_definition() {
    for (auto [s, p] : {std::pair{1uz, 0uz}, {1, 1}, {2, 1}, {2, 0}}) {
        auto x = random(Shape(2, 3, 7, 6)), w = random(Shape(4, 3, 3, 3));
        auto b = random(Shape(4));
        const Tensor<double, 4> y = conv2d(x, w, b, s, p);
        const auto want = direct_conv(x, w, b, s, p);
        check::equal(y.shape(), want.shape());
        for (std::size_t i = 0; i < y.size(); ++i) check::near(y.flat()[i], want.flat()[i], 1e-10);
    }
}

void conv2d_gradients() {
    Param X(random(Shape(2, 2, 5, 5)));
    Param W(random(Shape(3, 2, 3, 3)));
    Param b(random(Shape(3)));
    auto y = random(Shape(2, 3, 3, 3));
    gradcheck([&] { return sum(conv2d(X, W, b, 2, 1) * y); }, X, W, b);
    auto y1 = random(Shape(2, 3, 5, 5));
    gradcheck([&] { return sum(square(conv2d(X, W, b, 1, 1) - y1)); }, X, W, b);
}

void max_pool_takes_window_maxima() {
    Tensor<double, 4> x(Shape(1, 1, 2, 4), {1, 5, 2, 0, 3, 4, 8, 7});
    const Tensor<double, 4> y = max_pool2d<2>(x);
    check::equal(y.shape(), Shape(1, 1, 1, 2));
    check::equal(y(0, 0, 0, 0), 5.0);
    check::equal(y(0, 0, 0, 1), 8.0);
}

void max_pool_gradient() {
    // Distinct values, so no window has a tie at its maximum.
    Tensor<double, 4> x0(Shape(2, 2, 4, 4));
    for (std::size_t i = 0; i < x0.size(); ++i) x0.mut_flat()[i] = double((i * 37) % 64) / 7.0;
    Param X(x0);
    auto y = random(Shape(2, 2, 2, 2));
    gradcheck([&] { return sum(max_pool2d<2>(X) * y); }, X);
}

void reshape_shares_the_buffer_and_passes_gradients() {
    Matrix<double> m(Shape(2, 6), {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    const Tensor<double, 3> r = reshape(m, Shape(3, 2, 2));
    check::equal(r(2, 1, 0), 10.0);
    check::that(m.shared());   // r views m's buffer

    Param P(random(Shape(2, 6)));
    auto w = random(Shape(3, 4));
    gradcheck([&] { return sum(reshape(P, Shape(3, 4)) * w); }, P);
}

void a_small_cnn_trains_end_to_end() {
    Rng r{3};
    TinyCnn net(r);
    static_assert(param_tensor_count<TinyCnn> == 4);
    auto x = random(Shape(3, 1, 4, 4));
    const Matrix<double> t(Shape(3, 2), {1, 0, 0, 1, 1, 0});
    Adam opt(net, {.lr = 0.05});
    double first = 0, last = 0;
    for (int i = 0; i < 60; ++i) {
        opt.zero_grad();
        last = backward(softmax_cross_entropy(net(x), t));
        if (i == 0) first = last;
        opt.step();
    }
    check::that(last < first * 0.5);
}

void nested_parallel_loops_do_not_deadlock() {
    // A parallel convolution calls the (parallel) matrix kernel for each image.
    auto x = randn<float>(Shape(16, 8, 32, 32), rng);
    auto w = randn<float>(Shape(16, 8, 3, 3), rng);
    auto b = randn<float>(Shape(16), rng);
    const Tensor<float, 4> y = conv2d(x, w, b, 1, 1);
    check::equal(y.shape(), Shape(16, 16, 32, 32));
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
