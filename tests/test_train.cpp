// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <filesystem>
#include <set>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

struct Quadratic : Module {
    Param<double, 1> w{Vector<double>(Shape(3), {5, -3, 1})};
};

struct Two : Module {
    Dense<> a;
    Param<float, 0> s{Scalar<float>(Shape<0>{}, 1.0f)};
    explicit Two(Rng& rng) : a(2, 3, rng) {}
};

const std::filesystem::path data_dir = XINN_DATA_DIR;

}  // namespace

namespace tests {

void optimizers_are_optimizers() {
    Quadratic q;
    static_assert(Optimizer<decltype(SGD(q))>);
    static_assert(Optimizer<decltype(Adam(q))>);
}

void optimizer_state_follows_the_model_type() {
    Rng rng{1};
    Two m(rng);
    using P = params_of_t<Two>;
    static_assert(std::same_as<P, std::tuple<Param<float, 2>, Param<float, 1>, Param<float, 0>>>);
    static_assert(std::same_as<decltype(Adam(m)), Adam<P>>);
}

void sgd_momentum_by_hand() {
    Quadratic q;
    SGD opt(q, {.lr = 0.1, .momentum = 0.5});
    // loss = sum(3w): the gradient is 3 everywhere.
    opt.zero_grad(); backward(sum(q.w * 3)); opt.step();   // v = 3,   w -= 0.3
    check::near(q.w.tensor()(0), 5 - 0.3);
    opt.zero_grad(); backward(sum(q.w * 3)); opt.step();   // v = 4.5, w -= 0.45
    check::near(q.w.tensor()(0), 5 - 0.3 - 0.45);
}

void adam_first_step_is_lr_times_sign() {
    Quadratic q;
    Adam opt(q, {.lr = 0.01});
    opt.zero_grad();
    backward(sum(square(q.w)));   // g = 2w = (10, -6, 2)
    opt.step();
    // After one step m/(1-b1) = g and v/(1-b2) = g^2, so the step is lr * g/|g|.
    check::near(q.w.tensor()(0), 5 - 0.01, 1e-7);
    check::near(q.w.tensor()(1), -3 + 0.01, 1e-7);
    check::near(q.w.tensor()(2), 1 - 0.01, 1e-7);
}

void optimizers_minimize_a_quadratic() {
    const Vector<double> target(Shape(3), {1, 2, 3});
    auto run = [&](auto&& make) {
        Quadratic q;
        auto opt = make(q);
        for (int i = 0; i < 500; ++i) {
            opt.zero_grad();
            backward(sum(square(q.w - target)));
            opt.step();
        }
        return q.w.tensor();
    };
    const auto sgd = run([](auto& q) { return SGD(q, {.lr = 0.05, .momentum = 0.9}); });
    const auto adam = run([](auto& q) { return Adam(q, {.lr = 0.05}); });
    for (std::size_t i = 0; i < 3; ++i) {
        check::near(sgd(i), target(i), 1e-4);
        check::near(adam(i), target(i), 1e-2);
    }
}

void weight_decay_shrinks_unused_weights() {
    Quadratic q;
    Adam opt(q, {.lr = 0.1, .weight_decay = 0.5});
    const Vector<double> zero(Shape(3));
    for (int i = 0; i < 20; ++i) {
        opt.zero_grad();
        backward(sum(q.w * zero));   // gradient 0: only the decay acts
        opt.step();
    }
    check::near(q.w.tensor()(0), 5 * std::pow(1 - 0.1 * 0.5, 20), 1e-9);
}

void cosine_schedule_endpoints() {
    check::near(cosine_lr(0, 100, 1.0), 1.0);
    check::near(cosine_lr(50, 100, 1.0), 0.5);
    check::near(cosine_lr(100, 100, 1.0, 0.1), 0.1);
}

// ---- data ----------------------------------------------------------------------------

void batches_cover_every_sample_once_per_epoch() {
    Rng rng{3};
    std::multiset<std::size_t> seen;
    std::vector<std::size_t> sizes;
    for (auto idx : shuffled_batches(10, 4, rng)) {
        sizes.push_back(idx.size());
        seen.insert(idx.begin(), idx.end());
    }
    check::equal(sizes, std::vector<std::size_t>{4, 4, 2});
    check::equal(seen.size(), 10uz);
    for (std::size_t i = 0; i < 10; ++i) check::equal(seen.count(i), 1uz);
}

void batches_keep_rows_paired() {
    Rng rng{4};
    Matrix<float> x(Shape(6, 2), {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5});
    Matrix<float> y(Shape(6, 1), {0, 10, 20, 30, 40, 50});
    std::size_t rows = 0;
    for (auto [xb, yb] : batches(x, y, 4, rng)) {
        for (std::size_t i = 0; i < xb.shape()[0]; ++i) check::equal(yb(i, 0), xb(i, 0) * 10);
        rows += xb.shape()[0];
    }
    check::equal(rows, 6uz);
}

void gather_rows_copies_the_right_rows() {
    Matrix<int> x(Shape(3, 2), {1, 2, 3, 4, 5, 6});
    const std::vector<std::size_t> idx{2, 0};
    auto g = gather_rows(x, idx);
    check::equal(std::format("{}", g), std::string("[[5, 6], [1, 2]]"));
}

void standardizer_centres_and_scales() {
    Matrix<float> x(Shape(4, 2), {1, 100, 2, 200, 3, 300, 4, 400});
    auto st = Standardizer<float>::fit(x);
    const Matrix<float> z = st(x);
    check::near(Scalar<float>(mean(z))(), 0, 1e-6);
    check::near(z(0, 0), z(0, 1), 1e-5);            // both columns now on one scale
    check::near(Scalar<float>(mean(square(z)))(), 1, 1e-5);
}

// ---- metrics -----------------------------------------------------------------------------

void confusion_matrix_and_f1() {
    const std::vector<std::size_t> truth{0, 0, 0, 1, 1, 2};
    const std::vector<std::size_t> pred{0, 0, 1, 1, 1, 0};
    ConfusionMatrix cm(pred, truth, 3);
    check::equal(cm.counts[0][1], 1uz);
    check::equal(cm.counts[2][0], 1uz);
    check::near(cm.recall(0), 2.0 / 3);
    check::near(cm.precision(1), 2.0 / 3);
    check::near(cm.f1(2), 0.0);
    check::near(accuracy(pred, truth), 4.0 / 6);
}

void regression_errors() {
    const std::vector<float> p{11, 18}, t{10, 20};
    check::near(mean_absolute_error(p, t), 1.5);
    check::near(mean_absolute_percentage_error(p, t), 10.0);
}

// ---- data sets (skipped when tools/prepare_data.py has not been run) -----------------------

void mnist_loads_when_present() {
    auto train = datasets::load_mnist(data_dir / "mnist", true);
    if (!train) return std::println("    (skipped: {})", train.error());
    check::equal(train->images.shape(), Shape(60000, 784));
    check::equal(train->labels.size(), 60000uz);
    check::equal(train->labels[0], 5uz);   // the first training digit is a 5
    const float pixel_max = std::ranges::max(train->images.flat());
    check::near(pixel_max, 1.0);
}

void pems08_loads_when_present() {
    auto p = datasets::load_pems08(data_dir / "pems08");
    if (!p) return std::println("    (skipped: {})", p.error());
    check::equal(p->data.shape(), Shape(17856, 170, 3));
    check::near(p->data(0, 0, 0), 133.0);            // flow, veh/5 min
    check::near(p->data(0, 0, 1), 0.0603, 1e-6);     // occupancy
    check::near(p->data(0, 0, 2), 65.8, 1e-4);       // speed, mph
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
