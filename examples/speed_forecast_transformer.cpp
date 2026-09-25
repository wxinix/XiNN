// SPDX-License-Identifier: BSD-3-Clause
// Traffic speed forecasting on PEMS08 with a transformer encoder.
//
// The task (shared with the recurrent and graph-network chapters):
//   - PEMS08 speed, in km/h; 170 detectors, 62 days of 5-minute intervals.
//   - Input: a detector's last 12 intervals (one hour) of speed.
//   - Targets: its speed 3, 6 and 12 intervals ahead (15, 30, 60 minutes),
//     all three from one model.
//   - Days 0-49 train, days 50-61 test. Standardized with training statistics.
//   - MAE in km/h per horizon, over all test windows and over those whose
//     true speed is below 64 km/h (congested).
//
// Baselines: persistence, the historical average for the detector and time
// of day, and an MLP on the same 12 values. The transformer treats the 12
// intervals as tokens: each speed is projected to d_model features, a
// positional encoding is added, two encoder blocks follow, and the last
// token predicts the three horizons.
#include <array>
#include <chrono>
#include <cmath>
#include <print>
#include <string>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t history = 12, per_day = 288, train_days = 50, days = 62;
constexpr std::array<std::size_t, 3> horizons{3, 6, 12};
constexpr float congested_below = 64;   // km/h

struct Windows {
    Matrix<float> x, y;             // (n, 12) and (n, 3), km/h
    std::vector<std::size_t> t, d;   // the interval "now" and the detector
};

// Every window whose history and targets lie in [day_from, day_to), taking
// every `stride`-th interval and all detectors.
Windows make_windows(const Tensor<float, 2>& speed, std::size_t day_from, std::size_t day_to, std::size_t stride) {
    const std::size_t D = speed.shape()[1];
    std::vector<std::size_t> times;
    for (std::size_t t = day_from * per_day + history - 1; t + horizons.back() < day_to * per_day; t += stride)
        times.push_back(t);
    Windows w{Matrix<float>(Shape{times.size() * D, history}), Matrix<float>(Shape{times.size() * D, horizons.size()}),
              {}, {}};
    auto v = speed.view();
    auto X = w.x.mut();
    auto Y = w.y.mut();
    std::size_t i = 0;
    for (std::size_t t : times)
        for (std::size_t d = 0; d < D; ++d, ++i) {
            for (std::size_t k = 0; k < history; ++k) X[i, k] = v[t + 1 + k - history, d];   // oldest first
            for (std::size_t h = 0; h < horizons.size(); ++h) Y[i, h] = v[t + horizons[h], d];
            w.t.push_back(t);
            w.d.push_back(d);
        }
    return w;
}

// ---- models ------------------------------------------------------------------------------------

struct SpeedMlp : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit SpeedMlp(Rng& rng) : fc1(history, 64, rng), fc2(64, 64, rng), out(64, horizons.size(), rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

constexpr std::size_t d_model = 32;

struct SpeedTransformer : Module {
    TokenDense<> embed;                 // one speed -> d_model features
    TransformerBlock<> block1, block2;
    LayerNorm<> norm;
    Dense<> head;                       // last token -> 3 horizons
    Matrix<float> positions = sinusoidal_positions<float>(history, d_model);

    explicit SpeedTransformer(Rng& rng)
        : embed(1, d_model, rng), block1({.d_model = d_model, .heads = 4, .ffn = 2}, rng),
          block2({.d_model = d_model, .heads = 4, .ffn = 2}, rng), norm(d_model), head(d_model, horizons.size(), rng) {}

    auto forward(const Matrix<float>& x) const {
        const std::size_t B = x.shape()[0];
        auto tokens = embed(reshape(x, Shape{B, history, 1uz})) + positions;   // (B, 12, d)
        return head(take<1>(norm(block2(block1(tokens))), history - 1));
    }
};

// ---- training and evaluation -------------------------------------------------------------------

template <class Net>
Matrix<float> predict(const Net& net, const Matrix<float>& x) {
    Matrix<float> out(Shape{x.shape()[0], horizons.size()});
    std::vector<std::size_t> idx;
    for (std::size_t start = 0; start < x.shape()[0]; start += 4096) {
        idx.clear();
        for (std::size_t i = start; i < std::min(start + 4096, x.shape()[0]); ++i) idx.push_back(i);
        const Matrix<float> p = net(gather_rows(x, idx));
        std::ranges::copy(p.flat(), out.mut_flat().begin() + long(start * horizons.size()));
    }
    return out;
}

template <class Net>
void train(Net& net, const Matrix<float>& x, const Matrix<float>& y, std::size_t epochs, double lr0,
           const char* name) {
    Rng rng{5};
    Adam opt(net, {.lr = lr0});
    const std::size_t batch = 256, steps = epochs * ((x.shape()[0] + batch - 1) / batch);
    std::size_t step = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t e = 1; e <= epochs; ++e) {
        double total = 0;
        std::size_t n = 0;
        for (auto [xb, yb] : batches(x, y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, lr0, lr0 / 50));
            opt.zero_grad();
            total += backward(mse(net(xb), yb));
            ++n;
            opt.step();
        }
        std::println("  {} epoch {}  training MSE {:.4f} (standardized)  {:.0f} s", name, e, total / double(n),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
}

struct Mae {
    std::array<double, 3> all{}, congested{};
};

// pred and truth in km/h.
Mae mae(const Matrix<float>& pred, const Matrix<float>& truth) {
    Mae m;
    std::array<std::size_t, 3> n_all{}, n_cong{};
    auto p = pred.view();
    auto y = truth.view();
    for (std::size_t i = 0; i < truth.shape()[0]; ++i)
        for (std::size_t h = 0; h < 3; ++h) {
            const double e = std::abs(double(p[i, h]) - double(y[i, h]));
            m.all[h] += e;
            ++n_all[h];
            if (y[i, h] < congested_below) m.congested[h] += e, ++n_cong[h];
        }
    for (std::size_t h = 0; h < 3; ++h) m.all[h] /= double(n_all[h]), m.congested[h] /= double(n_cong[h]);
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t stride = argc > 1 ? std::stoul(argv[1]) : 10;
    const std::size_t epochs = argc > 2 ? std::stoul(argv[2]) : 6;
    auto pems = datasets::load_pems08(XINN_DATA_DIR "/pems08");
    if (!pems) return std::println("PEMS08 not available: {}", pems.error()), 0;
    const std::size_t T = pems->data.shape()[0], D = pems->data.shape()[1];

    Tensor<float, 2> speed(Shape{T, D});   // km/h
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < D; ++d) speed.set(t, d, pems->data(t, d, datasets::Pems08::speed) * 1.609344f);

    const auto train_w = make_windows(speed, 0, train_days, stride);
    const auto test_w = make_windows(speed, train_days, days, 1);
    std::println("{} training windows (every {}th interval of days 0-49, all {} detectors), {} test windows (days 50-61)",
                 train_w.x.shape()[0], stride, D, test_w.x.shape()[0]);

    // One mean and scale for all speeds, from training days only.
    double sum = 0, sq = 0;
    const std::size_t n_train = train_days * per_day * D;
    for (float v : speed.flat().first(n_train)) sum += v;
    const double mean = sum / double(n_train);
    for (float v : speed.flat().first(n_train)) sq += (v - mean) * (v - mean);
    const float mu = float(mean), sd = float(std::sqrt(sq / double(n_train)));
    std::println("training speeds: mean {:.1f} km/h, sd {:.1f} km/h", mu, sd);
    auto scale = [&](const Matrix<float>& m) { return Matrix<float>((m - mu) / sd); };
    auto unscale = [&](const Matrix<float>& m) { return Matrix<float>(m * sd + mu); };
    const Matrix<float> x_train = scale(train_w.x), y_train = scale(train_w.y), x_test = scale(test_w.x);

    std::size_t congested = 0;
    for (float v : test_w.y.flat()) congested += v < congested_below;
    std::println("test targets below {} km/h: {:.2f}%", congested_below, 100.0 * double(congested) / double(test_w.y.size()));

    // Persistence: the speed now, for every horizon.
    Matrix<float> persist(test_w.y.shape());
    for (std::size_t i = 0; i < persist.shape()[0]; ++i)
        for (std::size_t h = 0; h < 3; ++h) persist.set(i, h, test_w.x(i, history - 1));

    // Historical average: the mean speed of this detector at this time of day, over the training days.
    Tensor<float, 2> profile(Shape{per_day, D});
    for (std::size_t day = 0; day < train_days; ++day)
        for (std::size_t k = 0; k < per_day; ++k)
            for (std::size_t d = 0; d < D; ++d)
                profile.set(k, d, profile(k, d) + speed(day * per_day + k, d) / float(train_days));
    Matrix<float> hist(test_w.y.shape());
    for (std::size_t i = 0; i < hist.shape()[0]; ++i)
        for (std::size_t h = 0; h < 3; ++h) hist.set(i, h, profile((test_w.t[i] + horizons[h]) % per_day, test_w.d[i]));

    Rng rng{2026};
    SpeedMlp mlp(rng);
    std::println("\nMLP: {} parameters", mlp.parameter_count());
    train(mlp, x_train, y_train, epochs, 2e-3, "MLP");
    const auto mlp_pred = unscale(predict(mlp, x_test));

    SpeedTransformer tf(rng);
    std::println("\ntransformer: 2 blocks, d_model {}, 4 heads; {} parameters", d_model, tf.parameter_count());
    train(tf, x_train, y_train, epochs, 1e-3, "transformer");
    const auto t0 = std::chrono::steady_clock::now();
    const auto tf_pred = unscale(predict(tf, x_test));
    std::println("  transformer prediction on the test set: {:.1f} s",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    std::println("\nMAE (km/h) on the test days, {} windows", test_w.x.shape()[0]);
    std::println("{:<22} | {:>7}{:>7}{:>7} | {:>7}{:>7}{:>7}", "", "all", "", "", "congested (< 64 km/h)", "", "");
    std::println("{:<22} | {:>7}{:>7}{:>7} | {:>7}{:>7}{:>7}", "method", "15 min", "30 min", "60 min", "15 min", "30 min",
                 "60 min");
    auto row = [&](const char* name, const Matrix<float>& pred) {
        const auto m = mae(pred, test_w.y);
        std::println("{:<22} | {:>7.2f}{:>7.2f}{:>7.2f} | {:>7.2f}{:>7.2f}{:>7.2f}", name, m.all[0], m.all[1], m.all[2],
                     m.congested[0], m.congested[1], m.congested[2]);
    };
    row("persistence", persist);
    row("historical average", hist);
    row("MLP", mlp_pred);
    row("transformer", tf_pred);
}
