// SPDX-License-Identifier: BSD-3-Clause
// Speed forecasting on real detectors (PEMS08) with a recurrent network.
//
// Task, shared with the transformer and graph-network chapters:
//   input   the last 12 intervals (1 hour) of speed at one detector, km/h
//   output  its speed 3, 6 and 12 intervals ahead (15, 30, 60 min): one
//           model predicts all three
//   split   days 0-49 train, days 50-61 test, by time
//   scaling mean and standard deviation of the training speeds only
//
// Models: persistence (the speed now), the historical average by detector
// and time of day (training days), an MLP on the 12 values, and a GRU that
// reads them one step at a time; each network is trained with MSE and with
// a smooth absolute error. Errors are MAE in km/h, over all test targets,
// over the congested ones (true speed < 64 km/h), and over the windows that
// are congested now.
//
//   speed_forecast_rnn [epochs=4] [direct]   ("direct": predict the speed,
//                                              not its change from now)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <format>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t history = 12, per_day = 288, train_days = 50;
constexpr std::size_t horizons[] = {3, 6, 12};
constexpr std::size_t outputs = std::size(horizons), max_ahead = 12;
constexpr float congested_below = 64;   // km/h (40 mph)

struct Windows {
    Matrix<float> x, y;         // (n, 12) and (n, 3), km/h
    std::vector<float> ha;      // historical average for each target, n * 3
};

// Speed in km/h, (interval, detector).
Matrix<float> speeds(const datasets::Pems08& p) {
    const std::size_t T = p.data.shape()[0], D = p.data.shape()[1];
    Matrix<float> v(Shape{T, D});
    auto src = p.data.view();
    auto dst = v.mut();
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < D; ++d) dst[t, d] = src[t, d, datasets::Pems08::speed] * 1.609344f;
    return v;
}

// Mean speed by (time of day, detector) over the training days.
Matrix<float> historical_average(const Matrix<float>& v) {
    const std::size_t D = v.shape()[1];
    Matrix<float> ha(Shape{per_day, D});
    auto src = v.view();
    auto dst = ha.mut();
    for (std::size_t k = 0; k < per_day; ++k)
        for (std::size_t d = 0; d < D; ++d) {
            double s = 0;
            for (std::size_t day = 0; day < train_days; ++day) s += src[day * per_day + k, d];
            dst[k, d] = float(s / train_days);
        }
    return ha;
}

// All windows whose inputs and targets lie in days [day_from, day_to). With
// stride s, detector d keeps the windows ending at t = d mod s, d mod s + s, ...
// relative to the first one, so every offset is used by some detector.
Windows make_windows(const Matrix<float>& v, const Matrix<float>& ha, std::size_t day_from, std::size_t day_to,
                     std::size_t stride) {
    const std::size_t D = v.shape()[1], first = day_from * per_day + history - 1, end = day_to * per_day;
    auto src = v.view();
    auto h = ha.view();
    std::vector<float> xs, ys, has;
    for (std::size_t d = 0; d < D; ++d)
        for (std::size_t t = first + d % stride; t + max_ahead < end; t += stride) {
            for (std::size_t lag = history; lag-- > 0;) xs.push_back(src[t - lag, d]);   // oldest first
            for (auto k : horizons) {
                ys.push_back(src[t + k, d]);
                has.push_back(h[(t + k) % per_day, d]);
            }
        }
    const std::size_t n = ys.size() / outputs;
    Windows w{Matrix<float>(Shape{n, history}), Matrix<float>(Shape{n, outputs}), std::move(has)};
    std::ranges::copy(xs, w.x.mut_flat().begin());
    std::ranges::copy(ys, w.y.mut_flat().begin());
    return w;
}

// Squared error, or an absolute error made smooth at 0: sqrt(e^2 + c^2),
// with c = 0.1 standard deviations (about 1 km/h). Built from existing
// ops, so it needs no gradient rule of its own.
enum class Loss { mse, l1 };

template <class P, class Y>
float loss_backward(Loss loss, const P& pred, const Y& y) {
    if (loss == Loss::mse) return backward(mse(pred, y));
    return backward(mean(sqrt(square(pred - y) + 0.01f)));
}

struct MLP : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit MLP(Rng& rng) : fc1(history, 64, rng), fc2(64, 64, rng), out(64, outputs, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }

    // One training step on a batch; returns the loss.
    float train_step(Loss loss, const Matrix<float>& x, const Matrix<float>& y) const {
        return loss_backward(loss, forward(x), y);
    }
    Matrix<float> predict(const Matrix<float>& x) const { return forward(x); }
};

// A GRU reads the 12 speeds one at a time; a linear head maps its last
// state to the three forecasts.
struct SpeedGRU : Module {
    GRU<float> gru;
    Dense<> head;
    SpeedGRU(std::size_t hidden, Rng& rng) : gru(1, hidden, rng), head(hidden, outputs, rng) {}

    float train_step(Loss loss_kind, const Matrix<float>& x, const Matrix<float>& y) const {
        const auto xs = time_major(x);                        // (12, batch, 1)
        const auto hs = gru(xs);
        const Param<float, 2> last(hs.back());                // cut here: the head's backward stops at h_12
        const float loss = loss_backward(loss_kind, head(last), y);
        gru.bptt(xs, hs, last.grad());                        // and the GRU's starts there
        return loss;
    }
    Matrix<float> predict(const Matrix<float>& x) const { return head(gru(time_major(x)).back()); }
};

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

template <class Net>
void train(Net& net, std::string_view name, Loss loss, const Matrix<float>& x, const Matrix<float>& y,
           std::size_t epochs, double lr0) {
    Rng rng{17};
    Adam opt(net, {.lr = lr0});
    const std::size_t batch = 256, steps = epochs * ((x.shape()[0] + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 1; e <= epochs; ++e) {
        const auto t0 = std::chrono::steady_clock::now();
        double total = 0;
        std::size_t n = 0;
        for (auto [xb, yb] : batches(x, y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, lr0, lr0 / 100));
            opt.zero_grad();
            total += net.train_step(loss, xb, yb);
            ++n;
            opt.step();
        }
        std::println("  {} epoch {}  loss {:.4f}  ({:.1f} s)", name, e, total / double(n), seconds_since(t0));
        std::fflush(stdout);
    }
}

// The forecast as a change from the speed now (standardized): y - x_last.
Matrix<float> minus_last(const Matrix<float>& y, const Matrix<float>& x) {
    Matrix<float> out = y.clone();
    auto m = out.mut();
    for (std::size_t i = 0; i < y.shape()[0]; ++i)
        for (std::size_t k = 0; k < outputs; ++k) m[i, k] -= x(i, history - 1);
    return out;
}

// Predictions in km/h, n * 3, in chunks. x is standardized.
template <class Net>
std::vector<float> predict(const Net& net, const Matrix<float>& x, float mu, float sd, bool residual) {
    std::vector<float> out;
    std::vector<std::size_t> idx;
    for (std::size_t start = 0; start < x.shape()[0]; start += 4096) {
        idx.clear();
        for (std::size_t i = start; i < std::min(start + 4096, x.shape()[0]); ++i) idx.push_back(i);
        const Matrix<float> p = net.predict(gather_rows(x, idx));
        for (std::size_t i = 0; i < idx.size(); ++i)
            for (std::size_t k = 0; k < outputs; ++k)
                out.push_back((p(i, k) + (residual ? x(idx[i], history - 1) : 0.0f)) * sd + mu);
    }
    return out;
}

struct Errors {
    double all[outputs]{}, congested[outputs]{}, congested_now[outputs]{};
};

// MAE per horizon: over all targets, over the congested targets (true
// speed < 64 km/h), and over the windows that are congested now (speed at
// the last input < 64 km/h). The second selects by the outcome, the third
// by what the forecaster can see.
Errors mae(std::span<const float> pred, const Windows& w) {
    Errors e;
    std::size_t n_cong[outputs]{}, n_now = 0;
    auto t = w.y.flat();
    const std::size_t n = w.y.shape()[0];
    for (std::size_t i = 0; i < n; ++i) {
        const bool now = w.x(i, history - 1) < congested_below;
        n_now += now;
        for (std::size_t k = 0; k < outputs; ++k) {
            const double err = std::abs(double(pred[i * outputs + k]) - t[i * outputs + k]);
            e.all[k] += err;
            if (t[i * outputs + k] < congested_below) e.congested[k] += err, ++n_cong[k];
            if (now) e.congested_now[k] += err;
        }
    }
    for (std::size_t k = 0; k < outputs; ++k) {
        e.all[k] /= double(n);
        e.congested[k] /= double(std::max<std::size_t>(n_cong[k], 1));
        e.congested_now[k] /= double(std::max<std::size_t>(n_now, 1));
    }
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t epochs = argc > 1 ? std::stoul(argv[1]) : 4;
    const bool residual = argc > 2 ? std::string_view(argv[2]) != "direct" : true;
    const std::size_t stride = 12;

    auto pems = datasets::load_pems08(XINN_DATA_DIR "/pems08");
    if (!pems) return std::println("PEMS08 not found: {}", pems.error()), 0;
    const Matrix<float> v = speeds(*pems);
    const std::size_t days = v.shape()[0] / per_day;
    const Matrix<float> ha = historical_average(v);

    const auto train_w = make_windows(v, ha, 0, train_days, stride);
    const auto test_w = make_windows(v, ha, train_days, days, 1);

    // Standardize with the training speeds: one mean and scale for all detectors.
    double s = 0, sq = 0;
    const auto train_speeds = v.flat().first(train_days * per_day * v.shape()[1]);
    for (float x : train_speeds) s += x;
    const double mean = s / double(train_speeds.size());
    for (float x : train_speeds) sq += (x - mean) * (x - mean);
    const float mu = float(mean), sd = float(std::sqrt(sq / double(train_speeds.size())));
    auto scale = [&](const Matrix<float>& m) { return Matrix<float>((m - mu) / sd); };
    const Matrix<float> x_train = scale(train_w.x), x_test = scale(test_w.x);
    // The networks learn the change from the speed now (or, "direct", the speed itself).
    const Matrix<float> y_train = residual ? minus_last(scale(train_w.y), x_train) : scale(train_w.y);

    std::size_t cong[outputs]{};
    for (std::size_t i = 0; i < test_w.y.shape()[0]; ++i)
        for (std::size_t k = 0; k < outputs; ++k) cong[k] += test_w.y(i, k) < congested_below;
    std::println("PEMS08 speed: {} detectors, {} days; mean {:.1f} km/h, sd {:.1f}", v.shape()[1], days, mu, sd);
    std::println("{} training windows (days 0-{}, stride {}), {} test windows (days {}-{}, all)",
                 train_w.x.shape()[0], train_days - 1, stride, test_w.x.shape()[0], train_days, days - 1);
    std::size_t now = 0;
    for (std::size_t i = 0; i < test_w.x.shape()[0]; ++i) now += test_w.x(i, history - 1) < congested_below;
    std::println("congested test targets (< {} km/h): {} / {} / {} at 15 / 30 / 60 min; congested now: {}",
                 congested_below, cong[0], cong[1], cong[2], now);

    // Baselines.
    const std::size_t n_test = test_w.x.shape()[0];
    std::vector<float> persist(n_test * outputs);
    for (std::size_t i = 0; i < n_test; ++i)
        for (std::size_t k = 0; k < outputs; ++k) persist[i * outputs + k] = test_w.x(i, history - 1);

    std::println("targets: {}", residual ? "change from the speed now" : "speed");
    struct Row {
        std::string name;
        Errors e;
    };
    std::vector<Row> rows{{"persistence", mae(persist, test_w)}, {"historical average", mae(test_w.ha, test_w)}};

    // Each network, trained with each loss.
    auto run = [&](auto net, std::string_view name, Loss loss) {
        const auto label = std::format("{} ({})", name, loss == Loss::mse ? "MSE" : "L1");
        const auto t0 = std::chrono::steady_clock::now();
        train(net, label, loss, x_train, y_train, epochs, 2e-3);
        const double train_secs = seconds_since(t0);
        const auto t1 = std::chrono::steady_clock::now();
        const auto pred = predict(net, x_test, mu, sd, residual);
        std::println("  {}: {} parameters, training {:.0f} s, test predictions {:.1f} s", label,
                     net.parameter_count(), train_secs, seconds_since(t1));
        rows.push_back({label, mae(pred, test_w)});
    };
    for (Loss loss : {Loss::mse, Loss::l1}) {
        Rng rng{5};
        run(MLP(rng), "MLP", loss);
        run(SpeedGRU(64, rng), "GRU", loss);
    }

    std::println("\nMAE (km/h) on the test days");
    std::println("{:<20} | {:^23} | {:^23} | {:^23}", "", "all targets", "congested targets", "congested now");
    std::println("{:<20} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>7}", "model", "15 min", "30 min",
                 "60 min", "15 min", "30 min", "60 min", "15 min", "30 min", "60 min");
    for (const auto& r : rows)
        std::println("{:<20} | {:>7.2f} {:>7.2f} {:>7.2f} | {:>7.2f} {:>7.2f} {:>7.2f} | {:>7.2f} {:>7.2f} {:>7.2f}",
                     r.name, r.e.all[0], r.e.all[1], r.e.all[2], r.e.congested[0], r.e.congested[1],
                     r.e.congested[2], r.e.congested_now[0], r.e.congested_now[1], r.e.congested_now[2]);
}
