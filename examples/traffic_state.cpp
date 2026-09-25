// SPDX-License-Identifier: BSD-3-Clause
// Traffic state prediction: free, slowing, or congested, 15 minutes ahead.
//
// Input: the last 30 minutes (6 intervals) of flow, occupancy and speed at a
// loop detector, plus the time of day. Output: the detector's state three
// intervals later (and, for comparison, 30 and 60 minutes later), by speed:
//
//   free        v >= 88 km/h  (55 mph)
//   slowing     64 <= v < 88  (40-55 mph)
//   congested   v < 64 km/h   (40 mph)
//
// Two data sets, one model:
//   1. synthetic: a Cell Transmission Model corridor with a lane drop (60 days)
//   2. real:      PEMS08, 170 Caltrans detectors, Jul-Aug 2016 (62 days)
// The baseline is persistence -- "the state in 15 minutes is the state now" --
// which is hard to beat at short horizons.
#include <cmath>
#include <numbers>
#include <print>
#include <span>
#include <string>
#include <vector>
#include <traffic/ctm.hpp>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t history = 6, features = 3 * history + 2;
const std::vector<std::string> state_names{"free", "slowing", "congested"};

std::size_t state_of(float speed_kmh) { return speed_kmh >= 88 ? 0 : speed_kmh >= 64 ? 1 : 2; }

// Detector series in common units: (interval, detector, [flow veh/h, occupancy, speed km/h]).
struct Series {
    Tensor<float, 3> data;
    std::size_t per_day = 288;
    std::size_t days() const { return data.shape()[0] / per_day; }
};

struct Samples {
    Matrix<float> x;
    std::vector<std::size_t> y, persistence;
};

// Every `stride`-th (interval, detector) pair whose day is in [day_from, day_to).
Samples make_samples(const Series& s, std::size_t day_from, std::size_t day_to, std::size_t stride,
                     std::size_t horizon) {
    const std::size_t D = s.data.shape()[1];
    const auto v = s.data.view();
    std::vector<float> rows;
    Samples out;
    std::size_t counter = 0;
    for (std::size_t t = day_from * s.per_day + history - 1; t + horizon < day_to * s.per_day; ++t)
        for (std::size_t d = 0; d < D; ++d) {
            if (counter++ % stride) continue;
            for (std::size_t lag = 0; lag < history; ++lag)
                for (std::size_t f = 0; f < 3; ++f) rows.push_back(v[t - lag, d, f]);
            const double tod = 2 * std::numbers::pi * double(t % s.per_day) / double(s.per_day);
            rows.push_back(float(std::sin(tod)));
            rows.push_back(float(std::cos(tod)));
            out.y.push_back(state_of(v[t + horizon, d, 2]));   // the state `horizon` intervals ahead
            out.persistence.push_back(state_of(v[t, d, 2]));
        }
    out.x = Matrix<float>(Shape{out.y.size(), features});
    std::ranges::copy(rows, out.x.mut_flat().begin());
    return out;
}

struct StateNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit StateNet(Rng& rng) : fc1(features, 64, rng), fc2(64, 64, rng), out(64, 3, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

std::vector<std::size_t> predict(const StateNet& net, const Matrix<float>& x) {
    std::vector<std::size_t> out, idx;
    for (std::size_t start = 0; start < x.shape()[0]; start += 4096) {
        idx.clear();
        for (std::size_t i = start; i < std::min(start + 4096, x.shape()[0]); ++i) idx.push_back(i);
        auto p = argmax_rows(net(gather_rows(x, idx)));
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

struct Result {
    std::size_t minutes;
    ConfusionMatrix persistence, plain, weighted;
    double persistence_accuracy, plain_accuracy, weighted_accuracy;
};

// Train one network; `weight_power` 0 gives plain cross-entropy, 1 fully
// balanced class weights, 0.5 their square root (a middle way).
std::vector<std::size_t> train_and_predict(const Matrix<float>& x_train, std::span<const std::size_t> y,
                                           const Matrix<float>& x_test, double weight_power) {
    auto w = balanced_class_weights(y, 3);
    for (double& wi : w) wi = std::pow(wi, weight_power);
    const Matrix<float> y_train = one_hot(y, 3, w);

    Rng rng{11};
    StateNet net(rng);
    Adam opt(net, {.lr = 2e-3});
    const std::size_t epochs = 4, batch = 256, steps = epochs * ((y.size() + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 1; e <= epochs; ++e) {
        for (auto [xb, yb] : batches(x_train, y_train, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 2e-3, 1e-5));
            opt.zero_grad();
            backward(softmax_cross_entropy(net(xb), yb));
            opt.step();
        }
    }
    return predict(net, x_test);
}

Result run(const Series& s, std::size_t stride, std::size_t horizon, bool verbose, double weight_power) {
    const std::size_t test_days = s.days() / 5, train_days = s.days() - test_days;   // split by time
    auto train = make_samples(s, 0, train_days, stride, horizon);
    auto test = make_samples(s, train_days, s.days(), stride, horizon);

    if (verbose) {
        std::size_t counts[3]{};
        for (auto c : train.y) ++counts[c];
        std::println("{} training samples ({} days), {} test samples ({} days)", train.y.size(), train_days,
                     test.y.size(), test_days);
        std::println("states: free {:.1f}%, slowing {:.1f}%, congested {:.1f}%", 100.0 * counts[0] / train.y.size(),
                     100.0 * counts[1] / train.y.size(), 100.0 * counts[2] / train.y.size());
    }

    const auto scaler = Standardizer<float>::fit(train.x);   // statistics from training days only
    const Matrix<float> x_train = scaler(train.x), x_test = scaler(test.x);

    const auto plain = train_and_predict(x_train, train.y, x_test, 0.0);
    const auto weighted = train_and_predict(x_train, train.y, x_test, weight_power);
    return {horizon * 5,
            ConfusionMatrix(test.persistence, test.y, 3),
            ConfusionMatrix(plain, test.y, 3),
            ConfusionMatrix(weighted, test.y, 3),
            accuracy(test.persistence, test.y),
            accuracy(plain, test.y),
            accuracy(weighted, test.y)};
}

void report(std::string_view title, const Series& s, std::size_t stride, double weight_power) {
    std::println("\n=== {} ===", title);
    std::vector<Result> results;
    for (std::size_t h : {3uz, 6uz, 12uz}) results.push_back(run(s, stride, h, h == 3, weight_power));
    std::println("\nnetwork: plain cross-entropy;  weighted: class weights (balanced ^ {})", weight_power);
    std::println("\n{:>9} | {:>30} | {:>30} | {:>30}", "ahead", "accuracy (%)", "macro F1", "congested F1");
    std::println("{:>9} | {:>10}{:>10}{:>10} | {:>10}{:>10}{:>10} | {:>10}{:>10}{:>10}", "", "persist", "network",
                 "weighted", "persist", "network", "weighted", "persist", "network", "weighted");
    for (const auto& r : results)
        std::println("{:>5} min | {:>10.2f}{:>10.2f}{:>10.2f} | {:>10.3f}{:>10.3f}{:>10.3f} | {:>10.3f}{:>10.3f}{:>10.3f}",
                     r.minutes, 100 * r.persistence_accuracy, 100 * r.plain_accuracy, 100 * r.weighted_accuracy,
                     r.persistence.macro_f1(), r.plain.macro_f1(), r.weighted.macro_f1(), r.persistence.f1(2),
                     r.plain.f1(2), r.weighted.f1(2));
    std::print("\nweighted network, {} min ahead, test set:\n{}", results[1].minutes,
               results[1].weighted.table(state_names));
}

Series synthetic() {
    traffic::Corridor c;
    const auto days = traffic::simulate_days(c, 60, 2026);
    const std::size_t D = c.detectors();
    Series s{Tensor<float, 3>(Shape{days.size() * traffic::Day::intervals, D, std::size_t{3}})};
    auto m = s.data.mut();
    for (std::size_t day = 0; day < days.size(); ++day)
        for (std::size_t t = 0; t < traffic::Day::intervals; ++t)
            for (std::size_t d = 0; d < D; ++d) {
                const std::size_t row = day * traffic::Day::intervals + t;
                m[row, d, 0] = days[day].flow[t][d];
                m[row, d, 1] = days[day].occupancy[t][d];
                m[row, d, 2] = days[day].speed[t][d];
            }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    const double weight_power = argc > 1 ? std::stod(argv[1]) : 0.5;
    report("synthetic corridor (Cell Transmission Model)", synthetic(), 1, weight_power);

    auto pems = datasets::load_pems08(XINN_DATA_DIR "/pems08");
    if (!pems) return std::println("\nPEMS08 skipped: {}", pems.error()), 0;
    // To common units: flow veh/5 min -> veh/h, speed mph -> km/h.
    Series real{Tensor<float, 3>(pems->data.shape())};
    auto src = pems->data.view();
    auto dst = real.data.mut();
    for (std::size_t t = 0; t < src.extent(0); ++t)
        for (std::size_t d = 0; d < src.extent(1); ++d) {
            dst[t, d, 0] = src[t, d, 0] * 12;
            dst[t, d, 1] = src[t, d, 1];
            dst[t, d, 2] = src[t, d, 2] * 1.609344f;
        }
    report("real detectors (PEMS08, Caltrans District 8)", real, 2, weight_power);
}
