// SPDX-License-Identifier: BSD-3-Clause
// Reference solution for the chapter 5 project: travel-time prediction on a
// real PeMS corridor.
//
//   pems_travel_time META FWY DIR PM_FROM PM_TO FILE...
//
//   META      Station Metadata file (tab-separated, decompressed)
//   FWY DIR   freeway number and direction, e.g. 405 N
//   PM_FROM   corridor limits, absolute postmiles
//   PM_TO
//   FILE...   Station 5-Minute files (decompressed), one per day
//
// Example:
//   pems_travel_time d12_text_meta_2024_03_01.txt 405 N 0 12 d12_text_station_5min_2024_03_*.txt
//
// The model, features and baselines are those of examples/travel_time.cpp;
// only the data source changes. The target is the experienced travel time,
// computed from the speed field by the trajectory method.
#include <cmath>
#include <cstdlib>
#include <numbers>
#include <print>
#include <vector>
#include <xinn/xinn.hpp>

#include "pems.hpp"

using namespace xinn;

namespace {

constexpr std::size_t history = 3;
constexpr float min_observed = 50;   // % of a station's lanes really observed; below that PeMS imputed it

struct Samples {
    std::vector<float> rows, truth, instantaneous;
    std::vector<std::size_t> interval;   // of the day
    std::size_t features = 0;
};

// Samples for departures at the end of interval t, using data up to t only,
// for days in [from, to).
Samples make_samples(const pems::Series& s, const pems::Corridor& c, std::size_t from, std::size_t to) {
    const auto v = s.data.view();
    const auto obs = s.observed.view();
    const std::size_t D = c.stations.size();
    Samples out;
    out.features = history * (D + 1) + 2;
    std::vector<float> row;
    for (std::size_t t = from * 288 + history - 1; t < to * 288 && t + 1 < v.extent(0); ++t) {
        row.clear();
        bool ok = true;
        for (std::size_t lag = 0; lag < history && ok; ++lag) {
            for (std::size_t d = 0; d < D; ++d) {
                const float speed = v[t - lag, d, 2], q = obs[t - lag, d];
                ok &= !std::isnan(speed) && !std::isnan(q) && q >= min_observed;
                row.push_back(speed);
            }
            row.push_back(v[t - lag, 0, 0]);   // flow at the first station: demand arriving
            ok &= !std::isnan(row.back());
        }
        const float target = pems::experienced_travel_time(s, c, t + 1);
        if (!ok || std::isnan(target)) continue;
        const double tod = 2 * std::numbers::pi * double(t % 288) / 288;
        row.push_back(float(std::sin(tod)));
        row.push_back(float(std::cos(tod)));

        double inst = 0;
        for (std::size_t d = 0; d < D; ++d) inst += c.section_km[d] / v[t, d, 2] * 60;
        out.rows.insert(out.rows.end(), row.begin(), row.end());
        out.truth.push_back(target);
        out.instantaneous.push_back(float(inst));
        out.interval.push_back(t % 288);
    }
    return out;
}

Matrix<float> as_matrix(const std::vector<float>& v, std::size_t cols) {
    Matrix<float> m(Shape{v.size() / cols, cols});
    std::ranges::copy(v, m.mut_flat().begin());
    return m;
}

struct TravelTimeNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    TravelTimeNet(std::size_t in, Rng& rng) : fc1(in, 64, rng), fc2(64, 64, rng), out(64, 1, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

void print_row(std::string_view name, std::span<const float> pred, std::span<const float> truth, double free_tt) {
    std::vector<float> p, t;
    for (std::size_t i = 0; i < pred.size(); ++i)
        if (truth[i] > 1.2 * free_tt) p.push_back(pred[i]), t.push_back(truth[i]);
    std::print("{:<16}{:>9.2f}{:>9.1f}%", name, mean_absolute_error(pred, truth),
               mean_absolute_percentage_error(pred, truth));
    if (!t.empty()) std::print("   {:>12.2f}{:>9.1f}%", mean_absolute_error(p, t), mean_absolute_percentage_error(p, t));
    std::println("");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::println("usage: {} META FWY DIR PM_FROM PM_TO FILE...", argv[0]);
        return 1;
    }
    auto meta = pems::read_metadata(argv[1]);
    if (!meta) return std::println("{}", meta.error()), 1;
    auto corridor = pems::build_corridor(*meta, std::atoi(argv[2]), argv[3][0], std::atof(argv[4]), std::atof(argv[5]));
    if (!corridor) return std::println("{}", corridor.error()), 1;
    const auto& c = *corridor;
    std::println("{} mainline stations, {:.1f} km:", c.stations.size(), c.length_km());
    for (std::size_t i = 0; i < c.stations.size(); ++i)
        std::println("  {:>8}  PM {:7.2f}  {:>5.2f} km  {}", c.stations[i].id, c.stations[i].abs_pm, c.section_km[i],
                     c.stations[i].name);

    std::vector<std::filesystem::path> files(argv + 6, argv + argc);
    auto series = pems::read_station_5min(c, files);
    if (!series) return std::println("{}", series.error()), 1;
    const std::size_t unfilled = pems::fill_gaps(*series);
    std::println("{} days of data; {} values still missing after gap filling", series->days, unfilled);
    if (series->days < 5) return std::println("need at least 5 days"), 1;

    const std::size_t test_days = std::max<std::size_t>(1, series->days / 5), train_days = series->days - test_days;
    auto train = make_samples(*series, c, 0, train_days);
    auto test = make_samples(*series, c, train_days, series->days);
    if (train.truth.empty() || test.truth.empty()) return std::println("not enough clean samples"), 1;

    // Free-flow time from the 5th percentile of observed travel times.
    std::vector<float> sorted = train.truth;
    std::ranges::sort(sorted);
    const double free_tt = sorted[sorted.size() / 20];
    std::println("{} training and {} test departures; free-flow travel time ~{:.1f} min", train.truth.size(),
                 test.truth.size(), free_tt);

    std::vector<double> hsum(288), hn(288);
    for (std::size_t i = 0; i < train.truth.size(); ++i) hsum[train.interval[i]] += train.truth[i], hn[train.interval[i]] += 1;
    std::vector<float> historical;
    for (auto t : test.interval) historical.push_back(hn[t] ? float(hsum[t] / hn[t]) : float(free_tt));

    const Matrix<float> xtr_raw = as_matrix(train.rows, train.features), xte_raw = as_matrix(test.rows, test.features);
    const auto scaler = Standardizer<float>::fit(xtr_raw);
    const Matrix<float> x_train = scaler(xtr_raw), x_test = scaler(xte_raw);
    Matrix<float> y_train(Shape{train.truth.size(), std::size_t{1}});
    std::ranges::transform(train.truth, y_train.mut_flat().begin(), [&](float t) { return float((t - free_tt) / 5); });

    Rng rng{5};
    TravelTimeNet net(train.features, rng);
    Adam opt(net, {.lr = 2e-3, .weight_decay = 1e-4});
    const std::size_t epochs = 60, batch = 128, steps = epochs * ((train.truth.size() + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 0; e < epochs; ++e)
        for (auto [xb, yb] : batches(x_train, y_train, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 2e-3, 1e-5));
            opt.zero_grad();
            backward(mse(net(xb), yb));
            opt.step();
        }
    const Matrix<float> scaled = net(x_test);
    std::vector<float> pred;
    for (float v : scaled.flat()) pred.push_back(float(v * 5 + free_tt));

    std::println("\n{:<16}{:>9}{:>10}   {:>12}{:>10}", "", "MAE min", "MAPE", "congested MAE", "MAPE");
    print_row("instantaneous", test.instantaneous, test.truth, free_tt);
    print_row("historical", historical, test.truth, free_tt);
    print_row("neural network", pred, test.truth, free_tt);
}
