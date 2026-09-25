// SPDX-License-Identifier: BSD-3-Clause
// Corridor travel-time prediction.
//
// A vehicle entering the 15 km corridor now: how long will it take? The
// answer depends on traffic it has not met yet -- a queue ahead may grow
// or dissolve while it drives. Three predictors:
//
//   instantaneous   sum over sections of (length / current speed): what
//                   roadside travel-time signs typically show
//   historical      the average travel time at this time of day
//   neural network  from the last 15 minutes at all 15 detectors
//
// Data: 80 days of the Cell Transmission Model corridor (lane drop, peaks,
// incidents); each vehicle's *experienced* travel time is traced through
// the simulation. Trained on 64 days, tested on the last 16.
#include <cmath>
#include <numbers>
#include <print>
#include <vector>
#include <traffic/ctm.hpp>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t history = 3;

struct Samples {
    Matrix<float> x;
    Matrix<float> y;                          // (n, 1): travel time, minutes
    std::vector<float> truth, instantaneous;
    std::vector<std::size_t> interval;        // time of day, for the historical baseline
};

Samples make_samples(const traffic::Corridor& c, const std::vector<traffic::Day>& days, std::size_t from,
                     std::size_t to) {
    const std::size_t D = c.detectors();
    Samples s;
    std::vector<float> rows;
    for (std::size_t day = from; day < to; ++day)
        for (std::size_t t = history - 1; t < traffic::Day::intervals; ++t) {
            const auto& d = days[day];
            for (std::size_t lag = 0; lag < history; ++lag) {
                for (std::size_t k = 0; k < D; ++k) rows.push_back(d.speed[t - lag][k]);
                rows.push_back(d.flow[t - lag][0]);   // demand arriving at the entrance
            }
            const double tod = 2 * std::numbers::pi * double(t) / traffic::Day::intervals;
            rows.push_back(float(std::sin(tod)));
            rows.push_back(float(std::cos(tod)));

            double inst = 0;
            for (std::size_t k = 0; k < D; ++k) inst += c.detector_spacing_km / d.speed[t][k] * 60;
            s.instantaneous.push_back(float(inst));
            s.truth.push_back(d.travel_time[t]);
            s.interval.push_back(t);
        }
    const std::size_t n = s.truth.size();
    s.x = Matrix<float>(Shape{n, rows.size() / n});
    std::ranges::copy(rows, s.x.mut_flat().begin());
    s.y = Matrix<float>(Shape{n, std::size_t{1}});
    std::ranges::copy(s.truth, s.y.mut_flat().begin());
    return s;
}

struct TravelTimeNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit TravelTimeNet(std::size_t in, Rng& rng) : fc1(in, 64, rng), fc2(64, 64, rng), out(64, 1, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

void print_row(std::string_view name, std::span<const float> pred, const Samples& test, double free_tt) {
    std::vector<float> p_cong, t_cong;
    for (std::size_t i = 0; i < pred.size(); ++i)
        if (test.truth[i] > 1.2 * free_tt) p_cong.push_back(pred[i]), t_cong.push_back(test.truth[i]);
    std::println("{:<16}{:>9.2f}{:>9.1f}%   {:>12.2f}{:>9.1f}%", name, mean_absolute_error(pred, test.truth),
                 mean_absolute_percentage_error(pred, test.truth), mean_absolute_error(p_cong, t_cong),
                 mean_absolute_percentage_error(p_cong, t_cong));
}

}  // namespace

int main() {
    traffic::Corridor c;
    const auto days = traffic::simulate_days(c, 80, 99);
    const std::size_t train_days = 64;
    auto train = make_samples(c, days, 0, train_days);
    auto test = make_samples(c, days, train_days, days.size());
    const double free_tt = c.length_km / c.vf * 60;
    std::println("{} training samples, {} test samples; free-flow travel time {:.1f} min", train.truth.size(),
                 test.truth.size(), free_tt);

    // Historical average by time of day, from training days only.
    std::vector<double> hist_sum(traffic::Day::intervals), hist_n(traffic::Day::intervals);
    for (std::size_t i = 0; i < train.truth.size(); ++i) hist_sum[train.interval[i]] += train.truth[i], hist_n[train.interval[i]] += 1;
    std::vector<float> historical;
    for (auto t : test.interval) historical.push_back(float(hist_sum[t] / hist_n[t]));

    // The network learns the travel time in units of 5 minutes above free flow.
    const auto scaler = Standardizer<float>::fit(train.x);
    const Matrix<float> x_train = scaler(train.x), x_test = scaler(test.x);
    const Matrix<float> y_train = (train.y - float(free_tt)) / 5.0f;

    Rng rng{5};
    TravelTimeNet net(x_train.shape()[1], rng);
    Adam opt(net, {.lr = 2e-3, .weight_decay = 1e-4});
    const std::size_t epochs = 60, batch = 128;
    const std::size_t steps = epochs * ((train.truth.size() + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 1; e <= epochs; ++e) {
        double loss = 0;
        std::size_t n = 0;
        for (auto [xb, yb] : batches(x_train, y_train, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 2e-3, 1e-5));
            opt.zero_grad();
            loss += backward(mse(net(xb), yb));
            opt.step();
            ++n;
        }
        if (e % 15 == 0) std::println("epoch {:2}  loss {:.4f}", e, loss / n);
    }
    const Matrix<float> pred_scaled = net(x_test);
    std::vector<float> pred;
    for (float v : pred_scaled.flat()) pred.push_back(float(v * 5 + free_tt));

    std::size_t congested = 0;
    for (float t : test.truth) congested += t > 1.2 * free_tt;
    std::println("\ntest set: {} departures, {} of them in congestion (> 1.2 x free-flow time)\n", test.truth.size(),
                 congested);
    std::println("{:<16}{:>9}{:>10}   {:>12}{:>10}", "", "MAE min", "MAPE", "congested MAE", "MAPE");
    print_row("instantaneous", test.instantaneous, test, free_tt);
    print_row("historical", historical, test, free_tt);
    print_row("neural network", pred, test, free_tt);

    // The worst test days, around their peak, every 15 minutes: one day with
    // an incident and one without.
    const std::size_t per_day = traffic::Day::intervals - (history - 1);
    for (bool with_incident : {false, true}) {
        std::size_t best = days.size();
        float worst = 0;
        for (std::size_t d = train_days; d < days.size(); ++d)
            if (days[d].incident == with_incident && std::ranges::max(days[d].travel_time) > worst)
                worst = std::ranges::max(days[d].travel_time), best = d;
        if (best == days.size()) continue;
        const auto& tt = days[best].travel_time;
        const std::size_t peak = std::size_t(std::ranges::max_element(tt) - tt.begin());
        std::println("\ntest day {} ({}), around its peak (minutes)", best, with_incident ? "incident" : "no incident");
        std::println("{:>7}{:>10}{:>15}{:>12}{:>10}", "time", "actual", "instantaneous", "historical", "network");
        for (std::size_t t = std::max(peak, 20uz) - 18; t <= std::min(peak + 18, traffic::Day::intervals - 1); t += 3) {
            const std::size_t i = (best - train_days) * per_day + (t - (history - 1));
            std::println("{:>4}:{:02}{:>10.1f}{:>15.1f}{:>12.1f}{:>10.1f}", t / 12, (t % 12) * 5, test.truth[i],
                         test.instantaneous[i], historical[i], pred[i]);
        }
    }
}
