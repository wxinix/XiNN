// SPDX-License-Identifier: BSD-3-Clause
// Freeway incident detection from space-time speed maps.
//
// Traffic operators want an alarm soon after an incident (a crash, a stalled
// vehicle) blocks lanes -- but not for the queue that forms every weekday at
// the same bottleneck. Both slow traffic down; what differs is the pattern in
// space and time. That is what convolutional networks read well.
//
// Input: the last hour at all 15 detectors of the Cell Transmission Model
// corridor, as a 2-channel image (speed, occupancy) of 15 x 12 pixels.
// Output: the probability that a traffic-disrupting incident is active now.
//
// Three detectors are compared with the standard metrics of incident
// detection -- detection rate (DR), false-alarm rate (FAR), mean time to
// detect (MTTD) -- each at the same operating point: its threshold is set on
// the training days for a false-alarm rate of 1%, and an alarm must persist
// for two consecutive intervals (as in the classic California algorithms).
//   rule   the lowest detector speed, as the score
//   MLP    a dense network on the flattened window
//   CNN    two convolution layers on the window
#include <algorithm>
#include <cmath>
#include <print>
#include <vector>
#include <traffic/ctm.hpp>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr std::size_t window = 12;   // intervals: one hour
constexpr std::size_t D = 15;        // detectors
constexpr std::size_t T = traffic::Day::intervals;

// An incident "disrupts traffic" once some detector within 2 km upstream of it
// is below 70 km/h. Incidents in light traffic remove a lane nobody needs:
// no detector can see them, and they are reported separately.
bool disrupting_at(const traffic::Day& d, std::size_t t) {
    if (!d.incident || t < d.incident_first + 1 || t > d.incident_last) return false;
    for (std::size_t k = 0; k < D; ++k) {
        const double km = k + 0.5;
        if (km <= d.incident_km && km >= d.incident_km - 2.0 && d.speed[t][k] < 70) return true;
    }
    return false;
}

bool near_incident(const traffic::Day& d, std::size_t t) {   // the incident and the half hour after it
    return d.incident && t >= d.incident_first && t <= d.incident_last + 6;
}

// The window ending at interval t, as (2, D, window): speed / 110, occupancy * 4.
void write_window(const traffic::Day& d, std::size_t t, float* out) {
    for (std::size_t k = 0; k < D; ++k)
        for (std::size_t j = 0; j < window; ++j) {
            const std::size_t tt = t + 1 - window + j;
            out[(0 * D + k) * window + j] = d.speed[tt][k] / 110.0f;
            out[(1 * D + k) * window + j] = d.occupancy[tt][k] * 4.0f;
        }
}

struct CnnDetector : Module {
    Conv2D<{.kernel = 3, .padding = 1, .activation = Activation::relu}> conv1;
    Conv2D<{.kernel = 3, .stride = 2, .padding = 1, .activation = Activation::relu}> conv2;
    Flatten flatten;
    Dense<{.activation = Activation::relu}> fc;
    Dense<> out;
    explicit CnnDetector(Rng& rng) : conv1(2, 8, rng), conv2(8, 16, rng), fc(16 * 8 * 6, 32, rng), out(32, 2, rng) {}
    auto forward(const auto& x) const { return out(fc(flatten(conv2(conv1(x))))); }
};

struct MlpDetector : Module {
    Flatten flatten;
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit MlpDetector(Rng& rng) : fc1(2 * D * window, 64, rng), fc2(64, 32, rng), out(32, 2, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(flatten(x)))); }
};

template <class Net>
void train(Net& net, const Tensor<float, 4>& x, const std::vector<std::size_t>& labels, Rng& rng) {
    const Matrix<float> y = one_hot(labels, 2);
    Adam opt(net, {.lr = 2e-3});
    const std::size_t epochs = 25, batch = 64, steps = epochs * ((labels.size() + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 0; e < epochs; ++e)
        for (auto [xb, yb] : batches(x, y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 2e-3, 1e-5));
            opt.zero_grad();
            backward(softmax_cross_entropy(net(xb), yb));
            opt.step();
        }
}

// A score for every interval of a day (higher = more likely an incident).
using Scores = std::vector<float>;

template <class Net>
Scores net_scores(const Net& net, const traffic::Day& day) {
    const std::size_t n = T - (window - 1);
    Tensor<float, 4> x(Shape{n, 2uz, D, window});
    for (std::size_t i = 0; i < n; ++i) write_window(day, i + window - 1, x.mut_flat().data() + i * 2 * D * window);
    const Matrix<float> logits = net(x);
    Scores s(T, -1e9f);
    for (std::size_t i = 0; i < n; ++i) s[i + window - 1] = logits(i, 1) - logits(i, 0);
    return s;
}

Scores rule_scores(const traffic::Day& day) {   // slower = more suspicious
    Scores s(T, -1e9f);
    for (std::size_t t = window - 1; t < T; ++t) s[t] = -std::ranges::min(day.speed[t]);
    return s;
}

// Alarm at t when the score exceeds the threshold at t and at t - 1.
bool alarm(const Scores& s, std::size_t t, float threshold) { return t > 0 && s[t] > threshold && s[t - 1] > threshold; }

// The threshold that gives a 1% false-alarm rate on the given days.
float calibrate(const std::vector<Scores>& scores, const std::vector<traffic::Day>& days, std::size_t from, std::size_t to) {
    std::vector<float> quiet;   // min(score[t], score[t-1]) over incident-free intervals
    for (std::size_t d = from; d < to; ++d)
        for (std::size_t t = window; t < T; ++t)
            if (!near_incident(days[d], t)) quiet.push_back(std::min(scores[d][t], scores[d][t - 1]));
    std::ranges::sort(quiet);
    return quiet[std::size_t(0.99 * double(quiet.size()))];
}

struct Report {
    double dr = 0, far = 0, false_per_day = 0, mttd = 0;
    std::size_t disrupting = 0, detected = 0, invisible = 0;
};

Report evaluate(const std::vector<Scores>& scores, const std::vector<traffic::Day>& days, std::size_t from,
                float threshold) {
    Report r;
    std::size_t false_alarms = 0, quiet = 0;
    double ttd = 0;
    for (std::size_t d = from; d < days.size(); ++d) {
        const auto& day = days[d];
        bool visible = false, found = false;
        for (std::size_t t = window; t < T; ++t) {
            const bool on = alarm(scores[d], t, threshold);
            if (disrupting_at(day, t)) {
                visible = true;
                if (on && !found) found = true, ttd += double(t - day.incident_first) * 5;
            } else if (!near_incident(day, t)) {
                ++quiet;
                false_alarms += on;
            }
        }
        if (day.incident) {
            if (visible) ++r.disrupting, r.detected += found;
            else ++r.invisible;
        }
    }
    r.dr = r.disrupting ? 100.0 * double(r.detected) / double(r.disrupting) : 0;
    r.far = 100.0 * double(false_alarms) / double(quiet);
    r.false_per_day = double(false_alarms) / double(days.size() - from);
    r.mttd = r.detected ? ttd / double(r.detected) : 0;
    return r;
}

}  // namespace

int main() {
    traffic::Corridor c;
    c.incident_probability = 0.8;
    c.incident_capacity_min = 0.15;   // severe: most of the road blocked
    c.incident_capacity_max = 0.35;
    const std::size_t n_days = 500, train_days = 400;
    const auto days = traffic::simulate_days(c, n_days, 31);

    // Training windows: every window of a disrupting incident, plus three
    // times as many without one -- half of them from congested periods (the
    // hard cases: the daily queue at the lane drop), half calm.
    Rng rng{8};
    std::vector<std::pair<std::size_t, std::size_t>> pos, congested, calm;
    for (std::size_t d = 0; d < train_days; ++d)
        for (std::size_t t = window - 1; t < T; ++t) {
            if (disrupting_at(days[d], t)) pos.push_back({d, t});
            else if (near_incident(days[d], t)) continue;
            else if (std::ranges::min(days[d].speed[t]) < 64) congested.push_back({d, t});
            else calm.push_back({d, t});
        }
    std::ranges::shuffle(congested, rng);
    std::ranges::shuffle(calm, rng);
    const std::size_t neg_each = 3 * pos.size() / 2;
    auto at = pos;
    at.insert(at.end(), congested.begin(), congested.begin() + std::min(congested.size(), neg_each));
    at.insert(at.end(), calm.begin(), calm.begin() + std::min(calm.size(), neg_each));

    Tensor<float, 4> x(Shape{at.size(), 2uz, D, window});
    std::vector<std::size_t> labels;
    for (std::size_t i = 0; i < at.size(); ++i) {
        write_window(days[at[i].first], at[i].second, x.mut_flat().data() + i * 2 * D * window);
        labels.push_back(disrupting_at(days[at[i].first], at[i].second));
    }
    std::println("{} training windows: {} with a disrupting incident, {} without (half congested, half calm)",
                 labels.size(), pos.size(), labels.size() - pos.size());

    CnnDetector cnn(rng);
    MlpDetector mlp(rng);
    train(cnn, x, labels, rng);
    train(mlp, x, labels, rng);

    struct Candidate { const char* name; std::vector<Scores> scores; };
    std::vector<Candidate> detectors{{"rule: lowest speed", {}}, {"MLP", {}}, {"CNN", {}}};
    for (const auto& day : days) {
        detectors[0].scores.push_back(rule_scores(day));
        detectors[1].scores.push_back(net_scores(mlp, day));
        detectors[2].scores.push_back(net_scores(cnn, day));
    }

    std::println("\ntest: {} days; every detector's threshold set on the training days for FAR = 1%", n_days - train_days);
    std::println("\n{:<20}{:>9}{:>9}{:>12}{:>11}", "detector", "DR", "FAR", "false/day", "MTTD");
    Report last;
    for (const auto& det : detectors) {
        const float threshold = calibrate(det.scores, days, 0, train_days);
        last = evaluate(det.scores, days, train_days, threshold);
        std::println("{:<20}{:>8.1f}%{:>8.2f}%{:>12.2f}{:>7.1f} min   ({} of {})", det.name, last.dr, last.far,
                     last.false_per_day, last.mttd, last.detected, last.disrupting);
    }
    std::println("\n{} further test incidents disrupted no traffic (light demand) and are not counted.", last.invisible);
    std::println("DR: disrupting incidents detected while active. FAR: share of incident-free intervals with an alarm.");
    std::println("MTTD: minutes from the start of an incident to its first alarm.");
}
