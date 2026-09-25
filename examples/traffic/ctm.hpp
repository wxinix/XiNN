// SPDX-License-Identifier: BSD-3-Clause
// A freeway corridor simulated with the Cell Transmission Model (CTM).
//
// The CTM (Daganzo, 1994) is the discrete form of the LWR kinematic-wave
// model. The road is cut into cells of length dx = vf * dt, and each time
// step moves vehicles between neighbouring cells:
//
//   flow into cell i  =  min( sending(i-1), receiving(i) )
//   sending(i)        =  min( vf * k_i,  Q_i )            what cell i can pass on
//   receiving(i)      =  min( Q_i,  w * (kj_i - k_i) )    what cell i can accept
//
// The fundamental diagram has a curved free-flow branch: speed falls
// linearly from vf on an empty road to v_cap at capacity, as on real
// freeways (the textbook triangular diagram keeps speed at vf all the way to
// capacity). The congested branch is straight, with backward wave speed
// w = Q / (kj - kc). Per lane:
//
//   free flow  (k <= kc):  q = k (vf - a k),   a = (vf - v_cap) / kc,  kc = Q / v_cap
//   congested  (k >  kc):  q = w (kj - k)
//
// Sending is q(k) below kc and Q above; receiving is Q below kc and q(k)
// above. Queues form at bottlenecks and their tails travel upstream at the
// shockwave speed -- all from this one rule.
//
// This corridor has a lane drop (3 -> 2 lanes) near its end, random daily
// demand with morning and evening peaks, occasional incidents, and a
// capacity drop: a queue discharges a few percent below capacity. Loop
// detectors report 5-minute flow, occupancy and speed. Vehicles are traced
// through the evolving traffic to get their experienced travel times.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <random>
#include <vector>

namespace traffic {

struct Corridor {
    double length_km = 15.0;
    int lanes = 3;
    double bottleneck_from_km = 12.0, bottleneck_to_km = 13.0;   // lane drop to 2 lanes
    int bottleneck_lanes = 2;
    double vf = 110.0;          // km/h, empty road
    double v_cap = 85.0;        // km/h, at capacity
    double capacity = 2000.0;   // veh/h/lane
    double kj = 120.0;          // veh/km/lane
    double capacity_drop = 0.07;
    double dt_s = 10.0;
    double detector_spacing_km = 1.0;
    double incident_probability = 0.25;   // chance that a day has an incident
    double incident_capacity_min = 0.3;   // share of capacity an incident leaves, drawn
    double incident_capacity_max = 0.6;   // uniformly from [min, max]

    double kc() const { return capacity / v_cap; }   // veh/km/lane
    double a() const { return (vf - v_cap) / kc(); }
    double w() const { return capacity / (kj - kc()); }
    double q_free(double k_lane) const { return k_lane * (vf - a() * k_lane); }   // free-flow branch, per lane
    double dx_km() const { return vf * dt_s / 3600.0; }
    std::size_t cells() const { return std::size_t(std::ceil(length_km / dx_km())); }
    std::size_t detectors() const { return std::size_t(length_km / detector_spacing_km); }
};

// One simulated day.
struct Day {
    static constexpr std::size_t intervals = 288;   // 5-minute intervals
    // [interval][detector]: flow (veh/h, all lanes), occupancy (fraction), speed (km/h)
    std::vector<std::vector<float>> flow, occupancy, speed;
    // Experienced travel time over the whole corridor (minutes) for a vehicle
    // entering at the END of each interval.
    std::vector<float> travel_time;
    bool weekend = false, incident = false;
    // The incident, if any: position, and the 5-minute intervals it covers.
    double incident_km = 0;
    std::size_t incident_first = 0, incident_last = 0;   // inclusive
    double incident_capacity = 1;                          // share of capacity left
};

inline Day simulate_day(const Corridor& c, std::mt19937& rng, bool weekend) {
    std::uniform_real_distribution<double> U(0, 1);
    const std::size_t n = c.cells(), D = c.detectors();
    const double dx = c.dx_km(), dt_h = c.dt_s / 3600.0;
    const std::size_t steps_per_interval = std::size_t(300 / c.dt_s), steps = Day::intervals * steps_per_interval;

    // Lanes and capacity per cell.
    std::vector<double> Q(n), KJ(n), KC(n), L(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = (i + 0.5) * dx;
        const int lanes = (x >= c.bottleneck_from_km && x < c.bottleneck_to_km) ? c.bottleneck_lanes : c.lanes;
        L[i] = lanes;
        Q[i] = lanes * c.capacity;
        KJ[i] = lanes * c.kj;
        KC[i] = lanes * c.kc();
    }

    // Demand (veh/h entering the corridor): base plus two peaks, lower at weekends.
    auto peak = [](double t_h, double at, double width, double height) {
        return height * std::exp(-0.5 * (t_h - at) * (t_h - at) / (width * width));
    };
    const double scale = weekend ? 0.6 : 1.0;
    const double base = scale * (1600 + 600 * U(rng));
    const double am_at = 7.0 + 1.5 * U(rng), am_w = 0.5 + 0.5 * U(rng), am_h = scale * (1200 + 1800 * U(rng));
    const double pm_at = 16.5 + 1.5 * U(rng), pm_w = 0.6 + 0.6 * U(rng), pm_h = scale * (1200 + 1800 * U(rng));
    auto demand = [&](double t_h) {
        const double night = 0.35 + 0.65 * std::clamp((t_h - 4.5) / 2.0, 0.0, 1.0) * std::clamp((23.5 - t_h) / 2.5, 0.0, 1.0);
        return night * base + peak(t_h, am_at, am_w, am_h) + peak(t_h, pm_at, pm_w, pm_h);
    };

    // An incident on some days (a quarter by default): one lane-equivalent or more lost for a while.
    Day day;
    day.weekend = weekend;
    std::size_t inc_cell = 0, inc_from = steps, inc_to = steps;
    double inc_factor = 1;
    if (U(rng) < c.incident_probability) {
        day.incident = true;
        inc_cell = std::size_t(U(rng) * (n - 2));
        inc_from = std::size_t((6 + 14 * U(rng)) * 3600 / c.dt_s);
        inc_to = inc_from + std::size_t((15 + 45 * U(rng)) * 60 / c.dt_s);
        inc_factor = c.incident_capacity_min + (c.incident_capacity_max - c.incident_capacity_min) * U(rng);
        day.incident_km = (inc_cell + 0.5) * dx;
        day.incident_first = inc_from / steps_per_interval;
        day.incident_last = std::min(Day::intervals - 1, inc_to / steps_per_interval);
        day.incident_capacity = inc_factor;
    }

    std::vector<double> k(n, 0.0), y(n + 1, 0.0), v_cell(n, c.vf);
    double entry_queue = 0;   // vehicles waiting to enter (veh)

    // Detector cells and 5-minute accumulators.
    std::vector<std::size_t> det_cell(D);
    for (std::size_t d = 0; d < D; ++d) det_cell[d] = std::min(n - 1, std::size_t(((d + 0.5) * c.detector_spacing_km) / dx));
    std::vector<double> acc_count(D), acc_k(D), acc_q(D);

    // Vehicles traced for travel times: one released at the end of each
    // interval. If vehicles are waiting to enter, the tracer waits its turn.
    struct Tracer { double x; double t_start; std::size_t interval; double wait_h; };
    std::vector<Tracer> tracers;
    day.travel_time.assign(Day::intervals, 0.0f);

    std::normal_distribution<double> noise(0.0, 1.0);
    // One hour of warm-up at midnight demand, so the day starts with traffic
    // on the road instead of an empty corridor. Negative steps are not recorded.
    const long warmup = long(3600 / c.dt_s);
    for (long s_signed = -warmup; s_signed < long(steps); ++s_signed) {
        const bool recording = s_signed >= 0;
        const std::size_t s = recording ? std::size_t(s_signed) : 0;
        const double t_h = recording ? s * dt_h : 0.0;

        // Sending and receiving capacities.
        std::vector<double> S(n), R(n);
        for (std::size_t i = 0; i < n; ++i) {
            double Qi = Q[i];
            if (s >= inc_from && s < inc_to && i == inc_cell) Qi *= inc_factor;
            const bool queued = k[i] > KC[i] * 1.05;
            const double sending = k[i] <= KC[i] ? L[i] * c.q_free(k[i] / L[i]) : Q[i];
            const double receiving = k[i] <= KC[i] ? Q[i] : c.w() * (KJ[i] - k[i]);
            S[i] = std::min(sending, Qi * (queued ? 1 - c.capacity_drop : 1.0));
            R[i] = std::min(Qi, std::max(0.0, receiving));
        }
        // Flows across cell boundaries (veh/h).
        entry_queue += demand(t_h) * dt_h;
        y[0] = std::min(entry_queue / dt_h, R[0]);
        for (std::size_t i = 1; i < n; ++i) y[i] = std::min(S[i - 1], R[i]);
        y[n] = S[n - 1];   // free exit
        entry_queue -= y[0] * dt_h;
        for (std::size_t i = 0; i < n; ++i) {
            const double out = y[i + 1];
            v_cell[i] = k[i] > 1e-6 ? std::min(c.vf, out / k[i]) : c.vf;
            k[i] = std::max(0.0, k[i] + dt_h / dx * (y[i] - y[i + 1]));
        }

        if (!recording) continue;

        // Detectors.
        for (std::size_t d = 0; d < D; ++d) {
            acc_count[d] += y[det_cell[d] + 1] * dt_h;
            acc_k[d] += k[det_cell[d]];
            acc_q[d] += y[det_cell[d] + 1];
        }

        // Move tracers with the local speed; finished ones record their time.
        for (auto& tr : tracers) {
            if (tr.wait_h > 0) { tr.wait_h -= dt_h; continue; }
            const std::size_t cell = std::min(n - 1, std::size_t(tr.x / dx));
            tr.x += v_cell[cell] * dt_h;
        }
        std::erase_if(tracers, [&](const Tracer& tr) {
            if (tr.x < c.length_km) return false;
            day.travel_time[tr.interval] = float((t_h + dt_h - tr.t_start) * 60.0);
            return true;
        });

        if ((s + 1) % steps_per_interval == 0) {
            const std::size_t iv = s / steps_per_interval;
            std::vector<float> f(D), o(D), sp(D);
            for (std::size_t d = 0; d < D; ++d) {
                const double lanes = Q[det_cell[d]] / c.capacity;
                const double kbar = acc_k[d] / steps_per_interval, qbar = acc_q[d] / steps_per_interval;
                const double v = kbar > 0.05 ? std::min(c.vf, qbar / kbar) : c.vf;
                // Measurement noise, as real loop detectors have.
                f[d] = float(std::max(0.0, acc_count[d] * 12 * (1 + 0.05 * noise(rng))));
                o[d] = float(std::clamp(kbar / lanes * 0.0065 * (1 + 0.05 * noise(rng)), 0.0, 1.0));   // 6.5 m effective length
                sp[d] = float(std::clamp(v + 3.0 * noise(rng), 3.0, 120.0));
                acc_count[d] = acc_k[d] = acc_q[d] = 0;
            }
            day.flow.push_back(f);
            day.occupancy.push_back(o);
            day.speed.push_back(sp);
            tracers.push_back({0.0, t_h + dt_h, iv, y[0] > 1 ? entry_queue / y[0] : 0.0});
        }
    }
    // Vehicles still on the road at midnight: extrapolate at free-flow speed.
    for (auto& tr : tracers)
        day.travel_time[tr.interval] = float((24.0 - tr.t_start + (c.length_km - tr.x) / c.vf) * 60.0);
    return day;
}

// n days; every 7th pair of days is a weekend.
inline std::vector<Day> simulate_days(const Corridor& c, std::size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<Day> days;
    for (std::size_t d = 0; d < n; ++d) days.push_back(simulate_day(c, rng, d % 7 >= 5));
    return days;
}

}  // namespace traffic
