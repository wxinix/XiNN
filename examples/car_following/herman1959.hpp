// SPDX-License-Identifier: BSD-3-Clause
// Car following with a response delay: the model of
//
//   R. Herman, E. W. Montroll, R. B. Potts, R. W. Rothery (1959).
//   Traffic dynamics: analysis of stability in car following.
//   Operations Research 7(1), 86-106.
//
// Each follower n accelerates in proportion to how much faster its leader
// was going, Delta seconds ago:
//
//   a_n(t) = (lambda / M) * [ v_{n-1}(t - Delta) - v_n(t - Delta) ]
//
// Everything depends on one number, C = lambda * Delta / M:
//
//   local stability (two cars)       C < 1/e   spacing settles without oscillating
//                                    C < pi/2  oscillations die out
//                                    C > pi/2  oscillations grow
//   asymptotic stability (a line)    C < 1/2   a disturbance shrinks down the line
//                                    C > 1/2   it grows from car to car
//
// Units are the paper's: feet and seconds. Time steps follow the paper's
// numerical calculations: explicit Euler with dt = 0.01 s, the delay counted
// in whole steps, and followers holding their initial speed until the first
// delay has passed.
#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <span>
#include <vector>

namespace carfollow {

struct Scenario {
    double delta = 1.5;                  // response delay (s)
    double C = 1 / std::numbers::e;      // lambda * Delta / M
    double u = 70;                       // initial speed of every car (ft/s)
    double gap = 70;                     // initial spacing (ft)
    double duration = 20;                // simulated time (s)
    double dt = 0.01;                    // time step (s)
    std::size_t cars = 2;

    double lambda_m() const { return C / delta; }   // 1/s
    std::size_t steps() const { return std::size_t(std::lround(duration / dt)); }
    std::size_t delay_steps() const { return std::size_t(std::lround(delta / dt)); }
};

// The paper's lead-car manoeuvre: brake at 6 ft/s^2 from t = 2 to 4 s, then
// accelerate at 6 ft/s^2 from 4 to 6 s, back to the initial speed.
inline double lead_pulse(double t) {
    if (t > 2.0 && t <= 4.0) return -6.0;
    if (t > 4.0 && t <= 6.0) return 6.0;
    return 0.0;
}

// What a follower perceives, Delta seconds ago.
struct Stimulus {
    double dv;    // leader's speed minus own speed (ft/s)
    double v;     // own speed (ft/s)
    double gap;   // spacing to the leader (ft)
};

// A car-following model: the accelerations of all followers at one time
// step, from their delayed stimuli. Batched, so a neural network can
// evaluate the whole line of cars in one forward pass.
using Follower = std::function<void(std::span<const Stimulus>, std::span<double>)>;

// The 1959 model.
inline Follower linear_follower(double lambda_m) {
    return [lambda_m](std::span<const Stimulus> s, std::span<double> a) {
        for (std::size_t j = 0; j < s.size(); ++j) a[j] = lambda_m * s[j].dv;
    };
}

// a, v, x of every car at every step; car 0 leads.
struct Trajectories {
    std::size_t cars = 0, steps = 0;
    double dt = 0;
    std::vector<double> a, v, x;   // row-major: [car * (steps + 1) + i]

    double& acc(std::size_t car, std::size_t i) { return a[car * (steps + 1) + i]; }
    double& vel(std::size_t car, std::size_t i) { return v[car * (steps + 1) + i]; }
    double& pos(std::size_t car, std::size_t i) { return x[car * (steps + 1) + i]; }
    double acc(std::size_t car, std::size_t i) const { return a[car * (steps + 1) + i]; }
    double vel(std::size_t car, std::size_t i) const { return v[car * (steps + 1) + i]; }
    double pos(std::size_t car, std::size_t i) const { return x[car * (steps + 1) + i]; }
    double time(std::size_t i) const { return double(i) * dt; }
    // Spacing from car n-1 to car n.
    double spacing(std::size_t n, std::size_t i) const { return pos(n - 1, i) - pos(n, i); }
};

inline Trajectories simulate(const Scenario& s, const std::function<double(double)>& lead, const Follower& follow) {
    Trajectories tr{s.cars, s.steps(), s.dt, {}, {}, {}};
    const std::size_t len = (tr.steps + 1) * s.cars, delay = s.delay_steps();
    tr.a.assign(len, 0.0);
    tr.v.assign(len, 0.0);
    tr.x.assign(len, 0.0);
    for (std::size_t n = 0; n < s.cars; ++n) {
        tr.pos(n, 0) = -double(n) * s.gap;
        tr.vel(n, 0) = s.u;
    }
    for (std::size_t i = 0; i <= tr.steps; ++i) tr.acc(0, i) = lead(tr.time(i));

    std::vector<Stimulus> stim(s.cars > 0 ? s.cars - 1 : 0);
    std::vector<double> out(stim.size());
    for (std::size_t i = 0; i < tr.steps; ++i) {
        // Stimuli Delta ago; before the first delay has passed, nothing has changed yet.
        for (std::size_t n = 1; n < s.cars; ++n) {
            if (i >= delay) {
                const std::size_t d = i - delay + 1;
                stim[n - 1] = {tr.vel(n - 1, d) - tr.vel(n, d), tr.vel(n, d), tr.spacing(n, d)};
            } else {
                stim[n - 1] = {0.0, s.u, s.gap};
            }
        }
        follow(stim, out);
        for (std::size_t n = 0; n < s.cars; ++n) {
            if (n > 0) tr.acc(n, i) = out[n - 1];
            tr.vel(n, i + 1) = tr.vel(n, i) + tr.acc(n, i) * s.dt;
            tr.pos(n, i + 1) = tr.pos(n, i) + tr.vel(n, i) * s.dt;
        }
    }
    return tr;
}

// ---- measuring stability ----------------------------------------------------------------------

// How the spacing of car n (to car n-1) returns to its initial value.
struct Settling {
    std::size_t crossings;   // sign changes of (spacing - gap) after the lead car's manoeuvre
    double early, late;      // largest |spacing - gap| in the first and last third after it
    double growth() const { return early > 0 ? late / early : 0.0; }
};

inline Settling settling(const Trajectories& tr, std::size_t n, double gap, double from_time = 6.0) {
    const std::size_t i0 = std::size_t(from_time / tr.dt), i1 = tr.steps, third = (i1 - i0) / 3;
    Settling s{0, 0, 0};
    int last_sign = 0;
    for (std::size_t i = i0; i <= i1; ++i) {
        const double d = tr.spacing(n, i) - gap;
        const int sign = d > 0.05 ? 1 : d < -0.05 ? -1 : 0;   // ignore tiny wobbles
        if (sign != 0) {
            if (last_sign != 0 && sign != last_sign) ++s.crossings;
            last_sign = sign;
        }
        if (i < i0 + third) s.early = std::max(s.early, std::abs(d));
        if (i >= i1 - third) s.late = std::max(s.late, std::abs(d));
    }
    return s;
}

// A plain-words verdict on a two-car response. A single overshoot already
// counts as oscillation: the spacing passed its initial value and came back.
inline const char* local_verdict(const Settling& s) {
    if (s.growth() > 1.0) return "oscillation grows";
    if (s.crossings == 0) return "no oscillation";
    return "damped oscillation";
}

// Largest |spacing - gap| of each follower: how a disturbance travels down the line.
inline std::vector<double> disturbance_by_car(const Trajectories& tr, double gap) {
    std::vector<double> out;
    for (std::size_t n = 1; n < tr.cars; ++n) {
        double m = 0;
        for (std::size_t i = 0; i <= tr.steps; ++i) m = std::max(m, std::abs(tr.spacing(n, i) - gap));
        out.push_back(m);
    }
    return out;
}

// The first time any spacing falls to zero, or a negative number if none does.
inline double first_collision(const Trajectories& tr) {
    for (std::size_t i = 0; i <= tr.steps; ++i)
        for (std::size_t n = 1; n < tr.cars; ++n)
            if (tr.spacing(n, i) <= 0) return tr.time(i);
    return -1;
}

}  // namespace carfollow
