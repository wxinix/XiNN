// SPDX-License-Identifier: BSD-3-Clause
// The car-following simulator against the stability theory of Herman et al. (1959).
#include "check.hpp"

#include <car_following/herman1959.hpp>
#include <numbers>

using namespace carfollow;

namespace {

Trajectories run(double C, std::size_t cars, double duration = 40, double delta = 1.5, double u = 70,
                 double gap = 70) {
    Scenario s{.delta = delta, .C = C, .u = u, .gap = gap, .duration = duration, .cars = cars};
    return simulate(s, lead_pulse, linear_follower(s.lambda_m()));
}

}  // namespace

namespace tests {

void the_lead_car_returns_to_its_speed() {
    const auto tr = run(0.5, 2, 20);
    check::near(tr.vel(0, tr.steps), 70, 1e-9);            // -6 for 2 s, then +6 for 2 s
    check::near(tr.vel(0, 400), 70 - 6 * 2, 0.07);         // at t = 4 s: 12 ft/s slower
}

void nothing_happens_before_the_first_delay() {
    const auto tr = run(0.5, 3, 5);
    for (std::size_t i = 0; i < 150; ++i) {   // Delta = 1.5 s = 150 steps
        check::equal(tr.acc(1, i), 0.0);
        check::equal(tr.acc(2, i), 0.0);
    }
}

void followers_settle_back_to_the_initial_state_when_stable() {
    const auto tr = run(0.5, 4, 80);
    for (std::size_t n = 1; n < 4; ++n) {
        check::near(tr.vel(n, tr.steps), 70, 1e-3);
        check::near(tr.spacing(n, tr.steps), 70, 0.05);
    }
}

void local_stability_regimes() {
    const double pi_2 = std::numbers::pi / 2, inv_e = 1 / std::numbers::e;
    check::equal(std::string(local_verdict(settling(run(0.30, 2), 1, 70))), std::string("no oscillation"));
    check::equal(std::string(local_verdict(settling(run(inv_e, 2), 1, 70))), std::string("no oscillation"));
    check::equal(std::string(local_verdict(settling(run(0.80, 2), 1, 70))), std::string("damped oscillation"));
    check::equal(std::string(local_verdict(settling(run(pi_2 + 0.05, 2), 1, 70))), std::string("oscillation grows"));
}

void string_stability_threshold_is_one_half() {
    // Below C = 1/2 the disturbance shrinks from car to car; above it, it grows.
    const auto stable = disturbance_by_car(run(0.40, 8, 60), 70);
    const auto unstable = disturbance_by_car(run(0.75, 8, 60), 70);
    check::that(stable.back() < stable.front());
    check::that(unstable.back() > unstable.front());
}

void figure_6_ends_in_a_collision() {
    // C = 0.8, Delta = 2 s, 9 cars at 40 ft/s, 40 ft apart.
    const auto tr = run(0.8, 9, 30, 2.0, 40, 40);
    check::that(first_collision(tr) > 0);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
