// SPDX-License-Identifier: BSD-3-Clause
// Physical sanity checks of the traffic simulator used by the examples.
#include "check.hpp"

#include <algorithm>
#include <traffic/ctm.hpp>

namespace tests {

using traffic::Corridor;

void empty_road_travel_time_is_length_over_free_speed() {
    Corridor c;
    std::mt19937 rng{1};
    const auto day = traffic::simulate_day(c, rng, false);
    const double free_min = c.length_km / c.vf * 60;   // 8.18 min on an empty road
    // At 3 a.m. there is light traffic, and speed falls a little with density
    // on the curved free-flow branch; tracers also finish on 10 s steps.
    const double tt = day.travel_time[36];
    check::that(tt >= free_min - 0.05 && tt <= free_min * 1.05);
}

void wave_speed_follows_the_fundamental_diagram() {
    Corridor c;
    check::near(c.kc(), 2000.0 / 85, 1e-9);                     // 23.5 veh/km/lane
    check::near(c.q_free(c.kc()), c.capacity, 1e-9);             // the branches meet at capacity
    check::near(c.w(), 2000.0 / (120 - 2000.0 / 85), 1e-9);      // about 20.7 km/h
    check::that(c.w() < c.vf);                                   // needed for stability
}

void peaks_create_a_queue_upstream_of_the_lane_drop() {
    Corridor c;
    const auto days = traffic::simulate_days(c, 20, 7);
    const std::size_t D = c.detectors();
    const std::size_t upstream = 11, downstream = D - 1;   // km 11.5 (queue side) and km 14.5 (after the drop)
    std::size_t queued_days = 0;
    float slowest_downstream = 200;
    for (const auto& day : days) {
        float lowest = 200;
        for (std::size_t t = 0; t < traffic::Day::intervals; ++t) {
            lowest = std::min(lowest, day.speed[t][upstream]);
            slowest_downstream = std::min(slowest_downstream, day.speed[t][downstream]);
        }
        queued_days += lowest < 50;
    }
    check::that(queued_days >= 5 && queued_days < 20);   // congestion on many days, but not all
    std::println("    queued on {} of 20 days; slowest speed past the drop {:.0f} km/h", queued_days, slowest_downstream);
}

void congestion_lengthens_travel_time() {
    Corridor c;
    const auto days = traffic::simulate_days(c, 20, 7);
    float worst = 0;
    for (const auto& day : days) worst = std::max(worst, std::ranges::max(day.travel_time));
    check::that(worst > 1.5 * c.length_km / c.vf * 60);
    std::println("    worst travel time {:.1f} min (free flow {:.1f} min)", worst, c.length_km / c.vf * 60);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
