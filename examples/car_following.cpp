// SPDX-License-Identifier: BSD-3-Clause
// Stability in car following: the numerical calculations of Herman,
// Montroll, Potts & Rothery (1959), and what happens when the car-following
// law is learned from data instead.
//
//   car_following [output directory]
//
// Part 1 reproduces the paper's Figures 3-6 with the 1959 model and writes
// them as CSV files for plotting.
// Part 2 records noisy trajectories of drivers with a known C, learns their
// behaviour -- once as the 1959 model's single parameter lambda/M, once as a
// neural network that knows nothing about car following -- and puts each
// learned driver back into the paper's scenarios: does it show the same
// stability, or instability, as the real one?
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <print>
#include <random>
#include <vector>
#include <car_following/herman1959.hpp>
#include <car_following/learn.hpp>
#include <xinn/xinn.hpp>

using namespace xinn;
using namespace carfollow;

namespace {

const double inv_e = 1 / std::numbers::e, half_pi = std::numbers::pi / 2;

// ---- Part 1: the paper's figures ------------------------------------------------------------

void write_csv(const std::filesystem::path& path, const std::vector<std::string>& header,
               const std::vector<std::vector<double>>& columns) {
    std::ofstream f(path);
    for (std::size_t c = 0; c < header.size(); ++c) f << (c ? "," : "") << header[c];
    f << "\n";
    for (std::size_t r = 0; r < columns[0].size(); ++r) {
        for (std::size_t c = 0; c < columns.size(); ++c) f << (c ? "," : "") << columns[c][r];
        f << "\n";
    }
}

std::vector<double> times(const Trajectories& tr) {
    std::vector<double> t;
    for (std::size_t i = 0; i <= tr.steps; ++i) t.push_back(tr.time(i));
    return t;
}

void figures(const std::filesystem::path& out) {
    std::println("Part 1: the 1959 model (Delta = 1.5 s unless noted)\n");

    // Figure 3: two cars, C = 1/e.
    Scenario s3{.C = inv_e, .duration = 20, .cars = 2};
    const auto f3 = simulate(s3, lead_pulse, linear_follower(s3.lambda_m()));
    std::vector<double> a1, a2, v1, v2, dv, gap;
    for (std::size_t i = 0; i <= f3.steps; ++i) {
        a1.push_back(f3.acc(0, i)), a2.push_back(f3.acc(1, i));
        v1.push_back(f3.vel(0, i) - 70), v2.push_back(f3.vel(1, i) - 70);
        dv.push_back(f3.vel(0, i) - f3.vel(1, i)), gap.push_back(f3.spacing(1, i));
    }
    write_csv(out / "fig3.csv", {"t", "a1", "a2", "v1_minus_u", "v2_minus_u", "v1_minus_v2", "x1_minus_x2"},
              {times(f3), a1, a2, v1, v2, dv, gap});
    std::println("Figure 3 (C = 1/e): closest approach {:.1f} ft, largest speed difference {:.1f} ft/s",
                 std::ranges::min(gap), std::ranges::min(dv));

    // Figure 4: two cars, several C.
    std::println("\nFigure 4: two cars, spacing after the lead car's manoeuvre (40 s)");
    std::println("  {:>6}  {:<22} {:>10} {:>8}   theory", "C", "simulation", "crossings", "growth");
    std::vector<std::vector<double>> cols;
    std::vector<std::string> header{"t"};
    for (double C : {0.50, 0.80, 1.57, 1.60}) {
        Scenario s{.C = C, .duration = 40, .cars = 2};
        const auto tr = simulate(s, lead_pulse, linear_follower(s.lambda_m()));
        const auto st = settling(tr, 1, s.gap);
        const char* theory = C < inv_e ? "no oscillation" : C < half_pi ? "damped oscillation" : "oscillation grows";
        std::println("  {:>6.2f}  {:<22} {:>10} {:>8.2f}   {}", C, local_verdict(st), st.crossings, st.growth(), theory);
        if (cols.empty()) cols.push_back(times(tr));
        std::vector<double> d;
        for (std::size_t i = 0; i <= tr.steps; ++i) d.push_back(tr.spacing(1, i) - s.gap);
        cols.push_back(d);
        header.push_back(std::format("C_{:.2f}", C));
    }
    write_csv(out / "fig4.csv", header, cols);

    // Figure 5: eight cars.
    std::println("\nFigure 5: 8 cars, largest spacing disturbance of each follower (ft)");
    for (double C : {inv_e, 0.50, 0.75}) {
        Scenario s{.C = C, .duration = 60, .cars = 8};
        const auto tr = simulate(s, lead_pulse, linear_follower(s.lambda_m()));
        const auto d = disturbance_by_car(tr, s.gap);
        std::print("  C = {:.3f}:", C);
        for (double x : d) std::print(" {:5.1f}", x);
        std::println("   last/first {:.2f}  ({})", d.back() / d.front(),
                     C < 0.5 ? "shrinks: asymptotically stable" : C > 0.5 ? "grows: unstable" : "marginal");
        std::vector<std::vector<double>> c5{times(tr)};
        std::vector<std::string> h5{"t"};
        for (std::size_t n = 1; n < tr.cars; ++n) {
            std::vector<double> sp;
            for (std::size_t i = 0; i <= tr.steps; ++i) sp.push_back(tr.spacing(n, i));
            c5.push_back(sp);
            h5.push_back(std::format("gap_{}_{}", n, n + 1));
        }
        write_csv(out / std::format("fig5_C{:.3f}.csv", C), h5, c5);
    }

    // Figure 6: nine cars, C = 0.8, Delta = 2 s.
    Scenario s6{.delta = 2.0, .C = 0.8, .u = 40, .gap = 40, .duration = 30, .cars = 9};
    const auto f6 = simulate(s6, lead_pulse, linear_follower(s6.lambda_m()));
    std::vector<std::vector<double>> c6{times(f6)};
    std::vector<std::string> h6{"t"};
    for (std::size_t n = 0; n < f6.cars; ++n) {
        std::vector<double> rel;
        for (std::size_t i = 0; i <= f6.steps; ++i) rel.push_back(f6.pos(n, i) - s6.u * f6.time(i));
        c6.push_back(rel);
        h6.push_back(std::format("car_{}", n + 1));
    }
    write_csv(out / "fig6.csv", h6, c6);
    std::println("\nFigure 6 (9 cars, C = 0.8, Delta = 2 s): first collision at t = {:.2f} s", first_collision(f6));
}

void learned_drivers() {
    std::println("\nPart 2: drivers learned from noisy trajectories, put back into the paper's scenarios\n");
    Rng rng{1959};

    struct Case { const char* name; double C, delta; std::size_t cars; double u, gap, duration; };
    const Case cases[] = {
        {"Fig 4, two cars", 0.50, 1.5, 2, 70, 70, 40},  {"Fig 4, two cars", 0.80, 1.5, 2, 70, 70, 40},
        {"Fig 4, two cars", 1.57, 1.5, 2, 70, 70, 40},  {"Fig 4, two cars", 1.60, 1.5, 2, 70, 70, 40},
        {"Fig 5, 8 cars", inv_e, 1.5, 8, 70, 70, 60},   {"Fig 5, 8 cars", 0.50, 1.5, 8, 70, 70, 60},
        {"Fig 5, 8 cars", 0.75, 1.5, 8, 70, 70, 60},    {"Fig 6, 9 cars", 0.80, 2.0, 9, 40, 40, 30},
    };

    std::println("{:<16}{:>6} | {:>7}{:>9} | {:<30} | {:<30} | {:<30}", "scenario", "C", "C fit", "NN rmse",
                 "true driver", "fitted 1959 model", "neural network");
    for (const auto& c : cases) {
        const auto learned = learn(observe(c.C, c.delta, rng), rng);
        Scenario s{.delta = c.delta, .C = c.C, .u = c.u, .gap = c.gap, .duration = c.duration, .cars = c.cars};

        auto describe_run = [&](const Follower& f) -> std::string {
            const auto tr = simulate(s, lead_pulse, f);
            if (c.cars == 2) {
                const auto st = settling(tr, 1, s.gap);
                return std::format("{} (x{:.2f})", local_verdict(st), st.growth());
            }
            if (const double t = first_collision(tr); t > 0) return std::format("collision at {:.1f} s", t);
            const auto d = disturbance_by_car(tr, s.gap);
            return std::format("last/first {:.2f} ({})", d.back() / d.front(), d.back() < d.front() ? "shrinks" : "grows");
        };

        std::println("{:<16}{:>6.3f} | {:>7.3f}{:>9.2f} | {:<30} | {:<30} | {:<30}", c.name, c.C,
                     learned.lambda_m * c.delta, learned.net_rmse,
                     describe_run(linear_follower(s.lambda_m())),
                     describe_run(linear_follower(learned.lambda_m)), describe_run(neural_follower(learned)));
    }
    std::println("\n(C fit = learned lambda/M x Delta. NN rmse = the network's training error (ft/s^2); the");
    std::println(" measurement noise alone is 0.30, so 0.30 is a perfect fit. For two cars, (xg) is the growth of the late oscillation.)");
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path out = argc > 1 ? argv[1] : "car_following_out";
    std::filesystem::create_directories(out);
    figures(out);
    learned_drivers();
    std::println("\nCSV files for plotting: {}", std::filesystem::absolute(out).string());
}
