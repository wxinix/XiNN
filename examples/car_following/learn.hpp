// SPDX-License-Identifier: BSD-3-Clause
// Learning a car-following law from trajectories (used by the case study and
// by XiNN Lab).
//
//   observe(C, Delta, rng)  noisy records of a driver with a known C
//   learn(observations)     the 1959 law's lambda/M, and a neural network
//   neural_follower(L)      the network as a Follower for the simulator
#pragma once

#include <cmath>
#include <random>
#include <vector>
#include <car_following/herman1959.hpp>
#include <xinn/xinn.hpp>

namespace carfollow {

using namespace xinn;

struct Observations {
    Matrix<double> x;   // (n, 3): dv, v, gap -- Delta seconds ago
    Matrix<double> y;   // (n, 1): acceleration now, measured with noise
    std::vector<Trajectories> examples;   // the first few episodes, for display
};

// A driver with a given C is observed in 60 short episodes of a 6-car line:
// random speeds and spacings, and a lead car that brakes and accelerates at
// random. Accelerations are measured with noise, as from real trajectories.
Observations observe(double C, double delta, Rng& rng) {
    std::uniform_real_distribution<double> U(0, 1);
    std::normal_distribution<double> noise(0.0, 0.3);   // ft/s^2
    std::vector<double> xs, ys;
    std::vector<Trajectories> examples;
    for (int episode = 0; episode < 60; ++episode) {
        Scenario s{.delta = delta, .C = C, .u = 50 + 30 * U(rng), .gap = 50 + 40 * U(rng), .duration = 20, .cars = 6};
        struct Pulse { double start, length, accel; };
        std::vector<Pulse> pulses;
        for (int k = 0; k < 3; ++k) pulses.push_back({1 + 14 * U(rng), 0.5 + 2 * U(rng), (U(rng) < 0.5 ? -1 : 1) * (1 + 5 * U(rng))});
        auto lead = [&](double t) {
            double a = 0;
            for (const auto& p : pulses)
                if (t > p.start && t <= p.start + p.length) a += p.accel;
            return a;
        };
        const auto tr = simulate(s, lead, linear_follower(s.lambda_m()));
        if (episode < 3) examples.push_back(tr);
        const std::size_t delay = s.delay_steps();
        for (std::size_t i = delay; i < tr.steps; i += 10)   // every 0.1 s
            for (std::size_t n = 1; n < s.cars; ++n) {
                const std::size_t d = i - delay + 1;
                xs.insert(xs.end(), {tr.vel(n - 1, d) - tr.vel(n, d), tr.vel(n, d), tr.spacing(n, d)});
                ys.push_back(tr.acc(n, i) + noise(rng));
            }
    }
    Observations o{Matrix<double>(Shape{ys.size(), std::size_t{3}}), Matrix<double>(Shape{ys.size(), std::size_t{1}}),
                   std::move(examples)};
    std::ranges::copy(xs, o.x.mut_flat().begin());
    std::ranges::copy(ys, o.y.mut_flat().begin());
    return o;
}

// The 1959 law with its one parameter learned.
struct LinearDriver : Module {
    Param<double, 0> lambda_m{Scalar<double>(Shape<0>{}, 0.1)};
    auto forward(const auto& dv) const { return dv * lambda_m; }
};

// A neural network: acceleration from (dv, v, gap), no physics assumed.
struct NeuralDriver : Module {
    Dense<{.activation = Activation::tanh}, double> fc1, fc2;
    Dense<{}, double> out;
    explicit NeuralDriver(Rng& rng) : fc1(3, 32, rng), fc2(32, 32, rng), out(32, 1, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

struct Learned {
    double lambda_m;                 // from the linear fit
    NeuralDriver net;
    Standardizer<double> scaler;
    double a_scale;
    double net_rmse;                 // on the training data, ft/s^2
    double target_sd;                // spread of the measured accelerations, ft/s^2
};

Learned learn(const Observations& obs, Rng& rng) {
    // The one-parameter model, full batch.
    LinearDriver lin;
    Matrix<double> dv(Shape{obs.x.shape()[0], std::size_t{1}});
    for (std::size_t i = 0; i < dv.shape()[0]; ++i) dv.set(i, 0, obs.x(i, 0));
    Adam lin_opt(lin, {.lr = 0.02});
    for (int step = 0; step < 400; ++step) {
        lin_opt.zero_grad();
        backward(mse(lin(dv), obs.y));
        lin_opt.step();
    }

    // The network, mini-batches.
    Learned L{lin.lambda_m.tensor()(), NeuralDriver(rng), Standardizer<double>::fit(obs.x), 5.0, 0, 0};
    L.target_sd = Standardizer<double>::fit(obs.y).scale(0);
    const Matrix<double> x = L.scaler(obs.x), y = obs.y / L.a_scale;
    Adam opt(L.net, {.lr = 3e-3});
    const std::size_t epochs = 25, batch = 256, steps = epochs * ((x.shape()[0] + batch - 1) / batch);
    std::size_t step = 0;
    for (std::size_t e = 0; e < epochs; ++e)
        for (auto [xb, yb] : batches(x, y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 3e-3, 1e-5));
            opt.zero_grad();
            backward(mse(L.net(xb), yb));
            opt.step();
        }
    L.net_rmse = std::sqrt(Scalar<double>(mse(L.net(x), y)).eval()()) * L.a_scale;
    return L;
}

// The learned network as a car-following model: one forward pass for all followers.
Follower neural_follower(const Learned& L) {
    return [&L](std::span<const Stimulus> s, std::span<double> a) {
        Matrix<double> x(Shape{s.size(), std::size_t{3}});
        for (std::size_t j = 0; j < s.size(); ++j) {
            x.set(j, 0, s[j].dv);
            x.set(j, 1, s[j].v);
            x.set(j, 2, s[j].gap);
        }
        const Matrix<double> y = L.net(L.scaler(x));
        for (std::size_t j = 0; j < s.size(); ++j) a[j] = y(j, 0) * L.a_scale;
    };
}

}  // namespace carfollow
