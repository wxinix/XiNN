// SPDX-License-Identifier: BSD-3-Clause
// Fitting the fundamental diagram of traffic flow.
//
// Loop detectors on a freeway lane measure density k (veh/km) and speed v
// (km/h). Their relation v(k) is the fundamental diagram; flow is q = k·v
// (veh/h). The key numbers for traffic engineering follow from it:
//   free-flow speed  vf   speed on an empty road
//   critical density kc   the density at which flow peaks
//   capacity         qmax the peak flow
//
// The synthetic "detector data" comes from Drake's model,
//   v = vf · exp(-(k/kc)² / 2),  vf = 110 km/h, kc = 35 veh/km,
// plus measurement noise. Three models are then fitted with the same
// machinery -- Params, expressions, backward():
//
//   1. Greenshields  v = vf·(1 - k/kj)            2 physical parameters, linear
//   2. Drake         v = vf·exp(-(k/kc)²/2)       2 physical parameters, right form
//   3. an MLP        v = f(k)                     ~300 weights, no traffic knowledge
//
// A parametric model yields interpretable numbers; a neural network yields
// only a curve. Both are just expressions to autograd.
#include <algorithm>
#include <cmath>
#include <print>
#include <random>
#include <tuple>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

constexpr float true_vf = 110.0f;   // km/h
constexpr float true_kc = 35.0f;    // veh/km
constexpr float k_scale = 100.0f;   // densities and speeds are scaled to ~[0, 1.5]
constexpr float v_scale = 100.0f;   // for training, and scaled back for reporting

float drake(float k) { return true_vf * std::exp(-0.5f * (k / true_kc) * (k / true_kc)); }

Scalar<float> scalar_param_init(float v) { return Scalar<float>(Shape<0>{}, v); }

// Plain gradient descent over any set of Params.
template <class Loss, class... Ps>
float fit(Loss&& loss, float lr, int steps, Ps&... params) {
    float last = 0;
    for (int s = 0; s < steps; ++s) {
        (params.zero_grad(), ...);
        last = backward(loss());
        template for (auto& p : std::tie(params...)) p.assign(p - lr * p.grad());
    }
    return last;
}

// The flow-maximizing density of a speed curve, found on a grid.
template <class SpeedFn>
std::pair<float, float> capacity(SpeedFn&& v) {
    float best_k = 0, best_q = 0;
    for (float k = 1; k <= 150; k += 0.5f)
        if (float q = k * v(k); q > best_q) best_q = q, best_k = k;
    return {best_k, best_q};
}

}  // namespace

int main() {
    Rng rng{2026};

    // ---- detector data: 600 (density, speed) pairs ---------------------------
    const std::size_t n = 600;
    Vector<float> k(Shape{n}), v(Shape{n});
    {
        std::uniform_real_distribution<float> density(2.0f, 140.0f);
        std::normal_distribution<float> noise(0.0f, 4.0f);   // km/h
        auto ks = k.mut_flat();
        auto vs = v.mut_flat();
        for (std::size_t i = 0; i < n; ++i) {
            ks[i] = density(rng) / k_scale;
            vs[i] = std::max(0.0f, drake(ks[i] * k_scale) + noise(rng)) / v_scale;
        }
    }
    auto rmse_kmh = [&](float mse_scaled) { return std::sqrt(mse_scaled) * v_scale; };

    // ---- 1. Greenshields: v = vf (1 - k / kj) ----------------------------------
    Param gs_vf(scalar_param_init(1.0f));   // scaled: 100 km/h
    Param gs_kj(scalar_param_init(1.5f));   // scaled: 150 veh/km
    auto greenshields = [&](const auto& kk) { return gs_vf * (1 - kk / gs_kj); };
    const float gs_loss = fit([&] { return mse(greenshields(k), v); }, 0.5f, 4000, gs_vf, gs_kj);
    const float gvf = gs_vf.tensor()() * v_scale, gkj = gs_kj.tensor()() * k_scale;

    // ---- 2. Drake: v = vf exp(-(k / kc)^2 / 2) ----------------------------------
    Param dk_vf(scalar_param_init(1.0f));
    Param dk_kc(scalar_param_init(0.5f));   // scaled: 50 veh/km
    auto drake_model = [&](const auto& kk) { return dk_vf * exp(square(kk / dk_kc) * -0.5f); };
    const float dk_loss = fit([&] { return mse(drake_model(k), v); }, 0.5f, 4000, dk_vf, dk_kc);
    const float dvf = dk_vf.tensor()() * v_scale, dkc = dk_kc.tensor()() * k_scale;

    // ---- 3. an MLP: 1 -> 16 -> 16 -> 1, tanh ------------------------------------
    Param W1(glorot_uniform(1, 16, rng));
    Param b1(Vector<float>(Shape(16)));
    Param W2(glorot_uniform(16, 16, rng));
    Param b2(Vector<float>(Shape(16)));
    Param W3(glorot_uniform(16, 1, rng));
    Param b3(Vector<float>(Shape(1)));
    auto mlp = [&](const auto& x) {   // x: (batch, 1)
        return matmul(tanh(matmul(tanh(matmul(x, W1) + b1), W2) + b2), W3) + b3;
    };
    Matrix<float> K(Shape{n, 1}), V(Shape{n, 1});   // the data as (n, 1) columns
    std::ranges::copy(k.flat(), K.mut_flat().begin());
    std::ranges::copy(v.flat(), V.mut_flat().begin());
    const float nn_loss = fit([&] { return mse(mlp(K), V); }, 0.2f, 8000, W1, b1, W2, b2, W3, b3);

    // ---- report -------------------------------------------------------------------
    auto at = [&](auto&& model, float kk) {   // speed in km/h at density kk veh/km
        return Scalar<float>(sum(model(Vector<float>(Shape(1), {kk / k_scale}))))() * v_scale;
    };
    auto mlp_at = [&](float kk) {
        return Scalar<float>(sum(mlp(Matrix<float>(Shape(1, 1), {kk / k_scale}))))() * v_scale;
    };

    std::println("Fitted fundamental diagrams (truth: Drake, vf = {} km/h, kc = {} veh/km)\n", true_vf, true_kc);
    std::println("  model         RMSE     parameters");
    std::println("  Greenshields  {:4.1f} km/h  vf = {:5.1f} km/h, kj = {:5.1f} veh/km", rmse_kmh(gs_loss), gvf, gkj);
    std::println("  Drake         {:4.1f} km/h  vf = {:5.1f} km/h, kc = {:5.1f} veh/km", rmse_kmh(dk_loss), dvf, dkc);
    std::println("  MLP           {:4.1f} km/h  (313 weights, no physical meaning)\n", rmse_kmh(nn_loss));

    std::println("  speed (km/h) at density   truth  Greenshields  Drake    MLP");
    for (float kk : {10.0f, 35.0f, 70.0f, 120.0f})
        std::println("    k = {:5.1f} veh/km      {:6.1f}  {:10.1f}  {:7.1f}  {:6.1f}", kk, drake(kk),
                     at(greenshields, kk), at(drake_model, kk), mlp_at(kk));

    auto [tk, tq] = capacity(drake);
    auto [gk, gq] = capacity([&](float kk) { return at(greenshields, kk); });
    auto [dk, dq] = capacity([&](float kk) { return at(drake_model, kk); });
    auto [mk, mq] = capacity(mlp_at);
    std::println("\n  capacity (veh/h) at critical density (veh/km)");
    std::println("    truth         qmax = {:5.0f} at kc = {:5.1f}", tq, tk);
    std::println("    Greenshields  qmax = {:5.0f} at kc = {:5.1f}", gq, gk);
    std::println("    Drake         qmax = {:5.0f} at kc = {:5.1f}", dq, dk);
    std::println("    MLP           qmax = {:5.0f} at kc = {:5.1f}", mq, mk);
}
