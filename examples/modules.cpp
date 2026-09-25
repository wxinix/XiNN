// SPDX-License-Identifier: BSD-3-Clause
// Modules: models as structs, parameters found by reflection, saved as
// safetensors.
//
// Two models:
//   SpiralNet   a classifier built from Dense layers
//   Drake       a traffic-flow model with two physical parameters, vf and kc
// Each is trained, summarized, saved, loaded into a fresh instance, and
// checked to give identical results.
#include <cmath>
#include <numbers>
#include <print>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

struct SpiralNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;

    explicit SpiralNet(Rng& rng) : fc1(2, 32, rng), fc2(32, 32, rng), out(32, 2, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

// Drake's speed-density relation, v = vf * exp(-(k/kc)^2 / 2), in units of
// 100 km/h and 100 veh/km.
struct Drake : Module {
    Param<float, 0> vf{Scalar<float>(Shape<0>{}, 1.0f)};
    Param<float, 0> kc{Scalar<float>(Shape<0>{}, 0.5f)};
    auto forward(const auto& k) const { return vf * exp(square(k / kc) * -0.5f); }
};

// One step of gradient descent on every parameter of any module.
void sgd_step(ModuleType auto& model, float lr) {
    model.for_each_param([&](const std::string&, auto& p) { p.assign(p - lr * p.grad()); });
}

template <class Loss>
float train(ModuleType auto& model, Loss&& loss, float lr, int steps) {
    float last = 0;
    for (int s = 0; s < steps; ++s) {
        model.zero_grad();
        last = backward(loss());
        sgd_step(model, lr);
    }
    return last;
}

int main() {
    Rng rng{7};

    // ---- a classifier ---------------------------------------------------------
    const std::size_t per_class = 200, n = 2 * per_class;
    Matrix<float> X(Shape{n, 2});
    std::vector<std::size_t> labels(n);
    {
        std::normal_distribution<float> noise(0.0f, 0.15f);
        auto x = X.mut();
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < per_class; ++i) {
                const std::size_t row = c * per_class + i;
                const float r = 0.2f + 2.8f * i / per_class, a = 1.75f * r + c * std::numbers::pi_v<float>;
                x[row, 0] = r * std::cos(a) + noise(rng);
                x[row, 1] = r * std::sin(a) + noise(rng);
                labels[row] = c;
            }
    }
    const Matrix<float> Y = one_hot(labels, 2);

    SpiralNet net(rng);
    std::println("SpiralNet ({} parameter tensors, known at compile time)\n{}", param_tensor_count<SpiralNet>,
                 net.summary());
    const float loss = train(net, [&] { return softmax_cross_entropy(net(X), Y); }, 0.3f, 1500);

    auto accuracy = [&](const SpiralNet& m) {
        const auto pred = argmax_rows(m(X));
        std::size_t ok = 0;
        for (std::size_t i = 0; i < n; ++i) ok += pred[i] == labels[i];
        return 100.0 * ok / n;
    };
    std::println("trained: loss {:.4f}, accuracy {:.1f}%", loss, accuracy(net));

    if (auto r = save(net, "spiral.safetensors"); !r) return std::println("save failed: {}", r.error()), 1;
    Rng other{99};
    SpiralNet copy(other);   // different random weights ...
    std::println("fresh copy: accuracy {:.1f}%", accuracy(copy));
    if (auto r = load(copy, "spiral.safetensors"); !r) return std::println("load failed: {}", r.error()), 1;
    std::println("loaded:     accuracy {:.1f}%   (from spiral.safetensors)\n", accuracy(copy));

    // ---- a traffic-flow model -------------------------------------------------------
    const std::size_t m = 600;
    Vector<float> k(Shape{m}), v(Shape{m});
    {
        std::uniform_real_distribution<float> density(0.02f, 1.4f);   // x 100 veh/km
        std::normal_distribution<float> noise(0.0f, 0.04f);            // x 100 km/h
        auto ks = k.mut_flat();
        auto vs = v.mut_flat();
        for (std::size_t i = 0; i < m; ++i) {
            ks[i] = density(rng);
            vs[i] = 1.10f * std::exp(-0.5f * (ks[i] / 0.35f) * (ks[i] / 0.35f)) + noise(rng);
        }
    }
    Drake drake;
    std::println("Drake\n{}", drake.summary());
    train(drake, [&] { return mse(drake(k), v); }, 0.5f, 4000);
    std::println("fitted: vf = {:.1f} km/h, kc = {:.1f} veh/km   (truth: 110.0, 35.0)", drake.vf.tensor()() * 100,
                 drake.kc.tensor()() * 100);

    if (auto r = save(drake, "drake.safetensors"); !r) return std::println("save failed: {}", r.error()), 1;
    Drake restored;
    if (auto r = load(restored, "drake.safetensors"); !r) return std::println("load failed: {}", r.error()), 1;
    std::println("loaded:  vf = {:.1f} km/h, kc = {:.1f} veh/km   (from drake.safetensors)",
                 restored.vf.tensor()() * 100, restored.kc.tensor()() * 100);
}
