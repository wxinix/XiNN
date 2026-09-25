// SPDX-License-Identifier: BSD-3-Clause
// Train a small network to separate two interleaved spirals.
//
// The classes are not linearly separable: no straight line splits them, so
// hidden layers are needed. The two-hidden-layer MLP below reaches over 99%
// accuracy. Everything is plain XiNN: Params, expressions, backward(), and a
// hand-written gradient-descent step.
#include <cmath>
#include <numbers>
#include <print>
#include <tuple>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

int main() {
    Rng rng{7};

    // ---- data: 2 classes x 200 points --------------------------------------
    const std::size_t per_class = 200, n = 2 * per_class;
    Matrix<float> X(Shape{n, 2});
    std::vector<std::size_t> labels(n);
    {
        std::normal_distribution<float> noise(0.0f, 0.15f);
        auto x = X.mut();
        for (std::size_t c = 0; c < 2; ++c)
            for (std::size_t i = 0; i < per_class; ++i) {
                const std::size_t row = c * per_class + i;
                const float r = 0.2f + 2.8f * i / per_class;                       // radius
                const float a = 1.75f * r + c * std::numbers::pi_v<float>;          // angle
                x[row, 0] = r * std::cos(a) + noise(rng);
                x[row, 1] = r * std::sin(a) + noise(rng);
                labels[row] = c;
            }
    }
    const Matrix<float> Y = one_hot(labels, 2);

    // ---- model: 2 -> 32 -> 32 -> 2 ------------------------------------------
    Param W1(he_normal(2, 32, rng));
    Param b1(Vector<float>(Shape(32)));
    Param W2(he_normal(32, 32, rng));
    Param b2(Vector<float>(Shape(32)));
    Param W3(glorot_uniform(32, 2, rng));
    Param b3(Vector<float>(Shape(2)));
    auto params = std::tie(W1, b1, W2, b2, W3, b3);

    auto model = [&](const auto& x) {
        auto h1 = relu(matmul(x, W1) + b1);
        auto h2 = relu(matmul(h1, W2) + b2);
        return matmul(h2, W3) + b3;   // logits
    };

    // ---- training: full-batch gradient descent ------------------------------
    const float lr = 0.3f;
    for (int step = 0; step <= 2000; ++step) {
        template for (auto& p : params) p.zero_grad();

        const float loss = backward(softmax_cross_entropy(model(X), Y));

        template for (auto& p : params) p.assign(p - lr * p.grad());

        if (step % 250 == 0) {
            const auto pred = argmax_rows(model(X));
            std::size_t correct = 0;
            for (std::size_t i = 0; i < n; ++i) correct += pred[i] == labels[i];
            std::println("step {:4}  loss {:.4f}  accuracy {:5.1f}%", step, loss, 100.0 * correct / n);
        }
    }

    std::println("\nmodel: {}", describe(model(X)));
}
