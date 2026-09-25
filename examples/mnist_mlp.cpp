// SPDX-License-Identifier: BSD-3-Clause
// Handwritten digits: a 784 -> 256 -> 128 -> 10 network on MNIST.
//
//   python tools/prepare_data.py      # once
//   build/release/examples/mnist_mlp
//
// Mini-batches of 128, Adam, a cosine learning-rate schedule, 5 epochs.
#include <chrono>
#include <print>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

struct DigitNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;
    explicit DigitNet(Rng& rng) : fc1(784, 256, rng), fc2(256, 128, rng), out(128, 10, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

// Predicted digits for x, evaluated in chunks to bound memory.
std::vector<std::size_t> predict(const DigitNet& net, const Matrix<float>& x) {
    std::vector<std::size_t> out;
    std::vector<std::size_t> idx;
    for (std::size_t start = 0; start < x.shape()[0]; start += 1000) {
        idx.clear();
        for (std::size_t i = start; i < std::min(start + 1000, x.shape()[0]); ++i) idx.push_back(i);
        auto p = argmax_rows(net(gather_rows(x, idx)));
        out.insert(out.end(), p.begin(), p.end());
    }
    return out;
}

int main() {
    auto train = datasets::load_mnist(XINN_DATA_DIR "/mnist", true);
    auto test = datasets::load_mnist(XINN_DATA_DIR "/mnist", false);
    if (!train || !test) return std::println("{}", train ? test.error() : train.error()), 1;
    const Matrix<float> Y = one_hot(train->labels, 10);

    Rng rng{2026};
    DigitNet net(rng);
    std::print("{}", net.summary());

    const std::size_t epochs = 5, batch = 128;
    const std::size_t steps_per_epoch = (train->images.shape()[0] + batch - 1) / batch;
    Adam opt(net, {.lr = 1e-3});

    std::size_t step = 0;
    for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();
        double loss_sum = 0;
        for (auto [xb, yb] : batches(train->images, Y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, epochs * steps_per_epoch, 1e-3, 1e-5));
            opt.zero_grad();
            loss_sum += backward(softmax_cross_entropy(net(xb), yb));
            opt.step();
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::println("epoch {}  loss {:.4f}  test accuracy {:.2f}%  ({:.1f} s)", epoch, loss_sum / steps_per_epoch,
                     100 * accuracy(predict(net, test->images), test->labels), secs);
    }

    ConfusionMatrix cm(predict(net, test->images), test->labels, 10);
    std::vector<std::string> names;
    for (int d = 0; d < 10; ++d) names.push_back(std::to_string(d));
    std::print("\n{}", cm.table(names));

    if (auto r = save(net, "mnist_mlp.safetensors"); !r) std::println("save failed: {}", r.error());
}
