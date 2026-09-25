// SPDX-License-Identifier: BSD-3-Clause
// Handwritten digits with a convolutional network.
//
//   conv 3x3, 16 channels, ReLU -> max-pool 2 -> conv 3x3, 32 channels, ReLU
//   -> max-pool 2 -> flatten -> dense 128, ReLU -> dense 10
//
// Compare with mnist_mlp: the MLP sees 784 unrelated numbers; the CNN sees an
// image, and learns small filters that are applied at every position.
#include <chrono>
#include <print>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

struct DigitCnn : Module {
    Conv2D<{.kernel = 3, .padding = 1, .activation = Activation::relu}> conv1, conv2;
    MaxPool2D<2> pool;
    Flatten flatten;
    Dense<{.activation = Activation::relu}> fc;
    Dense<> out;

    explicit DigitCnn(Rng& rng) : conv1(1, 16, rng), conv2(16, 32, rng), fc(32 * 7 * 7, 128, rng), out(128, 10, rng) {}

    // x: (batch, 784) rows of pixels, viewed as (batch, 1, 28, 28) images.
    auto forward(const auto& x) const {
        auto img = reshape(x, Shape{x.shape()[0], 1uz, 28uz, 28uz});
        return out(fc(flatten(pool(conv2(pool(conv1(img)))))));
    }
};

std::vector<std::size_t> predict(const DigitCnn& net, const Matrix<float>& x) {
    std::vector<std::size_t> out, idx;
    for (std::size_t start = 0; start < x.shape()[0]; start += 500) {
        idx.clear();
        for (std::size_t i = start; i < std::min(start + 500, x.shape()[0]); ++i) idx.push_back(i);
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
    DigitCnn net(rng);
    std::print("{}", net.summary());

    const std::size_t epochs = 3, batch = 64, steps = epochs * ((train->images.shape()[0] + batch - 1) / batch);
    Adam opt(net, {.lr = 1e-3});
    std::size_t step = 0;
    for (std::size_t epoch = 1; epoch <= epochs; ++epoch) {
        const auto t0 = std::chrono::steady_clock::now();
        double loss = 0;
        std::size_t n = 0;
        for (auto [xb, yb] : batches(train->images, Y, batch, rng)) {
            opt.set_learning_rate(cosine_lr(step++, steps, 1e-3, 1e-5));
            opt.zero_grad();
            loss += backward(softmax_cross_entropy(net(xb), yb));
            opt.step();
            ++n;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::println("epoch {}  loss {:.4f}  test accuracy {:.2f}%  ({:.0f} s)", epoch, loss / n,
                     100 * accuracy(predict(net, test->images), test->labels), secs);
    }
    if (auto r = save(net, "mnist_cnn.safetensors"); !r) std::println("save failed: {}", r.error());
}
