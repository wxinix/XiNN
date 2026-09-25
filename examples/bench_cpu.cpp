// SPDX-License-Identifier: BSD-3-Clause
// What each CPU technique buys: scalar vs SIMD, one thread vs all.
//
//   bench_cpu
//
// Element-wise work is timed with xinn::config.simd and config.threads
// switched on and off; matmul with threads on and off.
#include <chrono>
#include <print>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

template <class F>
double best_ms(F&& f, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

float sink = 0;

template <class F>
void elementwise_row(std::string_view name, std::size_t n, F&& expr) {
    std::print("{:<28}", name);
    double base = 0;
    for (auto [simd, threads] : {std::pair{false, false}, {true, false}, {false, true}, {true, true}}) {
        config.simd = simd;
        config.threads = threads;
        const double ms = best_ms([&] { sink += expr().eval().flat()[0]; }, 7);
        if (base == 0) base = ms;
        std::print("{:>9.2f} ms {:>5.1f}x", ms, base / ms);
    }
    std::println("   ({:.2f} Gelem/s best)", double(n) / 1e6 / best_ms([&] { sink += expr().eval().flat()[0]; }, 3));
    config = {};
}

}  // namespace

int main() {
    Rng rng{1};
    std::println("{} hardware threads; {} floats per SIMD vector; threads {}\n", hardware_threads(), simd<float>::size(),
                 has_threads ? "available" : "not built in");

    const std::size_t n = 2048;
    const auto x = randn<float>(Shape{n, n}, rng), w = randn<float>(Shape{n, n}, rng);
    const auto b = randn<float>(Shape{n}, rng);
    std::println("element-wise on {}x{}          scalar,1 thread     SIMD,1 thread     scalar,threads     SIMD,threads",
                 n, n);
    elementwise_row("x + w", n * n, [&] { return x + w; });
    elementwise_row("x * w + b", n * n, [&] { return x * w + b; });
    elementwise_row("sigmoid(x * w + b) * 0.5", n * n, [&] { return sigmoid(x * w + b) * 0.5f; });
    elementwise_row("tanh(x) * exp(-square(w))", n * n, [&] { return tanh(x) * exp(-square(w)); });

    std::println("\nmatmul (float)        1 thread                 threads");
    for (std::size_t m : {256uz, 512uz, 1024uz, 2048uz}) {
        const auto a = randn<float>(Shape{m, m}, rng), c = randn<float>(Shape{m, m}, rng);
        const double flops = 2.0 * double(m) * m * m;
        std::print("{:>5}^3", m);
        for (bool threads : {false, true}) {
            config.threads = threads;
            const double ms = best_ms([&] { sink += matmul(a, c).eval().flat()[0]; }, m >= 2048 ? 2 : 5);
            std::print("{:>11.2f} ms {:>7.1f} GFLOP/s", ms, flops / ms / 1e6);
        }
        std::println("");
        config = {};
    }
    std::println("\n(checksum {})", sink);
}
