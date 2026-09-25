// SPDX-License-Identifier: BSD-3-Clause
// The same expressions on the CPU and on the GPU.
//
//   bench_gpu            time a fused element-wise expression and matmul
//   bench_gpu --source   also print the generated kernel
//
// GPU times come in two kinds:
//   total    host tensors in, host tensor out: upload + kernel + download
//   kernel   operands already on the device; the result stays there
// The difference is the cost of moving data over PCIe. Every time is the
// best of several runs, after one warm-up run that also builds the kernel.
//
// Without an OpenCL device it says so and exits with 0.
#include <array>
#include <chrono>
#include <print>
#include <string_view>

#include <xinn/xinn.hpp>
#include <xinn/gpu.hpp>

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

// Stop at the first GPU error: print it and exit.
template <class T>
T must(std::expected<T, gpu::Error> r) {
    if (!r) {
        std::println(stderr, "GPU error: {}", r.error().message);
        std::exit(1);
    }
    return *std::move(r);
}

void must(std::expected<void, gpu::Error> r) {
    if (!r) {
        std::println(stderr, "GPU error: {}", r.error().message);
        std::exit(1);
    }
}

double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d = std::max(d, double(std::abs(a[i] - b[i])));
    return d;
}

}  // namespace

int main(int argc, char** argv) {
    const bool show_source = argc > 1 && std::string_view(argv[1]) == "--source";
    auto opened = gpu::Device::open();
    if (!opened) {
        std::println("No OpenCL device: {}", opened.error().message);
        std::println("Nothing to compare. On a machine with a GPU driver, run this again.");
        return 0;
    }
    const gpu::Device& dev = *opened;
    std::println("GPU: {} ({}; {})", dev.name(), dev.platform(), dev.version());
    std::println("CPU: {} hardware threads, {} floats per SIMD vector\n", hardware_threads(), simd<float>::size());
    Rng rng{1};

    // ---- element-wise ------------------------------------------------------------
    auto expr = [](const auto& x, const auto& w, const auto& b) { return sigmoid(x * w + b) * 0.5f; };
    if (show_source) {
        using E = decltype(expr(Matrix<float>(), Matrix<float>(), Vector<float>()));
        std::println("generated kernel:\n{}", gpu::kernel_source<E>);
    }
    std::println("sigmoid(x * w + b) * 0.5     CPU            GPU total      GPU kernel     max |CPU - GPU|");
    for (std::size_t n : {512uz, 1024uz, 2048uz, 4096uz}) {
        const auto x = randn<float>(Shape{n, n}, rng), w = randn<float>(Shape{n, n}, rng);
        const auto b = randn<float>(Shape{n}, rng);
        const auto dx = must(gpu::upload(dev, x)), dw = must(gpu::upload(dev, w));
        const auto db = must(gpu::upload(dev, b));

        const Matrix<float> cpu = expr(x, w, b);
        const Matrix<float> gpu_result = must(gpu::eval(dev, expr(x, w, b)));   // warm-up, and the check
        const double cpu_ms = best_ms([&] { sink += expr(x, w, b).eval().flat()[0]; }, 5);
        const double total_ms = best_ms([&] { sink += must(gpu::eval(dev, expr(x, w, b))).flat()[0]; }, 5);
        const double kernel_ms = best_ms([&] {
            auto y = must(gpu::run(dev, expr(dx, dw, db)));
            must(gpu::finish(dev));
        }, 5);
        std::println("{:>5} x {:<5}           {:>8.2f} ms    {:>8.2f} ms    {:>8.2f} ms    {:.1e}", n, n, cpu_ms,
                     total_ms, kernel_ms, max_abs_diff(cpu.flat(), gpu_result.flat()));
    }

    // ---- matmul ------------------------------------------------------------------
    // The kernels of codegen.hpp: "tiled" computes one element of C per work
    // item; the blocked ones a 4 x 4 or 8 x 8 block in registers, loading
    // tiles one float or four ("f4") at a time. "GPU total" uses the default.
    using enum gpu::MatmulKernel;
    constexpr std::array kernels{tiled, blocked_4x4, blocked_4x4_float4, blocked_8x8, blocked_8x8_float4};
    std::println("\nmatmul (float), GFLOP/s   CPU   GPU total |  kernels: tiled    4x4  4x4 f4    8x8  8x8 f4 | max |CPU - GPU|");
    for (std::size_t m : {256uz, 512uz, 1024uz, 2048uz, 4096uz}) {
        const auto a = randn<float>(Shape{m, m}, rng), c = randn<float>(Shape{m, m}, rng);
        const auto da = must(gpu::upload(dev, a)), dc = must(gpu::upload(dev, c));
        const double flops = 2.0 * double(m) * double(m) * double(m);
        const int reps = m >= 2048 ? 3 : 5;

        const Matrix<float> cpu = matmul(a, c);
        double diff = 0;   // the largest difference over all kernels; the runs also warm them up
        for (auto kind : kernels) diff = std::max(diff, max_abs_diff(cpu.flat(), must(gpu::matmul(dev, a, c, kind)).flat()));
        const double cpu_ms = best_ms([&] { sink += matmul(a, c).eval().flat()[0]; }, reps);
        const double total_ms = best_ms([&] { sink += must(gpu::matmul(dev, a, c)).flat()[0]; }, reps);
        std::print("{:>5}^3                 {:>5.0f}   {:>7.0f}   |         ", m, flops / cpu_ms / 1e6, flops / total_ms / 1e6);
        for (auto kind : kernels) {
            const double ms = best_ms([&] {
                auto y = must(gpu::run_matmul(dev, da, dc, kind));
                must(gpu::finish(dev));
            }, reps);
            std::print(" {:>6.0f}", flops / ms / 1e6);
        }
        std::println(" | {:.1e}", diff);
    }
    std::println("\n(checksum {})", sink);
}
