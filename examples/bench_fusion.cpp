// SPDX-License-Identifier: BSD-3-Clause
// Fused vs. step-by-step evaluation of  y = sigmoid(x * w + b) * 0.5
//
// Step-by-step evaluates each operation into its own tensor, the way an eager
// framework does. Fused evaluates the whole expression in one loop.
#include <chrono>
#include <print>
#include <xinn/xinn.hpp>

using namespace xinn;

template <class F>
double best_ms(F&& f, int reps = 20) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

int main() {
    const std::size_t n = 1024;
    Matrix<float> x(Shape{n, n}, 0.5f);
    Matrix<float> w(Shape{n, n}, 2.0f);
    Vector<float> b(Shape{n}, -1.0f);

    float sink = 0;

    const double eager = best_ms([&] {
        Matrix<float> t1 = x * w;           // each line allocates and walks
        Matrix<float> t2 = t1 + b;          // the whole n*n array again
        Matrix<float> t3 = sigmoid(t2);
        Matrix<float> y  = t3 * 0.5f;
        sink += y(0, 0);
    });

    const double fused = best_ms([&] {
        Matrix<float> y = sigmoid(x * w + b) * 0.5f;   // one loop, one allocation
        sink += y(0, 0);
    });

    std::println("{}x{} elements", n, n);
    std::println("  step by step : {:7.2f} ms", eager);
    std::println("  fused        : {:7.2f} ms   ({:.1f}x faster)", fused, eager / fused);
    std::println("  (checksum {})", sink);
}
