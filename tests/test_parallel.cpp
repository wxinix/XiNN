// SPDX-License-Identifier: BSD-3-Clause
// SIMD and parallel paths must give the same results as the plain ones.
#include "check.hpp"

#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{8};

// The textbook triple loop, as a reference.
template <class T>
Matrix<T> reference_matmul(const Matrix<T>& a, const Matrix<T>& b) {
    const std::size_t m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    Matrix<T> c(Shape{m, n});
    auto C = c.mut();
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t p = 0; p < k; ++p) s += double(a(i, p)) * double(b(p, j));
            C[i, j] = T(s);
        }
    return c;
}

template <class T>
double max_abs_diff(const Matrix<T>& x, const Matrix<T>& y) {
    double d = 0;
    for (std::size_t i = 0; i < x.size(); ++i) d = std::max(d, std::abs(double(x.flat()[i]) - double(y.flat()[i])));
    return d;
}

}  // namespace

namespace tests {

void matmul_matches_the_reference_for_awkward_shapes() {
    // Sizes around the tile (4 x 16 floats), panel and block edges, and k past one 256-term slice.
    for (auto [m, k, n] : {std::array{1uz, 1uz, 1uz}, {3, 5, 7}, {4, 16, 16}, {17, 33, 15}, {97, 300, 50},
                          {200, 257, 130}, {1, 700, 3}, {130, 2, 200}}) {
        auto a = randn<float>(Shape{m, k}, rng), b = randn<float>(Shape{k, n}, rng);
        const Matrix<float> c = matmul(a, b);
        check::near(max_abs_diff(c, reference_matmul(a, b)), 0, 1e-4 * double(k));
    }
}

void transposed_operands_match() {
    auto a = randn<double>(Shape{37, 70}, rng), b = randn<double>(Shape{70, 45}, rng);
    const Matrix<double> at = transpose(a).eval(), bt = transpose(b).eval();   // stored transposed
    const auto want = reference_matmul(a, b);
    check::near(max_abs_diff(Matrix<double>(matmul(transpose(at), b)), want), 0, 1e-10);
    check::near(max_abs_diff(Matrix<double>(matmul(a, transpose(bt))), want), 0, 1e-10);
    check::near(max_abs_diff(Matrix<double>(matmul(transpose(at), transpose(bt))), want), 0, 1e-10);
}

void threads_do_not_change_matmul() {
    auto a = randn<float>(Shape{300, 400}, rng), b = randn<float>(Shape{400, 350}, rng);
    config.threads = false;
    const Matrix<float> one = matmul(a, b);
    config.threads = true;
    const Matrix<float> many = matmul(a, b);
    check::equal(max_abs_diff(one, many), 0.0);   // same arithmetic, same order within each element
}

void simd_and_threads_do_not_change_fused_results() {
    auto x = randn<float>(Shape{513, 257}, rng), w = randn<float>(Shape{513, 257}, rng);
    auto b = randn<float>(Shape{257}, rng);
    auto expr = [&] { return tanh(x * w + b) * exp(-square(w)) / 2 + relu(x - b); };
    static_assert(decltype(expr())::vectorizable);

    config = {.simd = false, .threads = false};
    const Matrix<float> plain = expr();
    for (auto [simd, threads] : {std::pair{true, false}, {false, true}, {true, true}}) {
        config = {.simd = simd, .threads = threads};
        const Matrix<float> fast = expr();
        check::near(max_abs_diff(plain, fast), 0, 1e-6);
    }
    config = {};
}

void broadcasting_that_wraps_mid_vector() {
    // A row of 13 values broadcast over rows: SIMD loads cross the row boundary.
    auto x = randn<float>(Shape{50, 13}, rng);
    auto r = randn<float>(Shape{13}, rng);
    config = {.simd = false, .threads = false};
    const Matrix<float> plain = x * r + 1;
    config = {};
    const Matrix<float> fast = x * r + 1;
    check::near(max_abs_diff(plain, fast), 0, 1e-6);
}

void lambdas_fall_back_to_scalar() {
    Vector<float> v(Shape(20), 1.0f);
    auto m = map([](float a) { return a * 3; }, v);
    static_assert(!decltype(m)::vectorizable);   // takes a float, not a SIMD vector
    check::equal(m.eval()(19), 3.0f);
}

void parallel_for_covers_every_index_once() {
    std::vector<int> hits(100'000);
    parallel_for(hits.size(), 1000, [&](std::size_t b, std::size_t e) {
        for (std::size_t i = b; i < e; ++i) ++hits[i];
    });
    check::that(std::ranges::all_of(hits, [](int h) { return h == 1; }));
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
