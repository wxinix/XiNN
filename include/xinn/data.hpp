// SPDX-License-Identifier: BSD-3-Clause
// Feeding data to training.
//
//   for (auto [xb, yb] : batches(X, Y, 128, rng)) { ... }   // one shuffled epoch
//
// Training on small random batches instead of the whole data set gives many
// cheap, noisy gradient steps per pass. The noise even helps: it keeps the
// optimizer from settling into sharp, poorly generalizing minima.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <generator>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "xinn/init.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

// Rows `idx` of x (along the first dimension), copied into a new tensor.
template <class T, std::size_t R>
    requires(R >= 1)
Tensor<T, R> gather_rows(const Tensor<T, R>& x, std::span<const std::size_t> idx) {
    Shape<R> s = x.shape();
    const std::size_t row = x.size() / s[0];
    s.dims[0] = idx.size();
    Tensor<T, R> out(s);
    auto src = x.flat();
    auto dst = out.mut_flat();
    for (std::size_t i = 0; i < idx.size(); ++i) {
        contract_assert(idx[i] < x.shape()[0]);
        std::copy_n(src.begin() + idx[i] * row, row, dst.begin() + i * row);
    }
    return out;
}

// The indices 0..n-1 in a random order, in chunks of `batch`. The last
// chunk may be smaller.
inline std::generator<std::span<const std::size_t>> shuffled_batches(std::size_t n, std::size_t batch, Rng& rng)
    pre(batch > 0)
{
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::ranges::shuffle(order, rng);
    for (std::size_t start = 0; start < n; start += batch)
        co_yield std::span<const std::size_t>(order).subspan(start, std::min(batch, n - start));
}

// One epoch of shuffled (x, y) mini-batches. x and y have one sample per row.
template <class TX, std::size_t RX, class TY, std::size_t RY>
std::generator<std::pair<Tensor<TX, RX>, Tensor<TY, RY>>> batches(Tensor<TX, RX> x, Tensor<TY, RY> y,
                                                                  std::size_t batch, Rng& rng)
    pre(x.shape()[0] == y.shape()[0])
{
    for (auto idx : shuffled_batches(x.shape()[0], batch, rng)) co_yield {gather_rows(x, idx), gather_rows(y, idx)};
}

// Standardization: shift and scale each column (feature) to mean 0 and
// standard deviation 1. Fit on the training data only, then apply the same
// numbers to validation and test data -- anything else leaks information
// about the test set into training.
template <class T = float>
struct Standardizer {
    Vector<T> mean, scale;

    static Standardizer fit(const Matrix<T>& x) {
        const std::size_t n = x.shape()[0], d = x.shape()[1];
        Vector<T> mu(Shape{d}), sd(Shape{d});
        auto v = x.view();
        auto m = mu.mut_flat();
        auto s = sd.mut_flat();
        for (std::size_t j = 0; j < d; ++j) {
            double sum = 0, sq = 0;
            for (std::size_t i = 0; i < n; ++i) sum += v[i, j];
            const double mean_j = sum / double(n);
            for (std::size_t i = 0; i < n; ++i) sq += (v[i, j] - mean_j) * (v[i, j] - mean_j);
            m[j] = T(mean_j);
            s[j] = T(std::max(std::sqrt(sq / double(n)), 1e-8));   // constant column: leave it at 0
        }
        return {mu, sd};
    }

    // Lazy: (x - mean) / scale, broadcast over rows.
    auto operator()(const Matrix<T>& x) const { return (x - mean) / scale; }
};

}  // namespace xinn
