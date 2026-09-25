// SPDX-License-Identifier: BSD-3-Clause
// Random tensors for initializing parameters.
//
// Why the scale matters: with weights of standard deviation s, each layer
// multiplies the spread of its signal by about s * sqrt(fan_in). Too large
// and activations explode; too small and they vanish. He initialization,
// s = sqrt(2 / fan_in), keeps the spread steady through ReLU layers.
#pragma once

#include <cmath>
#include <cstddef>
#include <random>

#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

using Rng = std::mt19937;

template <class T = float, std::size_t R>
Tensor<T, R> randn(const Shape<R>& shape, Rng& rng, T mean = 0, T stddev = 1) {
    std::normal_distribution<T> dist(mean, stddev);
    Tensor<T, R> t(shape);
    for (T& v : t.mut_flat()) v = dist(rng);
    return t;
}

template <class T = float, std::size_t R>
Tensor<T, R> uniform(const Shape<R>& shape, Rng& rng, T lo = 0, T hi = 1) {
    std::uniform_real_distribution<T> dist(lo, hi);
    Tensor<T, R> t(shape);
    for (T& v : t.mut_flat()) v = dist(rng);
    return t;
}

// A (fan_in, fan_out) weight matrix for a layer followed by ReLU.
template <class T = float>
Matrix<T> he_normal(std::size_t fan_in, std::size_t fan_out, Rng& rng) {
    return randn<T>(Shape{fan_in, fan_out}, rng, T{0}, static_cast<T>(std::sqrt(2.0 / fan_in)));
}

// A (fan_in, fan_out) weight matrix for tanh, sigmoid, or no activation.
template <class T = float>
Matrix<T> glorot_uniform(std::size_t fan_in, std::size_t fan_out, Rng& rng) {
    const T limit = static_cast<T>(std::sqrt(6.0 / (fan_in + fan_out)));
    return uniform<T>(Shape{fan_in, fan_out}, rng, -limit, limit);
}

}  // namespace xinn
