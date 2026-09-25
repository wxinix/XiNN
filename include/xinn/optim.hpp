// SPDX-License-Identifier: BSD-3-Clause
// Optimizers: how parameters move, given their gradients.
//
//   Adam opt(net, {.lr = 1e-3f});
//   for (...) { opt.zero_grad(); backward(loss); opt.step(); }
//
// An optimizer is built from a model. Its constructor collects the model's
// parameters as a typed tuple (param_tuple, found by reflection), and lays
// out its own state to match: one velocity tensor per parameter for SGD with
// momentum, two moment tensors for Adam, each with that parameter's exact
// type and shape. No maps, no type erasure; step() is a `template for` over
// the tuple.
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <tuple>
#include <utility>

#include "xinn/autograd.hpp"
#include "xinn/module.hpp"
#include "xinn/ops.hpp"
#include "xinn/param.hpp"

namespace xinn {

// The strategy interface every optimizer satisfies.
template <class O>
concept Optimizer = requires(O& o) {
    o.step();
    o.zero_grad();
    { o.learning_rate() } -> std::convertible_to<double>;
    o.set_learning_rate(0.1);
};

namespace detail {

// One zero tensor per parameter, shaped like it.
template <class... Ps>
auto zeros_like_each(const std::tuple<Ps...>& params) {
    return std::apply([](const auto&... p) {
        return std::tuple{Tensor<typename Ps::value_type, Ps::rank>(p.shape())...};
    }, params);
}

template <class Params>
using state_t = decltype(zeros_like_each(std::declval<const Params&>()));

}  // namespace detail

template <ModuleType M>
using params_of_t = decltype(detail::param_tuple(std::declval<M&>()));

// ---- SGD --------------------------------------------------------------------------
//
//   v <- momentum * v + g            (g includes weight_decay * p)
//   p <- p - lr * v
//
// Momentum averages recent gradients, so steps keep their direction through
// small bumps and speed up along long, shallow slopes.

struct SGDOptions {
    double lr = 0.01;
    double momentum = 0.0;
    double weight_decay = 0.0;
};

template <class Params>
class SGD {
public:
    SGDOptions options;

    template <ModuleType M>
    explicit SGD(M& model, SGDOptions opt = {})
        : options(opt), params_(detail::param_tuple(model)), velocity_(detail::zeros_like_each(params_)) {}

    void zero_grad() {
        template for (auto& p : params_) p.zero_grad();
    }

    void step() {
        template for (constexpr std::size_t I : detail::indices<std::tuple_size_v<Params>>) {
            auto& p = std::get<I>(params_);
            auto& v = std::get<I>(velocity_);
            using T = typename std::remove_cvref_t<decltype(p)>::value_type;
            const T lr = T(options.lr), mu = T(options.momentum), wd = T(options.weight_decay);
            if (options.momentum == 0 && options.weight_decay == 0) {
                p.assign(p - p.grad() * lr);
            } else {
                v = v * mu + p.grad() + p * wd;
                p.assign(p - v * lr);
            }
        }
    }

    double learning_rate() const { return options.lr; }
    void set_learning_rate(double lr) { options.lr = lr; }

private:
    Params params_;
    detail::state_t<Params> velocity_;
};

template <ModuleType M>
SGD(M&, SGDOptions = {}) -> SGD<params_of_t<M>>;

// ---- Adam (with decoupled weight decay, i.e. AdamW) --------------------------------------
//
//   m <- b1 m + (1 - b1) g            running mean of gradients
//   v <- b2 v + (1 - b2) g^2          running mean of squared gradients
//   p <- p - lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)  -  lr * wd * p
//
// Dividing by sqrt(v) gives every parameter its own step size: large where
// gradients are consistently small, small where they are large or noisy.
// The (1 - b^t) factors undo the bias of m and v towards their zero start.

struct AdamOptions {
    double lr = 1e-3;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.0;
};

template <class Params>
class Adam {
public:
    AdamOptions options;

    template <ModuleType M>
    explicit Adam(M& model, AdamOptions opt = {})
        : options(opt),
          params_(detail::param_tuple(model)),
          m_(detail::zeros_like_each(params_)),
          v_(detail::zeros_like_each(params_)) {}

    void zero_grad() {
        template for (auto& p : params_) p.zero_grad();
    }

    void step() {
        ++t_;
        const double c1 = 1 - std::pow(options.beta1, t_);
        const double c2 = 1 - std::pow(options.beta2, t_);
        template for (constexpr std::size_t I : detail::indices<std::tuple_size_v<Params>>) {
            auto& p = std::get<I>(params_);
            auto& m = std::get<I>(m_);
            auto& v = std::get<I>(v_);
            using T = typename std::remove_cvref_t<decltype(p)>::value_type;
            const auto& g = p.grad();
            const T b1 = T(options.beta1), b2 = T(options.beta2);
            m = m * b1 + g * (1 - b1);
            v = v * b2 + square(g) * (1 - b2);
            auto update = (m / T(c1)) / (sqrt(v / T(c2)) + T(options.eps));
            if (options.weight_decay == 0) p.assign(p - update * T(options.lr));
            else p.assign(p * T(1 - options.lr * options.weight_decay) - update * T(options.lr));
        }
    }

    double learning_rate() const { return options.lr; }
    void set_learning_rate(double lr) { options.lr = lr; }
    long steps() const { return t_; }

private:
    Params params_;
    detail::state_t<Params> m_, v_;
    long t_ = 0;
};

template <ModuleType M>
Adam(M&, AdamOptions = {}) -> Adam<params_of_t<M>>;

// ---- learning-rate schedules -------------------------------------------------------------

// Cosine decay from lr0 to lr_min over `total` steps: fast at first, gentle at the end.
inline double cosine_lr(std::size_t step, std::size_t total, double lr0, double lr_min = 0) {
    if (step >= total) return lr_min;
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + std::cos(std::numbers::pi * double(step) / double(total)));
}

}  // namespace xinn
