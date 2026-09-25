// SPDX-License-Identifier: BSD-3-Clause
// Losses and related helpers.
//
//   mse(pred, target)                     mean squared error, for regression
//   softmax(logits)                       probabilities along the last axis
//   softmax_cross_entropy(logits, onehot) the classification loss
//   one_hot(labels, classes [, class_weights]), balanced_class_weights(labels, classes)
//   argmax_rows(m)
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/ops.hpp"
#include "xinn/reduce.hpp"

namespace xinn {

namespace ops {

// dx = y * (g - sum_j g_j y_j), row by row: the Jacobian of softmax applied to g.
struct SoftmaxGrad {
    template <std::size_t... R> static constexpr std::size_t rank = std::max({R...});

    template <class G, class Y>
    Shape<Y::rank> shape(const G&, const Y& y) const { return y.shape(); }

    template <class T, std::size_t R>
    Tensor<T, R> eval(const Tensor<T, R>& g, const Tensor<T, R>& y) const {
        const std::size_t c = y.shape()[R - 1];
        Tensor<T, R> out(y.shape());
        auto gs = g.flat(), ys = y.flat();
        auto dst = out.mut_flat();
        for (std::size_t row = 0; row < ys.size(); row += c) {
            T dot{};
            for (std::size_t j = 0; j < c; ++j) dot += gs[row + j] * ys[row + j];
            for (std::size_t j = 0; j < c; ++j) dst[row + j] = ys[row + j] * (gs[row + j] - dot);
        }
        return out;
    }
};

// Softmax along the last axis: y_j = exp(x_j) / sum_k exp(x_k).
// Subtracting the row maximum first keeps exp() from overflowing.
struct Softmax {
    template <std::size_t... R> static constexpr std::size_t rank = (R + ...);

    template <class X>
    Shape<X::rank> shape(const X& x) const { return x.shape(); }

    template <class T, std::size_t R>
        requires(R >= 1)
    Tensor<T, R> eval(const Tensor<T, R>& x) const {
        const std::size_t c = x.shape()[R - 1];
        Tensor<T, R> out(x.shape());
        auto src = x.flat();
        auto dst = out.mut_flat();
        for (std::size_t row = 0; row < src.size(); row += c) {
            const T m = *std::max_element(src.begin() + row, src.begin() + row + c);
            T total{};
            for (std::size_t j = 0; j < c; ++j) total += dst[row + j] = std::exp(src[row + j] - m);
            for (std::size_t j = 0; j < c; ++j) dst[row + j] /= total;
        }
        return out;
    }

    template <std::size_t>
    auto grad(const auto& g, const auto& y, const auto&) const { return make_expr(SoftmaxGrad{}, g, y); }
};

// (softmax(x) * rowsum(t) - t) * g / n, row by row: the gradient of the loss below.
struct SoftmaxCrossEntropyGrad {
    template <std::size_t...> static constexpr std::size_t rank = 2;

    template <class G, class X, class Y>
    Shape<2> shape(const G&, const X& x, const Y&) const { return x.shape(); }

    template <class T>
    Tensor<T, 2> eval(const Tensor<T, 0>& g, const Tensor<T, 2>& x, const Tensor<T, 2>& t) const {
        const std::size_t n = x.shape()[0], c = x.shape()[1];
        const T scale = g() / static_cast<T>(n);
        Tensor<T, 2> out(x.shape());
        auto xs = x.flat(), ts = t.flat();
        auto dst = out.mut_flat();
        for (std::size_t row = 0; row < n * c; row += c) {
            const T m = *std::max_element(xs.begin() + row, xs.begin() + row + c);
            T total{}, weight{};
            for (std::size_t j = 0; j < c; ++j) {
                total += dst[row + j] = std::exp(xs[row + j] - m);
                weight += ts[row + j];
            }
            for (std::size_t j = 0; j < c; ++j) dst[row + j] = (dst[row + j] / total * weight - ts[row + j]) * scale;
        }
        return out;
    }
};

// Mean over rows of  -sum_j t_j log softmax(x)_j,  for logits x and targets t.
// Per row this is
//   (sum_j t_j) log(sum_k exp(x_k)) - sum_j t_j x_j,
// computed with the max trick, so no log(0) or overflow can occur.
// Rows of t are one-hot (or probabilities), so sum_j t_j = 1 -- or a class
// weight w: the row then counts w times, which is how rare classes are
// given more influence (see one_hot with class weights).
struct SoftmaxCrossEntropy {
    template <std::size_t...> static constexpr std::size_t rank = 0;

    template <class X, class Y>
    Shape<0> shape(const X& x, const Y& t) const {
        contract_assert(x.shape() == t.shape());
        return {};
    }

    template <class T>
    Tensor<T, 0> eval(const Tensor<T, 2>& x, const Tensor<T, 2>& t) const {
        const std::size_t n = x.shape()[0], c = x.shape()[1];
        auto xs = x.flat(), ts = t.flat();
        T loss{};
        for (std::size_t row = 0; row < n * c; row += c) {
            const T m = *std::max_element(xs.begin() + row, xs.begin() + row + c);
            T total{}, dot{}, weight{};
            for (std::size_t j = 0; j < c; ++j) {
                total += std::exp(xs[row + j] - m);
                dot += ts[row + j] * xs[row + j];
                weight += ts[row + j];
            }
            loss += weight * (m + std::log(total)) - dot;
        }
        return Tensor<T, 0>(Shape<0>{}, loss / static_cast<T>(n));
    }

    // d/dx = (softmax(x) * rowsum(t) - t) / n; for one-hot rows, (softmax(x) - t) / n.
    // Targets get no gradient.
    template <std::size_t I>
        requires(I == 0)
    auto grad(const auto& g, const auto&, const auto& x, const auto& t) const {
        return make_expr(SoftmaxCrossEntropyGrad{}, g, x, t);
    }
};

}  // namespace ops

template <TensorArg X>
    requires(std::remove_cvref_t<X>::rank >= 1)
auto softmax(X&& x) { return make_expr(ops::Softmax{}, std::forward<X>(x)); }

template <TensorArg X, TensorArg Y>
    requires(std::remove_cvref_t<X>::rank == 2 && std::remove_cvref_t<Y>::rank == 2)
auto softmax_cross_entropy(X&& logits, Y&& targets) {
    return make_expr(ops::SoftmaxCrossEntropy{}, std::forward<X>(logits), std::forward<Y>(targets));
}

template <TensorArg P, TensorArg Y>
auto mse(P&& pred, Y&& target) { return mean(square(std::forward<P>(pred) - std::forward<Y>(target))); }

// Class labels to one-hot rows: label 2 of 4 -> [0, 0, 1, 0]. With class
// weights, the 1 becomes the weight of the row's class: [0, 0, w2, 0].
template <class T = float>
Matrix<T> one_hot(std::span<const std::size_t> labels, std::size_t classes, std::span<const double> weights = {})
    pre(weights.empty() || weights.size() == classes)
{
    Matrix<T> out(Shape{labels.size(), classes});
    auto m = out.mut();
    for (std::size_t i = 0; i < labels.size(); ++i) {
        contract_assert(labels[i] < classes);
        m[i, labels[i]] = weights.empty() ? T{1} : static_cast<T>(weights[labels[i]]);
    }
    return out;
}

// Weights that give every class the same total influence:
// w_c = n / (classes * n_c), so the average weight over the samples is 1.
inline std::vector<double> balanced_class_weights(std::span<const std::size_t> labels, std::size_t classes) {
    std::vector<double> count(classes), w(classes);
    for (auto l : labels) count[l] += 1;
    for (std::size_t c = 0; c < classes; ++c)
        w[c] = count[c] > 0 ? double(labels.size()) / (double(classes) * count[c]) : 0.0;
    return w;
}

// The index of the largest element of each row: the predicted class.
template <TensorArg X>
    requires(std::remove_cvref_t<X>::rank == 2)
std::vector<std::size_t> argmax_rows(const X& x) {
    const auto t = x.eval();
    const std::size_t c = t.shape()[1];
    auto s = t.flat();
    std::vector<std::size_t> out;
    for (std::size_t row = 0; row < s.size(); row += c)
        out.push_back(std::size_t(std::max_element(s.begin() + row, s.begin() + row + c) - (s.begin() + row)));
    return out;
}

}  // namespace xinn
