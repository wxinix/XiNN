// SPDX-License-Identifier: BSD-3-Clause
// Param<T, R>: a trainable tensor.
//
// A Param holds a value and the gradient of the loss with respect to it.
// Copies of a Param are handles to the same parameter: an expression that
// uses W stores a copy, and backward() accumulates into the one gradient.
//
// Updates replace the value instead of writing into it, so a tensor that
// someone still holds (a recorded forward pass, say) never changes under them.
#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

template <class T, std::size_t R>
class Param {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;

    explicit Param(Tensor<T, R> init)
        pre(!init.empty())
        : s_(std::make_shared<State>(State{init, Tensor<T, R>(init.shape())})) {}

    const Shape<R>& shape() const noexcept { return s_->value.shape(); }
    Tensor<T, R> eval() const { return s_->value; }
    const Tensor<T, R>& tensor() const noexcept { return s_->value; }

    // dLoss/dParam, summed over every backward() since the last zero_grad().
    const Tensor<T, R>& grad() const noexcept { return s_->grad; }
    void zero_grad() { s_->grad = Tensor<T, R>(shape()); }

    // Add g to the gradient. g has this parameter's shape. It is const, like
    // writing through a const shared_ptr: the handle is unchanged; backward()
    // calls it on the copies stored inside expressions.
    template <TensorArg G>
    void accumulate(const G& g) const
        pre(g.shape() == shape())
    {
        s_->grad = s_->grad + g;
    }

    // Replace the value, e.g. p.assign(p - lr * p.grad()).
    template <TensorArg X>
    void assign(const X& x)
        pre(x.shape() == shape())
    {
        s_->value = x.eval();
    }

    // Same parameter (not: same value).
    bool same(const Param& other) const noexcept { return s_ == other.s_; }

private:
    struct State {
        Tensor<T, R> value;
        Tensor<T, R> grad;
    };
    std::shared_ptr<State> s_;
};

template <class T, std::size_t R>
Param(Tensor<T, R>) -> Param<T, R>;

template <class X> inline constexpr bool is_param_v = false;
template <class T, std::size_t R> inline constexpr bool is_param_v<Param<T, R>> = true;

}  // namespace xinn
