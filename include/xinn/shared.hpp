// SPDX-License-Identifier: BSD-3-Clause
// Shared<T, R>: a value computed once and used many times.
//
// An expression stores its operands by value. Using one subexpression twice,
// as in a residual connection  x + f(x),  therefore evaluates it twice, and
// backward() walks it twice. Stack such layers and the work doubles with
// every layer: a network of L residual blocks evaluates its first block
// 2^L times.
//
//   auto h = share(x + attention(norm(x)));   // evaluated here, once
//   auto y = share(h + ffn(norm(h)));         // h is a leaf: using it twice is free
//   float loss = backward(cross_entropy(head(y), targets));
//
// share(e) records e at once (the forward pass of chapter 3) and keeps the
// tape. To the expressions that use it, a Shared is a leaf that collects a
// gradient, exactly like a Param. backward() on a loss that contains
// Shared nodes first backpropagates to them, then lets every node, newest
// first, pass its summed gradient down its own tape. Newest first is a valid
// order: a node can only use nodes created before it.
//
// This is the one place where XiNN erases a type. The shared nodes form a
// graph whose shape is known only at run time, so each node keeps its tape
// behind a std::move_only_function.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "xinn/autograd.hpp"
#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/param.hpp"

namespace xinn {

namespace detail {

inline std::atomic<std::uint64_t> shared_counter{0};

// What the graph walk needs from a node, whatever its element type and rank.
struct SharedNode {
    std::uint64_t id = shared_counter++;
    std::vector<std::shared_ptr<SharedNode>> inputs;   // the Shared nodes its expression used

    virtual ~SharedNode() = default;
    virtual void propagate() = 0;   // pass the summed gradient down the tape, then drop the tape
};

template <class T, std::size_t R>
struct SharedState final : SharedNode {
    Tensor<T, R> value, grad;
    std::move_only_function<void(const Tensor<T, R>&)> back;

    void propagate() override {
        if (back && !grad.empty()) back(grad);
        back = nullptr;   // the tape's intermediate values are no longer needed
    }
};

}  // namespace detail

template <class T, std::size_t R>
class Shared {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;

    const Shape<R>& shape() const noexcept { return s_->value.shape(); }
    Tensor<T, R> eval() const { return s_->value; }
    const Tensor<T, R>& tensor() const noexcept { return s_->value; }

    // The gradient summed from all uses; empty before backward().
    const Tensor<T, R>& grad() const noexcept { return s_->grad; }

    // Called by backprop, like Param::accumulate.
    template <TensorArg G>
    void accumulate(const G& g) const
        pre(g.shape() == shape())
    {
        s_->grad = s_->grad.empty() ? Tensor<T, R>(g.eval()) : Tensor<T, R>(s_->grad + g);
    }

    const std::shared_ptr<detail::SharedState<T, R>>& node() const noexcept { return s_; }

    // Made by share(), below.
    explicit Shared(std::shared_ptr<detail::SharedState<T, R>> s) : s_(std::move(s)) {}

private:
    std::shared_ptr<detail::SharedState<T, R>> s_;
};

template <class X> inline constexpr bool is_shared_v = false;
template <class T, std::size_t R> inline constexpr bool is_shared_v<Shared<T, R>> = true;

// For backward(), a Shared is a leaf that receives a gradient: a parameter.
template <class T, std::size_t R> inline constexpr bool is_param_v<Shared<T, R>> = true;

namespace detail {

// Does expression X use a Shared node anywhere? Known from its type.
template <class X> struct contains_shared : std::bool_constant<is_shared_v<X>> {};
template <class Op, class... Args>
struct contains_shared<Expr<Op, Args...>> : std::bool_constant<(contains_shared<Args>::value || ...)> {};

template <class X>
void collect_shared(const X& x, std::vector<std::shared_ptr<SharedNode>>& out) {
    if constexpr (is_shared_v<X>) out.push_back(x.node());
    else if constexpr (contains_shared<X>::value)
        std::apply([&](const auto&... a) { (collect_shared(a, out), ...); }, x.args());
}

}  // namespace detail

// Evaluate x now and remember how to send a gradient back through it.
template <TensorArg X>
auto share(X&& x) {
    using B = std::remove_cvref_t<X>;
    using T = typename B::value_type;
    constexpr std::size_t R = B::rank;
    if constexpr (is_shared_v<B>) {
        return B(std::forward<X>(x));
    } else {
        auto s = std::make_shared<detail::SharedState<T, R>>();
        detail::collect_shared(x, s->inputs);
        auto tape = detail::record(x);
        s->value = tape.value().eval();
        if constexpr (Trainable<B>)
            s->back = [tape = std::move(tape)](const Tensor<T, R>& g) { detail::backprop(tape, g); };
        return Shared<T, R>(std::move(s));
    }
}

// Backpropagate from a scalar Shared loss through the whole graph of shared
// nodes behind it. Each node's tape is used once and then released.
template <class T>
T backward(const Shared<T, 0>& loss) {
    loss.accumulate(Tensor<T, 0>(Shape<0>{}, T{1}));   // dLoss/dLoss = 1
    std::vector<detail::SharedNode*> nodes;
    std::vector<detail::SharedNode*> stack{loss.node().get()};
    while (!stack.empty()) {   // every node reachable from the loss, once
        auto* n = stack.back();
        stack.pop_back();
        if (std::ranges::contains(nodes, n)) continue;
        nodes.push_back(n);
        for (const auto& in : n->inputs) stack.push_back(in.get());
    }
    std::ranges::sort(nodes, std::ranges::greater{}, &detail::SharedNode::id);   // newest first
    for (auto* n : nodes) n->propagate();
    return loss.tensor()();
}

// A loss expression that uses Shared nodes: make it one, then backpropagate.
// (More specialized than the backward() of chapter 3, so it is chosen for
// such expressions; for all others that one applies.)
template <class Op, class... Args>
    requires(Expr<Op, Args...>::rank == 0 && detail::contains_shared<Expr<Op, Args...>>::value)
auto backward(const Expr<Op, Args...>& loss) {
    return backward(share(loss));
}

}  // namespace xinn
