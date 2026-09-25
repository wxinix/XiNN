// SPDX-License-Identifier: BSD-3-Clause
// What it means to be "a tensor" in XiNN.
//
// The interface is a concept, so the compiler checks it; no category tags
// or SFINAE detection needed.
#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

// A tensor-like type has an element type, a compile-time rank, a run-time
// shape, and can be evaluated into the principal type Tensor<T, R>.
template <class X>
concept TensorLike = requires(const X& x) {
    typename X::value_type;
    requires std::same_as<std::remove_const_t<decltype(X::rank)>, std::size_t>;
    { x.shape() } -> std::same_as<const Shape<X::rank>&>;
    { x.eval() } -> std::same_as<Tensor<typename X::value_type, X::rank>>;
};

template <class X, std::size_t R>
concept TensorOfRank = TensorLike<X> && (X::rank == R);

// Handy for constraining forwarding references: TensorArg<const Tensor&> holds.
template <class X>
concept TensorArg = TensorLike<std::remove_cvref_t<X>>;

}  // namespace xinn
