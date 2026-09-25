// SPDX-License-Identifier: BSD-3-Clause
// Tensors whose elements all hold the same value.
//
//   Constant<T, R>     the value is known at run time
//   Filled<T, R, V>    the value V is part of the type
//   Zeros<T, R>        Filled<T, R, 0>
//   Ones<T, R>         Filled<T, R, 1>
//
// Each stores a shape and at most one number, not size() numbers. Because
// Zeros and Ones carry their value in the type, operations can simplify at
// compile time: x + zeros is x, x * ones is x.
#pragma once

#include <cstddef>

#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

template <class T, std::size_t R>
class Constant {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;

    constexpr Constant(const Shape<R>& shape, T value) : shape_(shape), value_(value) {}

    constexpr const Shape<R>& shape() const noexcept { return shape_; }
    constexpr T value() const noexcept { return value_; }

    Tensor<T, R> eval() const { return Tensor<T, R>(shape_, value_); }

private:
    Shape<R> shape_;
    T value_;
};

template <class T, std::size_t R>
Constant(Shape<R>, T) -> Constant<T, R>;

template <class T, std::size_t R, T V>
class Filled {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;
    static constexpr T fill_value = V;

    constexpr explicit Filled(const Shape<R>& shape) : shape_(shape) {}

    constexpr const Shape<R>& shape() const noexcept { return shape_; }
    constexpr T value() const noexcept { return V; }

    Tensor<T, R> eval() const { return Tensor<T, R>(shape_, V); }

private:
    Shape<R> shape_;
};

template <class T, std::size_t R> using Zeros = Filled<T, R, T{0}>;
template <class T, std::size_t R> using Ones  = Filled<T, R, T{1}>;

template <class T = float, std::size_t R>
constexpr Zeros<T, R> zeros(const Shape<R>& shape) { return Zeros<T, R>(shape); }

template <class T = float, std::size_t R>
constexpr Ones<T, R> ones(const Shape<R>& shape) { return Ones<T, R>(shape); }

// Scalar constants, used when a plain number appears in an expression (x * 2).
template <class T>
constexpr Constant<T, 0> scalar(T value) { return Constant<T, 0>(Shape<0>{}, value); }

}  // namespace xinn
