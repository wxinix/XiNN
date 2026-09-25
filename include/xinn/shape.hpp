// SPDX-License-Identifier: BSD-3-Clause
// Shape<R>: the extents of a rank-R tensor.
//
// The rank R is a compile-time constant; the extent of each dimension is a
// run-time value. A Shape<2> may be 3x5 or 7x10, but never 3x5x2.
#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <mdspan>
#include <tuple>

namespace xinn {

template <std::size_t R>
struct Shape {
    static constexpr std::size_t rank = R;

    std::array<std::size_t, R> dims{};

    constexpr Shape() = default;

    constexpr explicit Shape(std::convertible_to<std::size_t> auto... ds)
        requires(sizeof...(ds) == R)
        : dims{static_cast<std::size_t>(ds)...} {}

    constexpr explicit Shape(const std::array<std::size_t, R>& ds) : dims(ds) {}

    constexpr std::size_t operator[](std::size_t i) const
        pre(i < R)
    {
        return dims[i];
    }

    // Number of elements. A rank-0 shape (a scalar) has exactly one.
    constexpr std::size_t count() const {
        return std::ranges::fold_left(dims, std::size_t{1}, std::multiplies{});
    }

    // The same extents in the form std::mdspan wants.
    constexpr std::dextents<std::size_t, R> extents() const { return std::dextents<std::size_t, R>(dims); }

    constexpr bool operator==(const Shape&) const = default;
};

// Shape(3, 4) deduces Shape<2>.
template <std::convertible_to<std::size_t>... Ds>
Shape(Ds...) -> Shape<sizeof...(Ds)>;

// Broadcasting: align shapes at the trailing end; every
// dimension present in both must match exactly. The lower-rank operand is
// repeated along the extra leading dimensions. (NumPy additionally stretches
// size-1 dimensions; we do not.)
//
//   (4, 5)    with (3, 4, 5)  ->  (3, 4, 5)
//   (5)       with (3, 4, 5)  ->  (3, 4, 5)
//   (3, 4)    with (3, 4, 5)  ->  error
template <std::size_t A, std::size_t B>
constexpr bool broadcastable(const Shape<A>& a, const Shape<B>& b) {
    for (std::size_t i = 0; i < std::min(A, B); ++i)
        if (a.dims[A - 1 - i] != b.dims[B - 1 - i]) return false;
    return true;
}

// The common shape of any number of shapes: the one with the largest rank,
// provided every shape is broadcastable with it.
template <std::size_t... Rs>
    requires(sizeof...(Rs) >= 1)
constexpr Shape<std::max({Rs...})> broadcast(const Shape<Rs>&... shapes) {
    constexpr std::size_t widest = [] {
        constexpr std::array ranks{Rs...};
        return std::size_t(std::ranges::max_element(ranks) - ranks.begin());
    }();
    const auto& result = shapes...[widest];               // pack indexing
    template for (const auto& s : std::tie(shapes...))    // expansion statement
        contract_assert(broadcastable(s, result));
    return result;
}

}  // namespace xinn
