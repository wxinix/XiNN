// SPDX-License-Identifier: BSD-3-Clause
// std::format / std::print support for tensors.
//
//   std::println("{}", t);       // [[1, 2, 3], [4, 5, 6]]
//   std::println("{:.2f}", t);   // the spec applies to every element
//
// Any tensor-like value prints its elements; a lazy expression is evaluated
// first. (To see an expression's formula, use describe().)
#pragma once

#include <cstddef>
#include <format>
#include <mdspan>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/tensor.hpp"

template <class T, std::size_t R>
struct std::formatter<xinn::Tensor<T, R>> {
    std::formatter<T> elem;

    constexpr auto parse(std::format_parse_context& ctx) { return elem.parse(ctx); }

    auto format(const xinn::Tensor<T, R>& t, std::format_context& ctx) const {
        if (t.empty()) return std::format_to(ctx.out(), "[]");
        return write(t.view(), ctx);
    }

private:
    // Peel off the first dimension with submdspan until a single element is left.
    template <class V>
    auto write(const V& v, std::format_context& ctx) const {
        if constexpr (V::rank() == 0) {
            return elem.format(v[], ctx);
        } else {
            auto out = ctx.out();
            *out++ = '[';
            for (std::size_t i = 0; i < v.extent(0); ++i) {
                if (i) { *out++ = ','; *out++ = ' '; }
                ctx.advance_to(out);
                auto row = [&]<std::size_t... K>(std::index_sequence<K...>) {
                    return std::submdspan(v, i, ((void)K, std::full_extent)...);
                }(std::make_index_sequence<V::rank() - 1>{});
                out = write(row, ctx);
            }
            *out++ = ']';
            return out;
        }
    }
};

template <class X>
    requires(xinn::TensorLike<X> && !xinn::is_tensor_v<X>)
struct std::formatter<X> : std::formatter<xinn::Tensor<typename X::value_type, X::rank>> {
    auto format(const X& x, std::format_context& ctx) const {
        return std::formatter<xinn::Tensor<typename X::value_type, X::rank>>::format(x.eval(), ctx);
    }
};
