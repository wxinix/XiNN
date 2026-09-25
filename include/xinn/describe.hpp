// SPDX-License-Identifier: BSD-3-Clause
// describe(x): an expression as a formula, for debugging.
//
//   describe(sigmoid(matmul(x, w) + b))  ->  "sigmoid((matmul(T[4x3], T[3x2]) + T[2]))"
//
// Operation names come from static reflection: the name of the op's type,
// lower-cased (ops::Sigmoid -> "sigmoid"). An op can override it with a
// static `name`; infix operators give a `symbol`. A lambda has no name and
// prints as "map".
#pragma once

#include <cstddef>
#include <format>
#include <meta>
#include <string>
#include <string_view>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

namespace detail {

consteval std::string_view reflected_name(std::meta::info type) {
    using namespace std::meta;
    if (has_template_arguments(type)) type = template_of(type);   // MatMul<true, false> -> MatMul
    if (!has_identifier(type)) return "map";                      // closure types have no name
    std::string s(identifier_of(type));
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return define_static_string(s);
}

template <class Op>
consteval std::string_view op_name() {
    if constexpr (requires { Op::name; }) return Op::name;
    else return reflected_name(^^Op);
}

}  // namespace detail

template <TensorArg X>
std::string describe(const X& x) {
    using Bare = std::remove_cvref_t<X>;
    if constexpr (requires { typename Bare::op_type; }) {
        using Op = typename Bare::op_type;
        return std::apply([&](const auto&... a) {
            if constexpr (requires { Op::symbol; } && sizeof...(a) == 2) {
                const std::string parts[] = {describe(a)...};
                return std::format("({} {} {})", parts[0], Op::symbol, parts[1]);
            } else if constexpr (requires { Op::symbol; } && sizeof...(a) == 1) {
                return std::format("{}{}", Op::symbol, describe(a)...);
            } else {
                std::string out(detail::op_name<Op>());
                out += '(';
                bool first = true;
                ((out += (first ? "" : ", "), out += describe(a), first = false), ...);
                return out + ')';
            }
        }, x.args());
    } else if constexpr (is_tensor_v<Bare> || requires { x.grad(); }) {
        std::string out = is_tensor_v<Bare> ? "T[" : "P[";   // tensor or Param
        for (std::size_t d = 0; d < Bare::rank; ++d) out += std::format("{}{}", d ? "x" : "", x.shape()[d]);
        return out + ']';
    } else if constexpr (requires { x.value(); }) {
        return std::format("{}", x.value());
    } else {
        return "?";
    }
}

}  // namespace xinn
