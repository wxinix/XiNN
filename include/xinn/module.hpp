// SPDX-License-Identifier: BSD-3-Clause
// Modules: networks as plain structs.
//
//   struct MLP : Module {
//       Dense<{.activation = Activation::relu}> fc1;
//       Dense<> fc2;
//       auto forward(const auto& x) const { return fc2(fc1(x)); }
//   };
//
// The Module base gives every such struct, through deducing `this`:
//
//   net(x)                    calls net.forward(x)
//   net.for_each_param(f)     f(name, param) for every Param, found by
//                             reflection: "fc1.weight", "fc1.bias", ...
//   net.zero_grad(), net.parameter_count(), net.summary()
//
// No registration: a Param member is a parameter because of its type, and
// its name is the member's name.
#pragma once

#include <cstddef>
#include <format>
#include <meta>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xinn/param.hpp"

namespace xinn {

struct Module;

template <class X>
concept ModuleType = std::derived_from<std::remove_cvref_t<X>, Module>;

// A member of this type is not a parameter and is skipped (e.g. an absent bias).
struct Nothing {};

namespace detail {

// All data members of M, public or not.
template <class M>
consteval auto members_of() {
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(^^M, std::meta::access_context::unchecked()));
}

inline std::string join_name(std::string_view prefix, std::string_view name) {
    return prefix.empty() ? std::string(name) : std::format("{}.{}", prefix, name);
}

// A module whose parts are not named members (such as Sequential) lists
// them with children(), returning a tuple of references; they are named by
// position. All other modules are walked by reflection over their members.
template <class M>
concept HasChildren = requires(M& m) { m.children(); };

// Visit every Param in module m, depth first, in declaration order.
template <class M, class F>
void visit_params(M& m, F& f, std::string_view prefix) {
    using Bare = std::remove_cvref_t<M>;
    auto visit = [&](std::string_view name, auto& x) {
        using X = std::remove_cvref_t<decltype(x)>;
        if constexpr (is_param_v<X>) f(join_name(prefix, name), x);
        else if constexpr (ModuleType<X>) visit_params(x, f, join_name(prefix, name));
    };
    if constexpr (HasChildren<M>) {
        std::size_t i = 0;
        template for (auto& child : m.children()) visit(std::to_string(i++), child);
    } else {
        template for (constexpr auto member : members_of<Bare>()) visit(std::meta::identifier_of(member), m.[:member:]);
    }
}

// All Params of x as a tuple of handles, in the same order as visit_params.
// Unlike for_each_param, the result is typed: element I has the exact Param
// type of the I-th parameter. Optimizers use it to lay out their state.
template <class X>
auto param_tuple(X& x) {
    using Bare = std::remove_cvref_t<X>;
    if constexpr (is_param_v<Bare>) {
        return std::tuple<Bare>{x};
    } else if constexpr (ModuleType<Bare> && HasChildren<X>) {
        return std::apply([](auto&... c) { return std::tuple_cat(param_tuple(c)...); }, x.children());
    } else if constexpr (ModuleType<Bare>) {
        // The member list is looked up inside the splice, not captured: a
        // lambda that captured a consteval-only value would itself become
        // consteval, and could not touch the run-time object x.
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::tuple_cat(param_tuple(x.[:members_of<Bare>()[I]:])...);
        }(std::make_index_sequence<members_of<Bare>().size()>{});
    } else {
        return std::tuple<>{};
    }
}

// The number of Param members in M, counted from its type alone.
template <class M>
consteval std::size_t count_params() {
    if constexpr (requires { M::param_tensors; }) return M::param_tensors;   // e.g. Sequential
    std::size_t n = 0;
    template for (constexpr auto member : members_of<M>()) {
        using X = typename [:std::meta::type_of(member):];
        if constexpr (is_param_v<X>) n += 1;
        else if constexpr (ModuleType<X>) n += count_params<X>();
    }
    return n;
}

}  // namespace detail

struct Module {
    // net(x) is net.forward(x).
    template <class Self, class... Args>
    decltype(auto) operator()(this Self&& self, Args&&... args) {
        return std::forward<Self>(self).forward(std::forward<Args>(args)...);
    }

    // f(std::string name, Param& p) for every parameter.
    template <class Self, class F>
    void for_each_param(this Self& self, F&& f) {
        detail::visit_params(self, f, "");
    }

    void zero_grad(this auto& self) {
        self.for_each_param([](const std::string&, auto& p) { p.zero_grad(); });
    }

    std::size_t parameter_count(this const auto& self) {
        std::size_t n = 0;
        self.for_each_param([&](const std::string&, const auto& p) { n += p.shape().count(); });
        return n;
    }

    // A table of parameters: name, shape, number of values.
    std::string summary(this const auto& self) {
        std::string out = std::format("{:<24}{:<16}{:>10}\n", "parameter", "shape", "values");
        self.for_each_param([&](const std::string& name, const auto& p) {
            std::string shape = "[";
            for (std::size_t d = 0; d < p.shape().rank; ++d) shape += std::format("{}{}", d ? ", " : "", p.shape()[d]);
            out += std::format("{:<24}{:<16}{:>10}\n", name, shape + "]", p.shape().count());
        });
        return out + std::format("{:<40}{:>10}\n", "total", self.parameter_count());
    }
};

// Number of parameter tensors in a module type, at compile time.
template <ModuleType M>
inline constexpr std::size_t param_tensor_count = detail::count_params<M>();

}  // namespace xinn
