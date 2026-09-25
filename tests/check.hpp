// SPDX-License-Identifier: BSD-3-Clause
// A macro-free test harness.
//
// Every function in namespace `tests` is a test case. `run_tests<^^tests>()`
// finds them with static reflection and calls each in turn.
#pragma once

#include <cmath>
#include <cstdio>
#include <meta>
#include <print>
#include <source_location>
#include <string_view>
#include <vector>

namespace check {

inline int failures = 0;
inline std::string_view current_test;

inline void fail(std::source_location loc) {
    ++failures;
    std::println(stderr, "    FAIL in {} at {}:{}", current_test, loc.file_name(), loc.line());
}

inline void that(bool ok, std::source_location loc = std::source_location::current()) {
    if (!ok) fail(loc);
}

template <class A, class B>
void equal(const A& a, const B& b, std::source_location loc = std::source_location::current()) {
    if (!(a == b)) {
        fail(loc);
        if constexpr (std::formattable<A, char> && std::formattable<B, char>)
            std::println(stderr, "    expected {} == {}", a, b);
    }
}

inline void near(double a, double b, double eps = 1e-5,
                 std::source_location loc = std::source_location::current()) {
    if (!(std::abs(a - b) <= eps)) {
        fail(loc);
        std::println(stderr, "    expected {} ~= {} (eps {})", a, b, eps);
    }
}

// All functions declared directly in namespace NS, in declaration order.
template <std::meta::info NS>
consteval auto functions_in() {
    std::vector<std::meta::info> fs;
    for (auto m : std::meta::members_of(NS, std::meta::access_context::current()))
        if (std::meta::is_function(m)) fs.push_back(m);
    return std::define_static_array(fs);
}

template <std::meta::info NS>
int run_tests() {
    int count = 0;
    template for (constexpr auto f : functions_in<NS>()) {
        current_test = std::meta::identifier_of(f);
        std::println("[ run ] {}", current_test);
        [:f:]();
        ++count;
    }
    std::println("{} tests, {} failed checks", count, failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace check
