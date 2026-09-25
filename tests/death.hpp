// SPDX-License-Identifier: BSD-3-Clause
// Support for death tests.
//
// C++26 lets a program replace the contract-violation handler. Ours prints
// the violated condition and exits with status 1, which CTest's WILL_FAIL
// understands (an abnormal termination would count as a crash instead).
#pragma once

#include <contracts>
#include <cstdlib>
#include <print>

void handle_contract_violation(const std::contracts::contract_violation& v) {
    std::println(stderr, "contract violated: {}", v.comment());
    std::exit(1);
}
