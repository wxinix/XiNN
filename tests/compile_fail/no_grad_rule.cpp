// SPDX-License-Identifier: BSD-3-Clause
// Must not compile: a lambda op has no gradient rule, but a Param flows
// through it. The error message names the operation.
#include <xinn/xinn.hpp>

int main() {
    xinn::Param w(xinn::Vector<float>(xinn::Shape(2), {1, 2}));
    auto loss = xinn::sum(xinn::map([](float v) { return v * v; }, w));
    xinn::backward(loss);
}
