// SPDX-License-Identifier: BSD-3-Clause
// Must fail: (3, 4) and (3, 4, 5) do not broadcast. The shape is checked when
// the expression is built, before any evaluation.
#include "death.hpp"

#include <xinn/xinn.hpp>

int main() {
    xinn::Matrix<float> a(xinn::Shape(3, 4));
    xinn::Tensor<float, 3> b(xinn::Shape(3, 4, 5));
    auto e = a + b;   // contract violation -> handler exits(1)
    (void)e;
    return 0;
}
