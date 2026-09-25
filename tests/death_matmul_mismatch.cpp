// SPDX-License-Identifier: BSD-3-Clause
// Must fail: a 2x3 matrix cannot multiply a 2x3 matrix (inner dims 3 != 2).
#include "death.hpp"

#include <xinn/xinn.hpp>

int main() {
    xinn::Matrix<float> a(xinn::Shape(2, 3));
    auto e = xinn::matmul(a, a);   // contract violation -> handler exits(1)
    (void)e;
    return 0;
}
