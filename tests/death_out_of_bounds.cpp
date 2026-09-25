// SPDX-License-Identifier: BSD-3-Clause
// Must fail: index 3 is out of range for a 2x3 matrix's second dimension.
#include "death.hpp"

#include <xinn/tensor.hpp>

int main() {
    xinn::Matrix<float> m(xinn::Shape(2, 3));
    return static_cast<int>(m(0, 3));   // contract violation -> handler exits(1)
}
