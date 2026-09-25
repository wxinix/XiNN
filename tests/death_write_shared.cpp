// SPDX-License-Identifier: BSD-3-Clause
// Must fail: writing through a handle that shares its buffer violates
// Tensor::mut_flat()'s precondition. CTest marks this test WILL_FAIL.
#include "death.hpp"

#include <xinn/tensor.hpp>

int main() {
    xinn::Vector<float> a(xinn::Shape(3), {1, 2, 3});
    xinn::Vector<float> b = a;       // b shares a's buffer
    b.mut_flat()[0] = 42.0f;         // contract violation -> handler exits(1)
    return 0;
}
