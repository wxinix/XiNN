// SPDX-License-Identifier: BSD-3-Clause
// Must not compile: nothing to differentiate.
#include <xinn/xinn.hpp>

int main() {
    xinn::Vector<float> x(xinn::Shape(2), {1, 2});
    xinn::backward(xinn::sum(x));
}
