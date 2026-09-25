// SPDX-License-Identifier: BSD-3-Clause
// Must not compile: a lambda has no OpenCL code, so it cannot be part of a
// GPU kernel. The error message names the operation.
#include <xinn/xinn.hpp>
#include <xinn/gpu.hpp>

int main() {
    xinn::Vector<float> v(xinn::Shape(4), 1.0f);
    auto e = xinn::map([](float a) { return a * a; }, v);
    return int(xinn::gpu::kernel_source<decltype(e)>.size());
}
