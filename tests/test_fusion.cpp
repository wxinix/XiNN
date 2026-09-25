// SPDX-License-Identifier: BSD-3-Clause
// Fusion, measured: count heap allocations while building and evaluating
// expressions. This file replaces the global operator new, so it is its own
// test program.
#include "check.hpp"

#include <cstdlib>
#include <new>
#include <xinn/xinn.hpp>

namespace {
std::size_t allocations = 0;
}

// GCC's -Wmismatched-new-delete cannot see that these are a matched pair.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

void* operator new(std::size_t n) {
    ++allocations;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

using namespace xinn;

namespace tests {

void building_expressions_allocates_nothing() {
    Matrix<float> a(Shape(64, 64), 1.0f);
    Vector<float> b(Shape(64), 2.0f);

    const auto before = allocations;
    auto e = sigmoid((a + b) * a - 3) / 2;
    check::equal(allocations - before, 0uz);
    (void)e;
}

void a_fused_tree_allocates_only_its_result() {
    Matrix<float> a(Shape(64, 64), 1.0f);
    Vector<float> b(Shape(64), 2.0f);
    auto e = sigmoid((a + b) * a - 3) / 2;

    const auto before = allocations;
    Matrix<float> r = e.eval();
    check::equal(allocations - before, 1uz);   // the output buffer, nothing else
    check::near(r(0, 0), 1.0 / (1.0 + std::exp(0.0)) / 2);
}

void a_structural_node_is_evaluated_once() {
    Matrix<float> x(Shape(8, 4), 1.0f);
    Matrix<float> w(Shape(4, 4), 1.0f);
    Vector<float> b(Shape(4), 1.0f);
    auto y = relu(matmul(x, w) + b);

    const auto before = allocations;
    Matrix<float> r = y.eval();
    // matmul's result and its two packing buffers (chapter 6), then the
    // final output of the fused + b and relu.
    check::equal(allocations - before, 4uz);
    check::equal(r(3, 2), 5.0f);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
