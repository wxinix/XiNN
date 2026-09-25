// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <format>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

// Elements of any tensor-like value, as floats, for comparing.
template <class X>
std::vector<float> values(const X& x) {
    auto t = x.eval();
    return {t.flat().begin(), t.flat().end()};
}

}  // namespace

namespace tests {

void building_is_lazy_and_typed() {
    Matrix<float> a(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    Matrix<float> b(Shape(2, 3), {6, 5, 4, 3, 2, 1});

    auto e = a + b;
    static_assert(std::same_as<decltype(e), Expr<ops::Add, Matrix<float>, Matrix<float>>>);
    static_assert(TensorOfRank<decltype(e), 2>);
    check::equal(e.shape(), Shape(2, 3));   // known before evaluation
    check::that(a.shared());                // e holds shallow copies of a and b
}

void elementwise_arithmetic() {
    Vector<float> a(Shape(3), {1, 2, 3});
    Vector<float> b(Shape(3), {4, 5, 6});
    check::equal(values(a + b), std::vector<float>{5, 7, 9});
    check::equal(values(a - b), std::vector<float>{-3, -3, -3});
    check::equal(values(a * b), std::vector<float>{4, 10, 18});
    check::equal(values(b / a), std::vector<float>{4, 2.5f, 2});
    check::equal(values(-a), std::vector<float>{-1, -2, -3});
}

void numbers_mix_with_tensors() {
    Vector<float> a(Shape(3), {1, 2, 3});
    check::equal(values(a * 2), std::vector<float>{2, 4, 6});
    check::equal(values(1 - a), std::vector<float>{0, -1, -2});
    check::equal(values(a / 2.0), std::vector<float>{0.5f, 1, 1.5f});
    static_assert(std::same_as<decltype(a * 2)::value_type, float>);   // 2 became a float
}

void bias_broadcasts_over_rows() {
    Matrix<float> x(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    Vector<float> b(Shape(3), {10, 20, 30});
    auto y = x + b;
    check::equal(y.shape(), Shape(2, 3));
    check::equal(values(y), std::vector<float>{11, 22, 33, 14, 25, 36});
}

void nested_expressions_fuse() {
    Vector<float> a(Shape(3), {1, 2, 3});
    Vector<float> b(Shape(3), {1, 1, 1});
    auto e = (a + b) * (a - b) + 1;   // one loop, no temporaries
    check::equal(values(e), std::vector<float>{1, 4, 9});
}

void math_functions() {
    Vector<float> x(Shape(3), {-1, 0, 2});
    check::equal(values(relu(x)), std::vector<float>{0, 0, 2});
    check::equal(values(square(x)), std::vector<float>{1, 0, 4});
    check::near(sigmoid(x).eval()(1), 0.5);
    check::near(tanh(x).eval()(2), std::tanh(2.0f));
    check::near(log(exp(x)).eval()(2), 2.0);
    check::near(sqrt(square(x)).eval()(0), 1.0);
}

void map_takes_any_callable() {
    Vector<float> a(Shape(3), {1, 5, 3});
    Vector<float> b(Shape(3), {4, 2, 6});
    auto m = map([](float p, float q) { return p > q ? p : q; }, a, b);
    check::equal(values(m), std::vector<float>{4, 5, 6});

    auto is_big = map([](float v) { return v > 2; }, a);
    static_assert(std::same_as<decltype(is_big)::value_type, bool>);
}

void tensor_converts_from_expressions() {
    Vector<float> a(Shape(2), {1, 2});
    Vector<float> c = a * a + a;        // evaluated on conversion
    check::equal(c(1), 6.0f);
    check::that(!c.shared());
}

// ---- rewrite rules -------------------------------------------------------------

void adding_zeros_is_free() {
    Matrix<float> x(Shape(2, 2), {1, 2, 3, 4});
    auto e = x + zeros(Shape(2, 2));
    static_assert(std::same_as<decltype(e), Matrix<float>>);   // no Expr at all
    check::that(e.identical(x));

    auto f = zeros(Shape(2)) + x;                               // broadcast zeros
    static_assert(std::same_as<decltype(f), Matrix<float>>);
}

void multiplying_by_ones_and_zeros() {
    Matrix<float> x(Shape(2, 2), {1, 2, 3, 4});
    static_assert(std::same_as<decltype(x * ones(Shape(2, 2))), Matrix<float>>);
    static_assert(std::same_as<decltype(x / ones(Shape(2))), Matrix<float>>);

    auto z = x * zeros(Shape(2));
    static_assert(std::same_as<decltype(z), Zeros<float, 2>>);
    check::equal(z.shape(), Shape(2, 2));
}

void double_negation_cancels() {
    Vector<float> x(Shape(2), {1, 2});
    static_assert(std::same_as<decltype(-(-x)), Vector<float>>);
    static_assert(std::same_as<decltype(zeros(Shape(2)) - x), Expr<ops::Neg, Vector<float>>>);
}

// ---- structural operations -------------------------------------------------------

void matmul_values() {
    Matrix<float> a(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    Matrix<float> b(Shape(3, 2), {7, 8, 9, 10, 11, 12});
    auto c = matmul(a, b);
    check::equal(c.shape(), Shape(2, 2));
    check::equal(values(c), std::vector<float>{58, 64, 139, 154});
}

void transpose_values_and_rewrite() {
    Matrix<float> a(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    check::equal(transpose(a).shape(), Shape(3, 2));
    check::equal(values(transpose(a)), std::vector<float>{1, 4, 2, 5, 3, 6});
    static_assert(std::same_as<decltype(transpose(transpose(a))), Matrix<float>>);
}

void matmul_absorbs_transposes() {
    Matrix<float> a(Shape(3, 2), {1, 4, 2, 5, 3, 6});    // = transpose of [[1,2,3],[4,5,6]]
    Matrix<float> b(Shape(2, 3), {7, 9, 11, 8, 10, 12});  // = transpose of [[7,8],[9,10],[11,12]]

    auto tn = matmul(transpose(a), transpose(b));
    static_assert(std::same_as<decltype(tn), Expr<ops::MatMul<true, true>, Matrix<float>, Matrix<float>>>);
    check::equal(values(tn), std::vector<float>{58, 64, 139, 154});

    Matrix<float> a2(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    auto nt = matmul(a2, transpose(b));
    static_assert(std::same_as<decltype(nt)::op_type, ops::MatMul<false, true>>);
    check::equal(values(nt), std::vector<float>{58, 64, 139, 154});
}

void sums_and_means() {
    Matrix<float> m(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    check::equal(sum(m).eval()(), 21.0f);
    check::equal(values(sum<0>(m)), std::vector<float>{5, 7, 9});    // down the columns
    check::equal(values(sum<1>(m)), std::vector<float>{6, 15});      // along the rows
    check::near(mean(m).eval()(), 3.5);
    check::equal(values(mean<1>(m)), std::vector<float>{2, 5});
    static_assert(decltype(sum<1>(m))::rank == 1);
}

void structural_ops_nest_inside_elementwise() {
    Matrix<float> x(Shape(1, 2), {1, 2});
    Matrix<float> w(Shape(2, 2), {1, 0, 0, 1});
    Vector<float> b(Shape(2), {0.5f, -0.5f});
    auto y = relu(matmul(x, w) + b);   // the matmul is evaluated once, then fused
    check::equal(values(y), std::vector<float>{1.5f, 1.5f});
}

// ---- printing --------------------------------------------------------------------

void describe_shows_the_formula() {
    Matrix<float> x(Shape(4, 3));
    Matrix<float> w(Shape(3, 2));
    Vector<float> b(Shape(2));
    check::equal(describe(sigmoid(matmul(x, w) + b)),
                 std::string("sigmoid((matmul(T[4x3], T[3x2]) + T[2]))"));
    check::equal(describe(-x * 2), std::string("(-T[4x3] * 2)"));
    check::equal(describe(sum<1>(transpose(x))), std::string("sum(transpose(T[4x3]))"));
    check::equal(describe(map([](float v) { return v; }, b)), std::string("map(T[2])"));
}

void expressions_print_their_values() {
    Vector<int> a(Shape(3), {1, 2, 3});
    check::equal(std::format("{}", a * 10), std::string("[10, 20, 30]"));
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
