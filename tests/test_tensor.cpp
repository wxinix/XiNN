// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <format>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace tests {

void concepts_accept_tensors_only() {
    static_assert(TensorLike<Tensor<float, 2>>);
    static_assert(TensorLike<Scalar<double>>);
    static_assert(TensorLike<Constant<float, 3>>);
    static_assert(TensorLike<Ones<float, 1>>);
    static_assert(TensorOfRank<Matrix<float>, 2>);
    static_assert(!TensorOfRank<Matrix<float>, 1>);
    static_assert(!TensorLike<float>);
    static_assert(!TensorLike<std::vector<float>>);
}

void constructs_from_values_row_major() {
    Matrix<float> m(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    check::equal(m.size(), 6uz);
    check::equal(m(0, 0), 1.0f);
    check::equal(m(0, 2), 3.0f);
    check::equal(m(1, 0), 4.0f);
    check::equal(m(1, 2), 6.0f);
}

void new_tensors_are_zero() {
    Tensor<int, 3> t(Shape(2, 2, 2));
    for (int x : t.flat()) check::equal(x, 0);
}

void scalar_has_one_element() {
    Scalar<float> s(Shape<0>{}, 2.5f);
    check::equal(s.size(), 1uz);
    check::equal(s(), 2.5f);
    s.set(4.0f);
    check::equal(s(), 4.0f);
}

void copies_are_shallow() {
    Vector<float> a(Shape(3), {1, 2, 3});
    check::that(!a.shared());

    Vector<float> b = a;
    check::that(a.shared() && b.shared());
    check::that(a.identical(b));
    check::equal(b(1), 2.0f);
}

void clone_is_deep_and_writable() {
    Vector<float> a(Shape(3), {1, 2, 3});
    Vector<float> b = a;
    Vector<float> c = a.clone();
    check::that(!c.shared());
    check::that(!c.identical(a));

    c.set(1, 20.0f);
    check::equal(c(1), 20.0f);
    check::equal(a(1), 2.0f);
}

void set_takes_indices_then_value() {
    Tensor<float, 3> t(Shape(2, 3, 4));
    t.set(1, 2, 3, 7.0f);
    check::equal(t(1, 2, 3), 7.0f);
    check::equal(t.flat()[1 * 12 + 2 * 4 + 3], 7.0f);
}

void slice_shares_memory() {
    Matrix<float> m(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    {
        Vector<float> row = m[1];
        check::equal(row.shape(), Shape(3));
        check::equal(row(0), 4.0f);
        check::equal(row(2), 6.0f);
        check::that(m.shared());  // the row is another owner of the buffer
    }
    check::that(!m.shared());     // row is gone; m is writable again
}

void slice_of_slice() {
    Tensor<int, 3> t(Shape(2, 2, 2), {0, 1, 2, 3, 4, 5, 6, 7});
    check::equal(t[1][0](1), 5);
    check::equal(t[1][1](0), 6);
}

void mdspan_view_and_submdspan_column() {
    Matrix<float> m(Shape(3, 2), {1, 2, 3, 4, 5, 6});
    auto v = m.view();
    check::equal(v[2, 1], 6.0f);

    auto col = std::submdspan(v, std::full_extent, 1);
    check::equal(col.extent(0), 3uz);
    check::equal(col[0], 2.0f);
    check::equal(col[2], 6.0f);
}

void formats_nested() {
    Matrix<int> m(Shape(2, 3), {1, 2, 3, 4, 5, 6});
    check::equal(std::format("{}", m), std::string("[[1, 2, 3], [4, 5, 6]]"));

    Vector<float> v(Shape(2), {0.5f, 1.25f});
    check::equal(std::format("{:.1f}", v), std::string("[0.5, 1.2]"));

    check::equal(std::format("{}", Scalar<int>(Shape<0>{}, 7)), std::string("7"));
}

void constant_evaluates_to_filled_tensor() {
    auto z = zeros(Shape(2, 2));
    static_assert(std::same_as<decltype(z), Zeros<float, 2>>);
    static_assert(Zeros<float, 2>::fill_value == 0.0f);   // the value is in the type
    check::equal(z.value(), 0.0f);

    Tensor<float, 2> t = ones(Shape(2, 3)).eval();
    check::equal(t.shape(), Shape(2, 3));
    for (float x : t.flat()) check::equal(x, 1.0f);

    check::equal(std::format("{}", Constant(Shape(2), 3)), std::string("[3, 3]"));
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
