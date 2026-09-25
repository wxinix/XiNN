// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <xinn/shape.hpp>

using namespace xinn;

namespace tests {

void deduces_rank_from_arguments() {
    Shape s(3, 4);
    static_assert(std::same_as<decltype(s), Shape<2>>);
    check::equal(s.count(), 12uz);
    check::equal(s[0], 3uz);
    check::equal(s[1], 4uz);
}

void scalar_shape_has_one_element() {
    check::equal(Shape<0>{}.count(), 1uz);
}

void works_at_compile_time() {
    constexpr Shape s(2, 3, 4);
    static_assert(s.count() == 24);
    static_assert(s[2] == 4);
}

void broadcasting_aligns_trailing_dims() {
    check::that(broadcastable(Shape(4, 5), Shape(3, 4, 5)));
    check::that(broadcastable(Shape(5), Shape(3, 4, 5)));
    check::that(broadcastable(Shape<0>{}, Shape(3, 4)));
    check::that(!broadcastable(Shape(3, 4), Shape(3, 4, 5)));

    check::equal(broadcast(Shape(4, 5), Shape(3, 4, 5)), Shape(3, 4, 5));
    check::equal(broadcast(Shape(3, 4, 5), Shape(5)), Shape(3, 4, 5));
    static_assert(broadcast(Shape(2, 2), Shape(2)) == Shape(2, 2));
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
