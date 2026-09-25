// SPDX-License-Identifier: BSD-3-Clause
// Matrix operations: transpose and matmul.
//
// Both are structural: an output element depends on many input elements, so
// they cannot join a fused element-wise loop. Their operands are evaluated
// first.
//
// Rewrite rules:
//   transpose(transpose(x))   -> x
//   matmul(transpose(a), b)   -> one kernel that reads a transposed in place
//   matmul(a, transpose(b))   -> likewise for b
// The last two matter for training: gradients of a linear layer are exactly
// such products (chapter 3).
#pragma once

#include <cstddef>
#include <string_view>
#include <utility>

#include "xinn/concepts.hpp"
#include "xinn/expr.hpp"
#include "xinn/gemm.hpp"

namespace xinn {

namespace ops {

struct Transpose {
    template <std::size_t... R> static constexpr std::size_t rank = 2;

    template <class X>
    Shape<2> shape(const X& x) const { return Shape(x.shape()[1], x.shape()[0]); }

    template <class T>
    Tensor<T, 2> eval(const Tensor<T, 2>& x) const {
        Tensor<T, 2> out(shape(x));
        auto src = x.view();
        auto dst = out.mut();
        for (std::size_t i = 0; i < src.extent(0); ++i)
            for (std::size_t j = 0; j < src.extent(1); ++j) dst[j, i] = src[i, j];
        return out;
    }

    template <std::size_t> auto grad(const auto& g, const auto&, const auto&) const { return transpose(g); }
};

// C = op(A) · op(B), where op transposes when the flag is set.
template <bool TransA = false, bool TransB = false>
struct MatMul {
    static constexpr std::string_view name = TransA ? (TransB ? "matmul_tt" : "matmul_tn")
                                                    : (TransB ? "matmul_nt" : "matmul");

    template <std::size_t... R> static constexpr std::size_t rank = 2;

    template <class A, class B>
    Shape<2> shape(const A& a, const B& b) const {
        const auto [m, k]  = dims<TransA>(a.shape());
        const auto [k2, n] = dims<TransB>(b.shape());
        contract_assert(k == k2);   // inner dimensions must agree
        return Shape(m, n);
    }

    template <class T>
    Tensor<T, 2> eval(const Tensor<T, 2>& a, const Tensor<T, 2>& b) const {
        const auto [m, k] = dims<TransA>(a.shape());
        const std::size_t n = dims<TransB>(b.shape()).second;
        Tensor<T, 2> out(Shape(m, n));   // zeroed: the kernel adds into it
        if constexpr (simd_element<T>) {
            detail::gemm<TransA, TransB>(m, n, k, a.flat().data(), b.flat().data(), out.mut_flat().data());
        } else {   // integer matrices: the plain triple loop
            auto A = a.view();
            auto B = b.view();
            auto C = out.mut();
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t p = 0; p < k; ++p) {
                    const T aip = TransA ? A[p, i] : A[i, p];
                    for (std::size_t j = 0; j < n; ++j) C[i, j] += aip * (TransB ? B[j, p] : B[p, j]);
                }
        }
        return out;
    }

    // With A' = op(A) and B' = op(B), C = A'·B' gives
    //   dA' = G·B'^T      dB' = A'^T·G
    // and dA = op(dA'), dB = op(dB'). Written plainly with transpose();
    // the rewrite rules then fold every transpose into a MatMul flag, and
    // the kernel's packing step reads that operand in transposed order, so
    // no transposed tensor is built (e.g. dB for C = A·B is matmul_tn(A, G)).
    template <std::size_t I>
    auto grad(const auto& g, const auto&, const auto& a, const auto& b) const {
        if constexpr (I == 0) return maybe_transpose<TransA>(matmul(g, transpose(maybe_transpose<TransB>(b))));
        else return maybe_transpose<TransB>(matmul(transpose(maybe_transpose<TransA>(a)), g));
    }

private:
    template <bool Trans, class X>
    static auto maybe_transpose(X&& x) {
        if constexpr (Trans) return transpose(std::forward<X>(x));
        else return std::forward<X>(x);
    }

    // (rows, cols) of op(X).
    template <bool Trans>
    static std::pair<std::size_t, std::size_t> dims(const Shape<2>& s) {
        return Trans ? std::pair{s[1], s[0]} : std::pair{s[0], s[1]};
    }
};

}  // namespace ops

template <class X>
concept MatrixArg = TensorArg<X> && (std::remove_cvref_t<X>::rank == 2);

template <MatrixArg X>
auto transpose(X&& x) {
    if constexpr (ExprOf<X, ops::Transpose>) return x.template arg<0>();
    else return make_expr(ops::Transpose{}, std::forward<X>(x));
}

template <MatrixArg A, MatrixArg B>
auto matmul(A&& a, B&& b) {
    constexpr bool ta = ExprOf<A, ops::Transpose>;
    constexpr bool tb = ExprOf<B, ops::Transpose>;
    // Unwrap a transpose and tell the kernel instead.
    auto unwrap = []<bool T>(auto&& x) -> decltype(auto) {
        if constexpr (T) return x.template arg<0>();
        else return std::forward<decltype(x)>(x);
    };
    return make_expr(ops::MatMul<ta, tb>{},
                     unwrap.template operator()<ta>(std::forward<A>(a)),
                     unwrap.template operator()<tb>(std::forward<B>(b)));
}

}  // namespace xinn
