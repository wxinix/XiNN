// SPDX-License-Identifier: BSD-3-Clause
// OpenCL C kernels generated from expression types, at compile time.
//
// Chapter 2 fuses a tree of element-wise operations into one CPU loop: the
// compiler inlines the readers. A GPU runs code that is compiled by its
// driver, from OpenCL C source text. So here the tree is turned into that
// text, once per expression type, by a consteval function that walks the
// type:
//
//   sigmoid(x * w + b)   ->   __kernel void xinn_sigmoid_add_mul(__global float* out, const uint n,
//                                 __global const float* a0, const uint n0, ...)
//                             {
//                                 const uint i = get_global_id(0);
//                                 if (i >= n) return;
//                                 const float v0 = a0[i % n0];
//                                 ...
//                                 out[i] = v4;
//                             }
//
// Leaves become kernel arguments, in the order the tree is walked:
//
//   a tensor (or anything else)  __global const float* aK, const uint nK;
//                                broadcasting reads aK[i % nK], as in chapter 2
//   Constant (a run-time value)  const float cK
//   Filled (value in the type)   a literal: Zeros -> 0.0f, Ones -> 1.0f
//
// Each op gives its OpenCL C as an expression over $0, $1, ... (its
// operands): a static member `opencl` of the op, or a specialization of
// gpu::opencl_op<Op>. Kernel names come from the ops' reflected names, as
// in describe(). An op without GPU code (a lambda, say) is a compile error
// that names it.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <meta>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "xinn/describe.hpp"
#include "xinn/ops.hpp"

namespace xinn::gpu {

// OpenCL C for the element-wise ops of ops.hpp. Specialize for your own ops,
// or give the op a `static constexpr std::string_view opencl`.
template <class Op> struct opencl_op {};

template <> struct opencl_op<ops::Add>      { static constexpr std::string_view code = "$0 + $1"; };
template <> struct opencl_op<ops::Sub>      { static constexpr std::string_view code = "$0 - $1"; };
template <> struct opencl_op<ops::Mul>      { static constexpr std::string_view code = "$0 * $1"; };
template <> struct opencl_op<ops::Div>      { static constexpr std::string_view code = "$0 / $1"; };
template <> struct opencl_op<ops::Neg>      { static constexpr std::string_view code = "-$0"; };
template <> struct opencl_op<ops::Exp>      { static constexpr std::string_view code = "exp($0)"; };
template <> struct opencl_op<ops::Log>      { static constexpr std::string_view code = "log($0)"; };
template <> struct opencl_op<ops::Sqrt>     { static constexpr std::string_view code = "sqrt($0)"; };
template <> struct opencl_op<ops::Tanh>     { static constexpr std::string_view code = "tanh($0)"; };
template <> struct opencl_op<ops::Square>   { static constexpr std::string_view code = "$0 * $0"; };
template <> struct opencl_op<ops::Sigmoid>  { static constexpr std::string_view code = "1.0f / (1.0f + exp(-$0))"; };
template <> struct opencl_op<ops::Relu>     { static constexpr std::string_view code = "fmax($0, 0.0f)"; };
template <> struct opencl_op<ops::ReluGrad> { static constexpr std::string_view code = "$1 > 0.0f ? $0 : 0.0f"; };
template <> struct opencl_op<ops::Expand>   { static constexpr std::string_view code = "$0"; };

template <class Op>
concept HasOpenCL = requires { std::string_view(Op::opencl); } ||
                    requires { std::string_view(opencl_op<Op>::code); };

namespace detail {

template <class Op>
consteval std::string_view opencl_code() {
    if constexpr (requires { std::string_view(Op::opencl); }) return Op::opencl;
    else if constexpr (requires { std::string_view(opencl_op<Op>::code); }) return opencl_op<Op>::code;
    else return {};   // no code: emit() has already failed its static_assert
}

template <class Op>
consteval std::string_view no_opencl_message() {
    return std::define_static_string(
        std::string("xinn::gpu: operation `") + std::string(std::meta::display_string_of(^^Op)) +
        "` has no OpenCL code. Give the op a `static constexpr std::string_view opencl`, "
        "or specialize xinn::gpu::opencl_op for it.");
}

// ---- leaves ---------------------------------------------------------------------

// A node the kernel fuses: an element-wise expression.
template <class X>
concept FusedNode = requires { X::elementwise; } && X::elementwise;

enum class Leaf { buffer, constant, literal };

// How a leaf enters the kernel. Everything that is neither fused nor a
// constant is a buffer: tensors, device tensors, parameters, and structural
// expressions such as matmul (computed first, then read).
template <class X>
consteval Leaf leaf_kind() {
    if constexpr (requires { X::fill_value; }) return Leaf::literal;
    else if constexpr (requires(const X& x) { x.value(); }) return Leaf::constant;
    else return Leaf::buffer;
}

// ---- compile-time text ------------------------------------------------------------
//
// std::format does not run at compile time, so numbers are spelled by hand.

consteval std::string decimal(std::uint64_t v) {
    std::string s;
    do {
        s.insert(s.begin(), char('0' + v % 10));
        v /= 10;
    } while (v != 0);
    return s;
}

consteval std::string hex(std::uint64_t v, int digits) {
    std::string s(std::size_t(digits), '0');
    for (int d = digits - 1; d >= 0; --d, v >>= 4) s[std::size_t(d)] = "0123456789abcdef"[v & 15];
    return s;
}

// An exact OpenCL C literal for a float: whole numbers in decimal (1.0f),
// anything else as a hexadecimal float (0x1.8p-1f is 0.75), which C and
// OpenCL C read back bit for bit. Negative values are parenthesized, so a
// literal can replace $0 in "-$0".
consteval std::string float_literal(float v) {
    const auto bits = std::bit_cast<std::uint32_t>(v);
    const bool negative = bits >> 31;
    const int exponent = int((bits >> 23) & 0xFF);
    std::uint32_t mantissa = bits & 0x7FFFFF;
    std::string body;
    if (exponent == 0xFF) {
        body = mantissa ? "NAN" : "INFINITY";
    } else if (float m = negative ? -v : v; m < 16777216.0f && m == float(std::uint32_t(m))) {
        body = decimal(std::uint32_t(m)) + ".0f";
    } else {
        std::string digits = hex(std::uint64_t(mantissa) << 1, 6);   // 24 bits, 6 hex digits
        while (!digits.empty() && digits.back() == '0') digits.pop_back();
        const int e = exponent == 0 ? -126 : exponent - 127;       // subnormals: 0x0.xxxp-126
        body = std::string(exponent == 0 ? "0x0" : "0x1") + (digits.empty() ? "" : "." + digits) + "p" +
               (e < 0 ? "-" : "+") + decimal(std::uint64_t(e < 0 ? -e : e)) + "f";
    }
    return negative ? "(-" + body + ")" : body;
}

// The op's code with $0, $1, ... replaced by the operands' values.
consteval std::string substitute(std::string_view code, const std::string* operands, std::size_t count) {
    std::string out;
    for (std::size_t p = 0; p < code.size(); ++p) {
        if (code[p] == '$' && p + 1 < code.size() && code[p + 1] >= '0' && code[p + 1] <= '9') {
            const std::size_t k = std::size_t(code[++p] - '0');
            if (k >= count) throw "an op's OpenCL code names an operand it does not have";
            out += operands[k];
        } else {
            out += code[p];
        }
    }
    return out;
}

consteval std::uint32_t fnv1a(std::string_view s) {
    std::uint32_t h = 2166136261u;
    for (char c : s) h = (h ^ std::uint8_t(c)) * 16777619u;
    return h;
}

// ---- the walk -----------------------------------------------------------------

struct KernelText {
    std::string name = "xinn";   // op names, in pre-order
    std::string params;          // the leaves' arguments
    std::string body;            // one line per value
    int leaves = 0;              // leaf arguments so far (a0/n0, c1, ...)
    int values = 0;              // values so far (v0, v1, ...)
};

// Emit the code for node X; return the OpenCL C that holds its value at i.
template <class X>
consteval std::string emit(KernelText& k) {
    static_assert(std::same_as<typename X::value_type, float>, "xinn::gpu: kernels compute in float");
    if constexpr (FusedNode<X>) {
        using Op = typename X::op_type;
        static_assert(HasOpenCL<Op>, no_opencl_message<Op>());
        k.name += "_" + std::string(xinn::detail::op_name<Op>());
        std::string operands[X::arity];
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((operands[I] = emit<typename X::template arg_type<I>>(k)), ...);   // left to right
        }(std::make_index_sequence<X::arity>{});
        const std::string v = "v" + decimal(std::uint64_t(k.values++));
        k.body += "    const float " + v + " = " + substitute(opencl_code<Op>(), operands, X::arity) + ";\n";
        return v;
    } else if constexpr (leaf_kind<X>() == Leaf::literal) {
        return float_literal(X::fill_value);
    } else if constexpr (leaf_kind<X>() == Leaf::constant) {
        const std::string c = "c" + decimal(std::uint64_t(k.leaves++));
        k.params += ",\n    const float " + c;
        return c;
    } else {
        const std::string j = decimal(std::uint64_t(k.leaves++));
        const std::string v = "v" + decimal(std::uint64_t(k.values++));
        k.params += ",\n    __global const float* a" + j + ", const uint n" + j;
        k.body += "    const float " + v + " = a" + j + "[i % n" + j + "];\n";
        return v;
    }
}

template <class X>
consteval KernelText generate() {
    KernelText k;
    const std::string result = emit<X>(k);
    k.body += "    out[i] = " + result + ";\n";
    return k;
}

// Long trees would give long names: keep the first ops and add a hash.
template <class X>
consteval std::string kernel_name_of() {
    const KernelText k = generate<X>();
    if (k.name.size() <= 48) return k.name;
    return k.name.substr(0, 40) + "_" + hex(fnv1a(k.body + k.params), 8);
}

template <class X>
consteval std::string kernel_source_of() {
    const KernelText k = generate<X>();
    return "__kernel void " + kernel_name_of<X>() + "(__global float* out, const uint n" + k.params +
           ")\n{\n    const uint i = get_global_id(0);\n    if (i >= n) return;\n" + k.body + "}\n";
}

}  // namespace detail

// The kernel for an element-wise expression type, as static strings: made
// once, at compile time, and usable at run time.
template <class X>
    requires detail::FusedNode<std::remove_cvref_t<X>>
inline constexpr std::string_view kernel_name =
    std::define_static_string(detail::kernel_name_of<std::remove_cvref_t<X>>());

template <class X>
    requires detail::FusedNode<std::remove_cvref_t<X>>
inline constexpr std::string_view kernel_source =
    std::define_static_string(detail::kernel_source_of<std::remove_cvref_t<X>>());

// ---- matrix product -------------------------------------------------------------
//
// C = op(A) · op(B), op(A) of M x K, op(B) of K x N, all row-major. Each work
// group computes a TS x TS tile of C. It walks the inner dimension in steps
// of TS: the group loads one TS x TS tile of A and one of B into local memory
// (fast, shared by the group), waits at a barrier, and every work item adds
// TS products to its element from there. Each number read from global memory
// is then used TS times instead of once.
//
// TS, TRANS_A and TRANS_B are set when the program is built (-D options),
// so the four transpose combinations are four cached programs. This is the
// first, simple version; gpu::run uses matmul_blocked_source below.
inline constexpr std::string_view matmul_source = R"CL(
#if TRANS_A
#define A_AT(r, p) A[(p) * M + (r)]
#else
#define A_AT(r, p) A[(r) * K + (p)]
#endif
#if TRANS_B
#define B_AT(p, c) B[(c) * K + (p)]
#else
#define B_AT(p, c) B[(p) * N + (c)]
#endif

__kernel void xinn_matmul(const uint M, const uint N, const uint K,
                          __global const float* A, __global const float* B, __global float* C)
{
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    const uint lr = get_local_id(1), lc = get_local_id(0);
    const uint row = get_group_id(1) * TS + lr, col = get_group_id(0) * TS + lc;
    float acc = 0.0f;
    for (uint t = 0; t < K; t += TS) {
        As[lr][lc] = row < M && t + lc < K ? A_AT(row, t + lc) : 0.0f;
        Bs[lr][lc] = t + lr < K && col < N ? B_AT(t + lr, col) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint p = 0; p < TS; ++p) acc = fma(As[lr][p], Bs[p][lc], acc);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
)CL";

// The same product with register blocking. The kernel above does one
// multiply-add per two reads of local memory, and local memory, not
// arithmetic, sets its speed. Here each work item computes a WPT x WPT block
// of C, held in registers: per step of the inner dimension it reads WPT
// numbers of A and WPT of B from local memory and does WPT² multiply-adds
// with them. A group of RTS x RTS items then covers a TS x TS tile of C, TS =
// RTS * WPT (64 for 16 x 16 groups and WPT 4, 128 for WPT 8).
//
// An item's elements are RTS apart, not adjacent: rows lr, lr + RTS, ...,
// columns lc, lc + RTS, .... Neighbouring items then read neighbouring
// numbers of Bs and write neighbouring numbers of C, which the hardware
// combines. The local arrays get one column of padding, so that storing a
// column of a tile does not hit one memory bank TSK times.
//
// Build options: RTS (the group's side), WPT (4 or 8), VEC (1: load tiles
// from global memory four floats at a time, with vload4, along the direction
// in which each operand is contiguous), TRANS_A, TRANS_B.
inline constexpr std::string_view matmul_blocked_source = R"CL(
#define TS (RTS * WPT)
#define TSK 16
#define LPT (TSK * TS / (RTS * RTS))

#if TRANS_A
#define A_AT(r, p) A[(p) * M + (r)]
#else
#define A_AT(r, p) A[(r) * K + (p)]
#endif
#if TRANS_B
#define B_AT(p, c) B[(c) * K + (p)]
#else
#define B_AT(p, c) B[(p) * N + (c)]
#endif

#if VEC
/* Four consecutive floats at p; at an edge, only the first `left`, then zeros. */
float4 xinn_load4(__global const float* p, const uint left)
{
    if (left >= 4) return vload4(0, p);
    float4 v;
    v.x = left > 0 ? p[0] : 0.0f;
    v.y = left > 1 ? p[1] : 0.0f;
    v.z = left > 2 ? p[2] : 0.0f;
    v.w = 0.0f;
    return v;
}
#endif

__kernel void xinn_matmul_blocked(const uint M, const uint N, const uint K,
                                  __global const float* A, __global const float* B, __global float* C)
{
    __local float As[TSK][TS + 1];
    __local float Bs[TSK][TS + 1];
    const uint lr = get_local_id(1), lc = get_local_id(0), id = lr * RTS + lc;
    const uint row0 = get_group_id(1) * TS, col0 = get_group_id(0) * TS;
    float acc[WPT][WPT], a[WPT], b[WPT];
    for (uint i = 0; i < WPT; ++i)
        for (uint j = 0; j < WPT; ++j) acc[i][j] = 0.0f;

    for (uint t = 0; t < K; t += TSK) {
#if VEC
        for (uint l = 0; l < LPT / 4; ++l) {
            const uint e = id + l * RTS * RTS;   /* consecutive items read consecutive float4s */
#if TRANS_A
            const uint am = 4 * (e % (TS / 4)), ak = e / (TS / 4);   /* A^T is stored K x M: 4 rows of C */
            const uint left = t + ak < K && row0 + am < M ? M - row0 - am : 0;
            const float4 v = xinn_load4(&A_AT(row0 + am, t + ak), left);
            As[ak][am] = v.x; As[ak][am + 1] = v.y; As[ak][am + 2] = v.z; As[ak][am + 3] = v.w;
#else
            const uint ak = 4 * (e % (TSK / 4)), am = e / (TSK / 4);   /* A is stored M x K: 4 terms of K */
            const uint left = row0 + am < M && t + ak < K ? K - t - ak : 0;
            const float4 v = xinn_load4(&A_AT(row0 + am, t + ak), left);
            As[ak][am] = v.x; As[ak + 1][am] = v.y; As[ak + 2][am] = v.z; As[ak + 3][am] = v.w;
#endif
#if TRANS_B
            const uint bk = 4 * (e % (TSK / 4)), bn = e / (TSK / 4);   /* B^T is stored N x K: 4 terms of K */
            const uint bleft = col0 + bn < N && t + bk < K ? K - t - bk : 0;
            const float4 u = xinn_load4(&B_AT(t + bk, col0 + bn), bleft);
            Bs[bk][bn] = u.x; Bs[bk + 1][bn] = u.y; Bs[bk + 2][bn] = u.z; Bs[bk + 3][bn] = u.w;
#else
            const uint bn = 4 * (e % (TS / 4)), bk = e / (TS / 4);   /* B is stored K x N: 4 columns of C */
            const uint bleft = t + bk < K && col0 + bn < N ? N - col0 - bn : 0;
            const float4 u = xinn_load4(&B_AT(t + bk, col0 + bn), bleft);
            Bs[bk][bn] = u.x; Bs[bk][bn + 1] = u.y; Bs[bk][bn + 2] = u.z; Bs[bk][bn + 3] = u.w;
#endif
        }
#else
        for (uint l = 0; l < LPT; ++l) {
            const uint e = id + l * RTS * RTS;   /* consecutive items read consecutive addresses */
#if TRANS_A
            const uint am = e % TS, ak = e / TS;
#else
            const uint ak = e % TSK, am = e / TSK;
#endif
            As[ak][am] = row0 + am < M && t + ak < K ? A_AT(row0 + am, t + ak) : 0.0f;
#if TRANS_B
            const uint bk = e % TSK, bn = e / TSK;
#else
            const uint bn = e % TS, bk = e / TS;
#endif
            Bs[bk][bn] = t + bk < K && col0 + bn < N ? B_AT(t + bk, col0 + bn) : 0.0f;
        }
#endif
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint p = 0; p < TSK; ++p) {
            for (uint w = 0; w < WPT; ++w) {
                a[w] = As[p][lr + w * RTS];
                b[w] = Bs[p][lc + w * RTS];
            }
            for (uint i = 0; i < WPT; ++i)
                for (uint j = 0; j < WPT; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = 0; i < WPT; ++i)
        for (uint j = 0; j < WPT; ++j) {
            const uint r = row0 + lr + i * RTS, c = col0 + lc + j * RTS;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}
)CL";

}  // namespace xinn::gpu
