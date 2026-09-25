// SPDX-License-Identifier: BSD-3-Clause
// The matrix-product kernel behind matmul: C += op(A) · op(B).
//
// A direct triple loop reads each element of B from memory about m times.
// This kernel follows the well-known design of fast BLAS libraries, described
// in
//
//   K. Goto, R. A. van de Geijn. Anatomy of high-performance matrix
//   multiplication. ACM Transactions on Mathematical Software 34(3), 2008.
//   F. G. Van Zee, R. A. van de Geijn. BLIS: a framework for rapidly
//   instantiating BLAS functionality. ACM TOMS 41(3), 2015.
//
// in a simplified form, around three ideas:
//
//   1. Cache blocking. The k dimension is cut into slices of KC terms. One
//      slice of op(B) is packed into contiguous panels, NR columns wide, and
//      reused by every row of C while it sits in cache.
//   2. Register tiling. The innermost code computes an MR x NR tile of C in
//      SIMD registers (4 rows x 2 vectors), so each loaded value of A and B
//      is used for several multiply-adds before it is dropped.
//   3. Parallelism. Row blocks of C are independent, so they are computed
//      in parallel with xinn::parallel_for (std::execution).
//
// Transposed operands cost nothing extra: packing simply reads them in the
// other order.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <experimental/simd>
#include <vector>

#include "xinn/parallel.hpp"

namespace xinn::detail {

template <std::size_t N>
inline constexpr auto iota_array = [] {
    std::array<std::size_t, N> a{};
    for (std::size_t i = 0; i < N; ++i) a[i] = i;
    return a;
}();

// The register tile: R rows and 2 SIMD vectors of columns of C, accumulated
// over kc terms. ap: R packed rows of op(A), kc values each. bp: the packed
// B panel, 2W values per term. c: the top-left of the tile in C.
template <std::size_t R, class T>
void gemm_tile(std::size_t kc, const T* ap, const T* bp, T* c, std::size_t ldc, std::size_t cols) {
    using V = simd<T>;
    constexpr std::size_t W = V::size();
    namespace stdx = std::experimental;
    std::array<std::array<V, 2>, R> acc;
    template for (constexpr std::size_t r : iota_array<R>) acc[r] = {V(T{0}), V(T{0})};

    for (std::size_t p = 0; p < kc; ++p) {
        const V b0(bp + p * 2 * W, stdx::element_aligned);
        const V b1(bp + p * 2 * W + W, stdx::element_aligned);
        template for (constexpr std::size_t r : iota_array<R>) {   // unrolled: R is a constant
            const V a(ap[r * kc + p]);                                 // one value of A, in every lane
            using std::fma;
            acc[r][0] = fma(a, b0, acc[r][0]);
            acc[r][1] = fma(a, b1, acc[r][1]);
        }
    }

    template for (constexpr std::size_t r : iota_array<R>) {
        T* row = c + r * ldc;
        if (cols == 2 * W) {
            V c0(row, stdx::element_aligned), c1(row + W, stdx::element_aligned);
            (c0 + acc[r][0]).copy_to(row, stdx::element_aligned);
            (c1 + acc[r][1]).copy_to(row + W, stdx::element_aligned);
        } else {   // the right edge of C: fewer than 2W columns remain
            T tmp[2 * W];
            acc[r][0].copy_to(tmp, stdx::element_aligned);
            acc[r][1].copy_to(tmp + W, stdx::element_aligned);
            for (std::size_t j = 0; j < cols; ++j) row[j] += tmp[j];
        }
    }
}

// C (m x n, row-major) += op(A) · op(B). A is stored (m x k), or (k x m) when
// TransA; B is stored (k x n), or (n x k) when TransB.
template <bool TransA, bool TransB, class T>
void gemm(std::size_t m, std::size_t n, std::size_t k, const T* A, const T* B, T* C) {
    constexpr std::size_t W = simd<T>::size(), MR = 4, NR = 2 * W, KC = 256, MC = 96;
    auto a_at = [&](std::size_t i, std::size_t p) { return TransA ? A[p * m + i] : A[i * k + p]; };
    auto b_at = [&](std::size_t p, std::size_t j) { return TransB ? B[j * k + p] : B[p * n + j]; };

    const std::size_t panels = (n + NR - 1) / NR, blocks = (m + MC - 1) / MC;
    // Below about 128^3 multiply-adds, starting parallel tasks costs more than it saves.
    const bool parallel = double(m) * double(n) * double(k) >= 128.0 * 128 * 128;
    std::vector<T> Bp(KC * panels * NR);

    for (std::size_t p0 = 0; p0 < k; p0 += KC) {
        const std::size_t kc = std::min(KC, k - p0);

        // Pack rows p0..p0+kc of op(B) into panels of NR columns, zero-padded.
        parallel_for(panels, parallel ? 2 : panels, [&](std::size_t j0, std::size_t j1) {
            for (std::size_t jp = j0; jp < j1; ++jp) {
                T* dst = Bp.data() + jp * kc * NR;
                for (std::size_t p = 0; p < kc; ++p)
                    for (std::size_t c = 0; c < NR; ++c) {
                        const std::size_t j = jp * NR + c;
                        dst[p * NR + c] = j < n ? b_at(p0 + p, j) : T{0};
                    }
            }
        });

        // Row blocks of C, in parallel. Each task packs its own rows of op(A).
        parallel_for(blocks, parallel ? 1 : blocks, [&](std::size_t b0, std::size_t b1) {
            std::vector<T> Ap(MC * kc);
            for (std::size_t blk = b0; blk < b1; ++blk) {
                const std::size_t i0 = blk * MC, rows = std::min(MC, m - i0);
                for (std::size_t r = 0; r < rows; ++r)
                    for (std::size_t p = 0; p < kc; ++p) Ap[r * kc + p] = a_at(i0 + r, p0 + p);

                for (std::size_t jp = 0; jp < panels; ++jp) {
                    const T* bp = Bp.data() + jp * kc * NR;
                    const std::size_t cols = std::min(NR, n - jp * NR);
                    for (std::size_t r = 0; r < rows; r += MR) {
                        T* c = C + (i0 + r) * n + jp * NR;
                        const T* ap = Ap.data() + r * kc;
                        switch (std::min(MR, rows - r)) {
                            case 4: gemm_tile<4>(kc, ap, bp, c, n, cols); break;
                            case 3: gemm_tile<3>(kc, ap, bp, c, n, cols); break;
                            case 2: gemm_tile<2>(kc, ap, bp, c, n, cols); break;
                            default: gemm_tile<1>(kc, ap, bp, c, n, cols); break;
                        }
                    }
                }
            }
        });
    }
}

}  // namespace xinn::detail
