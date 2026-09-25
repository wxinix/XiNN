// SPDX-License-Identifier: BSD-3-Clause
// Parallel loops with C++26 std::execution, and SIMD types.
//
//   parallel_for(n, grain, [&](std::size_t begin, std::size_t end) { ... });
//
// splits [0, n) into chunks and runs them on the system's parallel
// scheduler:
//
//   sync_wait(schedule(get_parallel_scheduler())
//             | bulk(par, chunks, [&](std::size_t c) { f(begin(c), end(c)); }));
//
// This is standard C++26 (P2300 senders and receivers, P2079 parallel
// scheduler). GCC's library does not ship it yet, so the code runs on stdexec,
// the reference implementation of the proposal, under the same names; with a
// library that defines __cpp_lib_senders it uses <execution> itself. Without
// either, the loop runs on the calling thread.
#pragma once

#include <algorithm>
#include <cstddef>
#include <experimental/simd>
#include <thread>
#include <utility>
#include <version>

#if defined(__cpp_lib_senders)
#include <execution>
namespace xinn::detail {
namespace ex = std::execution;
using std::this_thread::sync_wait;
}  // namespace xinn::detail
#define XINN_PARALLEL 1
#elif defined(XINN_HAS_STDEXEC)
#include <stdexec/execution.hpp>
namespace xinn::detail {
namespace ex = stdexec;
using stdexec::sync_wait;
}  // namespace xinn::detail
#define XINN_PARALLEL 1
#else
#define XINN_PARALLEL 0
#endif

namespace xinn {

// Switches for experiments and benchmarks. The defaults are what you want.
struct Config {
    bool simd = true;           // vectorize fused element-wise loops
    bool threads = true;        // run large loops on the parallel scheduler
    std::size_t grain = 1 << 14;   // elements per chunk, at least
};
inline Config config;

inline constexpr bool has_threads = XINN_PARALLEL;

namespace detail {
// True on a thread that is running a task of parallel_for. A parallel loop
// started from inside another one runs serially: if every pool thread blocked
// in sync_wait on inner tasks, no thread would be left to run them.
inline thread_local bool in_parallel_region = false;
}  // namespace detail

inline std::size_t hardware_threads() {
    static const std::size_t n = std::max(1u, std::thread::hardware_concurrency());
    return n;
}

// Run f(begin, end) over [0, n). Loops of at least two grains are split into
// chunks and run in parallel; smaller ones run here, where the cost of
// starting tasks would outweigh the work.
template <class F>
void parallel_for(std::size_t n, std::size_t grain, F&& f) {
#if XINN_PARALLEL
    if (config.threads && n >= 2 * grain && hardware_threads() > 1 && !detail::in_parallel_region) {
        const std::size_t chunks = std::min(n / grain, 4 * hardware_threads());
        auto work = detail::ex::schedule(detail::ex::get_parallel_scheduler()) |
                    detail::ex::bulk(detail::ex::par, chunks, [&](std::size_t c) {
                        const bool outer = detail::in_parallel_region;
                        detail::in_parallel_region = true;
                        f(c * n / chunks, (c + 1) * n / chunks);
                        detail::in_parallel_region = outer;
                    });
        detail::sync_wait(std::move(work));
        return;
    }
#endif
    f(std::size_t{0}, n);
}

// The widest SIMD vector of T this CPU supports (std::simd's native ABI):
// 8 floats with AVX2, 16 with AVX-512.
template <class T>
using simd = std::experimental::native_simd<T>;

template <class T>
inline constexpr bool simd_element = std::same_as<T, float> || std::same_as<T, double>;

}  // namespace xinn
