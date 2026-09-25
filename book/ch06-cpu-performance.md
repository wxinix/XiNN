# 6. CPU Performance

Code: `include/xinn/parallel.hpp`, `gemm.hpp`, the readers in `expr.hpp`.
Tests: `tests/test_parallel.cpp`. Benchmark: `examples/bench_cpu.cpp`.

## 6.1 Where the time goes

Deep learning spends its time in two kinds of work:

- **Matrix products**: `n³` multiply-adds on `n²` numbers. There is plenty of
  arithmetic per byte, so the limit is how fast the CPU can compute.
- **Element-wise operations**: one or two operations per number. The limit
  is how fast memory can deliver the numbers.

A modern CPU core runs **SIMD** instructions that work on several numbers at
once (8 floats with AVX2), and a CPU has several cores. Using both matters
more for matrix products. Chapter 2's fusion already did the most important
thing for element-wise work: it touches memory once.

All numbers in this chapter come from `bench_cpu` on the machine used to
write the book: a virtual machine with 8 hardware threads and AVX2/FMA, in a
release build.

## 6.2 Standard parallelism: `std::execution`

C++26 adds **senders and receivers** (P2300), a model for asynchronous and
parallel work. A **scheduler** says where work runs. **Sender algorithms**
describe the work, and `sync_wait` runs it and waits:

```cpp
auto work = schedule(get_parallel_scheduler())
          | bulk(par, chunks, [&](std::size_t c) { f(begin(c), end(c)); });
sync_wait(std::move(work));
```

- `get_parallel_scheduler()` (P2079) is the system's thread pool, shared by
  the whole program. No pool has to be created or passed around.
- `bulk(par, n, f)` runs `f(0)` … `f(n − 1)`, in parallel if the scheduler
  can.
- The pipe `|` composes senders. Nothing runs until `sync_wait`.

XiNN wraps this in one function:

```cpp
template <class F>
void parallel_for(std::size_t n, std::size_t grain, F&& f);   // f(begin, end) over [0, n)
```

Loops shorter than two *grains* run on the calling thread, because starting
tasks costs more than the work they would do.

**Availability.** GCC 16's standard library does not ship `<execution>`
senders yet. XiNN uses **stdexec**, NVIDIA's reference implementation of
P2300, under the same names. CMake fetches it, pinned to one commit.
`parallel.hpp` switches to the standard header when the library defines
`__cpp_lib_senders`:

```cpp
#if defined(__cpp_lib_senders)
#include <execution>
namespace xinn::detail { namespace ex = std::execution; }
#elif defined(XINN_HAS_STDEXEC)
#include <stdexec/execution.hpp>
namespace xinn::detail { namespace ex = stdexec; }
#endif
```

A GPU is just another scheduler in this model. Chapter 7 relies on that.

## 6.3 Standard SIMD: `std::simd`

`std::simd<T>` (C++26, P1928) is a vector of `T` whose operators work on all
lanes at once. `native_simd<float>` has the width the CPU supports best. As
with senders, GCC 16 has the pre-standard `std::experimental::simd`, the
direct ancestor of `std::simd`, and XiNN uses that.

**Fused loops, vectorized.** Every reader of chapter 2 learns to load a whole
vector:

```cpp
template <class T>
struct PtrReader {
    const T* p;
    T operator()(std::size_t i) const { return p[i]; }
    template <class V> V load(std::size_t i) const { return V(p + i, element_aligned); }
};
```

`MapReader::load` applies the op to vectors instead of numbers. The same op
objects serve both, because their bodies are generic:

```cpp
struct Exp {
    auto operator()(auto a) const { using std::exp; return exp(a); }
};
```

For a `float`, `exp` is `std::exp`. For a SIMD vector, argument-dependent
lookup finds the `std::simd` overload. `Relu` uses `max`, which also exists
for vectors. `ReluGrad` needs a lane-wise select, `where(a > 0, out) = g`.

**Deciding at compile time.** An expression tree is vectorized only if every
op in it accepts and returns vectors:

```cpp
template <class Op, class T, class... Args>
consteval bool vectorizable_op() {
    ...
    return std::same_as<std::invoke_result_t<const Op&, simd<T>...>, simd<T>>;
}
static constexpr bool vectorizable = ...;   // a member of every Expr
```

A lambda such as `[](float v) { return v * 3; }` takes a `float`, not a
vector, so its tree falls back to the scalar loop. The test
`lambdas_fall_back_to_scalar` checks this with `static_assert`.

**Broadcasting in vectors.** Chapter 2 read a broadcast operand at
`i % n`. A vector of 8 lanes may cross the end of the repeated block. `Wrap`
handles three cases: the operand is full size (a plain load), a single value
(repeated in every lane), or wraps inside the vector (lanes are filled one by
one).

## 6.4 Results: element-wise

On 2048×2048 floats (times in ms, and speed-up against scalar on one
thread):

| expression | scalar, 1 thread | SIMD, 1 thread | scalar, threads | SIMD, threads |
|---|---|---|---|---|
| `x + w` | 4.09 | 4.19 (1.0×) | 2.13 (1.9×) | **1.86 (2.2×)** |
| `x * w + b` | 10.07 | 5.37 (1.9×) | 3.19 (3.2×) | **2.20 (4.6×)** |
| `sigmoid(x * w + b) * 0.5` | 16.05 | 13.70 (1.2×) | 3.16 (5.1×) | 3.35 (4.8×) |
| `tanh(x) * exp(-square(w))` | 44.05 | 40.65 (1.1×) | 8.08 (5.5×) | **6.37 (6.9×)** |

Three observations:

- **`x + w` is limited by memory.** Neither SIMD nor more cores help much:
  the CPU already adds faster than memory delivers. Only fusion (chapter 2)
  helps here.
- **SIMD pays for arithmetic** (`x * w + b`: 1.9×).
- **SIMD barely helps `exp` and `tanh` here.** libstdc++'s experimental
  `simd` computes these functions one lane at a time. That is a limitation of
  this implementation, not of `std::simd`. Threads still help, by up to 5.5×.

One more change made the threads pay off: results that are about to be
written in full are allocated without zero-filling
(`Tensor::uninitialized`). A zero-fill is a pass over memory that runs on a
single thread, before the parallel loop even starts.

## 6.5 A fast matrix product

A direct triple loop reads each element of B about `m` times. Fast
libraries (BLAS) organize the product around the memory hierarchy. The
design used here is the standard one, described by Goto and van de Geijn
(2008) and developed further in the BLIS framework (Van Zee and van de Geijn,
2015); see the references at the end of this chapter. `gemm.hpp` is a
simplified version of it:

1. **Cache blocking.** The inner dimension is cut into slices of 256 terms.
   A slice of B is **packed** into contiguous panels, 16 columns wide (2 SIMD
   vectors). The panels stay in cache while every row of C uses them.
2. **Register tiling.** The innermost code computes a 4 × 16 tile of C in 8
   SIMD registers. Each value of A is broadcast to a vector and multiplied
   with two vectors of B. Each load feeds several multiply-adds.
3. **Parallel row blocks.** Blocks of 96 rows of C are independent, so they
   go to `parallel_for`.

```cpp
template <std::size_t R, class T>
void gemm_tile(std::size_t kc, const T* ap, const T* bp, T* c, std::size_t ldc, std::size_t cols) {
    using V = simd<T>;
    std::array<std::array<V, 2>, R> acc;   // the tile, in registers
    ...
    for (std::size_t p = 0; p < kc; ++p) {
        const V b0(bp + p * 2 * W, element_aligned), b1(bp + p * 2 * W + W, element_aligned);
        template for (constexpr std::size_t r : iota_array<R>) {   // unrolled at compile time
            const V a(ap[r * kc + p]);
            acc[r][0] = fma(a, b0, acc[r][0]);
            acc[r][1] = fma(a, b1, acc[r][1]);
        }
    }
    ...
}
```

`template for` over a constant array unrolls the row loop. Each iteration is
a separate copy with a constant `r`, so the accumulators can live in
registers.

`fma` is called explicitly. In strict ISO mode (`-std=c++26`, not
`gnu++26`), GCC does not fuse `a * b + c` into one fused multiply-add
instruction on its own.

**Transposes are free.** `matmul(transpose(a), b)` becomes `MatMul<true,
false>` (chapter 2). The packing step then reads A in the other order, so no
transposed tensor is ever built.

**Results** (float, GFLOP/s):

| size | before this chapter | 1 thread | threads |
|---|---|---|---|
| 512³ | 58 | 104 | **184** |
| 1024³ | 37 | 117 | **305** |
| 2048³ | 27 | 110 | **433** |

One thread reaches about 110 GFLOP/s, which is close to one core's peak with
AVX2: 2 FMA units × 8 floats × 2 operations per cycle × about 3.5 GHz ≈ 112.
The old loop fell off at 1024 because its working set no longer fit in cache.
Blocking removes that cliff. At 2048³ the product is **16× faster** than
before.

The tests compare the kernel with a textbook triple loop on awkward shapes
(1×1×1, 17×33×15, k across a 256-term slice, and so on), on transposed
operands, and with threads on and off. Threads must not change a single bit
of the result.

## 6.6 What it means for training

| | before | after |
|---|---|---|
| MNIST, one epoch (chapter 5) | 3.7 s | **1.3 s** |
| travel time, whole example | 3.8 s | **1.8 s** |
| traffic state, whole example | 50 s | 51 s |

The traffic-state network is small: 20 inputs, 64 units, batches of 256. Each
product is about 256 × 20 × 64 multiply-adds, below the size where threads
pay off. That run is limited by fixed costs per operation, such as allocating
results and recording the tape, not by arithmetic. Faster kernels only help
work that is large enough to need them.

## 6.7 A compiler bug on Windows

The first SIMD build crashed as soon as vectors were used. The debugger
showed the instruction:

```
vmovaps %ymm0, -0x60(%rbp)     ; store a 32-byte vector to the stack
```

`vmovaps` requires a 32-byte-aligned address. GCC assumes the stack is 32-byte
aligned when AVX is on, but the 64-bit Windows ABI only guarantees 16 bytes.
This is GCC bug 54412, open for years on MinGW. The usual remedy tells the
assembler to use unaligned moves, which are just as fast on modern CPUs:

```cmake
target_compile_options(xinn INTERFACE -Wa,-muse-unaligned-vector-move)
```

Linux and macOS are not affected.

**What is new, and what is not.** The algorithm is not new. What is somewhat
unusual is how it is expressed: in portable standard C++26 (`std::simd`,
`fma`, `template for`, `std::execution`) instead of intrinsics or assembly.
That makes it a good teaching example and a demonstration of new language
features. It is not a new algorithm, and it gives no technical edge. The
~110 GFLOP/s per core is good because the textbook method works, not because
of anything new.

## 6.8 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| a hand-written thread pool, or OpenMP pragmas | `std::execution`: `schedule`, `bulk`, `sync_wait`, the parallel scheduler |
| compiler intrinsics (`_mm256_fmadd_ps`) | `std::simd` with `fma` |
| two copies of each op, scalar and vector | one generic op, overloads found by ADL |
| manual unrolling or macros | `template for` over a constant array |
| run-time checks for "can this be vectorized?" | a `consteval` test, as part of the type |

## 6.9 References

- K. Goto, R. A. van de Geijn. *Anatomy of high-performance matrix
  multiplication.* ACM Transactions on Mathematical Software 34(3), 2008.
  The packing, blocking and micro-kernel design used by `gemm.hpp`.
- F. G. Van Zee, R. A. van de Geijn. *BLIS: A framework for rapidly
  instantiating BLAS functionality.* ACM Transactions on Mathematical
  Software 41(3), 2015. The same design, organized around a small micro-kernel.
- ISO C++ proposals: P2300 (`std::execution`), P2079 (parallel scheduler),
  P1928 (`std::simd`), P1306 (expansion statements).

## 6.10 Exercises

1. Change the matmul tile to 6 × 16 or 4 × 24. Which gives the most GFLOP/s
   on your CPU? Why can a tile be too large?
2. Write a vectorized `exp` with a polynomial and `std::simd` (split into
   `2^k · e^r`). How much faster does `sigmoid(x * w + b)` get?
3. `SumLeading` (bias gradients) is still a single-threaded loop. Parallelize
   it with `parallel_for`. Is it worth it for a (256, 64) gradient?
4. Run `traffic_state` with batches of 2048 instead of 256. Does the time per
   epoch fall? Does accuracy change?
