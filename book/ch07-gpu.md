# 7. A Minimal GPU Backend

Code: `include/xinn/gpu.hpp`, and in `include/xinn/gpu/`: `opencl.hpp`,
`device.hpp`, `codegen.hpp`, `scheduler.hpp`. Tests: `tests/test_gpu.cpp`,
`tests/gpu_emulate/`. Benchmark: `examples/bench_gpu.cpp`.

This chapter runs XiNN expressions on a GPU. It covers forward evaluation
only: no gradients on the GPU, and only `float`. It is small on purpose. It
shows the three ideas a GPU backend is built on, and how C++26 expresses
them:

1. The GPU runs **kernels**: small programs that the driver compiles at run
   time from source text.
2. An expression type (chapter 2) can be turned into that source text **at
   compile time**, so a whole element-wise tree still becomes one fused
   kernel.
3. A GPU is another place to run work, so it gets a **scheduler** for
   `std::execution` (chapter 6).

**Where this was tested.** The book was written in a virtual machine with
no GPU. Its `OpenCL.dll` is present, but it finds no platform. Everything
that can be checked without a GPU was checked there (§7.9): the build, the
exact kernel text, the backend against a fake OpenCL driver, the kernels
compiled as C and run on the CPU, the scheduler, and the clean skip when
there is no device. The same programs were then run on the author's host
machine, which has an NVIDIA GeForce RTX 3060: every test passes there, and
§7.10 gives its timings.

## 7.1 Why a GPU

A CPU has a few fast cores. A GPU has thousands of simple ones, and memory
that delivers several hundred GB/s instead of a few tens. Chapter 6 showed
the two kinds of work in deep learning:

- **Element-wise operations** are limited by memory bandwidth. A GPU's
  memory is several times faster, so they are several times faster there.
- **Matrix products** are limited by arithmetic. A mid-range GPU is
  specified at over 10 TFLOP/s in `float`; one CPU core reaches about 0.1
  (§6.5).

There is a catch. The GPU has its own memory, at the end of a PCIe bus that
moves about 10 to 25 GB/s. A tensor must be copied there before a kernel can
read it, and the result copied back. For an element-wise operation that
reads each number once, the copies take longer than the work. A GPU pays
off when data **stays** on the device across many operations, or when the
work per byte is large (matmul). `bench_gpu` measures both cases, with and
without the copies.

## 7.2 The OpenCL model

XiNN uses **OpenCL**, the Khronos standard for running code on GPUs. Every
major GPU vendor ships it with its driver: NVIDIA (OpenCL 3.0), AMD and
Intel. CUDA is faster to write for on NVIDIA hardware, but it needs NVIDIA's
compiler at build time. OpenCL needs nothing at build time: kernels are
plain text, compiled by the driver when the program runs.

| Object | What it is |
|---|---|
| platform | one vendor's OpenCL implementation |
| device | one GPU (or CPU, or accelerator) |
| context | the device's memory space and compiled programs |
| command queue | where the host sends work; runs it in order |
| buffer | a block of device memory |
| program | OpenCL C source, compiled for the device |
| kernel | one `__kernel` function of a program, with its arguments |

A kernel is launched over an **NDRange**: a 1-, 2- or 3-D grid of *work
items*, each of which runs the kernel once with its own
`get_global_id(0)`. Work items are grouped into **work groups**. The items of
a group run on the same compute unit and share fast **local memory**, and
can wait for each other at a `barrier`.

## 7.3 Loading OpenCL at run time

XiNN needs no OpenCL headers or SDK to build. `opencl.hpp` declares the
types and constants it uses (their values are fixed by the specification),
and a table of function pointers:

```cpp
struct Api {
    cl_int (XINN_CL_CALL* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_mem (XINN_CL_CALL* CreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*);
    cl_int (XINN_CL_CALL* BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, ...);
    ...   // 21 functions in all
};
```

`cl::api()` opens the system's OpenCL library (the *ICD loader*:
`OpenCL.dll` on Windows, `libOpenCL.so.1` on Linux) the first time it is
called, and fills the table with `GetProcAddress` or `dlsym`. It returns
`const std::expected<Api, Error>&`: either the table or the reason there is
none. Every call in the backend reports failure the same way, as an `Error`
with a message and the OpenCL error code:

```
clGetPlatformIDs failed: CL_PLATFORM_NOT_FOUND_KHR (-1001)
```

That is exactly the message in the book's VM.

**A detail on Windows.** The header needs `LoadLibraryA` and
`GetProcAddress`, but including `<windows.h>` would define macros such as
`near`, `far`, `min` and `max` in every program that uses XiNN (the test
harness has a function called `check::near`). So the header declares the two
functions itself, with exactly the types `<windows.h>` uses. A program may
include both, in either order; this was checked.

## 7.4 Devices and device tensors

```cpp
auto dev = gpu::Device::open();          // std::expected<Device, Error>
if (!dev) { std::println("{}", dev.error().message); return; }
std::println("{}", dev->name());         // e.g. "NVIDIA GeForce RTX 3060"
```

`Device::open()` takes the first GPU of any platform; failing that, the
first OpenCL device of any kind; failing that, an `Error`. A `Device` holds
the context and the command queue. Copies share them, like copies of a
`Tensor` share a buffer.

`DeviceTensor<T, R>` is a shape plus a device buffer:

```cpp
auto dx = gpu::upload(*dev, x);          // host -> device: std::expected<DeviceTensor<float, 2>, Error>
auto back = dx->download();              // device -> host: std::expected<Matrix<float>, Error>
```

It is `TensorLike` (§1.8). Its `eval()` copies it back to the host, so a
device tensor can be an operand of any expression, and `gpu::run` (§7.7)
recognizes it and reads it where it is.

**A bug the tests caught.** The first `Buffer` built its state with
`std::make_shared<State>(State{dev, mem, bytes})`. That makes a temporary
`State` and copies it; the temporary's destructor then released the OpenCL
buffer that the copy still held. Nothing in the VM could run it on a GPU,
but the fake driver of §7.9 counts buffers, and reported zero live buffers
right after an upload. The fix constructs the state in place and deletes its
copy constructor, so the mistake cannot come back.

## 7.5 Kernels from types

Chapter 2 fuses an element-wise tree into one CPU loop: every reader is a
concrete type, and the compiler inlines them. A GPU runs code compiled by its
driver from text, so the same tree must become **text**. The expression type
already holds everything needed, so the text is made at compile time, by a
`consteval` function that walks the type:

```cpp
template <class X>
consteval std::string emit(KernelText& k) {       // returns the value of X at element i
    if constexpr (FusedNode<X>) {                  // an element-wise Expr
        using Op = typename X::op_type;
        static_assert(HasOpenCL<Op>, no_opencl_message<Op>());
        k.name += "_" + std::string(op_name<Op>());          // reflection, as in describe()
        std::string operands[X::arity];
        [&]<std::size_t... I>(std::index_sequence<I...>) {
            ((operands[I] = emit<typename X::template arg_type<I>>(k)), ...);
        }(std::make_index_sequence<X::arity>{});
        const std::string v = "v" + decimal(k.values++);
        k.body += "    const float " + v + " = " + substitute(opencl_code<Op>(), operands) + ";\n";
        return v;
    } else if constexpr (leaf_kind<X>() == Leaf::literal) {
        return float_literal(X::fill_value);        // Zeros -> 0.0f
    } else if constexpr (leaf_kind<X>() == Leaf::constant) {
        ...   // a `const float cK` argument
    } else {
        ...   // a `__global const float* aK, const uint nK` argument, read at i % nK
    }
}
```

The leaves become kernel arguments, like the readers of §2.4:

| Leaf | In the kernel |
|---|---|
| `Tensor`, `DeviceTensor`, `Param`, structural `Expr` | `__global const float* aK, const uint nK`, read as `aK[i % nK]` |
| `Constant` (a number in the expression) | `const float cK` |
| `Filled`, `Zeros`, `Ones` (value in the type) | a literal: `0.0f`, `1.0f`, `0x1.8p-1f` |

Broadcasting is the same one line as on the CPU: element `i` reads
element `i % nK` of an operand with `nK` elements. A structural operand
(such as a matmul inside the tree) is computed first, then read like a
tensor.

**Each op says what it is in OpenCL C**, as an expression over its operands
`$0`, `$1`. For the ops of `ops.hpp` this is a trait, `gpu::opencl_op`, so
`ops.hpp` does not change:

```cpp
template <> struct opencl_op<ops::Add>     { static constexpr std::string_view code = "$0 + $1"; };
template <> struct opencl_op<ops::Sigmoid> { static constexpr std::string_view code = "1.0f / (1.0f + exp(-$0))"; };
template <> struct opencl_op<ops::Relu>    { static constexpr std::string_view code = "fmax($0, 0.0f)"; };
```

A user op can bring its own code as a static member:

```cpp
struct Clamp01 {
    static constexpr std::string_view opencl = "clamp($0, 0.0f, 1.0f)";
    float operator()(float a) const { return std::clamp(a, 0.0f, 1.0f); }
};
```

A lambda has neither, and cannot be part of a GPU kernel. That is a compile
error, with a message built by reflection that names the operation
(the compile-fail test `gpu_op_without_code` checks it):

```
error: static assertion failed: xinn::gpu: operation `main()::<lambda(float)>` has no
OpenCL code. Give the op a `static constexpr std::string_view opencl`, or
specialize xinn::gpu::opencl_op for it.
```

Each node becomes one `const float` value, so an operand used twice (as in
`Square`, `$0 * $0`) is computed once. The kernel's name lists the ops in the
order the tree is walked, with names taken by reflection exactly as
`describe()` takes them (§2.8). Long names are cut and given a hash.

The result is stored as a static string that the program can use at run
time:

```cpp
template <class X>
inline constexpr std::string_view kernel_source =
    std::define_static_string(detail::kernel_source_of<X>());
```

For `sigmoid(x * w + b)`, with matrices `x`, `w` and a vector `b`, this is
the text, exactly as the test `fused_kernel_source_for_a_dense_layer`
checks it:

```c
__kernel void xinn_sigmoid_add_mul(__global float* out, const uint n,
    __global const float* a0, const uint n0,
    __global const float* a1, const uint n1,
    __global const float* a2, const uint n2)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v0 = a0[i % n0];
    const float v1 = a1[i % n1];
    const float v2 = v0 * v1;
    const float v3 = a2[i % n2];
    const float v4 = v2 + v3;
    const float v5 = 1.0f / (1.0f + exp(-v4));
    out[i] = v5;
}
```

One work item computes one element. The launch rounds the number of work
items up to a multiple of the work-group size (256), and the extra items
return at once (`if (i >= n)`).

**Literals are exact.** `std::format` cannot run at compile time, so
`float_literal` spells numbers itself: whole numbers in decimal (`2.0f`),
everything else as a hexadecimal float (`0.75f` is `0x1.8p-1f`, `0.1f` is
`0x1.99999ap-4f`). A hex float is read back bit for bit, so the kernel sees
the same `float` as the CPU. Numbers written in an expression (`x * 0.5f`)
are `Constant`s, known only at run time, so they are kernel arguments, and
changing the number does not make a new kernel.

**Compiled once per expression type.** The `Device` keeps the programs it
has built, keyed by their source text. The first `gpu::eval` of an
expression type compiles its kernel, which is why `bench_gpu` warms up
before it times anything; later ones reuse it. Two types with the same
text, such as the same formula over host tensors and over device tensors,
share one program. A build error returns an `Error` that carries the
compiler's log.

## 7.6 The matrix product

`matmul` gets one hand-written kernel, `gpu::matmul_source`: the classic
**tiled** product. Each work group computes a 16 × 16 tile of C. It walks the
inner dimension 16 terms at a time: the group loads a 16 × 16 tile of A and
one of B into local memory, waits at a barrier, and each work item adds the
16 products for its element from there:

```c
for (uint t = 0; t < K; t += TS) {
    As[lr][lc] = row < M && t + lc < K ? A_AT(row, t + lc) : 0.0f;
    Bs[lr][lc] = t + lr < K && col < N ? B_AT(t + lr, col) : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint p = 0; p < TS; ++p) acc = fma(As[lr][p], Bs[p][lc], acc);
    barrier(CLK_LOCAL_MEM_FENCE);
}
```

Each number read from device memory is used 16 times, instead of once. This
is the GPU form of the cache blocking of §6.5, with local memory in place of
the CPU cache. `TS`, `TRANS_A` and `TRANS_B` are set with `-D` options when
the program is built, so `matmul(transpose(a), b)` reads `a` transposed in
place, as on the CPU, and the four combinations are four cached programs.
Devices whose work groups cannot hold 256 items get 8 × 8 tiles.

This kernel is simple, not fast. On an RTX 3060 it reaches about 1 TFLOP/s
(§7.10), under a tenth of what the card can do. The reason is in the inner
loop: every `fma` needs two numbers from local memory. Local memory is fast,
but not as fast as the arithmetic units, so they wait.

**Register blocking.** `gpu::matmul_blocked_source` is the next step, the
one tuned libraries (cuBLAS, CLBlast) start from. Each work item computes a
4 × 4 block of C and keeps its 16 partial sums in registers. For each term
of the inner dimension it reads 4 numbers of A and 4 of B from local memory,
and does 16 multiply-adds with them: two reads per multiply-add become half
a read. A group of 16 × 16 items now covers a 64 × 64 tile of C, and walks
the inner dimension 16 terms at a time:

```c
for (uint p = 0; p < TSK; ++p) {
    for (uint w = 0; w < WPT; ++w) {
        a[w] = As[p][lr + w * RTS];
        b[w] = Bs[p][lc + w * RTS];
    }
    for (uint i = 0; i < WPT; ++i)
        for (uint j = 0; j < WPT; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
}
```

`WPT` is the work per item (4), `RTS` the group's side (16), `TS = RTS ·
WPT` the tile (64), `TSK` the step in K (16). Three details matter:

- **An item's elements are spread out.** Item (lr, lc) owns rows lr, lr +
  16, lr + 32, lr + 48, and the same for columns, not a 4 × 4 square.
  Neighbouring items then read neighbouring numbers of `Bs` and write
  neighbouring numbers of C, which the hardware combines into one wide
  access.
- **Loading is shared out.** A 64 × 16 tile of A has 1,024 numbers and the
  group 256 items, so each loads 4. Consecutive items load consecutive
  addresses of A (or of Aᵀ, when transposed), again so that loads combine.
- **The local arrays are padded**, `As[TSK][TS + 1]`. Without the extra
  column, storing a column of the tile would put 16 numbers in the same
  memory bank of local memory, and they would be stored one at a time.

**Larger blocks, wider loads.** `WPT` and `VEC` are build options too, so
one source gives four blocked kernels:

- **`WPT=8`**: 8 × 8 elements per item, 64 sums in registers, a 128 × 128
  tile per group. Each number read from local memory now serves 8
  multiply-adds, not 4.
- **`VEC=1`**: tiles come from device memory four floats at a time, with
  `vload4`, along the direction in which each operand is stored: along K
  for A, along M for Aᵀ, along N for B, along K for Bᵀ. At an edge,
  `xinn_load4` loads only what is inside the matrix and fills the rest with
  zeros. (The zeros matter less than the bounds: a zero-filled tile of B
  would hide stray numbers from A, but reading past the end of a buffer
  can fault, and a stray NaN times zero is still NaN.)

`gpu::MatmulKernel` names them all: `tiled`, `blocked_4x4`,
`blocked_4x4_float4`, `blocked_8x8`, `blocked_8x8_float4`, and `automatic`,
the default, which takes what was fastest on the RTX 3060 (§7.10): 8 × 8
for results of 512 × 512 and more, 4 × 4 below, where 128 × 128 tiles would
be too few to keep the GPU busy. `gpu::matmul(dev, a, b, kind)` chooses
explicitly. All are checked the same ways (§7.9), including as C, for all
four transpose combinations and for sizes that are not multiples of the
tile or of 4. On the RTX 3060 the 8 × 8 kernel reaches 4.6 TFLOP/s, the
4 × 4 one 2.7, the tiled one 1.0 (§7.10).

## 7.7 What runs where

`gpu.hpp` has two entry points:

```cpp
auto y = gpu::run(dev, x);    // std::expected<DeviceTensor<float, R>, Error>: the result stays on the device
auto z = gpu::eval(dev, x);   // std::expected<Tensor<float, R>, Error>: and is copied back
```

`run` decides by the type of `x`, at compile time:

| `x` | on the GPU |
|---|---|
| `Tensor` | copied to the device |
| `DeviceTensor` | used where it is |
| element-wise tree | one fused kernel (§7.5); its leaves are `run` first |
| `matmul(a, b)`, with any transposes | the register-blocked kernel (§7.6); `a` and `b` are `run` first |
| anything else (`sum`, a lone `transpose`, ...) | evaluated on the CPU, then copied |

Because leaves are `run` by the same rules, `gpu::eval(dev, relu(matmul(x,
w) + b))` runs the matmul kernel, then one fused kernel for `+ b` and
`relu`. With `x`, `w`, `b` already on the device, nothing crosses the bus
until the result is downloaded:

```cpp
auto dx = gpu::upload(dev, x), dw = gpu::upload(dev, w);
auto db = gpu::upload(dev, b);
auto h = gpu::run(dev, relu(matmul(*dx, *dw) + *db));
```

`gpu::matmul(dev, a, b)` is `gpu::eval(dev, matmul(a, b))`.

## 7.8 A scheduler for `std::execution`

Chapter 6 ran loops with `schedule(get_parallel_scheduler()) | bulk(...)`.
A scheduler says *where* work runs. For a GPU, a useful answer is: on a
thread that owns the device's command queue, so that GPU work is submitted
in order and the program's other threads stay free.

`gpu::Queue` is such a thread, with a list of waiting operations.
`queue.scheduler()` gives a scheduler for it:

```cpp
gpu::Queue q(*dev);
auto work = schedule(q.scheduler())
          | then([&] { return gpu::eval(*dev, sigmoid(x * w + b)); });   // runs on q's thread
auto [y] = sync_wait(std::move(work)).value();                        // std::expected<Matrix<float>, Error>
```

The whole scheduler is three small types, the minimum the concept asks for:

```cpp
class Queue::Scheduler {
public:
    using scheduler_concept = ex::scheduler_t;
    Sender schedule() const noexcept { return {queue_}; }
    bool operator==(const Scheduler&) const noexcept = default;
    ...
};

struct Sender {                                   // what schedule() returns
    using sender_concept = ex::sender_t;
    using completion_signatures = ex::completion_signatures<ex::set_value_t()>;
    template <class Receiver>
    Operation<Receiver> connect(Receiver r) const { return Operation<Receiver>(queue, std::move(r)); }
    Env get_env() const noexcept { return {queue}; }   // "I complete on this scheduler"
    Queue* queue;
};

template <class Receiver>
struct Operation : Task {                         // what connect() returns
    using operation_state_concept = ex::operation_state_t;
    void start() & noexcept { queue->push(this); }    // run() later calls set_value(receiver)
    ...
};
```

`start()` appends the operation to the queue's list; the queue's thread
takes it off and calls `set_value` on the receiver, so everything chained
after `schedule` runs on that thread. The list is intrusive (each operation
is its own list node), so scheduling allocates nothing, and operations are
not movable, because the list points at them. The tests check the concepts
with `static_assert(ex::scheduler<gpu::Queue::Scheduler>)`, that the work
runs on the queue's thread, and in the order it was started.

`gpu::async_eval(sched, x)` wraps the pattern above in one sender.

**What it does not do.** This is the smallest honest scheduler, not a full
GPU integration. `bulk` is not customized: on this scheduler, `bulk(par, n,
f)` runs `f(0)` … `f(n − 1)` one after another on the queue's thread, not as
a kernel. A real one (such as NVIDIA's `nvexec` for CUDA) turns `bulk` into a
kernel launch and completes when the GPU does, without a host thread
waiting. Stop requests are not supported either. The queue does not need a
device, so all of this is tested on the VM.

## 7.9 Testing without a GPU

A backend whose kernels never run is easy to get wrong. Four layers of
tests check everything short of a real GPU:

1. **The exact kernel text.** `test_gpu` compares the generated source,
   character for character, for several expressions: plain tensors,
   constants and `Filled` literals, a structural operand, a user op, a long
   name. Kernel names are also checked with `static_assert`.
2. **A fake OpenCL driver.** `Device::open(const cl::Api&)` accepts any
   function table. The test passes one whose "buffers" are host memory and
   which records every call: kernel arguments, launch sizes, build options,
   programs built and objects released. It checks that arguments arrive in
   the order the source names them, that sizes round up correctly, that a
   second evaluation reuses the program, that the matmul flags and sizes are
   right, that build errors carry the log, and that everything is released.
   It found the bug of §7.4.
3. **The kernels, compiled as C and run on the CPU.** The OpenCL C used here
   is also C99. `tests/gpu_emulate/prelude.h` defines what is missing
   (`__kernel`, `__global`, `uint`, `get_global_id`, `barrier`, ...), and a
   launcher calls the kernel once per work item. For the matmul kernel,
   the work items of a group are threads that meet at `barrier()` (a
   `pthread_barrier`), and `__local` arrays become `static`, shared by the
   running group. The test `gpu_kernels_as_c` writes the generated kernels
   with inputs and CPU results, compiles them with `-Wall -Wextra`, runs them,
   and compares: five fused kernels and all four transpose combinations of
   matmul, on awkward sizes (37 × 70 × 45). Deliberately broken kernels
   fail it.
4. **On a device.** The remaining tests compare `gpu::eval` with the CPU on
   fused expressions (with broadcasting, rank-0 tensors, constants),
   matmul on awkward shapes and with transposes, a layer kept on the device,
   and the scheduler driving a device. Without a device they print
   `skipped: no OpenCL platform` and pass.

| | this VM (no GPU) | the host (RTX 3060) |
|---|---|---|
| builds without warnings, no OpenCL SDK | verified | |
| exact generated kernel text | verified | verified |
| argument order, sizes, cache, errors, release (fake driver) | verified | verified |
| kernels compile as C and match the CPU (emulated) | verified | |
| scheduler concepts, thread, order | verified | verified |
| no platform: clear error, tests skip, `bench_gpu` exits 0 | verified | |
| the driver's OpenCL compiler accepts the kernels | | verified |
| results on the GPU match the CPU | | verified |
| speed | | measured (§7.10) |
| blocked matmul kernels, 4 × 4 and 8 × 8, with and without `float4` (§7.6) | verified (emulated, fake driver) | verified, and measured |

## 7.10 Results on the host

The author's host machine has an NVIDIA GeForce RTX 3060, whose driver
includes OpenCL 3.0. To produce the numbers (Windows, in a shell where the
compiler is on the path, as in chapter 0):

```
cmake --preset release
cmake --build --preset release
ctest --preset release -R gpu --output-on-failure
build\release\tests\test_gpu
build\release\examples\bench_gpu --source
```

`test_gpu` prints the device it found. None of its lines should say
`skipped`. `bench_gpu` prints each size's time on the CPU, on the GPU
including both copies (*total*), and on the GPU with the operands already on
the device and the result left there (*kernel*; it still includes
allocating the output buffer), plus the largest difference from the CPU
result.

**Results on the host.** Windows 11, an NVIDIA GeForce RTX 3060 (OpenCL
3.0 through the CUDA driver), and a CPU with 32 hardware threads and AVX2.
The programs were built in the VM for any AVX2 CPU (`-march=x86-64-v3`),
copied over, and run there. The host has four times the VM's threads, so its
CPU numbers are much higher than chapter 6's.

`test_gpu`: 21 tests, 0 failed checks, none skipped. CPU times vary by
about 10% from run to run; the two tables are from separate runs of
`bench_gpu`.

| `sigmoid(x * w + b) * 0.5` | CPU | GPU total | GPU kernel | max \|CPU − GPU\| |
|---|---|---|---|---|
| 512 × 512 | 0.30 ms | 0.76 ms | 0.04 ms | 8.9e-8 |
| 1024 × 1024 | 0.82 ms | 2.44 ms | 0.36 ms | 8.9e-8 |
| 2048 × 2048 | 2.35 ms | 8.77 ms | 0.92 ms | 8.9e-8 |
| 4096 × 4096 | 8.48 ms | 29.63 ms | 2.85 ms | 8.9e-8 |

Matmul, in GFLOP/s. *GPU total* includes the copies and used the 4 × 4
kernel, the default at the time; the kernel columns have the operands
already on the device.

| size | CPU | GPU total | tiled | 4 × 4 | 4 × 4, `float4` | 8 × 8 | 8 × 8, `float4` |
|---|---|---|---|---|---|---|---|
| 256³ | 152 | 220 | 491 | **603** | 534 | 392 | 347 |
| 512³ | 216 | 229 | 458 | 612 | 547 | **641** | 639 |
| 1024³ | 383 | 668 | 821 | 1,760 | 1,613 | **2,298** | 2,056 |
| 2048³ | 709 | 1,240 | 917 | 2,438 | 2,285 | **4,085** | 3,760 |
| 4096³ | 945 | 1,607 | 1,002 | 2,727 | 2,517 | **4,612** | 4,364 |

The largest difference from the CPU, over all kernels, was 0 at 256³ and
rose to 9.8e-4 at 4096³.

What they show:

- **The kernels are right.** The fused kernel differs from the CPU by one
  rounding step (8.9e-8, the GPU's `exp` is not the CPU's). Matmul differs
  more as `n` grows, because the CPU and the GPU sum the products in a
  different order; 1e-3 on sums of 4096 terms is normal for `float`. All
  GPU kernels have the same largest difference: each adds an element's
  products in the same order, k = 0, 1, 2, ..., so they give the same
  numbers.
- **Element-wise: the kernel wins, the copies lose.** With the data already
  on the device, the fused kernel is 3 to 8 times faster than 32 CPU
  threads. With the copies, the GPU is about 3 times *slower*: three
  matrices go over the bus and one comes back, for one pass of arithmetic.
  This is the case for keeping tensors on the device (§7.7).
- **The tiled kernel ties with the CPU.** It reaches about 1.0 TFLOP/s,
  some 8% of the RTX 3060's rated 12.7, and the host's CPU, with the kernel
  of chapter 6 on 32 threads, reaches about 1.0 too. Two reads of local
  memory per multiply-add are its limit (§7.6). A first draft of this
  chapter predicted a clear GPU win from 1024³; with this kernel, it did
  not come.
- **Register blocking wins, and more of it wins more.** 4 × 4 elements per
  work item make the kernel 2.7 times faster at 4096³; 8 × 8 make it 4.6
  times faster: 4.6 TFLOP/s, 36% of the card's rating and about 5 times the
  CPU. The memory and the arithmetic units were the same all along; only
  the reuse of each number read changed. That is the lesson of §6.5 again,
  on a different machine. With blocking, the GPU wins from 1024³ even
  with the copies, because the work grows as `n³` and the copies as `n²`.
- **Wider loads did not help, and we do not know why.** `float4` made both
  blocked kernels up to 11% *slower*, never faster. Part of the reason it
  did not *help* is clear: 32 neighbouring items reading 32 neighbouring
  floats is already one wide access to device memory, so the scalar loads
  had nothing to gain. Why it *hurt* is not clear. It is not bank conflicts
  in local memory, a first guess. Counting the banks for one warp shows at
  most 2-way conflicts on the stores into `As` for both versions, scalar
  and `float4` (each row of `As` is 65 or 129 floats, 1 more than a
  multiple of the 32 banks), and those stores are only about 6% of the
  kernel's local-memory traffic: per tile, an 8 × 8 item stores 16 numbers
  and reads 256. The branch in `xinn_load4`, more registers per item, or how
  the driver's compiler schedules the loads are all possible; telling them
  apart takes a profiler, such as NVIDIA Nsight Compute. The lesson stands
  either way: a standard optimization of tuned libraries paid nothing here.
  Measure; do not assume, and do not explain a result you have not
  checked.
- **Big blocks need big matrices.** At 256³ the 8 × 8 kernel is the
  *slowest* blocked one: a 256 × 256 result is only 4 groups of 128 × 128,
  for the card's 28 multiprocessors. The 4 × 4 kernel makes 16 groups and
  wins. Hence `automatic`, which switches at 512 × 512.

Tuned libraries go further, with double-buffered tiles, conflict-free
layouts and kernels tuned per GPU, and reach well over half of the rating.
Exercise 1 takes the next steps.

Run it on your own machine to see where your CPU and GPU stand.

On Linux, install the NVIDIA driver's OpenCL ICD and the loader
(`ocl-icd-libopencl1`), check with `clinfo`, and configure without the
Windows presets: `cmake -B build -DCMAKE_BUILD_TYPE=Release
-DCMAKE_CXX_COMPILER=g++-16`.

## 7.11 Old C++ and new

| Older way | C++26 way in XiNN |
|---|---|
| link against an OpenCL SDK, `#include <CL/cl.h>` | declare 21 functions, load them at run time; failures as `std::expected` |
| kernels written by hand, one per fused combination | kernel text generated by `consteval` code from the expression type |
| a string-building "JIT" at run time, on every call | text made once at compile time, stored with `define_static_string` |
| kernel names and op names typed as strings | names from reflection, as in `describe()` |
| "unsupported op" found when the driver fails | a `static_assert` whose message names the op |
| a hand-written task queue with callbacks or futures | a `std::execution` scheduler: `schedule(q) \| then(...)`, `sync_wait` |

## 7.12 Exercises

1. The 8 × 8 kernel reaches 36% of the RTX 3060's rating. Take it further,
   measuring each step with `bench_gpu`, and profiling (for example with
   Nsight Compute) before you explain a result:
   (a) **double buffering**: load the next tiles into a second pair of local
   arrays while computing on the current ones, with one barrier per step
   instead of two;
   (b) **wider local reads**: the inner loop reads 16 numbers from local
   memory per step, 8 of A and 8 of B, and that is where the traffic is.
   Store the tiles so that each item's 8 values are contiguous, and read
   them as two `float4`s each. (Its elements are now RTS apart so that
   writes to C combine; what does the new layout cost there?);
   (c) find out why `float4` loads from device memory made the kernel
   slower (§7.10).
2. `gpu::run` allocates a new output buffer on every call. Add a small pool
   to `Device`, keyed by size. How much does *kernel* time fall for 1024 ×
   1024?
3. The fused kernel reads every leaf at `i % nK`, even at full size, where
   `nK == n`. Generate `aK[i]` for leaves known to be full size, deciding at
   run time between two cached kernels. Is the modulo measurable?
4. Give `sum(x)` a GPU kernel: each work group reduces its part in local
   memory, then a second pass adds the groups' results.
5. Customize `bulk` for `gpu::Queue::Scheduler` so that `schedule(q) |
   bulk(par, n, f)` becomes a kernel launch. What must `f` be, for this to
   be possible?
