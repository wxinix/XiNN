# XiNN

**A small deep learning framework in C++26, and a book that explains it.**

XiNN (小 *xiǎo*, small + 新 *xīn*, new + NN) is a header-only framework for
learning how deep learning works inside, written in the newest C++: static
reflection, contracts, pack indexing, expansion statements, `std::execution`
and `std::simd`. It comes with a step-by-step book and examples from traffic
flow theory.

XiNN was motivated by teaching the graduate course **TR-GY 7353 Data Driven
Mobility Modeling and Simulation** (Spring 2025) in the Department of Civil and
Urban Engineering at the **New York University Tandon School of
Engineering**. It was first meant as part of the course material, but was not
used in class: it requires mastery of both modern C++ and the internals of
deep learning, more than the course could assume. It is published here for
readers who have both, or want to build them.

The goal: **手搓 (*shǒu cuō*, "make by hand") a C++ deep learning framework
from scratch, using the latest modern C++ features**, to truly understand how
it works. The features include:

- **C++26:** static reflection (`^^T`, `[: r :]`, P2996), expansion statements
  (`template for`, P1306), contracts (`pre`, `contract_assert`, P2900), pack
  indexing (`Ts...[I]`, P2662), computed `static_assert` messages (P2741),
  `std::execution` senders and the parallel scheduler (P2300, P2079),
  `std::simd` (P1928), `std::submdspan`
- **C++23:** deducing `this`, `std::mdspan`, `std::expected`, `std::generator`,
  `std::print`, `std::byteswap`
- **C++20:** concepts, class-type template parameters (`Dense<{.bias = false}>`),
  `consteval`, `std::span`, ranges

Tensors, lazy expressions, automatic differentiation, optimizers and the
matrix kernel are all written here; the only dependency of the framework
itself is the C++ standard library (with stdexec standing in for
`std::execution` until GCC ships it).

![XiNN Lab: car-following stability](book/images/lab_car_following_C0.75.png)

```cpp
struct DigitNet : Module {                       // parameters found by reflection:
    Dense<{.activation = Activation::relu}> fc1, fc2;   // no registration, no macros
    Dense<> out;
    explicit DigitNet(Rng& rng) : fc1(784, 256, rng), fc2(256, 128, rng), out(128, 10, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};

DigitNet net(rng);
Adam opt(net, {.lr = 1e-3});                     // optimizer state laid out from the model's type
for (auto [xb, yb] : batches(images, labels, 128, rng)) {   // std::generator
    opt.zero_grad();
    backward(softmax_cross_entropy(net(xb), yb));             // gradients from the expression type
    opt.step();
}
```

## Highlights

**Fast on the CPU, in standard C++.** The classic cache-blocked,
register-tiled matrix product of Goto and van de Geijn (2008), the design
behind BLIS, OpenBLAS and MKL, written in portable standard C++ with
`std::simd`, `fma` and C++26 `template for`, and run in parallel on the C++26
parallel scheduler (`std::execution`):

| float matmul | textbook loop | XiNN, 1 thread | XiNN, all threads |
|---|---|---|---|
| 1024³ | 37 GFLOP/s | 117 GFLOP/s | **305 GFLOP/s** |
| 2048³ | 27 GFLOP/s | 110 GFLOP/s | **433 GFLOP/s (16×)** |

One thread reaches about 110 GFLOP/s, close to what one core can do with AVX2
(2 FMA units × 8 floats × 2 operations per cycle at about 3.5 GHz). No BLAS
library and no intrinsics are used.

**What is new, and what is not.** The algorithm is not new. What is somewhat
unusual is how it is expressed: in portable standard C++26 (`std::simd`,
`fma`, `template for`, `std::execution`) instead of intrinsics or assembly.
That makes it a good teaching example and a demonstration of new language
features. It is not a new algorithm, and it gives no technical edge. The
~110 GFLOP/s per core is good because the textbook method works, not because
of anything new.

> K. Goto, R. A. van de Geijn. Anatomy of high-performance matrix
> multiplication. *ACM Transactions on Mathematical Software* 34(3), 2008.
>
> F. G. Van Zee, R. A. van de Geijn. BLIS: A framework for rapidly
> instantiating BLAS functionality. *ACM TOMS* 41(3), 2015.

**Fused, vectorized element-wise math.** `sigmoid(x * w + b) * 0.5` compiles
to one loop, with no temporary tensors (tested by counting allocations). The
loop runs on SIMD vectors when every op in the tree supports it, decided at
compile time, and in parallel chunks: up to **6.9× faster** than the scalar
loop on one thread.

**Automatic differentiation from types.** Each operation declares a local
gradient rule. `backward()` records a tape shaped like the expression type,
and **prunes parameter-free subtrees at compile time**. Every rule is checked
against finite differences. A missing rule is a compile error that names the
operation, with a message built by reflection.

**Compile-time rewrites.** `x + zeros` is `x`. `transpose(transpose(x))` is
`x`. `matmul(transpose(a), b)` becomes one kernel that reads `a` transposed,
so the gradients of a linear layer never copy a transpose.

**Real results.**

| task | result |
|---|---|
| MNIST, 784-256-128-10 MLP | **98.0%** test accuracy, **1.3 s per epoch** on the CPU |
| MNIST, small CNN (2 conv layers) | **98.9%** test accuracy after 3 epochs, 13 s per epoch |
| two spirals | 99.8% |
| traffic state 15–60 min ahead, real Caltrans detectors (PEMS08) | beats persistence; class weights give the best macro-F1 |
| corridor travel time in congestion | **0.75 min** error, vs. 1.20 for the instantaneous estimate |
| freeway incident detection from space–time maps (CNN) | detects incidents about **30% sooner** than a speed rule, at the same false-alarm rate |
| speed 15/30/60 min ahead, 170 PEMS08 detectors (MLP, GRU, transformer) | MAE **1.90 / 2.45 / 3.19 km/h** vs. 2.00 / 2.60 / 3.38 for persistence; the GRU and the transformer are no better than an MLP on a 1-hour window |
| speed forecasting with a graph network (chapter 11) | on the simulated corridor the road graph cuts congested-speed errors by 3–6 km/h; on PEMS08 it does no better than a random graph |
| character language model trained on this book (transformer) | **2.53 bits/char** on held-out text, vs. 4.93 for character frequencies |
| car-following stability (Herman et al. 1959) | learned drivers reproduce every stability regime in closed loop |

Measured on the machine the book was written on: a VM with 8 hardware
threads, AVX2/FMA, GCC 16.1, release build. Your numbers will differ.

**Modern C++, used where it helps.**

| | |
|---|---|
| static reflection (P2996) | modules, parameter names, test discovery, op names in formulas and errors |
| `template for` (P1306) | loops over tuples and reflected members; unrolled GEMM tiles |
| contracts (P2900) | shape and write rules as `pre` / `contract_assert` |
| pack indexing (P2662) | `args...[I]` instead of recursive templates |
| `std::execution` (P2300, P2079) | parallel loops on the system scheduler (via stdexec until GCC ships it) |
| `std::simd` (P1928) | vectorized fused loops and the matmul kernel |
| class-type template parameters | `Dense<{.bias = false}>` |
| `std::expected`, `std::generator`, `std::mdspan` | loaders, mini-batches, tensor views |

**Kernels from types, on the GPU.** `gpu::eval(dev, sigmoid(matmul(x, w) + b))`
runs a register-blocked matmul kernel and one fused OpenCL kernel whose source is
generated at compile time from the expression's type, with kernel names from
reflection. OpenCL is loaded at run time, so building needs no SDK. Tested
without a GPU (emulated kernels, a fake driver) and on an RTX 3060, where the
results match the CPU's and matmul reaches 4.6 TFLOP/s.

**Traffic flow theory.** Examples fit the fundamental diagram (Greenshields,
Drake), simulate a freeway with the Cell Transmission Model, predict traffic
states and travel times, and reproduce the car-following stability analysis
of Herman, Montroll, Potts and Rothery (1959).

**XiNN Lab.** An interactive front end built with Dear ImGui and ImPlot:
drag C across 1/e, ½ and π/2 and watch a line of cars go unstable, or train a
driver from data and let it drive.

## Why this repository?

To explore how much of a deep learning framework modern C++ can express
cleanly, transparently and efficiently, and to let readers understand the
entire implementation. See [Questions and Answers](book/faq.md).

## The book

**Read it online: <https://wxinix.github.io/XiNN/>**

[`book/`](book/README.md) builds XiNN chapter by chapter:
tensors → lazy operations → automatic differentiation → modules → training →
CPU performance → a minimal GPU backend → convolutional networks → recurrent
networks → attention and transformers → graph neural networks, plus a
car-following case study, XiNN Lab and a page of questions and answers.

## Building

XiNN needs **GCC 16** (for reflection and contracts) and CMake. On Windows:

```
winget install BrechtSanders.WinLibs.POSIX.UCRT
cmake --preset release
cmake --build --preset release
ctest --preset release
python tools/prepare_data.py          # MNIST and PEMS08, for the examples
build/release/examples/mnist_mlp
```

XiNN Lab: `cmake --preset gui && cmake --build --preset gui --target xinn_lab`.
See [chapter 0](book/ch00-setup.md) for details, including CLion.

## License

Code: [BSD 3-Clause](LICENSE). Book text: [CC BY 4.0](book/LICENSE.md).
Copyright © 2026 Wuping Xin.

The design owes much to Li Wei's book *动手打造深度学习框架* and its MetaNN
framework; see [Credits](book/credits.md).
