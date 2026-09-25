# XiNN: A Mini Deep Learning Framework in C++26

This book builds a small deep learning framework, **XiNN**, from scratch.

The name has two halves. **NN** stands for "neural network". **Xi** starts
two Chinese words at once:

| Word | Meaning |
|---|---|
| 小 (*xiǎo*) | small: a mini framework, small enough to read in full |
| 新 (*xīn*) | new: built with the newest C++; also the author's family name, Xin |

So **XiNN** is a small, new neural-network framework. In code it is written
in lower case: namespace `xinn`, headers under `include/xinn/`.

XiNN was motivated by teaching the graduate course TR-GY 7353 *Data Driven
Mobility Modeling and Simulation* (Spring 2025), Department of Civil and Urban
Engineering, New York University Tandon School of Engineering. It was first
meant as part of the course material, but was not used in class: it requires
mastery of both modern C++ and the internals of deep learning, more than the
course could assume.

The goal of this book: **手搓 (*shǒu cuō*, "make by hand") a C++ deep learning
framework from scratch, using the latest modern C++ features**, to truly
understand how it works. The features include:

- **C++26:** static reflection (`^^T`, `[: r :]`, P2996), expansion statements
  (`template for`, P1306), contracts (`pre`, `contract_assert`, P2900), pack
  indexing (`Ts...[I]`, P2662), computed `static_assert` messages (P2741),
  `std::execution` senders and the parallel scheduler (P2300, P2079),
  `std::simd` (P1928), `std::submdspan`
- **C++23:** deducing `this`, `std::mdspan`, `std::expected`, `std::generator`,
  `std::print`, `std::byteswap`
- **C++20:** concepts, class-type template parameters (`Dense<{.bias = false}>`),
  `consteval`, `std::span`, ranges

The book has two goals:

1. Learn how a deep learning framework works inside: tensors, operations,
   layers, automatic differentiation, training, and evaluation.
2. Learn the newest C++: concepts, `mdspan`, deducing `this`, contracts, pack
   indexing, expansion statements, and static reflection.

XiNN works the way compile-time C++ frameworks do: the structure of a network
is known when the program is compiled, so the compiler can check it and
optimize it. Until C++20 this took heavy template metaprogramming, such as
type lists, SFINAE, and recursive templates. C++26 replaces most of that with
ordinary code that runs at compile time. Where it helps, a chapter shows the
old way next to the new one.

## How to read

Each chapter matches one stage of the code. Read the chapter, then read the
code it names, then run the tests. The code is small enough to read in full.

| Chapter | Topic | New C++ | Examples |
|---|---|---|---|
| 0 | Setup | toolchain, CMake presets | |
| 1 | Tensors | concepts, `mdspan`, `submdspan`, contracts, pack indexing | |
| 2 | Lazy operations | expression templates, fusion, compile-time rewrite rules, reflection | fusion benchmark |
| 3 | Automatic differentiation | gradients from expression types, compile-time pruning | spirals; traffic fundamental diagram |
| 4 | Modules | deducing `this`, reflection over members, config structs, `std::expected` | model summary, safetensors; a traffic model as a module |
| 5 | Training | `std::generator`, optimizer state laid out by reflection, `std::expected` | MNIST; traffic state (synthetic + PEMS08); travel time |
| 6 | CPU performance | `std::simd`, `std::execution` (senders, `bulk`, parallel scheduler) | matmul 16× faster; MNIST epoch 3.7 → 1.3 s |
| 7 | A minimal GPU backend | a `std::execution` scheduler for the GPU; OpenCL kernels generated at compile time from expression types; OpenCL loaded at run time | fused expression and matmul, CPU vs GPU |
| 8 | Convolutional networks | im2col convolution, pooling, nested parallelism | MNIST CNN (98.9%); incident detection from space–time maps |
| 9 | Recurrent networks | a run-time loop over a fixed step type; vector-Jacobian products | GRU speed forecasting on PEMS08 (15/30/60 min) |
| 10 | Attention and transformers | batched matmul, `permute<P...>`, layer norm; shared nodes with `std::move_only_function` | a character language model trained on this book; PEMS08 forecasting |
| 11 | Graph neural networks | CSR sparse matrices, `spmm`, graph convolution | forecasting on the CTM corridor (the road graph helps) and PEMS08 (it does not) |

Every training chapter pairs a standard example with one from traffic flow
theory and transportation. A case study reproduces the
car-following stability analysis of Herman et al. (1959) and learns the
car-following law from data. [XiNN Lab](xinn-lab.md) is an interactive
front end for the traffic examples.


## Conventions

Code lives in `include/xinn/`. Tests live in `tests/`. Data sets are fetched
with `python tools/prepare_data.py` into `data/`, which is not in the
repository.

## License

The XiNN source code is licensed under the BSD 3-Clause License (`LICENSE`).
The text of this book is licensed under CC BY 4.0 (see
[License of this book](LICENSE.md)).

## Credits

The design of XiNN is based on Li Wei's book *动手打造深度学习框架* and its
MetaNN framework. See [Credits](credits.md) for what XiNN takes from it and
what it does differently.
