# Questions and Answers

## What is the point of this repository?

To explore **how much of a deep learning framework modern C++ can express
cleanly, transparently and efficiently**, and to let readers understand the
entire implementation. It combines two kinds of learning: how deep learning
frameworks work inside, and how to use modern C++.

It brings together three things:

- **A working explanation.** Tensors, automatic differentiation, optimizers
  and training are implemented alongside the book that explains them, and so
  are the main architecture families: convolutional, recurrent, transformer
  and graph networks.
- **A language-design demonstration.** Reflection discovers parameters,
  expression types encode computations, and compile-time rules simplify them.
- **An engineering experiment.** SIMD, fusion and parallelism show that the
  readable implementation can also reach useful CPU performance, and a
  minimal GPU backend generates its kernels from the same expression types.

The traffic examples give it a distinctive application. Readers connect
neural-network machinery to physical models and simulation, from the
fundamental diagram to car-following stability, beyond the standard
classification exercises.

Its main contribution is **making sophisticated implementation techniques
accessible in one coherent, inspectable system**. The performance results
matter because they show the approach can also be practical. Outperforming
established frameworks is not necessary for the repository to have value.

## Why C++, and not Python?

Python is where deep learning is *used*; the frameworks themselves, PyTorch
and TensorFlow, are C++ underneath, with Python on top. XiNN is about that
lower layer, and C++ lets it be written, read and understood in one language:

- **Nothing is hidden.** In Python, a tensor operation calls into compiled
  code you cannot step into. In XiNN, every operation from `a + b` to the
  matrix kernel is C++ you can read and debug.
- **The compiler works for you.** Tensor ranks, the shapes of layers, which
  parameters need gradients, and simplifications such as `x + zeros -> x` are
  decided while compiling. Python finds such errors, and makes such
  decisions, at run time.
- **Speed without a second language.** Fused loops, SIMD and threads are in
  the same code as the model, and even the GPU kernels are generated from the
  C++ expression types (chapter 7). In Python, speed means writing C, C++ or
  CUDA extensions anyway.
- **Deployment.** A trained XiNN model is a small native program, with no
  interpreter and no runtime to install: useful inside traffic simulators and
  other C++ applications.
- **The language itself is a subject.** C++26 brings reflection, contracts
  and `std::execution`. Building a framework is a demanding way to learn
  them.

The costs are real too: C++ is harder to learn, compile times are long (about
20–40 seconds for one of XiNN's test programs), error messages can be long,
and there is no interactive notebook. For experimenting with models, use
Python. For understanding what runs underneath, C++ is the better teacher.

## Who is this book for?

Readers who know C++ well (templates, lambdas, RAII) and want to learn how
deep learning frameworks work, or who know deep learning and want to see
modern C++ applied to a real system. Transportation students and engineers
will recognize the traffic examples. The book explains the deep learning it
uses; it does not replace a textbook on either subject.

## How big is it?

About 6,000 lines of library code in 35 headers, plus 3,300 lines of tests,
3,600 lines of examples and 550 lines for XiNN Lab. The book has about
33,000 words. The whole framework can be read in a week or two.

## Why header-only?

Almost everything is a template: tensors are templated on element type and
rank, and every expression has its own type. Templates must be visible where
they are used, so there is nothing left to compile into a library. The price
is compile time.

## Why expression templates instead of a run-time graph, like PyTorch?

Because the network's structure is known when the program is compiled, the
compiler can check it and optimize it: shapes of ranks, fusion of
element-wise operations, rewrite rules, and pruning of gradient work. The
limits are the other side of the same coin:

- Every different formula is a different type, so compile times grow.
- Structures whose size is only known at run time, such as a sequence of
  unknown length, cannot become one type. The recurrent network chapter
  (chapter 9) handles them with a loop that reuses one step's type.
- An expression is a *tree*: a value used twice is stored, evaluated and
  differentiated twice. For a GRU gate this is a small waste, avoided with a
  dedicated op (chapter 9). For a stack of transformer blocks, where each
  residual connection reuses its input, it would grow exponentially, so
  chapter 10 adds `share()`: a node evaluated once whose gradient is
  collected once, turning the tree into a graph where it matters.

PyTorch makes the opposite choice: fully dynamic, with optimization done by a
separate compiler (TorchScript, `torch.compile`).

## Does it implement transformers, CNNs, and other architectures?

| architecture | building blocks | chapter | examples |
|---|---|---|---|
| **MLP** (dense networks) | `Dense`, ReLU/tanh/sigmoid, `Sequential` | 3–5 | spirals, MNIST (98.0%), traffic state, travel time |
| **CNN** (convolutional networks) | `Conv2D` (im2col), `MaxPool2D`, `Flatten`, `reshape` | 8 | MNIST (98.9%), incident detection from space–time maps |
| **RNN** (recurrent networks) | `GRU`, backpropagation through time as a loop of per-step vector-Jacobian products | 9 | PEMS08 speed forecasting (no better than an MLP on a 1-hour window) |
| **Transformer** | multi-head attention, layer norm, embeddings, positional encoding, causal masks, pre-LN blocks, shared nodes | 10 | a character language model trained on this book (2.53 bits/char); PEMS08 forecasting |
| **GNN** (graph neural networks) | CSR sparse matrices, `spmm`, `GraphConv` (Kipf–Welling), optional self weight | 11 | forecasting on the CTM corridor (the road graph helps) and PEMS08 (it does not) |
| **GPU evaluation** | fused OpenCL kernels generated from expression types, tiled and register-blocked matmul, a `std::execution` scheduler | 7 | CPU vs GPU on an RTX 3060: fused kernels 3–15× faster than 32 CPU threads once data is on the device; matmul 4.6 TFLOP/s with 8 × 8 register blocking, about 5× the CPU |
| **Physical models as modules** | scalar `Param`s in any formula | 3, 4, case study | fundamental diagram (Greenshields, Drake), car-following law |

Every building block has a gradient rule checked against finite
differences, so the pieces can be combined freely into new models.

What XiNN does not attempt: very large models, a GPU training path,
distributed training, or a model zoo. It implements each family at the size
that teaches how it works and trains on one CPU.

## Which optimizers are implemented? Adam, or something else?

For training (chapter 5, `include/xinn/optim.hpp`):

| optimizer | options | update |
|---|---|---|
| **SGD** | `lr`, `momentum`, `weight_decay` | `v ← μv + g (+ wd·p)`, `p ← p − lr·v` |
| **Adam** | `lr`, `beta1`, `beta2`, `eps` | running means of `g` and `g²`, with bias correction |
| **AdamW** | Adam with `weight_decay` | Adam plus decoupled decay: `p ← p − lr·wd·p` |

Around them:

- a **cosine learning-rate schedule**, `cosine_lr(step, total, lr0, lr_min)`;
- **class weights** in the cross-entropy loss, for rare classes;
- **He and Glorot initialization**;
- **mini-batches** from a `std::generator`.

Both optimizers are built from a model, `Adam opt(net, {.lr = 1e-3})`, and lay
out their state from the model's *type*: one or two tensors per parameter,
found by reflection, with no maps and no type erasure. Any type with
`step()`, `zero_grad()` and a learning rate satisfies the `Optimizer` concept
and can be used instead.

Not implemented: RMSprop, Adagrad, LAMB, gradient clipping, learning-rate
warm-up and mixed precision. Each is a few lines on top of the same pattern,
and makes a good exercise.

If "optimization" means *making it fast*, see the next question and chapter
6: fused element-wise loops, compile-time rewrite rules, SIMD, a blocked
parallel matrix kernel, and pruning of gradient work at compile time.

## How is the computation graph implemented?

There is no graph object. **The graph is the C++ type of an expression.**

```cpp
auto y = sigmoid(matmul(x, W) + b);
// decltype(y) is  Expr<Sigmoid, Expr<Add, Expr<MatMul<>, Matrix<float>, Param<float,2>>, Param<float,1>>>
```

Each operation returns an `Expr<Op, Args...>` node that stores its operands
by value (tensors copy shallowly, so this is cheap) and computes only its
shape. The nesting of the types *is* the graph, fixed at compile time
(chapter 2). This gives the three phases of the framework:

1. **Build.** Writing the formula builds the typed tree. Shapes are checked
   at once, by contracts, and rewrite rules simplify the *type*:
   `x + zeros` is just `x`, and a transposed operand of `matmul` becomes a
   kernel flag.
2. **Evaluate.** `eval()` walks the tree. Element-wise subtrees are fused
   into a single loop through "readers", vectorized with `std::simd` and run
   in parallel. Structural operations (matmul, convolution, softmax) call
   their kernels (chapters 2 and 6). On a GPU, the same tree becomes one
   OpenCL kernel whose source is generated from the type (chapter 7).
3. **Differentiate.** `backward(loss)` records a *tape*, a tree of `Node`
   and `Leaf` objects shaped exactly like the expression type, holding the
   intermediate values. It then walks the tape in reverse: at each node, the
   op's `grad<I>` rule turns the output's gradient into each operand's, and a
   `Param` leaf accumulates it (chapter 3). Branches with no parameter in
   them are known from the types and pruned at compile time: they are
   evaluated once, fused, and never differentiated.

Compared with PyTorch, which builds its graph at run time as operations
execute, this graph costs nothing to build and is checked and optimized by
the compiler. The price is that its shape must be known when compiling.
Where the structure depends on run-time data, such as a sequence of unknown
length, the recurrent network chapter (chapter 9) keeps one step's type
fixed and loops over the steps at run time. And where one value feeds several
others, as in a transformer's residual connections, `share()` (chapter 10)
evaluates it once and makes the tree a graph.

## How are the gradients checked?

Every gradient rule is compared with finite differences,
`(f(x + h) − f(x − h)) / 2h`, in double precision, for every element of
every parameter. See chapter 3. A wrong rule does not crash; it quietly
trains worse, so these tests matter.

## Can I load weights trained in PyTorch?

XiNN reads and writes safetensors, the format of Hugging Face and PyTorch,
with parameters named by reflection (`fc1.weight`). Two differences remain:
names must match, and `torch.nn.Linear` stores its weight as (out, in),
while XiNN's `Dense` stores (in, out), so it needs a transpose.

## Does it run on Linux, macOS or ARM?

It is written in portable standard C++, and the only Windows-specific
setting is a workaround for a MinGW compiler bug (chapter 6). It has been
built and tested only on Windows so far. On Linux or macOS, GCC 16 should
build it; on ARM, `std::simd` picks the native vector width (NEON). Reports
are welcome.

## Why so many honest caveats in the book?

Because a teaching text that overclaims teaches the wrong lesson. Where a
result is weak, the book shows it and explains why: persistence is hard to
beat in traffic forecasting; a GRU and a transformer do no better than an MLP
on a one-hour window of one detector; a graph network on PEMS08 does no
better with the real detector graph than with a random one; an incident
detector first raised 30 false alarms a day. Knowing when a model does *not*
help is part of the craft.

## How can I contribute or report a problem?

Open an issue or a pull request at <https://github.com/wxinix/XiNN>.

## What makes XiNN worth studying?

It connects **the mathematics, the implementation, and the performance
consequences** in one project small enough to study. A gradient rule is
derived in the text, written as a few lines of C++, checked against finite
differences, and then timed.

Two further things set it apart. The traffic flow examples give the machinery
a purpose beyond digit recognition: fundamental diagrams, freeway
simulation, incident detection with a CNN, speed forecasting with recurrent,
transformer and graph networks on real detectors, and car-following
stability. And the book reports
where things do *not* work: where an optimization does not help, where a
baseline is hard to beat, and that its matrix-multiplication algorithm is
established work, not an invention. That makes the engineering discussion
more useful.

## What is the hardest part?

The learning curve. Understanding deep learning while also navigating
advanced C++26 (reflection, contracts, expression templates, senders) is
demanding. For someone interested in both, that intersection is exactly the
appeal; for others, it is a lot at once.

A gentler path through the book:

- Read chapters 1–3 for the ideas, and skim the C++ details the first time.
- Run the examples before reading their code: the results tell you what the
  code is for.
- Come back for reflection (chapters 1.12, 2.8, 4) and `std::execution`
  (chapter 6) once the deep learning is familiar.
- After chapter 5, the architecture chapters (8 CNNs, 9 RNNs, 10
  transformers, 11 graph networks) can be read in any order. Chapter 7 (GPU)
  is optional.

## Is XiNN correct? Can I trust its results?

It is tested, but it is not audited. Every gradient rule is checked against
finite differences; kernels are compared with direct reference loops, with
and without SIMD and threads; the generated GPU kernels are checked as text,
compiled as C and run against CPU results, and the GPU runtime is tested
against a fake OpenCL driver; each example reports results that can be
reproduced with one command. That catches many mistakes, but not all, and
nobody outside the project has reviewed the implementation.

So trust it as a teaching implementation: good for learning and
experimenting, not for results that matter without independent checking,
and not for production (see below). If you find a bug, please report it.

## Should I use XiNN in production?

No. XiNN is built for reading and learning. PyTorch, JAX, ONNX Runtime and
libtorch are the tools for production work. They offer GPU training, tuned
kernels for every platform, model import and export, and years of hardening.
XiNN trains on the CPU (with a minimal GPU path for forward evaluation),
covers the operations its examples need, and favours code you can read.

## Is anything in it new?

Mostly no, and it does not claim to be. The algorithms are the textbook ones:
reverse-mode differentiation, Adam, im2col convolution, the GRU of Cho et al.
(2014), the transformer of Vaswani et al. (2017), the graph convolution of
Kipf and Welling (2017), and the blocked matrix product of Goto and van de
Geijn (2008) that BLAS libraries use.

What is somewhat unusual is **how they are expressed**: in portable standard
C++26, with static reflection, expansion statements, contracts,
`std::execution` and `std::simd`, instead of intrinsics, macros or code
generators. Gradients are derived from expression *types*, parameter-free
branches are pruned at compile time, and GPU kernels are generated at compile
time from the same types. That makes XiNN a teaching example and a
demonstration of the new language, not a technical edge.

## How fast is it, compared with PyTorch?

We have not benchmarked against PyTorch, so we make no claim. What we
measured, on one machine (8 hardware threads, AVX2/FMA), is in chapter 6: a
float matrix product at 433 GFLOP/s, 16 times the textbook loop, a
1.3-second MNIST epoch for the MLP and 13 seconds for the CNN (chapter 8).
On an RTX 3060, the GPU backend's register-blocked matmul reaches 4.6
TFLOP/s, 36% of the card's rating and about 5 times the host's 32-thread
CPU (chapter 7). Optimized BLAS libraries are faster still on
large matrices, and PyTorch on a GPU, with cuBLAS and cuDNN, is in another
league.

## Why only GCC 16?

In 2026, GCC 16 is the compiler that implements static reflection (P2996) and
contracts (P2900), and XiNN uses both throughout. MSVC does not yet support
them, and upstream Clang is not there yet either. When other compilers catch
up, XiNN should need only small changes. The compiler bugs found along the
way are in chapters 0, 6 and 9: two internal compiler errors with contracts,
and an AVX stack-alignment bug on Windows.

## Why is `std::execution` provided by stdexec?

GCC's standard library does not ship C++26 senders yet. stdexec is NVIDIA's
reference implementation of the proposal (P2300), with the same names.
`xinn/parallel.hpp` switches to `<execution>` as soon as the library defines
`__cpp_lib_senders`. XiNN's code is written against the standard interface.

## Why traffic examples?

XiNN grew out of teaching TR-GY 7353 *Data Driven Mobility Modeling and
Simulation* at the NYU Tandon School of Engineering (Spring 2025). Traffic is
a good field for machine learning: it has well-understood physical models
(the fundamental diagram, kinematic waves, car following), real sensor data
(Caltrans PeMS), and questions where a learned model must be judged against
those models, not just scored on accuracy. The case study on car-following
stability shows why: a model can fit its data well and still misjudge
stability. And chapters 9–11 show the other side: on real detector data,
a larger network is not automatically a better forecaster.

## Why was it not used in the course?

It requires mastery of both modern C++ and the internals of deep learning,
more than the course could assume.

## How does it relate to Li Wei's book?

XiNN's overall design follows 李伟 (Li Wei), *动手打造深度学习框架*
(2022), and its MetaNN framework. It is an independent implementation in
C++26, smaller, with a different autograd design and additions such as
training, optimizers and traffic examples. See [Credits](credits.md) for the
details.

## Does a fancier model forecast traffic better?

Not on the data used here. Chapters 9, 10 and 11 forecast the speed at 170
real PEMS08 detectors 15, 30 and 60 minutes ahead, all on the same task
(errors in km/h):

| model | 15 min | 30 min | 60 min |
|---|---|---|---|
| persistence (speed now) | 2.00 | 2.60 | 3.38 |
| MLP | 1.90 | 2.45 | 3.19 |
| GRU (chapter 9) | 1.91 | 2.46 | 3.20 |
| transformer (chapter 10) | 1.95 | 2.53 | 3.28 |
| graph network with self weight (chapter 11) | 1.96 | 2.53 | 3.27 |

(MLP and GRU with the smooth-L1 loss of chapter 9; the others with squared
error, so compare within a chapter for the fine differences.)

All networks improve on persistence by about 5%, and none beats the MLP. A
one-hour window of one detector holds little that a small dense network
cannot use, and the PEMS08 detector graph carries little extra information
(chapter 11). On the simulated corridor of chapter 11, where queues visibly
travel from detector to detector, the graph network *does* win clearly.
Architecture helps when it matches structure that is really in the data.

## Can I train a language model with it?

A tiny one. Chapter 10 trains a character-level transformer (2 blocks,
121,000 parameters, context 64) on the text of this book in about two and a
half minutes on 8 CPU threads. It reaches 2.53 bits per character on held-out
text, against 4.93 for character frequencies alone, and its samples look like
Markdown, English words and C++ fragments. That is how language models work,
at a size that fits in a book; it is not a useful assistant.

## How do I run the GPU code on my machine?

You need an OpenCL device: any recent NVIDIA, AMD or Intel GPU driver
provides one. Then:

```
cmake --preset release
cmake --build --preset release
ctest --preset release -R gpu --output-on-failure
build/release/examples/bench_gpu --source
```

No test line should say "skipped". `bench_gpu --source` prints the generated
kernel, then compares CPU and GPU times. Chapter 7 has the details.

## Does XiNN run on a GPU?

Yes, a minimal backend (chapter 7): forward evaluation in `float` through
OpenCL, which ships with NVIDIA, AMD and Intel drivers and is loaded at run
time, so building needs no SDK. An element-wise tree becomes one fused kernel
whose source is generated at compile time from the expression's type; matmul
uses a register-blocked local-memory kernel; `gpu::Queue` is a `std::execution`
scheduler for GPU work. There is no autograd on the GPU, and performance is
far from cuBLAS.

The book was written in a virtual machine without GPU access. There, the
generated kernels are checked character by character, compiled as C and run
against the CPU's results, and the runtime is tested against a fake OpenCL
driver. The same tests then passed on an NVIDIA GeForce RTX 3060. There, a
fused element-wise kernel is 3 to 15 times faster than 32 CPU threads when
the data is already on the device, but about 3 times slower when it must
cross the PCIe bus. Matmul reaches 1 TFLOP/s with a simple tiled kernel, a
tie with the CPU, 2.7 TFLOP/s once each work item computes a 4 × 4 block in
registers, and 4.6 TFLOP/s with 8 × 8 blocks. Loading four floats at a time
(`float4`), a standard trick, made it slightly slower. Chapter 7 has the
tables.

## Where are the data sets?

They are not in the repository. `python tools/prepare_data.py` downloads
MNIST and PEMS08 from their publishers into `data/`. The traffic simulator
needs no downloads.

## How do I cite XiNN?

> Wuping Xin. *XiNN: a small deep learning framework in C++26, and a book
> that explains it.* 2026. <https://github.com/wxinix/XiNN>

## What is the license?

The code is BSD 3-Clause, and the book text is CC BY 4.0. See
[License](LICENSE.md).
