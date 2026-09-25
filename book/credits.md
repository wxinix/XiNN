# Credits

## The original work

XiNN is inspired by, and owes its overall design to:

> 李伟 (Li Wei). **《动手打造深度学习框架》** (*Building a Deep Learning
> Framework by Hand*). 人民邮电出版社 (Posts & Telecom Press), Beijing,
> 1st edition, April 2022.
>
> Source code: **MetaNN**, <https://github.com/liwei-cpp/MetaNN>
> (branch `book_v2`).

That book shows how C++ template metaprogramming can carry a whole deep
learning framework: the network is described in types, checked by the
compiler, and optimized before it runs. It is the best treatment of that idea
we know of. Anyone who reads Chinese and wants the full story should read it.
It goes further than this book in both depth and scope.

## What XiNN takes from it

The central ideas are Li Wei's:

- Tensors whose rank is part of the type and whose sizes are run-time values.
- Shallow-copied tensors that may be written only when unshared.
- Special tensor types, such as constants, that carry knowledge in their
  type.
- One generic, lazy expression node for all operations, with shapes checked
  when the expression is built and values computed later.
- Layers with a uniform forward/backward interface over named inputs and
  outputs.
- Networks composed from sublayers, with the wiring checked and sorted at
  compile time. Gradients flow in reverse order and are summed where one
  output feeds several layers.
- Working out at compile time which layers must produce gradients.
- A recurrent layer built as a loop over a step layer.
- A register-then-evaluate plan that removes repeated work and can fuse
  operations.

## What XiNN does differently

XiNN is a new, independent implementation. No code is copied from MetaNN.
The differences are deliberate.

**The language.** MetaNN is written in C++17, when a compile-time framework
had to be built from type lists, SFINAE, recursive templates and
inheritance tricks. A large part of the original book teaches that machinery.
XiNN uses C++26 instead:

| Need | MetaNN (C++17) | XiNN (C++26) |
|---|---|---|
| "Is this a tensor?" | category tags + SFINAE | concepts |
| Named compile-time options | policy objects, virtual-inheritance dominance, macros | a config struct as a template argument, designated initializers |
| Named inputs and outputs | a heterogeneous type-keyed dictionary | plain structs, walked with reflection |
| Network wiring and checks | type-level lists and recursive topological sort | `consteval` functions over `std::vector`, readable errors |
| Loops over sublayers | recursive index templates | `template for` over reflected members |
| Index arithmetic, slicing | hand-written | `std::mdspan`, `std::submdspan` |
| Checks | `assert`, exceptions, macros | contracts |

**The scope.** XiNN is smaller and aims to be read in full:

- CPU only; there is no device parameter.
- Fewer tensor types and operations; only what the examples need.
- A different backward pass. In MetaNN each layer implements its own
  backward step and keeps its inputs on stacks. XiNN derives all gradients
  from the expression type: each operation declares a local gradient rule,
  and `backward()` records a tape shaped like the type and walks it in
  reverse. The idea of working out at compile time which parts need
  gradients is kept, but applied per operation instead of per layer.

- A different way to compose networks. MetaNN declares sublayers and their
  connections in type lists and checks and sorts the wiring at compile
  time. In XiNN a model is a struct whose `forward()` is ordinary code;
  the compiler's checking of expression types does the wiring checks, and
  reflection over the struct's members finds and names the parameters.

**Additions.** The original book stops before training. XiNN adds
losses, weight initialization, gradient checking, saving and loading in the
safetensors format, and training examples, including models from traffic
flow theory.
It also adds optimizers (SGD with momentum, Adam), mini-batch training, MNIST,
and traffic prediction on simulated and real detector data.

**The book.** This book is written in English and follows the code stage by
stage. Each chapter teaches one piece of deep learning and the C++26 features
used to build it.

## Software used

- **stdexec** (NVIDIA; Apache-2.0 with LLVM exception), the reference
  implementation of C++26 `std::execution`, fetched at build time:
  <https://github.com/NVIDIA/stdexec>.

- **Dear ImGui** (Omar Cornut; MIT), **ImPlot** (Evan Pezent; MIT) and
  **GLFW** (zlib license), used by XiNN Lab and fetched at build time.

- **OpenCL** is a standard of the Khronos Group
  (<https://www.khronos.org/opencl/>). XiNN declares the few OpenCL types,
  constants and functions it uses, from the OpenCL specification, and loads
  the system's OpenCL library at run time; no Khronos headers or code are
  included.

## Data

- **MNIST**: Yann LeCun, Corinna Cortes, Christopher J. C. Burges, *The MNIST
  database of handwritten digits*. Downloaded from a public mirror by
  `tools/prepare_data.py`.
- **PEMS08**: loop-detector data from the Caltrans Performance Measurement
  System (PeMS), District 8, July–August 2016, as prepared and published by
  Guo et al. with ASTGCN (AAAI 2019) and ASTGNN (IEEE TKDE 2021),
  <https://github.com/guoshnBJTU/ASTGNN>.
- The matrix-product kernel of chapter 6 follows the design of K. Goto and
  R. A. van de Geijn, *Anatomy of high-performance matrix multiplication*,
  ACM TOMS 34(3) (2008), and of the BLIS framework, F. G. Van Zee and R. A.
  van de Geijn, ACM TOMS 41(3) (2015).
- The GRU of chapter 9 is from K. Cho et al., *Learning phrase representations
  using RNN encoder–decoder for statistical machine translation*, EMNLP 2014,
  in the reset-after-product variant used by cuDNN and PyTorch.
- The transformer of chapter 10 follows A. Vaswani et al., *Attention is all
  you need*, NeurIPS 2017 (including the sinusoidal positional encoding).
  Layer normalization is from J. L. Ba, J. R. Kiros, G. E. Hinton, *Layer
  normalization*, arXiv:1607.06450 (2016); the pre-LN block arrangement from
  R. Xiong et al., *On layer normalization in the transformer architecture*,
  ICML 2020.
- Graph convolution in chapter 11 follows T. N. Kipf and M. Welling,
  *Semi-supervised classification with graph convolutional networks*, ICLR
  2017. The Gaussian-kernel edge weights follow Y. Li, R. Yu, C. Shahabi and
  Y. Liu, *Diffusion convolutional recurrent neural network: data-driven
  traffic forecasting*, ICLR 2018.
- The car-following case study reproduces the numerical calculations of
  R. Herman, E. W. Montroll, R. B. Potts, R. W. Rothery, *Traffic dynamics:
  analysis of stability in car following*, Operations Research 7(1), 86–106
  (1959).
- The synthetic corridor uses the Cell Transmission Model of C. F. Daganzo
  (*Transportation Research Part B*, 1994), the discrete form of the
  Lighthill–Whitham–Richards model. The speed–density models in chapter 3 are
  those of Greenshields (1935) and Drake et al. (1967).

The data sets are not redistributed with XiNN; the script fetches them from
their publishers.

Any mistakes in XiNN or in this book are ours, not the original author's.
