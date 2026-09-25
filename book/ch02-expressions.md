# 2. Lazy Operations

Code: `include/xinn/expr.hpp`, `ops.hpp`, `linalg.hpp`, `reduce.hpp`,
`describe.hpp`. Tests: `tests/test_expr.cpp`, `tests/test_fusion.cpp`.
Example: `examples/bench_fusion.cpp`.

## 2.1 Why lazy

One layer of a neural network computes something like

```
y = sigmoid(x · W + b)
```

An *eager* framework evaluates each operation as soon as it is written.
`x · W` fills a temporary tensor. `+ b` reads it and fills a second one.
`sigmoid` fills a third. Each temporary costs a memory allocation plus a
full pass over memory, and on modern CPUs memory traffic, not arithmetic, is
usually the bottleneck.

A *lazy* framework records the operations and computes later. Once it sees
the whole formula, it can do better: run all the element-wise steps in one
loop, skip steps whose result is already known, and pick special kernels.

`examples/bench_fusion.cpp` measures `sigmoid(x * w + b) * 0.5` on
1024×1024 floats (release build):

```
step by step :    7.92 ms
fused        :    3.61 ms   (2.2x faster)
```

The two versions do the same arithmetic. The fused one touches memory once
instead of four times.

## 2.2 The expression node

Every operation in XiNN produces one type:

```cpp
template <class Op, TensorLike... Args>
class Expr {
public:
    using value_type = ...;                   // computed from Op and Args
    static constexpr std::size_t rank = ...;  // computed from Op and Args

    explicit Expr(Op op, Args... args);       // computes the shape, nothing else
    const Shape<rank>& shape() const;
    Tensor<value_type, rank> eval() const;    // does the work
private:
    Op op_;
    std::tuple<Args...> args_;
    Shape<rank> shape_;
};
```

`a + b` for two matrices has type `Expr<ops::Add, Matrix<float>,
Matrix<float>>`. `(a + b) * c` has type `Expr<ops::Mul, Expr<ops::Add, ...>,
Matrix<float>>`. The type *is* the formula.

Three points of the design:

- **Operands are stored by value.** Tensors copy shallowly (§1.5), so this
  is cheap. An expression owns its inputs, so it can never dangle.
- **The shape is computed when the expression is built.** A mismatch fails
  at the line that caused it, not later inside `eval()`. The check is a
  contract (§1.10).
- **An `Expr` is `TensorLike`.** It has a `value_type`, a `rank`, a
  `shape()` and an `eval()`, so it can be an operand of another expression,
  or be printed, or be converted to a `Tensor`:

```cpp
Matrix<float> c = a * a + a;   // evaluated here
```

## 2.3 Two kinds of operation

**Element-wise.** Each output element depends only on the input elements at
the same position. Such an operation is just a function object that works on
single numbers:

```cpp
struct Add {
    static constexpr std::string_view symbol = "+";
    constexpr auto operator()(auto a, auto b) const { return a + b; }
};
struct Sigmoid {
    auto operator()(auto a) const { return decltype(a){1} / (decltype(a){1} + std::exp(-a)); }
};
```

The concept that recognizes them is one line:

```cpp
template <class Op, class... Args>
concept ElementwiseOp = std::invocable<const Op&, typename Args::value_type...>;
```

So *any* callable is an element-wise operation, including a lambda:

```cpp
auto m = map([](float p, float q) { return p > q ? p : q; }, a, b);
```

The result's rank is the largest operand rank, and its shape follows
broadcasting (§1.4). Its `value_type` is whatever the callable returns, so
`map([](float v) { return v > 2; }, a)` is a tensor of `bool`.

**Structural.** An output element depends on many input elements: matrix
product, transpose, sum. Such an operation describes itself with three
members:

```cpp
struct Transpose {
    template <std::size_t... R> static constexpr std::size_t rank = 2;
    Shape<2> shape(const auto& x) const;               // shape rule
    Tensor<T, 2> eval(const Tensor<T, 2>& x) const;    // the kernel
};
```

`Expr::eval()` evaluates the operands into tensors, then calls `Op::eval`.

The two kinds are told apart at compile time, with `if constexpr (elementwise)`
inside `Expr`. No base classes and no virtual functions are involved.

## 2.4 Fused evaluation

To evaluate an element-wise tree in one loop, XiNN turns it into a
**reader**: a small function object that maps a flat index to one element of
the result.

| Operand | Reader |
|---|---|
| `Tensor` | `p[i]` |
| `Constant`, `Zeros`, `Ones` | the value; `i` is ignored |
| element-wise `Expr` | `op(child_0(i), child_1(i), ...)` |
| structural `Expr` | evaluate once into a tensor, then `p[i]` |

`eval()` builds the reader for the whole tree and runs one loop:

```cpp
Tensor<value_type, rank> out(shape_);
const auto read = reader();
auto dst = out.mut_flat();
for (std::size_t i = 0; i < dst.size(); ++i) dst[i] = read(i);
```

Every reader is a concrete type, known at compile time, so the compiler
inlines the whole tree into this loop. The loop for `(a + b) * c` compiles
to roughly `dst[i] = (pa[i] + pb[i]) * pc[i]`.

**Broadcasting costs one line.** Shapes align at the trailing end
(§1.4). In row-major order, a smaller operand therefore just repeats. Element
`i` of a (2, 3) result reads element `i % 3` of a (3) bias:

```cpp
auto operator()(std::size_t i) const { return read(i < n ? i : i % n); }
```

`tests/test_fusion.cpp` checks the claim by counting heap allocations with a
replaced `operator new`:

- Building `sigmoid((a + b) * a - 3) / 2` allocates **nothing**.
- Evaluating it allocates **once**: the output buffer.
- `relu(matmul(x, w) + b)` allocates **twice**: the matrix product, then
  the fused `+ b` and `relu`.

## 2.5 Structural operations

| Function | Result | Note |
|---|---|---|
| `matmul(a, b)` | (m, n) from (m, k) and (k, n) | inner dimensions checked by a contract |
| `transpose(m)` | (n, m) from (m, n) | |
| `sum(x)` | rank 0 | the total |
| `sum<A>(x)` | rank R−1 | sums along axis `A` |
| `mean(x)`, `mean<A>(x)` | as `sum` | `sum` times 1/n |

The axis is a template argument because it decides the result's type.
`sum<2>(matrix)` does not compile. The constraint is a `requires` clause:

```cpp
template <std::size_t Axis = all_axes, TensorArg X>
    requires(Axis == all_axes || Axis < std::remove_cvref_t<X>::rank)
auto sum(X&& x);
```

`matmul` accepts only rank-2 operands. It is constrained by a concept,
`MatrixArg`, so passing a vector is a compile error that names the
concept.

## 2.6 Numbers in expressions

In `a * 2`, the `2` becomes a rank-0 `Constant` of `a`'s element type. A
rank-0 tensor broadcasts with any shape, so no special case is needed later.
The result is `float`, not `int` or `double`:

```cpp
static_assert(std::same_as<decltype(a * 2)::value_type, float>);
```

## 2.7 Rewrite rules

Operand *types* can make the answer known before any computation. The
operators check for that at compile time and return something simpler. The
expression type is rewritten before the program runs.

| Written | Becomes |
|---|---|
| `x + zeros`, `zeros + x` | `x` |
| `x - zeros` | `x` |
| `zeros - x` | `-x` |
| `x * ones`, `x / ones` | `x` |
| `x * zeros` | `zeros` of the broadcast shape |
| `-(-x)` | `x` |
| `transpose(transpose(x))` | `x` |
| `matmul(transpose(a), b)` | `MatMul<true, false>` on `a`, `b` |
| `matmul(a, transpose(b))` | `MatMul<false, true>` on `a`, `b` |

Each rule is an `if constexpr` branch that returns a different type:

```cpp
template <class A, class B>
auto add(A a, B b) {
    if constexpr (ZerosLike<B> && A::rank >= B::rank) return a;
    else if constexpr (ZerosLike<A> && B::rank >= A::rank) return b;
    else return make_expr(ops::Add{}, std::move(a), std::move(b));
}
```

The function's return type is deduced from the branch that is kept, so
`x + zeros(...)` really has type `Matrix<float>`. The tests check this with
`static_assert`. The rank condition matters: `x + zeros` may only return `x`
when `x` already has the shape of the sum.

This is why §1.9 made `Zeros` and `Ones` their own types. A `Constant` with
value 0 is only known to be zero at run time, too late to change a type.

The `matmul` rules pay off in chapter 3. The gradients of a linear layer are
`xᵀ · g` and `g · Wᵀ`. With the rules, neither transpose is ever copied; the
kernel reads its operand in transposed order instead.

## 2.8 Printing formulas with reflection

`describe()` prints an expression as a formula:

```cpp
describe(sigmoid(matmul(x, w) + b))   // "sigmoid((matmul(T[4x3], T[3x2]) + T[2]))"
```

Infix operations give a `symbol`. All other names come from **reflection**,
so no operation has to spell out its own name:

```cpp
consteval std::string_view reflected_name(std::meta::info type) {
    using namespace std::meta;
    if (has_template_arguments(type)) type = template_of(type);   // MatMul<true, false> -> MatMul
    if (!has_identifier(type)) return "map";                      // lambdas have no name
    std::string s(identifier_of(type));
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return define_static_string(s);
}
```

- `^^Op` reflects the operation type.
- `template_of` goes from a specialization, such as `MatMul<true, false>`,
  to its template, `MatMul`.
- `identifier_of` gives the name. A lambda's closure type has none, which
  `has_identifier` detects.
- The lower-casing runs at compile time on a `std::string`. As with
  `define_static_array` in §1.12, a compile-time string cannot survive into
  run time. `define_static_string` stores a copy that can.

## 2.9 Pitfalls

**`auto` holds the formula, not the numbers.** `auto y = a + b;` is an
`Expr`. Every `y.eval()` computes again. To compute once, convert:
`Matrix<float> y = a + b;`.

**The most vexing parse.** With a variable `n`, this declares a *function*:

```cpp
Vector<float> b(Shape(n), -1.0f);   // a function b taking a Shape named n!
```

C++ reads `Shape(n)` as a parameter declaration. Use braces, `Shape{n}`,
whenever the arguments are plain names. With literals, as in `Shape(3)`,
there is no ambiguity.

## 2.10 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| one class template per arity (`UnaryOp`, `BinaryOp`, ...) | one variadic `Expr<Op, Args...>` |
| trait specializations per op for rank, shape, element type | computed in `Expr` from the op itself |
| `enable_if` guards on operator overloads | concepts: `BinaryArgs`, `MatrixArg` |
| `std::tuple_element_t<I, std::tuple<Args...>>` | `Args...[I]` |
| recursive templates to fold over shapes | `template for` over `std::tie(shapes...)` |
| hand-written name strings | names from reflection |

## 2.11 Exercises

1. Add `max(a, b)` and `pow(x, p)` as element-wise operations. Does either
   need a new class?
2. Add the rule `x * scalar(1)` → `x`. Why can it not be done at compile
   time, while `x * ones(...)` can?
3. `mean` builds `sum * (1/n)`. Add a rule so that `sum(x) * 0` becomes a
   `Zeros` of rank 0 without evaluating `sum`.
4. The fused loop ignores SIMD and threads. Before chapter 6, try
   `#pragma omp simd` on it, or split the loop across two `std::jthread`s,
   and measure with `bench_fusion`.
