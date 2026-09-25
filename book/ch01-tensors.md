# 1. Tensors

Code: `include/xinn/shape.hpp`, `tensor.hpp`, `concepts.hpp`,
`constant.hpp`, `format.hpp`. Tests: `tests/test_shape.cpp`,
`tests/test_tensor.cpp`.

## 1.1 What a tensor is

Every value in a neural network is a **tensor**: an array with some number of
dimensions. The number of dimensions is the **rank**.

| Rank | Name | Example |
|---|---|---|
| 0 | scalar | the loss of one training step |
| 1 | vector | the bias of a layer, one per output |
| 2 | matrix | the weights of a layer, inputs × outputs; a batch of 64 samples × 784 pixels |
| 4 | — | a batch of images: batch × channels × height × width |

A tensor is a flat block of numbers plus a shape. XiNN stores elements in
**row-major** order: the last index changes fastest. In a 2×3 matrix, element
(i, j) sits at offset `i*3 + j`.

## 1.2 Two design decisions

**Rank at compile time, sizes at run time.** `Tensor<float, 2>` is always a
matrix, but it may be 3×5 or 64×784. The rank must be static because
operations depend on it: the product of two matrices is a matrix, and a sum
over one dimension lowers the rank by one. With the rank in the type, the
compiler checks such rules. Sizes change with the data (batch size, sequence
length), so they stay dynamic.

**No device parameter.** A framework that targets GPUs adds a device tag,
as in `Tensor<Elem, Device, Rank>`, and passes it through every template.
XiNN runs on the CPU only and leaves the tag out.

## 1.3 Shape

```cpp
template <std::size_t R>
struct Shape {
    static constexpr std::size_t rank = R;
    std::array<std::size_t, R> dims{};

    constexpr explicit Shape(std::convertible_to<std::size_t> auto... ds)
        requires(sizeof...(ds) == R);

    constexpr std::size_t operator[](std::size_t i) const pre(i < R);
    constexpr std::size_t count() const;   // product of dims; 1 for R == 0
};

template <std::convertible_to<std::size_t>... Ds>
Shape(Ds...) -> Shape<sizeof...(Ds)>;
```

The deduction guide lets you write `Shape(3, 4)` and get a `Shape<2>`.
Everything is `constexpr`, so shapes also work at compile time:

```cpp
constexpr Shape s(2, 3, 4);
static_assert(s.count() == 24);
```

`pre(i < R)` is a **contract precondition**, covered in §1.10.

## 1.4 Broadcasting

A layer computes `x·W + b`. Here `x·W` is a batch × outputs matrix, and `b`
is a vector with one entry per output. The same `b` must be added to every
row. **Broadcasting** makes this legal without copying `b`.

The rule in XiNN: align the two shapes at their last dimension.
Every dimension present in both must be equal. The lower-rank tensor is
repeated along the extra leading dimensions.

| a | b | result |
|---|---|---|
| (4, 5) | (3, 4, 5) | (3, 4, 5) |
| (5) | (3, 4, 5) | (3, 4, 5) |
| () | (3, 4) | (3, 4) |
| (3, 4) | (3, 4, 5) | error |

NumPy also stretches dimensions of size 1, so (3, 1) + (3, 4) works there.
XiNN does not do that. The rule is smaller and still covers biases and
scalars.

```cpp
template <std::size_t A, std::size_t B>
constexpr Shape<std::max(A, B)> broadcast(const Shape<A>& a, const Shape<B>& b)
    pre(broadcastable(a, b));
```

Note the return type: the result rank is computed at compile time from the
input ranks.

## 1.5 Storage and shallow copies

```cpp
template <class T, std::size_t R>
class Tensor {
    Shape<R> shape_;
    std::shared_ptr<T[]> buf_;   // owns the elements
    T* data_;                    // first element of this tensor's window
};
```

Copying a tensor copies the pointer, not the elements. Two copies share one
buffer. This matters later: lazy operations (chapter 2) hold their operands
by value, and layers keep their inputs for the backward pass (chapter 3).
Deep copies would make both slow.

Sharing has a danger. A layer keeps its input `x` for the backward pass. If
the caller then changes `x` in place, the gradient is computed from the
wrong numbers. The rule that prevents this: **a tensor may be written only
through a handle that owns its buffer alone.**

```cpp
bool shared() const noexcept { return buf_.use_count() > 1; }

std::span<T> mut_flat() pre(!empty() && !shared());
MutView      mut()      pre(!empty() && !shared());
```

To change a tensor that is shared, `clone()` it first. A clone is a deep
copy with a buffer of its own.

`identical(other)` tests whether two handles see the same buffer window.
This is identity, not equality of values. The evaluator in chapter 6 uses it
to spot repeated work.

## 1.6 Views: `mdspan` and `submdspan`

`std::mdspan` (C++23) is a non-owning view of a multidimensional array: a
pointer plus extents plus a layout. It does the index-to-offset arithmetic
that would otherwise be written by hand.

```cpp
using View = std::mdspan<const T, std::dextents<std::size_t, R>>;
View view() const { return View(data_, shape_.dims); }

auto v = m.view();
float x = v[2, 1];          // C++23 multidimensional subscript
```

`std::submdspan` (C++26) cuts a view out of a view. Each argument either
fixes one dimension to an index or keeps it with `std::full_extent`:

```cpp
auto col = std::submdspan(v, std::full_extent, 1);   // column 1 of a matrix
```

`t[i]` returns the rank-(R−1) slice at index `i` of the first dimension. It
uses `submdspan` with one index followed by R−1 `full_extent`s:

```cpp
auto sub = [&]<std::size_t... K>(std::index_sequence<K...>) {
    return std::submdspan(raw_view(), i, ((void)K, std::full_extent)...);
}(std::make_index_sequence<R - 1>{});
return Tensor<T, R - 1>(buf_, sub.data_handle(), shape_of(sub));
```

The lambda turns the number R−1 into a pack of R−1 `full_extent`s. The slice
shares `buf_`, so it costs no copy. While a slice is alive, the parent counts
as shared and cannot be written.

## 1.7 `set`: the value comes last

To write one element, `t.set(i, j, k, value)` reads naturally. C++ cannot
declare a parameter after a pack (`(I... idx, T value)`). Before C++26 the
usual fix was a recursive template that peels off one argument at a time.
C++26 **pack indexing** picks elements of a pack directly:

```cpp
template <class... A> requires(sizeof...(A) == R + 1)
void set(A... args) {
    [&]<std::size_t... K>(std::index_sequence<K...>) {
        raw_view()[static_cast<std::size_t>(args...[K])...] = static_cast<T>(args...[R]);
    }(std::make_index_sequence<R>{});
}
```

`args...[R]` is the last argument, the value. `args...[K]...` expands to the
first R arguments, the indices.

## 1.8 The `TensorLike` concept

The classic C++17 way to say "this type is a matrix" is a category tag
(`using category = matrix_tag;`) found with SFINAE, while the interface a
type must provide lives only in documentation. In XiNN the interface is a
**concept**, so the compiler checks it:

```cpp
template <class X>
concept TensorLike = requires(const X& x) {
    typename X::value_type;
    requires std::same_as<std::remove_const_t<decltype(X::rank)>, std::size_t>;
    { x.shape() } -> std::same_as<const Shape<X::rank>&>;
    { x.eval() } -> std::same_as<Tensor<typename X::value_type, X::rank>>;
};
```

A tensor-like type has an element type, a compile-time rank, a run-time
shape, and an `eval()` that produces a real `Tensor`. `Tensor` is the
**principal type**: every tensor-like value can become one. For a `Tensor`,
`eval()` is a shallow copy.

`TensorOfRank<X, 2>` narrows the concept to matrices. Functions can now be
constrained plainly:

```cpp
auto transpose(const TensorOfRank<2> auto& m);
```

## 1.9 Constants

A constant tensor has all elements equal. It stores a shape and at most one
number. There are two kinds:

| Type | The value is known | Built by |
|---|---|---|
| `Constant<T, R>` | at run time | `Constant(shape, value)`, `scalar(v)` |
| `Filled<T, R, V>` | at compile time, as a template argument | `zeros(shape)`, `ones(shape)` |

`Zeros<T, R>` and `Ones<T, R>` are `Filled` with `V` = 0 and 1.
Floating-point template arguments have been allowed since C++20.

Why separate types, when a filled `Tensor` holds the same numbers? Because
the type carries knowledge. Code that sees a `Constant` knows every element
is equal without reading any of them. Code that sees `Zeros` knows even the
value, at compile time, so chapter 2 can rewrite `x + zeros` to plain `x`
before the program runs. A `Tensor` full of zeros gives no such guarantee.

All of them satisfy `TensorLike`. `eval()` turns them into a filled
`Tensor`.

## 1.10 Contracts

A contract states what must be true at a point in the program. C++26 has
three kinds:

```cpp
T f(int i) pre(i >= 0)          // precondition: checked on entry
           post(r: r != 0);     // postcondition: r names the result
contract_assert(x < n);         // assertion inside a body
```

XiNN uses contracts for shape rules and the write rule, checks that older
code does with `assert` or exceptions. A violation calls the **contract-violation
handler**. The default one prints the condition and terminates:

```
contract violation in function ... mut_flat() ... : !empty() && !shared()
[assertion_kind: pre, semantic: enforce, ...]
```

Contracts are better than `assert` in two ways.

- A precondition is part of the function's declaration. Callers and tools
  can see it.
- The build chooses how contracts are evaluated (`enforce`, `observe`,
  `ignore`) without editing the code. A release build can turn the checks
  off.

A contract checked during constant evaluation is a compile error. For
example, `static_assert(broadcast(Shape(3, 4), Shape(3, 4, 5)) == ...)`
does not compile.

A program may replace the handler. The death tests in `tests/death.hpp`
replace it with one that prints and exits with status 1:

```cpp
void handle_contract_violation(const std::contracts::contract_violation& v) {
    std::println(stderr, "contract violated: {}", v.comment());
    std::exit(1);
}
```

GCC 16.1 has a bug with `pre` on variadic templates; see §0.4.
`operator()` and `set` check their conditions with `contract_assert`
instead.

## 1.11 Printing

`format.hpp` teaches `std::format` about tensors:

```cpp
std::println("{}", m);        // [[1, 2, 3], [4, 5, 6]]
std::println("{:.2f}", v);    // the spec applies to each element
```

The formatter keeps a `std::formatter<T>` for the elements and hands it the
format spec. It then walks the tensor with `submdspan`, peeling one
dimension per level until a single element is left.

## 1.12 How the test runner finds tests

The test runner is the first use of **static reflection** in XiNN.

```cpp
template <std::meta::info NS>
consteval auto functions_in() {
    std::vector<std::meta::info> fs;
    for (auto m : std::meta::members_of(NS, std::meta::access_context::current()))
        if (std::meta::is_function(m)) fs.push_back(m);
    return std::define_static_array(fs);
}

template <std::meta::info NS>
int run_tests() {
    template for (constexpr auto f : functions_in<NS>()) {
        std::println("[ run ] {}", std::meta::identifier_of(f));
        [:f:]();
    }
    ...
}

int main() { return check::run_tests<^^tests>(); }
```

Line by line:

- `^^tests` is a **reflection**: a compile-time value of type
  `std::meta::info` that stands for the namespace `tests`.
- `members_of` lists the declarations in it. `is_function` keeps the
  functions. This runs in a `consteval` function, so it happens at compile
  time. `std::vector` can be used there.
- A compile-time `std::vector` cannot survive into run time.
  `define_static_array` copies its contents into a static array that can.
- `template for` is an **expansion statement**. It looks like a loop, but
  the compiler stamps out the body once per element, and each `f` is a
  constant.
- `[:f:]` is a **splice**. It turns the reflection back into the thing it
  reflects, here the function, so `[:f:]()` calls it.
- `identifier_of(f)` is the function's name as a string.

The result is a test framework with no registration macros. A new test is
just a new function in the namespace.

## 1.13 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| category tags + SFINAE detection | `TensorLike` concept |
| interface described in comments | interface checked by the concept |
| hand-written index-to-offset code | `mdspan` |
| sub-arrays by pointer arithmetic | `submdspan` |
| recursive templates to split a pack | pack indexing `args...[I]` |
| `assert` and exceptions | contracts |
| test registration macros | reflection over a namespace |

## 1.14 Exercises

1. Add `Tensor::reshape<R2>(Shape<R2>)`, which returns a tensor that shares
   the buffer with a new shape. What precondition does it need?
2. Write `transpose` for matrices using `std::layout_stride`, without
   copying. Why can it not return a `Tensor` as defined here?
3. Add NumPy-style size-1 stretching to `broadcastable`. Which tests change?
