# 4. Modules

Code: `include/xinn/module.hpp`, `layers.hpp`, `io.hpp`. Tests:
`tests/test_module.cpp`, `tests/test_io.cpp`, `tests/verify_safetensors.py`.
Example: `examples/modules.cpp`.

## 4.1 From loose parameters to models

In chapter 3 a network was six loose `Param`s, a lambda, and a
`std::tie(...)` to loop over them. That does not scale. A real model has
dozens of parameter tensors in nested parts, and every tool needs all of
them: the optimizer, `zero_grad`, saving, printing.

In XiNN a model is an ordinary struct:

```cpp
struct SpiralNet : Module {
    Dense<{.activation = Activation::relu}> fc1, fc2;
    Dense<> out;

    explicit SpiralNet(Rng& rng) : fc1(2, 32, rng), fc2(32, 32, rng), out(32, 2, rng) {}
    auto forward(const auto& x) const { return out(fc2(fc1(x))); }
};
```

Nothing is registered. A member is a parameter because its type is `Param`.
A member is a submodule because its type derives from `Module`. Its name is
the member's name. **Reflection** reads all of this from the struct
definition.

## 4.2 One base class, no CRTP

`Module` provides the shared operations. It must work on the *derived*
struct: `for_each_param` has to see `SpiralNet`'s members, not `Module`'s.
The C++17 answer is CRTP, `struct SpiralNet : Module<SpiralNet>`. C++23's
**deducing `this`** makes the object's real type a template parameter of the
member function:

```cpp
struct Module {
    template <class Self, class... Args>
    decltype(auto) operator()(this Self&& self, Args&&... args) {
        return std::forward<Self>(self).forward(std::forward<Args>(args)...);
    }

    template <class Self, class F>
    void for_each_param(this Self& self, F&& f) { detail::visit_params(self, f, ""); }

    void zero_grad(this auto& self);
    std::size_t parameter_count(this const auto& self);
    std::string summary(this const auto& self);
};
```

In `net(x)`, `Self` is `SpiralNet&`, so `self.forward(x)` calls
`SpiralNet::forward`. Constness follows the object too: `for_each_param` on
a `const` model hands out `const Param&`.

## 4.3 Finding parameters by reflection

```cpp
template <class M>
consteval auto members_of() {
    return std::define_static_array(
        std::meta::nonstatic_data_members_of(^^M, std::meta::access_context::unchecked()));
}

template <class M, class F>
void visit_params(M& m, F& f, std::string_view prefix) {
    auto visit = [&](std::string_view name, auto& x) {
        using X = std::remove_cvref_t<decltype(x)>;
        if constexpr (is_param_v<X>) f(join_name(prefix, name), x);
        else if constexpr (ModuleType<X>) visit_params(x, f, join_name(prefix, name));
    };
    template for (constexpr auto member : members_of<std::remove_cvref_t<M>>())
        visit(std::meta::identifier_of(member), m.[:member:]);
}
```

- `nonstatic_data_members_of(^^M, ...)` lists the data members of `M`.
  `access_context::unchecked()` includes private ones, so a module may keep
  its parameters private.
- `m.[:member:]` **splices** a member reflection into a member access. It is
  `m.fc1` when `member` reflects `fc1`.
- Recursing into submodules builds dotted names: `encoder.fc1.weight`.
- Members of other types (sizes, flags, `Nothing`) are skipped at compile
  time.

The same walk also runs purely at compile time. It counts parameter tensors
from the type alone:

```cpp
static_assert(param_tensor_count<SpiralNet> == 6);
```

`summary()` prints what the walk finds:

```
parameter               shape               values
fc1.weight              [2, 32]                 64
fc1.bias                [32]                    32
fc2.weight              [32, 32]              1024
fc2.bias                [32]                    32
out.weight              [32, 2]                 64
out.bias                [2]                      2
total                                         1218
```

## 4.4 Options as template arguments

A layer has options: bias or not, which activation. XiNN passes them as one
struct, used as a template argument:

```cpp
struct DenseOptions {
    bool bias = true;
    Activation activation = Activation::none;
};

template <DenseOptions Opt = {}, class T = float>
struct Dense : Module {
    Param<T, 2> weight;
    [[no_unique_address]] std::conditional_t<Opt.bias, Param<T, 1>, Nothing> bias;
    ...
};

Dense<{.activation = Activation::relu}> fc1;   // designated initializers
Dense<{.bias = false}> proj;
```

C++20 allows class types as template parameters when they are *structural*:
public members, no `mutable`, and all members structural themselves. With
designated initializers, the options read like named arguments. Unnamed
options keep their defaults.

Because the options are part of the type:

- `forward` picks the right code with `if constexpr (Opt.bias)`. No run-time
  flag is checked.
- An absent bias has type `Nothing`, and `[[no_unique_address]]` lets it
  take no space. The test checks
  `sizeof(Dense<{.bias = false}>) == sizeof(Param<float, 2>)`.
- The initializer follows the activation: He for ReLU, Glorot otherwise
  (§3.10).
- Equal options give the same type, whatever order you write them in.

## 4.5 Sequential

For a plain stack of layers:

```cpp
Sequential net{Dense<>{2, 16, rng}, ReLU{}, Dense<>{16, 1, rng}};
```

Class template argument deduction infers `Sequential<Dense<>, ReLU, Dense<>>`
from the constructor arguments. The layers live in a `std::tuple`, and
`forward` applies them in order. A tuple has no member names, so
`Sequential` lists its parts with a `children()` hook. The module walks use
it instead of reflection and name the children by position:

```cpp
template <class Self>
auto children(this Self& self) {
    return std::apply([](auto&... l) { return std::tie(l...); }, self.layers);
}

// in visit_params:
if constexpr (HasChildren<M>) {
    std::size_t i = 0;
    template for (auto& child : m.children()) visit(std::to_string(i++), child);
}
```

The parameters are named `0.weight`, `0.bias`, `2.weight`, `2.bias`, the
same scheme PyTorch uses. `ReLU` at position 1 has no parameters.

## 4.6 Physical models are modules too

A module is not limited to neural networks. The traffic model of §3.13
becomes:

```cpp
struct Drake : Module {
    Param<float, 0> vf{Scalar<float>(Shape<0>{}, 1.0f)};
    Param<float, 0> kc{Scalar<float>(Shape<0>{}, 0.5f)};
    auto forward(const auto& k) const { return vf * exp(square(k / kc) * -0.5f); }
};
```

Its summary lists `vf` and `kc`. It trains, saves and loads like any network,
and the saved file holds two named numbers a traffic engineer can read.

## 4.7 Saving and loading: safetensors

XiNN stores parameters in **safetensors**, the format used by Hugging Face
and PyTorch. It is simple and safe to load. Unlike Python's `pickle`, it
contains data only, never code.

```
8 bytes    N: header length, unsigned little-endian
N bytes    JSON: {"fc1.weight": {"dtype": "F32", "shape": [2, 32], "data_offsets": [0, 256]}, ...,
                  "__metadata__": {"format": "pt", "writer": "xinn"}}
rest       raw row-major tensor data
```

```cpp
if (auto r = save(net, "net.safetensors"); !r) std::println("{}", r.error());
if (auto r = load(net, "net.safetensors"); !r) std::println("{}", r.error());
```

Both return `std::expected<void, std::string>` (C++23). A failure is a
value: the file is missing, a tensor is missing, or a dtype or shape differs.
It is not an exception. `load` checks every parameter before changing any,
so a failed load leaves the model as it was.

The header is read by a small JSON parser in `io.hpp`, about 120 lines. It
uses `std::expected` throughout, so each step either returns a value or the
reason it failed.

Two checks guard compatibility:

- `tests/test_io.cpp` compares header bytes exactly, e.g.
  `"fc1.bias":{"dtype":"F32","shape":[5],"data_offsets":[60,80]}`.
- `tests/verify_safetensors.py` reads a written file with Python's standard
  library, following the published format. It checks the alignment, that
  the tensors tile the data exactly, and the metadata types. If the
  `safetensors` package is installed, it also loads the file with it. CTest
  runs it after `test_io`.

**A note for PyTorch users.** `torch.nn.Linear` stores its weight as
(out, in) and computes `x·Wᵀ`. `Dense` stores (in, out) and computes `x·W`.
Moving weights between the two needs a transpose.

## 4.8 Results

`examples/modules.cpp` trains both models, saves them, and loads them into
fresh instances:

```
trained: loss 0.0070, accuracy 99.8%
fresh copy: accuracy 56.0%
loaded:     accuracy 99.8%   (from spiral.safetensors)

fitted: vf = 109.9 km/h, kc = 35.1 veh/km   (truth: 110.0, 35.0)
loaded:  vf = 109.9 km/h, kc = 35.1 veh/km   (from drake.safetensors)
```

## 4.9 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| CRTP base `Module<Derived>` | deducing `this` |
| register each parameter by hand, with a name string | reflection over data members |
| sublayer lists and wiring declared in type lists | ordinary `forward()` code; the compiler checks shapes and ranks |
| options as policy classes or run-time flags | a structural struct as a template argument |
| exceptions for I/O errors | `std::expected` |
| recursion over a tuple of layers | `template for` over the tuple |

## 4.10 Exercises

1. Add `Dense<{.activation = Activation::tanh}>` layers to `SpiralNet` in
   place of ReLU. Does accuracy change? Why does the initializer change too?
2. Give `Module` a `freeze()` that excludes some parameters from
   `for_each_param` when a flag is set. Where does the flag live?
3. Write `count_values<M>()`, which computes the number of trainable values
   at compile time for modules whose shapes are also template arguments.
   What must change in `Dense`?
4. Save a `Drake` model and open it in Python with `safetensors.numpy`.
