# 3. Automatic Differentiation

Code: `include/xinn/param.hpp`, `autograd.hpp`, `loss.hpp`, `init.hpp`, and
the `grad` members in `ops.hpp`, `linalg.hpp`, `reduce.hpp`. Tests:
`tests/test_autograd.cpp`, `tests/compile_fail/`. Example:
`examples/spiral.cpp`, `examples/fundamental_diagram.cpp`.

## 3.1 What training needs

A network is a function of its inputs *and* its parameters. Training adjusts
the parameters to make a **loss** small: one number that says how wrong the
network is on the training data. The basic method is **gradient descent**:

```
W  ←  W − lr · ∂loss/∂W
```

The gradient `∂loss/∂W` points in the direction that increases the loss the
most, so a small step against it decreases the loss. `lr`, the *learning
rate*, sets the step size. Repeat this many times and the network learns.

Everything hinges on computing `∂loss/∂W` for every parameter, efficiently
and without writing derivatives by hand. That is **automatic
differentiation**.

## 3.2 The chain rule, backwards

Take `loss = f(g(h(W)))`. The chain rule gives

```
∂loss/∂W = f'(g(h(W))) · g'(h(W)) · h'(W)
```

Reverse mode evaluates this product from the left, starting at the loss:

1. The gradient of the loss with respect to itself is 1.
2. Each operation receives the gradient of its output, `g = ∂loss/∂y`, and
   turns it into the gradient of each input, `∂loss/∂a`. This needs only the
   operation's *local* derivative.
3. When the walk reaches a parameter, the gradient has arrived.

One backward walk yields the gradients of *all* parameters at once, at a cost
similar to one forward evaluation. That is why every deep learning framework
uses reverse mode.

## 3.3 Parameters

```cpp
template <class T, std::size_t R>
class Param {
    struct State { Tensor<T, R> value; Tensor<T, R> grad; };
    std::shared_ptr<State> s_;
public:
    const Tensor<T, R>& tensor() const;   // the current value
    const Tensor<T, R>& grad() const;     // ∂loss/∂this, accumulated
    void zero_grad();
    void assign(const auto& x);           // replace the value
};
```

A `Param` is a *handle*. Copies share one state, the way copies of a
`shared_ptr` share one object. An expression that uses `W` stores a copy,
and `backward()` accumulates into the one real gradient. `Param` is
`TensorLike`, so it works in expressions like any tensor.

`assign` replaces the value instead of writing into it. A tensor that
something still holds, such as a recorded forward pass, never changes behind
its back. This is the write rule of §1.5 again.

## 3.4 Gradient rules

Each operation says how to pass a gradient to operand `I`:

```cpp
struct Mul {
    constexpr auto operator()(auto a, auto b) const { return a * b; }
    template <std::size_t I> auto grad(const auto& g, const auto&, const auto&... a) const {
        return g * a...[1 - I];   // ∂(a0·a1)/∂a0 = a1, and the other way round
    }
};
struct Sigmoid {
    auto operator()(auto a) const { return 1 / (1 + std::exp(-a)); }
    template <std::size_t> auto grad(const auto& g, const auto& y, const auto&) const {
        return g * y * (1 - y);   // σ' = σ(1 − σ), written with the output y
    }
};
```

`grad<I>(g, y, a...)` receives the incoming gradient `g`, the operation's
output `y`, and its operands `a...`. It returns a **lazy expression**, so a
gradient rule is itself fused when it is evaluated. Some rules are cheaper
in terms of the output, like sigmoid's `y(1 − y)`, and some need the
operands, like `Mul`.

| Operation | Rule for `∂loss/∂a` |
|---|---|
| `a + b`, `a − b` | `g`, and `g` / `−g` for `b` |
| `a * b`, `a / b` | `g·b`, `g/b` (and `g·a`, `−g·y/b` for `b`) |
| `exp`, `log`, `sqrt` | `g·y`, `g/a`, `g/(2y)` |
| `tanh`, `sigmoid` | `g(1 − y²)`, `g·y(1 − y)` |
| `relu` | `g` where `a > 0`, else 0 |
| `transpose` | `transpose(g)` |
| `matmul(A, B)` | `g·Bᵀ` for A, `Aᵀ·g` for B |
| `sum<Axis>` | `g` repeated along the axis |
| `softmax` | `y(g − Σ g·y)` row by row |

**The matmul rule and chapter 2.** The rule is written plainly with
`transpose()`:

```cpp
if constexpr (I == 0) return maybe_transpose<TransA>(matmul(g, transpose(maybe_transpose<TransB>(b))));
else return maybe_transpose<TransB>(matmul(transpose(maybe_transpose<TransA>(a)), g));
```

The rewrite rules then remove every transpose: `transpose(transpose(b))`
becomes `b`, and a transposed operand of `matmul` becomes a kernel flag. The
test `matmul_gradients_need_no_transposed_copies` checks with
`static_assert` that `∂loss/∂B` is `MatMul<true, false>`, one kernel that
reads `A` transposed in place.

## 3.5 Broadcast operands

In `x·W + b`, the bias `b` has shape (n) and was repeated along every row of
a (batch, n) result. Each copy of `b` contributed to the loss, so the
gradient of `b` is the **sum** of the output gradient over the batch:

```cpp
template <std::size_t RK, class G>
auto unbroadcast(G&& g) { return sum_leading<G::rank - RK>(std::forward<G>(g)); }
```

Broadcasting repeats along leading dimensions (§1.4), so undoing it sums over
leading dimensions. When no broadcasting happened, `sum_leading<0>` returns
`g` unchanged.

## 3.6 Two passes: record, then backprop

`backward(loss)` first **records** the forward pass. It evaluates the
expression bottom-up and keeps each intermediate result that a gradient rule
may need. The result is a *tape*, a tree with the same shape as the
expression type:

```cpp
template <class L> struct Leaf { L leaf; };                     // Param, tensor, constant
template <class Op, class Out, class Kids>
struct Node { Op op; Out out; Kids kids; };                      // an op on a path to a Param
```

Then it walks the tape top-down:

```cpp
template <class Op, class Out, class... Kids, class G>
void backprop(const Node<Op, Out, std::tuple<Kids...>>& node, const G& g_in) {
    const Out g = g_in;                       // evaluate the incoming gradient once
    template for (constexpr std::size_t I : indices<sizeof...(Kids)>) {
        using Kid = Kids...[I];
        if constexpr (Kid::trainable) {
            auto gi = grad_of<I>(node.op, g, node.out, operand values...);
            backprop(std::get<I>(node.kids), unbroadcast<Kid rank>(std::move(gi)));
        }
    }
}
```

At a `Param` leaf, `backprop` calls `accumulate`. `backward` seeds the walk
with 1 and returns the loss value, so a training loop gets it for free.

The tape is ordinary data, built for this one call and thrown away. No
global graph and no registration are involved. **The expression type is the
tape's blueprint.**

## 3.7 Pruning at compile time

Whether a subexpression depends on any parameter is a property of its type:

```cpp
template <class X> struct depends_on_params : std::bool_constant<is_param_v<X>> {};
template <class Op, class... Args>
struct depends_on_params<Expr<Op, Args...>>
    : std::bool_constant<(depends_on_params<Args>::value || ...)> {};
```

`record` uses it. A subtree without parameters, such as input
preprocessing, is evaluated **fused** into one tensor and stored as a leaf.
`backprop` never visits it, because `Kid::trainable` is false, and the
compiler removes that branch. No gradient code is generated for work that
needs no gradient.

```cpp
auto tape = detail::record(exp(x * 2) + w);
// kids: Leaf<Vector<double>> (exp(x*2), evaluated once), Leaf<Param<double, 1>>
```

This decides, at compile time, which parts of a network must produce
gradients.

## 3.8 Errors at compile time

Two mistakes are caught by the compiler, and the tests in
`tests/compile_fail/` check the messages.

**An op without a gradient rule on a path to a parameter.** A lambda passed
to `map` has no `grad`. The message is built by reflection and C++26's
computed `static_assert` messages:

```
error: static assertion failed: xinn: operation `main()::<lambda(float)>` has no
gradient rule, but a Param is among its operands. ...
```

```cpp
template <class Op>
consteval std::string_view no_grad_message() {
    return std::define_static_string(
        std::string("xinn: operation `") + std::string(std::meta::display_string_of(^^Op)) + "` has ...");
}
static_assert(false, no_grad_message<Op>());   // in the branch that lacks a rule
```

**A loss that depends on no parameter.** `backward(sum(x))` would silently
do nothing, so it is rejected instead.

## 3.9 Losses

**Mean squared error**, for predicting numbers, is composed from existing
operations, so its gradient comes for free:

```cpp
auto mse(auto&& pred, auto&& target) { return mean(square(pred - target)); }
```

**Softmax cross-entropy**, for classification. The network outputs one
score (*logit*) per class. Softmax turns the scores into probabilities,
`pᵢ = exp(xᵢ) / Σₖ exp(xₖ)`. The loss is `−log p` of the correct class,
averaged over the batch.

Composed naively as `-log(softmax(x))`, this overflows for large scores and
takes `log(0)` for tiny probabilities. The fused op rewrites each row as

```
loss_row = max(x) + log Σₖ exp(xₖ − max(x)) − Σⱼ tⱼ xⱼ
```

which is exact and never overflows. Its gradient is also simple:
`(softmax(x) − t) / batch`. The test
`cross_entropy_matches_its_definition` compares it with the naive formula.

Targets are one-hot rows: `one_hot(labels, classes)` turns label 2 of 4 into
`[0, 0, 1, 0]`.

## 3.10 Initialization

Parameters start random. The spread matters: with weights of standard
deviation `s`, each layer multiplies the spread of its signal by about
`s·√fan_in`. **He initialization** (`s = √(2/fan_in)`) keeps it steady
through ReLU layers. **Glorot** does the same for tanh, sigmoid, and output
layers. Biases start at zero.

## 3.11 Testing gradients

A wrong gradient rule does not crash. Training just quietly works worse. So
every rule is checked against **finite differences**:

```
∂loss/∂pᵢ ≈ (loss(pᵢ + h) − loss(pᵢ − h)) / 2h
```

`gradcheck(loss, params...)` compares this with `backward()` for every
element of every parameter, in `double`. `tests/test_autograd.cpp` runs it
over every operation, over broadcasting, and over a parameter used twice,
where its gradients must add up.

One trap came up while writing the tests: ReLU has no derivative at 0. A test
input that makes some `x·W` exactly 0 fails the check, even though the
rule is correct. The test data avoids it.

## 3.12 Training a network

`examples/spiral.cpp` trains a 2 → 32 → 32 → 2 network on two interleaved
spirals, which no straight line can separate:

```cpp
auto model = [&](const auto& x) {
    auto h1 = relu(matmul(x, W1) + b1);
    auto h2 = relu(matmul(h1, W2) + b2);
    return matmul(h2, W3) + b3;
};

for (int step = 0; step <= 2000; ++step) {
    template for (auto& p : params) p.zero_grad();
    const float loss = backward(softmax_cross_entropy(model(X), Y));
    template for (auto& p : params) p.assign(p - lr * p.grad());
}
```

`params` is `std::tie(W1, b1, ...)`, and `template for` walks the tuple
(§1.12). Output from a release build:

```
step    0  loss 1.4615  accuracy  50.0%
step  250  loss 0.0168  accuracy  99.2%
step 2000  loss 0.0065  accuracy  99.8%
```

The whole run takes under a second. The model is one expression type:

```
(matmul(relu((matmul(relu((matmul(T[400x2], P[2x32]) + P[32])), P[32x32]) + P[32])), P[32x2]) + P[2])
```

## 3.13 Traffic: fitting the fundamental diagram

Autograd differentiates *any* expression, not only neural networks. A model
from traffic flow theory, with a handful of physical parameters, trains with
the same `backward()`.

On a freeway lane, loop detectors measure **density** k (veh/km) and
**speed** v (km/h). Their relation v(k) is the **fundamental diagram**.
Flow is q = k·v (veh/h). Three numbers matter most:

- **free-flow speed** vf: the speed on an empty road
- **critical density** kc: the density at which flow peaks
- **capacity** q_max: that peak flow

`examples/fundamental_diagram.cpp` generates 600 noisy detector readings from
**Drake's model**, v = vf·exp(−(k/kc)²/2) with vf = 110 km/h and
kc = 35 veh/km. It then fits three models:

```cpp
Param vf(...), kj(...);                                   // rank-0 Params: plain numbers
auto greenshields = [&](const auto& k) { return vf * (1 - k / kj); };

Param dvf(...), kc(...);
auto drake = [&](const auto& k) { return dvf * exp(square(k / kc) * -0.5f); };

auto mlp = [&](const auto& x) { return matmul(tanh(matmul(tanh(matmul(x, W1) + b1), W2) + b2), W3) + b3; };
```

Each is trained by gradient descent on `mse(model(k), v)`. A rank-0 `Param`
broadcasts over all 600 readings, and its gradient is summed back (§3.5).
Densities and speeds are divided by 100 for training, so every parameter is
of order 1. Unscaled, the gradients for vf and kj would differ by orders of
magnitude, and no single learning rate would suit both.

```
  model         RMSE       parameters
  Greenshields  15.1 km/h  vf =  96.3 km/h, kj = 110.7 veh/km
  Drake          3.6 km/h  vf = 109.7 km/h, kc =  35.2 veh/km
  MLP            3.5 km/h  (313 weights, no physical meaning)

  capacity (veh/h) at critical density (veh/km)
    truth         qmax =  2335 at kc =  35.0
    Greenshields  qmax =  2665 at kc =  55.5
    Drake         qmax =  2344 at kc =  35.0
    MLP           qmax =  2351 at kc =  35.5
```

The results teach three lessons:

- **A wrong model form stays wrong.** Greenshields' linear speed–density
  line cannot bend, so the best line misplaces the critical density by 20
  veh/km and overestimates capacity by 14%. At 120 veh/km it predicts a
  negative speed. More data would not fix this.
- **The right form recovers the physics.** Drake's model returns vf and kc
  within 0.5% of the truth. These are numbers a traffic engineer can use
  directly.
- **A network needs no form, but gives no parameters.** The MLP matches the
  curve as well as Drake does, and its capacity estimate is close too. But
  it offers no vf or kc to read off, and outside the observed densities it
  guesses.

The noise here has a standard deviation of 4 km/h, so an RMSE of about 3.6
km/h is as good as any model can do.

## 3.14 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| a backward function written by hand for each layer | gradients derived from the expression type |
| a run-time graph, built as operations execute | a tape whose shape is fixed by the type |
| run-time flags for "needs gradient" | `depends_on_params`, decided at compile time |
| cryptic template errors | `static_assert` messages computed with reflection |
| index recursion over operands | `template for` + pack indexing `Kids...[I]` |

## 3.15 Exercises

1. Add `abs(x)` with its gradient. What should the rule do at 0?
2. Add a gradient rule to `ops::Expand`, then differentiate a loss that uses
   `expand` on a Param. Check it with `gradcheck`.
3. `record` stores every intermediate on the path to a parameter. For
   element-wise nodes the value could be recomputed during backprop instead.
   What would that save, and what would it cost?
4. Fit the **triangular** fundamental diagram (flow rises linearly to
   capacity, then falls linearly to jam density). Its speed curve has a kink
   at kc. Write it with `relu`, and fit it to the Drake data.
5. Replace the hand-written update in `spiral.cpp` with momentum:
   `v ← 0.9v + g`, `W ← W − lr·v`. How much faster does the loss fall?
