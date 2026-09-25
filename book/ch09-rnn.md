# 9. Recurrent Networks

Code: `include/xinn/rnn.hpp`, and `backward(expr, seed)` in `autograd.hpp`.
Tests: `tests/test_rnn.cpp`. Example: `examples/speed_forecast_rnn.cpp`.
Data: PEMS08 (`python tools/prepare_data.py`).

## 9.1 Sequences

The networks so far map one input to one output. Many inputs are
sequences: the speeds a detector has reported over the last hour, the words
of a sentence. An MLP can take a sequence of fixed length as one flat
vector, but it treats position 3 and position 4 as unrelated inputs, each
with its own weights.

A **recurrent network** reads the sequence one step at a time and carries a
**hidden state** `h` from step to step:

```
h_t = f(x_t, h_{t−1})        t = 1, ..., T,   h_0 = 0
```

The same function `f`, with the same weights, is applied at every step. The
state `h_t` is a summary of everything seen up to `t`. A read-out layer
turns the last state (or every state) into predictions.

## 9.2 The GRU

A plain recurrent cell, `h_t = tanh(x_t W + h_{t−1} U + b)`, is hard to
train: the gradient is multiplied by `U` and by the slope of `tanh` once per
step, so over many steps it shrinks towards zero or blows up. The **gated
recurrent unit** (Cho et al., 2014) adds gates that let the state pass
through unchanged:

```
z  = σ(x Wz + h Uz + bz)            update gate: keep how much of h?
r  = σ(x Wr + h Ur + br)            reset gate:  use how much of h for n?
n  = tanh(x Wn + r ∘ (h Un) + bn)   candidate state
h' = (1 − z) ∘ n + z ∘ h
```

`∘` is element-wise. Where `z` is near 1, the unit copies its old state and
the gradient flows back undamped. XiNN applies the reset gate *after* the
product `h Un`, as cuDNN and PyTorch do; the original paper uses
`(r ∘ h) Un`. Both work about equally well.

In XiNN one step is a member function that returns a lazy expression:

```cpp
template <class T = float>
struct GRU : Module {
    Param<T, 2> Wz, Wr, Wn;   // (in, hidden)
    Param<T, 2> Uz, Ur, Un;   // (hidden, hidden)
    Param<T, 1> bz, br, bn;

    auto step(const auto& x, const auto& h) const {
        auto z = sigmoid(matmul(x, Wz) + matmul(h, Uz) + bz);
        auto r = sigmoid(matmul(x, Wr) + matmul(h, Ur) + br);
        auto n = tanh(matmul(x, Wn) + r * matmul(h, Un) + bn);
        return blend(z, n, h);   // (1 − z) n + z h
    }
};
```

The parameters are found by reflection, like those of any module
(chapter 4), so `Adam opt(net)` works unchanged.

**One op for the last line.** Written directly, `(1 - z) * n + z * h` uses
`z` twice. The tape of chapter 3 is a tree, not a graph with shared nodes,
so `z`, with its two matrix products, would be computed and differentiated
twice. `blend` is one element-wise op with three operands, so `z` appears
once:

```cpp
struct Blend {
    constexpr auto operator()(auto z, auto n, auto h) const { return n + z * (h - n); }
    template <std::size_t I>
    auto grad(const auto& g, const auto&, const auto& z, const auto& n, const auto& h) const {
        if constexpr (I == 0) return g * (h - n);
        else if constexpr (I == 1) return g * (1 - z);
        else return g * z;
    }
};
```

It also works on SIMD vectors, so the whole step after the matrix products
is still one fused, vectorized loop.

## 9.3 A step is a type; a sequence is not

Every expression in XiNN has a type that records its whole structure
(chapter 2). The type of one GRU step is long but fixed: it is the same at
every step and for every batch. So one step compiles to fused loops, and
its tape is laid out at compile time, as in chapter 3.

A whole sequence is different. Unrolled into one expression, the output
after `T` steps would contain `T` nested copies of the step type. Two things
go wrong:

1. **The type grows with `T`.** Twelve steps give a type twelve times as
   deep, and the compiler must instantiate all of it. A hundred steps would
   make compile times and error messages unusable.
2. **`T` is a run-time value.** The length of a sequence is data. A type
   that depends on it would need a separate instantiation for every length
   that can occur, and the lengths are not known when the program is
   compiled.

So the sequence is not an expression. It is a **run-time loop over a
compile-time type**:

```cpp
std::vector<Matrix<T>> forward(const Tensor<T, 3>& xs, Matrix<T> h0 = {}) const {
    // xs: (steps, batch, in); returns h_0, h_1, ..., h_T
    std::vector<Matrix<T>> hs{h0};
    for (std::size_t t = 0; t < xs.shape()[0]; ++t) hs.push_back(step(xs[t], hs.back()).eval());
    return hs;
}
```

Each step is evaluated at once and its result stored as a plain tensor.
`xs[t]` is a view into the input (chapter 1), not a copy: the input is kept
time-major, `(steps, batch, in)`, so the inputs of one step are contiguous.

Training needs gradients through this loop. `backward(loss)` of chapter 3
cannot provide them: the loss depends on the weights through all `T` steps,
but no single expression contains them.

## 9.4 Vector-Jacobian products

`backward(loss)` differentiates a *scalar*. Its first step seeds the walk
with `∂loss/∂loss = 1`. Nothing in the tape walk needs the seed to be 1, or
the expression to be a scalar. For an expression `y` of any shape and a seed
`s` of the same shape, the same walk computes

```
P.grad() += Σᵢ sᵢ · ∂yᵢ/∂P        for every Param P in y
```

This is a **vector-Jacobian product** (VJP). If `s = ∂loss/∂y`, it is
exactly the chain rule applied from `y` downwards: it continues a backward
pass that was stopped at `y`. XiNN adds one overload:

```cpp
template <TensorArg Y, TensorArg S>
    requires(Y::rank == S::rank)
auto backward(const Y& y, const S& seed) {
    contract_assert(y.shape() == seed.shape());
    const auto tape = detail::record(y);
    detail::backprop(tape, seed);
    return Tensor<value_type, rank>(tape.value());   // y's value
}
```

It is the scalar version with the seed as a parameter. The test
`vjp_equals_gradient_of_weighted_sum` checks it two ways: against
`backward(sum(y * s))`, which must give the same gradients, and against
finite differences.

(The shape check is a `contract_assert` in the body, not a `pre`. GCC 16.1
crashes with an internal error on `pre` for this template, whose return
type is deduced.)

## 9.5 Backpropagation through time

Now the loop can be run backwards. Suppose the loss sees the hidden states,
and `dh_t = ∂loss/∂h_t` arrives from outside (from a read-out layer). The
state `h_t` affects the loss directly and through all later steps, so its
full gradient is

```
g_t = dh_t + (∂h_{t+1}/∂h_t)ᵀ g_{t+1}
```

The second term is a VJP of step `t+1` with seed `g_{t+1}`, taken with
respect to its input `h_t`. The same VJP adds the step's share to the
gradients of the weights. To get the gradient with respect to `h_t`, `h_t`
must be a `Param`: a fresh leaf, made from the stored tensor, whose gradient
is read off after the step.

```cpp
Matrix<T> bptt(const Tensor<T, 3>& xs, std::span<const Matrix<T>> hs,
               std::span<const Matrix<T>> dh) const {
    Matrix<T> g(hs[0].shape());
    for (std::size_t t = xs.shape()[0]; t-- > 0;) {
        if (!dh[t].empty()) g = g + dh[t];   // ∂loss/∂h_{t+1} from outside
        const Param<T, 2> h(hs[t]);          // a leaf: the backward pass stops here
        backward(step(xs[t], h), g);         // weights' gradients += ..., h.grad() = ...
        g = h.grad();                        // ∂loss/∂h_t, the seed of the step before
    }
    return g;                                // ∂loss/∂h_0
}
```

This is **backpropagation through time** (BPTT), written as `T` per-step
VJPs. The weight gradients from all steps add up in the same `Param`s,
because every step uses the same handles.

A training step with a read-out head uses the same trick once more, to cut
between the head and the GRU:

```cpp
const auto xs = time_major(xb);                          // (12, batch, 1)
const auto hs = net.gru(xs);                             // forward, no tape
const Param<float, 2> last(hs.back());                   // cut at h_12
const float loss = backward(mse(net.head(last), yb));    // head's gradients, and last.grad()
net.gru.bptt(xs, hs, last.grad());                       // the GRU's gradients
opt.step();
```

**What it costs.** `backward(step(...), g)` records the step again, so every
step's forward work is done twice: once in `forward`, once in `bptt`. In
return only the hidden states are kept between the passes, `T + 1` tensors
of `(batch, hidden)`, and the tape of each step lives only during its own
VJP. Frameworks call this trade *recomputation* or *checkpointing*; here it
falls out of the design. On the forecasting task below, one epoch of the
GRU (203,675 sequences of 12 steps, hidden size 64, batches of 256) takes
17–21 s on 8 threads.

## 9.6 Type erasure only where needed

A framework with a dynamic graph (PyTorch, for example) builds a graph of
node objects at run time for every forward pass. The operations sit behind
a virtual interface, so a graph of any shape and any depth can be built and
walked. This is **type erasure**: the node types are hidden behind one
interface, and each node pays for it with a heap allocation and an indirect
call.

XiNN has used no type erasure so far. The structure of a network was known
when the program was compiled, so it could live in types. A recurrent
network is the first case where part of the structure is data: the number
of steps. The question is how much of the design must become dynamic to
handle it.

The answer here: only the loop. What varies at run time, the sequence
length, is handled by an ordinary run-time loop. What does not vary, the
step, stays a compile-time expression, with its fusion, its pruned tape and
its compile-time checks. Where the two meet, a tensor is wrapped in a fresh
`Param`: the boundary between the fixed-type world and the run-time loop is
a leaf of the tape. Strictly speaking, nothing is erased at all. The loop
never needs to hold "an expression of some type"; it holds tensors.

The same pattern handles other run-time structure:

| Varies at run time | Fixed at compile time | Boundary |
|---|---|---|
| sequence length | one step | `Param` leaf per step |
| sequence length, with a loss at every step | one step | `dh[t]` added to the seed |
| stacked recurrent layers | one layer's step | the `hs` of one layer as the `xs` of the next |

Type erasure, such as `std::function` or a virtual base class, becomes
necessary only when the *kind* of operation varies at run time, for
instance a network architecture read from a file. Even then it is enough to
erase at the level of whole blocks, such as a GRU step or a dense layer, and
keep each block a fixed type inside.

## 9.7 Testing

`tests/test_rnn.cpp` checks, in `double`:

- **VJP.** `backward(y, s)` agrees with `backward(sum(y * s))` and with
  finite differences. For `y = W`, the gradient is the seed itself.
- **`blend`.** Its values at `z = 0, ½, 1`, and its three gradient rules.
- **One step** against the GRU equations, written out element by element.
- **BPTT against finite differences.** A GRU with 2 inputs and 3 hidden
  units runs 6 steps from a random `h_0`. The loss is
  `Σ_t sum(h_t ∘ c_t)` for random `c_t`, so `dh_t = c_t`. Every weight is
  perturbed in turn, the whole sequence is re-run, and the difference
  quotient is compared with the gradient from `bptt`; so is the returned
  `∂loss/∂h_0`. A second test goes through a `Dense` head, with the loss on
  the last state only.
- **A task that needs memory.** Sequences of 8 random values in [−1, 1];
  the target is the *first* value. Predicting 0 gives an MSE of 1/3. A GRU
  with 16 hidden units, trained for 400 Adam steps, gets from 0.29 to
  0.001: it has learned to store a value and hold it for 7 steps.

## 9.8 Forecasting speed on real detectors

**Task.** This task is shared with the transformer and graph-network
chapters, so the models can be compared.

- Data: PEMS08 speeds (chapter 5), converted from mph to km/h; 170
  detectors, 62 days of 5-minute intervals.
- Input: the last 12 intervals (1 hour) of speed at one detector.
- Output: its speed 3, 6 and 12 intervals ahead (15, 30, 60 minutes). One
  model predicts all three.
- Split by time: days 0–49 train, days 50–61 test. Speeds are standardized
  with the mean and standard deviation of the training days (102.6 and
  10.6 km/h).
- Training windows: every 12th window per detector, with the starting
  offset shifted by detector, 203,675 windows. Test: all 583,610 windows of
  the test days.

**Models.**

- **Persistence:** the speed in 15, 30 or 60 minutes is the speed now.
- **Historical average:** the mean speed of that detector at that time of
  day, over the training days.
- **MLP:** 12 → 64 → 64 → 3 with ReLU, 5,187 parameters.
- **GRU:** reads the 12 speeds one per step (input size 1, hidden size 64),
  then a linear layer maps `h_12` to the three outputs, 12,867 parameters.

Both networks predict the *change* from the speed now, in standardized
units, so that "no change", the persistence forecast, is the easy default.
(With MSE, predicting the speed itself made no measurable difference after
one epoch.) Both train for 4 epochs with Adam, batches of 256, and a cosine
schedule from 2·10⁻³. Each is trained twice: with MSE, and with a smooth
absolute error, `mean(sqrt(square(pred − y) + 0.01))`, called L1 below. The
L1 loss is built from existing ops, so it needs no gradient rule.

**Metrics.** MAE in km/h per horizon, over three sets of test windows:

- *all targets*;
- *congested targets*, where the true future speed is below 64 km/h (40
  mph); 12,814–12,832 windows, 2.2% of the test set;
- *congested now*, where the current speed is below 64 km/h; 12,805
  windows.

Averages over all windows are dominated by free flow, where speed hardly
changes, hence the other two.

**Results** (release build, 8 threads; training time is per model):

| model | all: 15 / 30 / 60 min | congested targets | congested now | training |
|---|---|---|---|---|
| persistence | 2.00 / 2.60 / 3.38 | **8.60 / 13.47 / 20.83** | 8.65 / 13.50 / 21.10 | |
| historical average | 3.93 / 3.93 / 3.93 | 33.70 / 33.71 / 33.73 | 30.38 / 27.43 / 23.03 | |
| MLP (MSE) | 1.95 / 2.54 / 3.30 | 10.47 / 17.94 / 28.70 | 9.17 / 14.38 / 20.53 | 2 s |
| GRU (MSE) | 1.94 / 2.52 / 3.30 | 10.66 / 17.97 / 28.53 | 9.27 / 14.47 / 20.61 | 74 s |
| MLP (L1) | **1.90 / 2.45 / 3.19** | 9.42 / 15.31 / 24.98 | **8.41 / 12.90 / 19.23** | 1 s |
| GRU (L1) | 1.91 / 2.46 / 3.20 | 9.70 / 15.42 / 24.76 | 8.61 / 13.12 / 19.42 | 78 s |

What the table says:

- **Persistence is very hard to beat.** Over all windows the best network
  improves on it by 0.10 km/h at 15 minutes and 0.19 km/h at 60 minutes,
  about 5%. Speed at one detector changes slowly, and one hour of its own
  history says little about what comes next.
- **The GRU is no better than the MLP.** With 12 inputs of one detector,
  reading them one at a time gives the GRU no information the MLP lacks,
  and the MLP trains 40 to 80 times faster. A recurrent network earns its
  cost on long or variable-length sequences, not on a fixed window of 12
  numbers.
- **The loss matters more than the architecture.** L1 beats MSE for both
  networks, on every column. The reason is in the data. On the training
  days, when a detector is congested, the speed an hour later is on average
  17 km/h higher, but the median rise is only 7 km/h: most jams persist
  (57% are still congested an hour later), and a few clear quickly. MSE
  learns the mean, and so predicts recovery too eagerly; the MAE is
  smallest for the median, which is what an absolute loss learns.
- **"Congested targets" favours persistence.** Selecting the windows whose
  *outcome* is congested picks exactly the cases where speed stayed low or
  fell, where "no change" is right or too optimistic in the right
  direction. Every model that expects some recovery loses on this set.
  Selecting by what the forecaster can see, *congested now*, is the fairer
  test. There the L1 networks beat persistence at every horizon, by
  0.04–0.24 km/h at 15 minutes and 1.7–1.9 km/h at 60. The historical
  average, useless 15 minutes ahead, comes within 2 km/h of persistence at
  60.

A single detector's history cannot foresee the onset of congestion; the
traffic upstream and downstream can. Using the other detectors is the
subject of the attention and graph-network chapters.

Run it with `speed_forecast_rnn [epochs] [direct]`. It takes under three
minutes with the defaults.

## 9.9 Old C++ and new

| Earlier way | XiNN |
|---|---|
| unroll a fixed number of steps with recursive templates, one instantiation per length | a run-time loop over one fixed step type |
| a graph of virtual node objects for every forward pass | a compile-time tape per step, discarded after its VJP |
| a hand-written backward function for each recurrent cell | the step's gradient derived from its expression |
| `(1 − z) * n + z * h` with `z` evaluated twice | one three-operand op; `if constexpr` picks its gradient rule |
| checks with `assert` | `pre` and `contract_assert` (where GCC allows `pre`) |
| copying each time step out of the input | `xs[t]`, a view into a time-major tensor |

## 9.10 Exercises

1. Write `step` for the original GRU variant, `n = tanh(x Wn + (r ∘ h) Un + bn)`,
   and run the BPTT gradient check on it. Nothing else should need to change.
2. Stack two GRU layers: feed the hidden states of the first as the inputs
   of the second. Write the backward pass. (Hint: `bptt` of the second layer
   must also return the gradient with respect to its inputs `xs`. Add it
   the same way as for `h`.)
3. Initialize `bz` to +1 instead of 0, so the units start by copying their
   state. Does the recall test train faster with 20 steps instead of 8?
4. Measure how much of a GRU training step is the recomputation in `bptt`.
   Then keep the tapes from the forward pass instead. How much faster is
   it, and how much more memory does it use?
5. Add the time of day (as sine and cosine) to every step's input in
   `speed_forecast_rnn`. Does it help the congested windows?
6. Replace the smooth L1 loss with a loss for the 0.9 quantile
   (the "pinball" loss). What would such a forecast be useful for?
