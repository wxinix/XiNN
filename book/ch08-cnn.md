# 8. Convolutional Networks

Code: `include/xinn/conv.hpp`, and `Conv2D`, `MaxPool2D` and `Flatten` in
`layers.hpp`. Tests: `tests/test_conv.cpp`. Examples: `examples/mnist_cnn.cpp`,
`examples/incident_detection.cpp`.

## 8.1 Why convolution

An MLP treats an image as 784 unrelated numbers. It must learn separately
what a stroke looks like at every position. A **convolution** slides one
small filter over the whole image:

```
y[f, i, j] = b[f] + Σ_c Σ_kh Σ_kw  w[f, c, kh, kw] · x[c, i·s + kh − p, j·s + kw − p]
```

A 3×3 filter has 9 weights per input channel, whatever the image size, and
it finds its pattern anywhere. **Pooling** keeps the largest response in each
2×2 block, which halves the resolution and makes the result tolerant to small
shifts.

Images are rank-4 tensors in NCHW order: (batch, channels, height, width).

## 8.2 Convolution as a matrix product: im2col

A direct loop over those six indices is slow. The standard trick is
**im2col**. It unrolls every receptive field of an image into one column of a
matrix of shape (C·KH·KW, OH·OW). The convolution then becomes a single
matrix product:

```
out_n (F, OH·OW) = W (F, C·KH·KW) · col_n (C·KH·KW, OH·OW)
```

All the arithmetic runs in the matmul kernel of chapter 6. The gradients are
matrix products too:

| gradient | computation |
|---|---|
| ∂L/∂x | `dcol = Wᵀ · g`, then **col2im** folds the columns back into an image (adding where receptive fields overlap) |
| ∂L/∂W | `Σₙ gₙ · colₙᵀ` |
| ∂L/∂b | the sum of g over batch and positions |

Each is a structural operation with its own `eval`, as in chapter 2, and
`Conv2d::grad<I>` returns them lazily:

```cpp
template <std::size_t I>
auto grad(const auto& g, const auto&, const auto& x, const auto& w, const auto&) const {
    if constexpr (I == 0) return make_expr(Conv2dGradInput{stride, pad, x.shape()}, g, w);
    else if constexpr (I == 1) return make_expr(Conv2dGradWeight{stride, pad, w.shape()}, g, x);
    else return make_expr(ChannelSum{}, g);
}
```

The gradient of max-pooling sends each window's gradient to the position of
its maximum. `reshape` shares the tensor's buffer, and its gradient is the
same reshape backwards.

The tests compare the convolution with the six-loop definition for four
stride/padding combinations, and check every gradient against finite
differences.

## 8.3 Nested parallelism

`Conv2d::eval` processes the images of a batch in parallel with
`parallel_for`. Each task calls the matmul kernel, which is itself parallel.
That is a trap. If every thread of the pool blocks in `sync_wait`, waiting
for inner tasks, no thread is left to run them, and the program deadlocks.

`parallel_for` therefore marks its tasks with a `thread_local` flag, and a
parallel loop started inside one runs serially:

```cpp
inline thread_local bool in_parallel_region = false;
...
if (config.threads && n >= 2 * grain && !detail::in_parallel_region) { /* parallel */ }
```

The outer loop over images keeps all cores busy, so nothing is lost. The test
`nested_parallel_loops_do_not_deadlock` covers it.

## 8.4 Layers

```cpp
struct DigitCnn : Module {
    Conv2D<{.kernel = 3, .padding = 1, .activation = Activation::relu}> conv1, conv2;
    MaxPool2D<2> pool;
    Flatten flatten;
    Dense<{.activation = Activation::relu}> fc;
    Dense<> out;
    ...
    auto forward(const auto& x) const {
        auto img = reshape(x, Shape{x.shape()[0], 1uz, 28uz, 28uz});
        return out(fc(flatten(pool(conv2(pool(conv1(img)))))));
    }
};
```

`Conv2D`'s options are a struct template argument, like `Dense`'s in chapter
4. Its weights use He initialization with fan-in C·K·K.

## 8.5 MNIST

```
epoch 1  loss 0.1451  test accuracy 98.24%  (13 s)
epoch 2  loss 0.0434  test accuracy 98.76%  (13 s)
epoch 3  loss 0.0249  test accuracy 98.94%  (14 s)
```

That is **98.94%** after three epochs, against 98.0% for the MLP of chapter 5,
with fewer parameters (207k against 235k). The network knows that nearby
pixels belong together.

## 8.6 Traffic: incident detection from space–time maps

A freeway operator wants an alarm soon after an incident, such as a crash or
a stalled vehicle, blocks lanes. The alarm should *not* go off for the queue
that forms every weekday at the same bottleneck. Both slow traffic down. What
differs is the pattern in space and time, and that is what convolutions read.

**Data.** 500 days of the Cell Transmission Model corridor, 400 for training
and 100 for testing. An incident occurs on 80% of the days, blocking 65–85% of
the road for 15–60 minutes. The input is the last hour at all 15 detectors,
as a 2-channel image of 15 × 12 pixels (speed and occupancy). The label is
"an incident is disrupting traffic now". Training uses every incident window,
plus three times as many without one, half of them from the daily congestion:
the hard negatives.

**Fair comparison.** Incident detectors are compared at a fixed false-alarm
rate. Each detector's threshold is set on the training days so that 1% of
incident-free intervals raise an alarm. As in the classic California
algorithms, an alarm must also persist for two consecutive intervals.

**Detectors.**
- **rule**: the lowest speed on the corridor, as the score;
- **MLP**: a dense network on the flattened window;
- **CNN**: two convolution layers (the second with stride 2), then a dense
  layer.

**Results** (100 test days, 54 disrupting incidents):

| detector | detection rate | false-alarm rate | false alarms / day | mean time to detect |
|---|---|---|---|---|
| rule: lowest speed | 96.3% | 0.99% | 2.6 | 15.7 min |
| MLP | 98.1% | 1.01% | 2.7 | 13.0 min |
| **CNN** | **100%** | 1.18% | 3.1 | **11.2 min** |

At the same false-alarm rate, severe incidents are easy to *find*. The
difference is **how soon**. The CNN raises its alarm on average 4.5 minutes
before the rule, about 30% sooner. It recognizes the shape of a new queue
growing upstream from the incident before any one detector becomes slow
enough. The MLP sees the same numbers but not their neighbourhood, and lands
in between.

Another 25 test incidents disrupted no traffic at all: they happened in light
traffic and removed a lane nobody needed. No detector can see them, so they
are left out of the detection rate. Counting them as misses would be unfair
to every method; training on them as positives would teach the network noise.
An earlier version of this example made both mistakes. It reported 30 false
alarms a day, until the thresholds were calibrated and the invisible
incidents were set aside.

## 8.7 Exercises

1. Remove the persistence check (alarm on one interval). How do FAR and MTTD
   change for each detector?
2. Feed the CNN three channels by adding flow. Does detection get faster?
3. Make the incidents milder (`incident_capacity_min/max` = 0.3–0.6, the
   simulator's default). Which detector degrades most?
4. MNIST: replace max-pooling with stride-2 convolutions. Accuracy? Speed?
