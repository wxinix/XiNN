# 10. Attention and Transformers

Code: `include/xinn/attention.hpp` (the operations), `include/xinn/shared.hpp`
(shared nodes), `include/xinn/transformer.hpp` (the layers). Tests:
`tests/test_attention.cpp`. Examples: `examples/tiny_lm.cpp`,
`examples/speed_forecast_transformer.cpp`.

## 10.1 Attention from first principles

A sequence model must let each position use information from the others. A
recurrent network (chapter 9) carries it along step by step, in a state of
fixed size. **Attention** lets every position look at every other position
directly, and decide how much each one matters.

Give each position `i` three vectors, computed from its input `xᵢ` by three
learned matrices:

| vector | name | role |
|---|---|---|
| `qᵢ = xᵢ W_Q` | query | what position i is looking for |
| `kⱼ = xⱼ W_K` | key | what position j offers |
| `vⱼ = xⱼ W_V` | value | what position j passes on if chosen |

Position i scores every position j by the dot product `qᵢ · kⱼ`, turns the
scores into weights that sum to 1 with softmax, and returns the weighted
average of the values:

```
out_i = Σⱼ softmax_j(qᵢ · kⱼ / √d) · vⱼ
```

For all positions at once, with the vectors as rows of matrices Q, K, V:

```
Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
```

That is two matrix products and a softmax. The `√d` keeps the scores from
growing with the vector length d. Without it, softmax saturates and its
gradient vanishes.

Attention has no notion of order: permute the positions and the outputs
permute with them. Order is added to the input as a **positional encoding**,
a fixed pattern of sines and cosines of different frequencies (Vaswani et
al. 2017):

```
PE[pos, 2i] = sin(pos / 10000^(2i/d)),   PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

For a language model, position i must not see the characters after it. A
**causal mask** adds a large negative number to every score with j > i.
After the exponential in softmax, those weights are exactly zero.

## 10.2 The operations and their gradients

Sequences are rank-3 tensors (batch, tokens, features). Five new structural
operations cover a transformer; the rest is matmul, softmax, and element-wise
arithmetic from earlier chapters.

**Batched matrix product.** `bmm<TA, TB>(a, b)` multiplies matching matrices
of two batches: `(…, M, K) · (…, K, N) → (…, M, N)`. The flags transpose an
operand in place, as `MatMul`'s did in chapter 3, so `Q Kᵀ` is
`bmm<false, true>(q, k)` and no transposed copy of K is made. Each batch item
is one call of the chapter 6 kernel, and the items run in parallel. The
gradient rules follow from those of a single product. With `A' = op(A)`,
`B' = op(B)` and `C = A'B'`:

```
∂L/∂A' = G B'ᵀ        ∂L/∂B' = A'ᵀ G
```

and a transposed operand takes the transposed gradient. Each of the eight
cases (two operands, four flag settings) is again one batched product:

```cpp
template <std::size_t I>
auto grad(const auto& g, const auto&, const auto& a, const auto& b) const {
    if constexpr (I == 0) {
        if constexpr (!TransA) return bmm<false, !TransB>(g, b);   // G·op(B)ᵀ
        else return bmm<TransB, true>(b, g);                       // op(B)·Gᵀ
    } else {
        if constexpr (!TransB) return bmm<!TransA, false>(a, g);   // op(A)ᵀ·G
        else return bmm<true, TransA>(g, a);                       // Gᵀ·op(A)
    }
}
```

**Permutation of axes.** `permute<0, 2, 1, 3>(x)` swaps the two middle axes:
`out.shape[i] = x.shape[P[i]]`. The axes are template arguments, so a wrong
permutation is a compile error. The gradient is the inverse permutation,
computed at compile time:

```cpp
static constexpr auto inverse = [] {
    std::array<std::size_t, R> inv{};
    for (std::size_t i = 0; i < R; ++i) inv[perm[i]] = i;
    return inv;
}();
...
return make_expr(Permute<inverse[I]...>{}, g);   // I = 0 .. R-1
```

**Layer normalization.** Each token's feature vector is shifted and scaled to
mean 0 and variance 1, then multiplied by a learned gain and shifted by a
learned bias (Ba, Kiros, Hinton 2016). Only the first step needs a new
operation, `normalize`. The gain and bias are element-wise, so they fuse,
and their gradients come from the rules of chapter 3:

```cpp
auto layer_norm(X&& x, G&& gain, B&& bias, double eps = 1e-5) {
    return normalize(x, eps) * gain + bias;
}
```

For one row of n features, with `y = (x − μ) · r` and `r = 1/√(σ² + ε)`, the
gradient of `normalize` is

```
∂L/∂x = r · (g − mean(g) − y · mean(g ⊙ y))
```

The two means are there because every input moves μ and σ, and through
them every output. The rule needs y (the op's output, which the tape keeps)
and r, which is recomputed from x.

**Embedding.** `embedding(table, ids)` picks rows of a (vocab, d) table by
integer ids. The ids are a `Tensor<std::size_t, R>` stored in the operation,
not an operand, because they get no gradient. The table's gradient is a
**scatter-add**: row `ids[i]` receives the gradient of output row i. A
character used 500 times in a batch gets 500 contributions.

**Selecting one position.** `take<1>(x, i)` picks token i of every sequence:
`(B, T, d) → (B, d)`. Its gradient puts g back at index i and zeros
elsewhere. The forecaster below predicts from the last token.

The tests compare every operation with direct loops and check every gradient
against finite differences, including `bmm` for all four flag settings, and
softmax for rank 3 and 4.

## 10.3 Multi-head attention by reshape and permute

One attention can look for one kind of relation at a time. **Multi-head
attention** runs h of them side by side, each on d/h of the features, and
joins the results. No copy per head is needed: the heads are an axis.

```
x (B, T, d)
  -> query(x)                   (B, T, d)          a TokenDense layer
  -> reshape                    (B, T, h, dh)      dh = d / h
  -> permute<0, 2, 1, 3>        (B, h, T, dh)
  -> reshape                    (B·h, T, dh)       one batch of B·h sequences
scores  = bmm<false, true>(q, k) · 1/√dh           (B·h, T, T)
weights = softmax(scores + mask)                    the mask (T, T) broadcasts
context = bmm(weights, v)                           (B·h, T, dh)
  -> reshape, permute<0, 2, 1, 3>, reshape          (B, T, d)
  -> output(context)
```

A reshape shares the buffer (chapter 8). The permutation is the only real
data movement. The (T, T) mask broadcasts over the B·h score matrices under
the rule of chapter 2, which aligns shapes at the trailing end.

`TokenDense` is a dense layer applied to every token. It folds the leading
axes into one, `(B, T, d) → (B·T, d)`, so a whole batch of sequences is a
single matrix product.

## 10.4 A block, and the problem of reuse

A transformer block has two residual steps, each with layer normalization in
front (the "pre-LN" arrangement, Xiong et al. 2020):

```
h = x + MHA(LN(x))
y = h + FFN(LN(h))          FFN: TokenDense(d, 4d) + ReLU, TokenDense(4d, d)
```

Written as plain XiNN expressions, this has a cost that grows exponentially.
An expression stores its operands by value. `x + MHA(LN(x))` contains x
four times (the residual, and the query, key and value projections), and
evaluating it evaluates x each time. `y` contains h twice, so one block
contains its input eight times, and a stack of L blocks evaluates the
network's input 8^L times. `backward()` walks every copy too. For the residual chains of chapters 3–8 this never mattered;
here it does.

The fix is to evaluate such a value once and let it be used many times.
`share(e)` evaluates e now and records its tape (chapter 3). The result, a
`Shared<T, R>`, is a leaf to the expressions that use it:

```cpp
auto forward(const X& x_in) const {
    const auto x = share(x_in);
    const auto h = share(x + attention(norm1(x)));
    return share(h + ffn2(ffn1(norm2(h))));
}
```

For the backward pass, a Shared node is treated exactly like a Param: a leaf
that collects the gradient from all its uses. That is one line, a partial
specialization of the variable template from chapter 3:

```cpp
template <class T, std::size_t R> inline constexpr bool is_param_v<Shared<T, R>> = true;
```

`backward()` on a loss that contains Shared nodes then works in two stages.
First the loss's own tape is walked, which leaves gradients in the Shared
leaves it uses. Then every Shared node, newest first, passes its summed
gradient down its own tape into older nodes and Params. Newest first is
always a valid order, because a node can only use nodes that existed before
it. A test builds ten residual steps in a row. As plain expressions the first
step would be evaluated 1,024 times; with `share` once, and the gradients
still match finite differences.

This is the one place where XiNN erases a type. Each node's tape has its own
static type, but the graph of nodes is known only at run time. So a node
keeps its tape inside a `std::move_only_function` and exposes one virtual
function, `propagate()`. The static machinery of chapter 3 still does all
the work inside each node. The erased layer only fixes the order.

A `backward()` overload for expressions that contain a Shared node is
chosen by the normal rules of partial ordering, because
`Expr<Op, Args...>` is more specialized than the plain template parameter of
chapter 3's version. User code calls `backward(loss)` as before.

## 10.5 A tiny language model trained on this book

`examples/tiny_lm.cpp` reads every chapter of this book (`book/*.md`) as one
text of Unicode characters. Every tenth block of 1,024 characters is held
out for validation. The model predicts each next character from the 64
before it:

```cpp
struct CharLm : Module {
    Embedding<> embed;
    TransformerBlock<> block1, block2;
    LayerNorm<> norm;
    TokenDense<> head;
    auto forward(const Tensor<std::size_t, 2>& ids) const {
        const auto pe = sinusoidal_positions<float>(ids.shape()[1], embed.table.shape()[1]);
        return head(norm(block2(block1(embed(ids) + pe))));
    }
};
```

The loss is the cross-entropy of the next character, averaged over all 64
positions of 32 random windows per step: `softmax_cross_entropy` of chapter 5
on the logits reshaped to (32·64, vocab). Training uses Adam with a short
warm-up, cosine decay and weight decay 0.01.

When this chapter was written, the book had 16 files and 140,528
characters, 161 of them distinct (Chinese characters and mathematical
symbols included); 127,216 for training and 13,312 for validation. The model
has 2 blocks, d_model 64, 4 heads and 120,865 parameters. 2,000 steps take
141 s on 8 threads:

| step | training loss | validation loss | validation bits/char |
|---|---|---|---|
| 250 | 2.699 nats | 2.538 nats | 3.66 |
| 500 | 2.188 | 2.160 | 3.12 |
| 1,000 | 1.748 | 1.875 | 2.70 |
| 1,500 | 1.555 | 1.766 | 2.55 |
| 2,000 | 1.498 | 1.751 | **2.53** |

Bits per character is the loss divided by ln 2: the average number of yes/no
questions the model needs to guess the next character. Knowing only the
character frequencies gives 4.93 bits (the unigram line printed by the
example), so the model has learned about half of what there is to learn from
the preceding characters. It has seen 4.1 million characters in total, about
32 passes over a small text; models trained on corpora a thousand times
larger do much better.

The gap between training and validation loss opens after 1,000 steps. A
wider model (d_model 96, 254,366 parameters), run on an earlier state of
the book of 124,655 characters, fell to 1.61 bits/char on training data but
only 2.54 on validation: more capacity went into memorizing the text, not
into generalizing from it. Only more text would help.

Samples, 240 characters from a prompt (the prompt is given, the rest is
generated):

```
temperature 0.5:
A tensor and the training detectors of chapter 2
   test format.

## Why the training and the experiences of the trained backe refecepts one each is one a refull.

## 3.1 Macke and the parameter does the core this model compile
```

````
temperature 0.8:
The gradient is a every row ong.

```

The comporess it one the wire writtted this the test accuries tensor one learning. Its it with (stdexes). A pubm(shǒastrets), ...
````

The model has learned the surface of the book. It writes English words
mostly spelled right, Markdown section headings with chapter numbers, code
fences, list indentation, backquoted identifiers, and C++ fragments such as
`const auto& self` and `auto& const { return`. It has also picked up the
book's vocabulary: tensor, gradient, compiler, detectors, contracts,
congestion. It has no grammar beyond a few words and no meaning at all. That
is what 64 characters of context and 120 thousand parameters can hold, and it
is a fair picture of how language models start.


## 10.6 Traffic: speed forecasting with a transformer

`examples/speed_forecast_transformer.cpp` forecasts detector speeds on the
real PEMS08 data of chapter 5. The task is shared with chapters 9 and 11, so
the three model families can be compared on the same numbers:

- speed only, in km/h (mph × 1.609344), 170 detectors, 62 days of 5-minute
  intervals;
- input: one detector's last 12 intervals (one hour);
- targets: its speed 3, 6 and 12 intervals ahead (15, 30, 60 minutes), all
  three from one model;
- days 0–49 train, days 50–61 test; one mean and standard deviation for all
  speeds, from the training days only;
- training windows: every 10th interval of the training days, all detectors
  (244,460 windows); test: every window of the test days (583,610);
- MAE in km/h, over all test windows and over those whose true speed is
  below 64 km/h (congested: 2.20% of the test targets).

The transformer reads the 12 intervals as 12 tokens. Each speed is projected
to 32 features by a `TokenDense(1, 32)`, the positional encoding is added,
two encoder blocks follow (4 heads, no mask: every interval may look at
every other), and after a final layer norm the last token predicts the
three horizons through a `Dense(32, 3)`:

```cpp
auto forward(const Matrix<float>& x) const {
    auto tokens = embed(reshape(x, Shape{B, history, 1uz})) + positions;   // (B, 12, 32)
    return head(take<1>(norm(block2(block1(tokens))), history - 1));
}
```

The baselines are persistence (the speed now, for every horizon), the
historical average of the detector at the target's time of day over the
training days, and an MLP (12 → 64 → 64 → 3, ReLU). Both networks are
trained for 6 epochs with Adam and mean squared error on the standardized
speeds.

| method | parameters | 15 min | 30 min | 60 min | congested 15 | congested 30 | congested 60 |
|---|---|---|---|---|---|---|---|
| persistence | – | 2.00 | 2.60 | 3.38 | **8.60** | **13.47** | **20.83** |
| historical average | – | 3.93 | 3.93 | 3.93 | 33.70 | 33.71 | 33.73 |
| MLP | 5,187 | **1.94** | **2.51** | **3.27** | 10.31 | 17.75 | 28.58 |
| transformer | 17,315 | 1.95 | 2.53 | 3.28 | 10.35 | 17.75 | 28.41 |

MAE in km/h on the 12 test days. Training took 4 s for the MLP and 300 s for
the transformer (8 threads, on a machine shared with other jobs).

What this says, plainly:

- **The transformer is no better than the MLP.** Both beat persistence by
  3–4% overall, and they differ by 0.01–0.02 km/h. For twelve numbers from
  one detector, a dense layer already sees everything at once; attention has
  nothing to add that the MLP cannot learn directly. It costs 75 times the
  training time.
- **Both networks lose to persistence in congestion,** and by more at longer
  horizons. Congested targets are 2.2% of the data. A model trained on mean
  squared error over all windows learns that speeds usually recover, and
  pulls its forecasts towards free flow. Persistence never does, which is
  right while a queue lasts.
- **The historical average is poor here.** PEMS08 is mostly free-flowing, so
  the profile for the time of day misses exactly the days that matter.
- **What is missing is information, not capacity.** Congestion at a detector
  usually arrives from downstream. A model that sees only its own detector
  cannot know that a queue is on its way. The graph networks of chapter 11
  give it the neighbours.


## 10.7 Old C++ and new

| Need | Older C++ | C++26 in XiNN |
|---|---|---|
| a permutation of axes as a type | a recursive template to invert a type list | `Permute<P...>` with a `constexpr` array, `Permute<inverse[I]...>` |
| checking that P is a permutation | template metaprogram | a `constexpr` lambda and `static_assert` |
| a transposed operand in a batched product | a separate transposed copy, or a run-time flag | `bmm<TA, TB>`: `bool` template parameters fold into the kernel |
| "a leaf that receives a gradient" | a base class with virtual `accumulate` | a partial specialization of the variable template `is_param_v` |
| a graph whose shape is known only at run time | `std::function` (copyable only) | `std::move_only_function` (C++23) holding a move-only tape |
| options of a layer | constructor argument lists | `TransformerBlock({.d_model = 64, .heads = 4, .causal = true}, rng)` |

## 10.8 Exercises

1. Replace the sinusoidal encoding in `tiny_lm` with a learned position
   table (an `Embedding` of positions). Does validation loss change?
2. Remove the `share` calls in `TransformerBlock` and time one training step
   with one and with two blocks. Explain the ratio.
3. Add a third block to the language model, or double `d_model`. Which helps
   validation loss more for the same training time?
4. In the forecaster, predict the *change* from the last observed speed
   instead of the speed itself. How do MAE and congested MAE move?
5. Add time of day as two extra input features (sine and cosine) to both the
   MLP and the transformer. Which model gains more?

## 10.9 References

- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez,
  Ł. Kaiser, I. Polosukhin. *Attention is all you need.* NeurIPS 2017.
- J. L. Ba, J. R. Kiros, G. E. Hinton. *Layer normalization.*
  arXiv:1607.06450, 2016.
- R. Xiong et al. *On layer normalization in the transformer architecture.*
  ICML 2020.
