# 11. Graph Neural Networks

Code: `include/xinn/gnn.hpp`. Tests: `tests/test_gnn.cpp`. Example:
`examples/speed_forecast_gnn.cpp`.

## 11.1 Sensors on a graph

Loop detectors do not stand in a grid. They sit along roads, a few hundred
metres apart, and traffic moves between them: a queue that forms at a
bottleneck reaches the detector upstream a few minutes later. The natural
picture is a **graph**. Detectors are nodes, road links are edges, and an
edge may carry a weight such as the distance between its ends.

A convolution (chapter 8) mixes a pixel with its neighbours in a fixed 3×3
window. A graph has no such window: node 7 may have two neighbours and node 8
may have nine. A **graph neural network** mixes each node with its own
neighbours, whatever their number. Written with the adjacency matrix A
(A[i, j] ≠ 0 when j is a neighbour of i), one step of mixing is a matrix
product:

```
H'[i] = Σ_j A[i, j] · H[j]        i.e.   H' = A · H
```

where row H[j] holds the features of node j. This is **message passing**:
each node sends its features along its edges, and each node adds up what it
receives. Stacking k such steps lets information travel k edges.

## 11.2 Sparse matrices

The adjacency matrix of a road network is almost all zeros. PEMS08 has 170
detectors and 295 directed edges: 295 of 28,900 entries, 1%. Storing it dense
wastes memory, and a dense product A·H wastes 99% of its multiply-adds on
zeros.

XiNN stores it in **compressed sparse row** (CSR) form: three arrays.

```
values    the non-zero entries, row by row
columns   the column of each entry
offsets   row i's entries are values[offsets[i] .. offsets[i+1])
```

For the matrix

```
    [ 0  1  0 ]
A = [ 2  0  3 ]        values  = [1, 2, 3]
    [ 0  0  0 ]        columns = [1, 0, 2]
                       offsets = [0, 1, 3, 3]
```

`SparseMatrix<T>::from_edges(rows, cols, edges)` builds it from a list of
`Edge{from, to, weight}` with a counting sort by row, then sorts each row by
column and adds repeated entries. A product A·X then costs O(nnz · F) instead
of O(n² · F).

A sparse matrix is a **constant**. It is never trained, so it is not a
`Param`, and backward() never asks for its gradient. Copies share the arrays
through a `shared_ptr`, as tensors do (chapter 1).

## 11.3 spmm and its gradient

`spmm(A, X)` computes A·X for X of shape (N, F) or, batch item by batch item,
(B, N, F). It is a structural operation (chapter 2) with its own `eval`: one
task per group of output rows, over all batch items, on `parallel_for`
(chapter 6):

```cpp
for (std::size_t k = off[i]; k < off[i + 1]; ++k) {   // the stored entries of row i
    const T v = val[k];
    const T* xr = in + col[k] * F;
    for (std::size_t f = 0; f < F; ++f) out[f] += v * xr[f];
}
```

The inner loop runs over contiguous features, so it vectorizes. Rows are
independent, so no two tasks write the same memory.

The gradient follows from Y = A·X as for matmul (chapter 3): ∂L/∂X = Aᵀ·G.
So every `SparseMatrix` keeps its transpose, built once with the matrix by
the same counting sort, and `transposed()` just swaps the two halves:

```cpp
template <std::size_t>
auto grad(const auto& g, const auto&, const auto&) const { return make_expr(SpMM{a.transposed()}, g); }
```

The gradient is itself an `spmm`, lazily built, like the gradients of
convolution in chapter 8. The tests compare spmm with a dense product, for
rank 2 and rank 3, and check the gradient against finite differences.

## 11.4 Graph convolution

Using A as it is has two problems. A node does not see itself: A[i, i] = 0.
And a node with nine neighbours adds up nine messages while a node with two
adds two, so the scale of H' follows the degree, and after a few layers the
signal explodes at hubs and fades at the ends of the road.

The **graph convolutional network** (GCN) of Kipf and Welling (2017) fixes
both:

```
Â = D^-1/2 (A + I) D^-1/2,        D = diag(row sums of A + I)
H' = σ(Â · H · W + b)
```

- **A + I** adds a self-loop, so each node keeps its own features in the
  mix.
- **D^-1/2 … D^-1/2** scales entry (i, j) by 1/√(dᵢ dⱼ). Every row becomes a
  weighted average rather than a sum, and Â stays symmetric. Its largest
  eigenvalue is exactly 1, with eigenvector D^1/2·1, so repeated mixing
  neither blows up nor dies out.
- **W** is an ordinary weight matrix on the feature axis, shared by all
  nodes, as a convolution filter is shared by all pixels.

`gcn_normalize(A)` builds Â; `symmetrize(A)` first makes a directed graph
undirected (max of A and Aᵀ). The test `gcn_normalization_properties` checks
Â entry by entry, its symmetry, the eigenvector D^1/2·1, and that power
iteration never grows a vector.

**Weights from distances.** DCRNN (Li et al., 2018) weights an edge of
length d by a Gaussian kernel, exp(−(d/σ)²), with σ the standard deviation
of all lengths, and drops weights below 0.1: close detectors count more.
`gaussian_kernel(distances)` does this.

## 11.5 Layers

```cpp
struct Forecaster : Module {
    GraphConv<{.activation = Activation::relu}> g1, g2;
    NodeDense<> out;
    Forecaster(const SparseMatrix<float>& a_hat, Rng& rng)
        : g1(a_hat, 12, 64, rng), g2(a_hat, 64, 64, rng), out(64, 3, rng) {}
    auto forward(const auto& x) const { return out(g2(g1(x))); }
};
```

`GraphConv` computes Â·(H·W) + b on (B, N, F). The product with W needs a
matrix, so H is reshaped to (B·N, F), multiplied, and reshaped back; the
`reshape` of chapter 8 shares the buffer, and its gradient is the reshape
backwards. Â·(H·W) equals (Â·H)·W; XiNN applies W first, which costs less
whenever the layer has fewer outputs than inputs, and which the self weight
below needs.

The adjacency is a member of the module, but not a `Param`. The module walks
of chapter 4 find parameters by type, so they skip it with no extra code:
`param_tensor_count<GraphConv<>> == 2`.

`NodeDense` is a `Dense` layer applied at every node. With Â = I, a
`GraphConv` is exactly a `NodeDense`, and a `Forecaster` built with
`SparseMatrix::identity(N)` is a **per-node MLP** with the same layers and
the same number of parameters. That makes the comparison below clean: the
only difference between the models is Â.

**A self weight.** Kipf's layer puts a node and its neighbours through the
same W: the node's own history is one voice in an average. When neighbours
are only weakly related, that average dilutes the one signal that matters.
With `.self_weight = true` the layer is

```
H' = σ(Â · H · W + H · W_self + b)
```

It is implemented without using H twice. The weight is [W | W_self], of shape
(F, 2F'), so one matmul gives Z = H·[W | W_self], and one structural op,
`graph_mix(Â, Z) = Â·Z[:, :F'] + Z[:, F':]`, combines the halves. Its gradient
is [Âᵀ·G | G]. (XiNN expressions are trees, not graphs of shared nodes: using
the expression H twice would evaluate, and differentiate, the layer below it
twice.)

The tests check `graph_mix` against the formula and by finite differences,
and a `GraphConv` with and without a self weight against its dense
definition and by finite differences through its input and all its
parameters.

## 11.6 The task: speed forecasting at all detectors at once

The same task is used in the chapters on recurrent networks and
transformers, so the results can be compared:

- **Data:** speed only, in km/h (PEMS08 reports mph: × 1.609344).
- **Input:** the last 12 intervals (one hour) at each detector.
- **Target:** the speed 3, 6 and 12 intervals ahead (15, 30 and 60 minutes);
  one model predicts all three.
- **Split by time:** PEMS08 days 0–49 train, days 50–61 test (62 days of 288
  intervals). Speeds are standardized with one mean and scale from the
  training days.
- **Metric:** mean absolute error (km/h) per horizon, over all targets and
  over congested targets (true speed below 64 km/h), on every test time
  step.

Here one sample is a whole time step for all detectors: X is (B, N, 12) and
Y is (B, N, 3), with N = 170 on PEMS08. Training uses every time step of the
training days (14,377 on PEMS08), mini-batches of 32, Adam with cosine decay
from 2·10⁻³, MSE loss, 10 epochs on PEMS08 and 20 on the corridor. Every
network result is the mean of three seeds; the seeds differ by at most 0.05
km/h over all targets, and by at most 1.1 km/h on the few congested ones.
The whole example (30 training runs) took 31 minutes on the development machine,
most of it on PEMS08; `speed_forecast_gnn ctm` runs the corridor alone
in about 6 minutes.

The models are those of §11.5: two 64-unit ReLU layers and a linear output,
with shared weights at every node. They differ only in the graph:

| model | Â |
|---|---|
| per-node MLP | I |
| GCN | D^-1/2 (A + I) D^-1/2 of the detector graph |
| GCN, Gaussian distances | the same, with DCRNN kernel weights (PEMS08) |
| GCN, random graph | as many edges, placed at random |
| GCN + self | the detector graph, and a separate self weight |

Two baselines: **persistence** (the speed stays as it is now) and the
**historical average** of the detector at that time of day over the
training days.

## 11.7 A corridor where the graph must help

First a case where we know the answer. The Cell Transmission Model corridor
of chapter 5 has 15 detectors, 1 km apart, in a line, and the graph links
each detector to its neighbours on the road. Queues form at the lane drop
and spread upstream, one detector after another: the detector downstream
slows first, so it carries the signal for the one behind it. The example
uses a busier version of the corridor (capacity 1800 veh/h/lane instead of
2000), so that queues form on most weekdays: 2.6% of the test targets are
congested. 60 days, 48 for training and 12 for testing.

A first check of the data: the median correlation between the 15-minute
speed changes of two neighbouring detectors is **0.34**; for random pairs it
is 0.02.

| MAE (km/h) | 15 min | 30 min | 60 min | congested 15 | 30 | 60 |
|---|---|---|---|---|---|---|
| persistence | 4.07 | 4.85 | 6.27 | 15.74 | 27.30 | 45.97 |
| historical average | 5.51 | 5.51 | 5.51 | 59.64 | 59.64 | 59.64 |
| per-node MLP | 3.51 | 4.37 | 5.55 | 18.97 | 32.09 | 49.21 |
| GCN, road neighbours | 3.22 | 3.96 | 5.17 | 16.08 | 26.57 | 43.46 |
| **GCN + self, road** | **3.12** | **3.84** | **5.11** | **13.66** | **24.14** | **41.77** |
| GCN, random graph | 3.87 | 4.39 | 5.34 | 27.45 | 33.91 | 45.63 |

- **The graph helps, at every horizon.** The GCN beats the per-node MLP by
  0.3–0.4 km/h overall and by 3–6 km/h on congested targets. The only
  difference between the two models is Â.
- **It is the road that helps, not the mixing.** With a random graph, the
  GCN is worse than the MLP at 15 minutes, overall and much more on
  congested targets (27.5 against 19.0 km/h): it mixes in detectors that say
  nothing about the queue. At no horizon does it come close to the road
  graph.
- **The self weight helps further.** At 15 minutes only this model beats
  persistence on congested targets (13.7 against 15.7 km/h); at 30 and 60
  minutes both road-graph models do. Persistence is hard to beat there
  because a queue, once present, stays for a while. The per-node MLP,
  trained with MSE, pulls its forecasts towards the uncongested speeds that
  make up 97% of the targets.
- The historical average knows the daily pattern but not today's queue. It is
  useless on congested targets.

## 11.8 PEMS08

The same models on the 170 PEMS08 detectors, with the published graph: 295
directed edges, 274 undirected pairs after `symmetrize`. The edge lengths
average 316 (presumably metres); the Gaussian kernel, with σ = 216 (their
standard deviation), keeps 132 pairs.
2.2% of the test targets are congested.

| MAE (km/h) | 15 min | 30 min | 60 min | congested 15 | 30 | 60 |
|---|---|---|---|---|---|---|
| persistence | 2.00 | 2.59 | 3.38 | **8.60** | **13.47** | **20.83** |
| historical average | 3.92 | 3.93 | 3.93 | 33.70 | 33.71 | 33.73 |
| **per-node MLP** | **1.93** | **2.50** | **3.26** | 10.17 | 17.59 | 28.25 |
| GCN, detector graph | 4.05 | 4.29 | 4.68 | 25.02 | 28.68 | 34.93 |
| GCN, Gaussian distances | 3.29 | 3.65 | 4.16 | 22.76 | 27.60 | 35.12 |
| GCN, random graph | 4.11 | 4.36 | 4.74 | 24.50 | 28.98 | 35.78 |
| GCN + self, detector graph | 1.96 | 2.53 | 3.27 | 10.42 | 17.55 | 27.67 |
| GCN + self, random graph | 1.97 | 2.53 | 3.28 | 10.52 | 17.80 | 28.10 |

**The graph does not help on PEMS08.** Read the table from the bottom:

- **With a self weight, the real graph and a random graph give the same
  result**, and both are within 0.04 km/h of the per-node MLP overall. At 60
  minutes on congested targets the real graph is 0.6 km/h better than the
  MLP, but a random graph is 0.15 better too, and the spread over seeds is
  0.3.
  That is no evidence that the graph carries information.
- **Kipf's GCN is much worse than no graph at all**: 4.05 km/h against 1.93
  at 15 minutes, twice the error of persistence. And it is equally bad with
  a random graph. The damage comes from the averaging itself, not from the
  particular edges. A PEMS08 detector has 3.2 neighbours on average, so after
  normalization its own history is only about a quarter of what reaches the
  next layer, and after two layers it is mixed with detectors two edges away.
  And PEMS08 detectors differ in their typical speed: over the training
  days, their mean speeds range from 87 to 114 km/h, and two neighbours
  differ by 3.7 km/h in the median, more than the MLP's whole error. The
  average blurs the one series that predicts best. The Gaussian kernel
  keeps half the edges and loses less, for the same reason.
- **Persistence wins on congested targets.** All networks here are trained
  with MSE on data that is 98% free flow, and they pull congested forecasts
  towards the mean. The per-node MLP beats persistence only on average, by
  0.07–0.12 km/h.

Why does the graph help on the corridor and not here? §5.5 found that the
PEMS08 graph is not a corridor: along the longest path, neighbouring speeds
were barely more correlated than random pairs, and some detectors report
identical series. The example measures the quantity that matters for
forecasting, the correlation of 15-minute speed changes:

| | graph neighbours | random pairs |
|---|---|---|
| CTM corridor | 0.34 | 0.02 |
| PEMS08 | 0.066 | 0.034 |

On the corridor a neighbour's change tells a lot about a detector's next
change. On PEMS08 it tells almost nothing. A graph neural network can only
pass on information that is there. Whatever the published graph encodes,
the speed changes of connected detectors are nearly independent at a
5-minute resolution. That is consistent with §5.5's finding that some series
are imputed rather than measured.

Two lessons carry over to any data set:

1. **Always run the per-node model and a random-graph ablation.** A GCN that
   beats persistence or a historical average has not shown that the graph
   helps; only a model with everything the same but Â can show that.
2. **Kipf's layer assumes neighbours are alike** (it was designed for
   citation networks, where linked papers share topics). When a node's own
   signal matters most, give it its own weight.

## 11.9 Old C++ and new

| C++17 way | C++26 way in XiNN |
|---|---|
| a sparse-matrix class hierarchy with a virtual multiply | a value type captured by a structural op; `spmm` is one more `Expr` |
| a backward function written for each sparse layer | `grad<0>` returns another lazy `spmm`, with the transpose |
| non-trainable tensors registered by hand as "buffers" | reflection finds `Param` members by type; a `SparseMatrix` member is skipped |
| layer options as constructor flags, checked at run time | `GraphConv<{.activation = Activation::relu, .self_weight = true}>`; an absent bias costs nothing |
| one overload per input rank | `if constexpr` on the rank; the op's rank is `(R + ...)` of its operand |
| OpenMP pragmas over rows | `parallel_for` on `std::execution` |

## 11.10 Exercises

1. **Direction.** Queues travel upstream, but Â is symmetric. DCRNN diffuses
   along the road with two random-walk matrices, D_out⁻¹A and D_in⁻¹Aᵀ, each
   with its own weight. Implement it with two `spmm`s and compare with GCN
   + self on the corridor.
2. **Depth.** Add a third graph layer, so information travels three edges.
   Does the corridor gain at 60 minutes, where the queue comes from further
   away?
3. **A learned graph.** On PEMS08, link each detector to the 3 detectors
   whose speed changes on the training days correlate most with its own.
   Does this graph beat the published one, and a random one?
4. **Loss.** Train with an MAE loss (add `abs` with its gradient, chapter 3).
   Do congested errors fall towards those of persistence?
5. **Normalization.** Replace D^-1/2 (A + I) D^-1/2 with the row average
   D⁻¹(A + I). Check in a test that its rows sum to 1 and that it is not
   symmetric. Does it change the results?
6. **Cost.** Time `spmm` against a dense `matmul` with the same matrix for
   N = 170 and N = 2000 nodes, at 3 neighbours each. Where is the break-even
   point?

## References

- T. N. Kipf and M. Welling, "Semi-supervised classification with graph
  convolutional networks," ICLR 2017.
- Y. Li, R. Yu, C. Shahabi and Y. Liu, "Diffusion convolutional recurrent
  neural network: data-driven traffic forecasting," ICLR 2018.
