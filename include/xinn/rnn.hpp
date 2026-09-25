// SPDX-License-Identifier: BSD-3-Clause
// Recurrent networks: a GRU, trained by backpropagation through time.
//
//   GRU<> gru(in, hidden, rng);
//   auto hs = gru(xs);                        // xs: (steps, batch, in); hs[0..steps]
//   Param h_last(hs.back());                  // cut the graph at the last state
//   backward(mse(head(h_last), y));           // the head's gradients, and dLoss/dh_last
//   gru.bptt(xs, hs, h_last.grad());          // the GRU's gradients
//
// One step of the GRU is an expression with a fixed type. The number of
// steps is known only at run time, so the sequence is never one expression:
// forward() is a loop that keeps the hidden states as tensors, and bptt() is
// a loop in the other direction that differentiates one step at a time, by
// a vector-Jacobian product (backward(expr, seed)).
#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include "xinn/autograd.hpp"
#include "xinn/expr.hpp"
#include "xinn/init.hpp"
#include "xinn/linalg.hpp"
#include "xinn/module.hpp"
#include "xinn/ops.hpp"
#include "xinn/param.hpp"
#include "xinn/tensor.hpp"

namespace xinn {

namespace ops {

// blend(z, n, h) = n + z (h - n) = (1 - z) n + z h: the GRU's update, a mix
// of the candidate n and the old state h, weighted by the gate z. One op
// instead of (1 - z) * n + z * h, so z appears once in the expression and
// is computed (and differentiated) once.
struct Blend {
    constexpr auto operator()(auto z, auto n, auto h) const { return n + z * (h - n); }
    template <std::size_t I>
    auto grad(const auto& g, const auto&, const auto& z, const auto& n, const auto& h) const {
        if constexpr (I == 0) return g * (h - n);
        else if constexpr (I == 1) return g * (1 - z);
        else return g * z;
    }
};

}  // namespace ops

template <TensorArg Z, TensorArg N, TensorArg H>
auto blend(Z&& z, N&& n, H&& h) {
    return make_expr(ops::Blend{}, std::forward<Z>(z), std::forward<N>(n), std::forward<H>(h));
}

// A batch of sequences, (batch, steps * in) with one sequence per row, in
// time-major order (steps, batch, in): the layout the GRU reads, where the
// inputs of one step are contiguous and xs[t] is a view, not a copy.
template <class T>
Tensor<T, 3> time_major(const Matrix<T>& rows, std::size_t in = 1)
    pre(in > 0 && rows.shape()[1] % in == 0)
{
    const std::size_t batch = rows.shape()[0], steps = rows.shape()[1] / in;
    Tensor<T, 3> out(Shape{steps, batch, in});
    auto src = rows.view();
    auto dst = out.mut();
    for (std::size_t b = 0; b < batch; ++b)
        for (std::size_t t = 0; t < steps; ++t)
            for (std::size_t f = 0; f < in; ++f) dst[t, b, f] = src[b, t * in + f];
    return out;
}

// Gated recurrent unit (Cho et al., 2014), for x_t of shape (batch, in) and
// h of shape (batch, hidden):
//
//   z  = sigmoid(x Wz + h Uz + bz)          update gate
//   r  = sigmoid(x Wr + h Ur + br)          reset gate
//   n  = tanh(x Wn + r * (h Un) + bn)       candidate state
//   h' = (1 - z) * n + z * h
//
// The reset gate multiplies h Un, after the product, as in cuDNN and
// PyTorch; the original paper applies it before, n = tanh(x Wn + (r*h) Un).
template <class T = float>
struct GRU : Module {
    Param<T, 2> Wz, Wr, Wn;   // input weights (in, hidden)
    Param<T, 2> Uz, Ur, Un;   // recurrent weights (hidden, hidden)
    Param<T, 1> bz, br, bn;

    GRU(std::size_t in, std::size_t hidden, Rng& rng)
        : Wz(glorot_uniform<T>(in, hidden, rng)), Wr(glorot_uniform<T>(in, hidden, rng)),
          Wn(glorot_uniform<T>(in, hidden, rng)), Uz(glorot_uniform<T>(hidden, hidden, rng)),
          Ur(glorot_uniform<T>(hidden, hidden, rng)), Un(glorot_uniform<T>(hidden, hidden, rng)),
          bz(Vector<T>(Shape{hidden})), br(Vector<T>(Shape{hidden})), bn(Vector<T>(Shape{hidden})) {}

    std::size_t input_size() const { return Wz.shape()[0]; }
    std::size_t hidden_size() const { return Uz.shape()[0]; }

    // One step, as a lazy expression. h may be a tensor or a Param.
    template <TensorArg X, TensorArg H>
    auto step(const X& x, const H& h) const {
        auto z = sigmoid(matmul(x, Wz) + matmul(h, Uz) + bz);
        auto r = sigmoid(matmul(x, Wr) + matmul(h, Ur) + br);
        auto n = tanh(matmul(x, Wn) + r * matmul(h, Un) + bn);
        return blend(std::move(z), std::move(n), h);
    }

    // Run over xs, of shape (steps, batch, in), from h0 (zeros if empty).
    // Returns the hidden states h_0, h_1, ..., h_steps. Nothing is recorded.
    std::vector<Matrix<T>> forward(const Tensor<T, 3>& xs, Matrix<T> h0 = {}) const
        pre(xs.shape()[2] == input_size())
    {
        const std::size_t steps = xs.shape()[0];
        if (h0.empty()) h0 = Matrix<T>(Shape{xs.shape()[1], hidden_size()});
        std::vector<Matrix<T>> hs;
        hs.reserve(steps + 1);
        hs.push_back(std::move(h0));
        for (std::size_t t = 0; t < steps; ++t) {
            Matrix<T> next = step(xs[t], hs.back()).eval();
            hs.push_back(std::move(next));
        }
        return hs;
    }

    // Backpropagation through time. hs is what forward(xs) returned, and
    // dh[t] = dLoss/dh_{t+1} is the gradient that reaches h_{t+1} from
    // outside the GRU (an empty tensor for none). Adds dLoss/dW to every
    // weight's gradient, and returns dLoss/dh_0.
    //
    // Step t is differentiated on its own: h_t becomes a fresh Param leaf,
    // the step's expression is rebuilt (and re-evaluated) on it, and a
    // vector-Jacobian product with g = dLoss/dh_{t+1} fills the weights'
    // gradients and the leaf's -- which is dLoss/dh_t, the seed of step t-1.
    Matrix<T> bptt(const Tensor<T, 3>& xs, std::span<const Matrix<T>> hs, std::span<const Matrix<T>> dh) const
        pre(hs.size() == xs.shape()[0] + 1 && dh.size() == xs.shape()[0])
    {
        Matrix<T> g(hs[0].shape());
        for (std::size_t t = xs.shape()[0]; t-- > 0;) {
            if (!dh[t].empty()) g = g + dh[t];
            const Param<T, 2> h(hs[t]);
            backward(step(xs[t], h), g);
            g = h.grad();
        }
        return g;
    }

    // The common case: the loss sees only the last state.
    Matrix<T> bptt(const Tensor<T, 3>& xs, std::span<const Matrix<T>> hs, const Matrix<T>& dh_last) const
        pre(xs.shape()[0] > 0 && hs.size() == xs.shape()[0] + 1)
    {
        std::vector<Matrix<T>> dh(xs.shape()[0]);
        dh.back() = dh_last;
        return bptt(xs, hs, std::span<const Matrix<T>>(dh));
    }
};

}  // namespace xinn
