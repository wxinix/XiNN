// SPDX-License-Identifier: BSD-3-Clause
// Transformer layers (Vaswani et al. 2017), for sequences of shape
// (batch, tokens, features).
//
//   TokenDense<Opt>        a Dense layer applied to every token
//   LayerNorm              normalize each token's features; learned gain and bias
//   Embedding              token ids -> rows of a learned table
//   MultiHeadAttention     softmax(Q·K^T / sqrt(dh) [+ mask]) · V, in h heads
//   TransformerBlock       pre-LayerNorm:  x + MHA(LN(x)),  then  x + FFN(LN(x))
#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "xinn/attention.hpp"
#include "xinn/conv.hpp"
#include "xinn/init.hpp"
#include "xinn/layers.hpp"
#include "xinn/linalg.hpp"
#include "xinn/loss.hpp"
#include "xinn/module.hpp"
#include "xinn/shared.hpp"

namespace xinn {

// y = activation(x·W + b) for every token: (..., in) -> (..., out). The
// leading axes are folded into one, so it is one matrix product.
template <DenseOptions Opt = {}, class T = float>
struct TokenDense : Module {
    Param<T, 2> weight;
    [[no_unique_address]] std::conditional_t<Opt.bias, Param<T, 1>, Nothing> bias;

    TokenDense(std::size_t in, std::size_t out, Rng& rng)
        : weight(Opt.activation == Activation::relu ? he_normal<T>(in, out, rng) : glorot_uniform<T>(in, out, rng)),
          bias(make_bias(out)) {}

    template <TensorArg X>
    auto forward(X&& x) const {
        constexpr std::size_t R = std::remove_cvref_t<X>::rank;
        Shape<R> out = x.shape();
        out.dims[R - 1] = weight.shape()[1];
        const Shape<2> rows{x.shape().count() / x.shape()[R - 1], x.shape()[R - 1]};
        auto y = matmul(reshape(std::forward<X>(x), rows), weight);
        if constexpr (Opt.bias) return reshape(activate<Opt.activation>(std::move(y) + bias), out);
        else return reshape(activate<Opt.activation>(std::move(y)), out);
    }

private:
    static auto make_bias(std::size_t out) {
        if constexpr (Opt.bias) return Param<T, 1>(Vector<T>(Shape{out}));
        else return Nothing{};
    }
};

template <class T = float>
struct LayerNorm : Module {
    Param<T, 1> gain, bias;
    explicit LayerNorm(std::size_t d) : gain(Vector<T>(Shape{d}, T{1})), bias(Vector<T>(Shape{d})) {}

    template <TensorArg X>
    auto forward(X&& x) const { return layer_norm(std::forward<X>(x), gain, bias); }
};

template <class T = float>
struct Embedding : Module {
    Param<T, 2> table;   // (vocab, d)
    Embedding(std::size_t vocab, std::size_t d, Rng& rng) : table(randn<T>(Shape{vocab, d}, rng)) {}

    template <std::size_t R>
    auto forward(const Tensor<std::size_t, R>& ids) const { return embedding(table, ids); }
};

// (B, T, d) -> (B, T, d). Each head attends with its own dh = d / h
// features; splitting into heads is a reshape and a permutation, not a copy
// per head.
template <class T = float>
struct MultiHeadAttention : Module {
    TokenDense<{}, T> query, key, value, output;
    std::size_t heads;
    bool causal;

    MultiHeadAttention(std::size_t d_model, std::size_t heads_, Rng& rng, bool causal_ = false)
        pre(heads_ > 0 && d_model % heads_ == 0)
        : query(d_model, d_model, rng), key(d_model, d_model, rng), value(d_model, d_model, rng),
          output(d_model, d_model, rng), heads(heads_), causal(causal_) {}

    template <TensorArg X>
        requires(std::remove_cvref_t<X>::rank == 3)
    auto forward(const X& x) const {
        const std::size_t B = x.shape()[0], S = x.shape()[1], d = x.shape()[2], dh = d / heads;
        // (B, S, d) -> (B, S, h, dh) -> (B, h, S, dh) -> (B*h, S, dh)
        auto split = [&](auto&& y) {
            return reshape(permute<0, 2, 1, 3>(reshape(std::forward<decltype(y)>(y), Shape{B, S, heads, dh})),
                           Shape{B * heads, S, dh});
        };
        auto q = split(query(x)), k = split(key(x)), v = split(value(x));
        const T scale = T(1) / std::sqrt(T(dh));
        const Matrix<T> mask = causal ? causal_mask<T>(S) : Matrix<T>(Shape{S, S});   // (S, S), broadcast
        auto weights = softmax(bmm<false, true>(q, k) * scale + mask);                // (B*h, S, S)
        auto context = bmm(weights, v);                                               // (B*h, S, dh)
        // (B*h, S, dh) -> (B, h, S, dh) -> (B, S, h, dh) -> (B, S, d)
        return output(reshape(permute<0, 2, 1, 3>(reshape(context, Shape{B, heads, S, dh})), Shape{B, S, d}));
    }
};

struct TransformerOptions {
    std::size_t d_model = 64;
    std::size_t heads = 4;
    std::size_t ffn = 4;   // hidden width of the feed-forward part, as a multiple of d_model
    bool causal = false;
};

// Pre-LayerNorm block (Xiong et al. 2020): the residual path x stays
// un-normalized, which makes deep stacks train without warm-up tricks.
template <class T = float>
struct TransformerBlock : Module {
    LayerNorm<T> norm1;
    MultiHeadAttention<T> attention;
    LayerNorm<T> norm2;
    TokenDense<{.activation = Activation::relu}, T> ffn1;
    TokenDense<{}, T> ffn2;

    TransformerBlock(TransformerOptions o, Rng& rng)
        : norm1(o.d_model), attention(o.d_model, o.heads, rng, o.causal), norm2(o.d_model),
          ffn1(o.d_model, o.ffn * o.d_model, rng), ffn2(o.ffn * o.d_model, o.d_model, rng) {}

    // x is used twice in each residual step, so the steps are shared
    // (xinn/shared.hpp): each is evaluated once, whatever depth it sits at.
    template <TensorArg X>
        requires(std::remove_cvref_t<X>::rank == 3)
    auto forward(const X& x_in) const {
        const auto x = share(x_in);
        const auto h = share(x + attention(norm1(x)));
        return share(h + ffn2(ffn1(norm2(h))));
    }
};

}  // namespace xinn
