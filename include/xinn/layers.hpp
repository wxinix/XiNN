// SPDX-License-Identifier: BSD-3-Clause
// Ready-made layers.
//
//   Dense<{.activation = Activation::relu}> fc{in, out, rng};   // y = relu(x·W + b)
//   Dense<{.bias = false}>                  proj{in, out, rng}; // y = x·W
//   Sequential net{Dense<>{2, 16, rng}, ReLU{}, Dense<>{16, 1, rng}};
//
// A layer's options are a struct passed as a template argument (C++20
// class-type template parameters), written with designated initializers.
// Options are part of the type, so the compiler specializes the code, and
// an absent bias costs neither memory nor work.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xinn/conv.hpp"
#include "xinn/init.hpp"
#include "xinn/linalg.hpp"
#include "xinn/module.hpp"
#include "xinn/ops.hpp"
#include "xinn/param.hpp"

namespace xinn {

enum class Activation { none, relu, tanh, sigmoid };

template <Activation A, class X>
auto activate(X&& x) {
    if constexpr (A == Activation::relu) return relu(std::forward<X>(x));
    else if constexpr (A == Activation::tanh) return tanh(std::forward<X>(x));
    else if constexpr (A == Activation::sigmoid) return sigmoid(std::forward<X>(x));
    else return std::remove_cvref_t<X>(std::forward<X>(x));
}

struct DenseOptions {
    bool bias = true;
    Activation activation = Activation::none;
};

// A fully connected layer: y = activation(x·W + b), x of shape (batch, in).
template <DenseOptions Opt = {}, class T = float>
struct Dense : Module {
    static constexpr DenseOptions options = Opt;

    Param<T, 2> weight;
    [[no_unique_address]] std::conditional_t<Opt.bias, Param<T, 1>, Nothing> bias;

    Dense(std::size_t in, std::size_t out, Rng& rng)
        : weight(Opt.activation == Activation::relu ? he_normal<T>(in, out, rng) : glorot_uniform<T>(in, out, rng)),
          bias(make_bias(out)) {}

    template <TensorArg X>
    auto forward(X&& x) const {
        if constexpr (Opt.bias) return activate<Opt.activation>(matmul(std::forward<X>(x), weight) + bias);
        else return activate<Opt.activation>(matmul(std::forward<X>(x), weight));
    }

private:
    static auto make_bias(std::size_t out) {
        if constexpr (Opt.bias) return Param<T, 1>(Vector<T>(Shape{out}));
        else return Nothing{};
    }
};

// Activations as parameter-free modules, for use in Sequential.
struct ReLU : Module {
    auto forward(TensorArg auto&& x) const { return relu(std::forward<decltype(x)>(x)); }
};
struct Tanh : Module {
    auto forward(TensorArg auto&& x) const { return tanh(std::forward<decltype(x)>(x)); }
};
struct Sigmoid : Module {
    auto forward(TensorArg auto&& x) const { return sigmoid(std::forward<decltype(x)>(x)); }
};

// Layers applied in order. Parameters are named by position: "0.weight".
template <ModuleType... Layers>
    requires(sizeof...(Layers) >= 1)
struct Sequential : Module {
    std::tuple<Layers...> layers;

    explicit Sequential(Layers... ls) : layers(std::move(ls)...) {}

    template <std::size_t I>
    const auto& at() const { return std::get<I>(layers); }

    template <class X>
    auto forward(X&& x) const { return apply_from<0>(std::forward<X>(x)); }

    // The layers, for the module walks. Children without member names are
    // named by position.
    template <class Self>
    auto children(this Self& self) {
        return std::apply([](auto&... l) { return std::tie(l...); }, self.layers);
    }

    static constexpr std::size_t param_tensors = (detail::count_params<Layers>() + ...);

private:
    template <std::size_t I, class X>
    auto apply_from(X&& x) const {
        if constexpr (I == sizeof...(Layers)) return std::remove_cvref_t<X>(std::forward<X>(x));
        else return apply_from<I + 1>(std::get<I>(layers)(std::forward<X>(x)));
    }
};

template <class... Layers>
Sequential(Layers...) -> Sequential<Layers...>;

// ---- convolutional layers (chapter 8) ----------------------------------------------------------

struct Conv2DOptions {
    std::size_t kernel = 3;
    std::size_t stride = 1;
    std::size_t padding = 0;
    Activation activation = Activation::none;
};

// y = activation(conv2d(x, W, b)), x of shape (N, in_channels, H, W).
template <Conv2DOptions Opt = {}, class T = float>
struct Conv2D : Module {
    static constexpr Conv2DOptions options = Opt;

    Param<T, 4> weight;   // (out_channels, in_channels, kernel, kernel)
    Param<T, 1> bias;     // (out_channels)

    Conv2D(std::size_t in_channels, std::size_t out_channels, Rng& rng)
        : weight(randn<T>(Shape{out_channels, in_channels, Opt.kernel, Opt.kernel}, rng, T{0},
                          static_cast<T>(std::sqrt(2.0 / double(in_channels * Opt.kernel * Opt.kernel))))),
          bias(Vector<T>(Shape{out_channels})) {}

    template <TensorArg X>
    auto forward(X&& x) const {
        return activate<Opt.activation>(conv2d(std::forward<X>(x), weight, bias, Opt.stride, Opt.padding));
    }
};

template <std::size_t Size = 2>
struct MaxPool2D : Module {
    auto forward(TensorArg auto&& x) const { return max_pool2d<Size>(std::forward<decltype(x)>(x)); }
};

// (N, C, H, W) -> (N, C * H * W), for the dense layers that follow.
struct Flatten : Module {
    template <TensorArg X>
    auto forward(X&& x) const {
        const auto& s = x.shape();
        std::size_t rest = 1;
        for (std::size_t d = 1; d < std::remove_cvref_t<X>::rank; ++d) rest *= s[d];
        return reshape(std::forward<X>(x), Shape{s[0], rest});
    }
};

}  // namespace xinn
