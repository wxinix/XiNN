// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <string>
#include <vector>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

Rng rng{1};

struct MLP : Module {
    Dense<{.activation = Activation::relu}> fc1{2, 8, rng};
    Dense<> fc2{8, 3, rng};
    auto forward(const auto& x) const { return fc2(fc1(x)); }
};

struct Autoencoder : Module {
    MLP encoder;
    Dense<{.bias = false}> decoder{3, 2, rng};
    auto forward(const auto& x) const { return decoder(encoder(x)); }
};

// A physical model is a module too: two named scalar parameters.
struct Drake : Module {
    Param<float, 0> vf{Scalar<float>(Shape<0>{}, 1.1f)};
    Param<float, 0> kc{Scalar<float>(Shape<0>{}, 0.35f)};
    auto forward(const auto& k) const { return vf * exp(square(k / kc) * -0.5f); }
};

template <ModuleType M>
std::vector<std::string> names(const M& m) {
    std::vector<std::string> out;
    m.for_each_param([&](const std::string& name, const auto&) { out.push_back(name); });
    return out;
}

}  // namespace

namespace tests {

void parameters_are_found_by_reflection() {
    MLP net;
    check::equal(names(net), std::vector<std::string>{"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"});
    check::equal(net.parameter_count(), 2uz * 8 + 8 + 8 * 3 + 3);
}

void nested_modules_get_dotted_names() {
    Autoencoder ae;
    check::equal(names(ae), std::vector<std::string>{"encoder.fc1.weight", "encoder.fc1.bias", "encoder.fc2.weight",
                                                     "encoder.fc2.bias", "decoder.weight"});
}

void options_shape_the_type() {
    using NoBias = Dense<{.bias = false}>;
    static_assert(std::same_as<decltype(NoBias::bias), Nothing>);
    static_assert(param_tensor_count<NoBias> == 1);
    static_assert(param_tensor_count<MLP> == 4);
    static_assert(param_tensor_count<Autoencoder> == 5);
    static_assert(sizeof(NoBias) == sizeof(Param<float, 2>));   // an absent bias takes no space
}

void calling_a_module_calls_forward() {
    MLP net;
    Matrix<float> x(Shape(5, 2), 0.5f);
    auto y = net(x);
    check::equal(y.shape(), Shape(5, 3));
    check::that(describe(y).starts_with("(matmul(relu((matmul(T[5x2], P[2x8]) + P[8]))"));
}

void sequential_numbers_its_layers() {
    Sequential net{Dense<>{2, 4, rng}, ReLU{}, Dense<>{4, 1, rng}};
    check::equal(names(net), std::vector<std::string>{"0.weight", "0.bias", "2.weight", "2.bias"});
    static_assert(param_tensor_count<decltype(net)> == 4);
    check::equal(net(Matrix<float>(Shape(3, 2))).shape(), Shape(3, 1));
}

void zero_grad_reaches_every_parameter() {
    MLP net;
    Matrix<float> x(Shape(4, 2), 1.0f);
    backward(sum(net(x)));
    bool any = false;
    net.for_each_param([&](const std::string&, const auto& p) {
        for (float g : p.grad().flat()) any |= g != 0;
    });
    check::that(any);

    net.zero_grad();
    net.for_each_param([&](const std::string&, const auto& p) {
        for (float g : p.grad().flat()) check::equal(g, 0.0f);
    });
}

void summary_lists_parameters() {
    Drake d;
    const std::string s = d.summary();
    check::that(s.contains("vf"));
    check::that(s.contains("kc"));
    check::that(s.contains("total"));
    check::equal(d.parameter_count(), 2uz);
}

void a_physical_model_trains_as_a_module() {
    Drake d;
    Vector<float> k(Shape(3), {0.1f, 0.35f, 0.7f});
    backward(sum(d(k)));
    check::that(d.vf.grad()() != 0.0f);
    check::that(d.kc.grad()() != 0.0f);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
