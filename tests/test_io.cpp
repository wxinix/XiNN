// SPDX-License-Identifier: BSD-3-Clause
#include "check.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <xinn/xinn.hpp>

using namespace xinn;

namespace {

struct Net : Module {
    Dense<{.activation = Activation::tanh}> fc1;
    Dense<{.bias = false}> fc2;
    Net(std::size_t in, std::size_t hidden, std::size_t out, Rng& rng) : fc1(in, hidden, rng), fc2(hidden, out, rng) {}
    auto forward(const auto& x) const { return fc2(fc1(x)); }
};

std::filesystem::path temp_file(std::string_view name) {
    return std::filesystem::temp_directory_path() / std::format("xinn_{}", name);
}

std::string header_of(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::uint64_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof n);
    std::string h(n, '\0');
    in.read(h.data(), std::streamsize(n));
    return h;
}

}  // namespace

namespace tests {

void round_trip_restores_every_value() {
    Rng rng_a{1}, rng_b{2};
    Net a(3, 5, 2, rng_a), b(3, 5, 2, rng_b);   // different random weights
    const auto path = temp_file("roundtrip.safetensors");

    check::that(save(a, path).has_value());
    check::that(load(b, path).has_value());

    Matrix<float> x(Shape(4, 3), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const Matrix<float> ya = a(x), yb = b(x);
    for (std::size_t i = 0; i < ya.size(); ++i) check::equal(ya.flat()[i], yb.flat()[i]);
}

void header_follows_the_format() {
    Rng rng{3};
    Net a(3, 5, 2, rng);
    const auto path = temp_file("header.safetensors");
    check::that(save(a, path).has_value());

    const std::string h = header_of(path);
    check::equal(h.size() % 8, 0uz);   // data starts 8-byte aligned
    check::that(h.contains(R"("fc1.weight":{"dtype":"F32","shape":[3,5],"data_offsets":[0,60]})"));
    check::that(h.contains(R"("fc1.bias":{"dtype":"F32","shape":[5],"data_offsets":[60,80]})"));
    check::that(h.contains(R"("__metadata__":{)"));
    check::that(!h.contains("fc2.bias"));   // no bias, so nothing is written

    // The Python check (tests/verify_safetensors.py) reads this file too.
    std::filesystem::copy_file(path, temp_file("spec_check.safetensors"),
                               std::filesystem::copy_options::overwrite_existing);
}

void wrong_shape_is_an_error_and_changes_nothing() {
    Rng rng{4};
    Net small(3, 5, 2, rng), big(3, 6, 2, rng);
    const auto path = temp_file("shape.safetensors");
    check::that(save(small, path).has_value());

    const Matrix<float> before = big.fc1.weight.tensor();
    auto r = load(big, path);
    check::that(!r.has_value());
    check::that(r.error().contains("fc1.weight"));
    check::that(big.fc1.weight.tensor().identical(before));
}

void wrong_dtype_is_an_error() {
    struct Doubles : Module {
        Param<double, 1> w{Vector<double>(Shape(2), {1, 2})};
    };
    struct Floats : Module {
        Param<float, 1> w{Vector<float>(Shape(2), {1, 2})};
    };
    const auto path = temp_file("dtype.safetensors");
    Doubles d;
    check::that(save(d, path).has_value());
    check::that(header_of(path).contains(R"("dtype":"F64")"));

    Floats f;
    auto r = load(f, path);
    check::that(!r.has_value());
    check::that(r.error().contains("dtype F32"));
}

void missing_file_and_missing_tensor() {
    Rng rng{5};
    Net a(3, 5, 2, rng);
    auto r = load(a, temp_file("does_not_exist.safetensors"));
    check::that(!r.has_value());
    check::that(r.error().contains("cannot open"));

    struct Other : Module {
        Param<float, 1> something{Vector<float>(Shape(1))};
    };
    Other o;
    const auto path = temp_file("other.safetensors");
    check::that(save(o, path).has_value());
    auto r2 = load(a, path);
    check::that(!r2.has_value());
    check::that(r2.error().contains("'fc1.weight' not in file"));
}

void json_reader_handles_the_basics() {
    auto j = detail::JsonParser(R"( {"a": [1, 2, {"b": "x\"y"}], "c": true, "d": null} )").parse();
    check::that(j.has_value());
    check::that(j->find("a") != nullptr);
    check::that(j->find("zzz") == nullptr);
    check::that(!detail::JsonParser(R"({"a": )").parse().has_value());
    check::that(!detail::JsonParser(R"({"a": 1} x)").parse().has_value());
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
