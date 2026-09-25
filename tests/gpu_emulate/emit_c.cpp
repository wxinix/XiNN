// SPDX-License-Identifier: BSD-3-Clause
// Writes the generated OpenCL kernels as C programs that check themselves.
//
//   gpu_emit_c <dir>
//
// Each file holds the kernel source exactly as the GPU would receive it,
// after prelude.h (a little OpenCL C in plain C), the inputs, the result the
// CPU computed, and a main() that runs every work item and compares. run.cmake
// compiles and runs them. So the generated source is compiled and executed,
// on a machine without a GPU.
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <string>
#include <vector>

#include <xinn/xinn.hpp>
#include <xinn/gpu.hpp>

using namespace xinn;
namespace gd = xinn::gpu::detail;

namespace {

std::string literal(float v) { return std::format("{:.9e}f", v); }   // 9 digits: exact for float

std::string array(const std::string& name, std::span<const float> v) {
    std::string s = "static float " + name + "[] = {";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i % 6 ? " " : "\n    ") + literal(v[i]) + ",";
    return s + "\n};\n";
}

// The kernel's arguments after (out, n), in the order codegen.hpp numbers
// the leaves, as C: arrays for the data, and the call's argument list.
template <class X>
void leaves(const X& x, std::string& data, std::string& call, int& k) {
    if constexpr (gd::FusedNode<X>) {
        std::apply([&](const auto&... a) { (leaves(a, data, call, k), ...); }, x.args());
    } else if constexpr (gd::leaf_kind<X>() == gd::Leaf::literal) {
    } else if constexpr (gd::leaf_kind<X>() == gd::Leaf::constant) {
        call += ", " + literal(float(x.value()));
        ++k;
    } else {
        const auto t = x.eval();
        data += array("a" + std::to_string(k), t.flat());
        call += std::format(", a{}, {}u", k, t.size());
        ++k;
    }
}

template <class E>
void write_fused(const std::filesystem::path& file, const E& e) {
    const auto want = e.eval();
    std::string data, call;
    int k = 0;
    leaves(e, data, call, k);
    const std::size_t n = want.size(), global = (n + 255) / 256 * 256;   // as run_fused rounds it
    std::ofstream out(file);
    out << "#include \"prelude.h\"\n\n" << gpu::kernel_source<E> << "\n" << data << array("want", want.flat())
        << std::format("static float out[{}];\n\n", n)
        << std::format("static void item(void) {{ {}(out, {}u{}); }}\n\n", gpu::kernel_name<E>, n, call)
        << std::format("int main(void) {{\n    xcl_launch_1d(item, {});\n    return xcl_compare(out, want, {}, 1e-5f);\n}}\n",
                       global, n);
}

// side: the work group is side x side items. wpt 1 is the tiled kernel, one
// element of C per item; wpt 4 or 8 the blocked one, wpt x wpt per item,
// loading float4s if vec. The launch is as run_matmul's.
template <bool TA, bool TB>
void write_matmul(const std::filesystem::path& file, const Matrix<float>& a, const Matrix<float>& b,
                  std::size_t side, std::size_t wpt = 1, bool vec = false) {
    const bool blocked = wpt > 1;
    auto op = [](const Matrix<float>& m, bool t) { return t ? Matrix<float>(transpose(m)) : m; };
    const Matrix<float> want = matmul(op(a, TA), op(b, TB));   // a and b are stored as the kernel reads them
    const std::size_t m = want.shape()[0], n = want.shape()[1], kk = TA ? a.shape()[0] : a.shape()[1];
    const std::size_t tile = side * wpt;
    std::ofstream out(file);
    out << (blocked ? std::format("#define RTS {}\n#define WPT {}\n#define VEC {}\n", side, wpt, int(vec))
                    : std::format("#define TS {}\n", side))
        << std::format("#define TRANS_A {}\n#define TRANS_B {}\n", int(TA), int(TB))
        << "#include \"prelude.h\"\n" << (blocked ? gpu::matmul_blocked_source : gpu::matmul_source) << "\n"
        << array("A", a.flat()) << array("B", b.flat()) << array("want", want.flat())
        << std::format("static float C[{}];\n\n", m * n)
        << std::format("static void item(void) {{ {}({}u, {}u, {}u, A, B, C); }}\n\n",
                       blocked ? "xinn_matmul_blocked" : "xinn_matmul", m, n, kk)
        << std::format("int main(void) {{\n    xcl_launch_2d(item, {}, {}, {}, {});\n"
                       "    return xcl_compare(C, want, {}, 1e-5f);\n}}\n",
                       (n + tile - 1) / tile * side, (m + tile - 1) / tile * side, side, side, m * n);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::println(stderr, "usage: gpu_emit_c <dir>");
        return 2;
    }
    const std::filesystem::path dir = argv[1];
    std::filesystem::create_directories(dir);
    Rng rng{11};
    auto x = randn<float>(Shape(37, 29), rng), w = randn<float>(Shape(37, 29), rng);
    auto b = randn<float>(Shape(29), rng);
    auto s = randn<float>(Shape<0>{}, rng);

    write_fused(dir / "fused_dense.c", sigmoid(x * w + b));
    write_fused(dir / "fused_mixed.c", tanh(x) * exp(-square(w)) / 2 + relu(x - b));
    write_fused(dir / "fused_constants.c", (x - 2) * 0.5f + expand(b, x.shape()) + s);
    write_fused(dir / "fused_literal.c", -make_expr(ops::Mul{}, x, Filled<float, 0, -0.75f>(Shape<0>{})));
    write_fused(dir / "fused_log_sqrt.c", log(sqrt(square(x) + 1)) - b / 3);

    auto a = randn<float>(Shape(37, 70), rng), bb = randn<float>(Shape(70, 45), rng);
    const Matrix<float> at = transpose(a).eval(), bt = transpose(bb).eval();
    write_matmul<false, false>(dir / "matmul_nn.c", a, bb, 16);
    write_matmul<true, false>(dir / "matmul_tn.c", at, bb, 16);
    write_matmul<false, true>(dir / "matmul_nt.c", a, bt, 8);
    write_matmul<true, true>(dir / "matmul_tt.c", at, bt, 8);

    // Blocked, in each variant: several groups each way, ragged edges, and K,
    // M and N that are not multiples of 4, so float4 loads meet the edges.
    auto a2 = randn<float>(Shape(130, 37), rng), b2 = randn<float>(Shape(37, 75), rng);
    const Matrix<float> a2t = transpose(a2).eval(), b2t = transpose(b2).eval();
    for (std::size_t wpt : {4uz, 8uz})
        for (bool vec : {false, true}) {
            const std::string v = std::format("matmul_blocked{}{}_", wpt, vec ? "v" : "");
            write_matmul<false, false>(dir / (v + "nn.c"), a2, b2, 16, wpt, vec);
            write_matmul<true, false>(dir / (v + "tn.c"), a2t, b2, 8, wpt, vec);
            write_matmul<false, true>(dir / (v + "nt.c"), a2, b2t, 16, wpt, vec);
            write_matmul<true, true>(dir / (v + "tt.c"), a2t, b2t, 8, wpt, vec);
            write_matmul<false, false>(dir / (v + "small.c"), a, bb, 16, wpt, vec);   // less than one tile
        }
    std::println("wrote the kernels to {}", dir.string());
}
