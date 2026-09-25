// SPDX-License-Identifier: BSD-3-Clause
// The GPU backend.
//
// Kernel generation and the scheduler are tested everywhere: neither needs a
// GPU. The tests that run kernels need an OpenCL device; without one they
// print "skipped: no OpenCL platform" and pass.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <print>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <xinn/xinn.hpp>
#include <xinn/gpu.hpp>

using namespace xinn;

// A user-defined op with its own OpenCL code.
struct Clamp01 {
    static constexpr std::string_view opencl = "clamp($0, 0.0f, 1.0f)";
    float operator()(float a) const { return std::clamp(a, 0.0f, 1.0f); }
};

namespace {

Rng rng{7};

// Opened once. Every runtime test asks here and skips if there is no device.
const std::optional<gpu::Device>& device() {
    static const std::optional<gpu::Device> dev = [] -> std::optional<gpu::Device> {
        auto d = gpu::Device::open();
        if (!d) {
            std::println("    OpenCL: {}", d.error().message);
            return std::nullopt;
        }
        std::println("    OpenCL: {} ({}, {})", d->name(), d->platform(), d->version());
        return *d;
    }();
    if (!dev) std::println("    skipped: no OpenCL platform");
    return dev;
}

template <class T, std::size_t R>
double max_abs_diff(const Tensor<T, R>& x, const Tensor<T, R>& y) {
    if (x.shape() != y.shape()) return INFINITY;
    double d = 0;
    for (std::size_t i = 0; i < x.size(); ++i) d = std::max(d, std::abs(double(x.flat()[i]) - double(y.flat()[i])));
    return d;
}

// ---- a fake OpenCL driver ------------------------------------------------------------
//
// It keeps buffers in host memory and records what the backend asks of it,
// but compiles and runs nothing. That tests everything around the kernels
// (argument order, sizes, build options, the cache, errors, releases) on a
// machine without a GPU.
namespace fake {

using namespace gpu::cl;

struct Arg {
    cl_uint index;
    std::vector<std::byte> bytes;
};
struct Launch {
    cl_uint dims;
    std::size_t global[2], local[2];
};

struct Driver {
    int live_buffers = 0, programs = 0, released_programs = 0, contexts = 0;
    std::vector<std::string> options, kernel_names;
    std::vector<Arg> args;
    std::vector<Launch> launches;
} driver;

template <class T> T* handle(std::uintptr_t v) { return reinterpret_cast<T*>(v); }

cl_int XINN_CL_CALL platform_ids(cl_uint n, cl_platform_id* ps, cl_uint* count) {
    if (ps && n > 0) ps[0] = handle<_cl_platform_id>(1);
    if (count) *count = 1;
    return CL_SUCCESS;
}
cl_int info(const std::string& s, std::size_t size, void* value, std::size_t* ret) {
    if (ret) *ret = s.size() + 1;
    if (value) std::copy_n(s.c_str(), std::min(size, s.size() + 1), static_cast<char*>(value));
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL platform_info(cl_platform_id, cl_platform_info, std::size_t size, void* value, std::size_t* ret) {
    return info("Fake platform", size, value, ret);
}
cl_int XINN_CL_CALL device_ids(cl_platform_id, cl_device_type, cl_uint n, cl_device_id* ds, cl_uint* count) {
    if (ds && n > 0) ds[0] = handle<_cl_device_id>(2);
    if (count) *count = 1;
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL device_info(cl_device_id, cl_device_info what, std::size_t size, void* value, std::size_t* ret) {
    if (what == CL_DEVICE_MAX_WORK_GROUP_SIZE) {
        if (value) *static_cast<std::size_t*>(value) = 256;
        return CL_SUCCESS;
    }
    return info(what == CL_DEVICE_NAME ? "Fake GPU" : "OpenCL 3.0 fake", size, value, ret);
}
cl_context XINN_CL_CALL create_context(const cl_context_properties*, cl_uint, const cl_device_id*,
                                       Api::ContextCallback, void*, cl_int* rc) {
    ++driver.contexts;
    *rc = CL_SUCCESS;
    return handle<_cl_context>(3);
}
cl_command_queue XINN_CL_CALL create_queue(cl_context, cl_device_id, cl_command_queue_properties, cl_int* rc) {
    *rc = CL_SUCCESS;
    return handle<_cl_command_queue>(4);
}
cl_mem XINN_CL_CALL create_buffer(cl_context, cl_mem_flags, std::size_t bytes, void*, cl_int* rc) {
    ++driver.live_buffers;
    *rc = CL_SUCCESS;
    return reinterpret_cast<cl_mem>(new std::vector<std::byte>(bytes));
}
std::vector<std::byte>& memory(cl_mem m) { return *reinterpret_cast<std::vector<std::byte>*>(m); }
cl_program XINN_CL_CALL create_program(cl_context, cl_uint, const char** text, const std::size_t* length,
                                       cl_int* rc) {
    *rc = CL_SUCCESS;
    return reinterpret_cast<cl_program>(new std::string(text[0], length[0]));
}
cl_int XINN_CL_CALL build(cl_program p, cl_uint, const cl_device_id*, const char* opts, Api::BuildCallback, void*) {
    ++driver.programs;
    driver.options.emplace_back(opts);
    return reinterpret_cast<std::string*>(p)->contains("#error") ? -11 : CL_SUCCESS;
}
cl_int XINN_CL_CALL build_info(cl_program, cl_device_id, cl_program_build_info, std::size_t size, void* value,
                               std::size_t* ret) {
    return info("<source>:1:2: error: nope", size, value, ret);
}
cl_kernel XINN_CL_CALL create_kernel(cl_program, const char* name, cl_int* rc) {
    driver.kernel_names.emplace_back(name);
    *rc = CL_SUCCESS;
    return handle<_cl_kernel>(5);
}
cl_int XINN_CL_CALL set_arg(cl_kernel, cl_uint index, std::size_t size, const void* value) {
    const auto* p = static_cast<const std::byte*>(value);
    driver.args.push_back({index, std::vector<std::byte>(p, p + size)});
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL launch(cl_command_queue, cl_kernel, cl_uint dims, const std::size_t*, const std::size_t* global,
                           const std::size_t* local, cl_uint, const cl_event*, cl_event*) {
    Launch l{dims, {global[0], dims > 1 ? global[1] : 1}, {local[0], dims > 1 ? local[1] : 1}};
    driver.launches.push_back(l);
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL read(cl_command_queue, cl_mem m, cl_bool, std::size_t offset, std::size_t size, void* dst,
                         cl_uint, const cl_event*, cl_event*) {
    std::copy_n(memory(m).data() + offset, size, static_cast<std::byte*>(dst));
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL write(cl_command_queue, cl_mem m, cl_bool, std::size_t offset, std::size_t size,
                          const void* src, cl_uint, const cl_event*, cl_event*) {
    std::copy_n(static_cast<const std::byte*>(src), size, memory(m).data() + offset);
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL finish(cl_command_queue) { return CL_SUCCESS; }
cl_int XINN_CL_CALL release_mem(cl_mem m) {
    --driver.live_buffers;
    delete &memory(m);
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL release_kernel(cl_kernel) { return CL_SUCCESS; }
cl_int XINN_CL_CALL release_program(cl_program p) {
    ++driver.released_programs;
    delete reinterpret_cast<std::string*>(p);
    return CL_SUCCESS;
}
cl_int XINN_CL_CALL release_queue(cl_command_queue) { return CL_SUCCESS; }
cl_int XINN_CL_CALL release_context(cl_context) {
    --driver.contexts;
    return CL_SUCCESS;
}

const Api api{platform_ids, platform_info, device_ids, device_info, create_context, create_queue, create_buffer,
              create_program, build, build_info, create_kernel, set_arg, launch, read, write, finish,
              release_mem, release_kernel, release_program, release_queue, release_context};

template <class T>
T arg_value(std::size_t k) {
    T v;
    std::memcpy(&v, driver.args.at(k).bytes.data(), sizeof v);
    return v;
}

}  // namespace fake

}  // namespace

namespace tests {

// ---- code generation (no GPU needed) -------------------------------------------------

void fused_kernel_source_for_a_dense_layer() {
    auto x = Matrix<float>(Shape(4, 3)), w = Matrix<float>(Shape(4, 3));
    auto b = Vector<float>(Shape(3));
    using E = decltype(sigmoid(x * w + b));
    static_assert(gpu::kernel_name<E> == "xinn_sigmoid_add_mul");
    check::equal(gpu::kernel_source<E>, std::string_view(R"(__kernel void xinn_sigmoid_add_mul(__global float* out, const uint n,
    __global const float* a0, const uint n0,
    __global const float* a1, const uint n1,
    __global const float* a2, const uint n2)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v0 = a0[i % n0];
    const float v1 = a1[i % n1];
    const float v2 = v0 * v1;
    const float v3 = a2[i % n2];
    const float v4 = v2 + v3;
    const float v5 = 1.0f / (1.0f + exp(-v4));
    out[i] = v5;
}
)"));
}

void constants_become_arguments_and_filled_values_literals() {
    auto x = Matrix<float>(Shape(2, 5));
    auto b = Vector<float>(Shape(5));
    // Numbers in the expression are Constants (run-time values); expand()'s
    // Zeros operand carries its value in its type.
    using E = decltype((x - 2) * 0.5f + expand(b, x.shape()));
    check::equal(gpu::kernel_source<E>, std::string_view(R"(__kernel void xinn_add_mul_sub_expand(__global float* out, const uint n,
    __global const float* a0, const uint n0,
    const float c1,
    const float c2,
    __global const float* a3, const uint n3)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v0 = a0[i % n0];
    const float v1 = v0 - c1;
    const float v2 = v1 * c2;
    const float v3 = a3[i % n3];
    const float v4 = v3;
    const float v5 = v2 + v4;
    out[i] = v5;
}
)"));

    using F = decltype(-make_expr(ops::Mul{}, x, Filled<float, 0, -0.75f>(Shape<0>{})));
    check::equal(gpu::kernel_source<F>, std::string_view(R"(__kernel void xinn_neg_mul(__global float* out, const uint n,
    __global const float* a0, const uint n0)
{
    const uint i = get_global_id(0);
    if (i >= n) return;
    const float v0 = a0[i % n0];
    const float v1 = v0 * (-0x1.8p-1f);
    const float v2 = -v1;
    out[i] = v2;
}
)"));
}

void float_literals_are_exact() {
    using gpu::detail::float_literal;
    static_assert(float_literal(0.0f) == "0.0f");
    static_assert(float_literal(1.0f) == "1.0f");
    static_assert(float_literal(-2.0f) == "(-2.0f)");
    static_assert(float_literal(0.5f) == "0x1p-1f");
    static_assert(float_literal(0.75f) == "0x1.8p-1f");
    static_assert(float_literal(0.1f) == "0x1.99999ap-4f");
    static_assert(float_literal(1e30f) == "0x1.93e594p+99f");
    static_assert(float_literal(16777216.0f) == "0x1p+24f");     // 2^24: whole, but past the exact-integer range
    check::that(true);
}

void a_structural_operand_is_a_buffer() {
    auto x = Matrix<float>(Shape(4, 3)), w = Matrix<float>(Shape(3, 2));
    auto b = Vector<float>(Shape(2));
    using E = decltype(relu(matmul(x, w) + b));   // matmul runs first, then the fused kernel reads it
    static_assert(gpu::kernel_name<E> == "xinn_relu_add");
    check::that(gpu::kernel_source<E>.contains("const float v3 = fmax(v2, 0.0f);"));
    check::that(gpu::kernel_source<E>.contains("__global const float* a0, const uint n0,\n"
                                               "    __global const float* a1, const uint n1)"));
}

void user_ops_bring_their_own_code() {
    auto x = Vector<float>(Shape(8));
    using E = decltype(map(Clamp01{}, x * 3));
    static_assert(gpu::kernel_name<E> == "xinn_clamp01_mul");
    check::that(gpu::kernel_source<E>.contains("const float v2 = clamp(v1, 0.0f, 1.0f);"));
}

void long_names_are_shortened_with_a_hash() {
    auto x = Vector<float>(Shape(8));
    using E = decltype(tanh(sigmoid(exp(tanh(sigmoid(exp(tanh(sigmoid(exp(sqrt(x)))))))))));
    constexpr std::string_view name = gpu::kernel_name<E>;
    static_assert(name.size() == 49 && name.starts_with("xinn_tanh_sigmoid_exp_tanh_sigmoid_exp_t_"));
    check::that(gpu::kernel_source<E>.starts_with("__kernel void " + std::string(name) + "("));
}

void the_matmul_kernel_tiles_in_local_memory() {
    check::that(gpu::matmul_source.contains("__local float As[TS][TS];"));
    check::that(gpu::matmul_source.contains("barrier(CLK_LOCAL_MEM_FENCE);"));
}

void the_blocked_kernel_keeps_a_block_of_c_in_registers() {
    const auto& src = gpu::matmul_blocked_source;
    check::that(src.contains("float acc[WPT][WPT]"));
    check::that(src.contains("return vload4(0, p);"));   // with VEC, tiles arrive four floats at a time
    check::that(src.contains("__local float As[TSK][TS + 1];"));   // padded against bank conflicts
    check::that(src.contains("__kernel void xinn_matmul_blocked("));
}

// ---- the OpenCL library (no GPU needed) ------------------------------------------------

void no_device_is_an_error_not_a_crash() {
    const auto& api = gpu::cl::api();
    if (!api) std::println("    no OpenCL library: {}", api.error().message);
    else std::println("    OpenCL library loaded");
    auto dev = gpu::Device::open();
    if (!dev) {
        check::that(!dev.error().message.empty());
        std::println("    Device::open(): {}", dev.error().message);
    } else {
        check::that(!dev->name().empty());
    }
}

// ---- the backend on a fake driver (no GPU needed) ------------------------------------

void fake_device_copies_both_ways() {
    fake::driver = {};
    {
        auto dev = gpu::Device::open(fake::api);
        check::that(dev.has_value());
        check::equal(dev->name(), std::string_view("Fake GPU"));
        check::equal(dev->platform(), std::string_view("Fake platform"));
        auto x = randn<float>(Shape(3, 4), rng);
        auto dx = gpu::upload(*dev, x);
        check::equal(fake::driver.live_buffers, 1);
        check::equal(max_abs_diff(dx->eval(), x), 0.0);
        auto empty = gpu::upload(*dev, Matrix<float>(Shape(0, 4)));   // no OpenCL buffer for no data
        check::equal(fake::driver.live_buffers, 1);
        check::equal(empty->eval().size(), 0uz);
    }
    check::equal(fake::driver.live_buffers, 0);   // everything released
    check::equal(fake::driver.contexts, 0);
}

void fake_fused_kernel_gets_its_arguments_in_source_order() {
    fake::driver = {};
    auto dev = gpu::Device::open(fake::api);
    auto x = randn<float>(Shape(30, 40), rng);
    auto b = randn<float>(Shape(40), rng);
    auto e = (x - 2) * 0.5f + b;
    auto y = gpu::run(*dev, e);
    check::that(y.has_value());
    check::equal(y->shape(), x.shape());
    // out, n, a0, n0, c1, c2, a3, n3 -- as in the generated source
    using E = decltype(e);
    check::that(gpu::kernel_source<E>.contains("(__global float* out, const uint n,\n"
                                               "    __global const float* a0, const uint n0,\n"
                                               "    const float c1,\n    const float c2,\n"
                                               "    __global const float* a3, const uint n3)"));
    check::equal(fake::driver.args.size(), 8uz);
    for (std::size_t k = 0; k < fake::driver.args.size(); ++k) check::equal(fake::driver.args[k].index, gpu::cl::cl_uint(k));
    check::equal(fake::arg_value<gpu::cl::cl_uint>(1), 1200u);
    check::equal(fake::arg_value<gpu::cl::cl_uint>(3), 1200u);
    check::equal(fake::arg_value<float>(4), 2.0f);
    check::equal(fake::arg_value<float>(5), 0.5f);
    check::equal(fake::arg_value<gpu::cl::cl_uint>(7), 40u);   // b wraps every 40 elements
    check::equal(fake::arg_value<gpu::cl::cl_mem>(0), y->buffer().handle());
    check::equal(fake::driver.kernel_names.at(0), std::string(gpu::kernel_name<E>));
    check::equal(fake::driver.launches.at(0).dims, 1u);
    check::equal(fake::driver.launches.at(0).global[0], 1280uz);   // 1200 rounded up to groups of 256
    check::equal(fake::driver.launches.at(0).local[0], 256uz);

    auto again = gpu::run(*dev, (x - 3) * 0.25f + b);   // same type: the program is cached
    check::that(again.has_value());
    check::equal(fake::driver.programs, 1);
    check::equal(fake::driver.launches.size(), 2uz);
}

void fake_matmul_sets_sizes_and_flags() {
    fake::driver = {};
    auto dev = gpu::Device::open(fake::api);
    auto a = randn<float>(Shape(50, 20), rng), b = randn<float>(Shape(30, 50), rng);
    // matmul(a^T, b^T) with a stored 50x20 and b stored 30x50: C is 20 x 30, K = 50.
    auto c = gpu::run(*dev, relu(matmul(transpose(a), transpose(b))));
    check::that(c.has_value());
    check::equal(c->shape(), Shape(20, 30));
    check::equal(fake::driver.options.at(0), std::string("-D RTS=16 -D WPT=4 -D VEC=0 -D TRANS_A=1 -D TRANS_B=1"));
    check::equal(fake::driver.kernel_names.at(0), std::string("xinn_matmul_blocked"));
    check::equal(fake::arg_value<gpu::cl::cl_uint>(0), 20u);   // M
    check::equal(fake::arg_value<gpu::cl::cl_uint>(1), 30u);   // N
    check::equal(fake::arg_value<gpu::cl::cl_uint>(2), 50u);   // K
    const auto& l = fake::driver.launches.at(0);
    // One 64 x 64 tile covers C: one group of 16 x 16 items, 4 x 4 elements each.
    check::that(l.dims == 2 && l.global[0] == 16 && l.global[1] == 16 && l.local[0] == 16 && l.local[1] == 16);
    check::equal(fake::driver.kernel_names.at(1), std::string("xinn_relu"));   // then the fused relu
}

void fake_matmul_can_choose_its_kernel() {
    fake::driver = {};
    auto dev = gpu::Device::open(fake::api);
    auto a = randn<float>(Shape(20, 50), rng), b = randn<float>(Shape(50, 130), rng);
    auto tiled = gpu::run_matmul(*dev, a, b, gpu::MatmulKernel::tiled);
    auto blocked = gpu::run_matmul(*dev, a, b);
    auto big = gpu::run_matmul(*dev, a, b, gpu::MatmulKernel::blocked_8x8_float4);
    check::that(tiled.has_value() && blocked.has_value() && big.has_value());
    check::equal(fake::driver.kernel_names.at(0), std::string("xinn_matmul"));
    check::equal(fake::driver.options.at(0), std::string("-D TS=16 -D TRANS_A=0 -D TRANS_B=0"));
    const auto& t = fake::driver.launches.at(0);   // one item per element: 144 x 32
    check::that(t.global[0] == 144 && t.global[1] == 32);
    const auto& k = fake::driver.launches.at(1);   // 3 x 1 tiles of 64 x 64
    check::that(k.global[0] == 48 && k.global[1] == 16);
    check::equal(fake::driver.options.at(2), std::string("-D RTS=16 -D WPT=8 -D VEC=1 -D TRANS_A=0 -D TRANS_B=0"));
    const auto& g = fake::driver.launches.at(2);   // 2 x 1 tiles of 128 x 128
    check::that(g.global[0] == 32 && g.global[1] == 16 && g.local[0] == 16 && g.local[1] == 16);

    // automatic: 8 x 8 blocks from a 512 x 512 result on, 4 x 4 below.
    auto x = randn<float>(Shape(512, 3), rng), y = randn<float>(Shape(3, 512), rng), y1 = randn<float>(Shape(3, 511), rng);
    check::that(gpu::run_matmul(*dev, x, y).has_value() && gpu::run_matmul(*dev, x, y1).has_value());
    check::that(fake::driver.options.at(3).contains("-D WPT=8 -D VEC=0"));
    const auto& big8 = fake::driver.launches.at(3);    // 4 x 4 tiles of 128 x 128
    const auto& small4 = fake::driver.launches.at(4);  // 8 x 8 tiles of 64 x 64 (the cached 4 x 4 program)
    check::that(big8.global[0] == 64 && big8.global[1] == 64);
    check::that(small4.global[0] == 128 && small4.global[1] == 128);
}

void fake_build_errors_carry_the_log() {
    fake::driver = {};
    auto dev = gpu::Device::open(fake::api);
    auto guard = dev->lock();
    auto k = dev->kernel("#error broken", "k");
    check::that(!k.has_value());
    check::that(k.error().message.contains("CL_BUILD_PROGRAM_FAILURE"));
    check::that(k.error().message.contains("error: nope"));
    check::equal(k.error().code, -11);
    check::equal(fake::driver.released_programs, 1);   // the failed program is not leaked
}

// ---- the scheduler (no GPU needed) -----------------------------------------------------

void the_queue_scheduler_models_the_concept() {
    namespace ex = gpu::ex;
    using S = gpu::Queue::Scheduler;
    static_assert(ex::scheduler<S>);
    static_assert(ex::sender<decltype(ex::schedule(std::declval<S>()))>);
    static_assert(ex::sender_of<decltype(ex::schedule(std::declval<S>())), ex::set_value_t()>);

    gpu::Queue q;
    check::that(q.scheduler() == q.scheduler());
    gpu::Queue other;
    check::that(q.scheduler() != other.scheduler());
    // The sender says where it completes.
    auto s = ex::schedule(q.scheduler());
    check::that(ex::get_completion_scheduler<ex::set_value_t>(ex::get_env(s)) == q.scheduler());
}

void work_runs_on_the_queue_thread_in_order() {
    namespace ex = gpu::ex;
    gpu::Queue q;
    auto sched = q.scheduler();
    auto [id] = xinn::detail::sync_wait(ex::schedule(sched) | ex::then([] { return std::this_thread::get_id(); }))
                    .value();
    check::that(id == q.thread_id());
    check::that(id != std::this_thread::get_id());

    std::vector<int> order;   // touched only by the queue's thread
    auto step = [&](int k) { return ex::schedule(sched) | ex::then([&order, k] { order.push_back(k); }); };
    xinn::detail::sync_wait(ex::when_all(step(0), step(1), step(2), step(3)));
    check::that(order == std::vector{0, 1, 2, 3});
}

void async_eval_without_a_device_reports_it() {
    gpu::Queue q;   // no device
    auto x = Vector<float>(Shape(4), 1.0f);
    auto [y] = xinn::detail::sync_wait(gpu::async_eval(q.scheduler(), x * 2)).value();
    check::that(!y.has_value());
    check::equal(y.error().message, std::string("the queue has no device"));
}

// ---- on a device ---------------------------------------------------------------

void fused_kernels_match_the_cpu() {
    const auto& dev = device();
    if (!dev) return;
    auto x = randn<float>(Shape(300, 257), rng), w = randn<float>(Shape(300, 257), rng);
    auto b = randn<float>(Shape(257), rng);
    auto s = randn<float>(Shape<0>{}, rng);
    auto check_same = [&](const auto& e) {
        auto got = gpu::eval(*dev, e);
        if (!got) {
            std::println(stderr, "    {}", got.error().message);
            check::that(false);
            return;
        }
        check::near(max_abs_diff(*got, e.eval()), 0, 1e-5);
    };
    check_same(sigmoid(x * w + b));
    check_same(tanh(x) * exp(-square(w)) / 2 + relu(x - b));
    check_same((x - 2) * 0.5f + expand(b, x.shape()));
    check_same(x * s + sqrt(square(w) + 1) - log(exp(b)));   // a rank-0 tensor broadcast everywhere
    check_same(map(Clamp01{}, x));
    check_same(b * 3);                                       // a vector, rank 1
}

void matmul_matches_the_cpu() {
    const auto& dev = device();
    if (!dev) return;
    using enum gpu::MatmulKernel;
    for (auto kind : {tiled, blocked_4x4, blocked_4x4_float4, blocked_8x8, blocked_8x8_float4})
        for (auto [m, k, n] : {std::array{1uz, 1uz, 1uz}, {3, 5, 7}, {16, 16, 16}, {17, 33, 15}, {97, 300, 50},
                               {130, 2, 200}, {64, 64, 64}, {65, 17, 129}, {257, 100, 63}}) {
            auto a = randn<float>(Shape{m, k}, rng), b = randn<float>(Shape{k, n}, rng);
            auto got = gpu::matmul(*dev, a, b, kind);
            check::that(got.has_value());
            if (got) check::near(max_abs_diff(*got, Matrix<float>(matmul(a, b))), 0, 1e-4 * double(k));
        }
    auto a = randn<float>(Shape(37, 70), rng), b = randn<float>(Shape(70, 45), rng);
    const Matrix<float> at = transpose(a).eval(), bt = transpose(b).eval(), want = matmul(a, b);
    for (auto got : {gpu::eval(*dev, matmul(transpose(at), b)), gpu::eval(*dev, matmul(a, transpose(bt))),
                     gpu::eval(*dev, matmul(transpose(at), transpose(bt)))}) {
        check::that(got.has_value());
        if (got) check::near(max_abs_diff(*got, want), 0, 1e-3);
    }
    // Every kernel with every transpose combination: the float4 loads differ for each.
    auto a2 = randn<float>(Shape(130, 37), rng), b2 = randn<float>(Shape(37, 75), rng);
    const Matrix<float> want2 = matmul(a2, b2);
    auto da = gpu::upload(*dev, a2), db = gpu::upload(*dev, b2);
    auto dat = gpu::upload(*dev, Matrix<float>(transpose(a2))), dbt = gpu::upload(*dev, Matrix<float>(transpose(b2)));
    check::that(da && db && dat && dbt);
    if (!da || !db || !dat || !dbt) return;
    auto same = [&](auto got) {
        check::that(got.has_value());
        if (got) check::near(max_abs_diff(*got->download(), want2), 0, 1e-3);
    };
    for (auto kind : {tiled, blocked_4x4, blocked_4x4_float4, blocked_8x8, blocked_8x8_float4}) {
        same(gpu::detail::run_matmul<false, false>(*dev, *da, *db, kind));
        same(gpu::detail::run_matmul<true, false>(*dev, *dat, *db, kind));
        same(gpu::detail::run_matmul<false, true>(*dev, *da, *dbt, kind));
        same(gpu::detail::run_matmul<true, true>(*dev, *dat, *dbt, kind));
    }
}

void a_layer_runs_on_the_device_and_stays_there() {
    const auto& dev = device();
    if (!dev) return;
    auto x = randn<float>(Shape(64, 30), rng), w = randn<float>(Shape(30, 20), rng);
    auto b = randn<float>(Shape(20), rng);
    auto dx = gpu::upload(*dev, x), dw = gpu::upload(*dev, w);
    auto db = gpu::upload(*dev, b);
    check::that(dx && dw && db);
    if (!dx || !dw || !db) return;
    auto h = gpu::run(*dev, relu(matmul(*dx, *dw) + *db));   // no host copies in between
    check::that(h.has_value());
    if (!h) return;
    const Matrix<float> want = relu(matmul(x, w) + b);
    check::near(max_abs_diff(h->eval(), want), 0, 1e-4);
    check::near(max_abs_diff(Matrix<float>(*h * 2), Matrix<float>(want * 2)), 0, 1e-4);   // CPU math on a device tensor
}

void the_queue_drives_the_device() {
    const auto& dev = device();
    if (!dev) return;
    gpu::Queue q(*dev);
    auto x = randn<float>(Shape(1000), rng);
    auto [y] = xinn::detail::sync_wait(gpu::async_eval(q.scheduler(), sigmoid(x))).value();
    check::that(y.has_value());
    if (y) check::near(max_abs_diff(*y, sigmoid(x).eval()), 0, 1e-6);
}

}  // namespace tests

int main() { return check::run_tests<^^tests>(); }
