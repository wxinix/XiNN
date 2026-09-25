// SPDX-License-Identifier: BSD-3-Clause
// A minimal GPU backend: forward evaluation with OpenCL.
//
//   auto dev = gpu::Device::open();                     // std::expected<Device, Error>
//   auto y = gpu::eval(*dev, sigmoid(matmul(x, w) + b));   // std::expected<Matrix<float>, Error>
//
// gpu::run(dev, x) computes x on the device and leaves the result there;
// gpu::eval(dev, x) also copies it back. What runs where:
//
//   a Tensor                   copied to the device
//   a DeviceTensor             used as it is
//   an element-wise tree       one fused kernel, generated from its type (codegen.hpp)
//   matmul(a, b)               the register-blocked matmul kernel; transposes are flags, as on the CPU
//   anything else              evaluated on the CPU, then copied
//
// A tree's leaves are computed first by the same rules, so a matmul inside
// an element-wise tree runs on the GPU too. Only float is supported. There is
// no automatic differentiation on the GPU: this is forward evaluation.
//
// Opt-in: not part of <xinn/xinn.hpp>, because it loads a system library and
// most programs never need it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <expected>
#include <format>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "xinn/expr.hpp"
#include "xinn/gpu/codegen.hpp"
#include "xinn/gpu/device.hpp"
#include "xinn/gpu/opencl.hpp"
#include "xinn/gpu/scheduler.hpp"
#include "xinn/linalg.hpp"

namespace xinn::gpu {

template <TensorArg X>
auto run(const Device& dev, const X& x) -> std::expected<DeviceTensor<float, std::remove_cvref_t<X>::rank>, Error>;

// The matmul kernels of codegen.hpp: tiled (one element of C per work item),
// and register-blocked with 4 x 4 or 8 x 8 elements per item, loading tiles
// one float or four (float4) at a time. `automatic`, which gpu::run uses,
// picks what was fastest on the RTX 3060 of chapter 7: 8 x 8 for results of
// 512 x 512 elements and more, 4 x 4 for smaller ones, whose 128 x 128 tiles
// would be too few to keep the GPU busy. gpu::matmul can choose, to compare.
enum class MatmulKernel {
    automatic,
    tiled,
    blocked_4x4,
    blocked_4x4_float4,
    blocked_8x8,
    blocked_8x8_float4,
};

namespace detail {

template <class Op> struct matmul_flags { static constexpr bool is = false; };
template <bool A, bool B> struct matmul_flags<ops::MatMul<A, B>> {
    static constexpr bool is = true, trans_a = A, trans_b = B;
};

template <class X>
concept MatMulNode = requires { typename X::op_type; } && matmul_flags<typename X::op_type>::is;

// One kernel argument.
using Arg = std::variant<cl::cl_mem, cl::cl_uint, float>;

inline std::expected<void, Error> set_args(const Device& dev, cl::cl_kernel k, const std::vector<Arg>& args) {
    for (cl::cl_uint i = 0; i < args.size(); ++i) {
        const cl::cl_int rc = std::visit([&](const auto& v) { return dev.api().SetKernelArg(k, i, sizeof v, &v); },
                                         args[i]);
        if (rc != cl::CL_SUCCESS) return std::unexpected(cl::error("clSetKernelArg", rc));
    }
    return {};
}

inline std::expected<void, Error> enqueue(const Device& dev, cl::cl_kernel k, cl::cl_uint dims,
                                          const std::size_t* global, const std::size_t* local) {
    const cl::cl_int rc = dev.api().EnqueueNDRangeKernel(dev.queue(), k, dims, nullptr, global, local, 0, nullptr,
                                                         nullptr);
    if (rc != cl::CL_SUCCESS) return std::unexpected(cl::error("clEnqueueNDRangeKernel", rc));
    return {};
}

inline std::size_t round_up(std::size_t n, std::size_t m) { return (n + m - 1) / m * m; }

// The leaves' arguments, in the order codegen.hpp numbers them. Buffers are
// kept in `keep` until the kernel has been queued; after that, OpenCL itself
// keeps a released buffer alive until the commands that use it are done.
template <class X>
std::expected<void, Error> bind(const Device& dev, const X& x, std::vector<Arg>& args,
                                std::vector<Buffer>& keep) {
    if constexpr (FusedNode<X>) {
        std::expected<void, Error> ok;
        std::apply([&](const auto&... a) { ((ok = ok ? bind(dev, a, args, keep) : ok), ...); }, x.args());
        return ok;
    } else if constexpr (leaf_kind<X>() == Leaf::literal) {
        return {};   // written into the source
    } else if constexpr (leaf_kind<X>() == Leaf::constant) {
        args.emplace_back(float(x.value()));
        return {};
    } else {
        auto d = gpu::run(dev, x);
        if (!d) return std::unexpected(d.error());
        args.emplace_back(d->buffer().handle());
        args.emplace_back(cl::cl_uint(d->size()));
        keep.push_back(d->buffer());
        return {};
    }
}

template <class X>
auto run_fused(const Device& dev, const X& x) -> std::expected<DeviceTensor<float, X::rank>, Error> {
    auto out = DeviceTensor<float, X::rank>::allocate(dev, x.shape());
    if (!out || out->size() == 0) return out;
    contract_assert(out->size() <= 0xFFFFFFFFuz);   // indices are 32-bit uints in the kernel
    std::vector<Arg> args{out->buffer().handle(), cl::cl_uint(out->size())};
    std::vector<Buffer> keep;
    if (auto ok = bind(dev, x, args, keep); !ok) return std::unexpected(ok.error());

    auto guard = dev.lock();
    auto k = dev.kernel(kernel_source<X>, kernel_name<X>);
    if (!k) return std::unexpected(k.error());
    if (auto ok = set_args(dev, *k, args); !ok) return std::unexpected(ok.error());
    const std::size_t local = std::min<std::size_t>(256, dev.max_work_group_size());
    const std::size_t global = round_up(out->size(), local);
    if (auto ok = enqueue(dev, *k, 1, &global, &local); !ok) return std::unexpected(ok.error());
    return out;
}

template <bool TransA, bool TransB>
auto run_matmul(const Device& dev, const DeviceTensor<float, 2>& a, const DeviceTensor<float, 2>& b,
                MatmulKernel kind = MatmulKernel::automatic) -> std::expected<DeviceTensor<float, 2>, Error> {
    const std::size_t m = TransA ? a.shape()[1] : a.shape()[0], k = TransA ? a.shape()[0] : a.shape()[1];
    const std::size_t n = TransB ? b.shape()[0] : b.shape()[1];
    contract_assert(k == (TransB ? b.shape()[1] : b.shape()[0]));   // inner dimensions agree
    auto out = DeviceTensor<float, 2>::allocate(dev, Shape(m, n));
    if (!out || out->size() == 0) return out;
    if (kind == MatmulKernel::automatic)
        kind = m * n >= 512 * 512 ? MatmulKernel::blocked_8x8 : MatmulKernel::blocked_4x4;

    // Work groups of 16 x 16 items need 256 per group; smaller devices get 8 x 8.
    // The tiled kernel covers one element of C per item, the blocked ones
    // wpt x wpt, so a group covers a tile of side * wpt.
    const std::size_t side = dev.max_work_group_size() >= 256 ? 16 : 8;
    const bool blocked = kind != MatmulKernel::tiled;
    const std::size_t wpt = !blocked ? 1
                            : kind == MatmulKernel::blocked_8x8 || kind == MatmulKernel::blocked_8x8_float4 ? 8
                                                                                                            : 4;
    const bool vec = kind == MatmulKernel::blocked_4x4_float4 || kind == MatmulKernel::blocked_8x8_float4;
    const std::size_t tile = side * wpt;
    const std::string flags = std::string(" -D TRANS_A=") + (TransA ? "1" : "0") + " -D TRANS_B=" + (TransB ? "1" : "0");
    const std::string options = blocked ? std::format("-D RTS={} -D WPT={} -D VEC={}{}", side, wpt, int(vec), flags)
                                        : std::format("-D TS={}{}", side, flags);
    const std::vector<Arg> args{cl::cl_uint(m), cl::cl_uint(n), cl::cl_uint(k), a.buffer().handle(),
                                b.buffer().handle(), out->buffer().handle()};
    auto guard = dev.lock();
    auto kern = blocked ? dev.kernel(matmul_blocked_source, "xinn_matmul_blocked", options)
                        : dev.kernel(matmul_source, "xinn_matmul", options);
    if (!kern) return std::unexpected(kern.error());
    if (auto ok = set_args(dev, *kern, args); !ok) return std::unexpected(ok.error());
    const std::size_t global[2] = {(n + tile - 1) / tile * side, (m + tile - 1) / tile * side};
    const std::size_t local[2] = {side, side};
    if (auto ok = enqueue(dev, *kern, 2, global, local); !ok) return std::unexpected(ok.error());
    return out;
}

}  // namespace detail

// Compute x on the device; the result stays there.
template <TensorArg X>
auto run(const Device& dev, const X& x) -> std::expected<DeviceTensor<float, std::remove_cvref_t<X>::rank>, Error> {
    using B = std::remove_cvref_t<X>;
    static_assert(std::same_as<typename B::value_type, float>, "xinn::gpu computes in float");
    if constexpr (is_device_tensor_v<B>) {
        return x;
    } else if constexpr (is_tensor_v<B>) {
        return upload(dev, x);
    } else if constexpr (detail::FusedNode<B>) {
        return detail::run_fused(dev, x);
    } else if constexpr (detail::MatMulNode<B>) {
        using F = detail::matmul_flags<typename B::op_type>;
        auto a = run(dev, x.template arg<0>());
        if (!a) return a;
        auto b = run(dev, x.template arg<1>());
        if (!b) return b;
        return detail::run_matmul<F::trans_a, F::trans_b>(dev, *a, *b);
    } else {
        return upload(dev, x.eval());   // not a GPU operation: compute it on the CPU
    }
}

// Compute x on the device and copy the result back.
template <TensorArg X>
auto eval(const Device& dev, const X& x) -> std::expected<Tensor<float, std::remove_cvref_t<X>::rank>, Error> {
    return run(dev, x).and_then([](const auto& d) { return d.download(); });
}

// C = A · B with the chosen kernel; the result stays on the device.
template <MatrixArg A, MatrixArg B>
auto run_matmul(const Device& dev, const A& a, const B& b, MatmulKernel kind = MatmulKernel::automatic)
    -> std::expected<DeviceTensor<float, 2>, Error> {
    auto da = run(dev, a);
    if (!da) return da;
    auto db = run(dev, b);
    if (!db) return db;
    return detail::run_matmul<false, false>(dev, *da, *db, kind);
}

// C = A · B on the device, host tensors in and out.
template <MatrixArg A, MatrixArg B>
auto matmul(const Device& dev, const A& a, const B& b, MatmulKernel kind = MatmulKernel::automatic)
    -> std::expected<Matrix<float>, Error> {
    return run_matmul(dev, a, b, kind).and_then([](const auto& d) { return d.download(); });
}

// Wait until everything queued on the device has finished (for timing).
inline std::expected<void, Error> finish(const Device& dev) {
    auto guard = dev.lock();
    if (cl::cl_int rc = dev.api().Finish(dev.queue()); rc != cl::CL_SUCCESS)
        return std::unexpected(cl::error("clFinish", rc));
    return {};
}

#if XINN_PARALLEL
// eval() as a sender on a Queue's scheduler: the work is submitted from the
// queue's thread. Completes with std::expected<Tensor, Error>.
template <TensorArg X>
auto async_eval(Queue::Scheduler sched, X x) {
    return ex::then(ex::schedule(sched), [sched, x = std::move(x)] {
        using R = decltype(eval(*sched.device(), x));
        if (!sched.device()) return R(std::unexpected(Error{"the queue has no device"}));
        return eval(*sched.device(), x);
    });
}
#endif

}  // namespace xinn::gpu
