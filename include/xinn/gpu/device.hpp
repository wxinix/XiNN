// SPDX-License-Identifier: BSD-3-Clause
// A GPU, and tensors that live in its memory.
//
//   auto dev = gpu::Device::open();          // std::expected<Device, Error>
//   if (!dev) { std::println("{}", dev.error().message); return; }
//   auto dx = gpu::upload(*dev, x);          // host -> device
//   Matrix<float> back = dx->eval();         // device -> host
//
// A Device is the OpenCL platform, device, context and command queue of the
// first GPU found (or, if there is none, the first OpenCL device of any
// kind). Copies of a Device share them, like copies of a Tensor share a
// buffer. It also caches compiled kernels, keyed by their source text.
//
// DeviceTensor<T, R> is a shape plus a device buffer. It is TensorLike: its
// eval() copies it back to the host, so it can appear in expressions.
#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xinn/gpu/opencl.hpp"
#include "xinn/shape.hpp"
#include "xinn/tensor.hpp"

namespace xinn::gpu {

namespace detail {

using namespace cl;

struct DeviceState {
    const Api* api = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    std::string name, platform, version;
    std::size_t max_group = 1;
    std::mutex mutex;   // one thread at a time sets kernel arguments and enqueues
    struct Program {
        cl_program program;
        cl_kernel kernel;
    };
    std::unordered_map<std::string, Program> kernels;   // key: options + source

    ~DeviceState() {
        for (auto& [key, p] : kernels) {
            api->ReleaseKernel(p.kernel);
            api->ReleaseProgram(p.program);
        }
        if (queue) api->ReleaseCommandQueue(queue);
        if (context) api->ReleaseContext(context);
    }
};

template <class Info, class Handle, class Key>
std::string info_string(Info get, Handle h, Key key) {
    std::size_t size = 0;
    if (get(h, key, 0, nullptr, &size) != CL_SUCCESS || size == 0) return {};
    std::string s(size, '\0');
    get(h, key, size, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
}

}  // namespace detail

class Device {
public:
    // The first GPU of any platform; failing that, the first device of any
    // type (a CPU OpenCL runtime, say); failing that, an Error.
    static std::expected<Device, Error> open() {
        const auto& table = cl::api();
        if (!table) return std::unexpected(table.error());
        return open(*table);
    }

    // The same, through another function table: the tests pass a fake
    // OpenCL driver here. The table must outlive the device.
    static std::expected<Device, Error> open(const cl::Api& f) {
        using namespace cl;

        cl_uint count = 0;
        if (cl_int rc = f.GetPlatformIDs(0, nullptr, &count); rc != CL_SUCCESS)
            return std::unexpected(error("clGetPlatformIDs", rc));
        if (count == 0) return std::unexpected(Error{"no OpenCL platform", -1001});
        std::vector<cl_platform_id> platforms(count);
        f.GetPlatformIDs(count, platforms.data(), nullptr);

        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        for (cl_device_type type : {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ALL}) {
            for (cl_platform_id p : platforms) {
                cl_uint n = 0;
                if (f.GetDeviceIDs(p, type, 1, &device, &n) == CL_SUCCESS && n > 0) {
                    platform = p;
                    break;
                }
            }
            if (platform) break;
        }
        if (!platform) return std::unexpected(Error{"no OpenCL device", -1});

        auto s = std::make_shared<detail::DeviceState>();
        s->api = &f;
        s->device = device;
        s->name = detail::info_string(f.GetDeviceInfo, device, CL_DEVICE_NAME);
        s->version = detail::info_string(f.GetDeviceInfo, device, CL_DEVICE_VERSION);
        s->platform = detail::info_string(f.GetPlatformInfo, platform, CL_PLATFORM_NAME);
        f.GetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(std::size_t), &s->max_group, nullptr);

        cl_int rc = CL_SUCCESS;
        s->context = f.CreateContext(nullptr, 1, &device, nullptr, nullptr, &rc);
        if (rc != CL_SUCCESS) return std::unexpected(error("clCreateContext", rc));
        s->queue = f.CreateCommandQueue(s->context, device, 0, &rc);   // in order, no profiling
        if (rc != CL_SUCCESS) return std::unexpected(error("clCreateCommandQueue", rc));
        return Device(std::move(s));
    }

    std::string_view name() const noexcept { return s_->name; }
    std::string_view platform() const noexcept { return s_->platform; }
    std::string_view version() const noexcept { return s_->version; }
    std::size_t max_work_group_size() const noexcept { return s_->max_group; }

    // The kernel `name` from `source`, built with `options` the first time
    // and cached after that. A build error carries the compiler's log.
    // Call with the lock held (see lock()).
    std::expected<cl::cl_kernel, Error> kernel(std::string_view source, std::string_view name,
                                               std::string_view options = {}) const {
        using namespace cl;
        std::string key = std::string(options) + '\n' + std::string(source);
        if (auto it = s_->kernels.find(key); it != s_->kernels.end()) return it->second.kernel;

        const Api& f = *s_->api;
        const char* text = source.data();
        const std::size_t length = source.size();
        cl_int rc = CL_SUCCESS;
        cl_program program = f.CreateProgramWithSource(s_->context, 1, &text, &length, &rc);
        if (rc != CL_SUCCESS) return std::unexpected(error("clCreateProgramWithSource", rc));
        const std::string opts(options);
        if (rc = f.BuildProgram(program, 1, &s_->device, opts.c_str(), nullptr, nullptr); rc != CL_SUCCESS) {
            Error e = error("clBuildProgram", rc);
            e.message += "\n" + detail::info_string(
                [&](cl_program p, cl_program_build_info i, std::size_t n, void* v, std::size_t* r) {
                    return f.GetProgramBuildInfo(p, s_->device, i, n, v, r);
                }, program, CL_PROGRAM_BUILD_LOG);
            f.ReleaseProgram(program);
            return std::unexpected(std::move(e));
        }
        const std::string kernel_name(name);
        cl_kernel kernel = f.CreateKernel(program, kernel_name.c_str(), &rc);
        if (rc != CL_SUCCESS) {
            f.ReleaseProgram(program);
            return std::unexpected(error("clCreateKernel", rc));
        }
        s_->kernels.emplace(std::move(key), detail::DeviceState::Program{program, kernel});
        return kernel;
    }

    // Kernel objects hold their arguments, so setting them and enqueueing
    // must not interleave between threads. Everything in gpu.hpp takes this.
    std::unique_lock<std::mutex> lock() const { return std::unique_lock(s_->mutex); }

    // The raw handles, for code that calls OpenCL itself.
    const cl::Api& api() const noexcept { return *s_->api; }
    cl::cl_context context() const noexcept { return s_->context; }
    cl::cl_command_queue queue() const noexcept { return s_->queue; }
    cl::cl_device_id id() const noexcept { return s_->device; }

    bool operator==(const Device& other) const noexcept { return s_ == other.s_; }

private:
    explicit Device(std::shared_ptr<detail::DeviceState> s) : s_(std::move(s)) {}
    std::shared_ptr<detail::DeviceState> s_;
};

// ---- device memory --------------------------------------------------------------

// A block of device memory. Copies share it; the last one frees it.
class Buffer {
public:
    Buffer() = default;

    static std::expected<Buffer, Error> allocate(const Device& dev, std::size_t bytes) {
        if (bytes == 0) return Buffer(dev, nullptr, 0);   // OpenCL has no empty buffers
        cl::cl_int rc = cl::CL_SUCCESS;
        cl::cl_mem mem = dev.api().CreateBuffer(dev.context(), cl::CL_MEM_READ_WRITE, bytes, nullptr, &rc);
        if (rc != cl::CL_SUCCESS) return std::unexpected(cl::error("clCreateBuffer", rc));
        return Buffer(dev, mem, bytes);
    }

    // Blocking copies: when they return, the data has arrived.
    std::expected<void, Error> write(std::span<const std::byte> src) const
        pre(src.size() == bytes())
    {
        if (src.empty()) return {};
        auto guard = s_->device.lock();
        const cl::cl_int rc = s_->device.api().EnqueueWriteBuffer(s_->device.queue(), s_->mem, cl::CL_TRUE, 0,
                                                                  src.size(), src.data(), 0, nullptr, nullptr);
        if (rc != cl::CL_SUCCESS) return std::unexpected(cl::error("clEnqueueWriteBuffer", rc));
        return {};
    }

    std::expected<void, Error> read(std::span<std::byte> dst) const
        pre(dst.size() == bytes())
    {
        if (dst.empty()) return {};
        auto guard = s_->device.lock();
        const cl::cl_int rc = s_->device.api().EnqueueReadBuffer(s_->device.queue(), s_->mem, cl::CL_TRUE, 0,
                                                                 dst.size(), dst.data(), 0, nullptr, nullptr);
        if (rc != cl::CL_SUCCESS) return std::unexpected(cl::error("clEnqueueReadBuffer", rc));
        return {};
    }

    cl::cl_mem handle() const noexcept { return s_ ? s_->mem : nullptr; }
    std::size_t bytes() const noexcept { return s_ ? s_->bytes : 0; }
    const Device& device() const noexcept { return s_->device; }

private:
    // Owns one cl_mem. Not copyable: a copy would release it twice.
    struct State {
        State(const Device& d, cl::cl_mem m, std::size_t n) : device(d), mem(m), bytes(n) {}
        State(const State&) = delete;
        ~State() {
            if (mem) device.api().ReleaseMemObject(mem);
        }
        Device device;   // keeps the context alive while the buffer exists
        cl::cl_mem mem;
        std::size_t bytes;
    };
    Buffer(const Device& dev, cl::cl_mem mem, std::size_t bytes) : s_(std::make_shared<State>(dev, mem, bytes)) {}
    std::shared_ptr<State> s_;
};

template <class T, std::size_t R>
class DeviceTensor {
public:
    using value_type = T;
    static constexpr std::size_t rank = R;

    DeviceTensor() = default;

    static std::expected<DeviceTensor, Error> allocate(const Device& dev, const Shape<R>& shape) {
        return Buffer::allocate(dev, shape.count() * sizeof(T)).transform([&](Buffer b) {
            return DeviceTensor(std::move(b), shape);
        });
    }

    const Shape<R>& shape() const noexcept { return shape_; }
    std::size_t size() const noexcept { return shape_.count(); }
    const Buffer& buffer() const noexcept { return buf_; }
    const Device& device() const noexcept { return buf_.device(); }

    // Device -> host.
    std::expected<Tensor<T, R>, Error> download() const {
        auto out = Tensor<T, R>::uninitialized(shape_);
        return buf_.read(std::as_writable_bytes(out.mut_flat())).transform([&] { return out; });
    }

    // TensorLike: evaluating a device tensor brings it to the host. A failed
    // copy is a contract violation here; call download() to handle it.
    Tensor<T, R> eval() const {
        auto t = download();
        contract_assert(t.has_value());
        return *std::move(t);
    }

private:
    DeviceTensor(Buffer b, const Shape<R>& shape) : buf_(std::move(b)), shape_(shape) {}
    Buffer buf_;
    Shape<R> shape_{};
};

template <class X> inline constexpr bool is_device_tensor_v = false;
template <class T, std::size_t R> inline constexpr bool is_device_tensor_v<DeviceTensor<T, R>> = true;

// Host -> device.
template <class T, std::size_t R>
std::expected<DeviceTensor<T, R>, Error> upload(const Device& dev, const Tensor<T, R>& t) {
    auto d = DeviceTensor<T, R>::allocate(dev, t.shape());
    if (!d) return d;
    if (auto ok = d->buffer().write(std::as_bytes(t.flat())); !ok) return std::unexpected(ok.error());
    return d;
}

}  // namespace xinn::gpu
