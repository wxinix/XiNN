// SPDX-License-Identifier: BSD-3-Clause
// The OpenCL API, loaded at run time.
//
// XiNN needs no OpenCL headers and no SDK to build. This header declares the
// few OpenCL C types, constants and functions the GPU backend uses (their
// values are fixed by the Khronos specification), opens the system's OpenCL
// library (the "ICD loader") when first asked, and looks the functions up by
// name:
//
//   Windows   OpenCL.dll          installed with every GPU driver
//   Linux     libOpenCL.so.1      from the ocl-icd package or the driver
//   macOS     OpenCL.framework
//
// If the library or a platform is missing, cl::api() returns an Error
// instead of the table, and the program keeps running on the CPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <type_traits>

#if defined(_WIN32)
// The two Win32 functions needed, declared exactly as <windows.h> does, so a
// program may include both. (Including <windows.h> here would define macros
// such as `near`, `far`, `min` and `max` in every program that uses XiNN.)
struct HINSTANCE__;
extern "C" __declspec(dllimport) HINSTANCE__* __stdcall LoadLibraryA(const char*);
extern "C" __declspec(dllimport) std::intptr_t(__stdcall* __stdcall GetProcAddress(HINSTANCE__*, const char*))();
#define XINN_CL_CALL __stdcall
#else
#include <dlfcn.h>
#define XINN_CL_CALL
#endif

namespace xinn::gpu {

// What went wrong, for std::expected. `code` is the OpenCL error code, or 0
// when the failure was not an OpenCL call (no library, a missing function).
struct Error {
    std::string message;
    int code = 0;
};

namespace cl {

// ---- types (cl.h) -------------------------------------------------------------

using cl_int = std::int32_t;
using cl_uint = std::uint32_t;
using cl_ulong = std::uint64_t;
using cl_bool = cl_uint;
using cl_bitfield = cl_ulong;
using cl_device_type = cl_bitfield;
using cl_mem_flags = cl_bitfield;
using cl_command_queue_properties = cl_bitfield;
using cl_platform_info = cl_uint;
using cl_device_info = cl_uint;
using cl_program_build_info = cl_uint;
using cl_context_properties = std::intptr_t;

using cl_platform_id = struct _cl_platform_id*;
using cl_device_id = struct _cl_device_id*;
using cl_context = struct _cl_context*;
using cl_command_queue = struct _cl_command_queue*;
using cl_mem = struct _cl_mem*;
using cl_program = struct _cl_program*;
using cl_kernel = struct _cl_kernel*;
using cl_event = struct _cl_event*;

// ---- constants ----------------------------------------------------------------

inline constexpr cl_int CL_SUCCESS = 0;
inline constexpr cl_bool CL_TRUE = 1;

inline constexpr cl_device_type CL_DEVICE_TYPE_GPU = 1 << 2;
inline constexpr cl_device_type CL_DEVICE_TYPE_ALL = 0xFFFFFFFF;

inline constexpr cl_platform_info CL_PLATFORM_NAME = 0x0902;
inline constexpr cl_device_info CL_DEVICE_TYPE = 0x1000;
inline constexpr cl_device_info CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002;
inline constexpr cl_device_info CL_DEVICE_MAX_WORK_GROUP_SIZE = 0x1004;
inline constexpr cl_device_info CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F;
inline constexpr cl_device_info CL_DEVICE_NAME = 0x102B;
inline constexpr cl_device_info CL_DEVICE_VERSION = 0x102F;
inline constexpr cl_program_build_info CL_PROGRAM_BUILD_LOG = 0x1183;

inline constexpr cl_mem_flags CL_MEM_READ_WRITE = 1 << 0;
inline constexpr cl_mem_flags CL_MEM_READ_ONLY = 1 << 2;

// The error codes worth naming in a message.
inline constexpr std::string_view error_name(cl_int code) {
    switch (code) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -30: return "CL_INVALID_VALUE";
        case -34: return "CL_INVALID_CONTEXT";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        default: return "an OpenCL error";
    }
}

inline Error error(std::string_view call, cl_int code) {
    return {std::string(call) + " failed: " + std::string(error_name(code)) + " (" + std::to_string(code) + ")", code};
}

// ---- functions ----------------------------------------------------------------
//
// One table entry per function: its pointer type is spelled from the
// specification's signature, and load() fills it by name.

struct Api {
    using BuildCallback = void(XINN_CL_CALL*)(cl_program, void*);
    using ContextCallback = void(XINN_CL_CALL*)(const char*, const void*, std::size_t, void*);

    cl_int(XINN_CL_CALL* GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
    cl_int(XINN_CL_CALL* GetPlatformInfo)(cl_platform_id, cl_platform_info, std::size_t, void*, std::size_t*);
    cl_int(XINN_CL_CALL* GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
    cl_int(XINN_CL_CALL* GetDeviceInfo)(cl_device_id, cl_device_info, std::size_t, void*, std::size_t*);
    cl_context(XINN_CL_CALL* CreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*,
                                            ContextCallback, void*, cl_int*);
    cl_command_queue(XINN_CL_CALL* CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties,
                                                       cl_int*);
    cl_mem(XINN_CL_CALL* CreateBuffer)(cl_context, cl_mem_flags, std::size_t, void*, cl_int*);
    cl_program(XINN_CL_CALL* CreateProgramWithSource)(cl_context, cl_uint, const char**, const std::size_t*,
                                                      cl_int*);
    cl_int(XINN_CL_CALL* BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, BuildCallback,
                                       void*);
    cl_int(XINN_CL_CALL* GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, std::size_t,
                                              void*, std::size_t*);
    cl_kernel(XINN_CL_CALL* CreateKernel)(cl_program, const char*, cl_int*);
    cl_int(XINN_CL_CALL* SetKernelArg)(cl_kernel, cl_uint, std::size_t, const void*);
    cl_int(XINN_CL_CALL* EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*,
                                               const std::size_t*, const std::size_t*, cl_uint, const cl_event*,
                                               cl_event*);
    cl_int(XINN_CL_CALL* EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t, void*,
                                            cl_uint, const cl_event*, cl_event*);
    cl_int(XINN_CL_CALL* EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, std::size_t, std::size_t,
                                             const void*, cl_uint, const cl_event*, cl_event*);
    cl_int(XINN_CL_CALL* Finish)(cl_command_queue);
    cl_int(XINN_CL_CALL* ReleaseMemObject)(cl_mem);
    cl_int(XINN_CL_CALL* ReleaseKernel)(cl_kernel);
    cl_int(XINN_CL_CALL* ReleaseProgram)(cl_program);
    cl_int(XINN_CL_CALL* ReleaseCommandQueue)(cl_command_queue);
    cl_int(XINN_CL_CALL* ReleaseContext)(cl_context);
};

namespace loader {

inline void* open_library() {
#if defined(_WIN32)
    return LoadLibraryA("OpenCL.dll");
#elif defined(__APPLE__)
    return dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_NOW | RTLD_LOCAL);
#else
    void* lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib ? lib : dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

inline void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HINSTANCE__*>(lib), name));
#else
    return dlsym(lib, name);
#endif
}

inline std::expected<Api, Error> load() {
    void* lib = open_library();
    if (!lib) return std::unexpected(Error{"the OpenCL library (ICD loader) is not installed"});
    Api api{};
    std::string missing;
    auto get = [&](auto& fn, const char* name) {
        fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(symbol(lib, name));
        if (!fn) missing += std::string(missing.empty() ? "" : ", ") + name;
    };
    get(api.GetPlatformIDs, "clGetPlatformIDs");
    get(api.GetPlatformInfo, "clGetPlatformInfo");
    get(api.GetDeviceIDs, "clGetDeviceIDs");
    get(api.GetDeviceInfo, "clGetDeviceInfo");
    get(api.CreateContext, "clCreateContext");
    get(api.CreateCommandQueue, "clCreateCommandQueue");   // deprecated in 2.0, still in every loader
    get(api.CreateBuffer, "clCreateBuffer");
    get(api.CreateProgramWithSource, "clCreateProgramWithSource");
    get(api.BuildProgram, "clBuildProgram");
    get(api.GetProgramBuildInfo, "clGetProgramBuildInfo");
    get(api.CreateKernel, "clCreateKernel");
    get(api.SetKernelArg, "clSetKernelArg");
    get(api.EnqueueNDRangeKernel, "clEnqueueNDRangeKernel");
    get(api.EnqueueReadBuffer, "clEnqueueReadBuffer");
    get(api.EnqueueWriteBuffer, "clEnqueueWriteBuffer");
    get(api.Finish, "clFinish");
    get(api.ReleaseMemObject, "clReleaseMemObject");
    get(api.ReleaseKernel, "clReleaseKernel");
    get(api.ReleaseProgram, "clReleaseProgram");
    get(api.ReleaseCommandQueue, "clReleaseCommandQueue");
    get(api.ReleaseContext, "clReleaseContext");
    if (!missing.empty()) return std::unexpected(Error{"the OpenCL library lacks " + missing});
    return api;
}

}  // namespace loader

// The function table, loaded once (thread-safe: a function-local static).
// The library stays loaded until the program ends.
inline const std::expected<Api, Error>& api() {
    static const std::expected<Api, Error> table = loader::load();
    return table;
}

}  // namespace cl
}  // namespace xinn::gpu
