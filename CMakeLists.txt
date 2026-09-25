# SPDX-License-Identifier: BSD-3-Clause
cmake_minimum_required(VERSION 3.30)
project(xinn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)   # no C++20 modules (yet)

# Reflection and contracts need GCC 16. Say so now, not with an unknown flag later.
if(NOT (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16))
    message(FATAL_ERROR
        "XiNN needs GCC 16 or newer (-freflection, -fcontracts). This build uses "
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}: ${CMAKE_CXX_COMPILER}\n"
        "Use the presets (cmake --preset debug), or in CLion the debug/release profiles "
        "from CMakePresets.json, or a toolchain with GCC 16 (book chapter 0).")
endif()

# XiNN is header-only: everything is templates.
add_library(xinn INTERFACE)
target_include_directories(xinn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(xinn INTERFACE
    -freflection            # P2996 static reflection
    -fcontracts             # P2900 contracts: pre / post / contract_assert
    -Wall -Wextra -Wpedantic)

# Build for the CPU in this machine: AVX2/FMA and wider SIMD where available.
option(XINN_NATIVE "Optimize for the build machine's CPU (-march=native)" ON)
if(XINN_NATIVE)
    target_compile_options(xinn INTERFACE -march=native)
endif()

# Parallel evaluation through C++26 std::execution. The standard library does
# not ship it yet, so stdexec -- NVIDIA's reference implementation of the
# proposal (P2300) -- stands in, pinned to one commit. xinn/parallel.hpp
# switches to <execution> when the library provides it.
option(XINN_THREADS "Evaluate in parallel with std::execution (stdexec)" ON)
if(XINN_THREADS)
    include(FetchContent)
    FetchContent_Declare(stdexec
        GIT_REPOSITORY https://github.com/NVIDIA/stdexec.git
        GIT_TAG        04de75de5807e73a229acf047c6976575e313359
        SOURCE_SUBDIR  _headers_only)   # only the headers are needed; do not build stdexec
    FetchContent_MakeAvailable(stdexec)
    target_include_directories(xinn SYSTEM INTERFACE ${stdexec_SOURCE_DIR}/include)
    target_compile_definitions(xinn INTERFACE XINN_HAS_STDEXEC=1 STDEXEC_PARALLEL_SCHEDULER_HEADER_ONLY=1)
endif()

if(MINGW)
    # std::print needs libstdc++exp on MinGW; static linking avoids DLL hunting.
    target_link_libraries(xinn INTERFACE -static stdc++exp)
    # GCC bug 54412: with AVX, GCC assumes a 32-byte aligned stack, but 64-bit
    # Windows only guarantees 16 bytes, so aligned spills of SIMD registers
    # crash. Let the assembler use unaligned moves (same speed on modern CPUs).
    target_compile_options(xinn INTERFACE -Wa,-muse-unaligned-vector-move)
endif()

# XiNN Lab, the interactive front end (Dear ImGui + ImPlot). Optional: it
# fetches GLFW, Dear ImGui and ImPlot, and needs OpenGL 3.
option(XINN_GUI "Build XiNN Lab, the Dear ImGui front end" OFF)

enable_testing()
add_subdirectory(tests)
add_subdirectory(examples)
if(XINN_GUI)
    add_subdirectory(gui)
endif()
