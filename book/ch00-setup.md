# 0. Setup

## 0.1 Compiler

XiNN needs a compiler with C++26 static reflection and contracts. In 2026
that means **GCC 16**. MSVC 19.51 is close to C++23; it has no reflection, no
expansion statements, no pack indexing and no contracts.

On Windows, install the WinLibs build of GCC:

```
winget install BrechtSanders.WinLibs.POSIX.UCRT
```

It installs under
`%LOCALAPPDATA%\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_...\mingw64\bin`,
and brings `g++`, `gdb`, `cmake` and `ninja`.

Two flags turn the new features on:

| Flag | Enables |
|---|---|
| `-std=c++26` | the language standard |
| `-freflection` | static reflection (P2996): `^^T`, `[: r :]`, `<meta>` |
| `-fcontracts` | contracts (P2900): `pre`, `post`, `contract_assert` |

## 0.2 Building

The project is header-only. `CMakePresets.json` points CMake at the WinLibs
compiler.

```
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

Use `release` instead of `debug` for an optimized build.

Settings in `CMakeLists.txt` that deserve a note:

- `CMAKE_CXX_SCAN_FOR_MODULES OFF`. CMake 4 scans for C++20 modules by
  default at C++26. We do not use modules.
- `-lstdc++exp`. On MinGW, `std::print` lives in this extra library.
- `-static`. The test programs then run without the GCC DLLs on `PATH`.
- `XINN_NATIVE` (on by default): `-march=native`, so the build uses every
  SIMD instruction the CPU has. Binaries then run only on similar CPUs.
- `XINN_THREADS` (on by default): fetches stdexec for `std::execution`
  (chapter 6). With it off, everything runs on one thread.
- `XINN_GUI` (off by default; the `gui` preset turns it on): builds
  [XiNN Lab](xinn-lab.md), the interactive front end.

## 0.3 CLion

1. **Settings → Build, Execution, Deployment → Toolchains.** Add a *MinGW*
   toolchain and set its path to the WinLibs `mingw64` folder. Move it to
   the top, so that it is the default. CLion shows "Version: 14.0 w64": that
   is the MinGW-w64 runtime, not GCC; the compiler is GCC 16.1.
2. Open the project folder. CLion reads `CMakePresets.json` and offers the
   `debug` and `release` profiles. Enable them, and pick the toolchain from
   step 1.
3. Each test is a run target. For debugging, use CLion's bundled GDB
   (*Debugger → Debug Profiles*, with the executable left empty). WinLibs
   also ships a GDB, but it can be newer than the versions CLion supports
   (17.2, against 17.1 in CLion 2026.2).

CLion also creates its own `Debug` profile (folder `cmake-build-debug`),
which uses the MinGW bundled with CLion, GCC 15 at the time of writing. That
compiler has no reflection, so CMake stops with "XiNN needs GCC 16 or
newer". Disable that profile, or give it the toolchain from step 1 and run
*Reset Cache and Reload Project*.

CLion's code analysis is based on Clang, not GCC. It may underline valid
C++26 code, especially reflection (`^^tests`, `[:f:]`) and contracts. Trust
the compiler output.

## 0.4 What works in GCC 16.1

| Feature | Status |
|---|---|
| Static reflection, `template for`, `define_static_array` | yes |
| Pack indexing `Ts...[I]`, structured-binding packs | yes |
| Contracts, replaceable violation handler | yes, with one bug (below) |
| `constexpr` exceptions | yes |
| `std::mdspan`, `std::submdspan` | yes |
| `std::print`, `std::inplace_vector` | yes |
| `std::simd` | no; `<experimental/simd>`, its predecessor, works |
| `std::execution` (senders) | no; stdexec, the reference implementation, is fetched by CMake |
| `std::linalg` | no; XiNN has its own matrix kernel (chapter 6) |

**Compiler bug.** A `pre` or `post` on a variadic function template crashes
GCC 16.1 when the template is called with an empty pack:

```cpp
template <class... I> int f(I...) pre(true) { return 0; }
int main() { return f(); }   // internal compiler error
```

`contract_assert` inside the body works. XiNN uses it wherever a variadic
function can be called with no arguments, such as reading a rank-0 tensor.

**AVX on Windows.** GCC assumes a 32-byte aligned stack for AVX code, but
64-bit Windows guarantees only 16 bytes (GCC bug 54412). Aligned stores of
SIMD registers to the stack then crash. `CMakeLists.txt` passes
`-Wa,-muse-unaligned-vector-move` on MinGW; see §6.7.

## 0.5 Tests

Tests use a small harness in `tests/check.hpp` with no macros. Every
function in namespace `tests` is a test, and `main` runs them all:

```cpp
namespace tests {
void adds_up() { check::equal(1 + 1, 2); }
}
int main() { return check::run_tests<^^tests>(); }
```

`run_tests` uses reflection to list the functions in the namespace, then
calls each one. Chapter 1 explains how.

*Death tests* check that a contract fires. They replace the
contract-violation handler with one that exits with status 1, and CTest
expects that failure (`WILL_FAIL`).
