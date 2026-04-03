#pragma once
#include <array>

// Minimal SIMD abstraction for NanoVDB stencil kernels.
//
// Two implementations, selected automatically at compile time:
//
//   NANOVDB_USE_STD_SIMD (set when <experimental/simd> is available):
//     Simd<T,W> and SimdMask<T,W> are pure type aliases for
//     std::experimental::fixed_size_simd / fixed_size_simd_mask.
//     All arithmetic delegates to the standard type; the compiler emits
//     native vector instructions without relying on the auto-vectorizer.
//
//   Default (std::array backend):
//     Simd<T,W> wraps std::array<T,W> with element-wise operator loops.
//     Clang auto-vectorizes these loops; GCC does not.
//
// In both cases the interface is identical, so templated kernels (T=float
// for GPU, T=Simd<float,W> for CPU) compile unmodified.
//
// Mirrors the C++26 std::simd naming — migration will be a typedef swap.

// ---------------------------------------------------------------------------
// Portability: __hostdev__ is a no-op outside CUDA
// ---------------------------------------------------------------------------
#ifndef __CUDACC__
#  define NANOVDB_SIMD_HOSTDEV
#else
#  define NANOVDB_SIMD_HOSTDEV __host__ __device__
#endif

// ---------------------------------------------------------------------------
// Auto-detect std::experimental::simd (Parallelism TS v2)
// ---------------------------------------------------------------------------
#if !defined(NANOVDB_NO_STD_SIMD) && defined(__has_include) && __has_include(<experimental/simd>)
#  include <experimental/simd>
#  ifdef __cpp_lib_experimental_parallel_simd
#    define NANOVDB_USE_STD_SIMD 1
#  endif
#endif

namespace nanovdb {
namespace util {

// ---------------------------------------------------------------------------
// element_aligned_tag — portable load/store alignment descriptor.
// In the stdx backend this is an alias for stdx::element_aligned_tag so that
// nanovdb::util::element_aligned is the same token stdx constructors expect.
// In the std::array backend it is a standalone dummy struct (ignored).
// ---------------------------------------------------------------------------
#ifdef NANOVDB_USE_STD_SIMD
namespace stdx = std::experimental;
using element_aligned_tag = stdx::element_aligned_tag;
#else
struct element_aligned_tag {};
#endif
inline constexpr element_aligned_tag element_aligned{};

// ===========================================================================
// Implementation A: std::experimental::simd — pure type aliases
// ===========================================================================
#ifdef NANOVDB_USE_STD_SIMD

template<typename T, int W>
using SimdMask = stdx::fixed_size_simd_mask<T, W>;

template<typename T, int W>
using Simd = stdx::fixed_size_simd<T, W>;

template<typename T, int W>
inline Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) { return stdx::min(a, b); }

template<typename T, int W>
inline Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) { return stdx::max(a, b); }

// TS v2 where(mask, v) is a masked assignment proxy, not a 3-arg select.
// Wrap it into the select(mask, a, b) form our kernels expect.
template<typename T, int W>
inline Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    auto result = b;
    stdx::where(mask, result) = a;
    return result;
}

// ===========================================================================
// Implementation B: std::array backend (default)
// ===========================================================================
#else

template<typename T, int W>
struct SimdMask {
    std::array<bool, W> data{};
    NANOVDB_SIMD_HOSTDEV bool  operator[](int i) const { return data[i]; }
    NANOVDB_SIMD_HOSTDEV bool& operator[](int i)       { return data[i]; }
};

template<typename T, int W>
struct Simd {
    std::array<T, W> data{};

    Simd() = default;
    NANOVDB_SIMD_HOSTDEV Simd(T scalar) { data.fill(scalar); }               // broadcast
    NANOVDB_SIMD_HOSTDEV explicit Simd(const T* p, element_aligned_tag = {}) { // load
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    NANOVDB_SIMD_HOSTDEV T  operator[](int i) const { return data[i]; }
    NANOVDB_SIMD_HOSTDEV T& operator[](int i)       { return data[i]; }
    NANOVDB_SIMD_HOSTDEV void store(T* p, element_aligned_tag = {}) const {   // store
        for (int i = 0; i < W; i++) p[i] = data[i];
    }
    NANOVDB_SIMD_HOSTDEV Simd operator-() const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = -data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator+(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] + o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator-(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] - o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator*(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] * o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator/(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] / o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV SimdMask<T,W> operator>(Simd o) const {
        SimdMask<T,W> m;
        for (int i = 0; i < W; i++) m.data[i] = data[i] > o.data[i];
        return m;
    }
};

template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator+(T a, Simd<T,W> b) { return Simd<T,W>(a) + b; }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator+(Simd<T,W> a, T b) { return a + Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator-(T a, Simd<T,W> b) { return Simd<T,W>(a) - b; }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator-(Simd<T,W> a, T b) { return a - Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator*(T a, Simd<T,W> b) { return Simd<T,W>(a) * b; }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator*(Simd<T,W> a, T b) { return a * Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator/(T a, Simd<T,W> b) { return Simd<T,W>(a) / b; }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator/(Simd<T,W> a, T b) { return a / Simd<T,W>(b); }

template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; i++) r[i] = a[i] < b[i] ? a[i] : b[i]; return r;
}
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; i++) r[i] = a[i] > b[i] ? a[i] : b[i]; return r;
}
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; i++) r[i] = mask[i] ? a[i] : b[i]; return r;
}

#endif // NANOVDB_USE_STD_SIMD

// ---------------------------------------------------------------------------
// Scalar overloads — always present, for T=float (GPU / scalar path)
// ---------------------------------------------------------------------------
template<typename T> NANOVDB_SIMD_HOSTDEV T min(T a, T b)           { return a < b ? a : b; }
template<typename T> NANOVDB_SIMD_HOSTDEV T max(T a, T b)           { return a > b ? a : b; }
template<typename T> NANOVDB_SIMD_HOSTDEV T where(bool m, T a, T b) { return m ? a : b; }

} // namespace util
} // namespace nanovdb
