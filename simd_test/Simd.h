#pragma once
#include <array>

// Minimal SIMD abstraction for NanoVDB stencil kernels.
//
// Designed to be __hostdev__-compatible: on CUDA device code, instantiate
// kernels with T=float (scalar); on CPU, instantiate with T=Simd<float,W>.
// All arithmetic operators and where()/max() are overloaded for both cases,
// so a single templated kernel compiles correctly for both execution contexts.
//
// Mirrors the C++26 std::simd interface deliberately — migration is a typedef.

// ---------------------------------------------------------------------------
// Portability: __hostdev__ is a no-op outside CUDA
// ---------------------------------------------------------------------------
#ifndef __CUDACC__
#  define NANOVDB_SIMD_HOSTDEV
#else
#  define NANOVDB_SIMD_HOSTDEV __host__ __device__
#endif

namespace nanovdb {
namespace util {

template<typename T, int W> struct Simd;
template<typename T, int W> struct SimdMask;

// ---------------------------------------------------------------------------
// SimdMask<T, W>: result of a lane-wise comparison
// ---------------------------------------------------------------------------
template<typename T, int W>
struct SimdMask {
    std::array<bool, W> data{};

    NANOVDB_SIMD_HOSTDEV bool  operator[](int i) const { return data[i]; }
    NANOVDB_SIMD_HOSTDEV bool& operator[](int i)       { return data[i]; }
};

// ---------------------------------------------------------------------------
// Simd<T, W>: W-wide vector of T, backed by std::array<T, W>
// ---------------------------------------------------------------------------
template<typename T, int W>
struct Simd {
    std::array<T, W> data{};

    Simd() = default;
    NANOVDB_SIMD_HOSTDEV Simd(T scalar) { data.fill(scalar); }           // broadcast
    NANOVDB_SIMD_HOSTDEV explicit Simd(const T* p) {
        for (int i = 0; i < W; i++) data[i] = p[i];
    }

    NANOVDB_SIMD_HOSTDEV T  operator[](int i) const { return data[i]; }
    NANOVDB_SIMD_HOSTDEV T& operator[](int i)       { return data[i]; }

    NANOVDB_SIMD_HOSTDEV void store(T* p) const {
        for (int i = 0; i < W; i++) p[i] = data[i];
    }

    // Unary minus
    NANOVDB_SIMD_HOSTDEV Simd operator-() const {
        Simd r;
        for (int i = 0; i < W; i++) r.data[i] = -data[i];
        return r;
    }

    // Lane-wise arithmetic (Simd op Simd)
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

    // Lane-wise comparison → SimdMask
    NANOVDB_SIMD_HOSTDEV SimdMask<T,W> operator>(Simd o) const {
        SimdMask<T,W> m;
        for (int i = 0; i < W; i++) m.data[i] = data[i] > o.data[i];
        return m;
    }
};

// ---------------------------------------------------------------------------
// Mixed scalar/Simd arithmetic (enables e.g. 2.f * simd_val)
// Template argument deduction does not use implicit conversions, so these
// explicit overloads are required for scalar op Simd and Simd op scalar.
// ---------------------------------------------------------------------------
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator+(Simd<T,W> a, T b) { return a + Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator+(T a, Simd<T,W> b) { return Simd<T,W>(a) + b; }

template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator-(Simd<T,W> a, T b) { return a - Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator-(T a, Simd<T,W> b) { return Simd<T,W>(a) - b; }

template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator*(Simd<T,W> a, T b) { return a * Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator*(T a, Simd<T,W> b) { return Simd<T,W>(a) * b; }

template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator/(Simd<T,W> a, T b) { return a / Simd<T,W>(b); }
template<typename T, int W> NANOVDB_SIMD_HOSTDEV
Simd<T,W> operator/(T a, Simd<T,W> b) { return Simd<T,W>(a) / b; }

// ---------------------------------------------------------------------------
// min/max: lane-wise minimum and maximum
// Scalar overloads ensure templated kernels compile for T=float (GPU path)
// ---------------------------------------------------------------------------
template<typename T>
NANOVDB_SIMD_HOSTDEV T min(T a, T b) { return a < b ? a : b; }

template<typename T>
NANOVDB_SIMD_HOSTDEV T max(T a, T b) { return a > b ? a : b; }

template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = a[i] < b[i] ? a[i] : b[i];
    return r;
}

template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = a[i] > b[i] ? a[i] : b[i];
    return r;
}

// ---------------------------------------------------------------------------
// where: lane-wise select — returns a where mask is true, b otherwise
// Maps to VBLENDVPS on AVX2; no branching in vectorized code.
// Scalar overload: plain ternary, for T=float (GPU path)
// ---------------------------------------------------------------------------
template<typename T>
NANOVDB_SIMD_HOSTDEV T where(bool mask, T a, T b) { return mask ? a : b; }

template<typename T, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = mask[i] ? a[i] : b[i];
    return r;
}

} // namespace util
} // namespace nanovdb
