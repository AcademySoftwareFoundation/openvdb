// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file   nanovdb/util/Simd.h

    \author Efty Sifakis

    \date   April 28, 2026

    \brief  Minimal SIMD abstraction for NanoVDB stencil kernels.
*/

#ifndef NANOVDB_UTIL_SIMD_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_SIMD_H_HAS_BEEN_INCLUDED

#include <array>

#include <nanovdb/util/Util.h>  // __hostdev__

// Two implementations, selected automatically at compile time:
//
//   stdx backend (default, when <experimental/simd> is available):
//     Simd<T,W> and SimdMask<T,W> are aliases for
//     std::experimental::fixed_size_simd / fixed_size_simd_mask.
//     Internal flag: NANOVDB_USE_STDX_SIMD.
//
//   std::array backend (forced via NANOVDB_SIMD_ARRAY_BACKEND, or when
//   <experimental/simd> is unavailable):
//     Simd<T,W> wraps std::array<T,W> with element-wise operator loops.
//
// The interface is identical in both cases, so templated kernels
// (T=float for GPU, T=Simd<float,W> for CPU) compile unmodified.

// ---------------------------------------------------------------------------
// Auto-detect std::experimental::simd (Parallelism TS v2)
// ---------------------------------------------------------------------------
#if !defined(NANOVDB_SIMD_ARRAY_BACKEND) && defined(__has_include) && __has_include(<experimental/simd>)
#  include <experimental/simd>
#  ifdef __cpp_lib_experimental_parallel_simd
#    define NANOVDB_USE_STDX_SIMD 1
#  endif
#endif

namespace nanovdb {
namespace util {

// ===========================================================================
// nanovdb::util::experimental -- internal SIMD primitives.  Names in this
// nested namespace are unstable API by convention; external callers should
// not depend on them.
// ===========================================================================
namespace experimental {

// element_aligned_tag -- load/store alignment descriptor.  Aliases
// stdx::element_aligned_tag in the stdx backend; empty struct in the
// array backend.
#ifdef NANOVDB_USE_STDX_SIMD
namespace stdx = std::experimental;
using element_aligned_tag = stdx::element_aligned_tag;
#else
struct element_aligned_tag {};
#endif
inline constexpr element_aligned_tag element_aligned{};

// ===========================================================================
// Implementation A: std::experimental::simd -- pure type aliases
// ===========================================================================
#ifdef NANOVDB_USE_STDX_SIMD

template<typename T, int W>
using SimdMask = stdx::fixed_size_simd_mask<T, W>;

template<typename T, int W>
using Simd = stdx::fixed_size_simd<T, W>;

// ===========================================================================
// Implementation B: std::array backend (default)
// ===========================================================================
#else

template<typename T, int W> struct Simd;  // fwd-decl so SimdMask::simd_type can name it

template<typename T, int W>
struct SimdMask {
    using value_type = bool;
    using simd_type  = Simd<T, W>;
    static constexpr size_t size() { return size_t(W); }

    std::array<bool, W> data{};
    SimdMask() = default;
    __hostdev__ SimdMask(bool b) { data.fill(b); }                       // broadcast
    __hostdev__ explicit SimdMask(const bool* p, element_aligned_tag) {  // load (ctor)
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    __hostdev__ bool  operator[](int i) const { return data[i]; }
    __hostdev__ bool& operator[](int i)       { return data[i]; }
    __hostdev__ void copy_from(const bool* p, element_aligned_tag) {     // load (member)
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    __hostdev__ void copy_to(bool* p, element_aligned_tag) const {       // store (member)
        for (int i = 0; i < W; i++) p[i] = data[i];
    }
    __hostdev__ SimdMask operator!() const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = !data[i]; return r;
    }
    __hostdev__ SimdMask operator&(SimdMask o) const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = data[i] && o.data[i]; return r;
    }
    __hostdev__ SimdMask operator|(SimdMask o) const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = data[i] || o.data[i]; return r;
    }
};

template<typename T, int W>
struct Simd {
    using value_type = T;
    using mask_type  = SimdMask<T, W>;
    static constexpr size_t size() { return size_t(W); }

    std::array<T, W> data{};

    Simd() = default;
    __hostdev__ Simd(T scalar) { data.fill(scalar); }               // broadcast
    __hostdev__ explicit Simd(const T* p, element_aligned_tag) {    // load (ctor)
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    __hostdev__ T  operator[](int i) const { return data[i]; }
    __hostdev__ T& operator[](int i)       { return data[i]; }
    __hostdev__ void copy_from(const T* p, element_aligned_tag) {   // load (member)
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    __hostdev__ void copy_to(T* p, element_aligned_tag) const {     // store (member)
        for (int i = 0; i < W; i++) p[i] = data[i];
    }
    __hostdev__ Simd operator-() const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = -data[i]; return r;
    }
    __hostdev__ Simd operator+(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] + o.data[i]; return r;
    }
    __hostdev__ Simd operator-(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] - o.data[i]; return r;
    }
    __hostdev__ Simd operator*(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] * o.data[i]; return r;
    }
    __hostdev__ Simd operator/(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] / o.data[i]; return r;
    }
    __hostdev__ Simd& operator+=(Simd o) {
        for (int i = 0; i < W; i++) data[i] += o.data[i]; return *this;
    }
    __hostdev__ Simd& operator-=(Simd o) {
        for (int i = 0; i < W; i++) data[i] -= o.data[i]; return *this;
    }
    __hostdev__ Simd& operator*=(Simd o) {
        for (int i = 0; i < W; i++) data[i] *= o.data[i]; return *this;
    }
    __hostdev__ Simd& operator/=(Simd o) {
        for (int i = 0; i < W; i++) data[i] /= o.data[i]; return *this;
    }
    __hostdev__ SimdMask<T,W> operator<(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] <  o.data[i]; return m;
    }
    __hostdev__ SimdMask<T,W> operator<=(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] <= o.data[i]; return m;
    }
    __hostdev__ SimdMask<T,W> operator>(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] >  o.data[i]; return m;
    }
    __hostdev__ SimdMask<T,W> operator>=(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] >= o.data[i]; return m;
    }
    __hostdev__ SimdMask<T,W> operator==(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] == o.data[i]; return m;
    }
    __hostdev__ SimdMask<T,W> operator!=(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] != o.data[i]; return m;
    }
    // Bitwise and shift operators -- valid for integer element types.
    __hostdev__ Simd operator|(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] | o.data[i]; return r;
    }
    __hostdev__ Simd operator&(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] & o.data[i]; return r;
    }
    __hostdev__ Simd operator^(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] ^ o.data[i]; return r;
    }
    // Per-lane variable shift (shift count from corresponding lane of o).
    __hostdev__ Simd operator<<(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] << o.data[i]; return r;
    }
    __hostdev__ Simd operator>>(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] >> o.data[i]; return r;
    }
    // Uniform shift: all lanes shifted by the same scalar count (vpsllw imm8 / vpsrlw imm8).
    __hostdev__ Simd operator<<(T shift) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] << shift; return r;
    }
    __hostdev__ Simd operator>>(T shift) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] >> shift; return r;
    }
};

template<typename T, int W> __hostdev__
Simd<T,W> operator+(T a, Simd<T,W> b) { return Simd<T,W>(a) + b; }
template<typename T, int W> __hostdev__
Simd<T,W> operator+(Simd<T,W> a, T b) { return a + Simd<T,W>(b); }
template<typename T, int W> __hostdev__
Simd<T,W> operator-(T a, Simd<T,W> b) { return Simd<T,W>(a) - b; }
template<typename T, int W> __hostdev__
Simd<T,W> operator-(Simd<T,W> a, T b) { return a - Simd<T,W>(b); }
template<typename T, int W> __hostdev__
Simd<T,W> operator*(T a, Simd<T,W> b) { return Simd<T,W>(a) * b; }
template<typename T, int W> __hostdev__
Simd<T,W> operator*(Simd<T,W> a, T b) { return a * Simd<T,W>(b); }
template<typename T, int W> __hostdev__
Simd<T,W> operator/(T a, Simd<T,W> b) { return Simd<T,W>(a) / b; }
template<typename T, int W> __hostdev__
Simd<T,W> operator/(Simd<T,W> a, T b) { return a / Simd<T,W>(b); }

#endif // NANOVDB_USE_STDX_SIMD

} // namespace experimental
} // namespace util

// ---------------------------------------------------------------------------
// nanovdb::math::Min / Max / Select / Sqrt -- Simd<T,W> overloads.  Scalar
// overloads live in nanovdb/math/Math.h; defining the SIMD overloads here
// avoids a Math.h -> Simd.h dependency.
// ---------------------------------------------------------------------------
namespace math {
#ifdef NANOVDB_USE_STDX_SIMD
template<typename T, int W>
NANOVDB_FORCEINLINE util::experimental::Simd<T,W>
Min(util::experimental::Simd<T,W> a, util::experimental::Simd<T,W> b) {
    return std::experimental::min(a, b);
}
template<typename T, int W>
NANOVDB_FORCEINLINE util::experimental::Simd<T,W>
Max(util::experimental::Simd<T,W> a, util::experimental::Simd<T,W> b) {
    return std::experimental::max(a, b);
}
// 3-arg Select(mask, a, b): mask[i] ? a[i] : b[i], via TS v2's where() proxy.
template<typename T, int W>
NANOVDB_FORCEINLINE util::experimental::Simd<T,W>
Select(util::experimental::SimdMask<T,W> mask,
       util::experimental::Simd<T,W> a,
       util::experimental::Simd<T,W> b) {
    auto result = b;
    util::experimental::stdx::where(mask, result) = a;
    return result;
}
template<typename T, int W>
NANOVDB_FORCEINLINE util::experimental::Simd<T,W>
Sqrt(util::experimental::Simd<T,W> a) {
    return std::experimental::sqrt(a);
}
#else
template<typename T, int W>
__hostdev__ util::experimental::Simd<T,W>
Min(util::experimental::Simd<T,W> a, util::experimental::Simd<T,W> b) {
    util::experimental::Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = a[i] < b[i] ? a[i] : b[i];
    return r;
}
template<typename T, int W>
__hostdev__ util::experimental::Simd<T,W>
Max(util::experimental::Simd<T,W> a, util::experimental::Simd<T,W> b) {
    util::experimental::Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = a[i] > b[i] ? a[i] : b[i];
    return r;
}
template<typename T, int W>
__hostdev__ util::experimental::Simd<T,W>
Select(util::experimental::SimdMask<T,W> mask,
       util::experimental::Simd<T,W> a,
       util::experimental::Simd<T,W> b) {
    util::experimental::Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = mask[i] ? a[i] : b[i];
    return r;
}
template<typename T, int W>
__hostdev__ util::experimental::Simd<T,W>
Sqrt(util::experimental::Simd<T,W> a) {
    util::experimental::Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = Sqrt(a[i]);
    return r;
}
#endif
} // namespace math

} // namespace nanovdb

#endif // end of NANOVDB_UTIL_SIMD_H_HAS_BEEN_INCLUDED
