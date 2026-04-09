// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
///
/// @file Simd.h
///
/// @brief Generic-T SIMD abstraction for OpenVDB.
///
/// @details  Provides Simd<T,W> and SimdMask<T,W> wrapper types together with
///   free functions (where, hmin, hmax, hall, hany, hfirst, sqrt, square) that
///   have matching scalar overloads, enabling kernels to be written once as
///   templates on a value type T and compiled for both:
///
///     T = float / double  — scalar path, used directly or on GPU
///     T = Simd<float,W>   — W-wide SIMD path for CPU batch processing
///
///   The same source — operators, where(), hmin() etc. — compiles correctly
///   for both instantiations with no #ifdef or duplicated logic.
///
/// @note  Two backends are provided, selected automatically:
///
///   Backend A (OPENVDB_USE_STD_SIMD): wraps std::experimental::simd
///     (Parallelism TS v2) in a thin class to provide the full API including
///     explicit operator T() and value_type.  The compiler emits native SIMD
///     instructions directly; no auto-vectorizer involvement.
///
///   Backend B (default, C++17): wraps std::array<T,W> with element-wise
///     operator loops.  The compiler auto-vectorizes fixed-count loops.
///     Annotated __hostdev__ for CUDA compatibility (NanoVDB use).
///
/// @note  Migration to std::simd (C++26) will be a one-line change in the
///   backend detection guard; all call sites are unchanged.

#ifndef OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED
#define OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

#ifdef OPENVDB_USE_STD_SIMD
#include <experimental/simd>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace simd {

// ============================================================================
// Register-width constants — derived from OPENVDB_SIMD compile flags.
// SimdNativeT<T> uses this to select the natural lane count for element type T.
// ============================================================================

static constexpr size_t RegisterBits =
#if defined(OPENVDB_USE_AVX)
    256;
#elif defined(OPENVDB_USE_SSE42)
    128;
#else
    0; // no explicit ISA targeting; fall back to scalar (width 1)
#endif

// ============================================================================
// Backend A: std::experimental::simd — thin wrapper class
// ============================================================================
#ifdef OPENVDB_USE_STD_SIMD

namespace stdx = std::experimental;

/// @brief  SIMD mask wrapper for Backend A.
template<typename T, int W>
struct SimdMask
{
    using MaskT = stdx::fixed_size_simd_mask<T, W>;
    MaskT m;

    SimdMask() = default;
    /*implicit*/ SimdMask(MaskT mask) : m(mask) {}

    bool operator[](int i) const { return m[i]; }
};

/// @brief  SIMD value wrapper for Backend A.
template<typename T, int W>
struct Simd
{
    using value_type = T;
    using VecT = stdx::fixed_size_simd<T, W>;
    VecT v;

    Simd() = default;
    /*implicit*/ Simd(T scalar) : v(scalar) {}   // broadcast
    explicit     Simd(const T* p) : v(p, stdx::element_aligned) {} // load
    /*implicit*/ Simd(VecT vec) : v(vec) {}

    /// @brief  Extract lane 0.  When all lanes hold the same value (e.g.
    ///   after hmin/hmax), this converts a SIMD result back to a scalar.
    explicit operator T() const { return v[0]; }

    T operator[](int i) const { return v[i]; }

    void store(T* p) const { v.copy_to(p, stdx::element_aligned); }

    // Arithmetic operators
    Simd operator-() const { return Simd(-v); }
    Simd operator+(Simd o) const { return Simd(v + o.v); }
    Simd operator-(Simd o) const { return Simd(v - o.v); }
    Simd operator*(Simd o) const { return Simd(v * o.v); }
    Simd operator/(Simd o) const { return Simd(v / o.v); }

    // Compound assignment
    Simd& operator+=(Simd o) { v += o.v; return *this; }
    Simd& operator-=(Simd o) { v -= o.v; return *this; }
    Simd& operator*=(Simd o) { v *= o.v; return *this; }
    Simd& operator/=(Simd o) { v /= o.v; return *this; }

    // Comparison — return mask
    SimdMask<T,W> operator>=(Simd o) const { return SimdMask<T,W>(v >= o.v); }
    SimdMask<T,W> operator<=(Simd o) const { return SimdMask<T,W>(v <= o.v); }
    SimdMask<T,W> operator> (Simd o) const { return SimdMask<T,W>(v >  o.v); }
    SimdMask<T,W> operator< (Simd o) const { return SimdMask<T,W>(v <  o.v); }
    SimdMask<T,W> operator==(Simd o) const { return SimdMask<T,W>(v == o.v); }
    SimdMask<T,W> operator!=(Simd o) const { return SimdMask<T,W>(v != o.v); }
};

// Mixed scalar-SIMD operators
template<typename T, int W> Simd<T,W> operator+(T a, Simd<T,W> b) { return Simd<T,W>(a) + b; }
template<typename T, int W> Simd<T,W> operator+(Simd<T,W> a, T b) { return a + Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator-(T a, Simd<T,W> b) { return Simd<T,W>(a) - b; }
template<typename T, int W> Simd<T,W> operator-(Simd<T,W> a, T b) { return a - Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator*(T a, Simd<T,W> b) { return Simd<T,W>(a) * b; }
template<typename T, int W> Simd<T,W> operator*(Simd<T,W> a, T b) { return a * Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator/(T a, Simd<T,W> b) { return Simd<T,W>(a) / b; }
template<typename T, int W> Simd<T,W> operator/(Simd<T,W> a, T b) { return a / Simd<T,W>(b); }

// SIMD free functions — Backend A

template<typename T, int W>
Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    auto result = b.v;
    stdx::where(mask.m, result) = a.v;
    return Simd<T,W>(result);
}

template<typename T, int W>
Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) { return Simd<T,W>(stdx::min(a.v, b.v)); }

template<typename T, int W>
Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) { return Simd<T,W>(stdx::max(a.v, b.v)); }

template<typename T, int W>
Simd<T,W> sqrt(Simd<T,W> v) { return Simd<T,W>(stdx::sqrt(v.v)); }

template<typename T, int W>
Simd<T,W> abs(Simd<T,W> v) { return Simd<T,W>(stdx::abs(v.v)); }

/// @brief  Horizontal min — reduces all lanes to the minimum value and
///   broadcasts it back to all lanes.  The result stays in Simd<T,W> so
///   subsequent arithmetic (e.g. equality comparison to find winner lane)
///   remains in the same type domain.  Use explicit operator T() to extract
///   the scalar at the write boundary: T result = T(hmin(v));
template<typename T, int W>
Simd<T,W> hmin(Simd<T,W> v) {
    // stdx::reduce passes intermediate simd chunks to the binary op (tree reduction),
    // so the lambda must use auto parameters and stdx::min for element-wise selection.
    T m = stdx::reduce(v.v, [](auto a, auto b){ return stdx::min(a, b); });
    return Simd<T,W>(m);
}

template<typename T, int W>
Simd<T,W> hmax(Simd<T,W> v) {
    T m = stdx::reduce(v.v, [](auto a, auto b){ return stdx::max(a, b); });
    return Simd<T,W>(m);
}

/// @brief  Returns true if all lanes of the mask are set.
template<typename T, int W>
bool hall(SimdMask<T,W> mask) { return stdx::all_of(mask.m); }

/// @brief  Returns true if any lane of the mask is set.
template<typename T, int W>
bool hany(SimdMask<T,W> mask) { return stdx::any_of(mask.m); }

/// @brief  Returns the index of the first set lane, or -1 if none.
template<typename T, int W>
int hfirst(SimdMask<T,W> mask) {
    for (int i = 0; i < W; ++i) if (mask[i]) return i;
    return -1;
}

// ============================================================================
// Backend B: std::array fallback — auto-vectorizer target
// ============================================================================
#else // !OPENVDB_USE_STD_SIMD

/// @brief  SIMD mask wrapper for Backend B.
template<typename T, int W>
struct SimdMask
{
    std::array<bool, W> m{};

    SimdMask() = default;

    bool operator[](int i) const { return m[i]; }
    bool& operator[](int i) { return m[i]; }
};

/// @brief  SIMD value wrapper for Backend B.  Element-wise operator loops are
///   the auto-vectorizer target; fixed-count, no struct indirection.
template<typename T, int W>
struct Simd
{
    using value_type = T;
    std::array<T, W> v{};

    Simd() = default;
    /*implicit*/ Simd(T scalar) { v.fill(scalar); }        // broadcast
    explicit     Simd(const T* p) {                         // load
        for (int i = 0; i < W; ++i) v[i] = p[i];
    }

    /// @brief  Extract lane 0.  See hmin() for the intended usage pattern.
    explicit operator T() const { return v[0]; }

    T  operator[](int i) const { return v[i]; }
    T& operator[](int i)       { return v[i]; }

    void store(T* p) const {
        for (int i = 0; i < W; ++i) p[i] = v[i];
    }

    Simd operator-() const {
        Simd r; for (int i = 0; i < W; ++i) r[i] = -v[i]; return r;
    }
    Simd operator+(Simd o) const {
        Simd r; for (int i = 0; i < W; ++i) r[i] = v[i] + o[i]; return r;
    }
    Simd operator-(Simd o) const {
        Simd r; for (int i = 0; i < W; ++i) r[i] = v[i] - o[i]; return r;
    }
    Simd operator*(Simd o) const {
        Simd r; for (int i = 0; i < W; ++i) r[i] = v[i] * o[i]; return r;
    }
    Simd operator/(Simd o) const {
        Simd r; for (int i = 0; i < W; ++i) r[i] = v[i] / o[i]; return r;
    }

    Simd& operator+=(Simd o) { for (int i = 0; i < W; ++i) v[i] += o[i]; return *this; }
    Simd& operator-=(Simd o) { for (int i = 0; i < W; ++i) v[i] -= o[i]; return *this; }
    Simd& operator*=(Simd o) { for (int i = 0; i < W; ++i) v[i] *= o[i]; return *this; }
    Simd& operator/=(Simd o) { for (int i = 0; i < W; ++i) v[i] /= o[i]; return *this; }

    SimdMask<T,W> operator>=(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] >= o[i]; return r;
    }
    SimdMask<T,W> operator<=(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] <= o[i]; return r;
    }
    SimdMask<T,W> operator>(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] > o[i]; return r;
    }
    SimdMask<T,W> operator<(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] < o[i]; return r;
    }
    SimdMask<T,W> operator==(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] == o[i]; return r;
    }
    SimdMask<T,W> operator!=(Simd o) const {
        SimdMask<T,W> r; for (int i = 0; i < W; ++i) r[i] = v[i] != o[i]; return r;
    }
};

// Mixed scalar-SIMD operators
template<typename T, int W> Simd<T,W> operator+(T a, Simd<T,W> b) { return Simd<T,W>(a) + b; }
template<typename T, int W> Simd<T,W> operator+(Simd<T,W> a, T b) { return a + Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator-(T a, Simd<T,W> b) { return Simd<T,W>(a) - b; }
template<typename T, int W> Simd<T,W> operator-(Simd<T,W> a, T b) { return a - Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator*(T a, Simd<T,W> b) { return Simd<T,W>(a) * b; }
template<typename T, int W> Simd<T,W> operator*(Simd<T,W> a, T b) { return a * Simd<T,W>(b); }
template<typename T, int W> Simd<T,W> operator/(T a, Simd<T,W> b) { return Simd<T,W>(a) / b; }
template<typename T, int W> Simd<T,W> operator/(Simd<T,W> a, T b) { return a / Simd<T,W>(b); }

// SIMD free functions — Backend B

template<typename T, int W>
Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; ++i) r[i] = mask[i] ? a[i] : b[i]; return r;
}

template<typename T, int W>
Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; ++i) r[i] = a[i] < b[i] ? a[i] : b[i]; return r;
}

template<typename T, int W>
Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; ++i) r[i] = a[i] > b[i] ? a[i] : b[i]; return r;
}

template<typename T, int W>
Simd<T,W> sqrt(Simd<T,W> v) {
    Simd<T,W> r; for (int i = 0; i < W; ++i) r[i] = std::sqrt(v[i]); return r;
}

template<typename T, int W>
Simd<T,W> abs(Simd<T,W> v) {
    Simd<T,W> r; for (int i = 0; i < W; ++i) r[i] = std::abs(v[i]); return r;
}

template<typename T, int W>
Simd<T,W> hmin(Simd<T,W> v) {
    T m = v[0]; for (int i = 1; i < W; ++i) m = m < v[i] ? m : v[i];
    return Simd<T,W>(m);
}

template<typename T, int W>
Simd<T,W> hmax(Simd<T,W> v) {
    T m = v[0]; for (int i = 1; i < W; ++i) m = m > v[i] ? m : v[i];
    return Simd<T,W>(m);
}

template<typename T, int W>
bool hall(SimdMask<T,W> mask) {
    for (int i = 0; i < W; ++i) if (!mask[i]) return false;
    return true;
}

template<typename T, int W>
bool hany(SimdMask<T,W> mask) {
    for (int i = 0; i < W; ++i) if (mask[i]) return true;
    return false;
}

template<typename T, int W>
int hfirst(SimdMask<T,W> mask) {
    for (int i = 0; i < W; ++i) if (mask[i]) return i;
    return -1;
}

#endif // OPENVDB_USE_STD_SIMD

// ============================================================================
// Scalar overloads — always present.
//
// These make the Generic-T pattern complete: a kernel templated on T compiles
// identically for T=float (scalar/GPU) and T=Simd<float,W> (CPU batch).
// The scalar overloads are the identity in every case.
// ============================================================================

template<typename T> T    where(bool m, T a, T b) { return m ? a : b; }
template<typename T> T    min(T a, T b)            { return a < b ? a : b; }
template<typename T> T    max(T a, T b)            { return a > b ? a : b; }
template<typename T> T    sqrt(T v)                { return std::sqrt(v); }
template<typename T> T    abs(T v)                 { return std::abs(v); }
template<typename T> T    hmin(T v)                { return v; } // identity: one lane
template<typename T> T    hmax(T v)                { return v; } // identity: one lane
inline bool               hall(bool b)             { return b; } // identity: one lane
inline bool               hany(bool b)             { return b; } // identity: one lane
inline int                hfirst(bool b)           { return b ? 0 : -1; }

// square — useful shorthand, defined after scalar overloads so T works for both
template<typename T> T square(T v) { return v * v; }

// ============================================================================
// Scalar<T> — extract the underlying element type from T.
//
//   scalar_t<float>          = float   (primary: T has no value_type)
//   scalar_t<Simd<float,W>>  = float   (specialization: T::value_type exists)
//
// Detection uses std::void_t so it generalises to any type with a value_type
// member, without an explicit Simd<T,W> specialization.
// ============================================================================

template<typename T, typename = void>
struct Scalar { using type = T; };

template<typename T>
struct Scalar<T, std::void_t<typename T::value_type>> {
    using type = typename T::value_type;
};

template<typename T>
using scalar_t = typename Scalar<T>::type;

// ============================================================================
// SimdTraits<SimdT> — compile-time lane count for a SIMD type.
//
//   SimdTraits<float>::size           = 1
//   SimdTraits<Simd<float,W>>::size   = W
// ============================================================================

template<typename T>
struct SimdTraits { static constexpr size_t size = 1; };

template<typename T, int W>
struct SimdTraits<Simd<T,W>> { static constexpr size_t size = size_t(W); };

// ============================================================================
// SimdNativeT<T> — select the natural SIMD type for element type T.
//
//   SimdNativeT<double>::Type on AVX  → Simd<double, 4>  (256 / 64 bits)
//   SimdNativeT<float>::Type  on AVX  → Simd<float,  8>  (256 / 32 bits)
//   SimdNativeT<double>::Type no ISA  → Simd<double, 1>  (scalar fallback)
// ============================================================================

template<typename T>
struct SimdNativeT
{
    static constexpr size_t elemBits = sizeof(T) * CHAR_BIT;
    static constexpr size_t width    = (RegisterBits > 0)
        ? (RegisterBits / elemBits) : 1;
    using Type = Simd<T, int(width)>;
};

} // namespace simd
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_SIMD_SIMD_HAS_BEEN_INCLUDED
