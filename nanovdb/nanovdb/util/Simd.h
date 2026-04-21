#pragma once
#include <array>
#include <cstdint>

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

// [[gnu::always_inline]] forces these thin wrappers to inline at every
// call site.  Without it, GCC's cost model sometimes outlines them —
// each call then pays a function-call + vzeroupper + register-ABI
// transition that dominates the one-instruction body (vminps / vmaxps /
// vblendvps).  See BatchAccessor.md §8h for the analogous fix on the
// StencilAccessor path.
template<typename T, int W>
[[gnu::always_inline]] inline Simd<T,W> min(Simd<T,W> a, Simd<T,W> b) { return stdx::min(a, b); }

template<typename T, int W>
[[gnu::always_inline]] inline Simd<T,W> max(Simd<T,W> a, Simd<T,W> b) { return stdx::max(a, b); }

// TS v2 where(mask, v) is a masked assignment proxy, not a 3-arg select.
// Wrap it into the select(mask, a, b) form our kernels expect.
template<typename T, int W>
[[gnu::always_inline]] inline Simd<T,W> where(SimdMask<T,W> mask, Simd<T,W> a, Simd<T,W> b) {
    auto result = b;
    stdx::where(mask, result) = a;
    return result;
}
// Heterogeneous where: mask element type U ≠ value element type T.
// Converts the U-mask to a T-mask via a boolean round-trip.
template<typename T, typename U, int W>
[[gnu::always_inline]] inline Simd<T,W> where(SimdMask<U,W> mask, Simd<T,W> a, Simd<T,W> b) {
    bool arr[W];
    for (int i = 0; i < W; i++) arr[i] = static_cast<bool>(mask[i]);
    SimdMask<T,W> tmask(arr, element_aligned);
    auto result = b;
    stdx::where(tmask, result) = a;
    return result;
}

// 2-argument where: stdx-style masked-assignment proxy.
// where(mask, target) = value  writes value[i] into target[i] for lanes where mask[i] is true.
// Heterogeneous mask (mask element type U may differ from value element type T).
// stdx::fixed_size_simd operator[] returns by value, so the assignment delegates to
// a boolean round-trip + stdx::where rather than a per-lane scalar store.
template<typename T, typename U, int W>
struct WhereExpression {
    const SimdMask<U,W>& mask;
    Simd<T,W>& target;
    WhereExpression& operator=(const Simd<T,W>& value) {
        bool arr[W];
        for (int i = 0; i < W; ++i) arr[i] = static_cast<bool>(mask[i]);
        SimdMask<T,W> tmask(arr, element_aligned);
        stdx::where(tmask, target) = value;
        return *this;
    }
};
template<typename T, typename U, int W>
inline WhereExpression<T,U,W> where(const SimdMask<U,W>& mask, Simd<T,W>& target) {
    return {mask, target};
}

// Horizontal reduction: delegates to stdx::reduce.
// Mirrors std::experimental::reduce(v, binary_op) — same signature, same semantics.
// Use with std::bit_or<>{}, std::bit_and<>{}, std::plus<>{}, etc.
template<typename T, int W, typename BinaryOp>
inline T reduce(Simd<T,W> v, BinaryOp op) { return stdx::reduce(v, op); }

template<typename T, int W>
inline bool any_of(SimdMask<T,W> m) { return stdx::any_of(m); }
template<typename T, int W>
inline bool none_of(SimdMask<T,W> m) { return stdx::none_of(m); }
template<typename T, int W>
inline bool all_of(SimdMask<T,W> m) { return stdx::all_of(m); }

// Store W lanes of v into p[0..W-1] (stdx calls this copy_to).
template<typename T, int W>
inline void store(Simd<T,W> v, T* p, element_aligned_tag = {}) {
    v.copy_to(p, element_aligned);
}

// Unmasked gather: result[i] = ptr[idx[i]] for all lanes.
// IdxT may be int32_t or int64_t; the compiler selects the matching hardware
// instruction (vpgatherdps/vpgatherdq for 32-bit idx, vpgatherqq for 64-bit idx).
template<typename T, typename IdxT, int W>
inline Simd<T,W> gather(const T* __restrict__ ptr, Simd<IdxT,W> idx) {
    return Simd<T,W>([&](int i) { return ptr[idx[i]]; });
}

// Masked gather: result[i] = mask[i] ? ptr[idx[i]] : fallback.
// Implemented as a full gather + where-blend; ptr is accessed for ALL lanes,
// so every idx[i] must be a valid offset regardless of mask[i].
template<typename T, typename IdxT, int W>
inline Simd<T,W> gather(SimdMask<T,W> mask, const T* __restrict__ ptr,
                         Simd<IdxT,W> idx, T fallback = T(0)) {
    auto result = Simd<T,W>(fallback);
    stdx::where(mask, result) = Simd<T,W>([&](int i) { return ptr[idx[i]]; });
    return result;
}

// Merge-masked gather: dst[i] = mask[i] ? ptr[idx[i]] : dst[i]  (unchanged).
// MaskElemT may differ from T (heterogeneous mask, e.g. SimdMask<uint32_t,W>
// applied to Simd<uint64_t,W> data).  When T==MaskElemT, delegates directly
// to stdx::where; otherwise uses the WhereExpression boolean round-trip.
template<typename T, typename IdxT, typename MaskElemT, int W>
inline void gather_if(Simd<T,W>& dst, SimdMask<MaskElemT,W> mask,
                       const T* __restrict__ ptr, Simd<IdxT,W> idx) {
    if constexpr (std::is_same_v<T, MaskElemT>) {
        stdx::where(mask, dst) = Simd<T,W>([&](int i) { return ptr[idx[i]]; });
    } else {
        where(mask, dst) = Simd<T,W>([&](int i) { return ptr[idx[i]]; });
    }
}

// ===========================================================================
// Implementation B: std::array backend (default)
// ===========================================================================
#else

template<typename T, int W>
struct SimdMask {
    std::array<bool, W> data{};
    SimdMask() = default;
    NANOVDB_SIMD_HOSTDEV explicit SimdMask(const bool* p, element_aligned_tag) {
        for (int i = 0; i < W; i++) data[i] = p[i];
    }
    // Converting constructor: copy bool values from a mask over a different element type.
    // All SimdMask<U,W> are boolean arrays of the same width; this allows
    // where(SimdMask<float,W>, Simd<uint16_t,W>, Simd<uint16_t,W>) without explicit casting.
    template<typename U>
    NANOVDB_SIMD_HOSTDEV explicit SimdMask(SimdMask<U,W> const& o) {
        for (int i = 0; i < W; i++) data[i] = o[i];
    }
    NANOVDB_SIMD_HOSTDEV bool  operator[](int i) const { return data[i]; }
    NANOVDB_SIMD_HOSTDEV bool& operator[](int i)       { return data[i]; }
    NANOVDB_SIMD_HOSTDEV SimdMask operator!() const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = !data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV SimdMask operator&(SimdMask o) const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = data[i] && o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV SimdMask operator|(SimdMask o) const {
        SimdMask r; for (int i = 0; i < W; i++) r.data[i] = data[i] || o.data[i]; return r;
    }
};

template<typename T, int W>
struct Simd {
    std::array<T, W> data{};

    Simd() = default;
    NANOVDB_SIMD_HOSTDEV Simd(T scalar) { data.fill(scalar); }               // broadcast
    NANOVDB_SIMD_HOSTDEV explicit Simd(const T* p, element_aligned_tag) { // load
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
    NANOVDB_SIMD_HOSTDEV SimdMask<T,W> operator==(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] == o.data[i]; return m;
    }
    NANOVDB_SIMD_HOSTDEV SimdMask<T,W> operator!=(Simd o) const {
        SimdMask<T,W> m; for (int i = 0; i < W; i++) m.data[i] = data[i] != o.data[i]; return m;
    }
    // Bitwise and shift operators — valid for integer element types.
    NANOVDB_SIMD_HOSTDEV Simd operator|(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] | o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator&(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] & o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator^(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] ^ o.data[i]; return r;
    }
    // Per-lane variable shift (shift count from corresponding lane of o).
    NANOVDB_SIMD_HOSTDEV Simd operator<<(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] << o.data[i]; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator>>(Simd o) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] >> o.data[i]; return r;
    }
    // Uniform shift: all lanes shifted by the same scalar count (vpsllw imm8 / vpsrlw imm8).
    NANOVDB_SIMD_HOSTDEV Simd operator<<(T shift) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] << shift; return r;
    }
    NANOVDB_SIMD_HOSTDEV Simd operator>>(T shift) const {
        Simd r; for (int i = 0; i < W; i++) r.data[i] = data[i] >> shift; return r;
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
// Heterogeneous where: mask element type U need not match value element type T.
// Useful for applying PredicateT=SimdMask<float,W> to VoxelOffsetT=Simd<uint16_t,W>.
template<typename T, typename U, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> where(SimdMask<U,W> mask, Simd<T,W> a, Simd<T,W> b) {
    Simd<T,W> r; for (int i = 0; i < W; i++) r[i] = mask[i] ? a[i] : b[i]; return r;
}

// 2-argument where: stdx-style masked-assignment proxy.
// where(mask, target) = value  writes value[i] into target[i] for lanes where mask[i] is true.
// Heterogeneous mask (mask element type U may differ from value element type T).
template<typename T, typename U, int W>
struct WhereExpression {
    const SimdMask<U,W>& mask;
    Simd<T,W>& target;
    NANOVDB_SIMD_HOSTDEV WhereExpression& operator=(const Simd<T,W>& value) {
        for (int i = 0; i < W; ++i)
            if (mask[i]) target[i] = value[i];
        return *this;
    }
};
template<typename T, typename U, int W>
NANOVDB_SIMD_HOSTDEV WhereExpression<T,U,W> where(const SimdMask<U,W>& mask, Simd<T,W>& target) {
    return {mask, target};
}

// Horizontal reduction: fold all lanes with a binary operator.
// Mirrors std::experimental::reduce(v, binary_op).
// Use with std::bit_or<>{}, std::bit_and<>{}, std::plus<>{}, etc.
template<typename T, int W, typename BinaryOp>
NANOVDB_SIMD_HOSTDEV T reduce(Simd<T,W> v, BinaryOp op) {
    T r = v[0];
    for (int i = 1; i < W; ++i) r = op(r, v[i]);
    return r;
}

template<typename T, int W>
NANOVDB_SIMD_HOSTDEV bool any_of(SimdMask<T,W> m) {
    bool r = false; for (int i = 0; i < W; i++) r |= m[i]; return r;
}
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV bool none_of(SimdMask<T,W> m) { return !any_of(m); }
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV bool all_of(SimdMask<T,W> m) {
    bool r = true; for (int i = 0; i < W; i++) r &= m[i]; return r;
}

// Store W lanes of v into p[0..W-1] (array-backend passthrough to member).
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV void store(Simd<T,W> v, T* p, element_aligned_tag = {}) {
    v.store(p);
}

// Unmasked gather: result[i] = ptr[idx[i]] for all lanes.
template<typename T, typename IdxT, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> gather(const T* __restrict__ ptr, Simd<IdxT,W> idx) {
    Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = ptr[idx[i]];
    return r;
}

// Masked gather: result[i] = mask[i] ? ptr[idx[i]] : fallback.
// Scalar path: accesses ptr only for true lanes (ternary short-circuits).
template<typename T, typename IdxT, int W>
NANOVDB_SIMD_HOSTDEV Simd<T,W> gather(SimdMask<T,W> mask, const T* __restrict__ ptr,
                                        Simd<IdxT,W> idx, T fallback = T(0)) {
    Simd<T,W> r;
    for (int i = 0; i < W; i++) r[i] = mask[i] ? ptr[idx[i]] : fallback;
    return r;
}

// Merge-masked gather: dst[i] = mask[i] ? ptr[idx[i]] : dst[i]  (unchanged).
// MaskElemT may differ from T (heterogeneous mask).
// Scalar path: only accesses ptr for true lanes.
template<typename T, typename IdxT, typename MaskElemT, int W>
NANOVDB_SIMD_HOSTDEV void gather_if(Simd<T,W>& dst, SimdMask<MaskElemT,W> mask,
                                     const T* __restrict__ ptr, Simd<IdxT,W> idx) {
    for (int i = 0; i < W; i++)
        if (mask[i]) dst[i] = ptr[idx[i]];
}

#endif // NANOVDB_USE_STD_SIMD

// ---------------------------------------------------------------------------
// simd_cast<DstT> — element-wise static_cast between Simd types of the same W.
//
// Used for widening (uint16_t → uint32_t, uint32_t → uint64_t) and for
// reinterpreting signedness (uint32_t → int32_t) when building gather indices.
// Both backends: the array backend uses a lane loop; the stdx backend uses the
// generator constructor, which the compiler lowers to a vpmovsxbw / vpmovzxwd
// sequence or similar sign/zero-extend instruction depending on the types.
// Scalar overload: degrades to static_cast for plain scalar types.
// ---------------------------------------------------------------------------
template<typename DstT, typename SrcT, int W>
NANOVDB_SIMD_HOSTDEV Simd<DstT,W> simd_cast(Simd<SrcT,W> src) {
#ifdef NANOVDB_USE_STD_SIMD
    return Simd<DstT,W>([&](int i) { return static_cast<DstT>(src[i]); });
#else
    Simd<DstT,W> r;
    for (int i = 0; i < W; ++i) r[i] = static_cast<DstT>(src[i]);
    return r;
#endif
}
template<typename DstT, typename SrcT>
NANOVDB_SIMD_HOSTDEV DstT simd_cast(SrcT src) { return static_cast<DstT>(src); }

// ---------------------------------------------------------------------------
// simd_cast_if — masked element-wise cast (merge-masked).
//
// dst[i] = mask[i] ? static_cast<DstT>(src[i]) : dst[i]   (unchanged)
//
// Typical use: widen an integer index type into a wider type before arithmetic,
// keeping invalid (masked-out) lanes at their initial value (usually 0).
// On AVX-512 the compiler may emit a single masked vcvt/vpmovzx instruction.
// On AVX2 it lowers to an unmasked cast + blend.
//
// Scalar fallback: plain conditional cast.
// ---------------------------------------------------------------------------
template<typename DstT, typename SrcT, typename MaskElemT, int W>
NANOVDB_SIMD_HOSTDEV void simd_cast_if(Simd<DstT,W>& dst, SimdMask<MaskElemT,W> mask, Simd<SrcT,W> src) {
    dst = where(mask, simd_cast<DstT, SrcT, W>(src), dst);
}
template<typename DstT, typename SrcT>
NANOVDB_SIMD_HOSTDEV void simd_cast_if(DstT& dst, bool mask, SrcT src) {
    if (mask) dst = static_cast<DstT>(src);
}

// ---------------------------------------------------------------------------
// popcount64 — scalar SWAR popcount, always uses arithmetic (no __builtin_popcountll).
//
// Safe to call per-lane inside a vectorizable loop: every operation (>>, &, +, -)
// maps to an AVX2 instruction for 64-bit elements (vpsrlq, vpand, vpaddq, vpsubq).
// The final byte-sum uses a shift-and-add tree instead of the multiply trick
// (v * 0x0101...) since 64x64->64 multiply has no AVX2 equivalent (vpmullq is AVX-512).
// ---------------------------------------------------------------------------
NANOVDB_SIMD_HOSTDEV inline uint64_t popcount64(uint64_t v)
{
    v -= (v >> 1) & uint64_t(0x5555555555555555);
    v  = (v & uint64_t(0x3333333333333333)) + ((v >> 2) & uint64_t(0x3333333333333333));
    v  = (v + (v >> 4)) & uint64_t(0x0F0F0F0F0F0F0F0F);  // per-byte counts
    v += v >> 8;   v &= uint64_t(0x00FF00FF00FF00FF);
    v += v >> 16;  v &= uint64_t(0x0000FFFF0000FFFF);
    v += v >> 32;
    return v & uint64_t(63);
}

// Lane-wise SIMD popcount: applies popcount64 to every lane.
// Backend A: generator constructor; Backend B: element loop (auto-vectorized by GCC/Clang).
template<int W>
NANOVDB_SIMD_HOSTDEV Simd<uint64_t,W> popcount(Simd<uint64_t,W> v) {
#ifdef NANOVDB_USE_STD_SIMD
    return Simd<uint64_t,W>([&](int i) { return popcount64(v[i]); });
#else
    Simd<uint64_t,W> r;
    for (int i = 0; i < W; ++i) r[i] = popcount64(v[i]);
    return r;
#endif
}
NANOVDB_SIMD_HOSTDEV inline uint64_t popcount(uint64_t v) { return popcount64(v); }

// ---------------------------------------------------------------------------
// simd_traits — generic per-lane access for scalar and Simd<T,W> types.
//
// Lets algorithms be written once and work for both scalar (width=1) and
// vector (width=W) instantiations.  The class does not need to know whether
// it is working with scalars or SIMD vectors.
//
// Primary template: scalar types.
// Specializations below: Simd<T,W> and SimdMask<T,W> (both backends).
// ---------------------------------------------------------------------------
template<typename T>
struct simd_traits {
    static constexpr int width = 1;
    using scalar_type = T;
    NANOVDB_SIMD_HOSTDEV static T    get(T v, int)         { return v; }
    NANOVDB_SIMD_HOSTDEV static void set(T& v, int, T val) { v = val; }
};

template<>
struct simd_traits<bool> {
    static constexpr int width = 1;
    using scalar_type = bool;
    NANOVDB_SIMD_HOSTDEV static bool get(bool m, int)          { return m; }
    NANOVDB_SIMD_HOSTDEV static void set(bool& m, int, bool v) { m = v; }
};

// Simd<T,W> and SimdMask<T,W>: valid for both backends because the aliases
// are already resolved by the time these specializations are instantiated.
template<typename T, int W>
struct simd_traits<Simd<T,W>> {
    static constexpr int width = W;
    using scalar_type = T;
    NANOVDB_SIMD_HOSTDEV static T    get(Simd<T,W> v, int i)         { return v[i]; }
    NANOVDB_SIMD_HOSTDEV static void set(Simd<T,W>& v, int i, T val) { v[i] = val; }
};

template<typename T, int W>
struct simd_traits<SimdMask<T,W>> {
    static constexpr int width = W;
    using scalar_type = bool;
    NANOVDB_SIMD_HOSTDEV static bool get(SimdMask<T,W> m, int i)          { return m[i]; }
    NANOVDB_SIMD_HOSTDEV static void set(SimdMask<T,W>& m, int i, bool v) { m[i] = v; }
};

// ---------------------------------------------------------------------------
// scalar_traits — extract the scalar element type from T or Simd<T,W>.
//
// Primary template: a plain scalar type is its own element type.
// The = void default parameter reserves a slot for enable_if specialisations.
// Specialisation for Simd<T,W>: the element type is T.
// scalar_traits_t<U> is a convenience alias for typename scalar_traits<U>::type.
// ---------------------------------------------------------------------------
template<typename T, typename = void>
struct scalar_traits { using type = T; };

template<typename T, int W>
struct scalar_traits<Simd<T,W>> { using type = T; };

template<typename T>
using scalar_traits_t = typename scalar_traits<T>::type;


// ---------------------------------------------------------------------------
// to_bitmask — fold SimdMask<T,W> into a uint32_t (one bit per lane).
// T is the associated element type; only W matters.  Requires W <= 32.
// ---------------------------------------------------------------------------
template<typename T, int W>
NANOVDB_SIMD_HOSTDEV uint32_t to_bitmask(SimdMask<T,W> m) {
    static_assert(W <= 32, "to_bitmask: W must be <= 32");
    uint32_t r = 0;
    for (int i = 0; i < W; i++) if (m[i]) r |= (1u << i);
    return r;
}

// ---------------------------------------------------------------------------
// Scalar overloads — always present, for T=float (GPU / scalar path)
// ---------------------------------------------------------------------------
template<typename T> NANOVDB_SIMD_HOSTDEV T min(T a, T b)           { return a < b ? a : b; }
template<typename T> NANOVDB_SIMD_HOSTDEV T max(T a, T b)           { return a > b ? a : b; }
template<typename T> NANOVDB_SIMD_HOSTDEV T where(bool m, T a, T b) { return m ? a : b; }
template<typename T, typename BinaryOp>
NANOVDB_SIMD_HOSTDEV T reduce(T v, BinaryOp) { return v; }

// 2-argument where: scalar masked-assignment proxy matching the Simd form.
// where(mask, target) = value  writes value into target only if mask is true.
template<typename T>
struct ScalarWhereProxy {
    bool mask; T& target;
    NANOVDB_SIMD_HOSTDEV void operator=(const T& v) { if (mask) target = v; }
};
template<typename T>
NANOVDB_SIMD_HOSTDEV ScalarWhereProxy<T> where(bool mask, T& target) {
    return {mask, target};
}

// Unmasked scalar gather: result = ptr[idx].
template<typename T, typename IdxT>
NANOVDB_SIMD_HOSTDEV T gather(const T* __restrict__ ptr, IdxT idx) { return ptr[idx]; }

// Merge-masked scalar gather: dst = ptr[idx] only if mask, else dst unchanged.
template<typename T, typename IdxT>
NANOVDB_SIMD_HOSTDEV void gather_if(T& dst, bool mask, const T* __restrict__ ptr, IdxT idx) {
    if (mask) dst = ptr[idx];
}

} // namespace util
} // namespace nanovdb
