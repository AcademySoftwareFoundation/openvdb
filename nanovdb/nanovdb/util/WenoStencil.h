// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file WenoStencil.h

    \brief 19-tap WENO5 stencil data container + per-tap out-of-band
           extrapolation, templated on SIMD lane width.

    `WenoStencil<W>` holds the per-tap float values and per-tap activity
    flags for a single voxel (W=1, scalar / GPU-friendly) or a batch of W
    voxels (W>1, CPU SIMD).  The underlying element types switch via
    `std::conditional_t`:

       W == 1 :  ValueT = float            PredT = bool
       W >  1 :  ValueT = float[W]         PredT = bool[W]

    Storage is a plain C array (`ValueT mValues[SIZE]`) so the caller can
    fill it lane-by-lane with the same scalar syntax in both cases
    (`s.mValues[k][i] = ...` at W>1; `s.mValues[k] = ...` at W=1).

    The class's one substantive operation is `extrapolate(|background|)`,
    which repairs out-of-band lanes (mIsActive[k] == false) by applying
    copysign(|background|, mValues[innerTap]) via an ascending-|Δ|
    cascade.  After `extrapolate` returns, every tap holds either its
    true sidecar value (for active lanes) or a sign-corrected background
    magnitude (for inactive lanes) — ready for WENO5 arithmetic.

    The inner-tap mapping is spelled out explicitly (Weno5-specific,
    non-generic on purpose):

       |Δ|=1 taps  -->  inner = center tap (0,0,0)
       |Δ|=2 taps  -->  inner = |Δ|=1 tap on the same axis
       |Δ|=3 taps  -->  inner = |Δ|=2 tap on the same axis

    Cascade order (ascending-|Δ|) guarantees the inner tap is already
    resolved when the outer tap is processed, so distance-3 taps inherit
    sign via the |Δ|=1 → |Δ|=2 → |Δ|=3 chain without special casing.

    See BatchAccessor.md §11 for the full Phase-2 sidecar-WENO pipeline
    design and §11.2 for the extrapolation semantics.
*/

#pragma once

#include <nanovdb/util/Simd.h>
#include <nanovdb/util/StencilAccessor.h>  // StencilPoint, Weno5Stencil, detail::findIndex

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nanovdb {

template<int W = 1>
class WenoStencil
{
public:
    using Taps = Weno5Stencil::Taps;
    using Hull = Weno5Stencil::Hull;
    static constexpr int SIZE = int(std::tuple_size_v<Taps>);

    // Per-lane storage shape chosen by W:
    //   W == 1 :  plain scalar (GPU thread-per-voxel model)
    //   W >  1 :  W-wide array (CPU SIMD batch)
    using ValueT = std::conditional_t<W == 1, float, float[W]>;
    using PredT  = std::conditional_t<W == 1, bool,  bool[W]>;

    alignas(64) ValueT mValues  [SIZE];
    alignas(64) PredT  mIsActive[SIZE];

    static constexpr int size() { return SIZE; }

    // Compile-time named-tap access: returns the index of tap (DI,DJ,DK)
    // in the Taps tuple, matching StencilAccessor's convention.
    template<int DI, int DJ, int DK>
    static constexpr int tapIndex()
    {
        constexpr int I = detail::findIndex<Taps, DI, DJ, DK>(
            std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "WenoStencil::tapIndex: tap not in stencil");
        return I;
    }

    // Replace out-of-band lanes (mIsActive[k][i] == false) of mValues[k]
    // with copysign(absBackground, mValues[innerTap][i]).  Active lanes
    // are untouched.  Center tap (0,0,0) is assumed always in-band and
    // is not processed.
    //
    // Requires absBackground >= 0 (caller typically passes
    // std::abs(floatGrid.background()) or sidecar[0] for a narrow-band
    // level set where background > 0).
    void extrapolate(float absBackground);

private:
    // Bridge W=1 (scalar reference) and W>1 (array decays to pointer).
    // The address taken at W=1 is to the scalar member of mValues/mIsActive;
    // at W>1 an array-to-pointer decay works without extra syntax.
    static constexpr       float* addr(      ValueT& v) noexcept {
        if constexpr (W == 1) return &v; else return v;
    }
    static constexpr const float* addr(const ValueT& v) noexcept {
        if constexpr (W == 1) return &v; else return v;
    }
    static constexpr       bool*  addr(      PredT& p) noexcept {
        if constexpr (W == 1) return &p; else return p;
    }
    static constexpr const bool*  addr(const PredT& p) noexcept {
        if constexpr (W == 1) return &p; else return p;
    }

    // Hardcoded (tap, innerTap) pairs for Weno5Stencil::Taps, ordered by
    // ascending |Δ|.  Indices match the tuple definition in StencilAccessor.h.
    //
    //   idx  0     :  center     ( 0, 0, 0)
    //   idx  1.. 6 :  x-axis     (-3..+3)
    //   idx  7..12 :  y-axis     (-3..+3)
    //   idx 13..18 :  z-axis     (-3..+3)
    static constexpr int kNumPairs = 18;
    static constexpr int kPairs[kNumPairs][2] = {
        // |Δ|=1  (inner tap = center, idx 0)
        { 3, 0}, { 4, 0},        // x: -1, +1
        { 9, 0}, {10, 0},        // y: -1, +1
        {15, 0}, {16, 0},        // z: -1, +1
        // |Δ|=2  (inner tap = |Δ|=1 on same axis)
        { 2, 3}, { 5, 4},        // x: -2<-(-1), +2<-(+1)
        { 8, 9}, {11, 10},       // y
        {14, 15}, {17, 16},      // z
        // |Δ|=3  (inner tap = |Δ|=2 on same axis)
        { 1, 2}, { 6, 5},        // x: -3<-(-2), +3<-(+2)
        { 7, 8}, {12, 11},       // y
        {13, 14}, {18, 17}       // z
    };
};

// ---------------------------------------------------------------------------
// extrapolate — single-source implementation.
//
// Same body compiles for scalar (W=1) and SIMD (W>1): Simd.h's fixed_size<1>
// path collapses every instruction to a scalar store.  The only non-uniform
// bit is the addr() helper above.
// ---------------------------------------------------------------------------
template<int W>
void WenoStencil<W>::extrapolate(float absBackground)
{
    using FloatV = nanovdb::util::Simd    <float, W>;
    using MaskV  = nanovdb::util::SimdMask<float, W>;

    const FloatV absBg(absBackground);           // broadcast
    const FloatV zero (0.0f);

    for (int p = 0; p < kNumPairs; ++p) {
        const int k      = kPairs[p][0];
        const int kInner = kPairs[p][1];

        const MaskV  active(addr(mIsActive[k]),      nanovdb::util::element_aligned);
        const FloatV val   (addr(mValues  [k]),      nanovdb::util::element_aligned);
        const FloatV inner (addr(mValues  [kInner]), nanovdb::util::element_aligned);

        // copysign(absBg, inner): +absBg if inner >= 0, else -absBg.
        const MaskV  isNegInner = zero > inner;
        const FloatV extrap     = nanovdb::util::where(isNegInner, -absBg, absBg);

        // Active lanes keep `val`; inactive lanes take `extrap`.
        const FloatV result = nanovdb::util::where(active, val, extrap);

        nanovdb::util::store(result, addr(mValues[k]), nanovdb::util::element_aligned);
    }
}

} // namespace nanovdb
