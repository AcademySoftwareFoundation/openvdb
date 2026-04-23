// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file WenoStencil.h

    \brief 19-tap WENO5 stencil value container + out-of-band extrapolation +
           fifth-order upwind Godunov's norm-square gradient.  Templated on
           SIMD lane width W.

    `WenoStencil<W>` holds the per-tap float values and activity flags for a
    single voxel (W=1, scalar / GPU-friendly) or a batch of W voxels (W>1,
    CPU SIMD).  Storage is first-class Simd types directly:

       FloatV values  [19]      ≡  Simd<float, W>     values  [19]
       MaskV  isActive[19]      ≡  SimdMask<float, W> isActive[19]

    At W=1 the Simd types collapse to plain float / bool so scalar CUDA code
    reads as plain scalar arithmetic, and the class is pure-compute — the
    caller owns any fill-side C-array storage it wants to use for scalar
    scatter before an explicit load-into-Simd step.

    Grid-spacing scalars `mDx2` and `mInvDx2` stay scalar `float` at every
    W and are broadcast to FloatV at the point of use.

    Operations provided:
      - extrapolate(absBackground)
            Repair out-of-band lanes (isActive[k] == false) via
            copysign(absBackground, values[innerTap]), processed in
            ascending-|Δ| order so the inner tap is already resolved.
      - normSqGrad(isoValue = 0)
            Godunov's norm-square of the fifth-order WENO upwind gradient.
            Matches the semantics of WenoStencil::normSqGrad(isoValue) in
            nanovdb/math/Stencils.h (the ground-truth scalar reference).

    See BatchAccessor.md §11 for the full Phase-2 sidecar-WENO pipeline design
    and §11.2 for the extrapolation semantics.
*/

#pragma once

#include <nanovdb/util/Simd.h>
#include <nanovdb/math/Math.h>             // Pow2

#include <cstdint>
#include <tuple>
#include <utility>

namespace nanovdb {

namespace detail {

// ---------------------------------------------------------------------------
// Generic-T WENO5 reconstruction — templated on T ∈ {float, Simd<float, W>}.
//
// Structurally identical to nanovdb::math::WENO5 (ground-truth scalar WENO5
// in nanovdb/math/Stencils.h), transliterated to use only primitives that
// exist for both scalar T=float and Simd<T,W>: operator+/-/*, math::Pow2.
// No ternaries, no if/else — same source compiles to scalar or SIMD code
// via the Simd backend in Simd.h.
//
// scale2 is the optional reference magnitude (squared) used to scale the
// numerical epsilon; kept as a plain float for broadcast-on-demand.
// ---------------------------------------------------------------------------
template<typename T, typename RealT = T>
__hostdev__ NANOVDB_FORCEINLINE T
WENO5(const T& v1, const T& v2, const T& v3,
      const T& v4, const T& v5,
      float scale2 = 1.f)
{
    const RealT C   = RealT(13.f / 12.f);
    const RealT eps = RealT(1.e-6f * scale2);

    const RealT A1 = RealT(0.1f) / math::Pow2(
            C * math::Pow2(v1 - RealT(2)*v2 + v3)
          + RealT(0.25f) * math::Pow2(v1 - RealT(4)*v2 + RealT(3)*v3) + eps);
    const RealT A2 = RealT(0.6f) / math::Pow2(
            C * math::Pow2(v2 - RealT(2)*v3 + v4)
          + RealT(0.25f) * math::Pow2(v2 - v4) + eps);
    const RealT A3 = RealT(0.3f) / math::Pow2(
            C * math::Pow2(v3 - RealT(2)*v4 + v5)
          + RealT(0.25f) * math::Pow2(RealT(3)*v3 - RealT(4)*v4 + v5) + eps);

    return (A1 * (RealT( 2)*v1 - RealT(7)*v2 + RealT(11)*v3)
          + A2 * (RealT( 5)*v3 -           v2 + RealT( 2)*v4)
          + A3 * (RealT( 2)*v3 + RealT(5)*v4 -             v5))
         / (RealT(6) * (A1 + A2 + A3));
}

// ---------------------------------------------------------------------------
// Generic-T Godunov's norm-square gradient — templated on T (value type) and
// MaskT (mask type that `>` of T produces).  Ground-truth scalar version is
// nanovdb::math::GodunovsNormSqrd in nanovdb/math/Stencils.h, which uses a
// runtime if/else on `isOutside`.  Here we compute both branches uncondition-
// ally and blend via util::where, so the SIMD path has no control-flow
// divergence across lanes.  At T=float the scalar where(bool, T, T) overload
// degenerates this to the same semantics as the if/else.
// ---------------------------------------------------------------------------
template<typename T, typename MaskT>
__hostdev__ NANOVDB_FORCEINLINE T
GodunovsNormSqrd(MaskT isOutside,
                 T dP_xm, T dP_xp,
                 T dP_ym, T dP_yp,
                 T dP_zm, T dP_zp)
{
    using util::min; using util::max; using util::where;
    const T zero(0.f);

    const T outside = max(math::Pow2(max(dP_xm, zero)), math::Pow2(min(dP_xp, zero)))   // (dP/dx)²
                    + max(math::Pow2(max(dP_ym, zero)), math::Pow2(min(dP_yp, zero)))   // (dP/dy)²
                    + max(math::Pow2(max(dP_zm, zero)), math::Pow2(min(dP_zp, zero)));  // (dP/dz)²

    const T inside  = max(math::Pow2(min(dP_xm, zero)), math::Pow2(max(dP_xp, zero)))
                    + max(math::Pow2(min(dP_ym, zero)), math::Pow2(max(dP_yp, zero)))
                    + max(math::Pow2(min(dP_zm, zero)), math::Pow2(max(dP_zp, zero)));

    return where(isOutside, outside, inside);
}

} // namespace detail

// ---------------------------------------------------------------------------
// WenoStencil<W> — pure-compute container for a 19-tap WENO5 stencil state.
//
// The class holds only Simd-typed compute state + scalar grid constants.
// Fill-side responsibility (scalar writes into any raw float/bool buffers,
// followed by a SIMD load-per-tap into this stencil's values[] / isActive[])
// lives in the caller.  See WenoStencil.md §6 for usage patterns.
// ---------------------------------------------------------------------------
template<int W = 1>
class WenoStencil
{
public:
    using FloatV = util::Simd    <float, W>;
    using MaskV  = util::SimdMask<float, W>;

    // --- Tap-offset types (compile-time only) -----------------------------
    // TapPoint<DI,DJ,DK> carries the tap offset as a type.  Taps is the
    // 19-tap tuple in the canonical WenoPt<i,j,k>::idx ordering from
    // nanovdb/math/Stencils.h:
    //    idx  0     : center  < 0, 0, 0>
    //    idx  1.. 6 : x-axis  <-3,0,0> <-2,0,0> <-1,0,0> <+1,0,0> <+2,0,0> <+3,0,0>
    //    idx  7..12 : y-axis  <0,-3,0> <0,-2,0> <0,-1,0> <0,+1,0> <0,+2,0> <0,+3,0>
    //    idx 13..18 : z-axis  <0,0,-3> <0,0,-2> <0,0,-1> <0,0,+1> <0,0,+2> <0,0,+3>
    template<int DI, int DJ, int DK>
    struct TapPoint {
        static constexpr int di = DI, dj = DJ, dk = DK;
    };

    using Taps = std::tuple<
        TapPoint< 0, 0, 0>,
        TapPoint<-3, 0, 0>, TapPoint<-2, 0, 0>, TapPoint<-1, 0, 0>,
        TapPoint<+1, 0, 0>, TapPoint<+2, 0, 0>, TapPoint<+3, 0, 0>,
        TapPoint< 0,-3, 0>, TapPoint< 0,-2, 0>, TapPoint< 0,-1, 0>,
        TapPoint< 0,+1, 0>, TapPoint< 0,+2, 0>, TapPoint< 0,+3, 0>,
        TapPoint< 0, 0,-3>, TapPoint< 0, 0,-2>, TapPoint< 0, 0,-1>,
        TapPoint< 0, 0,+1>, TapPoint< 0, 0,+2>, TapPoint< 0, 0,+3>
    >;

    static constexpr int SIZE = int(std::tuple_size_v<Taps>);
    static constexpr int size() { return SIZE; }

    // Compute-side storage — first-class Simd values.  At W=1 these collapse
    // to plain scalar float / bool under the array backend.
    FloatV values  [SIZE];
    MaskV  isActive[SIZE];

    // Runtime grid-spacing constants — plain scalars at every W, broadcast
    // to FloatV at the use sites inside normSqGrad().  Storing them as
    // scalars saves YMM-register pressure (vbroadcastss folds into the FMA
    // consumer on x86) and keeps the W=1 code path free of any Simd wrapper.
    float mDx2{1.f};       // dx²      — fed to WENO5's epsilon via scale2
    float mInvDx2{1.f};    // 1 / dx²  — final normalisation in normSqGrad

    __hostdev__ WenoStencil() = default;
    __hostdev__ explicit WenoStencil(float dx)
        : mDx2(dx * dx), mInvDx2(1.f / (dx * dx)) {}

    // Compile-time named-tap access: returns the index of tap (DI,DJ,DK) in
    // the Taps tuple.  Ordering matches WenoPt<i,j,k>::idx in
    // nanovdb/math/Stencils.h.
    template<int DI, int DJ, int DK>
    static constexpr int tapIndex()
    {
        constexpr int I = findTap<DI, DJ, DK>(std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "WenoStencil::tapIndex: tap not in stencil");
        return I;
    }

    // ------------------------------------------------------------------
    // extrapolate — repair out-of-band lanes (isActive[k][i] == false) of
    // values[k] with copysign(absBackground, values[innerTap][i]).  Active
    // lanes are preserved.  Center tap (idx 0) is assumed always in-band
    // and is not processed.
    //
    // Processes 18 (tap, innerTap) pairs in ascending-|Δ| order so the
    // inner tap is already resolved when the outer tap is touched;
    // sign-inheritance through |Δ|=1 → |Δ|=2 → |Δ|=3 is automatic.
    //
    // Requires absBackground ≥ 0.
    // ------------------------------------------------------------------
    __hostdev__ NANOVDB_FORCEINLINE void extrapolate(float absBackground);

    // ------------------------------------------------------------------
    // normSqGrad — Godunov's norm-square of the fifth-order WENO upwind
    // gradient at the stencil center.  Returns |∇φ|².
    //
    // Semantics match WenoStencil::normSqGrad(isoValue) in
    // nanovdb/math/Stencils.h line-for-line: six axial WENO5 reconstructions
    // (one pair ±x, ±y, ±z), then Godunov's upwind combinator driven by the
    // sign of (center − iso).
    //
    // Call only after the stencil has been populated (see usage pattern in
    // WenoStencil.md §6).  extrapolate() is idempotent w.r.t. this — calling
    // normSqGrad after extrapolate is the typical pipeline shape, but the
    // method itself does not require extrapolate to have been called.
    // ------------------------------------------------------------------
    __hostdev__ NANOVDB_FORCEINLINE FloatV normSqGrad(float iso = 0.f) const;

private:
    // Compile-time inverse map: (DI,DJ,DK) → slot index in Taps.  Returns -1
    // if no matching tap exists; tapIndex() turns that into a static_assert.
    template<int DI, int DJ, int DK, size_t... Is>
    static constexpr int findTap(std::index_sequence<Is...>)
    {
        int result = -1;
        ((std::tuple_element_t<Is, Taps>::di == DI &&
          std::tuple_element_t<Is, Taps>::dj == DJ &&
          std::tuple_element_t<Is, Taps>::dk == DK &&
          result < 0 ? (result = int(Is)) : 0), ...);
        return result;
    }

    // Hardcoded (tap, innerTap) pairs for the 19-tap Taps tuple, ordered by
    // ascending |Δ| so the inner tap is always already resolved when the
    // outer tap is processed.  Indices match the Taps tuple above.
    //
    //   idx  0     :  center     ( 0, 0, 0)
    //   idx  1.. 6 :  x-axis     (-3..+3 in the order -3,-2,-1,+1,+2,+3)
    //   idx  7..12 :  y-axis     (-3..+3)
    //   idx 13..18 :  z-axis     (-3..+3)
    static constexpr int kNumPairs = 18;
    static constexpr int kPairs[kNumPairs][2] = {
        // |Δ|=1  (inner tap = center, idx 0)
        { 3, 0}, { 4, 0},        // x: -1, +1
        { 9, 0}, {10, 0},        // y: -1, +1
        {15, 0}, {16, 0},        // z: -1, +1
        // |Δ|=2  (inner tap = |Δ|=1 on same axis)
        { 2, 3}, { 5, 4},        // x: -2 ← (-1),  +2 ← (+1)
        { 8, 9}, {11, 10},       // y
        {14, 15}, {17, 16},      // z
        // |Δ|=3  (inner tap = |Δ|=2 on same axis)
        { 1, 2}, { 6, 5},        // x: -3 ← (-2),  +3 ← (+2)
        { 7, 8}, {12, 11},       // y
        {13, 14}, {18, 17}       // z
    };
};

// ---------------------------------------------------------------------------
// extrapolate — single-source implementation.
//
// values[] and isActive[] are already Simd-typed; the algorithm is a
// sequence of whole-SIMD blends (plus a broadcast of absBg) per pair.
// Same source body compiles at W=1 (Simd<float,1> collapses to scalar)
// and W>1 (native SIMD width).
// ---------------------------------------------------------------------------
template<int W>
__hostdev__ NANOVDB_FORCEINLINE void
WenoStencil<W>::extrapolate(float absBackground)
{
    const FloatV absBg(absBackground);
    const FloatV zero (0.f);

    for (int p = 0; p < kNumPairs; ++p) {
        const int k      = kPairs[p][0];
        const int kInner = kPairs[p][1];

        // copysign(absBg, inner): +absBg if inner >= 0, else -absBg.
        const MaskV  isNegInner = zero > values[kInner];
        const FloatV extrap     = util::where(isNegInner, -absBg, absBg);

        // Active lanes keep their own value; inactive lanes take the extrapolated sign-corrected background.
        values[k] = util::where(isActive[k], values[k], extrap);
    }
}

// ---------------------------------------------------------------------------
// normSqGrad — Godunov's upwind WENO norm-square gradient.
//
// Structurally mirrors WenoStencil::normSqGrad(isoValue) in
// nanovdb/math/Stencils.h: six axial WENO5 reconstructions driving
// GodunovsNormSqrd.  Tap indices 0..18 match WenoPt<i,j,k>::idx in that
// file.  mInvDx2 and iso are broadcast to FloatV at the final
// combinator only (free on x86; identity at W=1).
// ---------------------------------------------------------------------------
template<int W>
__hostdev__ NANOVDB_FORCEINLINE typename WenoStencil<W>::FloatV
WenoStencil<W>::normSqGrad(float iso) const
{
    const FloatV* v = values;

    const FloatV dP_xm = detail::WENO5<FloatV>(v[ 2]-v[ 1], v[ 3]-v[ 2], v[ 0]-v[ 3], v[ 4]-v[ 0], v[ 5]-v[ 4], mDx2);
    const FloatV dP_xp = detail::WENO5<FloatV>(v[ 6]-v[ 5], v[ 5]-v[ 4], v[ 4]-v[ 0], v[ 0]-v[ 3], v[ 3]-v[ 2], mDx2);
    const FloatV dP_ym = detail::WENO5<FloatV>(v[ 8]-v[ 7], v[ 9]-v[ 8], v[ 0]-v[ 9], v[10]-v[ 0], v[11]-v[10], mDx2);
    const FloatV dP_yp = detail::WENO5<FloatV>(v[12]-v[11], v[11]-v[10], v[10]-v[ 0], v[ 0]-v[ 9], v[ 9]-v[ 8], mDx2);
    const FloatV dP_zm = detail::WENO5<FloatV>(v[14]-v[13], v[15]-v[14], v[ 0]-v[15], v[16]-v[ 0], v[17]-v[16], mDx2);
    const FloatV dP_zp = detail::WENO5<FloatV>(v[18]-v[17], v[17]-v[16], v[16]-v[ 0], v[ 0]-v[15], v[15]-v[14], mDx2);

    return FloatV(mInvDx2) *
           detail::GodunovsNormSqrd<FloatV, MaskV>(v[0] > FloatV(iso),
                                                    dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp);
}

} // namespace nanovdb
