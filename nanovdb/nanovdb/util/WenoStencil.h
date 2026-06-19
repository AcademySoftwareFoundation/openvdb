// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file WenoStencil.h

    \brief 19-point WENO5 stencil value container + out-of-band extrapolation +
           fifth-order upwind Godunov's norm-square gradient.  Templated on
           a ValueType.

    `WenoStencil<ValueType>` holds the per-point values and activity flags.
    ValueType is typically a raw scalar `float` (e.g. for scalar / CUDA code)
    or a SIMD vector `Simd<float, W>` (CPU batch path).  The companion mask
    type is auto-deduced (bool for raw scalars, ValueType::mask_type
    otherwise):

       ValueType values  [19]
       MaskType  isActive[19]

    For raw scalars the class reads as plain scalar arithmetic; for SIMD
    vectors the same source compiles to whole-vector ops via Simd.h.  The
    class is pure-compute -- the caller owns any fill-side C-array storage
    and the per-point load step.

    Grid-spacing scalars `mDx2` and `mInvDx2` stay scalar `float` and are
    broadcast to ValueType at the point of use.

    Operations provided:
      - extrapolate(absBackground)
            Repair out-of-band lanes (isActive[k] == false) via
            copysign(absBackground, values[innerPoint]), processed in
            ascending-|d| order so the inner point is already resolved.
      - normSqGrad(isoValue = 0)
            Godunov's norm-square of the fifth-order WENO upwind gradient.
*/

#pragma once

#include <nanovdb/util/Simd.h>
#include <nanovdb/math/Math.h>             // Pow2

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace nanovdb {

namespace detail {

// ---------------------------------------------------------------------------
// mask_of<T>::type -- auto-deduced predicate type for ValueType T:
//   - T::mask_type if T has a nested mask_type (e.g. Simd<float,W>);
//   - bool otherwise (e.g. raw scalar T=float).
// ---------------------------------------------------------------------------
template<typename T, typename = void>
struct mask_of { using type = bool; };

template<typename T>
struct mask_of<T, std::void_t<typename T::mask_type>> {
    using type = typename T::mask_type;
};

// ---------------------------------------------------------------------------
// Generic WENO5 reconstruction -- templated on ValueType in {float,
// Simd<float,W>}.  Nominally fifth-order finite-difference WENO (Chi-Wang
// Shu, "High Order Finite Difference and Finite Volume WENO Schemes and
// Discontinuous Galerkin Methods for CFD", ICASE Report No 2001-11 page 6;
// see also ICASE 97-65 for a more complete reference, Shu 1997).
//
// Given v1=f(x-2dx), v2=f(x-dx), v3=f(x), v4=f(x+dx), v5=f(x+2dx), returns
// an interpolated f(x+dx/2) with the property that
// (f(x+dx/2)-f(x-dx/2))/dx = df/dx(x) + error, with error fifth-order in
// smooth regions: O(dx) <= error <= O(dx^5).
//
// Body uses only primitives common to scalar and Simd ValueType
// (operator+/-/*, math::Pow2), so the same source compiles in both modes
// via the Simd backend in Simd.h.  Integer coefficients carry explicit
// ValueType(...) casts for SIMD typed-operator dispatch, and float
// literals carry .f suffix because stdx::simd's broadcast ctor rejects
// double->float narrowing.
//
// ScalarType (defaults to float, deduced from scale2 if specified) is the
// arithmetic precision of the reference-magnitude epsilon scaling.  scale2
// stays scalar so callers can pass plain float/double grid constants
// without broadcasting.
// ---------------------------------------------------------------------------
template<typename ValueType, typename ScalarType = float>
__hostdev__ NANOVDB_FORCEINLINE ValueType
WENO5(const ValueType& v1, const ValueType& v2, const ValueType& v3,
      const ValueType& v4, const ValueType& v5,
      ScalarType scale2 = ScalarType(1)) // openvdb uses scale2 = 0.01
{
    using math::Pow2;
    const ValueType C = ValueType(13.f / 12.f);
    // WENO is formulated for non-dimensional equations, here the optional scale2
    // is a reference value (squared) for the function being interpolated.  For
    // example if 'v' is of order 1000, then scale2 = 10^6 is ok.  But in practice
    // leave scale2 = 1.
    const ValueType eps = ValueType(ScalarType(1.e-6) * scale2);
    // {\tilde \omega_k} = \gamma_k / (\beta_k + \epsilon)^2 in Shu's ICASE report.
    const ValueType A1 = ValueType(0.1f) / Pow2(
            C * Pow2(v1 - ValueType(2)*v2 + v3) +
            ValueType(0.25f) * Pow2(v1 - ValueType(4)*v2 + ValueType(3)*v3) + eps);
    const ValueType A2 = ValueType(0.6f) / Pow2(
            C * Pow2(v2 - ValueType(2)*v3 + v4) +
            ValueType(0.25f) * Pow2(v2 - v4) + eps);
    const ValueType A3 = ValueType(0.3f) / Pow2(
            C * Pow2(v3 - ValueType(2)*v4 + v5) +
            ValueType(0.25f) * Pow2(ValueType(3)*v3 - ValueType(4)*v4 + v5) + eps);

    return (A1 * (ValueType(2)*v1 - ValueType(7)*v2 + ValueType(11)*v3) +
            A2 * (ValueType(5)*v3 -              v2 + ValueType( 2)*v4) +
            A3 * (ValueType(2)*v3 + ValueType(5)*v4 -               v5))
         / (ValueType(6) * (A1 + A2 + A3));
}

// ---------------------------------------------------------------------------
// Generic Godunov's norm-square gradient -- templated on ValueType
// in {float, Simd<float,W>} and a companion MaskType (bool for raw scalar,
// ValueType::mask_type for Simd).  Differs from a textbook scalar form in
// shape: instead of an if/else on isOutside we compute both branches
// unconditionally and blend via math::Select, so the SIMD path has no
// control-flow divergence across lanes.  At ValueType=float the scalar
// math::Select(bool, ValueType, ValueType) overload reduces this to the
// equivalent if/else semantics.
// ---------------------------------------------------------------------------
template<typename ValueType, typename MaskType>
__hostdev__ NANOVDB_FORCEINLINE ValueType
GodunovsNormSqrd(MaskType isOutside,
                 ValueType dP_xm, ValueType dP_xp,
                 ValueType dP_ym, ValueType dP_yp,
                 ValueType dP_zm, ValueType dP_zp)
{
    using math::Min; using math::Max; using math::Pow2; using math::Select;
    const ValueType zero(0.f);

    const ValueType outside = Max(Pow2(Max(dP_xm, zero)), Pow2(Min(dP_xp, zero)))  // (dP/dx)^2
                            + Max(Pow2(Max(dP_ym, zero)), Pow2(Min(dP_yp, zero)))  // (dP/dy)^2
                            + Max(Pow2(Max(dP_zm, zero)), Pow2(Min(dP_zp, zero))); // (dP/dz)^2

    const ValueType inside  = Max(Pow2(Min(dP_xm, zero)), Pow2(Max(dP_xp, zero)))
                            + Max(Pow2(Min(dP_ym, zero)), Pow2(Max(dP_yp, zero)))
                            + Max(Pow2(Min(dP_zm, zero)), Pow2(Max(dP_zp, zero)));

    return Select(isOutside, outside, inside); // |\nabla\phi|^2
}

} // namespace detail

// ---------------------------------------------------------------------------
// WenoStencil<ValueType> -- pure-compute container for a 19-point WENO5
// stencil state.  Holds ValueType-typed values + MaskType-typed activity
// flags + scalar grid constants.  Fill-side responsibility (scalar writes
// into any raw buffers, followed by a per-point load into this stencil's
// values[] / isActive[]) lives in the caller.  See WenoStencil.md
// for usage patterns.
// ---------------------------------------------------------------------------
template<typename ValueType>
class WenoStencil
{
public:
    using MaskType = typename detail::mask_of<ValueType>::type;

    // --- Stencil-point offset types (compile-time only) -------------------
    // StencilPoint<i,j,k> carries the offset as a type.  StencilPoints is
    // the 19-point tuple in the canonical idx ordering:
    //    idx  0     : center  < 0, 0, 0>
    //    idx  1.. 6 : x-axis  <-3,0,0> <-2,0,0> <-1,0,0> <+1,0,0> <+2,0,0> <+3,0,0>
    //    idx  7..12 : y-axis  <0,-3,0> <0,-2,0> <0,-1,0> <0,+1,0> <0,+2,0> <0,+3,0>
    //    idx 13..18 : z-axis  <0,0,-3> <0,0,-2> <0,0,-1> <0,0,+1> <0,0,+2> <0,0,+3>
    template<int i, int j, int k>
    struct StencilPoint {
        static constexpr int di = i, dj = j, dk = k;
    };

    using StencilPoints = std::tuple<
        StencilPoint< 0, 0, 0>,
        StencilPoint<-3, 0, 0>, StencilPoint<-2, 0, 0>, StencilPoint<-1, 0, 0>,
        StencilPoint<+1, 0, 0>, StencilPoint<+2, 0, 0>, StencilPoint<+3, 0, 0>,
        StencilPoint< 0,-3, 0>, StencilPoint< 0,-2, 0>, StencilPoint< 0,-1, 0>,
        StencilPoint< 0,+1, 0>, StencilPoint< 0,+2, 0>, StencilPoint< 0,+3, 0>,
        StencilPoint< 0, 0,-3>, StencilPoint< 0, 0,-2>, StencilPoint< 0, 0,-1>,
        StencilPoint< 0, 0,+1>, StencilPoint< 0, 0,+2>, StencilPoint< 0, 0,+3>
    >;

    static constexpr int SIZE = int(std::tuple_size_v<StencilPoints>);
    static constexpr int size() { return SIZE; }

    // Compute-side storage.  At ValueType=float these are plain float / bool
    // arrays; at ValueType=Simd<float,W> they are whole-vector arrays.
    ValueType values  [SIZE];
    MaskType  isActive[SIZE];

    // Runtime grid-spacing constants -- plain scalars regardless of ValueType,
    // broadcast inside normSqGrad().
    float mDx2{1.f};       // dx^2      -- fed to WENO5's epsilon via scale2
    float mInvDx2{1.f};    // 1 / dx^2  -- final normalisation in normSqGrad

    __hostdev__ WenoStencil() = default;
    __hostdev__ explicit WenoStencil(float dx)
        : mDx2(dx * dx), mInvDx2(1.f / (dx * dx)) {}

    // Compile-time named-point access: returns the index of point (i,j,k)
    // in the StencilPoints tuple.
    template<int i, int j, int k>
    static constexpr int pointIndex()
    {
        constexpr int I = findPoint<i, j, k>(std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "WenoStencil::pointIndex: point not in stencil");
        return I;
    }

    // Resolve all SIZE stencil-point indices for the voxel at @a center via
    // Acc::getValue(center + offset).  Indices land in out[0..SIZE-1] in the
    // StencilPoints tuple ordering.  Acc is any NanoVDB accessor whose
    // getValue() returns a value convertible to uint64_t (e.g. ValueOnIndex's
    // sequential active-voxel indices).  The accessor's path cache is reused
    // across the SIZE getValue calls.
    template<typename Acc>
    static void gatherIndices(Acc& acc, const Coord& center, uint64_t* out)
    {
        gatherIndicesImpl(acc, center, out, std::make_index_sequence<SIZE>{});
    }

    // ------------------------------------------------------------------
    // extrapolate -- sign-correct out-of-band lanes (isActive[k][i] == false)
    // of values[k] by multiplying with Sign(values[innerPoint][i]).  Active
    // lanes are preserved.  Center point (idx 0) is assumed always in-band
    // and is not processed.
    //
    // Convention: the caller must pre-load inactive lanes of values[k] with
    // the sidecar slot-0 background value (which the standard NanoVDB fill
    // pattern produces automatically: a missing tap resolves to index 0,
    // and sidecar[0] is the background).  This routine then flips the sign
    // when the parent (innerPoint) is negative, leaves it alone when the
    // parent is positive, and zeros the lane when the parent is exactly 0.
    //
    // Processes 18 (point, innerPoint) pairs in ascending-|d| order so the
    // inner point is already resolved when the outer point is touched;
    // sign-inheritance through |d|=1 -> |d|=2 -> |d|=3 is automatic.
    // ------------------------------------------------------------------
    __hostdev__ NANOVDB_FORCEINLINE void extrapolate();

    // ------------------------------------------------------------------
    // normSqGrad -- Godunov's norm-square of the fifth-order WENO upwind
    // gradient at the stencil center.  Returns |\nabla\phi|^2.
    //
    // Six axial WENO5 reconstructions (one pair +/-x, +/-y, +/-z), then
    // Godunov's upwind combinator driven by the sign of (center - iso).
    //
    // Call only after the stencil has been populated (see usage pattern in
    // WenoStencil.md).  extrapolate() before normSqGrad() is the
    // typical pipeline shape but is not required by this method.
    // ------------------------------------------------------------------
    __hostdev__ NANOVDB_FORCEINLINE ValueType normSqGrad(float iso = 0.f) const;

private:
    // Compile-time inverse map: (i,j,k) -> slot index in StencilPoints.
    // Returns -1 if no matching point exists; pointIndex() turns that into
    // a static_assert.
    template<int i, int j, int k, size_t... Is>
    static constexpr int findPoint(std::index_sequence<Is...>)
    {
        int result = -1;
        ((std::tuple_element_t<Is, StencilPoints>::di == i &&
          std::tuple_element_t<Is, StencilPoints>::dj == j &&
          std::tuple_element_t<Is, StencilPoints>::dk == k &&
          result < 0 ? (result = int(Is)) : 0), ...);
        return result;
    }

    // Parameter-pack expansion driving gatherIndices(): unrolls SIZE getValue
    // calls into a single fold expression.
    template<typename Acc, size_t... Is>
    static void gatherIndicesImpl(Acc& acc, const Coord& center, uint64_t* out,
                                   std::index_sequence<Is...>)
    {
        ((out[Is] = static_cast<uint64_t>(acc.getValue(center + Coord(
            std::tuple_element_t<Is, StencilPoints>::di,
            std::tuple_element_t<Is, StencilPoints>::dj,
            std::tuple_element_t<Is, StencilPoints>::dk)))), ...);
    }

    // Hardcoded (point, innerPoint) pairs for the 19-point StencilPoints
    // tuple, ordered by ascending |d| so the inner point is always already
    // resolved when the outer point is processed.  Indices match the
    // StencilPoints tuple above.
    //
    //   idx  0     :  center     ( 0, 0, 0)
    //   idx  1.. 6 :  x-axis     (-3..+3 in the order -3,-2,-1,+1,+2,+3)
    //   idx  7..12 :  y-axis     (-3..+3)
    //   idx 13..18 :  z-axis     (-3..+3)
    static constexpr int kNumPairs = 18;
    static constexpr int kPairs[kNumPairs][2] = {
        // |d|=1  (inner point = center, idx 0)
        { 3, 0}, { 4, 0},        // x: -1, +1
        { 9, 0}, {10, 0},        // y: -1, +1
        {15, 0}, {16, 0},        // z: -1, +1
        // |d|=2  (inner point = |d|=1 on same axis)
        { 2, 3}, { 5, 4},        // x: -2 <- (-1),  +2 <- (+1)
        { 8, 9}, {11, 10},       // y
        {14, 15}, {17, 16},      // z
        // |d|=3  (inner point = |d|=2 on same axis)
        { 1, 2}, { 6, 5},        // x: -3 <- (-2),  +3 <- (+2)
        { 7, 8}, {12, 11},       // y
        {13, 14}, {18, 17}       // z
    };
};

// ---------------------------------------------------------------------------
// extrapolate -- single-source implementation.  At ValueType=float this is
// scalar code; at ValueType=Simd<float,W> the same source compiles to
// whole-SIMD blends via the math::Select dispatch.
// ---------------------------------------------------------------------------
template<typename ValueType>
__hostdev__ NANOVDB_FORCEINLINE void
WenoStencil<ValueType>::extrapolate()
{
    const ValueType zero(0.f);

    for (int p = 0; p < kNumPairs; ++p) {
        const int k      = kPairs[p][0];
        const int kInner = kPairs[p][1];

        // values[k] *= Sign(values[kInner]):
        //   parent > 0  -> values[k] (already pre-loaded with +background);
        //   parent < 0  -> -values[k];
        //   parent == 0 -> 0.
        const MaskType  isPosParent = values[kInner] > zero;
        const MaskType  isNegParent = values[kInner] < zero;
        const ValueType signed_k    = math::Select(isPosParent, values[k],
                                      math::Select(isNegParent, -values[k], zero));

        // Active lanes keep their own value; inactive lanes get the sign-corrected background.
        values[k] = math::Select(isActive[k], values[k], signed_k);
    }
}

// ---------------------------------------------------------------------------
// normSqGrad -- Godunov's upwind WENO norm-square gradient.  Six axial
// WENO5 reconstructions drive GodunovsNormSqrd; point indices 0..18 follow
// the StencilPoints tuple ordering above.  mInvDx2 and iso are broadcast to
// ValueType at the final combinator.
// ---------------------------------------------------------------------------
template<typename ValueType>
__hostdev__ NANOVDB_FORCEINLINE ValueType
WenoStencil<ValueType>::normSqGrad(float iso) const
{
    const ValueType* v = values;

    const ValueType dP_xm = detail::WENO5<ValueType>(v[ 2]-v[ 1], v[ 3]-v[ 2], v[ 0]-v[ 3], v[ 4]-v[ 0], v[ 5]-v[ 4], mDx2);
    const ValueType dP_xp = detail::WENO5<ValueType>(v[ 6]-v[ 5], v[ 5]-v[ 4], v[ 4]-v[ 0], v[ 0]-v[ 3], v[ 3]-v[ 2], mDx2);
    const ValueType dP_ym = detail::WENO5<ValueType>(v[ 8]-v[ 7], v[ 9]-v[ 8], v[ 0]-v[ 9], v[10]-v[ 0], v[11]-v[10], mDx2);
    const ValueType dP_yp = detail::WENO5<ValueType>(v[12]-v[11], v[11]-v[10], v[10]-v[ 0], v[ 0]-v[ 9], v[ 9]-v[ 8], mDx2);
    const ValueType dP_zm = detail::WENO5<ValueType>(v[14]-v[13], v[15]-v[14], v[ 0]-v[15], v[16]-v[ 0], v[17]-v[16], mDx2);
    const ValueType dP_zp = detail::WENO5<ValueType>(v[18]-v[17], v[17]-v[16], v[16]-v[ 0], v[ 0]-v[15], v[15]-v[14], mDx2);

    return ValueType(mInvDx2) *
           detail::GodunovsNormSqrd<ValueType, MaskType>(v[0] > ValueType(iso),
                                                          dP_xm, dP_xp, dP_ym, dP_yp, dP_zm, dP_zp);
}

} // namespace nanovdb
