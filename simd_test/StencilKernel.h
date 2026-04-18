#pragma once
#include <nanovdb/util/Simd.h>

// Portable __hostdev__ annotation — no-op outside CUDA, matching NanoVDB convention.
#ifndef __CUDACC__
#  ifndef __hostdev__
#    define __hostdev__
#  endif
#endif

// ---------------------------------------------------------------------------
// Prototype of the kernel-only stencil hierarchy for NanoVDB.
//
// Defines BaseStencilKernel<T, SIZE> and WenoStencilKernel<T>, where T is:
//   float             — scalar, __hostdev__-compatible, GPU per-thread path
//   Simd<float, W>    — W-wide SIMD, CPU per-batch path
//
// These are pure data + compute classes with no grid coupling.  They are
// intended to replace the compute portions of BaseStencil / WenoStencil in
// nanovdb/math/Stencils.h, with the legacy accessor-based classes deriving
// from these to retain backward compatibility during transition.
//
// Free functions WENO5 and GodunovsNormSqrd mirror their counterparts in
// Stencils.h, templatized on T so they work for both scalar and SIMD.
// ---------------------------------------------------------------------------

namespace nanovdb {
namespace math {

using namespace nanovdb::util; // min, max, where, Simd, SimdMask

// ---------------------------------------------------------------------------
// WENO5 — fifth-order upwind interpolation, templated on T.
// Mirrors WENO5<ValueType, RealT> in Stencils.h; here RealT == T throughout.
// ---------------------------------------------------------------------------
template<typename T>
__hostdev__ inline T WENO5(T v1, T v2, T v3, T v4, T v5, float scale2 = 1.f)
{
    const float C   = 13.f / 12.f;
    const T     eps = T(1.0e-6f * scale2);

    const T d12 = v1 - 2.f*v2 + v3;
    const T d13 = v1 - 4.f*v2 + 3.f*v3;
    const T d23 = v2 - 2.f*v3 + v4;
    const T d24 = v2 - v4;
    const T d34 = v3 - 2.f*v4 + v5;
    const T d35 = 3.f*v3 - 4.f*v4 + v5;

    const T w1 = C*d12*d12 + 0.25f*d13*d13 + eps;
    const T w2 = C*d23*d23 + 0.25f*d24*d24 + eps;
    const T w3 = C*d34*d34 + 0.25f*d35*d35 + eps;

    const T A1 = 0.1f / (w1*w1);
    const T A2 = 0.6f / (w2*w2);
    const T A3 = 0.3f / (w3*w3);

    return (A1*(2.f*v1 - 7.f*v2 + 11.f*v3) +
            A2*(5.f*v3 -     v2 +  2.f*v4) +
            A3*(2.f*v3 + 5.f*v4 -      v5)) / (6.f*(A1+A2+A3));
}

// ---------------------------------------------------------------------------
// GodunovsNormSqrd — templated on T (value type) and MaskT (mask type).
// Mirrors GodunovsNormSqrd<RealT> in Stencils.h.
// The if/else branch in the original is replaced by unconditionally computing
// both the outside and inside terms and blending via where(), so the SIMD
// path produces a lane-wise select with no control flow divergence.
// ---------------------------------------------------------------------------
template<typename T, typename MaskT>
__hostdev__ inline T GodunovsNormSqrd(MaskT isOutside,
                                      T dP_xm, T dP_xp,
                                      T dP_ym, T dP_yp,
                                      T dP_zm, T dP_zp)
{
    const T zero(0.f);
    T outside = max(max(dP_xm, zero) * max(dP_xm, zero),
                    min(dP_xp, zero) * min(dP_xp, zero))  // (dP/dx)^2
              + max(max(dP_ym, zero) * max(dP_ym, zero),
                    min(dP_yp, zero) * min(dP_yp, zero))  // (dP/dy)^2
              + max(max(dP_zm, zero) * max(dP_zm, zero),
                    min(dP_zp, zero) * min(dP_zp, zero)); // (dP/dz)^2

    T inside  = max(min(dP_xm, zero) * min(dP_xm, zero),
                    max(dP_xp, zero) * max(dP_xp, zero))  // (dP/dx)^2
              + max(min(dP_ym, zero) * min(dP_ym, zero),
                    max(dP_yp, zero) * max(dP_yp, zero))  // (dP/dy)^2
              + max(min(dP_zm, zero) * min(dP_zm, zero),
                    max(dP_zp, zero) * max(dP_zp, zero)); // (dP/dz)^2

    return where(isOutside, outside, inside);
}

// ---------------------------------------------------------------------------
// BaseStencilKernel<T, SIZE>
//
// Owns mValues[SIZE] and the grid spacing parameters mDx2 / mInvDx2.
// No grid accessor, no moveTo — pure data container for stencil compute.
// ---------------------------------------------------------------------------
template<typename T, int SIZE>
class BaseStencilKernel
{
protected:
    T     mValues[SIZE]{};
    float mDx2{1.f}, mInvDx2{1.f};

public:
    __hostdev__ BaseStencilKernel() = default;
    __hostdev__ explicit BaseStencilKernel(float dx)
        : mDx2(dx * dx), mInvDx2(1.f / (dx * dx)) {}

    __hostdev__ T&       operator[](int n)       { return mValues[n]; }
    __hostdev__ const T& operator[](int n) const { return mValues[n]; }

    __hostdev__ static constexpr int size() { return SIZE; }
};

// ---------------------------------------------------------------------------
// WenoStencilKernel<T>
//
// Derives from BaseStencilKernel<T, 19> and provides normSqGrad() and
// related compute methods.  Mirrors the compute interface of WenoStencil in
// nanovdb/math/Stencils.h.
//
// mValues layout (matching WenoPt<i,j,k>::idx):
//   [0]        = center  ( 0, 0, 0)
//   [1]..[6]   = x-axis  (-3,-2,-1, +1,+2,+3)
//   [7]..[12]  = y-axis  (-3,-2,-1, +1,+2,+3)
//   [13]..[18] = z-axis  (-3,-2,-1, +1,+2,+3)
// ---------------------------------------------------------------------------
template<typename T>
class WenoStencilKernel : public BaseStencilKernel<T, 19>
{
    using Base = BaseStencilKernel<T, 19>;

protected:
    using Base::mValues;
    using Base::mDx2;
    using Base::mInvDx2;

public:
    using Base::Base;

    /// @brief Return the norm-squared of the WENO upwind gradient at the
    /// buffered stencil location, using Godunov's scheme.
    /// Matches WenoStencil::normSqGrad() in Stencils.h.
    __hostdev__ inline T normSqGrad(float isoValue = 0.f) const
    {
        const T* v = mValues;
        const T
            dP_xm = WENO5<T>(v[ 2]-v[ 1], v[ 3]-v[ 2], v[ 0]-v[ 3], v[ 4]-v[ 0], v[ 5]-v[ 4], mDx2),
            dP_xp = WENO5<T>(v[ 6]-v[ 5], v[ 5]-v[ 4], v[ 4]-v[ 0], v[ 0]-v[ 3], v[ 3]-v[ 2], mDx2),
            dP_ym = WENO5<T>(v[ 8]-v[ 7], v[ 9]-v[ 8], v[ 0]-v[ 9], v[10]-v[ 0], v[11]-v[10], mDx2),
            dP_yp = WENO5<T>(v[12]-v[11], v[11]-v[10], v[10]-v[ 0], v[ 0]-v[ 9], v[ 9]-v[ 8], mDx2),
            dP_zm = WENO5<T>(v[14]-v[13], v[15]-v[14], v[ 0]-v[15], v[16]-v[ 0], v[17]-v[16], mDx2),
            dP_zp = WENO5<T>(v[18]-v[17], v[17]-v[16], v[16]-v[ 0], v[ 0]-v[15], v[15]-v[14], mDx2);

        return T(mInvDx2) * GodunovsNormSqrd(v[0] > T(isoValue),
                                             dP_xm, dP_xp,
                                             dP_ym, dP_yp,
                                             dP_zm, dP_zp);
    }
};

} // namespace math
} // namespace nanovdb
