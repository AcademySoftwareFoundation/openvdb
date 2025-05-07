// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file math/FiniteDifference.h

#ifndef OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED
#define OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb/util/Name.h>
#include "Math.h"
#include "Coord.h"
#include "Vec3.h"
#include <string>

#ifdef DWA_OPENVDB
#include <simd/Simd.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace math {


////////////////////////////////////////


/// @brief Different discrete schemes used in the first derivatives.
// Add new items to the *end* of this list, and update NUM_DS_SCHEMES.
enum DScheme {
    UNKNOWN_DS = -1,
    CD_2NDT =  0,   // center difference,    2nd order, but the result must be divided by 2
    CD_2ND,         // center difference,    2nd order
    CD_4TH,         // center difference,    4th order
    CD_6TH,         // center difference,    6th order
    FD_1ST,         // forward difference,   1st order
    FD_2ND,         // forward difference,   2nd order
    FD_3RD,         // forward difference,   3rd order
    BD_1ST,         // backward difference,  1st order
    BD_2ND,         // backward difference,  2nd order
    BD_3RD,         // backward difference,  3rd order
    FD_WENO5,       // forward difference,   weno5
    BD_WENO5,       // backward difference,  weno5
    FD_HJWENO5,     // forward differene,   HJ-weno5
    BD_HJWENO5      // backward difference, HJ-weno5
};

enum { NUM_DS_SCHEMES = BD_HJWENO5 + 1 };


inline std::string
dsSchemeToString(DScheme dss)
{
    std::string ret;
    switch (dss) {
        case UNKNOWN_DS:    ret = "unknown_ds"; break;
        case CD_2NDT:       ret = "cd_2ndt";    break;
        case CD_2ND:        ret = "cd_2nd";     break;
        case CD_4TH:        ret = "cd_4th";     break;
        case CD_6TH:        ret = "cd_6th";     break;
        case FD_1ST:        ret = "fd_1st";     break;
        case FD_2ND:        ret = "fd_2nd";     break;
        case FD_3RD:        ret = "fd_3rd";     break;
        case BD_1ST:        ret = "bd_1st";     break;
        case BD_2ND:        ret = "bd_2nd";     break;
        case BD_3RD:        ret = "bd_3rd";     break;
        case FD_WENO5:      ret = "fd_weno5";   break;
        case BD_WENO5:      ret = "bd_weno5";   break;
        case FD_HJWENO5:    ret = "fd_hjweno5"; break;
        case BD_HJWENO5:    ret = "bd_hjweno5"; break;
    }
    return ret;
}

inline DScheme
stringToDScheme(const std::string& s)
{
    DScheme ret = UNKNOWN_DS;

    std::string str = s;
    openvdb::string::trim(str);
    openvdb::string::to_lower(str);

    if (str == dsSchemeToString(CD_2NDT)) {
        ret = CD_2NDT;
    } else if (str == dsSchemeToString(CD_2ND)) {
        ret = CD_2ND;
    } else if (str == dsSchemeToString(CD_4TH)) {
        ret = CD_4TH;
    } else if (str == dsSchemeToString(CD_6TH)) {
        ret = CD_6TH;
    } else if (str == dsSchemeToString(FD_1ST)) {
        ret = FD_1ST;
    } else if (str == dsSchemeToString(FD_2ND)) {
        ret = FD_2ND;
    } else if (str == dsSchemeToString(FD_3RD)) {
        ret = FD_3RD;
    } else if (str == dsSchemeToString(BD_1ST)) {
        ret = BD_1ST;
    } else if (str == dsSchemeToString(BD_2ND)) {
        ret = BD_2ND;
    } else if (str == dsSchemeToString(BD_3RD)) {
        ret = BD_3RD;
    } else if (str == dsSchemeToString(FD_WENO5)) {
        ret = FD_WENO5;
    } else if (str == dsSchemeToString(BD_WENO5)) {
        ret = BD_WENO5;
    } else if (str == dsSchemeToString(FD_HJWENO5)) {
        ret = FD_HJWENO5;
    } else if (str == dsSchemeToString(BD_HJWENO5)) {
        ret = BD_HJWENO5;
    }

    return ret;
}

inline std::string
dsSchemeToMenuName(DScheme dss)
{
    std::string ret;
    switch (dss) {
        case UNKNOWN_DS:    ret = "Unknown DS scheme";                      break;
        case CD_2NDT:       ret = "Twice 2nd-order center difference";      break;
        case CD_2ND:        ret = "2nd-order center difference";            break;
        case CD_4TH:        ret = "4th-order center difference";            break;
        case CD_6TH:        ret = "6th-order center difference";            break;
        case FD_1ST:        ret = "1st-order forward difference";           break;
        case FD_2ND:        ret = "2nd-order forward difference";           break;
        case FD_3RD:        ret = "3rd-order forward difference";           break;
        case BD_1ST:        ret = "1st-order backward difference";          break;
        case BD_2ND:        ret = "2nd-order backward difference";          break;
        case BD_3RD:        ret = "3rd-order backward difference";          break;
        case FD_WENO5:      ret = "5th-order WENO forward difference";      break;
        case BD_WENO5:      ret = "5th-order WENO backward difference";     break;
        case FD_HJWENO5:    ret = "5th-order HJ-WENO forward difference";   break;
        case BD_HJWENO5:    ret = "5th-order HJ-WENO backward difference";  break;
    }
    return ret;
}



////////////////////////////////////////


/// @brief Different discrete schemes used in the second derivatives.
// Add new items to the *end* of this list, and update NUM_DD_SCHEMES.
enum DDScheme {
    UNKNOWN_DD  = -1,
    CD_SECOND   =  0,   // center difference, 2nd order
    CD_FOURTH,          // center difference, 4th order
    CD_SIXTH            // center difference, 6th order
};

enum { NUM_DD_SCHEMES = CD_SIXTH + 1 };


////////////////////////////////////////


/// @brief Biased Gradients are limited to non-centered differences
// Add new items to the *end* of this list, and update NUM_BIAS_SCHEMES.
enum BiasedGradientScheme {
    UNKNOWN_BIAS    = -1,
    FIRST_BIAS      = 0,    // uses FD_1ST & BD_1ST
    SECOND_BIAS,            // uses FD_2ND & BD_2ND
    THIRD_BIAS,             // uses FD_3RD & BD_3RD
    WENO5_BIAS,             // uses WENO5
    HJWENO5_BIAS            // uses HJWENO5
};

enum { NUM_BIAS_SCHEMES = HJWENO5_BIAS + 1 };

inline std::string
biasedGradientSchemeToString(BiasedGradientScheme bgs)
{
    std::string ret;
    switch (bgs) {
        case UNKNOWN_BIAS:  ret = "unknown_bias";   break;
        case FIRST_BIAS:    ret = "first_bias";     break;
        case SECOND_BIAS:   ret = "second_bias";    break;
        case THIRD_BIAS:    ret = "third_bias";     break;
        case WENO5_BIAS:    ret = "weno5_bias";     break;
        case HJWENO5_BIAS:  ret = "hjweno5_bias";   break;
    }
    return ret;
}

inline BiasedGradientScheme
stringToBiasedGradientScheme(const std::string& s)
{
    BiasedGradientScheme ret = UNKNOWN_BIAS;

    std::string str = s;
    openvdb::string::trim(str);
    openvdb::string::to_lower(str);

    if (str == biasedGradientSchemeToString(FIRST_BIAS)) {
        ret = FIRST_BIAS;
    } else if (str == biasedGradientSchemeToString(SECOND_BIAS)) {
        ret = SECOND_BIAS;
    } else if (str == biasedGradientSchemeToString(THIRD_BIAS)) {
        ret = THIRD_BIAS;
    } else if (str == biasedGradientSchemeToString(WENO5_BIAS)) {
        ret = WENO5_BIAS;
    } else if (str == biasedGradientSchemeToString(HJWENO5_BIAS)) {
        ret = HJWENO5_BIAS;
    }
    return ret;
}

inline std::string
biasedGradientSchemeToMenuName(BiasedGradientScheme bgs)
{
    std::string ret;
    switch (bgs) {
        case UNKNOWN_BIAS:  ret = "Unknown biased gradient";            break;
        case FIRST_BIAS:    ret = "1st-order biased gradient";          break;
        case SECOND_BIAS:   ret = "2nd-order biased gradient";          break;
        case THIRD_BIAS:    ret = "3rd-order biased gradient";          break;
        case WENO5_BIAS:    ret = "5th-order WENO biased gradient";     break;
        case HJWENO5_BIAS:  ret = "5th-order HJ-WENO biased gradient";  break;
    }
    return ret;
}

////////////////////////////////////////


/// @brief Temporal integration schemes
// Add new items to the *end* of this list, and update NUM_TEMPORAL_SCHEMES.
enum TemporalIntegrationScheme {
    UNKNOWN_TIS = -1,
    TVD_RK1,//same as explicit Euler integration
    TVD_RK2,
    TVD_RK3
};

enum { NUM_TEMPORAL_SCHEMES = TVD_RK3 + 1 };

inline std::string
temporalIntegrationSchemeToString(TemporalIntegrationScheme tis)
{
    std::string ret;
    switch (tis) {
        case UNKNOWN_TIS:   ret = "unknown_tis";    break;
        case TVD_RK1:       ret = "tvd_rk1";        break;
        case TVD_RK2:       ret = "tvd_rk2";        break;
        case TVD_RK3:       ret = "tvd_rk3";        break;
    }
    return ret;
}

inline TemporalIntegrationScheme
stringToTemporalIntegrationScheme(const std::string& s)
{
    TemporalIntegrationScheme ret = UNKNOWN_TIS;

    std::string str = s;
    openvdb::string::trim(str);
    openvdb::string::to_lower(str);

    if (str == temporalIntegrationSchemeToString(TVD_RK1)) {
        ret = TVD_RK1;
    } else if (str == temporalIntegrationSchemeToString(TVD_RK2)) {
        ret = TVD_RK2;
    } else if (str == temporalIntegrationSchemeToString(TVD_RK3)) {
        ret = TVD_RK3;
    }

    return ret;
}

inline std::string
temporalIntegrationSchemeToMenuName(TemporalIntegrationScheme tis)
{
    std::string ret;
    switch (tis) {
        case UNKNOWN_TIS:   ret = "Unknown temporal integration";   break;
        case TVD_RK1:       ret = "Forward Euler";                  break;
        case TVD_RK2:       ret = "2nd-order Runge-Kutta";          break;
        case TVD_RK3:       ret = "3rd-order Runge-Kutta";          break;
    }
    return ret;
}


//@}


/// @brief Implementation of nominally fifth-order finite-difference WENO
/// @details This function returns the numerical flux.  See "High Order Finite Difference and
/// Finite Volume WENO Schemes and Discontinuous Galerkin Methods for CFD" - Chi-Wang Shu
/// ICASE Report No 2001-11 (page 6).  Also see ICASE No 97-65 for a more complete reference
/// (Shu, 1997).
/// Given v1 = f(x-2dx), v2 = f(x-dx), v3 = f(x), v4 = f(x+dx) and v5 = f(x+2dx),
/// return an interpolated value f(x+dx/2) with the special property that
/// ( f(x+dx/2) - f(x-dx/2) ) / dx  = df/dx (x) + error,
/// where the error is fifth-order in smooth regions: O(dx) <= error <=O(dx^5)
template<typename ValueType>
inline typename ComputeTypeFor<ValueType>::type
WENO5(const ValueType& v1, const ValueType& v2, const ValueType& v3,
    const ValueType& v4, const ValueType& v5, float scale2 = 0.01f)
{
    using ComputeType = typename ComputeTypeFor<ValueType>::type;

    const ComputeType f1 = v1, f2 = v2, f3 = v3, f4 = v4, f5 = v5;

    const double C = 13.0 / 12.0;
    // WENO is formulated for non-dimensional equations, here the optional scale2
    // is a reference value (squared) for the function being interpolated.  For
    // example if 'v' is of order 1000, then scale2 = 10^6 is ok.  But in practice
    // leave scale2 = 1.
    const double eps = 1.0e-6 * static_cast<double>(scale2);
    // {\tilde \omega_k} = \gamma_k / ( \beta_k + \epsilon)^2 in Shu's ICASE report)
    const double A1=0.1/math::Pow2(C*math::Pow2(f1-2*f2+f3)+0.25*math::Pow2(f1-4*f2+3.0*f3)+eps),
                 A2=0.6/math::Pow2(C*math::Pow2(f2-2*f3+f4)+0.25*math::Pow2(f2-f4)+eps),
                 A3=0.3/math::Pow2(C*math::Pow2(f3-2*f4+f5)+0.25*math::Pow2(3.0*f3-4*f4+f5)+eps);

    return static_cast<ComputeType>(static_cast<ComputeType>(
        A1*(2.0*f1 - 7.0*f2 + 11.0*f3) +
        A2*(5.0*f3 -     f2 +  2.0*f4) +
        A3*(2.0*f3 + 5.0*f4 -      f5))/(6.0*(A1+A2+A3)));
}


template <typename Real>
inline typename ComputeTypeFor<Real>::type
GodunovsNormSqrd(bool isOutside,
                 Real dP_xm, Real dP_xp,
                 Real dP_ym, Real dP_yp,
                 Real dP_zm, Real dP_zp)
{
    using math::Max;
    using math::Min;
    using math::Pow2;

    using ComputeType = typename ComputeTypeFor<Real>::type;

    const ComputeType dpXm = dP_xm, dpYm = dP_ym, dpZm = dP_zm,
                      dpXp = dP_xp, dpYp = dP_yp, dpZp = dP_zp;

    const ComputeType zero(0);
    ComputeType dPLen2;
    if (isOutside) { // outside
        dPLen2  = Max(Pow2(Max(dpXm, zero)), Pow2(Min(dpXp, zero))); // (dP/dx)2
        dPLen2 += Max(Pow2(Max(dpYm, zero)), Pow2(Min(dpYp, zero))); // (dP/dy)2
        dPLen2 += Max(Pow2(Max(dpZm, zero)), Pow2(Min(dpZp, zero))); // (dP/dz)2
    } else { // inside
        dPLen2  = Max(Pow2(Min(dpXm, zero)), Pow2(Max(dpXp, zero))); // (dP/dx)2
        dPLen2 += Max(Pow2(Min(dpYm, zero)), Pow2(Max(dpYp, zero))); // (dP/dy)2
        dPLen2 += Max(Pow2(Min(dpZm, zero)), Pow2(Max(dpZp, zero))); // (dP/dz)2
    }

    return dPLen2; // |\nabla\phi|^2
}


template<typename Real>
inline typename ComputeTypeFor<Real>::type
GodunovsNormSqrd(bool isOutside, const Vec3<Real>& gradient_m, const Vec3<Real>& gradient_p)
{
    return GodunovsNormSqrd<Real>(isOutside,
                                  gradient_m[0], gradient_p[0],
                                  gradient_m[1], gradient_p[1],
                                  gradient_m[2], gradient_p[2]);
}


#ifdef DWA_OPENVDB
inline simd::Float4 simdMin(const simd::Float4& a, const simd::Float4& b) {
    return simd::Float4(_mm_min_ps(a.base(), b.base()));
}
inline simd::Float4 simdMax(const simd::Float4& a, const simd::Float4& b) {
    return simd::Float4(_mm_max_ps(a.base(), b.base()));
}

inline float simdSum(const simd::Float4& v);

inline simd::Float4 Pow2(const simd::Float4& v) { return v * v; }

template<>
inline simd::Float4
WENO5<simd::Float4>(const simd::Float4& v1, const simd::Float4& v2, const simd::Float4& v3,
                    const simd::Float4& v4, const simd::Float4& v5, float scale2)
{
    using math::Pow2;
    using F4 = simd::Float4;
    const F4
        C(13.f / 12.f),
        eps(1.0e-6f * scale2),
        two(2.0), three(3.0), four(4.0), five(5.0), fourth(0.25),
        A1 = F4(0.1f) / Pow2(C*Pow2(v1-two*v2+v3) + fourth*Pow2(v1-four*v2+three*v3) + eps),
        A2 = F4(0.6f) / Pow2(C*Pow2(v2-two*v3+v4) + fourth*Pow2(v2-v4) + eps),
        A3 = F4(0.3f) / Pow2(C*Pow2(v3-two*v4+v5) + fourth*Pow2(three*v3-four*v4+v5) + eps);
    return (A1 * (two * v1 - F4(7.0) * v2 + F4(11.0) * v3) +
            A2 * (five * v3 - v2 + two * v4) +
            A3 * (two * v3 + five * v4 - v5)) / (F4(6.0) * (A1 + A2 + A3));
}


inline float
simdSum(const simd::Float4& v)
{
    // temp = { v3+v3, v2+v2, v1+v3, v0+v2 }
    __m128 temp = _mm_add_ps(v.base(), _mm_movehl_ps(v.base(), v.base()));
    // temp = { v3+v3, v2+v2, v1+v3, (v0+v2)+(v1+v3) }
    temp = _mm_add_ss(temp, _mm_shuffle_ps(temp, temp, 1));
    return _mm_cvtss_f32(temp);
}

inline float
GodunovsNormSqrd(bool isOutside, const simd::Float4& dP_m, const simd::Float4& dP_p)
{
    const simd::Float4 zero(0.0);
    simd::Float4 v = isOutside
        ? simdMax(math::Pow2(simdMax(dP_m, zero)), math::Pow2(simdMin(dP_p, zero)))
        : simdMax(math::Pow2(simdMin(dP_m, zero)), math::Pow2(simdMax(dP_p, zero)));
    return simdSum(v);//should be v[0]+v[1]+v[2]
}
#endif


template<DScheme DiffScheme>
struct D1
{
    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk);

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S);
};

template<>
struct D1<CD_2NDT>
{
    // the difference operator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp1, const ValueType& xm1)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        return ComputeType(xp1) - ComputeType(xm1);
    }

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(1, 0, 0)),
            grid.getValue(ijk.offsetBy(-1, 0, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0, -1, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0, -1)));
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue< 1, 0, 0>(), S.template getValue<-1, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 1, 0>(), S.template getValue< 0,-1, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0,-1>());
    }
};

template<>
struct D1<CD_2ND>
{

    // the difference operator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp1, const ValueType& xm1)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        return (ComputeType(xp1) - ComputeType(xm1))*ComputeType(0.5);
    }
    static bool difference(const bool& xp1, const bool& /*xm1*/) {
        return xp1;
    }


    // random access
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(1, 0, 0)),
            grid.getValue(ijk.offsetBy(-1, 0, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0, -1, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0, -1)));
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference(S.template getValue< 1, 0, 0>(), S.template getValue<-1, 0, 0>());
    }
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 1, 0>(), S.template getValue< 0,-1, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0,-1>());
    }

};

template<>
struct D1<CD_4TH>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp2, const ValueType& xp1,
               const ValueType& xm1, const ValueType& xm2 )
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp1 = xp1, xcp2 = xp2, xcm1 = xm1, xcm2 = xm2;

        return ComputeType(2./3.)*(xcp1 - xcm1) + ComputeType(1./12.)*(xcm2 - xcp2);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 2,0,0)), grid.getValue(ijk.offsetBy( 1,0,0)),
            grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk.offsetBy(-2,0,0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 2, 0)), grid.getValue(ijk.offsetBy( 0, 1, 0)),
            grid.getValue(ijk.offsetBy( 0,-1, 0)), grid.getValue(ijk.offsetBy( 0,-2, 0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 2)), grid.getValue(ijk.offsetBy( 0, 0, 1)),
            grid.getValue(ijk.offsetBy( 0, 0,-1)), grid.getValue(ijk.offsetBy( 0, 0,-2)) );
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0,-2>() );
    }
};

template<>
struct D1<CD_6TH>
{

    // the difference operator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp3, const ValueType& xp2, const ValueType& xp1,
               const ValueType& xm1, const ValueType& xm2, const ValueType& xm3 )
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp1 = xp1, xcp2 = xp2, xcp3 = xp3, xcm1 = xm1, xcm2 = xm2, xcm3 = xm3;

        return ComputeType(3./4.)*(xcp1 - xcm1) - ComputeType(0.15)*(xcp2 - xcm2)
            + ComputeType(1./60.)*(xcp3 - xcm3);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 3,0,0)), grid.getValue(ijk.offsetBy( 2,0,0)),
            grid.getValue(ijk.offsetBy( 1,0,0)), grid.getValue(ijk.offsetBy(-1,0,0)),
            grid.getValue(ijk.offsetBy(-2,0,0)), grid.getValue(ijk.offsetBy(-3,0,0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0)), grid.getValue(ijk.offsetBy( 0, 2, 0)),
            grid.getValue(ijk.offsetBy( 0, 1, 0)), grid.getValue(ijk.offsetBy( 0,-1, 0)),
            grid.getValue(ijk.offsetBy( 0,-2, 0)), grid.getValue(ijk.offsetBy( 0,-3, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3)), grid.getValue(ijk.offsetBy( 0, 0, 2)),
            grid.getValue(ijk.offsetBy( 0, 0, 1)), grid.getValue(ijk.offsetBy( 0, 0,-1)),
            grid.getValue(ijk.offsetBy( 0, 0,-2)), grid.getValue(ijk.offsetBy( 0, 0,-3)));
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return  difference(S.template getValue< 3, 0, 0>(),
                           S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>(),
                           S.template getValue<-3, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {

        return  difference( S.template getValue< 0, 3, 0>(),
                            S.template getValue< 0, 2, 0>(),
                            S.template getValue< 0, 1, 0>(),
                            S.template getValue< 0,-1, 0>(),
                            S.template getValue< 0,-2, 0>(),
                            S.template getValue< 0,-3, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {

        return  difference( S.template getValue< 0, 0, 3>(),
                            S.template getValue< 0, 0, 2>(),
                            S.template getValue< 0, 0, 1>(),
                            S.template getValue< 0, 0,-1>(),
                            S.template getValue< 0, 0,-2>(),
                            S.template getValue< 0, 0,-3>());
    }
};


template<>
struct D1<FD_1ST>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp1, const ValueType& xp0)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        return ComputeType(xp1) - ComputeType(xp0);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(1, 0, 0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 1, 0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 0, 1)), grid.getValue(ijk));
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference(S.template getValue< 1, 0, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 1, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0, 0>());
    }
};


template<>
struct D1<FD_2ND>
{
    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp2, const ValueType& xp1, const ValueType& xp0)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcp2 = xp2;

        return ComputeType(2)*xp1 -(ComputeType(0.5)*xp2 + ComputeType(3./2.)*xp0);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(2,0,0)),
            grid.getValue(ijk.offsetBy(1,0,0)),
            grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0,2,0)),
            grid.getValue(ijk.offsetBy(0,1,0)),
            grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0,0,2)),
            grid.getValue(ijk.offsetBy(0,0,1)),
            grid.getValue(ijk));
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>() );
    }

};


template<>
struct D1<FD_3RD>
{

    // the difference opperator
    template<typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp3, const ValueType& xp2,
               const ValueType& xp1, const ValueType& xp0)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcp2 = xp2, xcp3 = xp3;

        return ComputeType(1.0/3.0)*xcp3 - ComputeType(1.5)*xcp2
            + ComputeType(3)*xcp1 - ComputeType(11.0/6.0)*xcp0;
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(3,0,0)),
                           grid.getValue(ijk.offsetBy(2,0,0)),
                           grid.getValue(ijk.offsetBy(1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,3,0)),
                           grid.getValue(ijk.offsetBy(0,2,0)),
                           grid.getValue(ijk.offsetBy(0,1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,0,3)),
                           grid.getValue(ijk.offsetBy(0,0,2)),
                           grid.getValue(ijk.offsetBy(0,0,1)),
                           grid.getValue(ijk) );
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference(S.template getValue< 3, 0, 0>(),
                          S.template getValue< 2, 0, 0>(),
                          S.template getValue< 1, 0, 0>(),
                          S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 3, 0>(),
                          S.template getValue< 0, 2, 0>(),
                          S.template getValue< 0, 1, 0>(),
                          S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 3>(),
                           S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>() );
    }
};


template<>
struct D1<BD_1ST>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xm1, const ValueType& xm0)
    {
        return -D1<FD_1ST>::difference(xm1, xm0);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0,-1,0)), grid.getValue(ijk));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference(grid.getValue(ijk.offsetBy(0, 0,-1)), grid.getValue(ijk));
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference(S.template getValue<-1, 0, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference(S.template getValue< 0,-1, 0>(), S.template getValue< 0, 0, 0>());
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0,-1>(), S.template getValue< 0, 0, 0>());
    }
};


template<>
struct D1<BD_2ND>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xm2, const ValueType& xm1, const ValueType& xm0)
    {
        return -D1<FD_2ND>::difference(xm2, xm1, xm0);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(-2,0,0)),
                           grid.getValue(ijk.offsetBy(-1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,-2,0)),
                           grid.getValue(ijk.offsetBy(0,-1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(0,0,-2)),
                           grid.getValue(ijk.offsetBy(0,0,-1)),
                           grid.getValue(ijk) );
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue<-2, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0, 0>() );
    }
};


template<>
struct D1<BD_3RD>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xm3, const ValueType& xm2,
               const ValueType& xm1, const ValueType& xm0)
    {
        return -D1<FD_3RD>::difference(xm3, xm2, xm1, xm0);
    }

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy(-3,0,0)),
                           grid.getValue(ijk.offsetBy(-2,0,0)),
                           grid.getValue(ijk.offsetBy(-1,0,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0,-3,0)),
                           grid.getValue(ijk.offsetBy( 0,-2,0)),
                           grid.getValue(ijk.offsetBy( 0,-1,0)),
                           grid.getValue(ijk) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0, 0,-3)),
                           grid.getValue(ijk.offsetBy( 0, 0,-2)),
                           grid.getValue(ijk.offsetBy( 0, 0,-1)),
                           grid.getValue(ijk) );
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue<-3, 0, 0>(),
                           S.template getValue<-2, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0,-3, 0>(),
                           S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0,-3>(),
                           S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0, 0>() );
    }

};

template<>
struct D1<FD_WENO5>
{
    // the difference operator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp3, const ValueType& xp2,
               const ValueType& xp1, const ValueType& xp0,
               const ValueType& xm1, const ValueType& xm2)
    {
        return WENO5(xp3, xp2, xp1, xp0, xm1) - WENO5(xp2, xp1, xp0, xm1, xm2);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(-2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,-2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,-2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {

        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 3, 0, 0>(),
            S.template getValue< 2, 0, 0>(),
            S.template getValue< 1, 0, 0>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue<-1, 0, 0>(),
            S.template getValue<-2, 0, 0>() ));

    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 0, 3, 0>(),
            S.template getValue< 0, 2, 0>(),
            S.template getValue< 0, 1, 0>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue< 0,-1, 0>(),
            S.template getValue< 0,-2, 0>() ));
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return static_cast<typename Stencil::ValueType>(difference(
            S.template getValue< 0, 0, 3>(),
            S.template getValue< 0, 0, 2>(),
            S.template getValue< 0, 0, 1>(),
            S.template getValue< 0, 0, 0>(),
            S.template getValue< 0, 0,-1>(),
            S.template getValue< 0, 0,-2>() ));
    }
};

template<>
struct D1<FD_HJWENO5>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp3, const ValueType& xp2,
               const ValueType& xp1, const ValueType& xp0,
               const ValueType& xm1, const ValueType& xm2)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcp2 = xp2, xcp3 = xp3, xcm1 = xm1, xcm2 = xm2;

        return WENO5<ComputeType>(xcp3 - xcp2, xcp2 - xcp1, xcp1 - xcp0, xcp0-xcm1, xcm1-xcm2);
    }

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(-2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);

    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,-2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,-2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {

        return difference( S.template getValue< 3, 0, 0>(),
                           S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),
                           S.template getValue<-2, 0, 0>() );

    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 3, 0>(),
                           S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>(),
                           S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {

        return difference( S.template getValue< 0, 0, 3>(),
                           S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),
                           S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>(),
                           S.template getValue< 0, 0,-2>() );
    }

};

template<>
struct D1<BD_WENO5>
{

    template<typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xm3, const ValueType& xm2, const ValueType& xm1,
               const ValueType& xm0, const ValueType& xp1, const ValueType& xp2)
    {
        return -D1<FD_WENO5>::difference(xm3, xm2, xm1, xm0, xp1, xp2);
    }


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(-3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(-2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,-3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,-2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,-3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,-2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue<-3, 0, 0>();
        V[1] = S.template getValue<-2, 0, 0>();
        V[2] = S.template getValue<-1, 0, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 1, 0, 0>();
        V[5] = S.template getValue< 2, 0, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0,-3, 0>();
        V[1] = S.template getValue< 0,-2, 0>();
        V[2] = S.template getValue< 0,-1, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 1, 0>();
        V[5] = S.template getValue< 0, 2, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0, 0,-3>();
        V[1] = S.template getValue< 0, 0,-2>();
        V[2] = S.template getValue< 0, 0,-1>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 0, 1>();
        V[5] = S.template getValue< 0, 0, 2>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }
};


template<>
struct D1<BD_HJWENO5>
{
    template<typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xm3, const ValueType& xm2, const ValueType& xm1,
               const ValueType& xm0, const ValueType& xp1, const ValueType& xp2)
    {
        return -D1<FD_HJWENO5>::difference(xm3, xm2, xm1, xm0, xp1, xp2);
    }

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(-3,0,0));
        V[1] = grid.getValue(ijk.offsetBy(-2,0,0));
        V[2] = grid.getValue(ijk.offsetBy(-1,0,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(1,0,0));
        V[5] = grid.getValue(ijk.offsetBy(2,0,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,-3,0));
        V[1] = grid.getValue(ijk.offsetBy(0,-2,0));
        V[2] = grid.getValue(ijk.offsetBy(0,-1,0));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,1,0));
        V[5] = grid.getValue(ijk.offsetBy(0,2,0));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType = typename Accessor::ValueType;
        ValueType V[6];
        V[0] = grid.getValue(ijk.offsetBy(0,0,-3));
        V[1] = grid.getValue(ijk.offsetBy(0,0,-2));
        V[2] = grid.getValue(ijk.offsetBy(0,0,-1));
        V[3] = grid.getValue(ijk);
        V[4] = grid.getValue(ijk.offsetBy(0,0,1));
        V[5] = grid.getValue(ijk.offsetBy(0,0,2));

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue<-3, 0, 0>();
        V[1] = S.template getValue<-2, 0, 0>();
        V[2] = S.template getValue<-1, 0, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 1, 0, 0>();
        V[5] = S.template getValue< 2, 0, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0,-3, 0>();
        V[1] = S.template getValue< 0,-2, 0>();
        V[2] = S.template getValue< 0,-1, 0>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 1, 0>();
        V[5] = S.template getValue< 0, 2, 0>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        using ValueType = typename Stencil::ValueType;
        ValueType V[6];
        V[0] = S.template getValue< 0, 0,-3>();
        V[1] = S.template getValue< 0, 0,-2>();
        V[2] = S.template getValue< 0, 0,-1>();
        V[3] = S.template getValue< 0, 0, 0>();
        V[4] = S.template getValue< 0, 0, 1>();
        V[5] = S.template getValue< 0, 0, 2>();

        return difference(V[0], V[1], V[2], V[3], V[4], V[5]);
    }
};


template<DScheme DiffScheme>
struct D1Vec
{
    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inX(grid, ijk)[n];
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inY(grid, ijk)[n];
    }
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<DiffScheme>::inZ(grid, ijk)[n];
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inX(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inX(S)[n];
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inY(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inY(S)[n];
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inZ(const Stencil& S, int n)
    {
        return D1<DiffScheme>::inZ(S)[n];
    }
};


template<>
struct D1Vec<CD_2NDT>
{

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy( 1, 0, 0))[n],
                                        grid.getValue(ijk.offsetBy(-1, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy(0, 1, 0))[n],
                                        grid.getValue(ijk.offsetBy(0,-1, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2NDT>::difference( grid.getValue(ijk.offsetBy(0, 0, 1))[n],
                                        grid.getValue(ijk.offsetBy(0, 0,-1))[n] );
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inX(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 1, 0, 0>()[n],
                                        S.template getValue<-1, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inY(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 0, 1, 0>()[n],
                                        S.template getValue< 0,-1, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inZ(const Stencil& S, int n)
    {
        return D1<CD_2NDT>::difference( S.template getValue< 0, 0, 1>()[n],
                                        S.template getValue< 0, 0,-1>()[n] );
    }
};

template<>
struct D1Vec<CD_2ND>
{

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy( 1, 0, 0))[n] ,
                                       grid.getValue(ijk.offsetBy(-1, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy(0, 1, 0))[n] ,
                                       grid.getValue(ijk.offsetBy(0,-1, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_2ND>::difference( grid.getValue(ijk.offsetBy(0, 0, 1))[n] ,
                                       grid.getValue(ijk.offsetBy(0, 0,-1))[n] );
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inX(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 1, 0, 0>()[n],
                                       S.template getValue<-1, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inY(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 0, 1, 0>()[n],
                                       S.template getValue< 0,-1, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inZ(const Stencil& S, int n)
    {
        return D1<CD_2ND>::difference( S.template getValue< 0, 0, 1>()[n],
                                       S.template getValue< 0, 0,-1>()[n] );
    }
};


template<>
struct D1Vec<CD_4TH> {
    // using value_type = typename Accessor::ValueType::value_type;


    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy(2, 0, 0))[n], grid.getValue(ijk.offsetBy( 1, 0, 0))[n],
            grid.getValue(ijk.offsetBy(-1,0, 0))[n], grid.getValue(ijk.offsetBy(-2, 0, 0))[n]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 2, 0))[n], grid.getValue(ijk.offsetBy( 0, 1, 0))[n],
            grid.getValue(ijk.offsetBy( 0,-1, 0))[n], grid.getValue(ijk.offsetBy( 0,-2, 0))[n]);
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_4TH>::difference(
            grid.getValue(ijk.offsetBy(0,0, 2))[n], grid.getValue(ijk.offsetBy( 0, 0, 1))[n],
            grid.getValue(ijk.offsetBy(0,0,-1))[n], grid.getValue(ijk.offsetBy( 0, 0,-2))[n]);
    }

    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inX(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 2, 0, 0>()[n],  S.template getValue< 1, 0, 0>()[n],
            S.template getValue<-1, 0, 0>()[n],  S.template getValue<-2, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inY(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 0, 2, 0>()[n],  S.template getValue< 0, 1, 0>()[n],
            S.template getValue< 0,-1, 0>()[n],  S.template getValue< 0,-2, 0>()[n]);
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inZ(const Stencil& S, int n)
    {
        return D1<CD_4TH>::difference(
            S.template getValue< 0, 0, 2>()[n],  S.template getValue< 0, 0, 1>()[n],
            S.template getValue< 0, 0,-1>()[n],  S.template getValue< 0, 0,-2>()[n]);
    }
};


template<>
struct D1Vec<CD_6TH>
{
    //using ValueType = typename Accessor::ValueType::value_type::value_type;

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inX(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 3, 0, 0))[n], grid.getValue(ijk.offsetBy( 2, 0, 0))[n],
            grid.getValue(ijk.offsetBy( 1, 0, 0))[n], grid.getValue(ijk.offsetBy(-1, 0, 0))[n],
            grid.getValue(ijk.offsetBy(-2, 0, 0))[n], grid.getValue(ijk.offsetBy(-3, 0, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inY(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0))[n], grid.getValue(ijk.offsetBy( 0, 2, 0))[n],
            grid.getValue(ijk.offsetBy( 0, 1, 0))[n], grid.getValue(ijk.offsetBy( 0,-1, 0))[n],
            grid.getValue(ijk.offsetBy( 0,-2, 0))[n], grid.getValue(ijk.offsetBy( 0,-3, 0))[n] );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType::value_type>::type
    inZ(const Accessor& grid, const Coord& ijk, int n)
    {
        return D1<CD_6TH>::difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3))[n], grid.getValue(ijk.offsetBy( 0, 0, 2))[n],
            grid.getValue(ijk.offsetBy( 0, 0, 1))[n], grid.getValue(ijk.offsetBy( 0, 0,-1))[n],
            grid.getValue(ijk.offsetBy( 0, 0,-2))[n], grid.getValue(ijk.offsetBy( 0, 0,-3))[n] );
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inX(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 3, 0, 0>()[n], S.template getValue< 2, 0, 0>()[n],
            S.template getValue< 1, 0, 0>()[n], S.template getValue<-1, 0, 0>()[n],
            S.template getValue<-2, 0, 0>()[n], S.template getValue<-3, 0, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inY(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 0, 3, 0>()[n], S.template getValue< 0, 2, 0>()[n],
            S.template getValue< 0, 1, 0>()[n], S.template getValue< 0,-1, 0>()[n],
            S.template getValue< 0,-2, 0>()[n], S.template getValue< 0,-3, 0>()[n] );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType::value_type>::type
    inZ(const Stencil& S, int n)
    {
        return D1<CD_6TH>::difference(
            S.template getValue< 0, 0, 3>()[n], S.template getValue< 0, 0, 2>()[n],
            S.template getValue< 0, 0, 1>()[n], S.template getValue< 0, 0,-1>()[n],
            S.template getValue< 0, 0,-2>()[n], S.template getValue< 0, 0,-3>()[n] );
    }
};

template<DDScheme DiffScheme>
struct D2
{

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk);

    // cross derivatives
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandY(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandZ(const Accessor& grid, const Coord& ijk);

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inYandZ(const Accessor& grid, const Coord& ijk);


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S);

    // cross derivatives
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandY(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandZ(const Stencil& S);

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inYandZ(const Stencil& S);
};

template<>
struct D2<CD_SECOND>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp1, const ValueType& xp0, const ValueType& xm1)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcm1 = xm1;

        return xcp1 + xcm1 - ComputeType(2)*xcp0;
    }

    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    crossdifference(const ValueType& xpyp, const ValueType& xpym,
                    const ValueType& xmyp, const ValueType& xmym)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcpyp = xpyp, xcpym = xpym, xcmyp = xmyp, xcmym = xmym;

        return ComputeType(0.25)*(xcpyp + xcmym - xcpym - xcmyp);
    }

    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 1,0,0)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy(-1,0,0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {

        return difference( grid.getValue(ijk.offsetBy(0, 1,0)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy(0,-1,0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
        return difference( grid.getValue(ijk.offsetBy( 0,0, 1)), grid.getValue(ijk),
                           grid.getValue(ijk.offsetBy( 0,0,-1)) );
    }

    // cross derivatives
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandY(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(1, 1,0)), grid.getValue(ijk.offsetBy( 1,-1,0)),
            grid.getValue(ijk.offsetBy(-1,1,0)), grid.getValue(ijk.offsetBy(-1,-1,0)));

    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandZ(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(1,0, 1)), grid.getValue(ijk.offsetBy(1, 0,-1)),
            grid.getValue(ijk.offsetBy(-1,0,1)), grid.getValue(ijk.offsetBy(-1,0,-1)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inYandZ(const Accessor& grid, const Coord& ijk)
    {
        return crossdifference(
            grid.getValue(ijk.offsetBy(0, 1,1)), grid.getValue(ijk.offsetBy(0, 1,-1)),
            grid.getValue(ijk.offsetBy(0,-1,1)), grid.getValue(ijk.offsetBy(0,-1,-1)) );
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue< 1, 0, 0>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 1, 0>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 1>(), S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>() );
    }

    // cross derivatives
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandY(const Stencil& S)
    {
        return crossdifference(S.template getValue< 1, 1, 0>(),  S.template getValue< 1,-1, 0>(),
                               S.template getValue<-1, 1, 0>(),  S.template getValue<-1,-1, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandZ(const Stencil& S)
    {
        return crossdifference(S.template getValue< 1, 0, 1>(),  S.template getValue< 1, 0,-1>(),
                               S.template getValue<-1, 0, 1>(),  S.template getValue<-1, 0,-1>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inYandZ(const Stencil& S)
    {
        return crossdifference(S.template getValue< 0, 1, 1>(),  S.template getValue< 0, 1,-1>(),
                               S.template getValue< 0,-1, 1>(),  S.template getValue< 0,-1,-1>() );
    }
};


template<>
struct D2<CD_FOURTH>
{

    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp2, const ValueType& xp1, const ValueType& xp0,
               const ValueType& xm1, const ValueType& xm2)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcp2 = xp2, xcm1 = xm1, xcm2 = xm2;

        return ComputeType(-1./12.)*(xcp2 + xcm2)
            + ComputeType(4./3.)*(xcp1 + xcm1) - ComputeType(2.5)*xcp0;
    }

    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    crossdifference(const ValueType& xp2yp2, const ValueType& xp2yp1,
                    const ValueType& xp2ym1, const ValueType& xp2ym2,
                    const ValueType& xp1yp2, const ValueType& xp1yp1,
                    const ValueType& xp1ym1, const ValueType& xp1ym2,
                    const ValueType& xm2yp2, const ValueType& xm2yp1,
                    const ValueType& xm2ym1, const ValueType& xm2ym2,
                    const ValueType& xm1yp2, const ValueType& xm1yp1,
                    const ValueType& xm1ym1, const ValueType& xm1ym2 )
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp2yp2 = xp2yp2, xcp2yp1 = xp2yp1, xcp2ym1 = xp2ym1, xcp2ym2 = xp2ym2,
                          xcp1yp2 = xp1yp2, xcp1yp1 = xp1yp1, xcp1ym1 = xp1ym1, xcp1ym2 = xp1ym2,
                          xcm2yp2 = xm2yp2, xcm2yp1 = xm2yp1, xcm2ym1 = xm2ym1, xcm2ym2 = xm2ym2,
                          xcm1yp2 = xm1yp2, xcm1yp1 = xm1yp1, xcm1ym1 = xm1ym1, xcm1ym2 = xm1ym2;

        const ComputeType tmp1 =
            ComputeType(2./3.)*(xcp1yp1 - xcm1yp1 - xcp1ym1 + xcm1ym1)-
            ComputeType(1./12.)*(xcp2yp1 - xcm2yp1 - xcp2ym1 + xcm2ym1);

        const ComputeType tmp2 =
            ComputeType(2./3.)*(xcp1yp2 - xcm1yp2 - xcp1ym2 + xcm1ym2)-
            ComputeType(1./12.)*(xcp2yp2 - xcm2yp2 - xcp2ym2 + xcm2ym2);

        return ComputeType(2./3.)*tmp1 - ComputeType(1./12.)*tmp2;
    }



    // random access version
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(2,0,0)),  grid.getValue(ijk.offsetBy( 1,0,0)),
            grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(-1,0,0)), grid.getValue(ijk.offsetBy(-2, 0, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy(0, 2,0)), grid.getValue(ijk.offsetBy(0, 1,0)),
            grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(0,-1,0)), grid.getValue(ijk.offsetBy(0,-2, 0)));
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {
         return difference(
             grid.getValue(ijk.offsetBy(0,0, 2)), grid.getValue(ijk.offsetBy(0, 0,1)),
             grid.getValue(ijk),
             grid.getValue(ijk.offsetBy(0,0,-1)), grid.getValue(ijk.offsetBy(0,0,-2)));
    }

    // cross derivatives
    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandY(const Accessor& grid, const Coord& ijk)
    {
        using ValueType   = typename Accessor::ValueType;
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        ComputeType tmp1 =
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 1, 0)) -
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0,-1, 0));
        ComputeType tmp2 =
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 2, 0)) -
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0,-2, 0));

        return ComputeType(2./3.)*tmp1 - ComputeType(1./12.)*tmp2;
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType   = typename Accessor::ValueType;
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        ComputeType tmp1 =
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0,-1));
        ComputeType tmp2 =
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_4TH>::inX<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0,-2));

        return ComputeType(2./3.)*tmp1 - ComputeType(1./12.)*tmp2;
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inYandZ(const Accessor& grid, const Coord& ijk)
    {
        using ValueType   = typename Accessor::ValueType;
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        ComputeType tmp1 =
            D1<CD_4TH>::inY<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_4TH>::inY<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0,-1));
        ComputeType tmp2 =
            D1<CD_4TH>::inY<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_4TH>::inY<ValueType, ComputeType>(grid, ijk.offsetBy(0, 0,-2));

        return ComputeType(2./3.)*tmp1 - ComputeType(1./12.)*tmp2;
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference(S.template getValue< 2, 0, 0>(), S.template getValue< 1, 0, 0>(),
                          S.template getValue< 0, 0, 0>(),
                          S.template getValue<-1, 0, 0>(), S.template getValue<-2, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference(S.template getValue< 0, 2, 0>(), S.template getValue< 0, 1, 0>(),
                          S.template getValue< 0, 0, 0>(),
                          S.template getValue< 0,-1, 0>(), S.template getValue< 0,-2, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference(S.template getValue< 0, 0, 2>(), S.template getValue< 0, 0, 1>(),
                          S.template getValue< 0, 0, 0>(),
                          S.template getValue< 0, 0,-1>(), S.template getValue< 0, 0,-2>() );
    }

    // cross derivatives
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandY(const Stencil& S)
     {
         return crossdifference(
             S.template getValue< 2, 2, 0>(), S.template getValue< 2, 1, 0>(),
             S.template getValue< 2,-1, 0>(), S.template getValue< 2,-2, 0>(),
             S.template getValue< 1, 2, 0>(), S.template getValue< 1, 1, 0>(),
             S.template getValue< 1,-1, 0>(), S.template getValue< 1,-2, 0>(),
             S.template getValue<-2, 2, 0>(), S.template getValue<-2, 1, 0>(),
             S.template getValue<-2,-1, 0>(), S.template getValue<-2,-2, 0>(),
             S.template getValue<-1, 2, 0>(), S.template getValue<-1, 1, 0>(),
             S.template getValue<-1,-1, 0>(), S.template getValue<-1,-2, 0>() );
     }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandZ(const Stencil& S)
    {
        return crossdifference(
            S.template getValue< 2, 0, 2>(), S.template getValue< 2, 0, 1>(),
            S.template getValue< 2, 0,-1>(), S.template getValue< 2, 0,-2>(),
            S.template getValue< 1, 0, 2>(), S.template getValue< 1, 0, 1>(),
            S.template getValue< 1, 0,-1>(), S.template getValue< 1, 0,-2>(),
            S.template getValue<-2, 0, 2>(), S.template getValue<-2, 0, 1>(),
            S.template getValue<-2, 0,-1>(), S.template getValue<-2, 0,-2>(),
            S.template getValue<-1, 0, 2>(), S.template getValue<-1, 0, 1>(),
            S.template getValue<-1, 0,-1>(), S.template getValue<-1, 0,-2>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inYandZ(const Stencil& S)
    {
        return crossdifference(
            S.template getValue< 0, 2, 2>(), S.template getValue< 0, 2, 1>(),
            S.template getValue< 0, 2,-1>(), S.template getValue< 0, 2,-2>(),
            S.template getValue< 0, 1, 2>(), S.template getValue< 0, 1, 1>(),
            S.template getValue< 0, 1,-1>(), S.template getValue< 0, 1,-2>(),
            S.template getValue< 0,-2, 2>(), S.template getValue< 0,-2, 1>(),
            S.template getValue< 0,-2,-1>(), S.template getValue< 0,-2,-2>(),
            S.template getValue< 0,-1, 2>(), S.template getValue< 0,-1, 1>(),
            S.template getValue< 0,-1,-1>(), S.template getValue< 0,-1,-2>() );
    }
};


template<>
struct D2<CD_SIXTH>
{
    // the difference opperator
    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    difference(const ValueType& xp3, const ValueType& xp2, const ValueType& xp1,
               const ValueType& xp0,
               const ValueType& xm1, const ValueType& xm2, const ValueType& xm3)
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp0 = xp0, xcp1 = xp1, xcp2 = xp2, xcp3 = xp3,
                          xcm1 = xm1, xcm2 = xm2, xcm3 = xm3;

        return ComputeType(1./90.)*(xcp3 + xcm3) - ComputeType(3./20.)*(xcp2 + xcm2)
            + ComputeType(1.5)*(xcp1 + xcm1) - ComputeType(49./18.)*xcp0;
    }

    template <typename ValueType>
    static typename ComputeTypeFor<ValueType>::type
    crossdifference(const ValueType& xp1yp1, const ValueType& xm1yp1,
                    const ValueType& xp1ym1, const ValueType& xm1ym1,
                    const ValueType& xp2yp1, const ValueType& xm2yp1,
                    const ValueType& xp2ym1, const ValueType& xm2ym1,
                    const ValueType& xp3yp1, const ValueType& xm3yp1,
                    const ValueType& xp3ym1, const ValueType& xm3ym1,
                    const ValueType& xp1yp2, const ValueType& xm1yp2,
                    const ValueType& xp1ym2, const ValueType& xm1ym2,
                    const ValueType& xp2yp2, const ValueType& xm2yp2,
                    const ValueType& xp2ym2, const ValueType& xm2ym2,
                    const ValueType& xp3yp2, const ValueType& xm3yp2,
                    const ValueType& xp3ym2, const ValueType& xm3ym2,
                    const ValueType& xp1yp3, const ValueType& xm1yp3,
                    const ValueType& xp1ym3, const ValueType& xm1ym3,
                    const ValueType& xp2yp3, const ValueType& xm2yp3,
                    const ValueType& xp2ym3, const ValueType& xm2ym3,
                    const ValueType& xp3yp3, const ValueType& xm3yp3,
                    const ValueType& xp3ym3, const ValueType& xm3ym3 )
    {
        using ComputeType = typename ComputeTypeFor<ValueType>::type;

        const ComputeType xcp1yp1 = xp1yp1, xcm1yp1 = xm1yp1, xcp1ym1 = xp1ym1, xcm1ym1 = xm1ym1,
                          xcp2yp1 = xp2yp1, xcm2yp1 = xm2yp1, xcp2ym1 = xp2ym1, xcm2ym1 = xm2ym1,
                          xcp3yp1 = xp3yp1, xcm3yp1 = xm3yp1, xcp3ym1 = xp3ym1, xcm3ym1 = xm3ym1,
                          xcp1yp2 = xp1yp2, xcm1yp2 = xm1yp2, xcp1ym2 = xp1ym2, xcm1ym2 = xm1ym2,
                          xcp2yp2 = xp2yp2, xcm2yp2 = xm2yp2, xcp2ym2 = xp2ym2, xcm2ym2 = xm2ym2,
                          xcp3yp2 = xp3yp2, xcm3yp2 = xm3yp2, xcp3ym2 = xp3ym2, xcm3ym2 = xm3ym2,
                          xcp1yp3 = xp1yp3, xcm1yp3 = xm1yp3, xcp1ym3 = xp1ym3, xcm1ym3 = xm1ym3,
                          xcp2yp3 = xp2yp3, xcm2yp3 = xm2yp3, xcp2ym3 = xp2ym3, xcm2ym3 = xm2ym3,
                          xcp3yp3 = xp3yp3, xcm3yp3 = xm3yp3, xcp3ym3 = xp3ym3, xcm3ym3 = xm3ym3;

        ComputeType tmp1 =
            ComputeType(0.7500)*(xcp1yp1 - xcm1yp1 - xcp1ym1 + xcm1ym1) -
            ComputeType(0.1500)*(xcp2yp1 - xcm2yp1 - xcp2ym1 + xcm2ym1) +
            ComputeType(1./60.)*(xcp3yp1 - xcm3yp1 - xcp3ym1 + xcm3ym1);

        ComputeType tmp2 =
            ComputeType(0.7500)*(xcp1yp2 - xcm1yp2 - xcp1ym2 + xcm1ym2) -
            ComputeType(0.1500)*(xcp2yp2 - xcm2yp2 - xcp2ym2 + xcm2ym2) +
            ComputeType(1./60.)*(xcp3yp2 - xcm3yp2 - xcp3ym2 + xcm3ym2);

        ComputeType tmp3 =
            ComputeType(0.7500)*(xcp1yp3 - xcm1yp3 - xcp1ym3 + xcm1ym3) -
            ComputeType(0.1500)*(xcp2yp3 - xcm2yp3 - xcp2ym3 + xcm2ym3) +
            ComputeType(1./60.)*(xcp3yp3 - xcm3yp3 - xcp3ym3 + xcm3ym3);

        return ComputeType(0.75)*tmp1 - ComputeType(0.15)*tmp2 + ComputeType(1./60)*tmp3;
    }

    // random access version

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inX(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 3, 0, 0)), grid.getValue(ijk.offsetBy( 2, 0, 0)),
            grid.getValue(ijk.offsetBy( 1, 0, 0)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy(-1, 0, 0)), grid.getValue(ijk.offsetBy(-2, 0, 0)),
            grid.getValue(ijk.offsetBy(-3, 0, 0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inY(const Accessor& grid, const Coord& ijk)
    {
        return difference(
            grid.getValue(ijk.offsetBy( 0, 3, 0)), grid.getValue(ijk.offsetBy( 0, 2, 0)),
            grid.getValue(ijk.offsetBy( 0, 1, 0)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy( 0,-1, 0)), grid.getValue(ijk.offsetBy( 0,-2, 0)),
            grid.getValue(ijk.offsetBy( 0,-3, 0)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inZ(const Accessor& grid, const Coord& ijk)
    {

        return difference(
            grid.getValue(ijk.offsetBy( 0, 0, 3)), grid.getValue(ijk.offsetBy( 0, 0, 2)),
            grid.getValue(ijk.offsetBy( 0, 0, 1)), grid.getValue(ijk),
            grid.getValue(ijk.offsetBy( 0, 0,-1)), grid.getValue(ijk.offsetBy( 0, 0,-2)),
            grid.getValue(ijk.offsetBy( 0, 0,-3)) );
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandY(const Accessor& grid, const Coord& ijk)
    {
        using ComputeType = typename ComputeTypeFor<typename Accessor::ValueType>::type;

        ComputeType tmp1 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 1, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-1, 0));
        ComputeType tmp2 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 2, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-2, 0));
        ComputeType tmp3 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 3, 0)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0,-3, 0));

        return ComputeType(0.75)*tmp1 - ComputeType(0.15)*tmp2 + ComputeType(1./60)*tmp3;
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inXandZ(const Accessor& grid, const Coord& ijk)
    {
        using ComputeType = typename ComputeTypeFor<typename Accessor::ValueType>::type;

        ComputeType tmp1 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-1));
        ComputeType tmp2 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-2));
        ComputeType tmp3 =
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0, 3)) -
            D1<CD_6TH>::inX(grid, ijk.offsetBy(0, 0,-3));

        return ComputeType(0.75)*tmp1 - ComputeType(0.15)*tmp2 + ComputeType(1./60)*tmp3;
    }

    template<typename Accessor>
    static typename ComputeTypeFor<typename Accessor::ValueType>::type
    inYandZ(const Accessor& grid, const Coord& ijk)
    {
        using ComputeType = typename ComputeTypeFor<typename Accessor::ValueType>::type;

        ComputeType tmp1 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 1)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-1));
        ComputeType tmp2 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 2)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-2));
        ComputeType tmp3 =
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0, 3)) -
            D1<CD_6TH>::inY(grid, ijk.offsetBy(0, 0,-3));

        return ComputeType(0.75)*tmp1 - ComputeType(0.15)*tmp2 + ComputeType(1./60)*tmp3;
    }


    // stencil access version
    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inX(const Stencil& S)
    {
        return difference( S.template getValue< 3, 0, 0>(),  S.template getValue< 2, 0, 0>(),
                           S.template getValue< 1, 0, 0>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue<-1, 0, 0>(),  S.template getValue<-2, 0, 0>(),
                           S.template getValue<-3, 0, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inY(const Stencil& S)
    {
        return difference( S.template getValue< 0, 3, 0>(),  S.template getValue< 0, 2, 0>(),
                           S.template getValue< 0, 1, 0>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0,-1, 0>(),  S.template getValue< 0,-2, 0>(),
                           S.template getValue< 0,-3, 0>() );

    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inZ(const Stencil& S)
    {
        return difference( S.template getValue< 0, 0, 3>(),  S.template getValue< 0, 0, 2>(),
                           S.template getValue< 0, 0, 1>(),  S.template getValue< 0, 0, 0>(),
                           S.template getValue< 0, 0,-1>(),  S.template getValue< 0, 0,-2>(),
                           S.template getValue< 0, 0,-3>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandY(const Stencil& S)
    {
        return crossdifference( S.template getValue< 1, 1, 0>(), S.template getValue<-1, 1, 0>(),
                                S.template getValue< 1,-1, 0>(), S.template getValue<-1,-1, 0>(),
                                S.template getValue< 2, 1, 0>(), S.template getValue<-2, 1, 0>(),
                                S.template getValue< 2,-1, 0>(), S.template getValue<-2,-1, 0>(),
                                S.template getValue< 3, 1, 0>(), S.template getValue<-3, 1, 0>(),
                                S.template getValue< 3,-1, 0>(), S.template getValue<-3,-1, 0>(),
                                S.template getValue< 1, 2, 0>(), S.template getValue<-1, 2, 0>(),
                                S.template getValue< 1,-2, 0>(), S.template getValue<-1,-2, 0>(),
                                S.template getValue< 2, 2, 0>(), S.template getValue<-2, 2, 0>(),
                                S.template getValue< 2,-2, 0>(), S.template getValue<-2,-2, 0>(),
                                S.template getValue< 3, 2, 0>(), S.template getValue<-3, 2, 0>(),
                                S.template getValue< 3,-2, 0>(), S.template getValue<-3,-2, 0>(),
                                S.template getValue< 1, 3, 0>(), S.template getValue<-1, 3, 0>(),
                                S.template getValue< 1,-3, 0>(), S.template getValue<-1,-3, 0>(),
                                S.template getValue< 2, 3, 0>(), S.template getValue<-2, 3, 0>(),
                                S.template getValue< 2,-3, 0>(), S.template getValue<-2,-3, 0>(),
                                S.template getValue< 3, 3, 0>(), S.template getValue<-3, 3, 0>(),
                                S.template getValue< 3,-3, 0>(), S.template getValue<-3,-3, 0>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inXandZ(const Stencil& S)
    {
        return crossdifference( S.template getValue< 1, 0, 1>(), S.template getValue<-1, 0, 1>(),
                                S.template getValue< 1, 0,-1>(), S.template getValue<-1, 0,-1>(),
                                S.template getValue< 2, 0, 1>(), S.template getValue<-2, 0, 1>(),
                                S.template getValue< 2, 0,-1>(), S.template getValue<-2, 0,-1>(),
                                S.template getValue< 3, 0, 1>(), S.template getValue<-3, 0, 1>(),
                                S.template getValue< 3, 0,-1>(), S.template getValue<-3, 0,-1>(),
                                S.template getValue< 1, 0, 2>(), S.template getValue<-1, 0, 2>(),
                                S.template getValue< 1, 0,-2>(), S.template getValue<-1, 0,-2>(),
                                S.template getValue< 2, 0, 2>(), S.template getValue<-2, 0, 2>(),
                                S.template getValue< 2, 0,-2>(), S.template getValue<-2, 0,-2>(),
                                S.template getValue< 3, 0, 2>(), S.template getValue<-3, 0, 2>(),
                                S.template getValue< 3, 0,-2>(), S.template getValue<-3, 0,-2>(),
                                S.template getValue< 1, 0, 3>(), S.template getValue<-1, 0, 3>(),
                                S.template getValue< 1, 0,-3>(), S.template getValue<-1, 0,-3>(),
                                S.template getValue< 2, 0, 3>(), S.template getValue<-2, 0, 3>(),
                                S.template getValue< 2, 0,-3>(), S.template getValue<-2, 0,-3>(),
                                S.template getValue< 3, 0, 3>(), S.template getValue<-3, 0, 3>(),
                                S.template getValue< 3, 0,-3>(), S.template getValue<-3, 0,-3>() );
    }

    template<typename Stencil>
    static typename ComputeTypeFor<typename Stencil::ValueType>::type
    inYandZ(const Stencil& S)
    {
        return crossdifference( S.template getValue< 0, 1, 1>(), S.template getValue< 0,-1, 1>(),
                                S.template getValue< 0, 1,-1>(), S.template getValue< 0,-1,-1>(),
                                S.template getValue< 0, 2, 1>(), S.template getValue< 0,-2, 1>(),
                                S.template getValue< 0, 2,-1>(), S.template getValue< 0,-2,-1>(),
                                S.template getValue< 0, 3, 1>(), S.template getValue< 0,-3, 1>(),
                                S.template getValue< 0, 3,-1>(), S.template getValue< 0,-3,-1>(),
                                S.template getValue< 0, 1, 2>(), S.template getValue< 0,-1, 2>(),
                                S.template getValue< 0, 1,-2>(), S.template getValue< 0,-1,-2>(),
                                S.template getValue< 0, 2, 2>(), S.template getValue< 0,-2, 2>(),
                                S.template getValue< 0, 2,-2>(), S.template getValue< 0,-2,-2>(),
                                S.template getValue< 0, 3, 2>(), S.template getValue< 0,-3, 2>(),
                                S.template getValue< 0, 3,-2>(), S.template getValue< 0,-3,-2>(),
                                S.template getValue< 0, 1, 3>(), S.template getValue< 0,-1, 3>(),
                                S.template getValue< 0, 1,-3>(), S.template getValue< 0,-1,-3>(),
                                S.template getValue< 0, 2, 3>(), S.template getValue< 0,-2, 3>(),
                                S.template getValue< 0, 2,-3>(), S.template getValue< 0,-2,-3>(),
                                S.template getValue< 0, 3, 3>(), S.template getValue< 0,-3, 3>(),
                                S.template getValue< 0, 3,-3>(), S.template getValue< 0,-3,-3>() );
    }

};

} // end math namespace
} // namespace OPENVDB_VERSION_NAME
} // end openvdb namespace

#endif // OPENVDB_MATH_FINITEDIFFERENCE_HAS_BEEN_INCLUDED
