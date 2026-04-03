#include "Simd.h"
#include <cmath>
#include <cstdio>

using namespace nanovdb::util;

// ---------------------------------------------------------------------------
// WENO5 upwind interpolation — templated on T
// T = float   : scalar, usable as __hostdev__ on GPU
// T = Simd<float,W> : W-wide vectorized version on CPU
// ---------------------------------------------------------------------------
template<typename T>
T weno5(T v1, T v2, T v3, T v4, T v5, float dx2 = 1.f)
{
    const float C   = 13.f / 12.f;
    const T     eps = T(1.0e-6f * dx2);

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
// godunovsNormSqrd — templated on T (value type) and MaskT (comparison result)
// MaskT = bool            when T = float          (GPU / scalar path)
// MaskT = SimdMask<float,W> when T = Simd<float,W> (CPU SIMD path)
// Mirrors GodunovsNormSqrd in nanovdb/math/Stencils.h
// ---------------------------------------------------------------------------
template<typename T, typename MaskT>
T godunovsNormSqrd(MaskT isOutside,
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
// normSqGrad — templated on T
// Input layout matches WenoStencil / WenoPt<i,j,k>::idx (19 values):
//   v0         = center  ( 0, 0, 0)
//   v1..v6     = x-axis  (-3,-2,-1, +1,+2,+3)
//   v7..v12    = y-axis  (-3,-2,-1, +1,+2,+3)
//   v13..v18   = z-axis  (-3,-2,-1, +1,+2,+3)
// ---------------------------------------------------------------------------
template<typename T>
T normSqGrad(T v0,
             T v1,  T v2,  T v3,  T v4,  T v5,  T v6,
             T v7,  T v8,  T v9,  T v10, T v11, T v12,
             T v13, T v14, T v15, T v16, T v17, T v18,
             float dx2, float invDx2, float isoValue = 0.f)
{
    const T dP_xm = weno5<T>(v2 -v1,  v3 -v2,  v0 -v3,  v4 -v0,  v5 -v4,  dx2);
    const T dP_xp = weno5<T>(v6 -v5,  v5 -v4,  v4 -v0,  v0 -v3,  v3 -v2,  dx2);
    const T dP_ym = weno5<T>(v8 -v7,  v9 -v8,  v0 -v9,  v10-v0,  v11-v10, dx2);
    const T dP_yp = weno5<T>(v12-v11, v11-v10, v10-v0,  v0 -v9,  v9 -v8,  dx2);
    const T dP_zm = weno5<T>(v14-v13, v15-v14, v0 -v15, v16-v0,  v17-v16, dx2);
    const T dP_zp = weno5<T>(v18-v17, v17-v16, v16-v0,  v0 -v15, v15-v14, dx2);

    return invDx2 * godunovsNormSqrd(v0 > T(isoValue),
                                     dP_xm, dP_xp,
                                     dP_ym, dP_yp,
                                     dP_zm, dP_zp);
}

// ---------------------------------------------------------------------------
// SIMD wrapper — noinline to prevent constant-folding in the test
// ---------------------------------------------------------------------------
constexpr int W = 16;
using FloatSimd = Simd<float, W>;

__attribute__((noinline))
FloatSimd runSimdNormSqGrad(const FloatSimd sv[19],
                            float dx2, float invDx2, float isoValue)
{
    return normSqGrad<FloatSimd>(
        sv[0],  sv[1],  sv[2],  sv[3],  sv[4],  sv[5],  sv[6],
        sv[7],  sv[8],  sv[9],  sv[10], sv[11], sv[12],
        sv[13], sv[14], sv[15], sv[16], sv[17], sv[18],
        dx2, invDx2, isoValue);
}

// ---------------------------------------------------------------------------
// Reference: scalar path — same kernel instantiated with T=float
// ---------------------------------------------------------------------------
float refNormSqGrad(const float v[19], float dx2, float invDx2, float isoValue = 0.f)
{
    return normSqGrad<float>(
        v[0],  v[1],  v[2],  v[3],  v[4],  v[5],  v[6],
        v[7],  v[8],  v[9],  v[10], v[11], v[12],
        v[13], v[14], v[15], v[16], v[17], v[18],
        dx2, invDx2, isoValue);
}

// ---------------------------------------------------------------------------
int main()
{
    const float dx = 0.1f, dx2 = dx*dx, invDx2 = 1.f/(dx*dx);

    // Storage: SoA layout — inData[n] holds W lane values for stencil position n
    float inData[19][W]{};
    float refValues[W][19];

    for (int i = 0; i < W; i++)
        for (int n = 0; n < 19; n++) {
            refValues[i][n] = std::sin(n * 0.3f + i * 0.5f);
            inData[n][i]    = refValues[i][n];
        }

    // Load into Simd — each FloatSimd holds one stencil position across all W lanes
    FloatSimd sv[19];
    for (int n = 0; n < 19; n++)
        sv[n] = FloatSimd(inData[n]);

    FloatSimd result = runSimdNormSqGrad(sv, dx2, invDx2, 0.f);

    printf("WenoNormSqGrad full 3-axis (W=%d, dx=%.2f):\n", W, dx);
    bool allOk = true;
    for (int i = 0; i < W; i++) {
        float ref = refNormSqGrad(refValues[i], dx2, invDx2, 0.f);
        float got = result[i];
        bool  ok  = std::abs(got - ref) < 1e-5f * std::abs(ref) + 1e-10f;
        printf("  lane %2d: %12.6f  ref: %12.6f  %s\n", i, got, ref, ok ? "OK" : "FAIL");
        allOk &= ok;
    }
    printf("\nOverall: %s\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
