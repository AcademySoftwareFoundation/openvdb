// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "VectorTypes.cuh"
#include <detail/ops/Ops.h>

#include <ATen/cuda/Atomic.cuh>

constexpr int NUM_THREADS = 1024;

namespace fvdb {
namespace detail {
namespace ops {

namespace {

template <typename T>
inline __device__ T
getval(const torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> coeffs, uint32_t k,
       uint32_t ci, uint32_t gi, uint32_t c) {
    return coeffs[k][ci][gi][c];
}

template <typename T>
inline __device__ T
getval(const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> coeffs, uint32_t k,
       uint32_t ci, uint32_t gi, uint32_t c) {
    return coeffs[k][gi][c];
}

template <typename T>
inline __device__ void
setval(torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> coeffs, uint32_t k,
       uint32_t ci, uint32_t gi, uint32_t c, T val) {
    coeffs[k][ci][gi][c] = val;
}

template <typename T>
inline __device__ void
setval(torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> coeffs, uint32_t k,
       uint32_t ci, uint32_t gi, uint32_t c, T val) {
    coeffs[k][gi][c] = val;
}

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
template <typename T, size_t N>
inline __device__ void
evalShFunction(const uint32_t                    degree,  // degree of SH to be evaluated
               const uint32_t                    ci,      // camera index
               const uint32_t                    gi,      // gaussian index
               const uint32_t                    c,       // render channel
               const typename Vec3Type<T>::type &viewDir, // [D]
               const torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> shCoeffs,
               T *outRenderQuantity                       // [D]
) {
    const T cSH0 = getval(shCoeffs, 0, ci, gi, c);

    T result = 0.2820947917738781f * cSH0;

    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        const T inorm =
            rsqrtf(viewDir.x * viewDir.x + viewDir.y * viewDir.y + viewDir.z * viewDir.z);
        const T x = viewDir.x * inorm;
        const T y = viewDir.y * inorm;
        const T z = viewDir.z * inorm;

        const T cSH1 = getval(shCoeffs, 1, ci, gi, c);
        const T cSH2 = getval(shCoeffs, 2, ci, gi, c);
        const T cSH3 = getval(shCoeffs, 3, ci, gi, c);

        result += 0.48860251190292f * (-y * cSH1 + z * cSH2 - x * cSH3);

        if (degree >= 2) {
            const T z2 = z * z;

            const T fTmp0B = -1.092548430592079f * z;
            const T fC1    = x * x - y * y;
            const T fS1    = 2.f * x * y;
            const T pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
            const T pSH7   = fTmp0B * x;
            const T pSH5   = fTmp0B * y;
            const T pSH8   = 0.5462742152960395f * fC1;
            const T pSH4   = 0.5462742152960395f * fS1;

            const T cSH4 = getval(shCoeffs, 4, ci, gi, c);
            const T cSH5 = getval(shCoeffs, 5, ci, gi, c);
            const T cSH6 = getval(shCoeffs, 6, ci, gi, c);
            const T cSH7 = getval(shCoeffs, 7, ci, gi, c);
            const T cSH8 = getval(shCoeffs, 8, ci, gi, c);

            result += (pSH4 * cSH4) + (pSH5 * cSH5) + (pSH6 * cSH6) + (pSH7 * cSH7) + (pSH8 * cSH8);

            if (degree >= 3) {
                const T fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                const T fTmp1B = 1.445305721320277f * z;
                const T fC2    = x * fC1 - y * fS1;
                const T fS2    = x * fS1 + y * fC1;
                const T pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
                const T pSH13  = fTmp0C * x;
                const T pSH11  = fTmp0C * y;
                const T pSH14  = fTmp1B * fC1;
                const T pSH10  = fTmp1B * fS1;
                const T pSH15  = -0.5900435899266435f * fC2;
                const T pSH9   = -0.5900435899266435f * fS2;

                const T cSH9  = getval(shCoeffs, 9, ci, gi, c);
                const T cSH10 = getval(shCoeffs, 10, ci, gi, c);
                const T cSH11 = getval(shCoeffs, 11, ci, gi, c);
                const T cSH12 = getval(shCoeffs, 12, ci, gi, c);
                const T cSH13 = getval(shCoeffs, 13, ci, gi, c);
                const T cSH14 = getval(shCoeffs, 14, ci, gi, c);
                const T cSH15 = getval(shCoeffs, 15, ci, gi, c);

                result += (pSH9 * cSH9) + (pSH10 * cSH10) + (pSH11 * cSH11) + (pSH12 * cSH12) +
                          (pSH13 * cSH13) + (pSH14 * cSH14) + (pSH15 * cSH15);

                if (degree >= 4) {
                    const T fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    const T fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    const T fTmp2B = -1.770130769779931f * z;
                    const T fC3    = x * fC2 - y * fS2;
                    const T fS3    = x * fS2 + y * fC2;
                    const T pSH20  = (1.984313483298443f * z * pSH12 - 1.006230589874905f * pSH6);
                    const T pSH21  = fTmp0D * x;
                    const T pSH19  = fTmp0D * y;
                    const T pSH22  = fTmp1C * fC1;
                    const T pSH18  = fTmp1C * fS1;
                    const T pSH23  = fTmp2B * fC2;
                    const T pSH17  = fTmp2B * fS2;
                    const T pSH24  = 0.6258357354491763f * fC3;
                    const T pSH16  = 0.6258357354491763f * fS3;

                    const T cSH16 = getval(shCoeffs, 16, ci, gi, c);
                    const T cSH17 = getval(shCoeffs, 17, ci, gi, c);
                    const T cSH18 = getval(shCoeffs, 18, ci, gi, c);
                    const T cSH19 = getval(shCoeffs, 19, ci, gi, c);
                    const T cSH20 = getval(shCoeffs, 20, ci, gi, c);
                    const T cSH21 = getval(shCoeffs, 21, ci, gi, c);
                    const T cSH22 = getval(shCoeffs, 22, ci, gi, c);
                    const T cSH23 = getval(shCoeffs, 23, ci, gi, c);
                    const T cSH24 = getval(shCoeffs, 24, ci, gi, c);

                    result += (pSH16 * cSH16) + (pSH17 * cSH17) + (pSH18 * cSH18) +
                              (pSH19 * cSH19) + (pSH20 * cSH20) + (pSH21 * cSH21) +
                              (pSH22 * cSH22) + (pSH23 * cSH23) + (pSH24 * cSH24);
                }
            }
        }
    }

    outRenderQuantity[c] = result + 0.5f;
}

// We repeat this code everywhere in evalShFunctionVJP to compute the gradient of the
// direction and write it out, so pull this into a function.
template <typename T>
__device__ inline void
writeDLossDViewDir(T x, T y, T z, T vX, T vY, T vZ, T inorm,
                   typename Vec3Type<T>::type *dLossDViewDir) {
    using vec3t                     = typename Vec3Type<T>::type;
    const T dLossDViewDirDotViewDir = x * vX + y * vY + z * vZ;

    dLossDViewDir->x = (vX - dLossDViewDirDotViewDir * x) * inorm;
    dLossDViewDir->y = (vY - dLossDViewDirDotViewDir * y) * inorm;
    dLossDViewDir->z = (vZ - dLossDViewDirDotViewDir * z) * inorm;
}

template <typename T, size_t N>
inline __device__ void
evalShFunctionVJP(const uint32_t                    degree, // degree of SH to be evaluated
                  const uint32_t                    ci,     // camera index
                  const uint32_t                    gi,     // gaussian index
                  const uint32_t                    c,      // render channel
                  const typename Vec3Type<T>::type &dir,    // [3]
                  const torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> coeffs,
                  const T *dLossDRenderQuantities,          // [D]
                  torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> dLossDShCoeffs,
                  typename Vec3Type<T>::type *dLossDViewDir // [3] optional
) {
    T dLossDRenderQuantitiesLocal = dLossDRenderQuantities[c];

    setval(dLossDShCoeffs, 0, ci, gi, c, 0.2820947917738781f * dLossDRenderQuantitiesLocal);

    if (degree < 1) {
        return;
    }
    const T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const T x     = dir.x * inorm;
    const T y     = dir.y * inorm;
    const T z     = dir.z * inorm;
    T       vX = 0.f, vY = 0.f, vZ = 0.f;

    setval(dLossDShCoeffs, 1, ci, gi, c, -0.48860251190292f * y * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 2, ci, gi, c, 0.48860251190292f * z * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 3, ci, gi, c, -0.48860251190292f * x * dLossDRenderQuantitiesLocal);

    if (dLossDViewDir != nullptr) {
        vX += -0.48860251190292f * getval(coeffs, 3, ci, gi, c) * dLossDRenderQuantitiesLocal;
        vY += -0.48860251190292f * getval(coeffs, 1, ci, gi, c) * dLossDRenderQuantitiesLocal;
        vZ += 0.48860251190292f * getval(coeffs, 2, ci, gi, c) * dLossDRenderQuantitiesLocal;
    }
    if (degree < 2) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T z2     = z * z;
    const T fTmp0B = -1.092548430592079f * z;
    const T fC1    = x * x - y * y;
    const T fS1    = 2.f * x * y;
    const T pSH6   = (0.9461746957575601f * z2 - 0.3153915652525201f);
    const T pSH7   = fTmp0B * x;
    const T pSH5   = fTmp0B * y;
    const T pSH8   = 0.5462742152960395f * fC1;
    const T pSH4   = 0.5462742152960395f * fS1;
    setval(dLossDShCoeffs, 4, ci, gi, c, pSH4 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 5, ci, gi, c, pSH5 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 6, ci, gi, c, pSH6 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 7, ci, gi, c, pSH7 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 8, ci, gi, c, pSH8 * dLossDRenderQuantitiesLocal);

    T fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y,
        pSH4_x, pSH4_y;
    if (dLossDViewDir != nullptr) {
        fTmp0B_z = -1.092548430592079f;
        fC1_x    = 2.f * x;
        fC1_y    = -2.f * y;
        fS1_x    = 2.f * y;
        fS1_y    = 2.f * x;
        pSH6_z   = 2.f * 0.9461746957575601f * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = 0.5462742152960395f * fC1_x;
        pSH8_y   = 0.5462742152960395f * fC1_y;
        pSH4_x   = 0.5462742152960395f * fS1_x;
        pSH4_y   = 0.5462742152960395f * fS1_y;

        vX += dLossDRenderQuantitiesLocal *
              (pSH4_x * getval(coeffs, 4, ci, gi, c) + pSH8_x * getval(coeffs, 8, ci, gi, c) +
               pSH7_x * getval(coeffs, 7, ci, gi, c));
        vY += dLossDRenderQuantitiesLocal *
              (pSH4_y * getval(coeffs, 4, ci, gi, c) + pSH8_y * getval(coeffs, 8, ci, gi, c) +
               pSH5_y * getval(coeffs, 5, ci, gi, c));
        vZ += dLossDRenderQuantitiesLocal *
              (pSH6_z * getval(coeffs, 6, ci, gi, c) + pSH7_z * getval(coeffs, 7, ci, gi, c) +
               pSH5_z * getval(coeffs, 5, ci, gi, c));
    }

    if (degree < 3) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    const T fTmp1B = 1.445305721320277f * z;
    const T fC2    = x * fC1 - y * fS1;
    const T fS2    = x * fS1 + y * fC1;
    const T pSH12  = z * (1.865881662950577f * z2 - 1.119528997770346f);
    const T pSH13  = fTmp0C * x;
    const T pSH11  = fTmp0C * y;
    const T pSH14  = fTmp1B * fC1;
    const T pSH10  = fTmp1B * fS1;
    const T pSH15  = -0.5900435899266435f * fC2;
    const T pSH9   = -0.5900435899266435f * fS2;

    setval(dLossDShCoeffs, 9, ci, gi, c, pSH9 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 10, ci, gi, c, pSH10 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 11, ci, gi, c, pSH11 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 12, ci, gi, c, pSH12 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 13, ci, gi, c, pSH13 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 14, ci, gi, c, pSH14 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 15, ci, gi, c, pSH15 * dLossDRenderQuantitiesLocal);

    T fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z,
        pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (dLossDViewDir != nullptr) {
        fTmp0C_z = -2.285228997322329f * 2.f * z;
        fTmp1B_z = 1.445305721320277f;
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
        pSH13_x  = fTmp0C;
        pSH13_z  = fTmp0C_z * x;
        pSH11_y  = fTmp0C;
        pSH11_z  = fTmp0C_z * y;
        pSH14_x  = fTmp1B * fC1_x;
        pSH14_y  = fTmp1B * fC1_y;
        pSH14_z  = fTmp1B_z * fC1;
        pSH10_x  = fTmp1B * fS1_x;
        pSH10_y  = fTmp1B * fS1_y;
        pSH10_z  = fTmp1B_z * fS1;
        pSH15_x  = -0.5900435899266435f * fC2_x;
        pSH15_y  = -0.5900435899266435f * fC2_y;
        pSH9_x   = -0.5900435899266435f * fS2_x;
        pSH9_y   = -0.5900435899266435f * fS2_y;

        const T cSH9  = getval(coeffs, 9, ci, gi, c);
        const T cSH10 = getval(coeffs, 10, ci, gi, c);
        const T cSH11 = getval(coeffs, 11, ci, gi, c);
        const T cSH12 = getval(coeffs, 12, ci, gi, c);
        const T cSH13 = getval(coeffs, 13, ci, gi, c);
        const T cSH14 = getval(coeffs, 14, ci, gi, c);
        const T cSH15 = getval(coeffs, 15, ci, gi, c);

        vX += dLossDRenderQuantitiesLocal * (pSH9_x * cSH9 + pSH15_x * cSH15 + pSH10_x * cSH10 +
                                             pSH14_x * cSH14 + pSH13_x * cSH13);

        vY += dLossDRenderQuantitiesLocal * (pSH9_y * cSH9 + pSH15_y * cSH15 + pSH10_y * cSH10 +
                                             pSH14_y * cSH14 + pSH11_y * cSH11);

        vZ += dLossDRenderQuantitiesLocal * (pSH12_z * cSH12 + pSH13_z * cSH13 + pSH11_z * cSH11 +
                                             pSH14_z * cSH14 + pSH10_z * cSH10);
    }

    if (degree < 4) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    const T fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    const T fTmp2B = -1.770130769779931f * z;
    const T fC3    = x * fC2 - y * fS2;
    const T fS3    = x * fS2 + y * fC2;
    const T pSH20  = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    const T pSH21  = fTmp0D * x;
    const T pSH19  = fTmp0D * y;
    const T pSH22  = fTmp1C * fC1;
    const T pSH18  = fTmp1C * fS1;
    const T pSH23  = fTmp2B * fC2;
    const T pSH17  = fTmp2B * fS2;
    const T pSH24  = 0.6258357354491763f * fC3;
    const T pSH16  = 0.6258357354491763f * fS3;

    setval(dLossDShCoeffs, 16, ci, gi, c, pSH16 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 17, ci, gi, c, pSH17 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 18, ci, gi, c, pSH18 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 19, ci, gi, c, pSH19 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 20, ci, gi, c, pSH20 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 21, ci, gi, c, pSH21 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 22, ci, gi, c, pSH22 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 23, ci, gi, c, pSH23 * dLossDRenderQuantitiesLocal);
    setval(dLossDShCoeffs, 24, ci, gi, c, pSH24 * dLossDRenderQuantitiesLocal);

    T fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y,
        pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z,
        pSH17_x, pSH17_y, pSH17_z, pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (dLossDViewDir != nullptr) {
        fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
        fTmp1C_z = 2.f * 3.31161143515146f * z;
        fTmp2B_z = -1.770130769779931f;
        fC3_x    = fC2 + x * fC2_x - y * fS2_x;
        fC3_y    = x * fC2_y - fS2 - y * fS2_y;
        fS3_x    = fS2 + y * fC2_x + x * fS2_x;
        fS3_y    = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z  = 1.984313483298443f * (pSH12 + z * pSH12_z) + -1.006230589874905f * pSH6_z;
        pSH21_x  = fTmp0D;
        pSH21_z  = fTmp0D_z * x;
        pSH19_y  = fTmp0D;
        pSH19_z  = fTmp0D_z * y;
        pSH22_x  = fTmp1C * fC1_x;
        pSH22_y  = fTmp1C * fC1_y;
        pSH22_z  = fTmp1C_z * fC1;
        pSH18_x  = fTmp1C * fS1_x;
        pSH18_y  = fTmp1C * fS1_y;
        pSH18_z  = fTmp1C_z * fS1;
        pSH23_x  = fTmp2B * fC2_x;
        pSH23_y  = fTmp2B * fC2_y;
        pSH23_z  = fTmp2B_z * fC2;
        pSH17_x  = fTmp2B * fS2_x;
        pSH17_y  = fTmp2B * fS2_y;
        pSH17_z  = fTmp2B_z * fS2;
        pSH24_x  = 0.6258357354491763f * fC3_x;
        pSH24_y  = 0.6258357354491763f * fC3_y;
        pSH16_x  = 0.6258357354491763f * fS3_x;
        pSH16_y  = 0.6258357354491763f * fS3_y;

        const T cSH16 = getval(coeffs, 16, ci, gi, c);
        const T cSH17 = getval(coeffs, 17, ci, gi, c);
        const T cSH18 = getval(coeffs, 18, ci, gi, c);
        const T cSH19 = getval(coeffs, 19, ci, gi, c);
        const T cSH20 = getval(coeffs, 20, ci, gi, c);
        const T cSH21 = getval(coeffs, 21, ci, gi, c);
        const T cSH22 = getval(coeffs, 22, ci, gi, c);
        const T cSH23 = getval(coeffs, 23, ci, gi, c);
        const T cSH24 = getval(coeffs, 24, ci, gi, c);

        vX += dLossDRenderQuantitiesLocal *
              (pSH16_x * cSH16 + pSH24_x * cSH24 + pSH17_x * cSH17 + pSH23_x * cSH23 +
               pSH18_x * cSH18 + pSH22_x * cSH22 + pSH21_x * cSH21);
        vY += dLossDRenderQuantitiesLocal *
              (pSH16_y * cSH16 + pSH24_y * cSH24 + pSH17_y * cSH17 + pSH23_y * cSH23 +
               pSH18_y * cSH18 + pSH22_y * cSH22 + pSH19_y * cSH19);
        vZ += dLossDRenderQuantitiesLocal *
              (pSH20_z * cSH20 + pSH21_z * cSH21 + pSH19_z * cSH19 + pSH22_z * cSH22 +
               pSH18_z * cSH18 + pSH23_z * cSH23 + pSH17_z * cSH17);

        writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
    }
}
} // namespace

// Evalute Spherical Harmonic functions at the given directions, assuming a uniform minibatch
// of C cameras, each with N gaussians, and K SH coefficients per gaussian.
template <typename T>
__global__ void
computeShForward(
    const uint32_t C, const uint32_t N, const uint32_t K, const uint32_t D,
    const uint32_t                                                      shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> viewDirs, // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> shCoeffs, // [K, C, N, D]
    const int *__restrict__ radii,                                                // [C, N]
    T *__restrict__ outRenderQuantities                                           // [C, N, D]
) {
    // parallelize over C * N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= C * N * D) {
        return;
    }

    const uint32_t eid = idx / D; // cidx * N + gidx
    const uint32_t cid = eid / N; // camera index
    const uint32_t gid = eid % N; // gaussian index
    const uint32_t c   = idx % D; // render channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    using vec3t             = typename Vec3Type<T>::type;
    const bool  hasViewDirs = viewDirs.size(0) > 0;
    const vec3t dir         = hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[cid][gid].data())
                                          : vec3t{ 0.f, 0.f, 0.f };
    T          *outRenderQuantityPtr = outRenderQuantities + eid * D;
    evalShFunction(shDegreeToUse, cid, gid, c, dir, shCoeffs, outRenderQuantityPtr);
}

// Evalute Spherical Harmonic functions at the given directions, assuming a N gaussians with K SH
// coefficients per gaussian.
template <typename T>
__global__ void
computeShForward(
    const uint32_t N, const uint32_t K, const uint32_t D, const uint32_t shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> viewDirs, // [N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> shCoeffs, // [K, N, D]
    const int *__restrict__ radii,                                                // [N]
    T *__restrict__ outRenderQuantities                                           // [N, D]
) {
    // parallelize over N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // gidx * D + kidx
    if (idx >= N * D) {
        return;
    }

    const uint32_t gid = idx / D; // gidx
    const uint32_t c   = idx % D; // render channel
    if (radii != nullptr && radii[gid] <= 0) {
        return;
    }

    using vec3t             = typename Vec3Type<T>::type;
    const bool  hasViewDirs = viewDirs.size(0) > 0;
    const vec3t viewDir =
        hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    T *outRenderQuantityPtr = outRenderQuantities + gid * D;
    evalShFunction(shDegreeToUse, 0, gid, c, viewDir, shCoeffs, outRenderQuantityPtr);
}

template <typename T>
__global__ void
computeShBackward(
    const uint32_t C, const uint32_t N, const uint32_t K, const uint32_t D,
    const uint32_t                                                      shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> viewDirs,    // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> shCoeffs,    // [K, C, N, D]
    const int *__restrict__ radii,                                                   // [C, N]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits>
        dLossDRenderQuantities,                                                      // [C, N, D]
    torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> outDLossDShCoeffs, // [K, C, N, D]
    T *__restrict__ outDLossDViewDirs // [C, N, 3] optiondl
) {
    // parallelize over C * N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= C * N * D) {
        return;
    }

    const uint32_t eid = idx / D; // cidx * N + gidx
    const uint32_t cid = eid / N; // camera index
    const uint32_t gid = eid % N; // gaussian index
    const uint32_t c   = idx % D; // render channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    using vec3t             = typename Vec3Type<T>::type;
    const bool  hasViewDirs = viewDirs.size(0) > 0;
    const vec3t viewDir     = hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[cid][gid].data())
                                          : vec3t{ T(0), T(0), T(0) };
    const T    *dLossDRenderQuantityPtr = dLossDRenderQuantities[cid][gid].data();

    vec3t  v_dir{ T(0), T(0), T(0) };
    vec3t *outDLossDViewDirPtr = outDLossDViewDirs == nullptr ? nullptr : &v_dir;

    evalShFunctionVJP(shDegreeToUse, cid, gid, c, viewDir, shCoeffs, dLossDRenderQuantityPtr,
                      outDLossDShCoeffs, outDLossDViewDirPtr);
    if (outDLossDViewDirs != nullptr) {
        gpuAtomicAdd(outDLossDViewDirs + eid * 3, v_dir.x);
        gpuAtomicAdd(outDLossDViewDirs + eid * 3 + 1, v_dir.y);
        gpuAtomicAdd(outDLossDViewDirs + eid * 3 + 2, v_dir.z);
    }
}

template <typename T>
__global__ void
computeShBackward_packed(
    const uint32_t N, const uint32_t K, const uint32_t D, const uint32_t shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> viewDirs, // [N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> shCoeffs, // [K, N, D]
    const int *__restrict__ radii,                                                // [N]
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits>
                                                                  dLossDRenderQuantities, // [N, D]
    torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> outDLossDShCoeffs, // [K, N, D]
    T *__restrict__ outDLossDViewDirs // [N, 3] optiondl
) {
    // parallelize over N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // gidx * D + c
    if (idx >= N * D) {
        return;
    }

    const uint32_t gid = idx / D; // gidx
    const uint32_t c   = idx % D; // render channel
    if (radii != nullptr && radii[gid] <= 0) {
        return;
    }

    using vec3t             = typename Vec3Type<T>::type;
    const bool  hasViewDirs = viewDirs.size(0) > 0;
    const vec3t viewDir =
        hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    const T *dLossDRenderQuantityPtr = dLossDRenderQuantities[gid].data();

    vec3t  v_dir{ 0.f, 0.f, 0.f };
    vec3t *outDLossDViewDirsPtr = outDLossDViewDirs == nullptr ? nullptr : &v_dir;

    evalShFunctionVJP(shDegreeToUse, static_cast<uint32_t>(0), gid, c, viewDir, shCoeffs,
                      dLossDRenderQuantityPtr, outDLossDShCoeffs, outDLossDViewDirsPtr);
    if (outDLossDViewDirs != nullptr) {
        gpuAtomicAdd(outDLossDViewDirs + gid * 3, v_dir.x);
        gpuAtomicAdd(outDLossDViewDirs + gid * 3 + 1, v_dir.y);
        gpuAtomicAdd(outDLossDViewDirs + gid * 3 + 2, v_dir.z);
    }
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCUDA>(
    const int            shDegreeToUse,
    const torch::Tensor &dirs,     // [N, 3] or [C, N, D] or empty for degree 0
    const torch::Tensor &shCoeffs, // [N, K, D] or [C, N, K, D]
    const torch::Tensor &radii     // [N] or [C, N]
) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(shCoeffs));

    const bool isPacked = shCoeffs.dim() == 3;

    TORCH_CHECK_VALUE(shCoeffs.is_cuda(), "shCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");

    if (isPacked) {
        TORCH_CHECK_VALUE(shCoeffs.dim() == 3, "shCoeffs must have shape [K, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have shape [N]");
        TORCH_CHECK_VALUE(shCoeffs.size(1) == radii.size(0),
                          "shCoeffs must have shape [K, N, D] and radii must have shape [N]");
    } else {
        TORCH_CHECK_VALUE(shCoeffs.dim() == 4, "shCoeffs must have shape [K, C, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have shape [C, N]");
        TORCH_CHECK_VALUE(shCoeffs.size(1) == radii.size(0),
                          "shCoeffs must have shape [K. C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(shCoeffs.size(2) == radii.size(1),
                          "shCoeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
    }

    const uint32_t K           = shCoeffs.size(0);
    const uint32_t N           = isPacked ? shCoeffs.size(1) : shCoeffs.size(2);
    const uint32_t C           = isPacked ? 1 : shCoeffs.size(1);
    const uint32_t D           = shCoeffs.size(-1);
    const uint32_t TOTAL_ELEMS = C * N * D;
    const uint32_t NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (K > 0 && shDegreeToUse > 0) {
        if (isPacked) {
            TORCH_CHECK_VALUE(dirs.dim() == 2,
                              "shCoeffs must have shape [K, N, D] and dirs must have shape [N, D]");
            TORCH_CHECK_VALUE(shCoeffs.size(1) == dirs.size(0),
                              "shCoeffs must have shape [K, N, D] and dirs must have shape [N, D]");
        } else {
            TORCH_CHECK_VALUE(
                dirs.dim() == 3,
                "shCoeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                shCoeffs.size(1) == dirs.size(0),
                "shCoeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                shCoeffs.size(2) == dirs.size(1),
                "shCoeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
        }
        TORCH_CHECK_VALUE(dirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(dirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(shCoeffs.device().index());

    // TODO (Francis): Might need to do zeros_like here
    const std::vector<int64_t> out_size =
        isPacked ? std::vector<int64_t>{ N, D } : std::vector<int64_t>{ C, N, D };
    torch::Tensor renderQuantities = torch::empty(out_size, shCoeffs.options()); // [..., D]

    using scalar_t = float;

    // parallelize over N * D
    if (!N) {
        return renderQuantities; // [..., D]
    }
    if (isPacked) {
        computeShForward<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            N, K, D, shDegreeToUse, dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            shCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(), renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        computeShForward<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, K, D, shDegreeToUse,
            dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            shCoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(), renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return renderQuantities; // [..., D]
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCPU>(const int            shDegreeToUse,
                                               const torch::Tensor &dirs,     // [N, 3]
                                               const torch::Tensor &shCoeffs, // [N, K, 3]
                                               const torch::Tensor &radii     // [N]
) {
    TORCH_CHECK(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCUDA>(
    const int            shDegreeToUse,
    const torch::Tensor &dirs,                   // [C, N, 3] or [N, 3] or empty for degree 0
    const torch::Tensor &shCoeffs,               // [C, N, K, 3] or [N, K, 3]
    const torch::Tensor &dLossDRenderQuantities, // [C, N, 3] or [N, 3]
    const torch::Tensor &radii,                  // [C, N] or [N]
    const bool           computeDLossDViewDirs) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(shCoeffs));

    const bool isPacked = shCoeffs.dim() == 3;

    TORCH_CHECK_VALUE(shCoeffs.is_cuda(), "shCoeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    TORCH_CHECK_VALUE(dLossDRenderQuantities.is_cuda(),
                      "dLossDRenderQuantities must be a CUDA tensor");

    if (isPacked) {
        TORCH_CHECK_VALUE(shCoeffs.dim() == 3, "shCoeffs must have shape [K, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have shape [N]");
        TORCH_CHECK_VALUE(shCoeffs.size(1) == radii.size(0),
                          "shCoeffs must have shape [K, N, D] and radii must have shape [N]");
        TORCH_CHECK_VALUE(
            shCoeffs.size(2) == dLossDRenderQuantities.size(0),
            "shCoeffs and dLossDRenderQuantities must have the same number of elements in dimension 0");
    } else {
        TORCH_CHECK_VALUE(shCoeffs.dim() == 4, "shCoeffs must have shape [K, C, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have shape [C, N]");
        TORCH_CHECK_VALUE(shCoeffs.size(1) == radii.size(0),
                          "shCoeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(shCoeffs.size(2) == radii.size(1),
                          "shCoeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            shCoeffs.size(1) == dLossDRenderQuantities.size(0),
            "shCoeffs and dLossDRenderQuantities must have the same number of elements in dimension 0");
        TORCH_CHECK_VALUE(
            shCoeffs.size(2) == dLossDRenderQuantities.size(1),
            "shCoeffs and dLossDRenderQuantities must have the same number of elements in dimension 1");
    }
    TORCH_CHECK_VALUE(
        shCoeffs.size(-1) == dLossDRenderQuantities.size(-1),
        "shCoeffs and dLossDRenderQuantities must have the same number of elements in the last dimension");

    const uint32_t K           = shCoeffs.size(0);
    const uint32_t N           = isPacked ? shCoeffs.size(1) : shCoeffs.size(2);
    const uint32_t C           = isPacked ? 1 : shCoeffs.size(1);
    const uint32_t D           = shCoeffs.size(-1);
    const uint32_t TOTAL_ELEMS = C * N * D;
    const uint32_t NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (K > 0 && shDegreeToUse > 0) {
        if (isPacked) {
            TORCH_CHECK_VALUE(dirs.dim() == 2,
                              "shCoeffs must have shape [K, N, D] and dirs must have shape [N, D]");
            TORCH_CHECK_VALUE(shCoeffs.size(1) == dirs.size(0),
                              "shCoeffs must have shape [K, N, D] and dirs must have shape [N, D]");
        } else {
            TORCH_CHECK_VALUE(
                dirs.dim() == 3,
                "shCoeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                shCoeffs.size(1) == dirs.size(0),
                "shCoeffs must have shape [K, C, N, 3] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                shCoeffs.size(2) == dirs.size(1),
                "shCoeffs must have shape [K, C, N, 3] and dirs must have shape [C, N, D]");
        }
        TORCH_CHECK_VALUE(dirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(dirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(shCoeffs.device().index());

    torch::Tensor dLossDShCoeffs = torch::zeros_like(shCoeffs);
    torch::Tensor dLossDViewDirs;
    if (computeDLossDViewDirs) {
        dLossDViewDirs = torch::zeros_like(dirs);
    }
    if (!N) {
        std::make_tuple(dLossDShCoeffs, dLossDViewDirs); // [..., K, 3], [..., 3]
    }

    using scalar_t = float;
    if (isPacked) {
        computeShBackward_packed<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            N, K, D, shDegreeToUse, dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            shCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(),
            dLossDRenderQuantities.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            dLossDShCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            computeDLossDViewDirs ? dLossDViewDirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        computeShBackward<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, K, D, shDegreeToUse,
            dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            shCoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(),
            dLossDRenderQuantities.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLossDShCoeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            computeDLossDViewDirs ? dLossDViewDirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(dLossDShCoeffs, dLossDViewDirs); // [..., K, 3], [..., 3]
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCPU>(const int            shDegreeToUse,
                                                const torch::Tensor &dirs,     // [N, 3]
                                                const torch::Tensor &shCoeffs, // [N, K, 3]
                                                const torch::Tensor &dLossDRenderQuantities,
                                                const torch::Tensor &radii,    // [N]
                                                const bool           computeDLossDViewDirs) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
