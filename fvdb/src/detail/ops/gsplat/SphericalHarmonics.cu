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
eval_sh_function(const uint32_t                    degree, // degree of SH to be evaluated
                 const uint32_t                    ci,     // camera index
                 const uint32_t                    gi,     // gaussian index
                 const uint32_t                    c,      // color channel
                 const typename Vec3Type<T>::type &dir,    // [D]
                 const torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> coeffs,
                 T                                                                  *colors // [D]
) {
    const T cSH0 = getval(coeffs, 0, ci, gi, c);

    T result = 0.2820947917738781f * cSH0;

    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        const T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        const T x     = dir.x * inorm;
        const T y     = dir.y * inorm;
        const T z     = dir.z * inorm;

        const T cSH1 = getval(coeffs, 1, ci, gi, c);
        const T cSH2 = getval(coeffs, 2, ci, gi, c);
        const T cSH3 = getval(coeffs, 3, ci, gi, c);

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

            const T cSH4 = getval(coeffs, 4, ci, gi, c);
            const T cSH5 = getval(coeffs, 5, ci, gi, c);
            const T cSH6 = getval(coeffs, 6, ci, gi, c);
            const T cSH7 = getval(coeffs, 7, ci, gi, c);
            const T cSH8 = getval(coeffs, 8, ci, gi, c);

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

                const T cSH9  = getval(coeffs, 9, ci, gi, c);
                const T cSH10 = getval(coeffs, 10, ci, gi, c);
                const T cSH11 = getval(coeffs, 11, ci, gi, c);
                const T cSH12 = getval(coeffs, 12, ci, gi, c);
                const T cSH13 = getval(coeffs, 13, ci, gi, c);
                const T cSH14 = getval(coeffs, 14, ci, gi, c);
                const T cSH15 = getval(coeffs, 15, ci, gi, c);

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

                    const T cSH16 = getval(coeffs, 16, ci, gi, c);
                    const T cSH17 = getval(coeffs, 17, ci, gi, c);
                    const T cSH18 = getval(coeffs, 18, ci, gi, c);
                    const T cSH19 = getval(coeffs, 19, ci, gi, c);
                    const T cSH20 = getval(coeffs, 20, ci, gi, c);
                    const T cSH21 = getval(coeffs, 21, ci, gi, c);
                    const T cSH22 = getval(coeffs, 22, ci, gi, c);
                    const T cSH23 = getval(coeffs, 23, ci, gi, c);
                    const T cSH24 = getval(coeffs, 24, ci, gi, c);

                    result += (pSH16 * cSH16) + (pSH17 * cSH17) + (pSH18 * cSH18) +
                              (pSH19 * cSH19) + (pSH20 * cSH20) + (pSH21 * cSH21) +
                              (pSH22 * cSH22) + (pSH23 * cSH23) + (pSH24 * cSH24);
                }
            }
        }
    }

    colors[c] = result + 0.5f;
}

// We repeat this code everywhere in eval_sh_function_vjp to compute the gradient of the
// direction and write it out, so pull this into a function.
template <typename T>
__device__ inline void
write_v_dir(T x, T y, T z, T v_x, T v_y, T v_z, T inorm, typename Vec3Type<T>::type *v_dir) {
    using vec3t               = typename Vec3Type<T>::type;
    const T v_dir_n_dot_dir_n = x * v_x + y * v_y + z * v_z;

    v_dir->x = (v_x - v_dir_n_dot_dir_n * x) * inorm;
    v_dir->y = (v_y - v_dir_n_dot_dir_n * y) * inorm;
    v_dir->z = (v_z - v_dir_n_dot_dir_n * z) * inorm;
}

template <typename T, size_t N>
inline __device__ void
eval_sh_function_vjp(const uint32_t                    degree, // degree of SH to be evaluated
                     const uint32_t                    ci,     // camera index
                     const uint32_t                    gi,     // gaussian index
                     const uint32_t                    c,      // color channel
                     const typename Vec3Type<T>::type &dir,    // [3]
                     const torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> coeffs,
                     const T                                                      *v_colors, // [D]
                     torch::PackedTensorAccessor32<T, N, torch::RestrictPtrTraits> v_coeffs,
                     typename Vec3Type<T>::type *v_dir // [3] optional
) {
    T v_colors_local = v_colors[c];

    setval(v_coeffs, 0, ci, gi, c, 0.2820947917738781f * v_colors_local);

    if (degree < 1) {
        return;
    }
    const T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const T x     = dir.x * inorm;
    const T y     = dir.y * inorm;
    const T z     = dir.z * inorm;
    T       v_x = 0.f, v_y = 0.f, v_z = 0.f;

    setval(v_coeffs, 1, ci, gi, c, -0.48860251190292f * y * v_colors_local);
    setval(v_coeffs, 2, ci, gi, c, 0.48860251190292f * z * v_colors_local);
    setval(v_coeffs, 3, ci, gi, c, -0.48860251190292f * x * v_colors_local);

    if (v_dir != nullptr) {
        v_x += -0.48860251190292f * getval(coeffs, 3, ci, gi, c) * v_colors_local;
        v_y += -0.48860251190292f * getval(coeffs, 1, ci, gi, c) * v_colors_local;
        v_z += 0.48860251190292f * getval(coeffs, 2, ci, gi, c) * v_colors_local;
    }
    if (degree < 2) {
        if (v_dir != nullptr) {
            write_v_dir(x, y, z, v_x, v_y, v_z, inorm, v_dir);
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
    setval(v_coeffs, 4, ci, gi, c, pSH4 * v_colors_local);
    setval(v_coeffs, 5, ci, gi, c, pSH5 * v_colors_local);
    setval(v_coeffs, 6, ci, gi, c, pSH6 * v_colors_local);
    setval(v_coeffs, 7, ci, gi, c, pSH7 * v_colors_local);
    setval(v_coeffs, 8, ci, gi, c, pSH8 * v_colors_local);

    T fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y,
        pSH4_x, pSH4_y;
    if (v_dir != nullptr) {
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

        v_x += v_colors_local *
               (pSH4_x * getval(coeffs, 4, ci, gi, c) + pSH8_x * getval(coeffs, 8, ci, gi, c) +
                pSH7_x * getval(coeffs, 7, ci, gi, c));
        v_y += v_colors_local *
               (pSH4_y * getval(coeffs, 4, ci, gi, c) + pSH8_y * getval(coeffs, 8, ci, gi, c) +
                pSH5_y * getval(coeffs, 5, ci, gi, c));
        v_z += v_colors_local *
               (pSH6_z * getval(coeffs, 6, ci, gi, c) + pSH7_z * getval(coeffs, 7, ci, gi, c) +
                pSH5_z * getval(coeffs, 5, ci, gi, c));
    }

    if (degree < 3) {
        if (v_dir != nullptr) {
            write_v_dir(x, y, z, v_x, v_y, v_z, inorm, v_dir);
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

    setval(v_coeffs, 9, ci, gi, c, pSH9 * v_colors_local);
    setval(v_coeffs, 10, ci, gi, c, pSH10 * v_colors_local);
    setval(v_coeffs, 11, ci, gi, c, pSH11 * v_colors_local);
    setval(v_coeffs, 12, ci, gi, c, pSH12 * v_colors_local);
    setval(v_coeffs, 13, ci, gi, c, pSH13 * v_colors_local);
    setval(v_coeffs, 14, ci, gi, c, pSH14 * v_colors_local);
    setval(v_coeffs, 15, ci, gi, c, pSH15 * v_colors_local);

    T fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z,
        pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (v_dir != nullptr) {
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

        v_x += v_colors_local * (pSH9_x * cSH9 + pSH15_x * cSH15 + pSH10_x * cSH10 +
                                 pSH14_x * cSH14 + pSH13_x * cSH13);

        v_y += v_colors_local * (pSH9_y * cSH9 + pSH15_y * cSH15 + pSH10_y * cSH10 +
                                 pSH14_y * cSH14 + pSH11_y * cSH11);

        v_z += v_colors_local * (pSH12_z * cSH12 + pSH13_z * cSH13 + pSH11_z * cSH11 +
                                 pSH14_z * cSH14 + pSH10_z * cSH10);
    }

    if (degree < 4) {
        if (v_dir != nullptr) {
            write_v_dir(x, y, z, v_x, v_y, v_z, inorm, v_dir);
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

    setval(v_coeffs, 16, ci, gi, c, pSH16 * v_colors_local);
    setval(v_coeffs, 17, ci, gi, c, pSH17 * v_colors_local);
    setval(v_coeffs, 18, ci, gi, c, pSH18 * v_colors_local);
    setval(v_coeffs, 19, ci, gi, c, pSH19 * v_colors_local);
    setval(v_coeffs, 20, ci, gi, c, pSH20 * v_colors_local);
    setval(v_coeffs, 21, ci, gi, c, pSH21 * v_colors_local);
    setval(v_coeffs, 22, ci, gi, c, pSH22 * v_colors_local);
    setval(v_coeffs, 23, ci, gi, c, pSH23 * v_colors_local);
    setval(v_coeffs, 24, ci, gi, c, pSH24 * v_colors_local);

    T fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y,
        pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z,
        pSH17_x, pSH17_y, pSH17_z, pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (v_dir != nullptr) {
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

        v_x += v_colors_local *
               (pSH16_x * cSH16 + pSH24_x * cSH24 + pSH17_x * cSH17 + pSH23_x * cSH23 +
                pSH18_x * cSH18 + pSH22_x * cSH22 + pSH21_x * cSH21);
        v_y += v_colors_local *
               (pSH16_y * cSH16 + pSH24_y * cSH24 + pSH17_y * cSH17 + pSH23_y * cSH23 +
                pSH18_y * cSH18 + pSH22_y * cSH22 + pSH19_y * cSH19);
        v_z += v_colors_local *
               (pSH20_z * cSH20 + pSH21_z * cSH21 + pSH19_z * cSH19 + pSH22_z * cSH22 +
                pSH18_z * cSH18 + pSH23_z * cSH23 + pSH17_z * cSH17);

        write_v_dir(x, y, z, v_x, v_y, v_z, inorm, v_dir);
    }
}
} // namespace

// Evalute Spherical Harmonic functions at the given directions, assuming a uniform minibatch
// of C cameras, each with N gaussians, and K SH coefficients per gaussian.
template <typename T>
__global__ void
compute_sh_fwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t K, const uint32_t D,
    const uint32_t                                                      degree_to_use,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> dirs,   // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> coeffs, // [K, C, N, D]
    const int *__restrict__ radii,                                              // [C, N]
    T *__restrict__ out_colors                                                  // [C, N, D]
) {
    // parallelize over C * N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= C * N * D) {
        return;
    }

    const uint32_t eid = idx / D; // cidx * N + gidx
    const uint32_t cid = eid / N; // camera index
    const uint32_t gid = eid % N; // gaussian index
    const uint32_t c   = idx % D; // color channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    using vec3t          = typename Vec3Type<T>::type;
    const bool  has_dirs = dirs.size(0) > 0;
    const vec3t dir =
        has_dirs ? *reinterpret_cast<vec3t *>(dirs[cid][gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    T *out_color_ptr = out_colors + eid * D;
    eval_sh_function(degree_to_use, cid, gid, c, dir, coeffs, out_color_ptr);
}

// Evalute Spherical Harmonic functions at the given directions, assuming a N gaussians with K SH
// coefficients per gaussian.
template <typename T>
__global__ void
compute_sh_fwd_kernel_packed(
    const uint32_t N, const uint32_t K, const uint32_t D, const uint32_t degree_to_use,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> dirs,   // [N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> coeffs, // [K, N, D]
    const int *__restrict__ radii,                                              // [N]
    T *__restrict__ out_colors                                                  // [N, D]
) {
    // parallelize over N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // gidx * D + kidx
    if (idx >= N * D) {
        return;
    }

    const uint32_t gid = idx / D; // gidx
    const uint32_t c   = idx % D; // color channel
    if (radii != nullptr && radii[gid] <= 0) {
        return;
    }

    using vec3t          = typename Vec3Type<T>::type;
    const bool  has_dirs = dirs.size(0) > 0;
    const vec3t dir =
        has_dirs ? *reinterpret_cast<vec3t *>(dirs[gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    T *out_color_ptr = out_colors + gid * D;
    eval_sh_function(degree_to_use, 0, gid, c, dir, coeffs, out_color_ptr);
}

template <typename T>
__global__ void
compute_sh_bwd_kernel(
    const uint32_t C, const uint32_t N, const uint32_t K, const uint32_t D,
    const uint32_t                                                      degree_to_use,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> dirs,     // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> coeffs,   // [K, C, N, D]
    const int *__restrict__ radii,                                                // [C, N]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> v_colors, // [C, N, D]
    torch::PackedTensorAccessor32<T, 4, torch::RestrictPtrTraits> out_v_coeffs,   // [K, C, N, D]
    T *__restrict__ out_v_dirs // [C, N, 3] optional
) {
    // parallelize over C * N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= C * N * D) {
        return;
    }

    const uint32_t eid = idx / D; // cidx * N + gidx
    const uint32_t cid = eid / N; // camera index
    const uint32_t gid = eid % N; // gaussian index
    const uint32_t c   = idx % D; // color channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    using vec3t          = typename Vec3Type<T>::type;
    const bool  has_dirs = dirs.size(0) > 0;
    const vec3t dir =
        has_dirs ? *reinterpret_cast<vec3t *>(dirs[cid][gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    const T *v_color_ptr = v_colors[cid][gid].data();

    vec3t  v_dir{ 0.f, 0.f, 0.f };
    vec3t *out_v_dir_ptr = out_v_dirs == nullptr ? nullptr : &v_dir;

    eval_sh_function_vjp(degree_to_use, cid, gid, c, dir, coeffs, v_color_ptr, out_v_coeffs,
                         out_v_dir_ptr);
    if (out_v_dirs != nullptr) {
        gpuAtomicAdd(out_v_dirs + eid * 3, v_dir.x);
        gpuAtomicAdd(out_v_dirs + eid * 3 + 1, v_dir.y);
        gpuAtomicAdd(out_v_dirs + eid * 3 + 2, v_dir.z);
    }
}

template <typename T>
__global__ void
compute_sh_bwd_kernel_packed(
    const uint32_t N, const uint32_t K, const uint32_t D, const uint32_t degree_to_use,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> dirs,         // [N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> coeffs,       // [K, N, D]
    const int *__restrict__ radii,                                                    // [N]
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> v_colors,     // [N, D]
    torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits>       out_v_coeffs, // [K, N, D]
    T *__restrict__ out_v_dirs // [N, 3] optional
) {
    // parallelize over N * D
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; // gidx * D + c
    if (idx >= N * D) {
        return;
    }

    const uint32_t gid = idx / D; // gidx
    const uint32_t c   = idx % D; // color channel
    if (radii != nullptr && radii[gid] <= 0) {
        return;
    }

    using vec3t          = typename Vec3Type<T>::type;
    const bool  has_dirs = dirs.size(0) > 0;
    const vec3t dir =
        has_dirs ? *reinterpret_cast<vec3t *>(dirs[gid].data()) : vec3t{ 0.f, 0.f, 0.f };
    const T *v_color_ptr = v_colors[gid].data();

    vec3t  v_dir{ 0.f, 0.f, 0.f };
    vec3t *out_v_dir_ptr = out_v_dirs == nullptr ? nullptr : &v_dir;

    eval_sh_function_vjp(degree_to_use, static_cast<uint32_t>(0), gid, c, dir, coeffs, v_color_ptr,
                         out_v_coeffs, out_v_dir_ptr);
    if (out_v_dirs != nullptr) {
        gpuAtomicAdd(out_v_dirs + gid * 3, v_dir.x);
        gpuAtomicAdd(out_v_dirs + gid * 3 + 1, v_dir.y);
        gpuAtomicAdd(out_v_dirs + gid * 3 + 2, v_dir.z);
    }
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCUDA>(
    const int            sh_degree_to_use,
    const torch::Tensor &dirs,      // [N, 3] or [C, N, D] or empty for degree 0
    const torch::Tensor &sh_coeffs, // [N, K, D] or [C, N, K, D]
    const torch::Tensor &radii      // [N] or [C, N]
) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(sh_coeffs));

    const bool is_packed = sh_coeffs.dim() == 3;

    TORCH_CHECK_VALUE(sh_coeffs.is_cuda(), "sh_coeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");

    if (is_packed) {
        TORCH_CHECK_VALUE(sh_coeffs.dim() == 3, "sh_coeffs must have shape [K, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have shape [N]");
        TORCH_CHECK_VALUE(sh_coeffs.size(1) == radii.size(0),
                          "sh_coeffs must have shape [K, N, D] and radii must have shape [N]");
    } else {
        TORCH_CHECK_VALUE(sh_coeffs.dim() == 4, "sh_coeffs must have shape [K, C, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(1) == radii.size(0),
            "sh_coeffs must have shape [K. C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(2) == radii.size(1),
            "sh_coeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
    }

    const uint32_t K           = sh_coeffs.size(0);
    const uint32_t N           = is_packed ? sh_coeffs.size(1) : sh_coeffs.size(2);
    const uint32_t C           = is_packed ? 1 : sh_coeffs.size(1);
    const uint32_t D           = sh_coeffs.size(-1);
    const uint32_t TOTAL_ELEMS = C * N * D;
    const uint32_t NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (K > 0 && sh_degree_to_use > 0) {
        if (is_packed) {
            TORCH_CHECK_VALUE(
                dirs.dim() == 2,
                "sh_coeffs must have shape [K, N, D] and dirs must have shape [N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(1) == dirs.size(0),
                "sh_coeffs must have shape [K, N, D] and dirs must have shape [N, D]");
        } else {
            TORCH_CHECK_VALUE(
                dirs.dim() == 3,
                "sh_coeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(1) == dirs.size(0),
                "sh_coeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(2) == dirs.size(1),
                "sh_coeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
        }
        TORCH_CHECK_VALUE(dirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(dirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(sh_coeffs.device().index());

    // TODO (Francis): Might need to do zeros_like here
    const std::vector<int64_t> out_size =
        is_packed ? std::vector<int64_t>{ N, D } : std::vector<int64_t>{ C, N, D };
    torch::Tensor colors = torch::empty(out_size, sh_coeffs.options()); // [..., D]

    using scalar_t = float;

    // parallelize over N * D
    if (!N) {
        return colors; // [..., D]
    }
    if (is_packed) {
        compute_sh_fwd_kernel_packed<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            N, K, D, sh_degree_to_use,
            dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sh_coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(), colors.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        compute_sh_fwd_kernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, K, D, sh_degree_to_use,
            dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sh_coeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(), colors.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return colors; // [..., D]
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCPU>(const int            sh_degree_to_use,
                                               const torch::Tensor &dirs,      // [N, 3]
                                               const torch::Tensor &sh_coeffs, // [N, K, 3]
                                               const torch::Tensor &radii      // [N]
) {
    TORCH_CHECK(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCUDA>(
    const int            sh_degree_to_use,
    const torch::Tensor &dirs,      // [C, N, 3] or [N, 3] or empty for degree 0
    const torch::Tensor &sh_coeffs, // [C, N, K, 3] or [N, K, 3]
    const torch::Tensor &v_colors,  // [C, N, 3] or [N, 3]
    const torch::Tensor &radii,     // [C, N] or [N]
    const bool           compute_v_dirs) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(sh_coeffs));

    const bool is_packed = sh_coeffs.dim() == 3;

    TORCH_CHECK_VALUE(sh_coeffs.is_cuda(), "sh_coeffs must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
    TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    TORCH_CHECK_VALUE(v_colors.is_cuda(), "v_colors must be a CUDA tensor");

    if (is_packed) {
        TORCH_CHECK_VALUE(sh_coeffs.dim() == 3, "sh_coeffs must have shape [K, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 1, "radii must have shape [N]");
        TORCH_CHECK_VALUE(sh_coeffs.size(1) == radii.size(0),
                          "sh_coeffs must have shape [K, N, D] and radii must have shape [N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(2) == v_colors.size(0),
            "sh_coeffs and v_colors must have the same number of elements in dimension 0");
    } else {
        TORCH_CHECK_VALUE(sh_coeffs.dim() == 4, "sh_coeffs must have shape [K, C, N, D]");
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(1) == radii.size(0),
            "sh_coeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(2) == radii.size(1),
            "sh_coeffs must have shape [K, C, N, D] and radii must have shape [C, N]");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(1) == v_colors.size(0),
            "sh_coeffs and v_colors must have the same number of elements in dimension 0");
        TORCH_CHECK_VALUE(
            sh_coeffs.size(2) == v_colors.size(1),
            "sh_coeffs and v_colors must have the same number of elements in dimension 1");
    }
    TORCH_CHECK_VALUE(
        sh_coeffs.size(-1) == v_colors.size(-1),
        "sh_coeffs and v_colors must have the same number of elements in the last dimension");

    const uint32_t K           = sh_coeffs.size(0);
    const uint32_t N           = is_packed ? sh_coeffs.size(1) : sh_coeffs.size(2);
    const uint32_t C           = is_packed ? 1 : sh_coeffs.size(1);
    const uint32_t D           = sh_coeffs.size(-1);
    const uint32_t TOTAL_ELEMS = C * N * D;
    const uint32_t NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (K > 0 && sh_degree_to_use > 0) {
        if (is_packed) {
            TORCH_CHECK_VALUE(
                dirs.dim() == 2,
                "sh_coeffs must have shape [K, N, D] and dirs must have shape [N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(1) == dirs.size(0),
                "sh_coeffs must have shape [K, N, D] and dirs must have shape [N, D]");
        } else {
            TORCH_CHECK_VALUE(
                dirs.dim() == 3,
                "sh_coeffs must have shape [K, C, N, D] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(1) == dirs.size(0),
                "sh_coeffs must have shape [K, C, N, 3] and dirs must have shape [C, N, D]");
            TORCH_CHECK_VALUE(
                sh_coeffs.size(2) == dirs.size(1),
                "sh_coeffs must have shape [K, C, N, 3] and dirs must have shape [C, N, D]");
        }
        TORCH_CHECK_VALUE(dirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(dirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(sh_coeffs.device().index());

    torch::Tensor v_coeffs = torch::zeros_like(sh_coeffs);
    torch::Tensor v_dirs;
    if (compute_v_dirs) {
        v_dirs = torch::zeros_like(dirs);
    }
    if (!N) {
        std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
    }

    using scalar_t = float;
    if (is_packed) {
        compute_sh_bwd_kernel_packed<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            N, K, D, sh_degree_to_use,
            dirs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            sh_coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(),
            v_colors.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            v_coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            compute_v_dirs ? v_dirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        compute_sh_bwd_kernel<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, K, D, sh_degree_to_use,
            dirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sh_coeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            radii.data_ptr<int>(),
            v_colors.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            v_coeffs.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            compute_v_dirs ? v_dirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(v_coeffs, v_dirs); // [..., K, 3], [..., 3]
}

template <>
std::tuple<torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCPU>(const int            sh_degree_to_use,
                                                const torch::Tensor &dirs,      // [N, 3]
                                                const torch::Tensor &sh_coeffs, // [N, K, 3]
                                                const torch::Tensor &v_colors,
                                                const torch::Tensor &radii,     // [N]
                                                const bool           compute_v_dirs) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb