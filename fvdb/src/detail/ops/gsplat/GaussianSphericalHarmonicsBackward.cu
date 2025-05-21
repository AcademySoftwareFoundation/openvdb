// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianVectorTypes.cuh"
#include <detail/ops/Ops.h>
#include <detail/utils/AccessorHelpers.cuh>

#include <ATen/cuda/Atomic.cuh>

constexpr int NUM_THREADS = 1024;

namespace fvdb {
namespace detail {
namespace ops {

namespace {

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

template <typename T>
inline __device__ void
evalShFunctionVJP(const int64_t                     degree, // degree of SH to be evaluated
                  const int64_t                     ci,     // camera index
                  const int64_t                     gi,     // gaussian index
                  const int64_t                     c,      // render channel
                  const typename Vec3Type<T>::type &dir,    // [3]
                  const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> coeffsN,
                  const T *dLossDRenderQuantities,          // [D]
                  torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> dLossDSh0Coeffs,
                  torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> dLossDShNCoeffs,
                  typename Vec3Type<T>::type *dLossDViewDir // [3] optional
) {
    T dLossDRenderQuantitiesLocal = dLossDRenderQuantities[c];

    dLossDSh0Coeffs[gi][0][c] = T(0.2820947917738781) * dLossDRenderQuantitiesLocal;

    if (degree < 1) {
        return;
    }
    const T inorm = rsqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    const T x     = dir.x * inorm;
    const T y     = dir.y * inorm;
    const T z     = dir.z * inorm;
    T       vX = 0.f, vY = 0.f, vZ = 0.f;

    dLossDShNCoeffs[gi][0][c] = T(-0.48860251190292) * y * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][1][c] = T(0.48860251190292) * z * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][2][c] = T(-0.48860251190292) * x * dLossDRenderQuantitiesLocal;

    if (dLossDViewDir != nullptr) {
        vX += T(-0.48860251190292) * coeffsN[gi][2][c] * dLossDRenderQuantitiesLocal;
        vY += T(-0.48860251190292) * coeffsN[gi][0][c] * dLossDRenderQuantitiesLocal;
        vZ += T(0.48860251190292) * coeffsN[gi][1][c] * dLossDRenderQuantitiesLocal;
    }
    if (degree < 2) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T z2                = z * z;
    const T fTmp0B            = T(-1.092548430592079) * z;
    const T fC1               = x * x - y * y;
    const T fS1               = 2.f * x * y;
    const T pSH6              = (T(0.9461746957575601) * z2 - T(0.3153915652525201));
    const T pSH7              = fTmp0B * x;
    const T pSH5              = fTmp0B * y;
    const T pSH8              = T(0.5462742152960395) * fC1;
    const T pSH4              = T(0.5462742152960395) * fS1;
    dLossDShNCoeffs[gi][3][c] = pSH4 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][4][c] = pSH5 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][5][c] = pSH6 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][6][c] = pSH7 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][7][c] = pSH8 * dLossDRenderQuantitiesLocal;

    T fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y, pSH5_z, pSH8_x, pSH8_y,
        pSH4_x, pSH4_y;
    if (dLossDViewDir != nullptr) {
        fTmp0B_z = T(-1.092548430592079);
        fC1_x    = T(2.0) * x;
        fC1_y    = T(-2.0) * y;
        fS1_x    = T(2.0) * y;
        fS1_y    = T(2.0) * x;
        pSH6_z   = T(2.0) * T(0.9461746957575601) * z;
        pSH7_x   = fTmp0B;
        pSH7_z   = fTmp0B_z * x;
        pSH5_y   = fTmp0B;
        pSH5_z   = fTmp0B_z * y;
        pSH8_x   = T(0.5462742152960395) * fC1_x;
        pSH8_y   = T(0.5462742152960395) * fC1_y;
        pSH4_x   = T(0.5462742152960395) * fS1_x;
        pSH4_y   = T(0.5462742152960395) * fS1_y;

        vX +=
            dLossDRenderQuantitiesLocal *
            (pSH4_x * coeffsN[gi][3][c] + pSH8_x * coeffsN[gi][7][c] + pSH7_x * coeffsN[gi][6][c]);
        vY +=
            dLossDRenderQuantitiesLocal *
            (pSH4_y * coeffsN[gi][3][c] + pSH8_y * coeffsN[gi][7][c] + pSH5_y * coeffsN[gi][4][c]);
        vZ +=
            dLossDRenderQuantitiesLocal *
            (pSH6_z * coeffsN[gi][5][c] + pSH7_z * coeffsN[gi][6][c] + pSH5_z * coeffsN[gi][4][c]);
    }

    if (degree < 3) {
        if (dLossDViewDir != nullptr) {
            writeDLossDViewDir(x, y, z, vX, vY, vZ, inorm, dLossDViewDir);
        }
        return;
    }

    const T fTmp0C = T(-2.285228997322329) * z2 + T(0.4570457994644658);
    const T fTmp1B = T(1.445305721320277) * z;
    const T fC2    = x * fC1 - y * fS1;
    const T fS2    = x * fS1 + y * fC1;
    const T pSH12  = z * (T(1.865881662950577) * z2 - T(1.119528997770346));
    const T pSH13  = fTmp0C * x;
    const T pSH11  = fTmp0C * y;
    const T pSH14  = fTmp1B * fC1;
    const T pSH10  = fTmp1B * fS1;
    const T pSH15  = T(-0.5900435899266435) * fC2;
    const T pSH9   = T(-0.5900435899266435) * fS2;

    dLossDShNCoeffs[gi][8][c]  = pSH9 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][9][c]  = pSH10 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][10][c] = pSH11 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][11][c] = pSH12 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][12][c] = pSH13 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][13][c] = pSH14 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][14][c] = pSH15 * dLossDRenderQuantitiesLocal;

    T fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x, pSH13_z, pSH11_y, pSH11_z,
        pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y, pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;
    if (dLossDViewDir != nullptr) {
        fTmp0C_z = T(-2.285228997322329) * T(2.0) * z;
        fTmp1B_z = T(1.445305721320277);
        fC2_x    = fC1 + x * fC1_x - y * fS1_x;
        fC2_y    = x * fC1_y - fS1 - y * fS1_y;
        fS2_x    = fS1 + x * fS1_x + y * fC1_x;
        fS2_y    = x * fS1_y + fC1 + y * fC1_y;
        pSH12_z  = T(3.0) * T(1.865881662950577) * z2 - T(1.119528997770346);
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
        pSH15_x  = T(-0.5900435899266435) * fC2_x;
        pSH15_y  = T(-0.5900435899266435) * fC2_y;
        pSH9_x   = T(-0.5900435899266435) * fS2_x;
        pSH9_y   = T(-0.5900435899266435) * fS2_y;

        const T cSH9  = coeffsN[gi][8][c];
        const T cSH10 = coeffsN[gi][9][c];
        const T cSH11 = coeffsN[gi][10][c];
        const T cSH12 = coeffsN[gi][11][c];
        const T cSH13 = coeffsN[gi][12][c];
        const T cSH14 = coeffsN[gi][13][c];
        const T cSH15 = coeffsN[gi][14][c];

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

    const T fTmp0D = z * (T(-4.683325804901025) * z2 + T(2.007139630671868));
    const T fTmp1C = T(3.31161143515146) * z2 - T(0.47308734787878);
    const T fTmp2B = -1.770130769779931f * z;
    const T fC3    = x * fC2 - y * fS2;
    const T fS3    = x * fS2 + y * fC2;
    const T pSH20  = (T(1.984313483298443) * z * pSH12 + T(-1.006230589874905) * pSH6);
    const T pSH21  = fTmp0D * x;
    const T pSH19  = fTmp0D * y;
    const T pSH22  = fTmp1C * fC1;
    const T pSH18  = fTmp1C * fS1;
    const T pSH23  = fTmp2B * fC2;
    const T pSH17  = fTmp2B * fS2;
    const T pSH24  = T(0.6258357354491763) * fC3;
    const T pSH16  = T(0.6258357354491763) * fS3;

    dLossDShNCoeffs[gi][15][c] = pSH16 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][16][c] = pSH17 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][17][c] = pSH18 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][18][c] = pSH19 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][19][c] = pSH20 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][20][c] = pSH21 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][21][c] = pSH22 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][22][c] = pSH23 * dLossDRenderQuantitiesLocal;
    dLossDShNCoeffs[gi][23][c] = pSH24 * dLossDRenderQuantitiesLocal;

    T fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z, pSH21_x, pSH21_z, pSH19_y,
        pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x, pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z,
        pSH17_x, pSH17_y, pSH17_z, pSH24_x, pSH24_y, pSH16_x, pSH16_y;
    if (dLossDViewDir != nullptr) {
        fTmp0D_z = T(3.0) * T(-4.683325804901025) * z2 + T(2.007139630671868);
        fTmp1C_z = T(2.0) * 3.31161143515146f * z;
        fTmp2B_z = T(-1.770130769779931);
        fC3_x    = fC2 + x * fC2_x - y * fS2_x;
        fC3_y    = x * fC2_y - fS2 - y * fS2_y;
        fS3_x    = fS2 + y * fC2_x + x * fS2_x;
        fS3_y    = x * fS2_y + fC2 + y * fC2_y;
        pSH20_z  = T(1.984313483298443) * (pSH12 + z * pSH12_z) + T(-1.006230589874905) * pSH6_z;
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
        pSH24_x  = T(0.6258357354491763) * fC3_x;
        pSH24_y  = T(0.6258357354491763) * fC3_y;
        pSH16_x  = T(0.6258357354491763) * fS3_x;
        pSH16_y  = T(0.6258357354491763) * fS3_y;

        const T cSH16 = coeffsN[gi][15][c];
        const T cSH17 = coeffsN[gi][16][c];
        const T cSH18 = coeffsN[gi][17][c];
        const T cSH19 = coeffsN[gi][18][c];
        const T cSH20 = coeffsN[gi][19][c];
        const T cSH21 = coeffsN[gi][20][c];
        const T cSH22 = coeffsN[gi][21][c];
        const T cSH23 = coeffsN[gi][22][c];
        const T cSH24 = coeffsN[gi][23][c];

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

template <typename T>
__global__ void
computeShBackward(
    const int64_t C, const int64_t N, const int64_t K, const int64_t D, const int64_t shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> viewDirs,     // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> shNCoeffs,    // [K-1, N, D]
    const int *__restrict__ radii,                                                    // [C, N]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits>
        dLossDRenderQuantities,                                                       // [C, N, D]
    torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> outDLossDSh0Coeffs, // [N, 1, D]
    torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> outDLossDShNCoeffs, // [N, K-1, D]
    T *__restrict__ outDLossDViewDirs // [C, N, 3] optiondl
) {
    // parallelize over C * N * D
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= C * N * D) {
        return;
    }

    const auto eid = idx / D; // cidx * N + gidx
    const auto cid = eid / N; // camera index
    const auto gid = eid % N; // gaussian index
    const auto c   = idx % D; // render channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    using vec3t             = typename Vec3Type<T>::type;
    const bool  hasViewDirs = viewDirs.size(0) > 0;
    const vec3t viewDir     = hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[cid][gid].data())
                                          : vec3t{ T(0), T(0), T(0) };
    const T    *dLossDRenderQuantityPtr = dLossDRenderQuantities[cid][gid].data();

    vec3t  dLossDViewDir{ T(0), T(0), T(0) };
    vec3t *outDLossDViewDirPtr = outDLossDViewDirs == nullptr ? nullptr : &dLossDViewDir;

    evalShFunctionVJP(shDegreeToUse, cid, gid, c, viewDir, shNCoeffs, dLossDRenderQuantityPtr,
                      outDLossDSh0Coeffs, outDLossDShNCoeffs, outDLossDViewDirPtr);
    if (outDLossDViewDirs != nullptr) {
        gpuAtomicAdd(outDLossDViewDirs + eid * 3, dLossDViewDir.x);
        gpuAtomicAdd(outDLossDViewDirs + eid * 3 + 1, dLossDViewDir.y);
        gpuAtomicAdd(outDLossDViewDirs + eid * 3 + 2, dLossDViewDir.z);
    }
}

template <typename T>
__global__ void
computeShDiffuseOnlyBackward(
    const int64_t C, const int64_t N, const int64_t D,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits>
        dLossDRenderQuantities,                                                      // [C, N, D]
    const int *__restrict__ radii,                                                   // [C, N]
    torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> outDLossDSh0Coeffs // [N, 1, D]
) {
    // parallelize over C * N * D
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + c
    if (idx >= C * N * D) {
        return;
    }

    const auto eid = idx / D; // cidx * N + gidx
    const auto cid = eid / N; // camera index
    const auto gid = eid % N; // gaussian index
    const auto c   = idx % D; // render channel
    if (radii != nullptr && radii[eid] <= 0) {
        return;
    }

    outDLossDSh0Coeffs[gid][0][c] = T(0.2820947917738781) * dLossDRenderQuantities[cid][gid][c];
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCUDA>(
    const int64_t shDegreeToUse, const int64_t numCameras, const int64_t numGaussians,
    const torch::Tensor &viewDirs,               // [C, N, 3]
    const torch::Tensor &shNCoeffs,              // [N, K-1, D]
    const torch::Tensor &dLossDRenderQuantities, // [C, N, D]
    const torch::Tensor &radii,                  // [C, N]
    const bool           computeDLossDViewDirs) {
    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(dLossDRenderQuantities));

    const bool hasShNCoeffs = shNCoeffs.defined();
    const bool hasViewirs   = viewDirs.defined();
    const bool hasRadii     = radii.defined();

    if (hasShNCoeffs) {
        TORCH_CHECK_VALUE(hasViewirs, "viewDirs must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(shNCoeffs.is_cuda(), "shNCoeffs must be a CUDA tensor");
        TORCH_CHECK_VALUE(shNCoeffs.dim() == 3, "shNCoeffs must have shape [N, K-1, D]");
        TORCH_CHECK_VALUE(shNCoeffs.size(0) == numGaussians,
                          "shNCoeffs must have shape [N, K-1, D]");
    } else {
        TORCH_CHECK_VALUE(shDegreeToUse == 0, "shDegreeToUse must be 0 if no shNCoeffs");
    }
    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have two dimensions with shape [C, N]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1),
                          "radii must have shape [C, N] but got shape = ", radii.sizes());
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    const int64_t K           = hasShNCoeffs ? shNCoeffs.size(1) + 1 : 1;
    const int64_t N           = dLossDRenderQuantities.size(1);
    const int64_t C           = numCameras;
    const int64_t D           = dLossDRenderQuantities.size(2);
    const int64_t TOTAL_ELEMS = C * N * D;
    const int64_t NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (hasShNCoeffs && K > 1 && shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(viewDirs.dim() == 3, "viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(
            shNCoeffs.size(0) == viewDirs.size(1),
            "shNCoeffs must have shape [N, K-1, D] and viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(viewDirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(viewDirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    at::cuda::CUDAStream stream =
        at::cuda::getCurrentCUDAStream(dLossDRenderQuantities.device().index());

    using scalar_t = float;

    const int *radiiPtr = hasRadii ? radii.data_ptr<int>() : nullptr;

    const auto tensorOptions = dLossDRenderQuantities.options();
    if (hasShNCoeffs && K > 1) {
        torch::Tensor dLossDShNCoeffs = torch::zeros_like(shNCoeffs);
        torch::Tensor dLossDSh0Coeffs = torch::zeros({ N, 1, D }, tensorOptions);
        torch::Tensor dLossDViewDirs;
        if (computeDLossDViewDirs) {
            dLossDViewDirs = torch::zeros_like(viewDirs);
        }
        if (N == 0) {
            return std::make_tuple(dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs);
        }

        computeShBackward<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, K, D, shDegreeToUse,
            viewDirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            shNCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(), radiiPtr,
            dLossDRenderQuantities.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLossDSh0Coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            dLossDShNCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            computeDLossDViewDirs ? dLossDViewDirs.data_ptr<scalar_t>() : nullptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return std::make_tuple(dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs);
    } else {
        torch::Tensor dLossDSh0Coeffs = torch::zeros({ N, 1, D }, tensorOptions);
        torch::Tensor dLossDShNCoeffs;
        torch::Tensor dLossDViewDirs;
        if (N == 0) {
            return std::make_tuple(dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs);
        }

        computeShDiffuseOnlyBackward<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C, N, D,
            dLossDRenderQuantities.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr, dLossDSh0Coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return std::make_tuple(dLossDSh0Coeffs, dLossDShNCoeffs, dLossDViewDirs);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
dispatchSphericalHarmonicsBackward<torch::kCPU>(const int64_t        shDegreeToUse,
                                                const int64_t        numCameras,
                                                const int64_t        numGaussians,
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
