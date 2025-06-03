// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/ops/gsplat/GaussianVectorTypes.cuh>

#include <ATen/cuda/Atomic.cuh>

constexpr int NUM_THREADS = 1024;

namespace fvdb {
namespace detail {
namespace ops {

namespace {

// Evaluate spherical harmonics bases at unit direction for high orders using
// approach described by Efficient Spherical Harmonic Evaluation, Peter-Pike
// Sloan, JCGT 2013 See https://jcgt.org/published/0002/02/06/ for reference
// implementation
template <typename T>
inline __device__ T
evalShFunction(const int64_t degree,                      // degree of SH to be evaluated
               const int64_t ci,                          // camera index
               const int64_t gi,                          // gaussian index
               const int64_t c,                           // render channel
               const typename Vec3Type<T>::type &viewDir, // [D]
               const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> sh0Coeffs,
               const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> shNCoeffs) {
    const T cSH0 = sh0Coeffs[gi][0][c];

    T result = T(0.2820947917738781) * cSH0;

    if (degree >= 1) {
        // Normally rsqrt is faster than sqrt, but --use_fast_math will optimize
        // sqrt on single precision, so we use sqrt here.
        const T inorm =
            rsqrtf(viewDir.x * viewDir.x + viewDir.y * viewDir.y + viewDir.z * viewDir.z);
        const T x = viewDir.x * inorm;
        const T y = viewDir.y * inorm;
        const T z = viewDir.z * inorm;

        const T cSH1 = shNCoeffs[gi][0][c];
        const T cSH2 = shNCoeffs[gi][1][c];
        const T cSH3 = shNCoeffs[gi][2][c];

        result += 0.48860251190292f * (-y * cSH1 + z * cSH2 - x * cSH3);

        if (degree >= 2) {
            const T z2 = z * z;

            const T fTmp0B = T(-1.092548430592079) * z;
            const T fC1    = x * x - y * y;
            const T fS1    = 2.f * x * y;
            const T pSH6   = (T(0.9461746957575601) * z2 - T(0.3153915652525201));
            const T pSH7   = fTmp0B * x;
            const T pSH5   = fTmp0B * y;
            const T pSH8   = T(0.5462742152960395) * fC1;
            const T pSH4   = T(0.5462742152960395) * fS1;

            const T cSH4 = shNCoeffs[gi][3][c];
            const T cSH5 = shNCoeffs[gi][4][c];
            const T cSH6 = shNCoeffs[gi][5][c];
            const T cSH7 = shNCoeffs[gi][6][c];
            const T cSH8 = shNCoeffs[gi][7][c];

            result += (pSH4 * cSH4) + (pSH5 * cSH5) + (pSH6 * cSH6) + (pSH7 * cSH7) + (pSH8 * cSH8);

            if (degree >= 3) {
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

                const T cSH9  = shNCoeffs[gi][8][c];
                const T cSH10 = shNCoeffs[gi][9][c];
                const T cSH11 = shNCoeffs[gi][10][c];
                const T cSH12 = shNCoeffs[gi][11][c];
                const T cSH13 = shNCoeffs[gi][12][c];
                const T cSH14 = shNCoeffs[gi][13][c];
                const T cSH15 = shNCoeffs[gi][14][c];

                result += (pSH9 * cSH9) + (pSH10 * cSH10) + (pSH11 * cSH11) + (pSH12 * cSH12) +
                          (pSH13 * cSH13) + (pSH14 * cSH14) + (pSH15 * cSH15);

                if (degree >= 4) {
                    const T fTmp0D = z * (T(-4.683325804901025) * z2 + T(2.007139630671868));
                    const T fTmp1C = T(3.31161143515146) * z2 - T(0.47308734787878);
                    const T fTmp2B = -1.770130769779931f * z;
                    const T fC3    = x * fC2 - y * fS2;
                    const T fS3    = x * fS2 + y * fC2;
                    const T pSH20 =
                        (T(1.984313483298443) * z * pSH12 - T(1.006230589874905) * pSH6);
                    const T pSH21 = fTmp0D * x;
                    const T pSH19 = fTmp0D * y;
                    const T pSH22 = fTmp1C * fC1;
                    const T pSH18 = fTmp1C * fS1;
                    const T pSH23 = fTmp2B * fC2;
                    const T pSH17 = fTmp2B * fS2;
                    const T pSH24 = T(0.6258357354491763) * fC3;
                    const T pSH16 = T(0.6258357354491763) * fS3;

                    const T cSH16 = shNCoeffs[gi][15][c];
                    const T cSH17 = shNCoeffs[gi][16][c];
                    const T cSH18 = shNCoeffs[gi][17][c];
                    const T cSH19 = shNCoeffs[gi][18][c];
                    const T cSH20 = shNCoeffs[gi][19][c];
                    const T cSH21 = shNCoeffs[gi][20][c];
                    const T cSH22 = shNCoeffs[gi][21][c];
                    const T cSH23 = shNCoeffs[gi][22][c];
                    const T cSH24 = shNCoeffs[gi][23][c];

                    result += (pSH16 * cSH16) + (pSH17 * cSH17) + (pSH18 * cSH18) +
                              (pSH19 * cSH19) + (pSH20 * cSH20) + (pSH21 * cSH21) +
                              (pSH22 * cSH22) + (pSH23 * cSH23) + (pSH24 * cSH24);
                }
            }
        }
    }

    return result + T(0.5);
}

} // namespace

// Evalute Spherical Harmonic functions at the given directions, assuming a uniform minibatch
// of C cameras, each with N gaussians, and K SH coefficients per gaussian.
template <typename T>
__global__ void
computeSh(
    const int64_t C,
    const int64_t N,
    const int64_t D,
    const int64_t shDegreeToUse,
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> viewDirs,  // [C, N, 3]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> sh0Coeffs, // [1, N, D]
    const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> shNCoeffs, // [K-1, N, D]
    const int *__restrict__ radii,                                                 // [C, N]
    T *__restrict__ outRenderQuantities                                            // [C, N, D]
) {
    // parallelize over C * N * D
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= C * N * D) {
        return;
    }

    const auto eid = idx / D; // cidx * N + gidx
    const auto cid = eid / N; // camera index
    const auto gid = eid % N; // gaussian index
    const auto c   = idx % D; // render channel

    T result = T(0);
    if (!(radii != nullptr && radii[eid] <= 0)) {
        using vec3t            = typename Vec3Type<T>::type;
        const bool hasViewDirs = viewDirs.size(0) > 0;
        const vec3t dir        = hasViewDirs ? *reinterpret_cast<vec3t *>(viewDirs[cid][gid].data())
                                             : vec3t{0.f, 0.f, 0.f};
        result = evalShFunction(shDegreeToUse, cid, gid, c, dir, sh0Coeffs, shNCoeffs);
    }
    outRenderQuantities[eid * D + c] = result;
}

template <typename T>
__global__ void
computeShDiffuseOnly(const int64_t C,
                     const int64_t N,
                     const int64_t D,
                     const torch::PackedTensorAccessor32<T, 3, torch::RestrictPtrTraits> sh0Coeffs,
                     const int *__restrict__ radii, // [C, N]
                     T *__restrict__ outRenderQuantities) {
    // parallelize over C * N * D
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x; // cidx * N * D + gidx * D + kidx
    if (idx >= C * N * D) {
        return;
    }

    const auto eid = idx / D; // cidx * N + gidx
    const auto gid = eid % N; // gaussian index
    const auto c   = idx % D; // render channel
    if (radii != nullptr && radii[eid] <= 0) {
        outRenderQuantities[eid * D + c] = T(0);
    } else {
        outRenderQuantities[eid * D + c] = T(0.2820947917738781) * sh0Coeffs[gid][0][c] + T(0.5);
    }
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCUDA>(const int64_t shDegreeToUse,
                                                const int64_t numCameras,
                                                const torch::Tensor &viewDirs,  // [C, N, 3]
                                                const torch::Tensor &sh0Coeffs, // [N, 1, D]
                                                const torch::Tensor &shNCoeffs, // [N, K-1, D]
                                                const torch::Tensor &radii      // [C, N]
) {
    // Valid modes:
    // 0: sh0Coeffs only
    // 1: sh0Coeffs + radii
    // 2: sh0Coeffs + shNCoeffs + viewDirs
    // 2: sh0Coeffs + shNCoeffs + viewDirs + radii

    const at::cuda::OptionalCUDAGuard device_guard(at::device_of(sh0Coeffs));

    const bool hasShNCoeffs = shNCoeffs.defined();
    const bool hasViewirs   = viewDirs.defined();
    const bool hasRadii     = radii.defined();

    const int64_t numGaussians = sh0Coeffs.size(0);

    TORCH_CHECK_VALUE(sh0Coeffs.dim() == 3, "sh0Coeffs must have shape [K, N, D]");
    TORCH_CHECK_VALUE(sh0Coeffs.is_cuda(), "sh0Coeffs must be a CUDA tensor");

    if (hasShNCoeffs) {
        TORCH_CHECK_VALUE(hasViewirs, "viewDirs must be defined if shNCoeffs is defined");
        TORCH_CHECK_VALUE(shNCoeffs.is_cuda(), "shNCoeffs must be a CUDA tensor");
        TORCH_CHECK_VALUE(shNCoeffs.dim() == 3, "shNCoeffs must have shape [N, K, D]");
        TORCH_CHECK_VALUE(shNCoeffs.size(0) == numGaussians, "shNCoeffs must have shape [N, K, D]");
    } else {
        TORCH_CHECK_VALUE(shDegreeToUse == 0, "shDegreeToUse must be 0 if no shNCoeffs");
    }

    if (hasRadii) {
        TORCH_CHECK_VALUE(radii.dim() == 2, "radii must have two dimensions with shape [C, N]");
        TORCH_CHECK_VALUE(numGaussians == radii.size(1), "radii must have shape [C, N]");
        TORCH_CHECK_VALUE(radii.size(0) == numCameras,
                          "radii must have shape [C, N] and C must match numCameras");
        TORCH_CHECK_VALUE(radii.is_cuda(), "radii must be a CUDA tensor");
        TORCH_CHECK_VALUE(radii.is_contiguous(), "radii must be a contiguous");
    }

    const int64_t K        = hasShNCoeffs ? shNCoeffs.size(1) + 1 : 1;
    const int64_t N        = sh0Coeffs.size(0);
    const int64_t C        = numCameras;
    const int64_t D        = sh0Coeffs.size(2);
    const auto TOTAL_ELEMS = C * N * D;
    const auto NUM_BLOCKS  = (TOTAL_ELEMS + NUM_THREADS - 1) / NUM_THREADS;

    // If you are using degree > 0, then we are going to use the directions tensor which means
    // we need to check it has the right shape
    if (hasShNCoeffs && K > 0 && shDegreeToUse > 0) {
        TORCH_CHECK_VALUE(viewDirs.dim() == 3, "viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(
            shNCoeffs.size(0) == viewDirs.size(1),
            "shNCoeffs must have shape [N, K, D] and viewDirs must have shape [C, N, 3]");
        TORCH_CHECK_VALUE(viewDirs.is_cuda(), "dirs must be a CUDA tensor");
        TORCH_CHECK_VALUE(viewDirs.size(-1) == 3, "dirs must have last dimension 3");
    }

    if (N == 0) {
        return torch::empty({int64_t(C), N, D}, sh0Coeffs.options());
    }

    using scalar_t                    = float;
    const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(sh0Coeffs.device().index());

    const int *radiiPtr            = hasRadii ? radii.data_ptr<int>() : nullptr;
    torch::Tensor renderQuantities = torch::empty({int64_t(C), N, D}, sh0Coeffs.options());
    if (hasShNCoeffs && shDegreeToUse > 0) {
        computeSh<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C,
            N,
            D,
            shDegreeToUse,
            viewDirs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            sh0Coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            shNCoeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        computeShDiffuseOnly<scalar_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            C,
            N,
            D,
            sh0Coeffs.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            radiiPtr,
            renderQuantities.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return renderQuantities;
}

template <>
torch::Tensor
dispatchSphericalHarmonicsForward<torch::kCPU>(const int64_t shDegreeToUse,
                                               const int64_t numCameras,
                                               const torch::Tensor &dirs,      // [N, 3]
                                               const torch::Tensor &sh0Coeffs, // [1, N, D]
                                               const torch::Tensor &shNCoeffs, // [K-1, N, D]
                                               const torch::Tensor &radii      // [N]
) {
    TORCH_CHECK(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
