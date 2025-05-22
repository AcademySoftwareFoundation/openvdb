// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianMacros.cuh"
#include "GaussianUtils.cuh"
#include "GaussianWarpUtils.cuh"

#include <detail/ops/Ops.h>

#include <nanovdb/math/Math.h>

#include <ATen/cuda/Atomic.cuh>

#include <cooperative_groups.h>

constexpr int NUM_THREADS = 256;

namespace fvdb {
namespace detail {
namespace ops {

namespace cg = cooperative_groups;

template <typename T, bool TRACK_MAX_RADII>
__global__ void
computeGradientState(const uint32_t C,
                     const uint32_t N,
                     const int32_t imageWidth,
                     const int32_t imageHeight,
                     const int32_t *__restrict__ radii,
                     const T *__restrict__ dLossDMeans2d,
                     T *__restrict__ outDLossDMeans2dNormAccum,
                     int32_t *__restrict__ outMaxRadiiAccum,
                     int32_t *__restrict__ outGradientStepCounts) {
    const auto idx = cg::this_grid().thread_rank();

    if (idx >= N) {
        return;
    }

    T accum       = T(0);
    int32_t count = 0;
    for (auto i = 0; i < C * N; i += N) {
        const int32_t ri = radii[idx + i];
        if (ri <= 0) {
            continue;
        }
        const T dldm2x = dLossDMeans2d[(idx + i) * 2] * (T(imageWidth) / T(2) * T(C));
        const T dldm2y = dLossDMeans2d[(idx + i) * 2 + 1] * (T(imageHeight) / T(2) * T(C));
        accum += nanovdb::math::Sqrt(dldm2x * dldm2x + dldm2y * dldm2y);
        count += 1;
    }
    if constexpr (TRACK_MAX_RADII) {
        int32_t maxRad = 0;
        for (auto i = 0; i < C * N; i += N) {
            const int32_t ri = radii[idx + i];
            if (ri <= 0) {
                continue;
            }
            maxRad = nanovdb::math::Max(maxRad, ri);
        }
        outMaxRadiiAccum[idx] = nanovdb::math::Max(outMaxRadiiAccum[idx], maxRad);
    }
    outDLossDMeans2dNormAccum[idx] += accum;
    outGradientStepCounts[idx] += count;
}

template <typename T, bool Ortho>
__global__ void
projectionBackwardKernel(
    // fwd inputs
    const uint32_t C,
    const uint32_t N,
    const T *__restrict__ means,              // [N, 3]
    const T *__restrict__ covars,             // [N, 6] optional
    const T *__restrict__ quats,              // [N, 4] optional
    const T *__restrict__ scales,             // [N, 3] optional
    const T *__restrict__ worldToCamMatrices, // [C, 4, 4]
    const T *__restrict__ projectionMatrices, // [C, 3, 3]
    const int32_t imageWidth,
    const int32_t imageHeight,
    const T eps2d,
    // fwd outputs
    const int32_t *__restrict__ radii,   // [C, N]
    const T *__restrict__ conics,        // [C, N, 3]
    const T *__restrict__ compensations, // [C, N] optional
    // grad outputs
    const T *__restrict__ dLossDMeans2d,       // [C, N, 2]
    const T *__restrict__ dLossDDepths,        // [C, N]
    const T *__restrict__ dLossDConics,        // [C, N, 3]
    const T *__restrict__ dLossDCompensations, // [C, N] optional
    // grad inputs
    T *__restrict__ outDLossDMeans,             // [N, 3]
    T *__restrict__ outDLossDCovars,            // [N, 6] optional
    T *__restrict__ outDLossDQuats,             // [N, 4] optional
    T *__restrict__ outDLossDScales,            // [N, 3] optional
    T *__restrict__ outDLossDWorldToCamMatrices // [C, 4, 4] optional
) {
    // parallelize over C * N.
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= C * N || radii[idx] <= 0) {
        return;
    }
    const uint32_t cId = idx / N; // camera id
    const uint32_t gId = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gId * 3;
    worldToCamMatrices += cId * 16;
    projectionMatrices += cId * 9;

    conics += idx * 3;

    dLossDMeans2d += idx * 2;
    dLossDDepths += idx;
    dLossDConics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    const nanovdb::math::Mat2<T> covar2dInverse(conics[0], conics[1], conics[1], conics[2]);
    const nanovdb::math::Mat2<T> dLossDCovar2dInverse(
        dLossDConics[0], dLossDConics[1] * .5f, dLossDConics[1] * .5f, dLossDConics[2]);
    nanovdb::math::Mat2<T> dLossDCovar2d =
        inverseVectorJacobianProduct(covar2dInverse, dLossDCovar2dInverse);

    if (dLossDCompensations != nullptr) {
        // vjp: compensation term
        const T compensation       = compensations[idx];
        const T dLossDCompensation = dLossDCompensations[idx];
        dLossDCovar2d += generateBlurVectorJacobianProduct(
            eps2d, covar2dInverse, compensation, dLossDCompensation);
    }

    // transform Gaussian to camera space
    const nanovdb::math::Mat3<T> R(worldToCamMatrices[0],
                                   worldToCamMatrices[1],
                                   worldToCamMatrices[2],   // 1st row
                                   worldToCamMatrices[4],
                                   worldToCamMatrices[5],
                                   worldToCamMatrices[6],   // 2nd row
                                   worldToCamMatrices[8],
                                   worldToCamMatrices[9],
                                   worldToCamMatrices[10]); // 3rd row
    const nanovdb::math::Vec3<T> t(
        worldToCamMatrices[3], worldToCamMatrices[7], worldToCamMatrices[11]);
    nanovdb::math::Mat3<T> covar;
    nanovdb::math::Vec4<T> quat;
    nanovdb::math::Vec3<T> scale;
    if (covars != nullptr) {
        covars += gId * 6;
        covar = nanovdb::math::Mat3<T>(covars[0],
                                       covars[1],
                                       covars[2], // 1st row
                                       covars[1],
                                       covars[3],
                                       covars[4], // 2nd row
                                       covars[2],
                                       covars[4],
                                       covars[5]  // 3rd row
        );
    } else {
        // compute from quaternions and scales
        quats += gId * 4;
        scales += gId * 3;
        quat  = nanovdb::math::Vec4<T>(quats[0], quats[1], quats[2], quats[3]);
        scale = nanovdb::math::Vec3<T>(scales[0], scales[1], scales[2]);

        covar = quaternionAndScaleToCovariance<T>(quat, scale);
    }

    const nanovdb::math::Vec3<T> &meansCamSpace =
        transformPointWorldToCam(R, t, nanovdb::math::Vec3<T>(means[0], means[1], means[2]));

    const nanovdb::math::Mat3<T> &covarCamSpace = transformCovarianceWorldToCam(R, covar);

    // vjp: camera projection
    const T fx = projectionMatrices[0], cx = projectionMatrices[2], fy = projectionMatrices[4],
            cy                                     = projectionMatrices[5];
    auto [dLossDCovarCamSpace, dLossDMeanCamSpace] = [&]() {
        if constexpr (Ortho) {
            return projectGaussianOrthographicVectorJacobianProduct<T>(
                meansCamSpace,
                covarCamSpace,
                fx,
                fy,
                cx,
                cy,
                imageWidth,
                imageHeight,
                dLossDCovar2d,
                nanovdb::math::Vec2<T>(dLossDMeans2d[0], dLossDMeans2d[1]));
        } else {
            return projectGaussianPerspectiveVectorJacobianProduct<T>(
                meansCamSpace,
                covarCamSpace,
                fx,
                fy,
                cx,
                cy,
                imageWidth,
                imageHeight,
                dLossDCovar2d,
                nanovdb::math::Vec2<T>(dLossDMeans2d[0], dLossDMeans2d[1]));
        }
    }();

    // add contribution from dLossDDepths
    dLossDMeanCamSpace[2] += dLossDDepths[0];

    // vjp: transform Gaussian covariance to camera space
    auto [dLossDRotation, dLossDTranslation, dLossDPoint] =
        transformPointWorldToCamVectorJacobianProduct(
            R, t, nanovdb::math::Vec3<T>(means[0], means[1], means[2]), dLossDMeanCamSpace);

    auto [dLossDRotationCov, dLossDCovar] =
        transformCovarianceWorldToCamVectorJacobianProduct(R, covar, dLossDCovarCamSpace);

    dLossDRotation += dLossDRotationCov;

    // write out results with warp-level reduction
    auto warp         = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gId);
    if (outDLossDMeans != nullptr) {
        warpSum(dLossDPoint, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            outDLossDMeans += gId * 3;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(outDLossDMeans + i, dLossDPoint[i]);
            }
        }
    }
    if (outDLossDCovars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(dLossDCovar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            outDLossDCovars += gId * 6;
            gpuAtomicAdd(outDLossDCovars, dLossDCovar[0][0]);
            gpuAtomicAdd(outDLossDCovars + 1, dLossDCovar[0][1] + dLossDCovar[1][0]);
            gpuAtomicAdd(outDLossDCovars + 2, dLossDCovar[0][2] + dLossDCovar[2][0]);
            gpuAtomicAdd(outDLossDCovars + 3, dLossDCovar[1][1]);
            gpuAtomicAdd(outDLossDCovars + 4, dLossDCovar[1][2] + dLossDCovar[2][1]);
            gpuAtomicAdd(outDLossDCovars + 5, dLossDCovar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        const nanovdb::math::Mat3<T> &rotmat = quaternionToRotationMatrix<T>(quat);

        auto [dLossDQuat, dLossDScale] = quaternionAndScaleToCovarianceVectorJacobianProduct<T>(
            quat, scale, rotmat, dLossDCovar);

        warpSum(dLossDScale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            outDLossDQuats += gId * 4;
            outDLossDScales += gId * 3;
            gpuAtomicAdd(outDLossDQuats, dLossDQuat[0]);
            gpuAtomicAdd(outDLossDQuats + 1, dLossDQuat[1]);
            gpuAtomicAdd(outDLossDQuats + 2, dLossDQuat[2]);
            gpuAtomicAdd(outDLossDQuats + 3, dLossDQuat[3]);
            gpuAtomicAdd(outDLossDScales, dLossDScale[0]);
            gpuAtomicAdd(outDLossDScales + 1, dLossDScale[1]);
            gpuAtomicAdd(outDLossDScales + 2, dLossDScale[2]);
        }
    }
    if (outDLossDWorldToCamMatrices != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cId);
        warpSum(dLossDRotation, warp_group_c);
        warpSum(dLossDTranslation, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            outDLossDWorldToCamMatrices += cId * 16;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t i = 0; i < 3; i++) {     // rows
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(outDLossDWorldToCamMatrices + i * 4 + j, dLossDRotation[i][j]);
                }
                gpuAtomicAdd(outDLossDWorldToCamMatrices + i * 4 + 3, dLossDTranslation[i]);
            }
        }
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionBackward<torch::kCUDA>(
    // fwd inputs
    const torch::Tensor &means,                       // [N, 3]
    const torch::Tensor &quats,                       // [N, 4]
    const torch::Tensor &scales,                      // [N, 3]
    const torch::Tensor &worldToCamMatrices,          // [C, 4, 4]
    const torch::Tensor &projectionMatrices,          // [C, 3, 3]
    const at::optional<torch::Tensor> &compensations, // [N, 6] optional
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &conics, // [C, N, 3]
    // grad outputs
    const torch::Tensor &dLossDMeans2d,                              // [C, N, 2]
    const torch::Tensor &dLossDDepths,                               // [C, N]
    const torch::Tensor &dLossDConics,                               // [C, N, 3]
    const at::optional<torch::Tensor> &dLossDCompensations,          // [C, N] optional
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho,
    at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum, // [N]
    at::optional<torch::Tensor> outNormalizedMaxRadiiAccum,          // [N]
    at::optional<torch::Tensor> outGradientStepCounts                // [N]
) {
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars = std::nullopt;
    // const at::optional<torch::Tensor> &compensations = std::nullopt;
    // const at::optional<torch::Tensor> &dLossDCompensations = std::nullopt;

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        GSPLAT_CHECK_INPUT(quats);
        GSPLAT_CHECK_INPUT(scales);
    }
    GSPLAT_CHECK_INPUT(worldToCamMatrices);
    GSPLAT_CHECK_INPUT(projectionMatrices);
    GSPLAT_CHECK_INPUT(radii);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(dLossDMeans2d);
    GSPLAT_CHECK_INPUT(dLossDDepths);
    GSPLAT_CHECK_INPUT(dLossDConics);
    if (compensations.has_value()) {
        GSPLAT_CHECK_INPUT(compensations.value());
    }
    if (dLossDCompensations.has_value()) {
        GSPLAT_CHECK_INPUT(dLossDCompensations.value());
        assert(compensations.has_value());
    }

    const uint32_t N            = means.size(0);              // number of gaussians
    const uint32_t C            = worldToCamMatrices.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor dLossDMeans = torch::zeros_like(means);
    torch::Tensor dLossDCovars, dLossDQuats, dLossDScales; // optional
    if (covars.has_value()) {
        dLossDCovars = torch::zeros_like(covars.value());
    } else {
        dLossDQuats  = torch::zeros_like(quats);
        dLossDScales = torch::zeros_like(scales);
    }
    torch::Tensor dLossDWorldToCamMatrices;
    if (worldToCamMatricesRequiresGrad) {
        dLossDWorldToCamMatrices = torch::zeros_like(worldToCamMatrices);
    }
    if (C && N) {
        if (ortho) {
            projectionBackwardKernel<float, true>
                <<<(C * N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    C,
                    N,
                    means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(),
                    projectionMatrices.data_ptr<float>(),
                    imageWidth,
                    imageHeight,
                    eps2d,
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<float>(),
                    compensations.has_value() ? compensations.value().data_ptr<float>() : nullptr,
                    dLossDMeans2d.data_ptr<float>(),
                    dLossDDepths.data_ptr<float>(),
                    dLossDConics.data_ptr<float>(),
                    dLossDCompensations.has_value() ? dLossDCompensations.value().data_ptr<float>()
                                                    : nullptr,
                    dLossDMeans.data_ptr<float>(),
                    covars.has_value() ? dLossDCovars.data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : dLossDQuats.data_ptr<float>(),
                    covars.has_value() ? nullptr : dLossDScales.data_ptr<float>(),
                    worldToCamMatricesRequiresGrad ? dLossDWorldToCamMatrices.data_ptr<float>()
                                                   : nullptr);
        } else {
            projectionBackwardKernel<float, false>
                <<<(C * N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    C,
                    N,
                    means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(),
                    projectionMatrices.data_ptr<float>(),
                    imageWidth,
                    imageHeight,
                    eps2d,
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<float>(),
                    compensations.has_value() ? compensations.value().data_ptr<float>() : nullptr,
                    dLossDMeans2d.data_ptr<float>(),
                    dLossDDepths.data_ptr<float>(),
                    dLossDConics.data_ptr<float>(),
                    dLossDCompensations.has_value() ? dLossDCompensations.value().data_ptr<float>()
                                                    : nullptr,
                    dLossDMeans.data_ptr<float>(),
                    covars.has_value() ? dLossDCovars.data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : dLossDQuats.data_ptr<float>(),
                    covars.has_value() ? nullptr : dLossDScales.data_ptr<float>(),
                    worldToCamMatricesRequiresGrad ? dLossDWorldToCamMatrices.data_ptr<float>()
                                                   : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (outNormalizeddLossdMeans2dNormAccum.has_value()) {
            float *outNormalizeddLossdMeans2dNormAccumPtr =
                outNormalizeddLossdMeans2dNormAccum.value().data_ptr<float>();
            int32_t *outGradientStepCountsPtr = outGradientStepCounts.value().data_ptr<int32_t>();
            constexpr size_t NUM_GRAD_STATE_THREADS = 1024;
            if (outNormalizedMaxRadiiAccum.has_value()) {
                int32_t *outNormalizedMaxRadiiAccumPtr =
                    outNormalizedMaxRadiiAccum.value().data_ptr<int32_t>();
                computeGradientState<float, true>
                    <<<(N + NUM_GRAD_STATE_THREADS - 1) / NUM_GRAD_STATE_THREADS,
                       NUM_GRAD_STATE_THREADS,
                       0,
                       stream>>>(C,
                                 N,
                                 imageWidth,
                                 imageHeight,
                                 radii.data_ptr<int32_t>(),
                                 dLossDMeans2d.data_ptr<float>(),
                                 outNormalizeddLossdMeans2dNormAccumPtr,
                                 outNormalizedMaxRadiiAccumPtr,
                                 outGradientStepCountsPtr);
            } else {
                computeGradientState<float, false>
                    <<<(N + NUM_GRAD_STATE_THREADS - 1) / NUM_GRAD_STATE_THREADS,
                       NUM_GRAD_STATE_THREADS,
                       0,
                       stream>>>(C,
                                 N,
                                 imageWidth,
                                 imageHeight,
                                 radii.data_ptr<int32_t>(),
                                 dLossDMeans2d.data_ptr<float>(),
                                 outNormalizeddLossdMeans2dNormAccumPtr,
                                 nullptr,
                                 outGradientStepCountsPtr);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
    return std::make_tuple(
        dLossDMeans, dLossDCovars, dLossDQuats, dLossDScales, dLossDWorldToCamMatrices);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionBackward<torch::kCPU>(
    // fwd inputs
    const torch::Tensor &means,                       // [N, 3]
    const torch::Tensor &quats,                       // [N, 4]
    const torch::Tensor &scales,                      // [N, 3]
    const torch::Tensor &worldToCamMatrices,          // [C, 4, 4]
    const torch::Tensor &projectionMatrices,          // [C, 3, 3]
    const at::optional<torch::Tensor> &compensations, // [N, 6] optional
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,  // [C, N]
    const torch::Tensor &conics, // [C, N, 3]
    // grad outputs
    const torch::Tensor &dLossDMeans2d,                     // [C, N, 2]
    const torch::Tensor &dLossDDepths,                      // [C, N]
    const torch::Tensor &dLossDConics,                      // [C, N, 3]
    const at::optional<torch::Tensor> &dLossDCompensations, // [C, N] optional
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho,
    at::optional<torch::Tensor> outNormalizeddLossdMeans2dNormAccum,
    at::optional<torch::Tensor> outNormalizedMaxRadiiAccum,
    at::optional<torch::Tensor> outGradientStepCounts) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
