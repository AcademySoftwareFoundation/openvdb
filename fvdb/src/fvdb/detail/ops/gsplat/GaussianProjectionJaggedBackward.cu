// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/ops/gsplat/GaussianMacros.cuh>
#include <fvdb/detail/ops/gsplat/GaussianUtils.cuh>
#include <fvdb/detail/ops/gsplat/GaussianWarpUtils.cuh>

#include <ATen/cuda/Atomic.cuh>

#include <cooperative_groups.h>

constexpr int NUM_THREADS = 256;

namespace fvdb {
namespace detail {
namespace ops {

namespace cg = cooperative_groups;

template <typename T, bool Ortho>
__global__ void
jaggedProjectionBackwardKernel(
    // fwd inputs
    const uint32_t B,
    const int64_t N,
    const int64_t *__restrict__ gSizes,       // [B]
    const int64_t *__restrict__ cSizes,       // [B]
    const int64_t *__restrict__ gIndexPtr,    // [B] start indices
    const int64_t *__restrict__ cIndexPtr,    // [B] start indices
    const int64_t *__restrict__ nIndexPtr,    // [B] start indices
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
    const int32_t *__restrict__ radii,   // [N]
    const T *__restrict__ conics,        // [N, 3]
    const T *__restrict__ compensations, // [N] optional
    // grad outputs
    const T *__restrict__ dLossDMeans2d,       // [N, 2]
    const T *__restrict__ dLossDDepths,        // [N]
    const T *__restrict__ dLossDConics,        // [N, 3]
    const T *__restrict__ dLossDCompensations, // [N] optional
    // grad inputs
    T *__restrict__ outDLossDMeans,                // [N, 3]
    T *__restrict__ outDLossDCovars,               // [N, 6] optional
    T *__restrict__ outDLossDQuats,                // [N, 4] optional
    T *__restrict__ outDLossDScales,               // [N, 3] optional
    T *__restrict__ outDLossDWorldToCamMatrices) { // [C, 4, 4] optional
    // parallelize over N.
    const uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N || radii[idx] <= 0) {
        return;
    }

    // TODO: too many global memory accesses.
    const int64_t bId      = binSearch(nIndexPtr, B, static_cast<int64_t>(idx)); // batch id
    const int64_t idxLocal = idx - nIndexPtr[bId];      // local elem idx within Ci * Ni
    const int64_t cIdLocal = idxLocal / gSizes[bId];    // local camera id within Ci
    const int64_t gIdLocal = idxLocal % gSizes[bId];    // local gaussian id within Ni
    const int64_t cId      = cIdLocal + cIndexPtr[bId]; // camera id
    const int64_t gId      = gIdLocal + gIndexPtr[bId]; // gaussian id

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
dispatchGaussianProjectionJaggedBackward<torch::kCUDA>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,              // [N, 4] optional
    const torch::Tensor &scales,             // [N, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
    const torch::Tensor &projectionMatrices, // [C, 3, 3]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    // fwd outputs
    const torch::Tensor &radii,  // [N]
    const torch::Tensor &conics, // [N, 3]
    // grad outputs
    const torch::Tensor &dLossDMeans2d, // [N, 2]
    const torch::Tensor &dLossDDepths,  // [N]
    const torch::Tensor &dLossDConics,  // [N, 3]
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho) {
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars              = std::nullopt;
    const at::optional<torch::Tensor> &compensations       = std::nullopt;
    const at::optional<torch::Tensor> &dLossDCompensations = std::nullopt;

    GSPLAT_DEVICE_GUARD(means);
    GSPLAT_CHECK_INPUT(gSizes);
    GSPLAT_CHECK_INPUT(means);
    if (covars.has_value()) {
        GSPLAT_CHECK_INPUT(covars.value());
    } else {
        GSPLAT_CHECK_INPUT(quats);
        GSPLAT_CHECK_INPUT(scales);
    }
    GSPLAT_CHECK_INPUT(cSizes);
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

    // TODO: use inclusive sum
    const uint32_t B        = gSizes.size(0);
    const int64_t N         = dLossDMeans2d.size(0);
    torch::Tensor cIndexPtr = torch::cumsum(cSizes, 0, torch::kInt64) - cSizes;
    torch::Tensor gIndexPtr = torch::cumsum(gSizes, 0, torch::kInt64) - gSizes;
    torch::Tensor nSizes    = cSizes * gSizes; // element size = Ci * Ni
    torch::Tensor nIndexPtr = torch::cumsum(nSizes, 0, torch::kInt64) - nSizes;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor outDLossDMeans = torch::zeros_like(means);
    torch::Tensor outDLossDCovars, outDLossDQuats, outDLossDScales; // optional
    if (covars.has_value()) {
        outDLossDCovars = torch::zeros_like(covars.value());
    } else {
        outDLossDQuats  = torch::zeros_like(quats);
        outDLossDScales = torch::zeros_like(scales);
    }
    torch::Tensor outDLossDWorldToCamMatrices;
    if (worldToCamMatricesRequiresGrad) {
        outDLossDWorldToCamMatrices = torch::zeros_like(worldToCamMatrices);
    }
    if (N) {
        if (ortho) {
            jaggedProjectionBackwardKernel<float, true>
                <<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B,
                    N,
                    gSizes.data_ptr<int64_t>(),
                    cSizes.data_ptr<int64_t>(),
                    gIndexPtr.data_ptr<int64_t>(),
                    cIndexPtr.data_ptr<int64_t>(),
                    nIndexPtr.data_ptr<int64_t>(),
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
                    outDLossDMeans.data_ptr<float>(),
                    covars.has_value() ? outDLossDCovars.data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : outDLossDQuats.data_ptr<float>(),
                    covars.has_value() ? nullptr : outDLossDScales.data_ptr<float>(),
                    worldToCamMatricesRequiresGrad ? outDLossDWorldToCamMatrices.data_ptr<float>()
                                                   : nullptr);
        } else {
            jaggedProjectionBackwardKernel<float, false>
                <<<(N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B,
                    N,
                    gSizes.data_ptr<int64_t>(),
                    cSizes.data_ptr<int64_t>(),
                    gIndexPtr.data_ptr<int64_t>(),
                    cIndexPtr.data_ptr<int64_t>(),
                    nIndexPtr.data_ptr<int64_t>(),
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
                    outDLossDMeans.data_ptr<float>(),
                    covars.has_value() ? outDLossDCovars.data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : outDLossDQuats.data_ptr<float>(),
                    covars.has_value() ? nullptr : outDLossDScales.data_ptr<float>(),
                    worldToCamMatricesRequiresGrad ? outDLossDWorldToCamMatrices.data_ptr<float>()
                                                   : nullptr);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(outDLossDMeans,
                           outDLossDCovars,
                           outDLossDQuats,
                           outDLossDScales,
                           outDLossDWorldToCamMatrices);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedBackward<torch::kCPU>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,              // [N, 4] optional
    const torch::Tensor &scales,             // [N, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
    const torch::Tensor &projectionMatrices, // [C, 3, 3]
    const uint32_t imageWidth,
    const uint32_t imageHeight,
    const float eps2d,
    const torch::Tensor &radii,         // [N]
    const torch::Tensor &conics,        // [N, 3]
    const torch::Tensor &dLossDMeans2d, // [N, 2]
    const torch::Tensor &dLossDDepths,  // [N]
    const torch::Tensor &dLossDConics,  // [N, 3]
    const bool worldToCamMatricesRequiresGrad,
    const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
