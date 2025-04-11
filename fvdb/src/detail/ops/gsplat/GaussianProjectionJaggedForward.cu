// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GsplatUtils.cuh"
#include <detail/ops/Ops.h>

#include <ATen/cuda/Atomic.cuh>

constexpr int NUM_THREADS = 256;

namespace fvdb {
namespace detail {
namespace ops {

template <typename T, bool Ortho>
__global__ void
jaggedProjectionForwardKernel(const uint32_t B, const int64_t M,
                              const int64_t *__restrict__ gSizes,       // [B]
                              const int64_t *__restrict__ cSizes,       // [B]
                              const int64_t *__restrict__ gIndex,       // [B] start indices
                              const int64_t *__restrict__ cIndex,       // [B] start indices
                              const int64_t *__restrict__ nIndex,       // [B] start indices
                              const T *__restrict__ means,              // [M, 3]
                              const T *__restrict__ covars,             // [M, 6] optional
                              const T *__restrict__ quats,              // [M, 4] optional
                              const T *__restrict__ scales,             // [M, 3] optional
                              const T *__restrict__ worldToCamMatrices, // [BC, 4, 4]
                              const T *__restrict__ projectionMatrices, // [BC, 3, 3]
                              const int32_t imageWidth, const int32_t imageHeight, const T eps2d,
                              const T nearPlane, const T farPlane, const T radiusClip,
                              // outputs
                              int32_t *__restrict__ radii,     // [M]
                              T *__restrict__ means2d,         // [M, 2]
                              T *__restrict__ depths,          // [M]
                              T *__restrict__ conics,          // [M, 3]
                              T *__restrict__ compensations) { // [M] optional
    // parallelize over M.
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) {
        return;
    }

    // TODO: too many global memory accesses.
    const int64_t bId      = bin_search(nIndex, B, static_cast<int64_t>(idx)); // batch id
    const int64_t idxLocal = idx - nIndex[bId];      // local elem idx within Ci * Ni
    const int64_t cidLocal = idxLocal / gSizes[bId]; // local camera id within Ci
    const int64_t gidLocal = idxLocal % gSizes[bId]; // local gaussian id within Ni
    const int64_t cId      = cidLocal + cIndex[bId]; // camera id
    const int64_t gId      = gidLocal + gIndex[bId]; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gId * 3;
    worldToCamMatrices += cId * 16;
    projectionMatrices += cId * 9;

    // input is row-major
    const nanovdb::math::Mat3<T> R(
        worldToCamMatrices[0], worldToCamMatrices[1], worldToCamMatrices[2],   // 1st row
        worldToCamMatrices[4], worldToCamMatrices[5], worldToCamMatrices[6],   // 2nd row
        worldToCamMatrices[8], worldToCamMatrices[9], worldToCamMatrices[10]); // 3rd row
    const nanovdb::math::Vec3<T> t(worldToCamMatrices[3], worldToCamMatrices[7],
                                   worldToCamMatrices[11]);
    // transform Gaussian center to camera space
    const nanovdb::math::Vec3<T> meansCamSpace =
        transformPointWorldToCam(R, t, nanovdb::math::Vec3<T>(means[0], means[1], means[2]));

    if (meansCamSpace[2] < nearPlane || meansCamSpace[2] > farPlane) {
        radii[idx] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    nanovdb::math::Mat3<T> covar;
    if (covars != nullptr) {
        covars += gId * 6;
        covar = nanovdb::math::Mat3<T>(covars[0], covars[1], covars[2], // 1st row
                                       covars[1], covars[3], covars[4], // 2nd row
                                       covars[2], covars[4], covars[5]  // 3rd row
        );
    } else {
        // compute from quaternions and scales
        quats += gId * 4;
        scales += gId * 3;
        covar = quatAndScaleToCovariance<T>(
            nanovdb::math::Vec4<T>(quats[0], quats[1], quats[2], quats[3]),
            nanovdb::math::Vec3<T>(scales[0], scales[1], scales[2]));
    }
    const nanovdb::math::Mat3<T> covarCamSpace = transformCovarianceWorldToCam(R, covar);

    // camera projection
    const T fx = projectionMatrices[0], cx = projectionMatrices[2], fy = projectionMatrices[4],
            cy             = projectionMatrices[5];
    auto [covar2d, mean2d] = [&]() {
        if constexpr (Ortho) {
            return ortho_proj<T>(meansCamSpace, covarCamSpace, fx, fy, cx, cy, imageWidth,
                                 imageHeight);
        } else {
            return persp_proj<T>(meansCamSpace, covarCamSpace, fx, fy, cx, cy, imageWidth,
                                 imageHeight);
        }
    }();

    T       compensation;
    const T det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    const nanovdb::math::Mat2<T> covar2dInverse = covar2d.inverse();

    // take 3 sigma as the radius (non differentiable)
    const T b      = 0.5f * (covar2d[0][0] + covar2d[1][1]);
    const T v1     = b + sqrt(max(0.01f, b * b - det));
    const T radius = ceil(3.f * sqrt(v1));
    // T v2 = b - sqrt(max(0.1f, b * b - det));
    // T radius = ceil(3.f * sqrt(max(v1, v2)));

    if (radius <= radiusClip) {
        radii[idx] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d[0] + radius <= 0 || mean2d[0] - radius >= imageWidth || mean2d[1] + radius <= 0 ||
        mean2d[1] - radius >= imageHeight) {
        radii[idx] = 0;
        return;
    }

    // write to outputs
    radii[idx]           = (int32_t)radius;
    means2d[idx * 2]     = mean2d[0];
    means2d[idx * 2 + 1] = mean2d[1];
    depths[idx]          = meansCamSpace[2];
    conics[idx * 3]      = covar2dInverse[0][0];
    conics[idx * 3 + 1]  = covar2dInverse[0][1];
    conics[idx * 3 + 2]  = covar2dInverse[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward<torch::kCUDA>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [M, 3]
    const torch::Tensor &quats,              // [M, 4] optional
    const torch::Tensor &scales,             // [M, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [BC, 4, 4]
    const torch::Tensor &projectionMatrices, // [BC, 3, 3]
    const uint32_t imageWidth, const uint32_t imageHeight, const float eps2d, const float nearPlane,
    const float farPlane, const float minRadius2d, const bool ortho) {
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars             = std::nullopt;
    constexpr bool                     calc_compensations = false;

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

    // TODO: use inclusive sum
    const uint32_t B      = gSizes.size(0);
    torch::Tensor  cIndex = torch::cumsum(cSizes, 0, torch::kInt64) - cSizes;
    torch::Tensor  gIndex = torch::cumsum(gSizes, 0, torch::kInt64) - gSizes;
    torch::Tensor  nSize  = cSizes * gSizes;            // element size = Ci * Ni
    torch::Tensor  nIndex = torch::cumsum(nSize, 0, torch::kInt64);
    const int64_t  M      = nIndex[-1].item<int64_t>(); // total number of elements
    nIndex                = nIndex - nSize;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor radii   = torch::empty({ M }, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({ M, 2 }, means.options());
    torch::Tensor depths  = torch::empty({ M }, means.options());
    torch::Tensor conics  = torch::empty({ M, 3 }, means.options());
    torch::Tensor compensations;
    if (calc_compensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({ M }, means.options());
    }
    if (M) {
        if (ortho) {
            jaggedProjectionForwardKernel<float, true>
                <<<(M + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B, M, gSizes.data_ptr<int64_t>(), cSizes.data_ptr<int64_t>(),
                    gIndex.data_ptr<int64_t>(), cIndex.data_ptr<int64_t>(),
                    nIndex.data_ptr<int64_t>(), means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(), projectionMatrices.data_ptr<float>(),
                    imageWidth, imageHeight, eps2d, nearPlane, farPlane, minRadius2d,
                    radii.data_ptr<int32_t>(), means2d.data_ptr<float>(), depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        } else {
            jaggedProjectionForwardKernel<float, false>
                <<<(M + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    B, M, gSizes.data_ptr<int64_t>(), cSizes.data_ptr<int64_t>(),
                    gIndex.data_ptr<int64_t>(), cIndex.data_ptr<int64_t>(),
                    nIndex.data_ptr<int64_t>(), means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(), projectionMatrices.data_ptr<float>(),
                    imageWidth, imageHeight, eps2d, nearPlane, farPlane, minRadius2d,
                    radii.data_ptr<int32_t>(), means2d.data_ptr<float>(), depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calc_compensations ? compensations.data_ptr<float>() : nullptr);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionJaggedForward<torch::kCPU>(
    const torch::Tensor &gSizes,             // [B] gaussian sizes
    const torch::Tensor &means,              // [M, 3]
    const torch::Tensor &quats,              // [M, 4] optional
    const torch::Tensor &scales,             // [M, 3] optional
    const torch::Tensor &cSizes,             // [B] camera sizes
    const torch::Tensor &worldToCamMatrices, // [BC, 4, 4]
    const torch::Tensor &projectionMatrices, // [BC, 3, 3]
    const uint32_t imageWidth, const uint32_t imageHeight, const float eps2d, const float nearPlane,
    const float farPlane, const float minRadius2d, const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
