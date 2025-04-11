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
projectionForwardKernel(const uint32_t C, const uint32_t N,
                        const T *__restrict__ means,              // [N, 3]
                        const T *__restrict__ covars,             // [N, 6] optional
                        const T *__restrict__ quats,              // [N, 4] optional
                        const T *__restrict__ scales,             // [N, 3] optional
                        const T *__restrict__ worldToCamMatrices, // [C, 4, 4]
                        const T *__restrict__ projectionMatrices, // [C, 3, 3]
                        const int32_t imageWidth, const int32_t imageHeight, const T eps2d,
                        const T nearPlane, const T farPlane, const T radiusClip,
                        // outputs
                        int32_t *__restrict__ radii,     // [C, N]
                        T *__restrict__ means2d,         // [C, N, 2]
                        T *__restrict__ depths,          // [C, N]
                        T *__restrict__ conics,          // [C, N, 3]
                        T *__restrict__ compensations) { // [C, N] optional
    // parallelize over C * N.
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= C * N) {
        return;
    }
    const uint32_t cid = idx / N; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += gid * 3;
    worldToCamMatrices += cid * 16;
    projectionMatrices += cid * 9;

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
        covars += gid * 6;
        covar = nanovdb::math::Mat3<T>(covars[0], covars[1], covars[2], // 1st row
                                       covars[1], covars[3], covars[4], // 2nd row
                                       covars[2], covars[4], covars[5]  // 3rd row
        );
    } else {
        // compute from quaternions and scales
        quats += gid * 4;
        scales += gid * 3;
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
dispatchGaussianProjectionForward<torch::kCUDA>(
    const torch::Tensor &means,              // [N, 3]
    const torch::Tensor &quats,              // [N, 4]
    const torch::Tensor &scales,             // [N, 3]
    const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
    const torch::Tensor &projectionMatrices, // [C, 3, 3]
    const uint32_t imageWidth, const uint32_t imageHeight, const float eps2d, const float nearPlane,
    const float farPlane, const float radiusClip, const bool calcCompensations, const bool ortho) {
    // These are supported by the underlying kernel, but they are not exposed
    const at::optional<torch::Tensor> &covars = std::nullopt;
    // const bool                         calc_compensations = false;

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

    const uint32_t       N      = means.size(0);              // number of gaussians
    const uint32_t       C      = worldToCamMatrices.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor radii   = torch::empty({ C, N }, means.options().dtype(torch::kInt32));
    torch::Tensor means2d = torch::empty({ C, N, 2 }, means.options());
    torch::Tensor depths  = torch::empty({ C, N }, means.options());
    torch::Tensor conics  = torch::empty({ C, N, 3 }, means.options());
    torch::Tensor compensations;
    if (calcCompensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        compensations = torch::zeros({ C, N }, means.options());
    }
    if (C && N) {
        if (ortho) {
            projectionForwardKernel<float, true>
                <<<(C * N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    C, N, means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(), projectionMatrices.data_ptr<float>(),
                    imageWidth, imageHeight, eps2d, nearPlane, farPlane, radiusClip,
                    radii.data_ptr<int32_t>(), means2d.data_ptr<float>(), depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calcCompensations ? compensations.data_ptr<float>() : nullptr);
        } else {
            projectionForwardKernel<float, false>
                <<<(C * N + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS, 0, stream>>>(
                    C, N, means.data_ptr<float>(),
                    covars.has_value() ? covars.value().data_ptr<float>() : nullptr,
                    covars.has_value() ? nullptr : quats.data_ptr<float>(),
                    covars.has_value() ? nullptr : scales.data_ptr<float>(),
                    worldToCamMatrices.data_ptr<float>(), projectionMatrices.data_ptr<float>(),
                    imageWidth, imageHeight, eps2d, nearPlane, farPlane, radiusClip,
                    radii.data_ptr<int32_t>(), means2d.data_ptr<float>(), depths.data_ptr<float>(),
                    conics.data_ptr<float>(),
                    calcCompensations ? compensations.data_ptr<float>() : nullptr);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(radii, means2d, depths, conics, compensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForward<torch::kCPU>(const torch::Tensor &means,              // [N, 3]
                                               const torch::Tensor &quats,              // [N, 4]
                                               const torch::Tensor &scales,             // [N, 3]
                                               const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                               const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                               const uint32_t       imageWidth,
                                               const uint32_t imageHeight, const float eps2d,
                                               const float nearPlane, const float farPlane,
                                               const float radiusClip, const bool calcCompensations,
                                               const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
