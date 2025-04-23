// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include "GaussianUtils.cuh"
#include <detail/ops/Ops.h>
#include <detail/utils/cuda/Utils.cuh>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

template <typename T, bool Ortho> struct ProjectionForward {
    using Mat3 = nanovdb::math::Mat3<T>;
    using Vec3 = nanovdb::math::Vec3<T>;
    using Vec4 = nanovdb::math::Vec4<T>;
    using Mat2 = nanovdb::math::Mat2<T>;
    using Vec2 = nanovdb::math::Vec2<T>;

    const int64_t C;
    const int64_t N;

    const int32_t mImageWidth;
    const int32_t mImageHeight;
    const T       mEps2d;
    const T       mNearPlane;
    const T       mFarPlane;
    const T       mRadiusClip;

    // TODO: We don't support raw covariances but we could
    // fvdb::TorchRAcc64<T, 2> mCovarsAcc;             // [N, 6] optional

    // Inputs
    fvdb::TorchRAcc64<T, 2> mMeansAcc;              // [N, 3]
    fvdb::TorchRAcc64<T, 2> mQuatsAcc;              // [N, 4]
    fvdb::TorchRAcc64<T, 2> mScalesAcc;             // [N, 3]
    fvdb::TorchRAcc32<T, 3> mWorldToCamMatricesAcc; // [C, 4, 4]
    fvdb::TorchRAcc32<T, 3> mProjectionMatricesAcc; // [C, 3, 3]

    // Outputs
    fvdb::TorchRAcc64<int32_t, 2> mOutRadiiAcc;   // [C, N]
    fvdb::TorchRAcc64<T, 3>       mOutMeans2dAcc; // [C, N, 2]
    fvdb::TorchRAcc64<T, 2>       mOutDepthsAcc;  // [C, N]
    fvdb::TorchRAcc64<T, 3>       mOutConicsAcc;  // [C, N, 3]

    // Tensor accessors are not default constructible so this needs to be a pointer.
    // This is okay since we allocate the memory and know the striding apriori.
    T *__restrict__ mOutCompensationsAcc; // [C, N] optional

    Mat3 *__restrict__ projectionMatsShared    = nullptr;
    Mat3 *__restrict__ worldToCamRotMatsShared = nullptr;
    Vec3 *__restrict__ worldToCamTranslation   = nullptr;

    ProjectionForward(const int64_t imageWidth, const int64_t imageHeight, const T eps2d,
                      const T nearPlane, const T farPlane, const T radiusClip,
                      const bool calcCompensations, const torch::Tensor &means, // [N, 3]
                      const torch::Tensor &quats,                               // [N, 4]
                      const torch::Tensor &scales,                              // [N, 3]
                      const torch::Tensor &worldToCamMatrices,                  // [C, 4, 4]
                      const torch::Tensor &projectionMatrices,                  // [C, 3, 3]
                      torch::Tensor        outRadii,                            // [C, N]
                      torch::Tensor        outMeans2d,                          // [C, N, 2]
                      torch::Tensor        outDepths,                           // [C, N]
                      torch::Tensor        outConics,                           // [C, N, 3]
                      torch::Tensor        outCompensations)                           // [C, N] optional
        : C(worldToCamMatrices.size(0)), N(means.size(0)), mImageWidth(imageWidth),
          mImageHeight(imageHeight), mEps2d(eps2d), mNearPlane(nearPlane), mFarPlane(farPlane),
          mRadiusClip(radiusClip),
          mMeansAcc(means.packed_accessor64<T, 2, torch::RestrictPtrTraits>()),
          mQuatsAcc(quats.packed_accessor64<T, 2, torch::RestrictPtrTraits>()),
          mScalesAcc(scales.packed_accessor64<T, 2, torch::RestrictPtrTraits>()),
          mWorldToCamMatricesAcc(
              worldToCamMatrices.packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          mProjectionMatricesAcc(
              projectionMatrices.packed_accessor32<T, 3, torch::RestrictPtrTraits>()),
          mOutRadiiAcc(outRadii.packed_accessor64<int32_t, 2, torch::RestrictPtrTraits>()),
          mOutMeans2dAcc(outMeans2d.packed_accessor64<T, 3, torch::RestrictPtrTraits>()),
          mOutDepthsAcc(outDepths.packed_accessor64<T, 2, torch::RestrictPtrTraits>()),
          mOutConicsAcc(outConics.packed_accessor64<T, 3, torch::RestrictPtrTraits>()),
          mOutCompensationsAcc(outCompensations.defined() ? outCompensations.data_ptr<T>()
                                                          : nullptr) {
        TORCH_CHECK_VALUE(means.is_cuda(), "means must be a CUDA tensor");
        TORCH_CHECK_VALUE(quats.is_cuda(), "quats must be a CUDA tensor");
        TORCH_CHECK_VALUE(scales.is_cuda(), "scales must be a CUDA tensor");
        TORCH_CHECK_VALUE(worldToCamMatrices.is_cuda(), "worldToCamMatrices must be a CUDA tensor");
        TORCH_CHECK_VALUE(projectionMatrices.is_cuda(), "projectionMatrices must be a CUDA tensor");

        mMeansAcc  = means.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
        mQuatsAcc  = quats.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
        mScalesAcc = scales.packed_accessor64<T, 2, torch::RestrictPtrTraits>();
        mWorldToCamMatricesAcc =
            worldToCamMatrices.packed_accessor32<T, 3, torch::RestrictPtrTraits>();
        mProjectionMatricesAcc =
            projectionMatrices.packed_accessor32<T, 3, torch::RestrictPtrTraits>();
    }

    inline __device__ Mat3
    computeCovarianceMatrix(int64_t gid) const {
        // TODO: Support raw covariances. Leaving this commented out for now.
        //     const auto covarAcc = mCovarsAcc[gid];
        //     return Mat3(covarAcc[0], covarAcc[1], covarAcc[2], // 1st row
        //                 covarAcc[1], covarAcc[3], covarAcc[4], // 2nd row
        //                 covarAcc[2], covarAcc[4], covarAcc[5]  // 3rd row
        //     );
        const auto quatAcc  = mQuatsAcc[gid];
        const auto scaleAcc = mScalesAcc[gid];
        return quatAndScaleToCovariance<T>(Vec4(quatAcc[0], quatAcc[1], quatAcc[2], quatAcc[3]),
                                           Vec3(scaleAcc[0], scaleAcc[1], scaleAcc[2]));
    }

    inline __device__ void
    projectionForward(int idx) {
        if (idx >= C * N) {
            return;
        }
        const auto cid = idx / N; // camera id
        const auto gid = idx % N; // gaussian id

        const Mat3 &projectionMatrix    = projectionMatsShared[cid];
        const Mat3 &worldToCamRotMatrix = worldToCamRotMatsShared[cid];
        const Vec3 &worldToCamTrans     = worldToCamTranslation[cid];

        // transform Gaussian center to camera space
        const Vec3 meanWorldSpace(mMeansAcc[gid][0], mMeansAcc[gid][1], mMeansAcc[gid][2]);
        const nanovdb::math::Vec3<T> meansCamSpace =
            transformPointWorldToCam(worldToCamRotMatrix, worldToCamTrans, meanWorldSpace);
        if (meansCamSpace[2] < mNearPlane || meansCamSpace[2] > mFarPlane) {
            mOutRadiiAcc[cid][gid] = 0;
            return;
        }

        // transform Gaussian covariance to camera space
        const Mat3 covar         = computeCovarianceMatrix(gid);
        const Mat3 covarCamSpace = transformCovarianceWorldToCam(worldToCamRotMatrix, covar);

        // camera projection
        const T fx = projectionMatrix[0][0], cx = projectionMatrix[0][2],
                fy = projectionMatrix[1][1], cy = projectionMatrix[1][2];
        auto [covar2d, mean2d] = [&]() {
            if constexpr (Ortho) {
                return projectGaussianOrthographic<T>(meansCamSpace, covarCamSpace, fx, fy, cx, cy,
                                                      mImageWidth, mImageHeight);
            } else {
                return projectGaussianPerspective<T>(meansCamSpace, covarCamSpace, fx, fy, cx, cy,
                                                     mImageWidth, mImageHeight);
            }
        }();

        T       compensation;
        const T det = add_blur(mEps2d, covar2d, compensation);
        if (det <= 0.f) {
            mOutRadiiAcc[cid][gid] = 0;
            return;
        }

        const Mat2 covar2dInverse = covar2d.inverse();

        // take 3 sigma as the radius
        const T b      = 0.5f * (covar2d[0][0] + covar2d[1][1]);
        const T v1     = b + sqrt(max(0.01f, b * b - det));
        const T radius = ceil(3.f * sqrt(v1));

        if (radius <= mRadiusClip) {
            mOutRadiiAcc[cid][gid] = 0;
            return;
        }

        // Mask out gaussians outside the image region
        if (mean2d[0] + radius <= 0 || mean2d[0] - radius >= mImageWidth ||
            mean2d[1] + radius <= 0 || mean2d[1] - radius >= mImageHeight) {
            mOutRadiiAcc[cid][gid] = 0;
            return;
        }

        // Write outputs
        mOutRadiiAcc[cid][gid]      = int32_t(radius);
        mOutMeans2dAcc[cid][gid][0] = mean2d[0];
        mOutMeans2dAcc[cid][gid][1] = mean2d[1];
        mOutDepthsAcc[cid][gid]     = meansCamSpace[2];
        mOutConicsAcc[cid][gid][0]  = covar2dInverse[0][0];
        mOutConicsAcc[cid][gid][1]  = covar2dInverse[0][1];
        mOutConicsAcc[cid][gid][2]  = covar2dInverse[1][1];
        if (mOutCompensationsAcc != nullptr) {
            mOutCompensationsAcc[idx] = compensation;
        }
    }

    inline __device__ void
    loadCamerasIntoSharedMemory() {
        // Load projection matrices and world-to-camera matrices into shared memory
        extern __shared__ T sharedMemory[];
        projectionMatsShared    = reinterpret_cast<Mat3 *>(sharedMemory);
        worldToCamRotMatsShared = reinterpret_cast<Mat3 *>(sharedMemory + C * 9);
        worldToCamTranslation   = reinterpret_cast<Vec3 *>(sharedMemory + 2 * C * 9);

        if (threadIdx.x < C * (9 + 9 + 3)) {
            if (threadIdx.x < C * 9) {
                const auto camId   = threadIdx.x / 9;
                const auto entryId = threadIdx.x % 9;
                const auto rowId   = entryId / 3;
                const auto colId   = entryId % 3;
                projectionMatsShared[camId][rowId][colId] =
                    mProjectionMatricesAcc[camId][rowId][colId];
            } else if (threadIdx.x < 2 * C * 9) {
                const auto baseIdx = threadIdx.x - C * 9;
                const auto camId   = baseIdx / 9;
                const auto entryId = baseIdx % 9;
                const auto rowId   = entryId / 3;
                const auto colId   = entryId % 3;
                worldToCamRotMatsShared[camId][rowId][colId] =
                    mWorldToCamMatricesAcc[camId][rowId][colId];
            } else {
                const auto baseIdx                    = threadIdx.x - 2 * C * 9;
                const auto camId                      = baseIdx / 3;
                const auto entryId                    = baseIdx % 3;
                worldToCamTranslation[camId][entryId] = mWorldToCamMatricesAcc[camId][entryId][3];
            }
        }
    }
};

template <typename T, bool Ortho>
__global__ void
projectionForwardKernel(ProjectionForward<T, Ortho> projectionForward) {
    projectionForward.loadCamerasIntoSharedMemory();
    __syncthreads();

    // parallelize over C * N.
    const auto problemSize = projectionForward.C * projectionForward.N;
    const auto idx         = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < problemSize;
         idx += blockDim.x * gridDim.x) {
        projectionForward.projectionForward(idx);
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
    const int64_t imageWidth, const int64_t imageHeight, const float eps2d, const float nearPlane,
    const float farPlane, const float radiusClip, const bool calcCompensations, const bool ortho) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means));

    const auto           N      = means.size(0);              // number of gaussians
    const auto           C      = worldToCamMatrices.size(0); // number of cameras
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(means.device().index());

    torch::Tensor outRadii   = torch::empty({ C, N }, means.options().dtype(torch::kInt32));
    torch::Tensor outMeans2d = torch::empty({ C, N, 2 }, means.options());
    torch::Tensor outDepths  = torch::empty({ C, N }, means.options());
    torch::Tensor outConics  = torch::empty({ C, N, 3 }, means.options());
    torch::Tensor outCompensations;
    if (calcCompensations) {
        // we dont want NaN to appear in this tensor, so we zero intialize it
        outCompensations = torch::zeros({ C, N }, means.options());
    }

    if (N == 0 || C == 0) {
        // Early exit if there are no gaussians or cameras
        return std::make_tuple(outRadii, outMeans2d, outDepths, outConics, outCompensations);
    }

    using scalar_t = float;

    const size_t NUM_THREADS    = 256;
    const size_t NUM_BLOCKS     = C * N / NUM_THREADS + 1;
    const size_t SHARD_MEM_SIZE = C * (9 + 9 + 3) * sizeof(scalar_t);

    if (ortho) {
        ProjectionForward<scalar_t, true> projectionForward(
            imageWidth, imageHeight, eps2d, nearPlane, farPlane, radiusClip, calcCompensations,
            means, quats, scales, worldToCamMatrices, projectionMatrices, outRadii, outMeans2d,
            outDepths, outConics, outCompensations);
        projectionForwardKernel<scalar_t, true>
            <<<NUM_BLOCKS, NUM_THREADS, SHARD_MEM_SIZE, stream>>>(projectionForward);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
        ProjectionForward<scalar_t, false> projectionForward(
            imageWidth, imageHeight, eps2d, nearPlane, farPlane, radiusClip, calcCompensations,
            means, quats, scales, worldToCamMatrices, projectionMatrices, outRadii, outMeans2d,
            outDepths, outConics, outCompensations);
        projectionForwardKernel<scalar_t, false>
            <<<NUM_BLOCKS, NUM_THREADS, SHARD_MEM_SIZE, stream>>>(projectionForward);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return std::make_tuple(outRadii, outMeans2d, outDepths, outConics, outCompensations);
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchGaussianProjectionForward<torch::kCPU>(const torch::Tensor &means,              // [N, 3]
                                               const torch::Tensor &quats,              // [N, 4]
                                               const torch::Tensor &scales,             // [N, 3]
                                               const torch::Tensor &worldToCamMatrices, // [C, 4, 4]
                                               const torch::Tensor &projectionMatrices, // [C, 3, 3]
                                               const int64_t imageWidth, const int64_t imageHeight,
                                               const float eps2d, const float nearPlane,
                                               const float farPlane, const float radiusClip,
                                               const bool calcCompensations, const bool ortho) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

} // namespace ops
} // namespace detail
} // namespace fvdb
