// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH
#define FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH

#include <fvdb/Config.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/GridBatchImpl.h>
#include <fvdb/detail/utils/cuda/GridDim.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAStream.h>

namespace fvdb {

namespace _private {

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachLeafPrivateUse1Kernel(int64_t leafChannelCount,
                             int64_t leafChannelOffset,
                             fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                             const int32_t channelsPerLeaf,
                             Func func,
                             Args... args) {
    for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < leafChannelCount;
         idx += blockDim.x * gridDim.x) {
        const int64_t leafChannelIdx = idx + leafChannelOffset;
        const int64_t cumLeafIdx     = leafChannelIdx / channelsPerLeaf;
        const int32_t channelIdx     = leafChannelIdx % channelsPerLeaf;

        const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

        func(batchIdx, leafIdx, channelIdx, grid, args...);
    }
}

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachVoxelPrivateUse1Kernel(int64_t leafVoxelChannelCount,
                              int64_t leafVoxelChannelOffset,
                              fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                              int64_t channelsPerVoxel,
                              Func func,
                              Args... args) {
    constexpr auto VOXELS_PER_LEAF =
        nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;

    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < leafVoxelChannelCount;
         idx += blockDim.x * gridDim.x) {
        const int64_t leafVoxelChannelIdx   = idx + leafVoxelChannelOffset;
        const int64_t voxelsChannelsPerLeaf = VOXELS_PER_LEAF * channelsPerVoxel;

        const int64_t cumLeafIdx = leafVoxelChannelIdx / voxelsChannelsPerLeaf;
        const int64_t leafVoxelIdx =
            (leafVoxelChannelIdx - cumLeafIdx * voxelsChannelsPerLeaf) / channelsPerVoxel;
        const int64_t channelIdx = leafVoxelChannelIdx - cumLeafIdx * voxelsChannelsPerLeaf -
                                   leafVoxelIdx * channelsPerVoxel;

        const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

        func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
    }
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void
forEachJaggedElementChannelPrivateUse1Kernel(int64_t numel,
                                             int64_t offset,
                                             JaggedRAcc32<ScalarT, NDIMS> jaggedAcc,
                                             int64_t channelsPerElement,
                                             Func func,
                                             Args... args) {
    for (int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
         idx < numel * channelsPerElement;
         idx += blockDim.x * gridDim.x) {
        const int64_t elementIdx      = (idx + offset) / channelsPerElement;
        const fvdb::JIdxType batchIdx = jaggedAcc.batchIdx(elementIdx);
        const int64_t channelIdx      = (idx + offset) % channelsPerElement;

        func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
    }
}

} // namespace _private

template <typename GridType, typename Func, typename... Args>
void
forEachLeafPrivateUse1(int64_t numChannels,
                       const fvdb::detail::GridBatchImpl &batchHdl,
                       Func func,
                       Args... args) {
    TORCH_CHECK(batchHdl.device().is_privateuseone(), "Grid batch must be on a PrivateUse1 device");

    const int64_t leafCount = batchHdl.totalLeaves();

    auto batchAccessor = batchHdl.deviceAccessor<GridType>();

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        auto deviceLeafCount =
            (leafCount + c10::cuda::device_count() - 1) / c10::cuda::device_count();
        const auto deviceLeafOffset = deviceLeafCount * deviceId;
        deviceLeafCount             = std::min(deviceLeafCount, leafCount - deviceLeafOffset);

        const auto deviceLeafChannelCount  = deviceLeafCount * numChannels;
        const auto deviceLeafChannelOffset = deviceLeafOffset * numChannels;

        constexpr int64_t kNumThreads = 256;
        const int64_t deviceNumBlocks = GET_BLOCKS(deviceLeafChannelCount, kNumThreads);
        TORCH_INTERNAL_ASSERT(deviceNumBlocks <
                                  static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
                              "Too many blocks in forEachLeafPrivateUse1");
        if (deviceNumBlocks > 0) {
            _private::forEachLeafPrivateUse1Kernel<<<deviceNumBlocks, kNumThreads, 0, stream>>>(
                deviceLeafChannelCount,
                deviceLeafChannelOffset,
                batchAccessor,
                numChannels,
                func,
                args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }
}

template <typename GridType, typename Func, typename... Args>
void
forEachVoxelPrivateUse1(int64_t numChannels,
                        const fvdb::detail::GridBatchImpl &batchHdl,
                        Func func,
                        Args... args) {
    TORCH_CHECK(batchHdl.device().is_privateuseone(), "Grid batch must be on a PrivateUse1 device");
    TORCH_CHECK(!fvdb::Config::global().ultraSparseAccelerationEnabled());

    const int64_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const int64_t leafCount = batchHdl.totalLeaves();

    auto batchAccessor = batchHdl.deviceAccessor<GridType>();

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(deviceId).stream();

        auto deviceLeafCount =
            (leafCount + c10::cuda::device_count() - 1) / c10::cuda::device_count();
        const auto deviceLeafOffset = deviceLeafCount * deviceId;
        deviceLeafCount             = std::min(deviceLeafCount, leafCount - deviceLeafOffset);

        const auto deviceLeafVoxelChannelCount  = deviceLeafCount * VOXELS_PER_LEAF * numChannels;
        const auto deviceLeafVoxelChannelOffset = deviceLeafOffset * VOXELS_PER_LEAF * numChannels;

        const int64_t kNumThreads     = 256;
        const int64_t deviceNumBlocks = GET_BLOCKS(deviceLeafVoxelChannelCount, kNumThreads);
        TORCH_INTERNAL_ASSERT(deviceNumBlocks <
                                  static_cast<int64_t>(std::numeric_limits<unsigned int>::max()),
                              "Too many blocks in forEachVoxelPrivateUse1");

        if (deviceNumBlocks > 0) {
            _private::forEachVoxelPrivateUse1Kernel<<<deviceNumBlocks, kNumThreads, 0, stream>>>(
                deviceLeafVoxelChannelCount,
                deviceLeafVoxelChannelOffset,
                batchAccessor,
                numChannels,
                func,
                args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }
}

template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachJaggedElementChannelPrivateUse1(int64_t numChannels,
                                       const JaggedTensor &jaggedTensor,
                                       Func func,
                                       Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_privateuseone(),
                "JaggedTensor must be on a PrivateUse1 device");

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        cudaStream_t stream    = c10::cuda::getCurrentCUDAStream(deviceId).stream();
        auto deviceNumElements = (jaggedTensor.element_count() + c10::cuda::device_count() - 1) /
                                 c10::cuda::device_count();
        const auto deviceOffset = deviceNumElements * deviceId;
        deviceNumElements =
            std::min(deviceNumElements, jaggedTensor.element_count() - deviceOffset);

        const int64_t kNumThreads     = 256;
        const int64_t deviceNumBlocks = GET_BLOCKS(deviceNumElements, kNumThreads);
        if (deviceNumBlocks > 0) {
            _private::forEachJaggedElementChannelPrivateUse1Kernel<NDIMS, ScalarT, Func, Args...>
                <<<deviceNumBlocks, kNumThreads, 0, stream>>>(
                    deviceNumElements,
                    deviceOffset,
                    jaggedTensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                    numChannels,
                    func,
                    args...);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }

    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        c10::cuda::getCurrentCUDAStream(deviceId).synchronize();
    }
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_FOREACHPRIVATEUSE1_CUH
