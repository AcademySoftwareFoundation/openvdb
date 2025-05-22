// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_FOREACHCUDA_CUH
#define FVDB_DETAIL_UTILS_CUDA_FOREACHCUDA_CUH

#include <Config.h>
#include <JaggedTensor.h>
#include <detail/GridBatchImpl.h>
#include <detail/utils/cuda/Utils.cuh>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAStream.h>

namespace fvdb {

namespace _private {

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachLeafCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                      const bool returnIfOutOfRange,
                      const int32_t channelsPerLeaf,
                      Func func,
                      Args... args) {
    const uint64_t leafChannelIdx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (leafChannelIdx >= static_cast<uint64_t>(grid.totalLeaves()) * channelsPerLeaf) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, -1, grid, args...);
            return;
        }
    }

    const int64_t cumLeafIdx = static_cast<int64_t>(leafChannelIdx / channelsPerLeaf);
    const int32_t channelIdx = leafChannelIdx % channelsPerLeaf;

    const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, channelIdx, grid, args...);
}

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachLeafSingleGridCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
                                const bool returnIfOutOfRange,
                                const int32_t channelsPerLeaf,
                                const int32_t bidx,
                                Func func,
                                Args... args) {
    const typename nanovdb::NanoGrid<GridType> *gpuGrid = batchAccessor.grid(bidx);

    const uint64_t leafChannelIdx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (leafChannelIdx >= static_cast<uint64_t>(gpuGrid->tree().nodeCount(0)) * channelsPerLeaf) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(gpuGrid, -1, -1, args...);
            return;
        }
    }

    const int64_t leafIdx    = static_cast<int64_t>(leafChannelIdx / channelsPerLeaf);
    const int32_t channelIdx = static_cast<int32_t>(leafChannelIdx % channelsPerLeaf);

    func(gpuGrid, leafIdx, channelIdx, args...);
}

template <typename GridType>
__global__ void
voxelMetaIndexCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
                         TorchRAcc32<int64_t, 2> metaIndex) {
    constexpr int32_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const int64_t lvIdx = ((int64_t)blockIdx.x * (int64_t)blockDim.x) + threadIdx.x;

    if (lvIdx >= gridAccessor.totalLeaves() * VOXELS_PER_LEAF) {
        return;
    }

    const int64_t cumLeafIdx   = (lvIdx / VOXELS_PER_LEAF);
    const int64_t leafVoxelIdx = lvIdx % VOXELS_PER_LEAF;

    const int64_t batchIdx = gridAccessor.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx  = cumLeafIdx - gridAccessor.leafOffset(batchIdx);

    const nanovdb::NanoGrid<GridType> *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];

    if (leaf.isActive(leafVoxelIdx)) {
        const int64_t baseOffset = gridAccessor.voxelOffset(batchIdx);
        const int64_t idx        = baseOffset + (int64_t)leaf.getValue(leafVoxelIdx) - 1;

        metaIndex[idx][0] = batchIdx;
        metaIndex[idx][1] = leafIdx;
        metaIndex[idx][2] = leafVoxelIdx;
    }
}

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachVoxelWithMetaCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                               TorchRAcc32<int64_t, 2> metaIndex,
                               const bool returnIfOutOfRange,
                               const int64_t channelsPerVoxel,
                               Func func,
                               Args... args) {
    const int64_t vcIdx = ((int64_t)blockIdx.x * (int64_t)blockDim.x) + threadIdx.x;

    if (vcIdx >= grid.totalVoxels() * channelsPerVoxel) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, -1, -1, grid, args...);
            return;
        }
    }

    const int64_t voxelIdx   = vcIdx / channelsPerVoxel;
    const int64_t channelIdx = vcIdx % channelsPerVoxel;

    const int64_t batchIdx     = metaIndex[voxelIdx][0];
    const int64_t leafIdx      = metaIndex[voxelIdx][1];
    const int64_t leafVoxelIdx = metaIndex[voxelIdx][2];

    func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
}

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachVoxelCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                       const bool returnIfOutOfRange,
                       const int64_t channelsPerVoxel,
                       Func func,
                       Args... args) {
    constexpr uint64_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const uint64_t lvcIdx = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const uint64_t voxelsChannelsPerLeaf = VOXELS_PER_LEAF * channelsPerVoxel;

    if (lvcIdx >= static_cast<uint64_t>(grid.totalLeaves()) * voxelsChannelsPerLeaf) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, -1, -1, grid, args...);
            return;
        }
    }

    const int64_t cumLeafIdx = static_cast<int64_t>(lvcIdx / voxelsChannelsPerLeaf);
    const int64_t leafVoxelIdx =
        static_cast<int64_t>((lvcIdx - cumLeafIdx * voxelsChannelsPerLeaf) / channelsPerVoxel);
    const int64_t channelIdx = static_cast<int64_t>(lvcIdx - cumLeafIdx * voxelsChannelsPerLeaf -
                                                    leafVoxelIdx * channelsPerVoxel);

    const fvdb::JIdxType batchIdx = grid.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx         = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void
__launch_bounds__(1024)
    forEachJaggedElementChannelCUDAKernel(JaggedRAcc32<ScalarT, NDIMS> jaggedAcc,
                                          const bool returnIfOutOfRange,
                                          const int64_t channelsPerElement,
                                          Func func,
                                          Args... args) {
    const uint64_t idx        = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t numElements = jaggedAcc.elementCount();
    if (idx >= static_cast<uint64_t>(numElements) * channelsPerElement) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, -1, jaggedAcc, args...);
            return;
        }
    }
    const int64_t elementIdx      = static_cast<int64_t>(idx / channelsPerElement);
    const fvdb::JIdxType batchIdx = jaggedAcc.batchIdx(elementIdx);
    const int64_t channelIdx      = static_cast<int64_t>(idx % channelsPerElement);

    func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void
forEachTensorElementChannelCUDAKernel(TorchRAcc32<ScalarT, NDIMS> tensorAcc,
                                      const bool returnIfOutOfRange,
                                      const int64_t channelsPerElement,
                                      Func func,
                                      Args... args) {
    const uint64_t idx        = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t numElements = tensorAcc.size(0);
    if (idx >= static_cast<uint64_t>(numElements) * channelsPerElement) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, tensorAcc, args...);
            return;
        }
    }
    const int64_t elementIdx = static_cast<int64_t>(idx / channelsPerElement);
    const int64_t channelIdx = static_cast<int64_t>(idx % channelsPerElement);

    func(elementIdx, channelIdx, tensorAcc, args...);
}

} // namespace _private

/// @brief Run the given function on each leaf in the grid batch in parallel on the GPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t cidx,
///            fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///        Where:
///            - bidx is the batch index of the current leaf
///            - lidx is the index of the leaf within the bidx^th grid in the batch
///            - cidx is the channel index
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each leaf. It must be a callable of the
/// form
///         void(int32_t, int32_t, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType>,
///         Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param stream Which cuda stream to run the kernel on
/// @param sharedMemBytes The amount of shared memory to use for the kernel. If 0, no shared memory
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void
forEachLeafCUDA(const at::cuda::CUDAStream &stream,
                const size_t sharedMemBytes,
                const bool returnIfOutOfRange,
                const int64_t numThreads,
                const int64_t numChannels,
                const fvdb::detail::GridBatchImpl &batchHdl,
                Func func,
                Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const int64_t numBlocks = GET_BLOCKS(batchHdl.totalLeaves() * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t)(4294967295), "Too many blocks");
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(_private::forEachLeafCUDAKernel<GridType, Func, Args...>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachLeafCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        auto batchAccessor = batchHdl.deviceAccessor<GridType>();
        _private::forEachLeafCUDAKernel<<<numBlocks, numThreads, sharedMemBytes, stream>>>(
            batchAccessor, returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename GridType, typename Func, typename... Args>
void
forEachLeafCUDA(const int64_t numThreads,
                const int64_t numChannels,
                const fvdb::detail::GridBatchImpl &batchHdl,
                Func func,
                Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
    return forEachLeafCUDA<GridType, Func, Args...>(
        stream, 0, true, numThreads, numChannels, batchHdl, func, args...);
}

/// @brief Run the given function on each leaf in the specified grid (at index batchIdx) in the
/// batch in parallel on the GPU.
///        The callback has the form:
///            void(const nanovdb::NanoGrid<GridType>*, int32_t lidx, int32_t cidx, Args...)
///        Where:
///            - grid is a pointer to the batchIdx^th grid in the batch
///            - lidx is the index of the leaf within the batchIdx^th grid in the batch
///            - cidx is the channel index
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each leaf.
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param stream Which cuda stream to run the kernel on
/// @param sharedMemBytes The amount of shared memory to use for the kernel. If 0, no shared memory
/// @param returnIfOutOfRange Whether to skip the callback if the element is out of range. If false,
///                           the callback will be run with -1 for the batchIdx, leafIdx, and
///                           channelIdx
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchIdx The index of the grid in the batch to run the callback on
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void
forEachLeafInOneGridCUDA(const at::cuda::CUDAStream &stream,
                         const size_t sharedMemBytes,
                         const bool returnIfOutOfRange,
                         const int64_t numThreads,
                         const int64_t numChannels,
                         const int64_t batchIdx,
                         const fvdb::detail::GridBatchImpl &batchHdl,
                         Func func,
                         Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    TORCH_CHECK(batchIdx >= 0 && batchIdx < batchHdl.batchSize(), "Batch index out of range");
    const int64_t numBlocks = GET_BLOCKS(batchHdl.numLeaves(batchIdx) * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t)(4294967295), "Too many blocks");
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachLeafSingleGridCUDAKernel<GridType, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachLeafSingleGridCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        auto batchAccessor = batchHdl.deviceAccessor<GridType>();
        _private::
            forEachLeafSingleGridCUDAKernel<<<numBlocks, numThreads, sharedMemBytes, stream>>>(
                batchAccessor, returnIfOutOfRange, numChannels, batchIdx, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename GridType, typename Func, typename... Args>
void
forEachLeafInOneGridCUDA(const int64_t numThreads,
                         const int64_t numChannels,
                         const int64_t batchIdx,
                         const fvdb::detail::GridBatchImpl &batchHdl,
                         Func func,
                         Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
    return forEachLeafInOneGridCUDA<GridType, Func, Args...>(
        stream, 0, true, numThreads, numChannels, batchIdx, batchHdl, func, args...);
}

/// @brief Run the given function on each voxel in the grid batch in parallel on the GPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx,
///            fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///         Where:
///             - bidx is the batch index of the current voxel
///             - lidx is the index of the leaf containing the voxelwithin the bidx^th grid in the
///             batch
///             - vidx is the index of the voxel within the leaf
///             - cidx is the channel index
/// @note This function will run on inactive voxels within a leaf so you need to chekc if the voxel
/// is active
///
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each voxel. It must be a callable of
/// the form
///         void(int32_t, int32_t, int32_t, int32_t,
///         fvdb::detail::GridBatchImpl::Accessor<GridType>, Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param stream Which cuda stream to run the kernel on
/// @param sharedMemBytes The amount of shared memory to use for the kernel. If 0, no shared memory
/// @param returnIfOutOfRange Whether to skip the callback if the element is out of range. If false,
///                           the callback will be run with -1 for the batchIdx, leafIdx, voxelIdx,
///                           and channelIdx
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void
forEachVoxelCUDA(const at::cuda::CUDAStream &stream,
                 const size_t sharedMemBytes,
                 const bool returnIfOutOfRange,
                 const int64_t numThreads,
                 const int64_t numChannels,
                 const fvdb::detail::GridBatchImpl &batchHdl,
                 Func func,
                 Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());

    const int64_t VOXELS_PER_LEAF =
        nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const int64_t numLeaves = batchHdl.totalLeaves();
    const int64_t numVoxels = batchHdl.totalVoxels();

    if (numVoxels == 0)
        return;
    auto batchAccessor = batchHdl.deviceAccessor<GridType>();

    if (fvdb::Config::global().ultraSparseAccelerationEnabled()) {
        torch::Tensor metaIndex =
            torch::empty({numVoxels, 3}, torch::dtype(torch::kInt64).device(batchHdl.device()));
        auto metaIndexAcc = metaIndex.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>();

        const int64_t numMetaBlocks = GET_BLOCKS(numLeaves * VOXELS_PER_LEAF, 128);
        _private::voxelMetaIndexCUDAKernel<<<numMetaBlocks, 128, 0, stream>>>(batchAccessor,
                                                                              metaIndexAcc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachVoxelWithMetaCUDAKernel<GridType, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachVoxelWithMetaCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        const int64_t numBlocks = GET_BLOCKS(numVoxels * numChannels, numThreads);
        TORCH_INTERNAL_ASSERT(numBlocks < (int64_t)(4294967295),
                              "Too many blocks in forEachVoxelCUDA");
        _private::forEachVoxelWithMetaCUDAKernel<<<numBlocks, numThreads, sharedMemBytes, stream>>>(
            batchAccessor, metaIndexAcc, returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

    } else {
        const int64_t numBlocks = GET_BLOCKS(numLeaves * VOXELS_PER_LEAF * numChannels, numThreads);
        TORCH_INTERNAL_ASSERT(numBlocks < (int64_t)(4294967295),
                              "Too many blocks in forEachVoxelCUDA");

        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(_private::forEachVoxelCUDAKernel<GridType, Func, Args...>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachVoxelCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachVoxelCUDAKernel<<<numBlocks, numThreads, sharedMemBytes, stream>>>(
            batchAccessor, returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename GridType, typename Func, typename... Args>
void
forEachVoxelCUDA(const int64_t numThreads,
                 const int64_t numChannels,
                 const fvdb::detail::GridBatchImpl &batchHdl,
                 Func func,
                 Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
    return forEachVoxelCUDA<GridType, Func, Args...>(
        stream, 0, true, numThreads, numChannels, batchHdl, func, args...);
}

/// @brief Run the given function on each element in the jagged tensor on the GPU
///        The callback has the form:
///            void(int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAccessor<ScalarT, NDIMS>
///            jaggedAcc, Args...)
///         Where:
///             - bidx is the batch index of the current element
///             - eidx is the offset of the current element in the data tensor of the jagged tensor
///             - cidx is the channel index
///             - jaggedAcc is an accessor to the jagged tensor
/// @tparam ScalarT The scalar type of the elements in the jagged tensor
/// @tparam Func The type of the callback function
/// @tparam ...Args The types of any extra arguments to pass to the callback function
/// @tparam NDIMS The number of data dimensions in the jagged tensor
///
/// @param stream Which cuda stream to run the kernel on
/// @param sharedMemBytes The amount of shared memory to use for the kernel. If 0, no shared memory
/// @param returnIfOutOfRange Whether to skip the callback if the element is out of range. If false,
///                           the callback will be called with -1 for the batchIdx, elementIdx, and
///                           channelIdx
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each jagged element being parallelized
/// over
/// @param jaggedTensor The jagged tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachJaggedElementChannelCUDA(const at::cuda::CUDAStream &stream,
                                const size_t sharedMemBytes,
                                const bool returnIfOutOfRange,
                                const int64_t numThreads,
                                const int64_t numChannels,
                                const JaggedTensor &jaggedTensor,
                                Func func,
                                Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cuda(), "JaggedTensor must be on a CUDA device");
    TORCH_CHECK(jaggedTensor.device().has_index(), "JaggedTensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(jaggedTensor.device());
    const int64_t numElements = jaggedTensor.element_count();
    const int64_t numBlocks   = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachJaggedElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachJaggedElementChannelCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachJaggedElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>
            <<<numBlocks, numThreads, sharedMemBytes, stream>>>(
                jaggedTensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                returnIfOutOfRange,
                numChannels,
                func,
                args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachJaggedElementChannelCUDA(const int64_t numThreads,
                                const int64_t numChannels,
                                const JaggedTensor &jaggedTensor,
                                Func func,
                                Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cuda(), "JaggedTensor must be on a CUDA device");
    TORCH_CHECK(jaggedTensor.device().has_index(), "JaggedTensor device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(jaggedTensor.device().index());
    return forEachJaggedElementChannelCUDA<ScalarT, NDIMS, Func, Args...>(
        stream, 0, true, numThreads, numChannels, jaggedTensor, func, args...);
}

/// @brief Run the given function on each element in the tensor (first dimension) on the GPU
///        The callback has the form:
///            void(int32_t eidx, int32_t cidx, fvdb::TensorAccessor<ScalarT, NDIMS> tensorAcc,
///            Args...)
///         Where:
///             - eidx is the batch id of the tensor
///             - cidx is the channel index
///             - tensorAcc is an accessor to the tensor
/// @tparam ScalarT The scalar type of the elements in the tensor
/// @tparam Func The type of the callback function
/// @tparam ...Args The types of any extra arguments to pass to the callback function
/// @tparam NDIMS The number of data dimensions in the tensor
///
/// @param stream Which cuda stream to run the kernel on
/// @param sharedMemBytes The amount of shared memory to use for the kernel. If 0, no shared memory
/// @param returnIfOutOfRange Whether to skip the callback if the element is out of range. If false,
///                           the callback will be called with -1 for the elementIdx and channelIdx
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each tensor element being parallelized
/// over
/// @param tensor The tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachTensorElementChannelCUDA(const at::cuda::CUDAStream &stream,
                                const size_t sharedMemBytes,
                                const bool returnIfOutOfRange,
                                const int64_t numThreads,
                                const int64_t numChannels,
                                const torch::Tensor &tensor,
                                Func func,
                                Args... args) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on a CUDA device");
    TORCH_CHECK(tensor.device().has_index(), "Tensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(tensor.device());
    const int64_t numElements = tensor.size(0);
    const int64_t numBlocks   = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachTensorElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachTensorElementChannelCUDAKernel (requested ",
                    sharedMemBytes,
                    " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachTensorElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>
            <<<numBlocks, numThreads, sharedMemBytes, stream>>>(
                tensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                returnIfOutOfRange,
                numChannels,
                func,
                args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachTensorElementChannelCUDA(const int64_t numThreads,
                                const int64_t numChannels,
                                const torch::Tensor &tensor,
                                Func func,
                                Args... args) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on a CUDA device");
    TORCH_CHECK(tensor.device().has_index(), "Tensor device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    return forEachTensorElementChannelCUDA<ScalarT, NDIMS, Func, Args...>(
        stream, 0, true, numThreads, numChannels, tensor, func, args...);
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_FOREACHCUDA_CUH
