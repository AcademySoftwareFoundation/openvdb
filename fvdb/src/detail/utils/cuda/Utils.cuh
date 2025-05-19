// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
#define FVDB_DETAIL_UTILS_CUDA_UTILS_CUH

#include <Config.h>
#include <JaggedTensor.h>
#include <detail/GridBatchImpl.h>
#include <detail/ops/Ops.h>

#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAStream.h>

namespace fvdb {

/// @brief Shorthand for torch::PackedTensorAccessor32 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchRAcc32 = torch::PackedTensorAccessor32<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for torch::PackedTensorAccessor64 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchRAcc64 = torch::PackedTensorAccessor64<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for a torch::TensorAccessor with DefaultPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using TorchAcc = torch::TensorAccessor<ScalarType, DIMS>;

/// @brief Shorthand for fvdb::PackedJaggedAccessor32 with RestrictPtrTraits
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using JaggedRAcc32 = fvdb::PackedJaggedAccessor32<ScalarType, DIMS, torch::RestrictPtrTraits>;

/// @brief Shorthand for fvdb::JaggedAccessor
/// @tparam ScalarType The type of the tensor
/// @tparam DIMS The number of dimensions of the tensor
template <typename ScalarType, int32_t DIMS>
using JaggedAcc = fvdb::JaggedAccessor<ScalarType, DIMS>;

/// @brief Get the number of blocks for a CUDA kernel launch given the number of elements and the
/// maximum number of threads per block
/// @param N The number of elements to parallelize over
/// @param maxThreadsPerBlock The maximum number of threads per block
/// @return The number of blocks for a CUDA kernel launch
static int
GET_BLOCKS(const int64_t N, const int64_t maxThreadsPerBlock) {
    if (N <= 0) {
        return 0;
    }

    constexpr int64_t max_int = std::numeric_limits<int>::max();

    // Round up division for positive number that cannot cause integer overflow
    auto block_num = (N - 1) / maxThreadsPerBlock + 1;
    TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

    return static_cast<int>(block_num);
}

namespace _private {

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachLeafCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid,
                      const bool returnIfOutOfRange, const int32_t channelsPerLeaf, Func func,
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
    const int64_t        leafIdx  = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, channelIdx, grid, args...);
}

template <typename GridType, typename Func, typename... Args>
__global__ void
forEachLeafSingleGridCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor,
                                const bool returnIfOutOfRange, const int32_t channelsPerLeaf,
                                const int32_t bidx, Func func, Args... args) {
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
                         TorchRAcc32<int64_t, 2>                         metaIndex) {
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

    const nanovdb::NanoGrid<GridType>                        *grid = gridAccessor.grid(batchIdx);
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
                               TorchRAcc32<int64_t, 2> metaIndex, const bool returnIfOutOfRange,
                               const int64_t channelsPerVoxel, Func func, Args... args) {
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
                       const bool returnIfOutOfRange, const int64_t channelsPerVoxel, Func func,
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
    const int64_t        leafIdx  = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void
__launch_bounds__(1024)
    forEachJaggedElementChannelCUDAKernel(JaggedRAcc32<ScalarT, NDIMS> jaggedAcc,
                                          const bool                   returnIfOutOfRange,
                                          const int64_t channelsPerElement, Func func,
                                          Args... args) {
    const uint64_t idx         = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t  numElements = jaggedAcc.elementCount();
    if (idx >= static_cast<uint64_t>(numElements) * channelsPerElement) {
        if (returnIfOutOfRange) {
            return;
        } else {
            func(-1, -1, -1, jaggedAcc, args...);
            return;
        }
    }
    const int64_t        elementIdx = static_cast<int64_t>(idx / channelsPerElement);
    const fvdb::JIdxType batchIdx   = jaggedAcc.batchIdx(elementIdx);
    const int64_t        channelIdx = static_cast<int64_t>(idx % channelsPerElement);

    func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void
forEachTensorElementChannelCUDAKernel(TorchRAcc32<ScalarT, NDIMS> tensorAcc,
                                      const bool                  returnIfOutOfRange,
                                      const int64_t channelsPerElement, Func func, Args... args) {
    const uint64_t idx         = (static_cast<uint64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const int64_t  numElements = tensorAcc.size(0);
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

/// @brief Per-leaf callback function for counting the number of enabled voxels in each leaf used by
///        countUnmaskedPerLeafShiftedByOne
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void
countEnabledPerLeafShiftByOneLeafCallback(
    int32_t batchIdx, int32_t leafIdx, fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
    TorchAccessor<int32_t, 1> outUnmaskedPerLeafAcc) {
    const nanovdb::NanoGrid<GridType>                        *grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType &leaf =
        grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t numUnmasked =
        static_cast<int64_t>(leaf.template get<fvdb::TotalUnmaskedPerLeaf<GridType>>(1111));
    outUnmaskedPerLeafAcc[gridAccessor.leafOffset(batchIdx) + leafIdx + 1] = numUnmasked;
}

} // namespace _private

/// @brief Get an accessor for the given tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the tensor
/// @tparam N The number of dimensions of the tensor
/// @tparam IndexT The type of index to use for packed tensors on the GPU (default is int32_t)
/// @param tensor The tensor to get an accessor for
/// @return A tensor accessor (either torch::TensorAccessor or torch::PackedTensorAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N, typename IndexT = int32_t>
typename std::conditional<
    DeviceTag == torch::kCUDA,
    torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, IndexT>,
    torch::TensorAccessor<T, N>>::type
tensorAccessor(const torch::Tensor &tensor) {
    if constexpr (DeviceTag == torch::kCUDA) {
        return tensor.generic_packed_accessor<T, N, torch::RestrictPtrTraits, IndexT>();
    } else {
        return tensor.accessor<T, N>();
    }
}

/// @brief Get an accessor for the given batched grid handle with scalar type T
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam GridT The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param batchHdl The batched grid handle to get an accessor for
/// @return A fvdb::detail::GridBatchImpl::Accessor of the given type on the appropriate device
template <c10::DeviceType DeviceTag, typename GridT>
typename fvdb::detail::GridBatchImpl::Accessor<GridT>
gridBatchAccessor(const fvdb::detail::GridBatchImpl &batchHdl) {
    if constexpr (DeviceTag == torch::kCUDA) {
        return batchHdl.deviceAccessor<GridT>();
    } else {
        return batchHdl.hostAccessor<GridT>();
    }
}

/// @brief Get an accessor for the given jagged tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the JaggedTensor
/// @tparam N The number of dimensions of the JaggedTensor data tensor
/// @param jaggedTensor The JaggedTensor to get an accessor for
/// @return A JaggedTensor accessor (either JaggedAccessor or PackedJaggedAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N>
typename std::conditional<DeviceTag == torch::kCUDA, fvdb::JaggedRAcc32<T, N>,
                          fvdb::JaggedAcc<T, N>>::type
jaggedAccessor(const fvdb::JaggedTensor &jaggedTensor) {
    if constexpr (DeviceTag == torch::kCUDA) {
        return jaggedTensor.packed_accessor32<T, N, torch::RestrictPtrTraits>();
    } else {
        return jaggedTensor.accessor<T, N>();
    }
}

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
forEachLeafCUDA(const at::cuda::CUDAStream &stream, const size_t sharedMemBytes,
                const bool returnIfOutOfRange, const int64_t numThreads, const int64_t numChannels,
                const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const int64_t        numBlocks = GET_BLOCKS(batchHdl.totalLeaves() * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t)(4294967295), "Too many blocks");
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(_private::forEachLeafCUDAKernel<GridType, Func, Args...>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachLeafCUDAKernel (requested ",
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
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
forEachLeafCUDA(const int64_t numThreads, const int64_t numChannels,
                const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
    return forEachLeafCUDA<GridType, Func, Args...>(stream, 0, true, numThreads, numChannels,
                                                    batchHdl, func, args...);
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
forEachLeafInOneGridCUDA(const at::cuda::CUDAStream &stream, const size_t sharedMemBytes,
                         const bool returnIfOutOfRange, const int64_t numThreads,
                         const int64_t numChannels, const int64_t batchIdx,
                         const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
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
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachLeafSingleGridCUDAKernel (requested ",
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
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
forEachLeafInOneGridCUDA(const int64_t numThreads, const int64_t numChannels,
                         const int64_t batchIdx, const fvdb::detail::GridBatchImpl &batchHdl,
                         Func func, Args... args) {
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
forEachVoxelCUDA(const at::cuda::CUDAStream &stream, const size_t sharedMemBytes,
                 const bool returnIfOutOfRange, const int64_t numThreads, const int64_t numChannels,
                 const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
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
            torch::empty({ numVoxels, 3 }, torch::dtype(torch::kInt64).device(batchHdl.device()));
        auto metaIndexAcc = metaIndex.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>();

        const int64_t numMetaBlocks = GET_BLOCKS(numLeaves * VOXELS_PER_LEAF, 128);
        _private::voxelMetaIndexCUDAKernel<<<numMetaBlocks, 128, 0, stream>>>(batchAccessor,
                                                                              metaIndexAcc);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachVoxelWithMetaCUDAKernel<GridType, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachVoxelWithMetaCUDAKernel (requested ",
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
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
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachVoxelCUDAKernel<<<numBlocks, numThreads, sharedMemBytes, stream>>>(
            batchAccessor, returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename GridType, typename Func, typename... Args>
void
forEachVoxelCUDA(const int64_t numThreads, const int64_t numChannels,
                 const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(batchHdl.device().index());
    return forEachVoxelCUDA<GridType, Func, Args...>(stream, 0, true, numThreads, numChannels,
                                                     batchHdl, func, args...);
}

/// @brief Run the given function on each leaf in the grid batch on the CPU.
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
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
__host__ void
forEachLeafCPU(int64_t channelsPerLeaf, const fvdb::detail::GridBatchImpl &batchHdl, Func func,
               Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    for (uint64_t leafChannelIdx = 0;
         leafChannelIdx < static_cast<uint64_t>(batchAccessor.totalLeaves()) * channelsPerLeaf;
         leafChannelIdx += 1) {
        const int64_t cumLeafIdx = static_cast<int64_t>(leafChannelIdx / channelsPerLeaf);
        const int64_t channelIdx = static_cast<int64_t>(leafChannelIdx % channelsPerLeaf);

        const fvdb::JIdxType batchIdx = batchAccessor.leafBatchIndex(cumLeafIdx);
        const int64_t        leafIdx  = cumLeafIdx - batchAccessor.leafOffset(batchIdx);

        func(batchIdx, leafIdx, channelIdx, batchAccessor, args...);
    }
}

/// @brief Run the given function on each leaf in the specified grid (at index batchIdx) in the
/// batch on the CPU.
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
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchIdx The index of the grid in the batch to run the callback on
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void
forEachLeafInOneGridCPU(int64_t numChannels, int64_t batchIdx,
                        const fvdb::detail::GridBatchImpl &batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    TORCH_CHECK(batchIdx >= 0 && batchIdx < batchHdl.batchSize(), "Batch index out of range");
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    const typename nanovdb::NanoGrid<GridType> *cpuGrid = batchAccessor.grid(batchIdx);

    for (uint64_t leafChannelIdx = 0;
         leafChannelIdx < static_cast<uint64_t>(cpuGrid->tree().nodeCount(0)) * numChannels;
         leafChannelIdx += 1) {
        const int64_t leafIdx    = static_cast<int64_t>(leafChannelIdx / numChannels);
        const int64_t channelIdx = static_cast<int64_t>(leafChannelIdx % numChannels);

        func(cpuGrid, leafIdx, channelIdx, args...);
    }
}

/// @brief Run the given function on each voxel in the grid batch on the CPU.
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
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
__host__ void
forEachVoxelCPU(int64_t numChannels, const fvdb::detail::GridBatchImpl &batchHdl, Func func,
                Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    constexpr int64_t VOXELS_PER_LEAF =
        static_cast<int64_t>(nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES);
    auto batchAccessor = batchHdl.hostAccessor<GridType>();
    for (fvdb::JIdxType batchIdx = 0; batchIdx < batchAccessor.batchSize(); batchIdx += 1) {
        const nanovdb::NanoGrid<GridType> *grid = batchAccessor.grid(batchIdx);
        for (int64_t leafIdx = 0; leafIdx < grid->tree().nodeCount(0); leafIdx += 1) {
            const typename nanovdb::NanoGrid<GridType>::LeafNodeType &leaf =
                grid->tree().template getFirstNode<0>()[leafIdx];
            for (int64_t voxIdx = 0; voxIdx < VOXELS_PER_LEAF; voxIdx += 1) {
                if (leaf.isActive(voxIdx)) {
                    for (int64_t channelIdx = 0; channelIdx < numChannels; channelIdx += 1) {
                        func(batchIdx, leafIdx, voxIdx, channelIdx, batchAccessor, args...);
                    }
                }
            }
        }
    }
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
forEachJaggedElementChannelCUDA(const at::cuda::CUDAStream &stream, const size_t sharedMemBytes,
                                const bool returnIfOutOfRange, const int64_t numThreads,
                                const int64_t numChannels, const JaggedTensor &jaggedTensor,
                                Func func, Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cuda(), "JaggedTensor must be on a CUDA device");
    TORCH_CHECK(jaggedTensor.device().has_index(), "JaggedTensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(jaggedTensor.device());
    const int64_t        numElements = jaggedTensor.element_count();
    const int64_t        numBlocks   = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachJaggedElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachJaggedElementChannelCUDAKernel (requested ",
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachJaggedElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>
            <<<numBlocks, numThreads, sharedMemBytes, stream>>>(
                jaggedTensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachJaggedElementChannelCUDA(const int64_t numThreads, const int64_t numChannels,
                                const JaggedTensor &jaggedTensor, Func func, Args... args) {
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
forEachTensorElementChannelCUDA(const at::cuda::CUDAStream &stream, const size_t sharedMemBytes,
                                const bool returnIfOutOfRange, const int64_t numThreads,
                                const int64_t numChannels, const torch::Tensor &tensor, Func func,
                                Args... args) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on a CUDA device");
    TORCH_CHECK(tensor.device().has_index(), "Tensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(tensor.device());
    const int64_t        numElements = tensor.size(0);
    const int64_t        numBlocks   = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        if (sharedMemBytes > 0) {
            if (cudaFuncSetAttribute(
                    _private::forEachTensorElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemBytes) != cudaSuccess) {
                AT_ERROR(
                    "Failed to set maximum shared memory size for forEachTensorElementChannelCUDAKernel (requested ",
                    sharedMemBytes, " bytes), try lowering sharedMemBytes.");
            }
        }
        _private::forEachTensorElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...>
            <<<numBlocks, numThreads, sharedMemBytes, stream>>>(
                tensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(),
                returnIfOutOfRange, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachTensorElementChannelCUDA(const int64_t numThreads, const int64_t numChannels,
                                const torch::Tensor &tensor, Func func, Args... args) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on a CUDA device");
    TORCH_CHECK(tensor.device().has_index(), "Tensor device must have an index");
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(tensor.device().index());
    return forEachTensorElementChannelCUDA<ScalarT, NDIMS, Func, Args...>(
        stream, 0, true, numThreads, numChannels, tensor, func, args...);
}

/// @brief Run the given function on each element in the jagged tensor on the CPU
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
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each jagged element being parallelized
/// over
/// @param jaggedTensor The jagged tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachJaggedElementChannelCPU(int64_t numChannels, const JaggedTensor &jaggedTensor, Func func,
                               Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cpu(), "JaggedTensor must be on the CPU");
    const int64_t numElements = jaggedTensor.element_count() * numChannels;
    auto          jaggedAcc   = jaggedTensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t        elementIdx = idx / numChannels;
        const fvdb::JIdxType batchIdx   = jaggedAcc.batchIdx(elementIdx);
        const int64_t        channelIdx = idx % numChannels;

        func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
    }
}

/// @brief Run the given function on each element in the tensor (first dimension) on the CPU
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
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each tensor element being parallelized
/// over
/// @param tensor The tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void
forEachTensorElementChannelCPU(int64_t numChannels, const torch::Tensor &tensor, Func func,
                               Args... args) {
    TORCH_CHECK(tensor.device().is_cpu(), "Tensor must be on the CPU");
    const int64_t numElements = tensor.size(0) * numChannels;
    auto          tensorAcc   = tensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t elementIdx = idx / numChannels;
        const int64_t channelIdx = idx % numChannels;

        func(elementIdx, channelIdx, tensorAcc, args...);
    }
}

/// @brief Count the number of unmasked voxels in each leaf node in the grid batch on the GPU
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param gridBatch The grid batch
/// @return A tensor of size [total_leaves + 1] containing the number of unmasked voxels in each
/// leaf (and zero in the first index)
///         i.e. [0, #unmasked_in_leaf_0, #unmasked_in_leaf_1, ... #unmasksed_in_leaf_N]
template <typename GridType, c10::DeviceType DeviceTag>
__host__ inline torch::Tensor
countEnabledPerLeafShiftedByOne(const fvdb::detail::GridBatchImpl &gridBatch) {
    auto          opts = torch::TensorOptions().dtype(torch::kInt32).device(gridBatch.device());
    torch::Tensor unmaskedPerLeaf = torch::zeros({ gridBatch.totalLeaves() + 1 }, opts);

    auto outUnmaskedPerLeafAcc = tensorAccessor<DeviceTag, int32_t, 1>(unmaskedPerLeaf);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__(int32_t batchIdx, int32_t leafIdx, int32_t,
                                 fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchRAcc32>(
                batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        fvdb::forEachLeafCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=](int32_t batchIdx, int32_t leafIdx, int32_t,
                      fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchAcc>(
                batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        fvdb::forEachLeafCPU<GridType>(1, gridBatch, cb);
    }
    return unmaskedPerLeaf;
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
