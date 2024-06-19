#include <nanovdb/NanoVDB.h>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>

#include "detail/GridBatchImpl.h"
#include "detail/ops/Ops.h"
#include "JaggedTensor.h"


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


/// @brief Get the number of blocks for a CUDA kernel launch given the number of elements and the maximum number of threads per block
/// @param N The number of elements to parallelize over
/// @param maxThreadsPerBlock The maximum number of threads per block
/// @return The number of blocks for a CUDA kernel launch
static int GET_BLOCKS(const int64_t N, const int64_t maxThreadsPerBlock) {
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
__global__ void forEachLeafCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid, int32_t channelsPerLeaf, Func func, Args... args) {
    const int32_t leafChannelIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (leafChannelIdx >= grid.totalLeafs() * channelsPerLeaf) {
        return;
    }

    const int32_t cumLeafIdx = leafChannelIdx / channelsPerLeaf;
    const int32_t channelIdx = leafChannelIdx % channelsPerLeaf;

    const int32_t batchIdx = grid.leafBatchIndex(cumLeafIdx);
    const int32_t leafIdx = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, channelIdx, grid, args...);
}


template <typename GridType, typename Func, typename... Args>
__global__ void forEachLeafSingleGridCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> batchAccessor, int32_t channelsPerLeaf, int32_t bidx, Func func, Args... args) {
    const typename nanovdb::NanoGrid<GridType>* gpuGrid = batchAccessor.grid(bidx);

    const int32_t leafChannelIdx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (leafChannelIdx >= gpuGrid->tree().nodeCount(0) * channelsPerLeaf) {
        return;
    }

    const int32_t leafIdx = leafChannelIdx / channelsPerLeaf;
    const int32_t channelIdx = leafChannelIdx % channelsPerLeaf;

    func(gpuGrid, leafIdx, channelIdx, args...);
}


template <typename GridType, typename Func, typename... Args>
__global__ void forEachVoxelCUDAKernel(fvdb::detail::GridBatchImpl::Accessor<GridType> grid, int64_t channelsPerVoxel, Func func, Args... args) {
    constexpr int32_t VOXELS_PER_LEAF = nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const int64_t lvcIdx = ((int64_t)blockIdx.x * (int64_t)blockDim.x) + threadIdx.x;
    const int64_t voxelsChannelsPerLeaf = (int64_t) VOXELS_PER_LEAF * (int64_t) channelsPerVoxel;

    if (lvcIdx >= grid.totalLeafs() * voxelsChannelsPerLeaf) {
        return;
    }

    const int64_t cumLeafIdx = (lvcIdx / voxelsChannelsPerLeaf);
    const int64_t leafVoxelIdx = (lvcIdx  - cumLeafIdx * voxelsChannelsPerLeaf) / channelsPerVoxel;
    const int64_t channelIdx = lvcIdx  - cumLeafIdx * voxelsChannelsPerLeaf - leafVoxelIdx * channelsPerVoxel;

    const int64_t batchIdx = grid.leafBatchIndex(cumLeafIdx);
    const int64_t leafIdx = cumLeafIdx - grid.leafOffset(batchIdx);

    func(batchIdx, leafIdx, leafVoxelIdx, channelIdx, grid, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void forEachJaggedElementChannelCUDAKernel(JaggedRAcc32<ScalarT, NDIMS> jaggedAcc,
                                                      int64_t channelsPerElement,
                                                      Func func, Args... args) {
    const int64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int64_t numElements = jaggedAcc.elementCount();
    if (idx >= numElements * channelsPerElement) {
        return;
    }
    const int64_t elementIdx = idx / channelsPerElement;
    const int64_t batchIdx = jaggedAcc.batchIdx(elementIdx);
    const int64_t channelIdx = idx % channelsPerElement;

    func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
}

template <int32_t NDIMS, typename ScalarT, typename Func, typename... Args>
__global__ void forEachTensorElementChannelCUDAKernel(TorchRAcc32<ScalarT, NDIMS> tensorAcc,
                                                      int64_t channelsPerElement,
                                                      Func func, Args... args) {
    const int64_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int64_t numElements = tensorAcc.size(0);
    if (idx >= numElements * channelsPerElement) {
        return;
    }
    const int64_t elementIdx = idx / channelsPerElement;
    const int64_t channelIdx = idx % channelsPerElement;

    func(elementIdx, channelIdx, tensorAcc, args...);
}

/// @brief Per-leaf callback function for counting the number of enabled voxels in each leaf used by
///        countUnmaskedPerLeafShiftedByOne
template <typename GridType, template <typename T, int32_t D> typename TorchAccessor>
__hostdev__ inline void countEnabledPerLeafShiftByOneLeafCallback(int32_t batchIdx, int32_t leafIdx,
                                                                  fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor,
                                                                  TorchAccessor<int32_t, 1> outUnmaskedPerLeafAcc) {
    const nanovdb::NanoGrid<GridType>* grid = gridAccessor.grid(batchIdx);
    const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
    const int64_t numUnmasked = (int64_t) leaf.template get<fvdb::TotalUnmaskedPerLeaf<GridType>>(1111);
    outUnmaskedPerLeafAcc[gridAccessor.leafOffset(batchIdx) + leafIdx + 1] = numUnmasked;
}

/// @brief Helper class to extract the correct TensorAccessor type for a given device
/// @tparam DeviceTag The device tag (torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the tensor
/// @tparam N The number of dimensions of the tensor
template <c10::DeviceType DeviceTag, typename T, int N, typename IndexType = int32_t>
struct TensorAccessorExtractor {
    using AccType = torch::TensorAccessor<T, N>;
    AccType get(const torch::Tensor& tensor) {
        return tensor.accessor<T, N>();
    }
};
template <typename T, int N, typename IndexType>
struct TensorAccessorExtractor<torch::kCUDA, T, N, IndexType> {
    using AccType = torch::GenericPackedTensorAccessor<T, N, torch::RestrictPtrTraits, IndexType>;
    AccType get(const torch::Tensor& tensor) {
        return tensor.generic_packed_accessor<T, N, torch::RestrictPtrTraits, IndexType>();
    }
};

/// @brief Helper class to extract the correct GridBatchImpl::Accessor type for a given device
/// @tparam DeviceTag The device tag (torch::kCUDA or torch::kCPU)
/// @tparam T The GridType of the grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
template <c10::DeviceType DeviceTag, typename T>
struct GridBatchAccessorExtractor {
    using AccType = fvdb::detail::GridBatchImpl::Accessor<T>;
    AccType get(const fvdb::detail::GridBatchImpl& batchHdl) {
        return batchHdl.hostAccessor<T>();
    }
};
template <typename T>
struct GridBatchAccessorExtractor<torch::kCUDA, T> {
    using AccType = fvdb::detail::GridBatchImpl::Accessor<T>;
    AccType get(const fvdb::detail::GridBatchImpl& batchHdl) {
        return batchHdl.deviceAccessor<T>();
    }
};

/// @brief Helper class to extract the correct JaggedAccessor type for a given device
/// @tparam DeviceTag The device tag (torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the jagged tensor
/// @tparam N The number of dimensions of the jagged data tensor
template <c10::DeviceType DeviceTag, typename T, int N>
struct JaggedAccessorExtractor {
    using AccType = fvdb::JaggedAcc<T, N>;
    AccType get(const fvdb::JaggedTensor& tensor) {
        return tensor.accessor<T, N>();
    }
};
template <typename T, int N>
struct JaggedAccessorExtractor<torch::kCUDA, T, N> {
    using AccType = fvdb::JaggedRAcc32<T, N>;
    AccType get(const fvdb::JaggedTensor& tensor) {
        return tensor.packed_accessor32<T, N, torch::RestrictPtrTraits>();
    }
};

}  // namespace _private


/// @brief Run the given function on each leaf in the grid batch in parallel on the GPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t cidx, fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///        Where:
///            - bidx is the batch index of the current leaf
///            - lidx is the index of the leaf within the bidx^th grid in the batch
///            - cidx is the channel index
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each leaf. It must be a callable of the form
///         void(int32_t, int32_t, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType>, Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void forEachLeafCUDA(int64_t numThreads, int64_t numChannels, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const int64_t numBlocks = GET_BLOCKS(batchHdl.totalLeaves() * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t) (4294967295), "Too many blocks");
    if (numBlocks > 0) {
        auto batchAccessor = batchHdl.deviceAccessor<GridType>();
        _private::forEachLeafCUDAKernel<<<numBlocks, numThreads>>>(batchAccessor, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


/// @brief Run the given function on each leaf in the specified grid (at index batchIdx) in the batch in parallel on the GPU.
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
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchIdx The index of the grid in the batch to run the callback on
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void forEachLeafInOneGridCUDA(int64_t numThreads, int64_t numChannels, int64_t batchIdx, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    TORCH_CHECK(batchIdx >= 0 && batchIdx < batchHdl.batchSize(), "Batch index out of range");
    const int64_t numBlocks = GET_BLOCKS(batchHdl.numLeaves(batchIdx) * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t) (4294967295), "Too many blocks");
    if (numBlocks > 0) {
        auto batchAccessor = batchHdl.deviceAccessor<GridType>();
        _private::forEachLeafSingleGridCUDAKernel<<<numBlocks, numThreads>>>(batchAccessor, numChannels, batchIdx, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


/// @brief Run the given function on each voxel in the grid batch in parallel on the GPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///         Where:
///             - bidx is the batch index of the current voxel
///             - lidx is the index of the leaf containing the voxelwithin the bidx^th grid in the batch
///             - vidx is the index of the voxel within the leaf
///             - cidx is the channel index
/// @note This function will run on inactive voxels within a leaf so you need to chekc if the voxel is active
///
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each voxel. It must be a callable of the form
///         void(int32_t, int32_t, int32_t, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType>, Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void forEachVoxelCUDA(int64_t numThreads, int64_t numChannels, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cuda(), "Grid batch must be on a CUDA device");
    TORCH_CHECK(batchHdl.device().has_index(), "Grid batch device must have an index");
    c10::cuda::CUDAGuard deviceGuard(batchHdl.device());
    const int64_t VOXELS_PER_LEAF = nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    const int64_t numBlocks = GET_BLOCKS(batchHdl.totalLeaves() * VOXELS_PER_LEAF * numChannels, numThreads);
    TORCH_INTERNAL_ASSERT(numBlocks < (int64_t) (4294967295), "Too many blocks in forEachVoxelCUDA");
    if (numBlocks > 0) {
        auto batchAccessor = batchHdl.deviceAccessor<GridType>();
        _private::forEachVoxelCUDAKernel<<<numBlocks, numThreads>>>(batchAccessor, numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


/// @brief Run the given function on each leaf in the grid batch on the CPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t cidx, fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///        Where:
///            - bidx is the batch index of the current leaf
///            - lidx is the index of the leaf within the bidx^th grid in the batch
///            - cidx is the channel index
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each leaf. It must be a callable of the form
///         void(int32_t, int32_t, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType>, Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
__host__ void forEachLeafCPU(int64_t channelsPerLeaf, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    for (int64_t leafChannelIdx = 0; leafChannelIdx < batchAccessor.totalLeafs() * channelsPerLeaf; leafChannelIdx += 1) {

        const int64_t cumLeafIdx = leafChannelIdx / channelsPerLeaf;
        const int64_t channelIdx = leafChannelIdx % channelsPerLeaf;

        const int64_t batchIdx = batchAccessor.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx = cumLeafIdx - batchAccessor.leafOffset(batchIdx);

        func(batchIdx, leafIdx, channelIdx, batchAccessor, args...);
    }
}


/// @brief Run the given function on each leaf in the specified grid (at index batchIdx) in the batch on the CPU.
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
/// @param numThreads The number of threads per block to use
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchIdx The index of the grid in the batch to run the callback on
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
void forEachLeafInOneGridCPU(int64_t numChannels, int64_t batchIdx, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    TORCH_CHECK(batchIdx >= 0 && batchIdx < batchHdl.batchSize(), "Batch index out of range");
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    const typename nanovdb::NanoGrid<GridType>* cpuGrid = batchAccessor.grid(batchIdx);

    for (int64_t leafChannelIdx = 0; leafChannelIdx < cpuGrid->tree().nodeCount(0) * numChannels; leafChannelIdx += 1) {

        const int32_t leafIdx = leafChannelIdx / numChannels;
        const int32_t channelIdx = leafChannelIdx % numChannels;

        func(cpuGrid, leafIdx, channelIdx, args...);
    }
}


/// @brief Run the given function on each voxel in the grid batch on the CPU.
///        The callback has the form:
///            void(int32_t bidx, int32_t lidx, int32_t vidx, int32_t cidx, fvdb::detail::GridBatchImpl::Accessor<GridType> batchAcc, Args...)
///         Where:
///             - bidx is the batch index of the current voxel
///             - lidx is the index of the leaf containing the voxelwithin the bidx^th grid in the batch
///             - vidx is the index of the voxel within the leaf
///             - cidx is the channel index
/// @note This function will run on inactive voxels within a leaf so you need to chekc if the voxel is active
///
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @tparam Func The type of the callback function to run on each voxel. It must be a callable of the form
///         void(int32_t, int32_t, int32_t, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType>, Args...)
/// @tparam Args... The types of any extra arguments to pass to the callback function
///
/// @param numChannels The number of channels per item in each leaf being parallelized over
/// @param batchHdl A batch of index grids
/// @param func The callback function to run on each leaf
/// @param args Any extra arguments to pass to the callback function
template <typename GridType, typename Func, typename... Args>
__host__ void forEachVoxelCPU(int64_t numChannels, const fvdb::detail::GridBatchImpl& batchHdl, Func func, Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    constexpr int64_t VOXELS_PER_LEAF = nanovdb::NanoTree<nanovdb::ValueOnIndex>::LeafNodeType::NUM_VALUES;
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    for (int64_t batchIdx = 0; batchIdx < batchAccessor.batchSize(); batchIdx += 1) {
        const nanovdb::NanoGrid<GridType>* grid = batchAccessor.grid(batchIdx);
        for (uint64_t leafIdx = 0; leafIdx < grid->tree().nodeCount(0); leafIdx += 1) {
            const typename nanovdb::NanoGrid<GridType>::LeafNodeType& leaf = grid->tree().template getFirstNode<0>()[leafIdx];
            for (uint64_t voxIdx = 0; voxIdx < VOXELS_PER_LEAF; voxIdx += 1) {
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
///            void(int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAccessor<ScalarT, NDIMS> jaggedAcc, Args...)
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
/// @param numChannels The number of channels per item in each jagged element being parallelized over
/// @param jaggedTensor The jagged tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachJaggedElementChannelCUDA(int64_t numThreads, int64_t numChannels, const JaggedTensor& jaggedTensor, Func func, Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cuda(), "JaggedTensor must be on a CUDA device");
    TORCH_CHECK(jaggedTensor.device().has_index(), "JaggedTensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(jaggedTensor.device());
    const int64_t numElements = jaggedTensor.element_count();
    const int64_t numBlocks = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        _private::forEachJaggedElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...><<<numBlocks, numThreads>>>(
                jaggedTensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(), numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


/// @brief Run the given function on each element in the tensor (first dimension) on the GPU
///        The callback has the form:
///            void(int32_t eidx, int32_t cidx, fvdb::TensorAccessor<ScalarT, NDIMS> tensorAcc, Args...)
///         Where:
///             - eidx is the batch id of the tensor
///             - cidx is the channel index
///             - tensorAcc is an accessor to the tensor
/// @tparam ScalarT The scalar type of the elements in the tensor
/// @tparam Func The type of the callback function
/// @tparam ...Args The types of any extra arguments to pass to the callback function
/// @tparam NDIMS The number of data dimensions in the tensor
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each tensor element being parallelized over
/// @param tensor The tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachTensorElementChannelCUDA(int64_t numThreads, int64_t numChannels, const torch::Tensor& tensor, Func func, Args... args) {
    TORCH_CHECK(tensor.device().is_cuda(), "Tensor must be on a CUDA device");
    TORCH_CHECK(tensor.device().has_index(), "Tensor device must have an index");
    c10::cuda::CUDAGuard deviceGuard(tensor.device());
    const int64_t numElements = tensor.size(0);
    const int64_t numBlocks = GET_BLOCKS(numElements * numChannels, numThreads);
    if (numBlocks > 0) {
        _private::forEachTensorElementChannelCUDAKernel<NDIMS, ScalarT, Func, Args...><<<numBlocks, numThreads>>>(
                tensor.packed_accessor32<ScalarT, NDIMS, torch::RestrictPtrTraits>(), numChannels, func, args...);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}


/// @brief Run the given function on each element in the jagged tensor on the CPU
///        The callback has the form:
///            void(int32_t bidx, int32_t eidx, int32_t cidx, fvdb::JaggedAccessor<ScalarT, NDIMS> jaggedAcc, Args...)
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
/// @param numChannels The number of channels per item in each jagged element being parallelized over
/// @param jaggedTensor The jagged tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachJaggedElementChannelCPU(int64_t numChannels, const JaggedTensor& jaggedTensor, Func func, Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cpu(), "JaggedTensor must be on the CPU");
    const int64_t numElements = jaggedTensor.element_count() * numChannels;
    auto jaggedAcc = jaggedTensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t elementIdx = idx / numChannels;
        const int64_t batchIdx = jaggedAcc.batchIdx(elementIdx);
        const int64_t channelIdx = idx % numChannels;

        func(batchIdx, elementIdx, channelIdx, jaggedAcc, args...);
    }
}


/// @brief Run the given function on each element in the tensor (first dimension) on the CPU
///        The callback has the form:
///            void(int32_t eidx, int32_t cidx, fvdb::TensorAccessor<ScalarT, NDIMS> tensorAcc, Args...)
///         Where:
///             - eidx is the batch id of the tensor
///             - cidx is the channel index
///             - tensorAcc is an accessor to the tensor
/// @tparam ScalarT The scalar type of the elements in the tensor
/// @tparam Func The type of the callback function
/// @tparam ...Args The types of any extra arguments to pass to the callback function
/// @tparam NDIMS The number of data dimensions in the tensor
/// @param numThreads The number of threads to use per block
/// @param numChannels The number of channels per item in each tensor element being parallelized over
/// @param tensor The tensor to parallelize over
/// @param func The callback function to run on each element
/// @param ...args Any extra arguments to pass to the callback function
template <typename ScalarT, int32_t NDIMS, typename Func, typename... Args>
void forEachTensorElementChannelCPU(int64_t numChannels, const torch::Tensor& tensor, Func func, Args... args) {
    TORCH_CHECK(tensor.device().is_cpu(), "Tensor must be on the CPU");
    const int64_t numElements = tensor.size(0) * numChannels;
    auto tensorAcc = tensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t elementIdx = idx / numChannels;
        const int64_t channelIdx = idx % numChannels;

        func(elementIdx, channelIdx, tensorAcc, args...);
    }
}


/// @brief Get an accessor for the given tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the tensor
/// @tparam N The number of dimensions of the tensor
/// @tparam IndexType The type of index to use for packed tensors on the GPU (default is int32_t)
/// @param tensor The tensor to get an accessor for
/// @return A tensor accessor (either torch::TensorAccessor or torch::PackedTensorAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N, typename IndexType = int32_t>
typename _private::TensorAccessorExtractor<DeviceTag, T, N, IndexType>::AccType tensorAccessor(const torch::Tensor& tensor) {
    return  _private::TensorAccessorExtractor<DeviceTag, T, N, IndexType>().get(tensor);
}


/// @brief Get an accessor for the given batched grid handle with scalar type T
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param batchHdl The batched grid handle to get an accessor for
/// @return A fvdb::detail::GridBatchImpl::Accessor of the given type on the appropriate device
template <c10::DeviceType DeviceTag, typename GridType>
typename _private::GridBatchAccessorExtractor<DeviceTag, GridType>::AccType gridBatchAccessor(const fvdb::detail::GridBatchImpl& batchHdl) {
    return  _private::GridBatchAccessorExtractor<DeviceTag, GridType>().get(batchHdl);
}


/// @brief Get an accessor for the given jagged tensor with scalar type T and N dimensions
/// @tparam DeviceTag The device tag to use for the accessor (either torch::kCUDA or torch::kCPU)
/// @tparam T The scalar type of the JaggedTensor
/// @tparam N The number of dimensions of the JaggedTensor data tensor
/// @param jaggedTensor The JaggedTensor to get an accessor for
/// @return A JaggedTensor accessor (either JaggedAccessor or PackedJaggedAccessor32)
template <c10::DeviceType DeviceTag, typename T, size_t N>
typename _private::JaggedAccessorExtractor<DeviceTag, T, N>::AccType jaggedAccessor(const fvdb::JaggedTensor& jaggedTensor) {
    return  _private::JaggedAccessorExtractor<DeviceTag, T, N>().get(jaggedTensor);
}


/// @brief Count the number of unmasked voxels in each leaf node in the grid batch on the GPU
/// @tparam GridType The type of grid (either nanovdb::ValueOnIndex or nanovdb::ValueOnIndexMask)
/// @param gridBatch The grid batch
/// @return A tensor of size [total_leaves + 1] containing the number of unmasked voxels in each leaf (and zero in the first index)
///         i.e. [0, #unmasked_in_leaf_0, #unmasked_in_leaf_1, ... #unmasksed_in_leaf_N]
template <typename GridType, c10::DeviceType DeviceTag>
__host__ inline torch::Tensor countEnabledPerLeafShiftedByOne(const fvdb::detail::GridBatchImpl& gridBatch) {
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(gridBatch.device());
    torch::Tensor unmaskedPerLeaf = torch::zeros({gridBatch.totalLeaves() + 1}, opts);

    auto outUnmaskedPerLeafAcc = tensorAccessor<DeviceTag, int32_t, 1>(unmaskedPerLeaf);
    if constexpr (DeviceTag == torch::kCUDA) {
        auto cb = [=] __device__ (int32_t batchIdx, int32_t leafIdx, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchRAcc32>(batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        fvdb::forEachLeafCUDA<GridType>(1024, 1, gridBatch, cb);
    } else {
        auto cb = [=] (int32_t batchIdx, int32_t leafIdx, int32_t, fvdb::detail::GridBatchImpl::Accessor<GridType> gridAccessor) {
            _private::countEnabledPerLeafShiftByOneLeafCallback<GridType, TorchAcc>(batchIdx, leafIdx, gridAccessor, outUnmaskedPerLeafAcc);
        };
        fvdb::forEachLeafCPU<GridType>(1, gridBatch, cb);
    }
    return unmaskedPerLeaf;
}


/// @brief A wrapper around a raw device buffer that automatically frees the buffer when it goes out of scope
/// @tparam T The type of data this buffer points to
template <typename T>
struct RAIIRawDeviceBuffer {
    T* devicePtr = nullptr;
    size_t bufferSize = 0;
    cudaStream_t stream = 0;

    RAIIRawDeviceBuffer() = default;
    RAIIRawDeviceBuffer(const RAIIRawDeviceBuffer&) = delete;

    RAIIRawDeviceBuffer(RAIIRawDeviceBuffer&& other) {
        devicePtr = other.devicePtr;
        bufferSize = other.bufferSize;
        stream = other.stream;
        other.devicePtr = nullptr;
        other.bufferSize = 0;
        other.stream = 0;
    }

    /// @brief Create a buffer containing size elements of type T on the specified device
    /// @param size The number of elements to allocate
    /// @param device The device to allocate the buffer on
    RAIIRawDeviceBuffer(size_t size, torch::Device device) {
        TORCH_CHECK(device.has_index(), "Device must specify an index");
        stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
        bufferSize = size * sizeof(T);
        devicePtr = reinterpret_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(bufferSize, stream));
        // devicePtr = reinterpret_cast<T*>(c10::cuda::CUDACachingAllocator::raw_alloc(bufferSize));
    }

    ~RAIIRawDeviceBuffer() {
        if (devicePtr == nullptr) {
            return;
        }
        c10::cuda::CUDACachingAllocator::raw_delete(devicePtr);
        devicePtr = nullptr;
    }

    /// @brief Copy data from the host to the device
    /// @param hostData The host data to copy (assumes it points to a buffer of size bufferSize)
    void setData(const T* hostData, bool blocking) {
        cudaMemcpyAsync((void*) devicePtr, (const void*) hostData, bufferSize, cudaMemcpyHostToDevice, stream);
        if (blocking) {
            C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        // cudaMemcpy((void*) devicePtr, (const void*) hostData, bufferSize, cudaMemcpyHostToDevice);
    }
};

}  // namespace fvdb