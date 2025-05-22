// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_FOREACHCPU_H
#define FVDB_DETAIL_UTILS_FOREACHCPU_H

#include <JaggedTensor.h>
#include <detail/GridBatchImpl.h>

#include <nanovdb/NanoVDB.h>

namespace fvdb {

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
forEachLeafCPU(int64_t channelsPerLeaf,
               const fvdb::detail::GridBatchImpl &batchHdl,
               Func func,
               Args... args) {
    TORCH_CHECK(batchHdl.device().is_cpu(), "Grid batch must be on the CPU");
    auto batchAccessor = batchHdl.hostAccessor<GridType>();

    for (uint64_t leafChannelIdx = 0;
         leafChannelIdx < static_cast<uint64_t>(batchAccessor.totalLeaves()) * channelsPerLeaf;
         leafChannelIdx += 1) {
        const int64_t cumLeafIdx = static_cast<int64_t>(leafChannelIdx / channelsPerLeaf);
        const int64_t channelIdx = static_cast<int64_t>(leafChannelIdx % channelsPerLeaf);

        const fvdb::JIdxType batchIdx = batchAccessor.leafBatchIndex(cumLeafIdx);
        const int64_t leafIdx         = cumLeafIdx - batchAccessor.leafOffset(batchIdx);

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
forEachLeafInOneGridCPU(int64_t numChannels,
                        int64_t batchIdx,
                        const fvdb::detail::GridBatchImpl &batchHdl,
                        Func func,
                        Args... args) {
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
forEachVoxelCPU(int64_t numChannels,
                const fvdb::detail::GridBatchImpl &batchHdl,
                Func func,
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
forEachJaggedElementChannelCPU(int64_t numChannels,
                               const JaggedTensor &jaggedTensor,
                               Func func,
                               Args... args) {
    TORCH_CHECK(jaggedTensor.device().is_cpu(), "JaggedTensor must be on the CPU");
    const int64_t numElements = jaggedTensor.element_count() * numChannels;
    auto jaggedAcc            = jaggedTensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t elementIdx      = idx / numChannels;
        const fvdb::JIdxType batchIdx = jaggedAcc.batchIdx(elementIdx);
        const int64_t channelIdx      = idx % numChannels;

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
forEachTensorElementChannelCPU(int64_t numChannels,
                               const torch::Tensor &tensor,
                               Func func,
                               Args... args) {
    TORCH_CHECK(tensor.device().is_cpu(), "Tensor must be on the CPU");
    const int64_t numElements = tensor.size(0) * numChannels;
    auto tensorAcc            = tensor.accessor<ScalarT, NDIMS>();

    for (int64_t idx = 0; idx < numElements; idx += 1) {
        const int64_t elementIdx = idx / numChannels;
        const int64_t channelIdx = idx % numChannels;

        func(elementIdx, channelIdx, tensorAcc, args...);
    }
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_FOREACHCPU_H
