// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*
    \file DeviceGridTraits.cuh

    \author Efty Sifakis

    \date Aug 8, 2025

    \brief This file implements helpers probing grid traits of device-resident NanoVDB grids

*/

#ifndef NANOVDB_UTIL_CUDA_DEVICEGRIDTRAITS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_DEVICEGRIDTRAITS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/cuda/Util.h> // for cudaCheck

namespace nanovdb::util {

namespace cuda {

/// @brief Utility class for directly probing a device-resident IndexGrid for fundamental traits
///        of the tree topology
/// @note Each getter returns a host copy, so it blocks: the read is issued on @c stream and
///       that stream is synchronized before returning.
/// @tparam BuildT Build type for the grid
template <typename BuildT>
struct DeviceGridTraits
{
    using GridT = typename nanovdb::NanoGrid<BuildT>;
    using TreeT = typename nanovdb::NanoTree<BuildT>;
    using RootT = typename nanovdb::NanoRoot<BuildT>;
    using TreeDataT = typename TreeT::DataType;

    static TreeDataT getTreeData(const GridT *d_grid, cudaStream_t stream = 0)
    {
        TreeDataT treeData{};
        std::size_t offset = GridT::memUsage();
        cudaCheck(cudaMemcpyAsync(&treeData, util::PtrAdd(d_grid, offset), sizeof(TreeDataT), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return treeData;
    }

    static uint64_t getActiveVoxelCount(const GridT *d_grid, cudaStream_t stream = 0)
    {
        uint64_t activeVoxelCount = 0;
        std::size_t offset = GridT::memUsage() + offsetof(TreeT, mVoxelCount);
        cudaCheck(cudaMemcpyAsync(&activeVoxelCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return activeVoxelCount;
    }

    static uint64_t getValueCount(const GridT *d_grid, cudaStream_t stream = 0)
    {
        uint64_t valueCount = 0;
        std::size_t offset = offsetof(GridT, mData1);
        cudaCheck(cudaMemcpyAsync(&valueCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return valueCount;
    }

    static uint64_t getGridSize(const GridT *d_grid, cudaStream_t stream = 0)
    {
        uint64_t gridSize = 0;
        std::size_t offset = offsetof(GridT, mGridSize);
        cudaCheck(cudaMemcpyAsync(&gridSize, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return gridSize;
    }

    static CoordBBox getIndexBBox(const GridT *d_grid, const TreeDataT &h_treeData, cudaStream_t stream = 0)
    {
        CoordBBox box;
        std::size_t offset = GridT::memUsage() + h_treeData.mNodeOffset[3] + offsetof(RootT, mBBox);
        cudaCheck(cudaMemcpyAsync(&box, util::PtrAdd(d_grid, offset), sizeof(CoordBBox), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return box;
    }

    static uint32_t getRootTableSize(const GridT *d_grid, const TreeDataT &h_treeData, cudaStream_t stream = 0)
    {
        uint32_t tableSize = 0;
        std::size_t offset = GridT::memUsage() + h_treeData.mNodeOffset[3] + offsetof(RootT, mTableSize);
        cudaCheck(cudaMemcpyAsync(&tableSize, util::PtrAdd(d_grid, offset), sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        return tableSize;
    }

};

} // namespace cuda

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_DEVICEGRIDTRAITS_H_HAS_BEEN_INCLUDED
