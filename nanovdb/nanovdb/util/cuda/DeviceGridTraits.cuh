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

namespace nanovdb::util {

namespace cuda {

/// @brief Utility class for directly probing a device-resident IndexGrid for fundamental traits
///        of the tree topology
/// @tparam BuildT Build type for the grid
template <typename BuildT>
struct DeviceGridTraits
{
    using GridT = typename nanovdb::NanoGrid<BuildT>;
    using TreeT = typename nanovdb::NanoTree<BuildT>;
    using RootT = typename nanovdb::NanoRoot<BuildT>;
    using TreeDataT = typename TreeT::DataType;

    static TreeDataT getTreeData(const GridT *d_grid)
    {
        TreeDataT treeData;
        std::size_t offset = GridT::memUsage();
        cudaCheck(cudaMemcpy(&treeData, util::PtrAdd(d_grid, offset), sizeof(TreeDataT), cudaMemcpyDeviceToHost));
        return treeData;
    }

    static uint64_t getActiveVoxelCount(const GridT *d_grid)
    {
        uint64_t activeVoxelCount = 0;
        std::size_t offset = GridT::memUsage() + offsetof(TreeT, mVoxelCount);
        cudaCheck(cudaMemcpy(&activeVoxelCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return activeVoxelCount;
    }

    static uint64_t getValueCount(const GridT *d_grid)
    {
        uint64_t valueCount = 0;
        std::size_t offset = offsetof(GridT, mData1);
        cudaCheck(cudaMemcpy(&valueCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return valueCount;
    }

    static uint64_t getGridSize(const GridT *d_grid)
    {
        uint64_t gridSize = 0;
        std::size_t offset = offsetof(GridT, mGridSize);
        cudaCheck(cudaMemcpy(&gridSize, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return gridSize;
    }

    static CoordBBox getIndexBBox(const GridT *d_grid, const TreeDataT &h_treeData)
    {
        CoordBBox box;
        std::size_t offset = GridT::memUsage() + h_treeData.mNodeOffset[3] + offsetof(RootT, mBBox);
        cudaCheck(cudaMemcpy(&box, util::PtrAdd(d_grid, offset), sizeof(CoordBBox), cudaMemcpyDeviceToHost));
        return box;
    }

    static uint32_t getRootTableSize(const GridT *d_grid, const TreeDataT &h_treeData)
    {
        uint32_t tableSize;
        std::size_t offset = GridT::memUsage() + h_treeData.mNodeOffset[3] + offsetof(RootT, mTableSize);
        cudaCheck(cudaMemcpy(&tableSize, util::PtrAdd(d_grid, offset), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        return tableSize;
    }

};

} // namespace cuda

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_DEVICEGRIDTRAITS_H_HAS_BEEN_INCLUDED
