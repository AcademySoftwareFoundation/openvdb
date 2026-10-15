// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*
    \file GridHelpers.cuh

    \author Efty Sifakis

    \date Jun 5, 2025

    \brief This file implements helpers for coordinating use of device-hosted grid data

*/

#ifndef NANOVDB_UTIL_CUDA_GRIDHELPERS_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_CUDA_GRIDHELPERS_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>

namespace nanovdb::util {

namespace cuda {

template <typename BuildT>
struct DeviceGridTraits
{
    using GridT = typename nanovdb::NanoGrid<BuildT>;
    using TreeT = typename nanovdb::NanoTree<BuildT>;
    using RootT = typename nanovdb::NanoRoot<BuildT>;
    using TreeDataT = typename TreeT::DataType;

    static TreeDataT getTreeData(GridT *d_grid)
    {
        TreeDataT treeData;
        std::size_t offset = GridT::memUsage();
        cudaCheck(cudaMemcpy(&treeData, util::PtrAdd(d_grid, offset), sizeof(TreeDataT), cudaMemcpyDeviceToHost));
        return treeData;
    }

    static uint64_t getActiveVoxelCount(GridT *d_grid)
    {
        uint64_t activeVoxelCount = 0;
        std::size_t offset = GridT::memUsage() + offsetof(TreeT, mVoxelCount);
        cudaCheck(cudaMemcpy(&activeVoxelCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return activeVoxelCount;
    }

    static uint64_t getValueCount(GridT *d_grid)
    {
        uint64_t valueCount = 0;
        std::size_t offset = offsetof(GridT, mData1);
        cudaCheck(cudaMemcpy(&valueCount, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return valueCount;
    }

    static uint64_t getGridSize(GridT *d_grid)
    {
        uint64_t gridSize = 0;
        std::size_t offset = offsetof(GridT, mGridSize);
        cudaCheck(cudaMemcpy(&gridSize, util::PtrAdd(d_grid, offset), sizeof(uint64_t), cudaMemcpyDeviceToHost));
        return gridSize;
    }

    static CoordBBox getIndexBBox(GridT *d_grid, const TreeDataT &h_treeData)
    {
        CoordBBox box;
        std::size_t offset = GridT::memUsage() + h_treeData.mNodeOffset[3] + offsetof(RootT, mBBox);
        cudaCheck(cudaMemcpy(&box, util::PtrAdd(d_grid, offset), sizeof(CoordBBox), cudaMemcpyDeviceToHost));
        return box;
    }

};

} // namespace cuda

} // namespace nanovdb::util

#endif // NANOVDB_UTIL_CUDA_GRIDHELPERS_H_HAS_BEEN_INCLUDED
