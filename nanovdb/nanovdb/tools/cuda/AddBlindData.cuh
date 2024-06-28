// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/cuda/AddBlindData.cuh

    \author Ken Museth

    \date August 3, 2023

    \brief Defines function that appends blind device data to and existing device NanoGrid

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>

#include <cstring> // for std::strcpy

namespace nanovdb {// ================================================

namespace tools::cuda {// ============================================

/// @brief This function appends blind data to and existing NanoGrid
/// @tparam BuildT Build type of the grid
/// @tparam BlindDataT Type of the blind data
/// @tparam BufferT Type of the buffer used for allocation
/// @param d_grid Pointer to device grid
/// @param d_blindData Pointer to device blind data
/// @param valueCount number of values in the blind data
/// @param blindClass class of the blind data
/// @param semantics semantics of the blind data
/// @param name optional name of the blind data
/// @param pool optional pool used for allocation
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return GridHandle with blind data appended
template<typename BuildT, typename BlindDataT, typename BufferT = nanovdb::cuda::DeviceBuffer>
GridHandle<BufferT>
addBlindData(const NanoGrid<BuildT> *d_grid,
             const BlindDataT *d_blindData,
             uint64_t valueCount,
             GridBlindDataClass blindClass   = GridBlindDataClass::Unknown,
             GridBlindDataSemantic semantics = GridBlindDataSemantic::Unknown,
             const char *name = "",
             const BufferT &pool = BufferT(),
             cudaStream_t stream = 0)
{
    // In:  |-----------|--------- |-----------|
    //        old grid    old meta   old data
    // Out: |-----------|----------|----------|-----------|------------|
    //        old grid    old meta   new meta    old data    new data

    static_assert(BufferTraits<BufferT>::hasDeviceDual, "Expected BufferT to support device allocation");

    // extract byte sizes of the grid, blind meta data and blind data
    enum {GRID=0, META=1, DATA=2, CHECKSUM=3};
    uint64_t tmp[4], *d_tmp;
    cudaCheck(util::cuda::mallocAsync((void**)&d_tmp, 4*sizeof(uint64_t), stream));
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        if (auto count  = d_grid->blindDataCount()) {
            d_tmp[GRID] = util::PtrDiff(&d_grid->blindMetaData(0), d_grid);
            d_tmp[META] = count*sizeof(GridBlindMetaData);
            d_tmp[DATA] = d_grid->gridSize() - d_tmp[GRID] - d_tmp[META];
        } else {
            d_tmp[GRID] = d_grid->gridSize();
            d_tmp[META] = d_tmp[DATA] = 0u;
        }
        d_tmp[CHECKSUM] = d_grid->checksum().full();
    }); cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&tmp, d_tmp, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

    GridBlindMetaData metaData{int64_t(sizeof(GridBlindMetaData) + tmp[DATA]), valueCount,
                               sizeof(BlindDataT), semantics, blindClass, toGridType<BlindDataT>()};
    if (!metaData.isValid()) throw std::runtime_error("cudaAddBlindData: invalid combination of blind meta data");
    std::strcpy(metaData.mName, name);
    auto buffer = BufferT::create(tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA] + metaData.blindDataSize(), &pool, false);
    void *d_data = buffer.deviceData();

    // 1:   |-----------|----------|
    //        old grid    old meta
    cudaCheck(cudaMemcpyAsync(d_data, d_grid, tmp[GRID] + tmp[META], cudaMemcpyDeviceToDevice, stream));

    // 2:   |-----------|----------|----------|
    //        old grid    old meta   new meta
    cudaCheck(cudaMemcpyAsync((char*)d_data + tmp[GRID] + tmp[META], &metaData, sizeof(GridBlindMetaData), cudaMemcpyHostToDevice, stream));

    // 3:   |-----------|----------|----------|-----------|
    //        old grid    old meta   new meta   old data
    cudaCheck(cudaMemcpyAsync((char*)d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData),
                 (const char*)d_grid + tmp[GRID] + tmp[META], tmp[DATA], cudaMemcpyDeviceToDevice, stream));

    // 4:   |-----------|----------|----------|-----------|------------|
    //        old grid    old meta   new meta    old data    new data
    const size_t dataSize = valueCount*sizeof(BlindDataT);// no padding
    cudaCheck(cudaMemcpyAsync((char*)d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA],
                              d_blindData, dataSize, cudaMemcpyDeviceToDevice, stream));
    if (auto padding = metaData.blindDataSize() - dataSize) {// zero out possible padding
        cudaCheck(cudaMemsetAsync((char*)d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA] + dataSize, 0, padding, stream));
    }

    // increment grid size and blind data counter in output grid
    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        auto &grid = *reinterpret_cast<NanoGrid<BuildT>*>(d_data);
        grid.mBlindMetadataCount += 1;
        grid.mBlindMetadataOffset = d_tmp[GRID];
        auto *meta = util::PtrAdd<GridBlindMetaData>(d_data, grid.mBlindMetadataOffset);// points to first blind meta data
        for (uint32_t i=0, n=grid.mBlindMetadataCount-1; i<n; ++i, ++meta) meta->mDataOffset += sizeof(GridBlindMetaData);
        grid.mGridSize += sizeof(GridBlindMetaData) + meta->blindDataSize();// expansion with 32 byte alignment
    }); cudaCheckError();
    cudaCheck(util::cuda::freeAsync(d_tmp, stream));

    Checksum cs(tmp[CHECKSUM]);
    cuda::updateChecksum(reinterpret_cast<GridData*>(d_data), cs.mode(), stream);

    return GridHandle<BufferT>(std::move(buffer));
}// cudaAddBlindData

}// namespace tools::cuda

template<typename BuildT, typename BlindDataT, typename BufferT = cuda::DeviceBuffer>
[[deprecated("Use nanovdb::cuda::addBlindData instead")]]
GridHandle<BufferT>
cudaAddBlindData(const NanoGrid<BuildT> *d_grid,
                 const BlindDataT *d_blindData,
                 uint64_t valueCount,
                 GridBlindDataClass blindClass   = GridBlindDataClass::Unknown,
                 GridBlindDataSemantic semantics = GridBlindDataSemantic::Unknown,
                 const char *name = "",
                 const BufferT &pool = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::addBlindData<BuildT, BlindDataT, BufferT>(d_grid, d_blindData, valueCount, blindClass, semantics, name, pool, stream);
}

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED