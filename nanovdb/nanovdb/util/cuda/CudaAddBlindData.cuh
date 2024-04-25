// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaAddBlindData.cuh

    \author Ken Museth

    \date August 3, 2023

    \brief Defines function that appends blind device data to and existing device NanoGrid

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_CUDA_ADD_BLIND_DATA_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_CUDA_ADD_BLIND_DATA_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include "CudaDeviceBuffer.h"
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/cuda/CudaUtils.h>
#include <nanovdb/util/GridChecksum.h>
#include <nanovdb/util/cuda/CudaGridChecksum.cuh>

#include <cstring> // for std::strcpy

namespace nanovdb {

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
template<typename BuildT, typename BlindDataT, typename BufferT = CudaDeviceBuffer>
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
    // In:  |-----------|--------- |-----------|
    //        old grid    old meta   old data
    // Out: |-----------|----------|----------|-----------|------------|
    //        old grid    old meta   new meta    old data    new data

    static_assert(BufferTraits<BufferT>::hasDeviceDual, "Expected BufferT to support device allocation");

    // extract byte sizes of the grid, blind meta data and blind data
    enum {GRID=0, META=1, DATA=2, CHECKSUM=3};
    uint64_t tmp[4], *d_tmp;
    cudaCheck(CUDA_MALLOC((void**)&d_tmp, 4*sizeof(uint64_t), stream));
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        if (auto count = d_grid->blindDataCount()) {
            d_tmp[GRID] = PtrDiff(&d_grid->blindMetaData(0), d_grid);
            d_tmp[META] = count*sizeof(GridBlindMetaData);
            d_tmp[DATA] = d_grid->gridSize() - d_tmp[GRID] - d_tmp[META];
        } else {
            d_tmp[GRID] = d_grid->gridSize();
            d_tmp[META] = d_tmp[DATA] = 0u;
        }
        d_tmp[CHECKSUM] = d_grid->checksum();
    }); cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&tmp, d_tmp, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));

    GridBlindMetaData metaData{int64_t(sizeof(GridBlindMetaData) + tmp[DATA]), valueCount,
                               sizeof(BlindDataT), semantics, blindClass, mapToGridType<BlindDataT>()};
    if (!metaData.isValid()) throw std::runtime_error("cudaAddBlindData: invalid combination of blind meta data");
    std::strcpy(metaData.mName, name);
    auto buffer = BufferT::create(tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA] + metaData.blindDataSize(), &pool, false);
    auto d_data = buffer.deviceData();

    // 1:   |-----------|----------|
    //        old grid    old meta
    cudaCheck(cudaMemcpyAsync(d_data, d_grid, tmp[GRID] + tmp[META], cudaMemcpyDeviceToDevice, stream));

    // 2:   |-----------|----------|----------|
    //        old grid    old meta   new meta
    cudaCheck(cudaMemcpyAsync(d_data + tmp[GRID] + tmp[META], &metaData, sizeof(GridBlindMetaData), cudaMemcpyHostToDevice, stream));

    // 3:   |-----------|----------|----------|-----------|
    //        old grid    old meta   new meta   old data
    cudaCheck(cudaMemcpyAsync(d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData),
                 (const char*)d_grid + tmp[GRID] + tmp[META], tmp[DATA], cudaMemcpyDeviceToDevice, stream));

    // 4:   |-----------|----------|----------|-----------|------------|
    //        old grid    old meta   new meta    old data    new data
    const size_t dataSize = valueCount*sizeof(BlindDataT);// no padding
    cudaCheck(cudaMemcpyAsync(d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA],
                              d_blindData, dataSize, cudaMemcpyDeviceToDevice, stream));
    if (auto padding = metaData.blindDataSize() - dataSize) {// zero out possible padding
        cudaCheck(cudaMemsetAsync(d_data + tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA] + dataSize, 0, padding, stream));
    }

    // increment grid size and blind data counter in output grid
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        auto &grid = *reinterpret_cast<NanoGrid<BuildT>*>(d_data);
        grid.mBlindMetadataCount += 1;
        grid.mBlindMetadataOffset = d_tmp[GRID];
        auto *meta = PtrAdd<GridBlindMetaData>(d_data, grid.mBlindMetadataOffset);// points to first blind meta data
        for (uint32_t i=0, n=grid.mBlindMetadataCount-1; i<n; ++i, ++meta) meta->mDataOffset += sizeof(GridBlindMetaData);
        grid.mGridSize += sizeof(GridBlindMetaData) + meta->blindDataSize();// expansion with 32 byte alignment
    }); cudaCheckError();
    cudaCheck(CUDA_FREE(d_tmp, stream));

    GridChecksum cs(tmp[CHECKSUM]);
    cudaGridChecksum(reinterpret_cast<GridData*>(d_data), cs.mode());

    return GridHandle<BufferT>(std::move(buffer));
}// cudaAddBlindData

}// nanovdb namespace

#endif // NVIDIA_CUDA_ADD_BLIND_DATA_CUH_HAS_BEEN_INCLUDED