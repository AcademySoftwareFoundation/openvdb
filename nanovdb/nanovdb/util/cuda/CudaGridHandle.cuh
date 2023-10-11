// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaGridHandle.cuh

    \author Ken Museth, Doyub Kim

    \date August 3, 2023

    \brief Contains cuda kernels for GridHandle

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_CUDA_GRID_HANDLE_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_GRID_HANDLE_CUH_HAS_BEEN_INCLUDED

#include "CudaDeviceBuffer.h"// required for instantiation of move c-tor of GridHandle
#include "CudaGridChecksum.cuh"// for cudaUpdateChecksum
#include "../GridHandle.h"

namespace nanovdb {

namespace {// anonymous namespace
__global__ void cudaCpyMetaData(const GridData *data, GridHandleMetaData *meta){cpyMetaData(data, meta);}
__global__ void cudaUpdateGridCount(GridData *data, uint32_t gridIndex, uint32_t gridCount, bool *d_dirty){
    NANOVDB_ASSERT(gridIndex < gridCount);
    if (*d_dirty = data->mGridIndex != gridIndex || data->mGridCount != gridCount) {
        data->mGridIndex = gridIndex;
        data->mGridCount = gridCount;
        if (data->mChecksum == GridChecksum::EMPTY) *d_dirty = false;// no need to update checksum if it didn't already exist
        //data->mChecksum  = GridChecksum::EMPTY;// disable the checksum (in the future this should call cudaGridChecksum)
    }
}
}// anonymous namespace

template<typename BufferT>
template<typename T, typename enable_if<BufferTraits<T>::hasDeviceDual, int>::type>
GridHandle<BufferT>::GridHandle(T&& buffer)
{
    static_assert(is_same<T,BufferT>::value, "Expected U==BufferT");
    mBuffer = std::move(buffer);
    if (auto *data = reinterpret_cast<const GridData*>(mBuffer.data())) {
        if (!data->isValid()) throw std::runtime_error("GridHandle was constructed with an invalid host buffer");
        mMetaData.resize(data->mGridCount);
        cpyMetaData(data, mMetaData.data());
    } else {
        if (auto *d_data = reinterpret_cast<const GridData*>(mBuffer.deviceData())) {
            GridData tmp;
            cudaCheck(cudaMemcpy(&tmp, d_data, sizeof(GridData), cudaMemcpyDeviceToHost));
            if (!tmp.isValid()) throw std::runtime_error("GridHandle was constructed with an invalid device buffer");
            GridHandleMetaData *d_metaData;
            cudaMalloc((void**)&d_metaData, tmp.mGridCount*sizeof(GridHandleMetaData));
            cudaCpyMetaData<<<1,1>>>(d_data, d_metaData);
            mMetaData.resize(tmp.mGridCount);
            cudaCheck(cudaMemcpy(mMetaData.data(), d_metaData,tmp.mGridCount*sizeof(GridHandleMetaData), cudaMemcpyDeviceToHost));
            cudaCheck(cudaFree(d_metaData));
        }
    }
}// GridHandle(T&& buffer)

// Dummy function that ensures instantiation of the move-constructor above when BufferT=CudaDeviceBuffer
namespace {auto __dummy(){return GridHandle<CudaDeviceBuffer>(std::move(CudaDeviceBuffer()));}}

template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline typename enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
cudaSplitGridHandles(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    const uint8_t *ptr = handle.deviceData();
    if (ptr == nullptr) return VectorT<GridHandle<BufferT>>();
    VectorT<GridHandle<BufferT>> handles(handle.gridCount());
    bool dirty, *d_dirty;// use this to check if the checksum needs to be recomputed
    cudaCheck(cudaMallocAsync((void**)&d_dirty, sizeof(bool), stream));
    for (uint32_t n=0; n<handle.gridCount(); ++n) {
        auto buffer = BufferT::create(handle.gridSize(n), other, false, stream);
        GridData *dst = reinterpret_cast<GridData*>(buffer.deviceData());
        const GridData *src = reinterpret_cast<const GridData*>(ptr);
        cudaCheck(cudaMemcpyAsync(dst, src, handle.gridSize(n), cudaMemcpyDeviceToDevice, stream));
        cudaUpdateGridCount<<<1, 1, 0, stream>>>(dst, 0u, 1u, d_dirty);
        cudaCheckError();
        cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        if (dirty) cudaGridChecksum(dst, ChecksumMode::Partial);
        handles[n] = GridHandle<BufferT>(std::move(buffer));
        ptr += handle.gridSize(n);
    }
    cudaCheck(cudaFreeAsync(d_dirty, stream));
    //cudaCheck(cudaFreeAsync(d_lut,  stream));
    return std::move(handles);
}// cudaSplitGridHandles

template<typename BufferT, template <class, class...> class VectorT = std::vector>
inline typename enable_if<BufferTraits<BufferT>::hasDeviceDual, VectorT<GridHandle<BufferT>>>::type
splitDeviceGrids(const GridHandle<BufferT> &handle, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cudaSplitGridHandles(handle, other, stream); }

template<typename BufferT, template <class, class...> class VectorT>
inline typename enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
cudaMergeGridHandles(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{
    uint64_t size = 0u;
    uint32_t counter = 0u, gridCount = 0u;
    for (auto &h : handles) {
        gridCount += h.gridCount();
        for (uint32_t n=0; n<h.gridCount(); ++n) size += h.gridSize(n);
    }
    auto buffer = BufferT::create(size, other, false, stream);
    uint8_t *dst = buffer.deviceData();
    bool dirty, *d_dirty;// use this to check if the checksum needs to be recomputed
    cudaCheck(cudaMallocAsync((void**)&d_dirty, sizeof(bool), stream));
    for (auto &h : handles) {
        const uint8_t *src = h.deviceData();
        for (uint32_t n=0; n<h.gridCount(); ++n) {
            cudaCheck(cudaMemcpyAsync(dst, src, h.gridSize(n), cudaMemcpyDeviceToDevice, stream));
            GridData *data = reinterpret_cast<GridData*>(dst);
            cudaUpdateGridCount<<<1, 1, 0, stream>>>(data, counter++, gridCount, d_dirty);
            cudaCheckError();
            cudaCheck(cudaMemcpyAsync(&dirty, d_dirty, sizeof(bool), cudaMemcpyDeviceToHost, stream));
            if (dirty) cudaGridChecksum(data, ChecksumMode::Partial);
            dst += h.gridSize(n);
            src += h.gridSize(n);
        }
    }
    cudaCheck(cudaFreeAsync(d_dirty, stream));
    return GridHandle<BufferT>(std::move(buffer));
}// cudaMergeGridHandles

template<typename BufferT, template <class, class...> class VectorT>
inline typename enable_if<BufferTraits<BufferT>::hasDeviceDual, GridHandle<BufferT>>::type
mergeDeviceGrids(const VectorT<GridHandle<BufferT>> &handles, const BufferT* other = nullptr, cudaStream_t stream = 0)
{ return cudaMergeGridHandles(handles, other, stream); }

} // namespace nanovdb

#endif // NANOVDB_CUDA_GRID_HANDLE_CUH_HAS_BEEN_INCLUDED
