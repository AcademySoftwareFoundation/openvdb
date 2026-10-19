// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/AddBlindData.cuh

    \author Ken Museth

    \date August 3, 2023

    \brief Defines function that appends blind device data to and existing device NanoGrid

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/tools/GridChecksum.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <cstddef> // for std::byte
#include <stdexcept> // for std::invalid_argument
#include <type_traits> // for the stream-argument check

#include <cstring> // for std::strcpy, std::strlen
#include <nanovdb/cuda/HandleStorage.h>

namespace nanovdb {// ================================================

namespace tools::cuda {// ============================================

namespace detail {

/// @brief Shared body of both addBlindData forms.
/// @param pool prototype buffer whose resource (or create() pool) allocates the output
/// @param resource resource the scratch allocates through
template<typename BuildT, typename BlindDataT, typename BufferT, typename ResourceT>
GridHandle<BufferT>
addBlindData(const NanoGrid<BuildT> *d_grid,
             const BlindDataT *d_blindData,
             uint64_t valueCount,
             GridBlindDataClass blindClass,
             GridBlindDataSemantic semantics,
             const char *name,
             const BufferT &pool,
             cudaStream_t stream,
             ResourceT &resource)
{
    // In:  |-----------|--------- |-----------|
    //        old grid    old meta   old data
    // Out: |-----------|----------|----------|-----------|------------|
    //        old grid    old meta   new meta    old data    new data

    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "addBlindData allocates stream-ordered scratch and requires an AsyncResource");

    // extract byte sizes of the grid, blind meta data and blind data
    enum {GRID=0, META=1, DATA=2, CHECKSUM=3};
    using ScratchT = nanovdb::cuda::Buffer<uint64_t, nanovdb::cuda::ResourceRef<ResourceT>>;
    uint64_t tmp[4];
    ScratchT tmpBuf(stream, nanovdb::cuda::ResourceRef<ResourceT>(resource), 4, nanovdb::cuda::noInit);
    uint64_t *d_tmp = tmpBuf.data();
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
    if (name == nullptr) throw std::invalid_argument("addBlindData: null name (pass \"\" for none)");
    if (std::strlen(name) >= GridBlindMetaData::MaxNameSize)
        throw std::invalid_argument("addBlindData: name longer than GridBlindMetaData::MaxNameSize - 1 characters");
    if (d_grid == nullptr) throw std::invalid_argument("addBlindData: null device grid");
    if (d_blindData == nullptr && valueCount > 0) throw std::invalid_argument("addBlindData: null blind data with a nonzero value count");
    std::strcpy(metaData.mName, name);
    auto buffer = nanovdb::cuda::detail::createDeviceStorage<BufferT>(tmp[GRID] + tmp[META] + sizeof(GridBlindMetaData) + tmp[DATA] + metaData.blindDataSize(),
                                                              &pool, util::cuda::currentDevice(), stream);
    void *d_data = nanovdb::cuda::detail::deviceStorageData(buffer);

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
        grid.mGridIndex = 0u; // Possibly overwriting input; returned grid has batch size 1
        grid.mGridCount = 1u;
        grid.mBlindMetadataCount += 1;
        grid.mBlindMetadataOffset = d_tmp[GRID];
        auto *meta = util::PtrAdd<GridBlindMetaData>(d_data, grid.mBlindMetadataOffset);// points to first blind meta data
        for (uint32_t i=0, n=grid.mBlindMetadataCount-1; i<n; ++i, ++meta) meta->mDataOffset += sizeof(GridBlindMetaData);
        grid.mGridSize += sizeof(GridBlindMetaData) + meta->blindDataSize();// expansion with 32 byte alignment
    }); cudaCheckError();
    tmpBuf.destroy(stream);

    Checksum cs(tmp[CHECKSUM]);
    cuda::updateChecksum(reinterpret_cast<GridData*>(d_data), cs.mode(), stream);

    nanovdb::cuda::detail::orderBeforeHandleConstruction<BufferT>(stream);
    return GridHandle<BufferT>(std::move(buffer));
}// detail::addBlindData

}// namespace detail

/// @brief This function appends blind data to and existing NanoGrid
/// @tparam BuildT Build type of the grid
/// @tparam BlindDataT Type of the blind data
/// @tparam BufferT Type of the buffer used for allocation
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param d_grid Pointer to device grid
/// @param d_blindData Pointer to device blind data
/// @param valueCount number of values in the blind data
/// @param blindClass class of the blind data
/// @param semantics semantics of the blind data
/// @param name optional name of the blind data
/// @param pool prototype buffer whose resource (or create() pool) allocates the output
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return GridHandle with blind data appended
template<typename BuildT, typename BlindDataT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
addBlindData(const NanoGrid<BuildT> *d_grid,
             const BlindDataT *d_blindData,
             uint64_t valueCount,
             GridBlindDataClass blindClass   = GridBlindDataClass::Unknown,
             GridBlindDataSemantic semantics = GridBlindDataSemantic::Unknown,
             const char *name = "",
             const BufferT &pool = BufferT(),
             cudaStream_t stream = 0)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "addBlindData: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    return detail::addBlindData(d_grid, d_blindData, valueCount, blindClass, semantics, name, pool, stream,
                                nanovdb::cuda::default_resource<ResourceT>());
}// addBlindData

/// @brief Resource form of addBlindData: the output and the scratch allocate
///        through @c resource, on @c stream. Passing a cudaStream_t selects
///        this form (a literal 0 does not: write cudaStream_t(0)); the
///        prototype-buffer form above keeps its signature and behaviour.
/// @tparam ResourceT stream-ordered memory resource, held by value; pass a
///         cuda::ResourceRef<R> to allocate through a borrowed instance
template<typename BuildT, typename BlindDataT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<nanovdb::cuda::is_async_resource<ResourceT>::value, GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
addBlindData(const NanoGrid<BuildT> *d_grid,
             const BlindDataT *d_blindData,
             uint64_t valueCount,
             GridBlindDataClass blindClass,
             GridBlindDataSemantic semantics,
             const char *name,
             cudaStream_t stream,
             ResourceT resource = ResourceT())
{
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    return detail::addBlindData(d_grid, d_blindData, valueCount, blindClass, semantics, name,
                                BufferT(stream, resource, 0, nanovdb::cuda::noInit), stream, resource);
}// addBlindData (resource form)

}// namespace tools::cuda

template<typename BuildT, typename BlindDataT, typename BufferT = cuda::DualDeviceBuffer>
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

#endif // NANOVDB_TOOLS_CUDA_ADDBLINDDATA_CUH_HAS_BEEN_INCLUDED
