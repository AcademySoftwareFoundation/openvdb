// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaGridChecksum.cuh

    \author Ken Museth

    \date September 28, 2023

    \brief Compute CRC32 checksum of NanoVDB grids

*/

#ifndef NANOVDB_CUDA_GRID_CHECKSUM_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_CUDA_GRID_CHECKSUM_CUH_HAS_BEEN_INCLUDED

#include "CudaDeviceBuffer.h"// required for instantiation of move c-tor of GridHandle
#include "CudaNodeManager.cuh"
#include "../GridChecksum.h"// for
#include "../GridHandle.h"

namespace nanovdb {

namespace crc32 {

/// @bried Cuda kernel to initiate lookup table for CRC32 computation
/// @tparam T Dummy template parameter used to avoid multiple instantiations. T should be uint32_t!
/// @param d_lut Device pointer to lookup table of size 256
template <typename T>
__global__ void initLutKernel(T *d_lut)
{
    static_assert(is_same<T, uint32_t>::value,"Expected uint32_t");
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 256u) crc32::initLut(d_lut, tid);
}

/// @brief Cuda kernel that computes CRC32 checksums of blocks of data using a look-up-table
/// @param d_data device pointer to raw data from wich to compute the CRC32 checksums
/// @param d_blockCRC device pointer to array of @c blockCount checksums for each block
/// @param blockCount number of blocks and checksums
/// @param blockSize size of each block in bytes
/// @param d_lut device pointer to CRC32 Lookup Table
template <typename T>
__global__ void checksumKernel(const T *d_data, uint32_t* d_blockCRC, uint32_t blockCount, uint32_t blockSize, const uint32_t *d_lut)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < blockCount) d_blockCRC[tid] = crc32::checksum((const uint8_t*)d_data + tid * blockSize, blockSize, d_lut);
}

/// @brief Cuda kernel that computes CRC32 checksums of blocks of data (without using a look-up-table)
/// @param d_data device pointer to raw data from wich to compute the CRC32 checksums
/// @param d_blockCRC device pointer to array of @c blockCount checksums for each block
/// @param blockCount number of blocks and checksums
/// @param blockSize size of each block in bytes
template <typename T>
__global__ void checksumKernel(const T *d_data, uint32_t* d_blockCRC, uint32_t blockCount, uint32_t blockSize)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < blockCount) d_blockCRC[tid] = crc32::checksum((const uint8_t*)d_data + tid * blockSize, blockSize);
}

/// @brief Host function to allocate and initiate a Look-Up-Table of size 256 for subsequent CRC32 computation on the device
/// @param stream optional cuda stream (defaults to zero)
/// @return returns a device point to a lookup-table for CRC32 computation
/// @warning It is the responsibility of the caller to delete the returned array
inline uint32_t* cudaCreateLut(cudaStream_t stream = 0)
{
    uint32_t *d_lut;
    cudaCheck(cudaMallocAsync((void**)&d_lut, 256*sizeof(uint32_t), stream));
    initLutKernel<<<1, 256, 0, stream>>>(d_lut);
    cudaCheckError();
    return d_lut;
}

}// namespace crc

#ifdef NANOVDB_CRC32_LOG2_BLOCK_SIZE// new approach computes CRC32 checksums for each 4 KB block

/// @brief Update the checksum of a device grid
/// @param d_gridData device pointer to GridData
/// @param mode Mode of computation for the checksum.
/// @param stream optional cuda stream (defaults to zero)
/// @return The actual mode used for checksum computation. Eg. if @c d_gridData is NULL (or @c mode = ChecksumMode::Disable)
///         then ChecksumMode::Disable is always returned. Elseif the grid has no nodes or blind data ChecksumMode::Partial
///         is always returnd (even if @c mode = ChecksumMode::Full).
inline ChecksumMode cudaGridChecksum(GridData *d_gridData, ChecksumMode mode = ChecksumMode::Partial, cudaStream_t stream = 0)
{
    if (d_gridData == nullptr || mode == ChecksumMode::Disable) return ChecksumMode::Disable;

    static constexpr unsigned int mNumThreads = 128;// seems faster than the old value of 256!
    auto numBlocks = [&](unsigned int n)->unsigned int{return (n + mNumThreads - 1) / mNumThreads;};
    uint8_t  *d_begin = reinterpret_cast<uint8_t*>(d_gridData);
    uint32_t *d_lut = crc32::cudaCreateLut(stream);// allocate and generate device LUT for CRC32
    uint64_t size[2], *d_size;// {total size of grid, partial size for first checksum}
    cudaCheck(cudaMallocAsync((void**)&d_size, 2*sizeof(uint64_t), stream));

    // Compute CRC32 checksum of GridData, TreeData, RootData (+tiles), but exclude GridData::mMagic and GridData::mChecksum
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        d_size[0] = d_gridData->mGridSize;
        uint8_t *d_mid = d_gridData->template nodePtr<2>();
        if (d_mid == nullptr) {// no upper nodes
            if (d_gridData->mBlindMetadataCount) {
                d_mid = d_begin + d_gridData->mBlindMetadataOffset;// exclude blind data from partial checksum
            } else {
                d_mid = d_begin + d_gridData->mGridSize;// no nodes or blind data, so partial checksum is computed on the entire grid buffer
            }
        }
        d_size[1] = d_mid - d_begin;
        uint32_t *p = reinterpret_cast<uint32_t*>(&(d_gridData->mChecksum));
        p[0] = crc32::checksum(d_begin + 16u, d_mid, d_lut);// exclude GridData::mMagic and GridData::mChecksum
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(size, d_size, 2*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaFreeAsync(d_size, stream));

    if (mode != ChecksumMode::Full || size[0] == size[1]) return ChecksumMode::Partial;

    // Compute CRC32 checksum of 4K block of everything remaining in the buffer, i.e. nodes and blind data
    const uint8_t *d_mid = d_begin + size[1], *d_end = d_begin + size[0];
    uint32_t *d_checksums;// 4096 byte chunks
    const uint64_t checksumCount = (d_end - d_mid) >> NANOVDB_CRC32_LOG2_BLOCK_SIZE;// 4 KB (4096 byte)
    cudaCheck(cudaMallocAsync((void**)&d_checksums, checksumCount*sizeof(uint32_t), stream));
    cudaLambdaKernel<<<numBlocks(checksumCount), mNumThreads, 0, stream>>>(checksumCount, [=] __device__(size_t tid) {
        uint32_t size = 1<<NANOVDB_CRC32_LOG2_BLOCK_SIZE;
        if (tid+1 == checksumCount) size += d_end - d_mid - (checksumCount<<NANOVDB_CRC32_LOG2_BLOCK_SIZE);
        d_checksums[tid] = crc32::checksum(d_mid + (tid<<NANOVDB_CRC32_LOG2_BLOCK_SIZE), size, d_lut);
    });
    // Compute a final CRC32 checksum of all the 4K blocks
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        uint32_t *p = reinterpret_cast<uint32_t*>(&(d_gridData->mChecksum));
        p[1] = crc32::checksum((const uint8_t*)d_checksums, checksumCount*sizeof(uint32_t), d_lut);
    });
    cudaCheckError();
    cudaCheck(cudaFreeAsync(d_checksums, stream));
    cudaCheck(cudaFreeAsync(d_lut, stream));

    return ChecksumMode::Full;
}// cudaGridChecksum

template <typename BuildT>
inline ChecksumMode cudaGridChecksum(NanoGrid<BuildT> *d_grid, ChecksumMode mode = ChecksumMode::Partial, cudaStream_t stream = 0)
{
    return cudaGridChecksum(reinterpret_cast<GridData*>(d_grid), mode, stream);
}

inline GridChecksum cudaGetGridChecksum(GridData *d_gridData, cudaStream_t stream = 0)
{
    uint64_t checksum, *d_checksum;
    cudaCheck(cudaMallocAsync((void**)&d_checksum, sizeof(uint64_t), stream));
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {*d_checksum = d_gridData->mChecksum;});
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(&checksum, d_checksum, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaFreeAsync(d_checksum, stream));
    return GridChecksum(checksum);;
}

inline ChecksumMode cudaUpdateGridChecksum(GridData *d_gridData, cudaStream_t stream = 0)
{
    return cudaGridChecksum(d_gridData, cudaGetGridChecksum(d_gridData, stream).mode(), stream);
}

#else

template <typename ValueT>
void cudaGridChecksum(NanoGrid<ValueT> *d_grid, ChecksumMode mode = ChecksumMode::Partial, cudaStream_t stream = 0)
{
    if (d_grid == nullptr || mode == ChecksumMode::Disable) return;

    static constexpr unsigned int mNumThreads = 128;// seems faster than the old value of 256!
    auto numBlocks = [&](unsigned int n)->unsigned int{return (n + mNumThreads - 1) / mNumThreads;};

    uint32_t *d_lut = crc32::cudaCreateLut(stream);// allocate and generate device LUT for CRC32
    uint64_t size[2], *d_size;
    cudaCheck(cudaMallocAsync((void**)&d_size, 2*sizeof(uint64_t), stream));
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        d_size[0] = d_grid->gridSize();
        d_size[1] = d_grid->memUsage() + d_grid->tree().memUsage() + d_grid->tree().root().memUsage();
        const uint8_t *begin = reinterpret_cast<const uint8_t*>(d_grid);
        uint32_t *p = reinterpret_cast<uint32_t*>(&(d_grid->mChecksum));
        p[0] = crc32::checksum(begin + 16u, begin + d_size[1], d_lut);// exclude mMagic and mChecksum
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(size, d_size, 2*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    cudaCheckError();

    if (mode != ChecksumMode::Full) return;

    // Get node counts
    uint32_t nodeCount[3], *d_nodeCount, *d_checksums, *d_ptr;
    cudaCheck(cudaMallocAsync((void**)&d_nodeCount, 3*sizeof(uint32_t), stream));
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        auto &tree = d_grid->tree();
        for (int i = 0; i < 3; ++i) d_nodeCount[i] = tree.nodeCount(i);
    });
    cudaCheckError();
    cudaCheck(cudaMemcpyAsync(nodeCount, d_nodeCount, 3*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaFreeAsync(d_nodeCount, stream));
    cudaCheck(cudaMallocAsync((void**)&d_checksums, (nodeCount[0]+nodeCount[1]+nodeCount[2])*sizeof(uint32_t), stream));

    auto nodeMgrHandle = cudaCreateNodeManager<ValueT, CudaDeviceBuffer>(d_grid, CudaDeviceBuffer(), stream);
    auto *d_nodeMgr = nodeMgrHandle.template deviceMgr<ValueT>();
    NANOVDB_ASSERT(isValid(d_nodeMgr));
    d_ptr = d_checksums;

    // very slow due to large nodes
    cudaLambdaKernel<<<numBlocks(nodeCount[2]), mNumThreads, 0, stream>>>(nodeCount[2], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->upper(uint32_t(tid));
        d_ptr[tid] = crc32::checksum((const uint8_t*)&node, node.memUsage(), d_lut);
    });
    cudaCheckError();

    d_ptr += nodeCount[2];
    cudaLambdaKernel<<<numBlocks(nodeCount[1]), mNumThreads, 0, stream>>>(nodeCount[1], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->lower(uint32_t(tid));
        d_ptr[tid] = crc32::checksum((const uint8_t*)&node, node.memUsage(), d_lut);
    });
    cudaCheckError();

    d_ptr += nodeCount[1];
    cudaLambdaKernel<<<numBlocks(nodeCount[0]), mNumThreads, 0, stream>>>(nodeCount[0], [=] __device__(size_t tid) {
        auto &node = d_nodeMgr->leaf(uint32_t(tid));
        d_ptr[tid] = crc32::checksum((const uint8_t*)&node, node.memUsage(), d_lut);
    });
    cudaCheckError();

    // to-do: process blind data
    cudaLambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {
        uint32_t *p = reinterpret_cast<uint32_t*>(&(d_grid->mChecksum));
        const uint8_t *begin = reinterpret_cast<const uint8_t*>(d_checksums);
        p[1] = crc32::checksum(begin, d_nodeMgr->tree().totalNodeCount()*sizeof(uint32_t), d_lut);
    });
    cudaCheckError();

    cudaCheck(cudaFreeAsync(d_size, stream));
    cudaCheck(cudaFreeAsync(d_checksums, stream));
    cudaCheck(cudaFreeAsync(d_lut, stream));
}// cudaGridChecksum

#endif

}// namespace nanovdb

#endif // NANOVDB_CUDA_GRID_CHECKSUM_CUH_HAS_BEEN_INCLUDED
