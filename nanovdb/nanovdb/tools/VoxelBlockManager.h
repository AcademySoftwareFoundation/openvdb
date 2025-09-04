// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/VoxelBlockManager.h

    \author Eftychios Sifakis

    \date July 24, 2025

    \brief Implements a handle class holding the metadata involved in invocations
           of the VoxelBlockManager structure.
           The VoxelBlockManager is an acceleration structure for sequential access
           and stencil operations over solely the active voxels of an OnIndexGrid
           which enables SIMT parallelism independent of occupancy.
*/


#ifndef NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED
#define NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/cuda/Util.h>

namespace nanovdb {

namespace tools {

/// @brief This class serves to manage the raw memory buffers for the metadata of a NanoVDB VoxelBlockManager
template<typename BufferT>
class VoxelBlockManagerHandle
{
    BufferT mFirstLeafID;
    BufferT mJumpMap;
    uint64_t mBlockCount{0};
    uint64_t mFirstOffset{0};
    uint64_t mLastOffset{0};
    uint32_t mLowerCount{0};

public:
    /// @brief Move constructor from metadata buffers
    VoxelBlockManagerHandle(BufferT&& firstLeafID, BufferT&& jumpMap, const uint64_t blockCount,
        const uint64_t firstOffset, const uint64_t lastOffset, const uint64_t lowerCount) {
        mFirstLeafID = std::move(firstLeafID);
        mJumpMap = std::move(jumpMap);
        mBlockCount = blockCount;
        mFirstOffset = firstOffset;
        mLastOffset = lastOffset;
        mLowerCount = lowerCount;
    }

    VoxelBlockManagerHandle() = default;
    VoxelBlockManagerHandle(const VoxelBlockManagerHandle&) = delete;
    VoxelBlockManagerHandle& operator=(const VoxelBlockManagerHandle&) = delete;

    VoxelBlockManagerHandle& operator=(VoxelBlockManagerHandle&& other) noexcept {
        mFirstLeafID = std::move(other.mFirstLeafID);
        mJumpMap = std::move(other.mJumpMap);
        mBlockCount = other.mblockCount;
        mFirstOffset = other.mfirstOffset;
        mLastOffset = other.mlastOffset;
        mLowerCount = other.mlowerCount;
        return *this;
    }

    VoxelBlockManagerHandle(VoxelBlockManagerHandle&& other) noexcept {
        mFirstLeafID = std::move(other.mFirstLeafID);
        mJumpMap = std::move(other.mJumpMap);
        mBlockCount = other.mblockCount;
        mFirstOffset = other.mfirstOffset;
        mLastOffset = other.mlastOffset;
        mLowerCount = other.mlowerCount;
    }

    ~VoxelBlockManagerHandle() { this->reset(); }
    /// @brief clear the buffer
    void reset() { mFirstLeafID.clear(); mJumpMap.clear(); mBlockCount = 0; }

    /// @brief Returns a non-const pointer to the firstLeafID device-hosted data
    ///
    /// @warning Note that the return pointer can be NULL if the VoxelBlockManagerHandle was not initialized
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, uint32_t*>::type
    deviceFirstLeafID() { return static_cast<uint32_t*>(mFirstLeafID.deviceData()); }

    /// @brief Returns a const pointer to the firstLeafID device-hosted data
    ///
    /// @warning Note that the return pointer can be NULL if the VoxelBlockManagerHandle was not initialized
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const uint32_t*>::type
    deviceFirstLeafID() const { return static_cast<const uint32_t*>(mFirstLeafID.deviceData()); }

    /// @brief Returns a non-const pointer to the jumpMap device-hosted data
    ///
    /// @warning Note that the return pointer can be NULL if the VoxelBlockManagerHandle was not initialized
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, uint64_t*>::type
    deviceJumpMap() { return static_cast<uint64_t*>(mJumpMap.deviceData()); }

    /// @brief Returns a const pointer to the jumpMap device-hosted data
    ///
    /// @warning Note that the return pointer can be NULL if the VoxelBlockManagerHandle was not initialized
    template<typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, const uint64_t*>::type
    deviceJumpMap() const { return static_cast<const uint64_t*>(mJumpMap.deviceData()); }

    /// @brief Returns the number of voxel blocks in the VoxelBlockManager
    uint64_t blockCount() const { return mBlockCount; }

    /// @brief Sets the first and last voxel indices (linear offsets) associated with this VoxelBlockManager
    void setOffsets(const uint64_t firstOffset, const uint64_t lastOffset) { mFirstOffset = firstOffset; mLastOffset = lastOffset; }

    /// @brief Returns the first voxel index (linear offset) associated with this VoxelBlockManager
    uint64_t firstOffset() const { return mFirstOffset; }

    /// @brief Returns the last voxel index (linear offset) associated with this VoxelBlockManager
    uint64_t lastOffset() const { return mLastOffset; }

    /// @brief Sets the count of lower internal nodes in the IndexGrid associated with this VoxelBlockManager
    void setLowerCount(const uint32_t lowerCount) { mLowerCount = lowerCount; }

    /// @brief Returns the count of lower internal nodes in the IndexGrid associated with this VoxelBlockManager
    uint32_t lowerCount() const { return mLowerCount; }

    /// @brief Allocates new buffers for metadata, adequate for storing a VoxelBlockManager of the specified block size
    template<int BlockWidthLog2, typename U = BufferT>
    typename util::enable_if<BufferTraits<U>::hasDeviceDual, void>::type
    deviceResize(const uint64_t blockCount, cudaStream_t stream = 0) {
        static constexpr int BlockWidth = 1 << BlockWidthLog2;
        reset();
        int device = 0;
        cudaCheck(cudaGetDevice(&device));
        mBlockCount = blockCount;
        mFirstLeafID = BufferT::create(blockCount*sizeof(uint32_t), nullptr, device, stream);
        mJumpMap = BufferT::create(blockCount*(BlockWidth/8), nullptr, device, stream);
    }

}; // VoxelBlockManagerHandle

} // namespace tools

} // namespace nanovdb

#if defined(__CUDACC__)
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#endif// defined(__CUDACC__)

#endif // NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED
