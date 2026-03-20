// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/VoxelBlockManager.h

    \author Eftychios Sifakis

    \date July 24, 2025

    \brief Implements a handle class holding the metadata involved in invocations
           of the VoxelBlockManager structure, and host-side build functions.
           The VoxelBlockManager is an acceleration structure for sequential access
           and stencil operations over solely the active voxels of an OnIndexGrid
           which enables SIMT parallelism independent of occupancy.
*/


#ifndef NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED
#define NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>

#include <algorithm>
#include <cstring>
#include <execution>

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

public:
    /// @brief Constructor from metadata buffers (used by buildVoxelBlockManager)
    VoxelBlockManagerHandle(BufferT&& firstLeafID, BufferT&& jumpMap, const uint64_t blockCount,
        const uint64_t firstOffset, const uint64_t lastOffset) {
        mFirstLeafID = std::move(firstLeafID);
        mJumpMap = std::move(jumpMap);
        mBlockCount = blockCount;
        mFirstOffset = firstOffset;
        mLastOffset = lastOffset;
    }

    VoxelBlockManagerHandle() = default;
    VoxelBlockManagerHandle(const VoxelBlockManagerHandle&) = delete;
    VoxelBlockManagerHandle& operator=(const VoxelBlockManagerHandle&) = delete;

    VoxelBlockManagerHandle& operator=(VoxelBlockManagerHandle&& other) noexcept {
        mFirstLeafID = std::move(other.mFirstLeafID);
        mJumpMap = std::move(other.mJumpMap);
        mBlockCount = other.mBlockCount;
        mFirstOffset = other.mFirstOffset;
        mLastOffset = other.mLastOffset;
        return *this;
    }

    VoxelBlockManagerHandle(VoxelBlockManagerHandle&& other) noexcept {
        mFirstLeafID = std::move(other.mFirstLeafID);
        mJumpMap = std::move(other.mJumpMap);
        mBlockCount = other.mBlockCount;
        mFirstOffset = other.mFirstOffset;
        mLastOffset = other.mLastOffset;
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

    /// @brief Returns the first voxel index (linear offset) associated with this VoxelBlockManager
    uint64_t firstOffset() const { return mFirstOffset; }

    /// @brief Returns the last voxel index (linear offset) associated with this VoxelBlockManager
    uint64_t lastOffset() const { return mLastOffset; }

    /// @brief Returns a non-const pointer to the firstLeafID host-side data
    uint32_t* hostFirstLeafID() { return static_cast<uint32_t*>(mFirstLeafID.data()); }

    /// @brief Returns a const pointer to the firstLeafID host-side data
    const uint32_t* hostFirstLeafID() const { return static_cast<const uint32_t*>(mFirstLeafID.data()); }

    /// @brief Returns a non-const pointer to the jumpMap host-side data
    uint64_t* hostJumpMap() { return static_cast<uint64_t*>(mJumpMap.data()); }

    /// @brief Returns a const pointer to the jumpMap host-side data
    const uint64_t* hostJumpMap() const { return static_cast<const uint64_t*>(mJumpMap.data()); }

}; // VoxelBlockManagerHandle

// --------------------------> buildVoxelBlockManager (CPU) <---------------------------------

/// @brief Rebuild a VoxelBlockManager in-place using a pre-allocated handle.
///        Zeros the jumpMap and recomputes both metadata arrays. This overload
///        performs no memory allocation and is suitable for repeated builds
///        (e.g. benchmarking, or rebuilding after a topology-preserving update).
/// @tparam BlockWidthLog2 Log2 of the number of active voxels per VBM block
/// @tparam BufferT Buffer type of the handle (must provide host-accessible data())
/// @param grid Host-accessible ValueOnIndex grid (must satisfy isSequential())
/// @param handle Pre-allocated handle whose blockCount/firstOffset/lastOffset are
///               already set to match the grid
template<int BlockWidthLog2, typename BufferT>
void buildVoxelBlockManager(const NanoGrid<ValueOnIndex>* grid, VoxelBlockManagerHandle<BufferT>& handle)
{
    static constexpr uint64_t BlockWidth = uint64_t(1) << BlockWidthLog2;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    NANOVDB_ASSERT(grid->isSequential());
    if (!handle.blockCount()) return;

    uint32_t* firstLeafID = handle.hostFirstLeafID();
    uint64_t* jumpMap = handle.hostJumpMap();
    const uint64_t nBlocks = handle.blockCount();
    const uint64_t firstOffset = handle.firstOffset();
    const uint64_t lastOffset = handle.lastOffset();

    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)

    std::memset(jumpMap, 0, nBlocks * JumpMapLength * sizeof(uint64_t));

    const auto& tree = grid->tree();
    const auto* firstLeaf = tree.getFirstNode<0>();
    const uint32_t leafCount = tree.nodeCount(0);

    std::for_each(std::execution::par, firstLeaf, firstLeaf + leafCount,
        [&](const NanoLeaf<ValueOnIndex>& leaf) {
            const uint64_t leafFirstOffset = leaf.data()->firstOffset();
            const uint64_t leafValueCount = leaf.data()->valueCount();
            const uint64_t leafLastOffset = leafFirstOffset + leafValueCount - 1;

            if (leafFirstOffset > lastOffset || leafLastOffset < firstOffset) return;

            const uint64_t leafIndex = static_cast<uint64_t>(&leaf - firstLeaf);

            const uint64_t lastBlock = std::min<uint64_t>(
                (leafLastOffset - firstOffset) >> BlockWidthLog2, nBlocks - 1);
            const uint64_t firstBlock = (leafFirstOffset < firstOffset) ? 0 :
                (leafFirstOffset - firstOffset) >> BlockWidthLog2;

            // For blocks after firstBlock, this leaf is the first leaf of each
            for (uint64_t b = lastBlock; b > firstBlock; --b)
                firstLeafID[b] = static_cast<uint32_t>(leafIndex);

            if (leafFirstOffset < firstOffset) {
                firstLeafID[0] = static_cast<uint32_t>(leafIndex);
                return;
            }

            const uint64_t offsetInBlock = (leafFirstOffset - 1) & (BlockWidth - 1);
            if (!offsetInBlock) {
                // Leaf starts exactly at a block boundary: register in firstLeafID
                firstLeafID[firstBlock] = static_cast<uint32_t>(leafIndex);
            } else {
                // Leaf starts in the interior of a block: mark in jumpMap with atomic OR
                util::atomicOr(&jumpMap[firstBlock * JumpMapLength + (offsetInBlock >> 6)],
                               uint64_t(1) << (offsetInBlock & 0x3f));
            }
        });
}

/// @brief Allocate buffers and build a VoxelBlockManager on the host from a
///        ValueOnIndex grid. Uses std::execution::par to process lower internal
///        nodes in parallel.
/// @tparam BlockWidthLog2  Log2 of the number of active voxels per VBM block
/// @tparam BufferT         Buffer type for the returned handle (default: HostBuffer)
/// @param grid         Host-accessible ValueOnIndex grid (must satisfy isSequential())
/// @param firstOffset  First active-voxel offset covered by this VBM; must satisfy
///                     firstOffset == 1 (mod BlockWidth). Pass 0 (default) to use 1,
///                     which covers the full grid from the first active voxel.
/// @param lastOffset   Last active-voxel offset covered by this VBM. Pass 0 (default)
///                     to use grid->activeVoxelCount(), covering the full grid.
/// @param nBlocks      Allocated capacity in blocks; must be >=
///                     ceil((lastOffset - firstOffset + 1) / BlockWidth). Pass 0
///                     (default) to use the minimum required capacity.
/// @param pool         Optional pool buffer for allocation (passed to BufferT::create)
/// @return A fully constructed VoxelBlockManagerHandle
template<int BlockWidthLog2, typename BufferT = HostBuffer>
VoxelBlockManagerHandle<BufferT>
buildVoxelBlockManager(
    const NanoGrid<ValueOnIndex>* grid,
    uint64_t firstOffset = 0,
    uint64_t lastOffset = 0,
    uint64_t nBlocks = 0,
    const BufferT* pool = nullptr)
{
    static constexpr uint64_t BlockWidth = uint64_t(1) << BlockWidthLog2;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    if (!firstOffset) firstOffset = 1;
    if (!lastOffset) lastOffset = grid->activeVoxelCount();
    if (lastOffset < firstOffset) return VoxelBlockManagerHandle<BufferT>{};
    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)
    if (!nBlocks) nBlocks = (lastOffset - firstOffset + BlockWidth) >> BlockWidthLog2;

    auto firstLeafIDBuf = BufferT::create(nBlocks * sizeof(uint32_t), pool);
    auto jumpMapBuf = BufferT::create(nBlocks * JumpMapLength * sizeof(uint64_t), pool);

    VoxelBlockManagerHandle<BufferT> handle(
        std::move(firstLeafIDBuf), std::move(jumpMapBuf),
        nBlocks, firstOffset, lastOffset);

    buildVoxelBlockManager<BlockWidthLog2>(grid, handle);
    return handle;
}

} // namespace tools

} // namespace nanovdb

#if defined(__CUDACC__)
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#endif// defined(__CUDACC__)

#endif // NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED
