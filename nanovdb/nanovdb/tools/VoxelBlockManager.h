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
#include <nanovdb/util/MaskPrefixSum.h>

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

// --------------------------> VoxelBlockManager (CPU) <--------------------------------------

/// @brief CPU counterpart of tools::cuda::VoxelBlockManager. Provides host-side
///        decode of the inverse maps (sequential index -> leaf + voxel offset)
///        for a single voxel block. No SIMD or parallelism: the implementation
///        is intentionally plain and single-threaded so the caller can parallelize
///        across blocks however it likes (e.g. std::execution::par or OpenMP).
template <int Log2BlockWidth>
struct VoxelBlockManager
{
    static constexpr int BlockWidth    = 1 << Log2BlockWidth;
    static_assert(Log2BlockWidth >= 6, "BlockWidth must be at least 64 (one jumpMap word per block)");
    static constexpr int JumpMapLength = BlockWidth / 64;

    static constexpr uint32_t UnusedLeafIndex   = 0xffffffff;
    static constexpr uint16_t UnusedVoxelOffset = 0xffff;

    /// @brief One pass of bit-parallel compaction: branchless two-buffer conditional blend.
    ///
    /// For j in [0, 512-Shift):
    ///   m = 0xFFFF if (shifts[j+Shift] & Shift) != 0, else 0x0000
    ///   dst[j] = (src[j+Shift] & m) | (src[j] & ~m)
    /// For j in [512-Shift, 512): dst[j] = src[j]
    ///
    /// The predicate tests the SOURCE element's shift (bit log2(Shift) of
    /// shifts[j+Shift]).  Separate __restrict__ src/dst buffers allow the
    /// vectorizer to emit vpblendvb (AVX2) or masked vmovdqu16 (AVX-512).
    /// The arithmetic mask avoids conditional branches in the SIMD loop.
    ///
    /// @note __restrict__ is GCC/Clang syntax; MSVC spells it __restrict.
    ///       A NANOVDB_RESTRICT portability macro should be added before
    ///       this header is used in a MSVC build.
    ///
    /// @param src     Source buffer (512 uint16_t, read-only this pass).
    /// @param shifts  shifts[i] = i - exclusive_prefix[i], unchanged across passes.
    /// @param dst     Destination buffer (512 uint16_t, write-only this pass).
    template <int Shift>
    static void shflDownSep(const uint16_t* __restrict__ src,
                            const uint16_t* __restrict__ shifts,
                                  uint16_t* __restrict__ dst)
    {
        #pragma omp simd
        for (int j = 0; j < 512 - Shift; j++) {
            const uint16_t m = static_cast<uint16_t>(
                -static_cast<int>((shifts[j + Shift] & static_cast<uint16_t>(Shift)) != 0));
            dst[j] = (src[j + Shift] & m) | (src[j] & ~m);
        }
        for (int j = 512 - Shift; j < 512; j++)
            dst[j] = src[j];
    }

    /// @brief Decode the inverse maps for a single voxel block on the host.
    ///
    /// Given the VBM metadata for one block (firstLeafID and the block's slice of
    /// the jumpMap) and the block's base sequential offset, fills leafIndex[] and
    /// voxelOffset[] so that for each position p in [0, BlockWidth):
    ///   - leafIndex[p]   = index of the leaf node containing sequential voxel
    ///                      (blockFirstOffset + p), or UnusedLeafIndex if that
    ///                      index is beyond the last active voxel.
    ///   - voxelOffset[p] = local (0..511) offset of that voxel within its leaf,
    ///                      or UnusedVoxelOffset.
    ///
    /// This is the direct CPU analogue of the CUDA decodeInverseMaps: same inputs,
    /// same outputs, same sentinel values. The GPU version runs cooperatively across
    /// a thread block; this version is a plain sequential loop intended to be called
    /// once per voxel block from a single thread (or from one thread per block in a
    /// parallel outer loop).
    ///
    /// @tparam BuildT     Build type of the grid (must be an index type)
    /// @param grid            Host-accessible grid
    /// @param firstLeafID  Index of the first leaf overlapping this block
    ///                     (i.e. firstLeafID[blockID] from the VBM metadata)
    /// @param jumpMap         Pointer to the JumpMapLength words for this block
    ///                        (i.e. &jumpMap[blockID * JumpMapLength])
    /// @param blockFirstOffset  Sequential index of the first voxel in this block
    /// @param leafIndex       Output array of length BlockWidth
    /// @param voxelOffset     Output array of length BlockWidth
    template <class BuildT>
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    decodeInverseMaps(
        const NanoGrid<BuildT>* grid,
        const uint32_t          firstLeafID,
        const uint64_t*         jumpMap,
        const uint64_t          blockFirstOffset,
        uint32_t*               leafIndex,
        uint16_t*               voxelOffset)
    {
        NANOVDB_ASSERT(grid->isSequential());

        // Count how many additional leaves follow firstLeafID in this block
        int nExtraLeaves = 0;
        for (int i = 0; i < JumpMapLength; i++)
            nExtraLeaves += util::countOn(jumpMap[i]);

        // Initialize outputs to sentinel values
        std::fill(leafIndex,   leafIndex   + BlockWidth, UnusedLeafIndex);
        std::fill(voxelOffset, voxelOffset + BlockWidth, UnusedVoxelOffset);

        const auto &tree = grid->tree();
        for (int leafID = firstLeafID; leafID <= firstLeafID + nExtraLeaves; leafID++) {
            const auto &leaf = tree.template getFirstNode<0>()[leafID];

            const uint64_t leafFirstOffset = leaf.data()->firstOffset();
            if (leafFirstOffset >= blockFirstOffset + BlockWidth) break;

            // Compute shifts[i] = number of inactive voxels at positions 0..i-1, i.e. the
            // exclusive prefix count of 0-bits over the inverted mask.  Using the 513-entry
            // exclusive layout (shifts[0]=0, buildMaskPrefixSums<true> writes inclusive
            // 0-bit counts into shifts[1..512]):
            //   shifts[i]   = exclusive 0-bit prefix at i  (used by shflDownSep passes)
            //   shifts[512] = total inactive voxel count  = 512 - leafValueCount
            uint16_t shifts[513];
            shifts[0] = 0;
            util::buildMaskPrefixSums<true>(leaf.valueMask(), leaf.data()->mPrefixSum, shifts + 1);

            const uint16_t leafValueCount = static_cast<uint16_t>(512u) - shifts[512];

            // Build leafLocalOffsets via 9 shfl_down passes using ping-pong buffers.
            // buf0 is initialized with the identity (buf0[i] = i).  Passes alternate
            // source and destination so each call has fully separate __restrict__ buffers,
            // enabling vpblendvb / masked-vmovdqu16 vectorization.  After 9 passes (odd),
            // the result is in buf1: buf1[j] = leaf-local position of j-th active voxel.
            // Only shifts[0..511] are accessed by the passes; shifts[512] was used above.
            uint16_t buf0[512], buf1[512];
            for (int i = 0; i < 512; i++) buf0[i] = static_cast<uint16_t>(i);
            shflDownSep<  1>(buf0, shifts, buf1);
            shflDownSep<  2>(buf1, shifts, buf0);
            shflDownSep<  4>(buf0, shifts, buf1);
            shflDownSep<  8>(buf1, shifts, buf0);
            shflDownSep< 16>(buf0, shifts, buf1);
            shflDownSep< 32>(buf1, shifts, buf0);
            shflDownSep< 64>(buf0, shifts, buf1);
            shflDownSep<128>(buf1, shifts, buf0);
            shflDownSep<256>(buf0, shifts, buf1);
            const uint16_t *leafLocalOffsets = buf1;

            // Intersect this leaf's active range with the block's range.
            // Active voxels span [leafFirstOffset, leafFirstOffset+leafValueCount) globally.
            // Block output slots span [blockFirstOffset, blockFirstOffset+BlockWidth).
            const uint64_t globalStart = std::max(leafFirstOffset, blockFirstOffset);
            const uint64_t globalEnd   = std::min(leafFirstOffset + leafValueCount,
                                                  blockFirstOffset + BlockWidth);
            const uint64_t jStart = globalStart - leafFirstOffset;  // first dense index in leaf
            const uint64_t pStart = globalStart - blockFirstOffset; // first output slot in block
            const uint64_t count  = globalEnd   - globalStart;

            std::fill(leafIndex   + pStart, leafIndex   + pStart + count,
                      static_cast<uint32_t>(leafID));
            std::copy(leafLocalOffsets + jStart, leafLocalOffsets + jStart + count,
                      voxelOffset + pStart);
        }
    }
}; // VoxelBlockManager

// --------------------------> buildVoxelBlockManager (CPU) <---------------------------------

/// @brief Rebuild a VoxelBlockManager in-place using a pre-allocated handle.
///        Zeros the jumpMap and recomputes both metadata arrays. This overload
///        performs no memory allocation and is suitable for repeated builds
///        (e.g. benchmarking, or rebuilding after a topology-preserving update).
/// @tparam Log2BlockWidth Log2 of the number of active voxels per VBM block
/// @tparam BufferT Buffer type of the handle (must provide host-accessible data())
/// @param grid Host-accessible ValueOnIndex grid (must satisfy isSequential())
/// @param handle Pre-allocated handle whose blockCount/firstOffset/lastOffset are
///               already set to match the grid
template<int Log2BlockWidth, typename BufferT>
void buildVoxelBlockManager(const NanoGrid<ValueOnIndex>* grid, VoxelBlockManagerHandle<BufferT>& handle)
{
    static constexpr uint64_t BlockWidth = uint64_t(1) << Log2BlockWidth;
    static_assert(Log2BlockWidth >= 6, "BlockWidth must be at least 64 (one jumpMap word per block)");
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    NANOVDB_ASSERT(grid->isSequential());
    if (!handle.blockCount()) return;

    uint32_t *firstLeafID = handle.hostFirstLeafID();
    uint64_t *jumpMap = handle.hostJumpMap();
    const uint64_t nBlocks = handle.blockCount();
    const uint64_t firstOffset = handle.firstOffset();
    const uint64_t lastOffset = handle.lastOffset();

    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)

    std::memset(jumpMap, 0, nBlocks * JumpMapLength * sizeof(uint64_t));

    const auto &tree = grid->tree();
    const auto *firstLeaf = tree.getFirstNode<0>();
    const uint32_t leafCount = tree.nodeCount(0);

    std::for_each(std::execution::par, firstLeaf, firstLeaf + leafCount,
        [&](const NanoLeaf<ValueOnIndex>& leaf) {
            const uint64_t leafFirstOffset = leaf.data()->firstOffset();
            const uint64_t leafValueCount = leaf.data()->valueCount();
            const uint64_t leafLastOffset = leafFirstOffset + leafValueCount - 1;

            if (leafFirstOffset > lastOffset || leafLastOffset < firstOffset) return;

            const uint64_t leafIndex = static_cast<uint64_t>(&leaf - firstLeaf);

            const uint64_t lastBlock = std::min<uint64_t>(
                (leafLastOffset - firstOffset) >> Log2BlockWidth, nBlocks - 1);
            const uint64_t firstBlock = (leafFirstOffset < firstOffset) ? 0 :
                (leafFirstOffset - firstOffset) >> Log2BlockWidth;

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
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
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
template<int Log2BlockWidth, typename BufferT = HostBuffer>
VoxelBlockManagerHandle<BufferT>
buildVoxelBlockManager(
    const NanoGrid<ValueOnIndex>* grid,
    uint64_t firstOffset = 0,
    uint64_t lastOffset = 0,
    uint64_t nBlocks = 0,
    const BufferT* pool = nullptr)
{
    static constexpr uint64_t BlockWidth = uint64_t(1) << Log2BlockWidth;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    if (!firstOffset) firstOffset = 1;
    if (!lastOffset) lastOffset = grid->activeVoxelCount();
    if (lastOffset < firstOffset) return VoxelBlockManagerHandle<BufferT>{};
    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)
    if (!nBlocks) nBlocks = (lastOffset - firstOffset + BlockWidth) >> Log2BlockWidth;

    auto firstLeafIDBuf = BufferT::create(nBlocks * sizeof(uint32_t), pool);
    auto jumpMapBuf = BufferT::create(nBlocks * JumpMapLength * sizeof(uint64_t), pool);

    VoxelBlockManagerHandle<BufferT> handle(
        std::move(firstLeafIDBuf), std::move(jumpMapBuf),
        nBlocks, firstOffset, lastOffset);

    buildVoxelBlockManager<Log2BlockWidth>(grid, handle);
    return handle;
}

} // namespace tools

} // namespace nanovdb

#if defined(__CUDACC__)
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#endif// defined(__CUDACC__)

#endif // NANOVDB_VOXELBLOCKMANAGER_H_HAS_BEEN_INCLUDED
