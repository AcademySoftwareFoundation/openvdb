// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/VoxelBlockManager.cuh

    \author Efty Sifakis

    \date January 27, 2025

    \brief VoxelBlockManager: CUDA device kernels for building and decoding the
           VBM metadata on the GPU.

    \details Implements the SIMT-parallel counterpart of the host-side decode in
      nanovdb/tools/VoxelBlockManager.h.  The VoxelBlockManager is an acceleration
      structure for voxel-sequential, SIMT-parallel access over the active voxels
      of an OnIndexGrid, independent of occupancy.  This file provides:
      - buildVoxelBlockManager (device): constructs the firstLeafID array and
        jumpMap on the GPU from a device-resident NanoGrid.
      - decodeInverseMaps (device): per-block SIMT decode of the inverse maps
        (sequential active-voxel index -> leaf ID + intra-leaf voxel offset),
        with one thread per output slot across a CUDA thread block.
*/

#ifndef NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/tools/VoxelBlockManager.h>

namespace nanovdb {

namespace tools::cuda {

/// @brief Device-side VoxelBlockManager: SIMT-parallel decode of the inverse
/// maps (sequential active-voxel index -> leaf ID + intra-leaf voxel offset)
/// for voxel-sequential, occupancy-independent access over an OnIndexGrid.
/// @tparam Log2BlockWidth Log2 of the number of active voxels per VBM block
template <int Log2BlockWidth>
struct VoxelBlockManager : nanovdb::tools::VoxelBlockManagerBase<Log2BlockWidth>
{
    using Base = nanovdb::tools::VoxelBlockManagerBase<Log2BlockWidth>;
    using Base::BlockWidth;
    using Base::JumpMapLength;
    using Base::UnusedLeafIndex;
    using Base::UnusedVoxelOffset;

    // The efficiency of the functions in this class are contingent on
    // threadblock-level coordination, which manifests either as using shared
    // memory for synchronization, or warp-level shift operations.

    /// @brief Decode the inverse maps for a single voxel block on the device.
    ///
    /// Given the VBM metadata for one block (firstLeafID and the block's slice of
    /// the jumpMap) and the block's base sequential offset, fills smem_leafIndex[]
    /// and smem_voxelOffset[] in shared memory so that for each position p in
    /// [0, BlockWidth):
    ///   - smem_leafIndex[p]   = index of the leaf node containing sequential voxel
    ///                           (blockFirstOffset + p), or UnusedLeafIndex if that
    ///                           index is beyond the last active voxel.
    ///   - smem_voxelOffset[p] = local (0..511) offset of that voxel within its leaf,
    ///                           or UnusedVoxelOffset.
    ///
    /// Must be called by all threads in the block (uses __syncthreads internally).
    /// Do not call from divergent threads within a thread block.
    ///
    /// @tparam BuildT  Build type of the grid (must be an index type)
    /// @param grid              Device-accessible OnIndex grid
    /// @param firstLeafID       Index of the first leaf overlapping this block
    /// @param jumpMap           Pointer to the JumpMapLength words for this block
    /// @param blockFirstOffset  Sequential index of the first voxel in this block
    /// @param smem_leafIndex    Output array of length BlockWidth in shared memory
    /// @param smem_voxelOffset  Output array of length BlockWidth in shared memory
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    decodeInverseMaps(
        const NanoGrid<BuildT> *grid,
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint64_t blockFirstOffset,
        uint32_t *smem_leafIndex,
        uint16_t *smem_voxelOffset)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());
        NANOVDB_ASSERT(blockDim.x <= 512);

        // Select-based decode: one thread per output slot. Here each
        // slot ranks itself into its leaf via the jumpMap popcount,
        // then locates its voxel with the leaf's 9-bit prefix sums
        // plus an in-word bit select - O(1) per slot.
        const int tID = threadIdx.x;
        const auto *leaf0 = grid->tree().template getFirstNode<0>();
        for (int blockOffset = tID; blockOffset < BlockWidth; blockOffset += blockDim.x) {
            // rank this slot into its leaf: count leaves beginning at in-block positions
            // [1, blockOffset] (bit 0 is never set) via the jumpMap popcount
            uint32_t leafRank = 0;
            const int jumpWord = blockOffset >> 6;  // index into jumpMap
            // count the number of leaves that begin before blockOffset
            #pragma unroll
            for (int i = 0; i < JumpMapLength; ++i) {
                if (i < jumpWord) leafRank += util::countOn(jumpMap[i]); // count leaves before the current jump word
                else if (i == jumpWord) {
                    // count leaves in the current jump word, masking those that are before blockOffset
                    leafRank += util::countOn(jumpMap[i] & ((uint64_t(2) << (blockOffset & 63)) - 1u));
                }

            }
            const uint32_t leafID = firstLeafID + leafRank;
            const auto& leafData = *leaf0[leafID].data();
            // 0-based rank among the leaf's actives
            const uint64_t rankInLeaf = (blockFirstOffset + blockOffset) - leafData.firstOffset();
            if (rankInLeaf < leafData.valueCount()) { // if the rank is within the leaf's active voxels (bounds check)
                // select the rankInLeaf-th active voxel: find its 64-bit mask word via the
                // leaf's 9-bit prefix sums, then its bit within that word
                int wordID = 0;
                uint32_t activesBeforeWord = 0;
                #pragma unroll
                for (int candidateWord = 1; candidateWord < 8; ++candidateWord) {
                    // the number of active voxels before the candidateWord's mask word (& 0x1ffu masks to 9 bits)
                    const uint32_t cumulative = uint32_t(leafData.mPrefixSum >> (9*(candidateWord-1))) & 0x1ffu;
                    if (cumulative <= rankInLeaf) {
                        wordID = candidateWord; // word ID of the mask word that contains the rankInLeaf-th active voxel
                        activesBeforeWord = cumulative; // the number of active voxels before the wordID's mask word
                    }
                }

                uint32_t rankInWord = uint32_t(rankInLeaf) - activesBeforeWord; // active voxel's rank within the mask word (0-based)
                // select the in-word bit position of the voxel using __fns (find n-th set bit)
                // /__fns(mask, base, k) is the hardware find-nth-set-bit intrinsic - the k-th set bit (1-based) at/after base
                // but it's a 32-bit op while `maskWord` is 64-bit, so we need to split it into two 32-bit halves
                const uint64_t maskWord = leafData.mValueMask.words()[wordID];
                const uint32_t lowHalf = uint32_t(maskWord);  // low 32 bits of the mask word
                const uint32_t lowHalfCount = util::countOn(uint64_t(lowHalf));
                int bit;
                // if rank is less than the number of active voxels in the low half, __fns finds the bit in the lower half
                // otherwise, shift maskWord by 32 bits and __fns finds the bit in the upper half
                if (rankInWord < lowHalfCount) bit = __fns(lowHalf, 0, rankInWord + 1);
                else                           bit = 32 + __fns(uint32_t(maskWord >> 32), 0, rankInWord - lowHalfCount + 1);
                smem_leafIndex[blockOffset] = leafID;
                smem_voxelOffset[blockOffset] = uint16_t((wordID << 6) + bit);
            } else {// beyond the last active voxel in the grid
                smem_leafIndex[blockOffset] = UnusedLeafIndex;
                smem_voxelOffset[blockOffset] = UnusedVoxelOffset;
            }
        }
        __syncthreads();
    }

    /// @brief Given a grid and its decoded voxel map, compute the stencil.
    /// This function accesses shared memory but does not synchronize threads
    /// so it may be called from divergent threads within a thread block.
    /// offsets for a 3x3x3 box stencil.
    /// @tparam BuildT Build type of the grid
    /// @param grid
    /// @param smem_leafIndex Leaf indices stored in shared memory
    /// @param smem_voxelOffset Voxel offsets stored in shared memory
    /// @param stencilIndices Pointer to output stencil indices. Must have
    /// length of at least 27 (corresponding to the 3x3x3 stencil)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeBoxStencil(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        uint64_t *stencilIndices)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        int tID = threadIdx.x;
        const auto& tree = grid->tree();
        if (smem_leafIndex[tID] != UnusedLeafIndex) {
            // This presumes that leaf nodes are fixed-size and sequentially accessible in memory
            const auto& leaf = tree.template getFirstNode<0>()[ smem_leafIndex[tID] ];
            const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
            const auto index = leaf.getValue( smem_voxelOffset[tID] );
            for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
                const auto neighbor = coord.offsetBy( di, dj, dk );
                stencilIndices[spokeID] = tree.getValue( neighbor );
            }
        }
    }
};

/// @brief This functor calculates the firstLeafID and jumpMap for the
/// VoxelBlockManager over the subset of the Tree nodes specified by
/// firstOffset, lastOffset, and nBlocks.
template<int Log2BlockWidth>
struct BuildVoxelBlockManagerFunctor
{
    static constexpr int BlockWidth = 1 << Log2BlockWidth;
    static constexpr int JumpMapLength = BlockWidth/64;
    static constexpr int SlicesPerLowerNode = 8;
    static constexpr int LeafNodesPerSlice = 4096/SlicesPerLowerNode;

    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    void __device__
    operator()(
        uint64_t firstOffset,
        uint64_t lastOffset,
        int nBlocks,
        const NanoGrid<ValueOnIndex> *grid,
        uint32_t *firstLeafID,
        uint64_t *jumpMap)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        using JumpMapType = uint64_t (&)[][JumpMapLength];

        int tID = threadIdx.x;
        int blockID = blockIdx.x;
        int sliceID = blockIdx.y;

        const auto& tree = grid->tree();

        const auto& lower = tree.getFirstNode<1>()[blockID];
        for ( std::size_t jj = sliceID*LeafNodesPerSlice; jj < (sliceID+1)*LeafNodesPerSlice; jj += MaxThreadsPerBlock )
            if ( lower.childMask().isOn(jj+tID) )
            {
                auto& leaf = *lower.getChild(jj+tID);
                const auto leafFirstOffset = leaf.data()->firstOffset();
                const auto leafValueCount = leaf.data()->valueCount();
                const auto leafLastOffset = leafFirstOffset + leafValueCount - 1;

                auto leafIndex = &leaf - tree.getFirstNode<0>();

                if ( ( leafFirstOffset > lastOffset ) || (leafLastOffset < firstOffset) ) continue;

                int lastBlock = (leafLastOffset - firstOffset) >> Log2BlockWidth;
                lastBlock = min(lastBlock, nBlocks-1);
                uint64_t firstBlock = (leafFirstOffset < firstOffset) ? 0 :
                    (leafFirstOffset - firstOffset) >> Log2BlockWidth;

                // For all but the first block touched, mark the firstLeaf as being this one
                for ( uint64_t b = lastBlock; b > firstBlock; --b )
                    firstLeafID[b] = leafIndex;
                if (leafFirstOffset < firstOffset) { firstLeafID[0] = leafIndex; continue; }

                const auto offsetInBlock = (leafFirstOffset - 1) & (BlockWidth - 1);
                if ( !offsetInBlock ) {
                    // If the first leaf starts exactly at the beginning of a
                    // block, register it in mFirstLeaf too
                    firstLeafID[firstBlock] = leafIndex;
                } else {
                    // Otherwise, mark it in the jumpMap
                    // The specific uint64_t in the jumpMap to be marked is at element offset (offsetInBlock>>6), i.e. offsetBlock/64
                    // and bit offset (offsetInBlock & 0x3f), i.e. offsetInBlock%64
                    util::atomicOr(&jumpMap[firstBlock * JumpMapLength + (offsetInBlock>>6)],
                                   UINT64_C(1) << (offsetInBlock & 0x3f));
                }
            }

        return;
    }

};

/// @brief Rebuild a VoxelBlockManager in-place using a pre-allocated handle.
///        Zeros the jumpMap on-stream and relaunches the build kernel. No memory
///        allocation is performed; the handle must already have correctly-sized
///        device buffers. Suitable for repeated builds and benchmarking.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam BufferT         Device buffer type (deduced from handle)
/// @param d_grid  Device-side grid pointer passed to the build kernel; lowerCount
///                is read from device memory via DeviceGridTraits
/// @param handle  Pre-allocated handle (blockCount/firstOffset/lastOffset already set)
/// @param stream  CUDA stream (default 0)
template<int Log2BlockWidth, typename BufferT>
void buildVoxelBlockManager(
    NanoGrid<ValueOnIndex>*                            d_grid,
    nanovdb::tools::VoxelBlockManagerHandle<BufferT>&  handle,
    cudaStream_t                                       stream = 0)
{
    static constexpr uint64_t BlockWidth    = uint64_t(1) << Log2BlockWidth;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    if (!handle.blockCount()) return;
    NANOVDB_ASSERT(!((handle.firstOffset() - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)

    // DeviceBuffer::create uses cudaMalloc (no zero-init); jumpMap must be zeroed each build
    cudaCheck(cudaMemsetAsync(handle.deviceJumpMap(), 0,
        handle.blockCount() * JumpMapLength * sizeof(uint64_t), stream));

    using Traits = util::cuda::DeviceGridTraits<ValueOnIndex>;
    const uint32_t lowerCount = Traits::getTreeData(d_grid).mNodeCount[1];
    using Op = BuildVoxelBlockManagerFunctor<Log2BlockWidth>;
    util::cuda::operatorKernel<Op>
        <<<dim3(lowerCount, Op::SlicesPerLowerNode, 1), Op::MaxThreadsPerBlock, 0, stream>>>(
            handle.firstOffset(), handle.lastOffset(),
            static_cast<int>(handle.blockCount()),
            d_grid, handle.deviceFirstLeafID(), handle.deviceJumpMap());
}

/// @brief Allocate device buffers and build a VoxelBlockManager on the device.
///        Returns a fully-constructed VoxelBlockManagerHandle backed by device memory.
///        Grid dimensions (when not supplied) are read from device memory via DeviceGridTraits.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam BufferT         Device buffer type (default: nanovdb::cuda::DeviceBuffer)
/// @param d_grid       Device-side grid pointer
/// @param firstOffset  First active-voxel offset covered by this VBM; must satisfy
///                     firstOffset == 1 (mod BlockWidth). Pass 0 (default) to use 1,
///                     which covers the full grid from the first active voxel.
/// @param lastOffset   Last active-voxel offset covered by this VBM. Pass 0 (default)
///                     to read activeVoxelCount from device memory via DeviceGridTraits.
/// @param nBlocks      Allocated capacity in blocks; must be >=
///                     ceil((lastOffset - firstOffset + 1) / BlockWidth). Pass 0
///                     (default) to use the minimum required capacity.
/// @param stream       CUDA stream (default 0)
/// @return A fully constructed VoxelBlockManagerHandle backed by device memory
template<int Log2BlockWidth, typename BufferT = nanovdb::cuda::DeviceBuffer>
nanovdb::tools::VoxelBlockManagerHandle<BufferT>
buildVoxelBlockManager(
    NanoGrid<ValueOnIndex>* d_grid,
    uint64_t                firstOffset = 0,
    uint64_t                lastOffset  = 0,
    uint64_t                nBlocks     = 0,
    cudaStream_t            stream      = 0)
{
    static constexpr uint64_t BlockWidth    = uint64_t(1) << Log2BlockWidth;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    using Traits = util::cuda::DeviceGridTraits<ValueOnIndex>;
    if (!firstOffset) firstOffset = 1;
    if (!lastOffset)  lastOffset  = Traits::getActiveVoxelCount(d_grid);
    if (lastOffset < firstOffset) return nanovdb::tools::VoxelBlockManagerHandle<BufferT>{};
    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)
    if (!nBlocks)     nBlocks     = (lastOffset - firstOffset + BlockWidth) >> Log2BlockWidth;

    int device = 0;
    cudaCheck(cudaGetDevice(&device));

    auto firstLeafIDBuf = BufferT::create(nBlocks * sizeof(uint32_t),                nullptr, device, stream);
    auto jumpMapBuf     = BufferT::create(nBlocks * JumpMapLength * sizeof(uint64_t), nullptr, device, stream);

    nanovdb::tools::VoxelBlockManagerHandle<BufferT> handle(
        std::move(firstLeafIDBuf), std::move(jumpMapBuf),
        nBlocks, firstOffset, lastOffset);

    buildVoxelBlockManager<Log2BlockWidth>(d_grid, handle, stream);
    return handle;
}

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
