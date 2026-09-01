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
      - decodeInverseMap (device): per-slot, thread-local decode of one inverse
        map entry (sequential active-voxel index -> leaf ID + intra-leaf voxel
        offset) into registers; no shared memory or synchronization, callable
        from divergent threads. Block-level facts a cooperative consumer might
        otherwise read from a materialized map are derivable directly from the
        VBM metadata: the block's first leaf is firstLeafID, slot p starts a
        new leaf iff jumpMap bit p is set, and the number of leaves spanned by
        the block is 1 + the jumpMap popcount.
      - decodeInverseMaps (device): cooperative wrapper over decodeInverseMap
        that materializes all BlockWidth slots into shared memory, preserving
        the pre-existing signature for consumers that read across slots.
*/

#ifndef NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/math/Stencils.h>

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

    // The decode is a rank + select over the VBM's bit-vectors and is
    // thread-local per output slot: decodeInverseMap requires no threadblock
    // coordination. Consumers that need cross-slot information can derive it
    // from the VBM metadata (firstLeafID + jumpMap), or materialize all slots
    // in shared memory via the cooperative decodeInverseMaps wrapper.

    /// @brief Rank one output slot into its leaf: the number of leaves that
    /// begin at in-block positions [1, blockOffset] (bit 0 is never set),
    /// counted via popcounts over the block's jumpMap.
    /// Thread-local; no synchronization.
    __device__
    static uint32_t jumpMapRank(const uint64_t *jumpMap, const int blockOffset)
    {
        uint32_t leafRank = 0;
        const int jumpWord = blockOffset >> 6;  // index into jumpMap
        #pragma unroll
        for (int i = 0; i < JumpMapLength; ++i) {
            if (i < jumpWord) leafRank += util::countOn(jumpMap[i]); // count leaves before the current jump word
            else if (i == jumpWord) {
                // count leaves in the current jump word, masking those that are at or before blockOffset
                leafRank += util::countOn(jumpMap[i] & ((uint64_t(2) << (blockOffset & 63)) - 1u));
            }
        }
        return leafRank;
    }

    /// @brief Select the voxel with sequential index @c globalOffset inside leaf
    /// @c leafID: find its 64-bit mask word via the leaf's precomputed 9-bit
    /// prefix sums (mPrefixSum), then its bit within that word via __fns.
    /// Writes the sentinels if globalOffset lies beyond the leaf's last active
    /// voxel (i.e. beyond the last active voxel of the grid).
    /// Thread-local; no synchronization.
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    selectVoxelInLeaf(
        const NanoGrid<BuildT> *grid,
        const uint32_t leafID,
        const uint64_t globalOffset,
        uint32_t &leafIndex,
        uint16_t &voxelOffset)
    {
        const auto& leafData = *grid->tree().template getFirstNode<0>()[leafID].data();
        // 0-based rank among the leaf's actives
        const uint64_t rankInLeaf = globalOffset - leafData.firstOffset();
        if (rankInLeaf < leafData.valueCount()) { // if the rank is within the leaf's active voxels (bounds check)
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
            // __fns(mask, base, k) is the hardware find-nth-set-bit intrinsic - the k-th set bit (1-based) at/after base
            // but it's a 32-bit op while `maskWord` is 64-bit, so we need to split it into two 32-bit halves
            const uint64_t maskWord = leafData.mValueMask.words()[wordID];
            const uint32_t lowHalf = uint32_t(maskWord);  // low 32 bits of the mask word
            const uint32_t lowHalfCount = __popc(lowHalf);
            int bit;
            // if rank is less than the number of active voxels in the low half, __fns finds the bit in the lower half
            // otherwise, shift maskWord by 32 bits and __fns finds the bit in the upper half
            if (rankInWord < lowHalfCount) bit = __fns(lowHalf, 0, rankInWord + 1);
            else                           bit = 32 + __fns(uint32_t(maskWord >> 32), 0, rankInWord - lowHalfCount + 1);
            leafIndex = leafID;
            voxelOffset = uint16_t((wordID << 6) + bit);
        } else { // beyond the last active voxel in the grid
            leafIndex = UnusedLeafIndex;
            voxelOffset = UnusedVoxelOffset;
        }
    }

    /// @brief Decode a single inverse-map entry into registers on the device.
    ///
    /// Given the VBM metadata for one block (firstLeafID and the block's slice of
    /// the jumpMap), the block's base sequential offset, and a slot position
    /// blockOffset in [0, BlockWidth), computes:
    ///   - leafIndex   = index of the leaf node containing sequential voxel
    ///                   (blockFirstOffset + blockOffset), or UnusedLeafIndex if
    ///                   that index is beyond the last active voxel.
    ///   - voxelOffset = local (0..511) offset of that voxel within its leaf,
    ///                   or UnusedVoxelOffset.
    ///
    /// No shared memory or synchronization; may be called from divergent threads.
    ///
    /// @tparam BuildT  Build type of the grid (must be an index type)
    /// @param grid              Device-accessible OnIndex grid
    /// @param firstLeafID       Index of the first leaf overlapping this block
    /// @param jumpMap           Pointer to the JumpMapLength words for this block
    /// @param blockFirstOffset  Sequential index of the first voxel in this block
    /// @param blockOffset       Slot position within the block, in [0, BlockWidth)
    /// @param leafIndex         Output leaf index (register)
    /// @param voxelOffset       Output intra-leaf voxel offset (register)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    decodeInverseMap(
        const NanoGrid<BuildT> *grid,
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint64_t blockFirstOffset,
        const int blockOffset,
        uint32_t &leafIndex,
        uint16_t &voxelOffset)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());
        NANOVDB_ASSERT(blockOffset >= 0 && blockOffset < BlockWidth);

        const uint32_t leafRank = jumpMapRank(jumpMap, blockOffset);
        selectVoxelInLeaf(grid, firstLeafID + leafRank,
                          blockFirstOffset + blockOffset, leafIndex, voxelOffset);
    }

    /// @brief Decode the inverse maps for a single voxel block into shared memory.
    ///
    /// Cooperative form of decodeInverseMap, preserving the pre-existing signature:
    /// fills smem_leafIndex[] and smem_voxelOffset[] so that for each position p in
    /// [0, BlockWidth):
    ///   - smem_leafIndex[p]   = index of the leaf node containing sequential voxel
    ///                           (blockFirstOffset + p), or UnusedLeafIndex if that
    ///                           index is beyond the last active voxel.
    ///   - smem_voxelOffset[p] = local (0..511) offset of that voxel within its leaf,
    ///                           or UnusedVoxelOffset.
    ///
    /// Each slot is decoded independently by decodeInverseMap; this wrapper only
    /// materializes the results, striding over the slots so any blockDim.x <= 512
    /// fills the maps correctly. Must be called by all threads in the block (ends
    /// in __syncthreads); do not call from divergent threads. Consumers that only
    /// read their own slot's result should prefer decodeInverseMap, which needs no
    /// shared memory or barrier.
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
        NANOVDB_ASSERT(blockDim.x <= 512);
        for (int blockOffset = threadIdx.x; blockOffset < BlockWidth; blockOffset += blockDim.x)
            decodeInverseMap(grid, firstLeafID, jumpMap, blockFirstOffset, blockOffset,
                             smem_leafIndex[blockOffset], smem_voxelOffset[blockOffset]);
        __syncthreads();
    }

    /// @brief Given a grid and one decoded inverse-map entry (from
    /// decodeInverseMap), compute the stencil indices for a 3x3x3 box stencil.
    /// Thread-local: no shared memory, no synchronization; may be called from
    /// divergent threads within a thread block. Leaves stencilIndices untouched
    /// when leafIndex is UnusedLeafIndex (a slot beyond the last active voxel).
    /// @tparam BuildT Build type of the grid
    /// @param grid         Device-accessible OnIndex grid
    /// @param leafIndex    This thread's decoded leaf index (or UnusedLeafIndex)
    /// @param voxelOffset  This thread's decoded intra-leaf voxel offset
    /// @param stencilIndices Pointer to output stencil indices. Must have
    /// length of at least 27 (corresponding to the 3x3x3 stencil)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeBoxStencil(
        const NanoGrid<BuildT> *grid,
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        uint64_t *stencilIndices)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        const auto& tree = grid->tree();
        if (leafIndex != UnusedLeafIndex) {
            // This presumes that leaf nodes are fixed-size and sequentially accessible in memory
            const auto& leaf = tree.template getFirstNode<0>()[ leafIndex ];
            const Coord coord = leaf.offsetToGlobalCoord( voxelOffset );
            for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
                const auto neighbor = coord.offsetBy( di, dj, dk );
                stencilIndices[spokeID] = tree.getValue( neighbor );
            }
        }
    }

    /// @brief Auxiliary type holding the resolved neighbor leaf pointers for
    /// the WENO5 stencil. ptrs[axis][0] is the lo neighbor along that axis
    /// (nullptr if outside the narrow band), ptrs[axis][1] is always the
    /// center leaf, and ptrs[axis][2] is the hi neighbor (nullptr if outside).
    template<class BuildT>
    struct WenoLeafPtrs {
        const NanoLeaf<BuildT>* ptrs[3][3];
    };

    /// @brief Resolve the neighbor leaf pointers needed by computeWenoStencil.
    /// Performs exactly one probeLeaf call per axis (three total). Safe to call
    /// per-thread; does not synchronize.
    /// @tparam BuildT Build type of the grid (must be an index type)
    /// @param grid    Device-resident grid
    /// @param leaf    Center leaf node for the current voxel
    /// @param voxelOffset Intra-leaf voxel offset for the current voxel
    /// @return WenoLeafPtrs with center entries set to &leaf and lo/hi entries
    ///         set to the probeLeaf result (nullptr if outside the narrow band)
    template<class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, WenoLeafPtrs<BuildT>>::type
    resolveWenoLeafPtrs(
        const NanoGrid<BuildT>* grid,
        const NanoLeaf<BuildT>& leaf,
        uint16_t                voxelOffset)
    {
        WenoLeafPtrs<BuildT> result;
        const auto coord      = leaf.offsetToGlobalCoord(voxelOffset);
        const auto localCoord = leaf.OffsetToLocalCoord(voxelOffset);
        const auto& tree      = grid->tree();

        for (int axis = 0; axis < 3; ++axis) {
            result.ptrs[axis][0] = nullptr;
            result.ptrs[axis][1] = &leaf;
            result.ptrs[axis][2] = nullptr;

            auto neighborCoord = coord;
            neighborCoord[axis] += (localCoord[axis] & 0x4) ? 4 : -4;
            result.ptrs[axis][(localCoord[axis] & 0x4) >> 1] =
                tree.root().probeLeaf(neighborCoord);
        }
        return result;
    }

    /// @brief Compute global sequential indices for the 19 WENO5 stencil
    /// points of the given voxel, using pre-resolved leaf pointers.
    ///
    /// Output layout follows nanovdb::math::WenoPt<i,j,k>::idx. Note that
    /// this convention differs from OpenVDB's NineteenPt<i,j,k>::idx.
    ///
    /// Entries for neighbors outside the narrow band are left unchanged;
    /// the caller must zero-initialize data[] before calling this function.
    /// Does not synchronize; safe to call from divergent threads.
    ///
    /// The voxelOffset arithmetic uses octal notation to exploit the fact that
    /// the NanoVDB leaf layout encodes (x,y,z) as x*64 + y*8 + z, making x, y,
    /// and z strides exactly 0100, 010, and 1 in octal respectively.
    ///
    /// @tparam BuildT Build type of the grid (must be an index type)
    /// @param leaf        Center leaf node for the current voxel
    /// @param voxelOffset Intra-leaf voxel offset for the current voxel
    /// @param leafPtrs    Resolved neighbor leaf pointers from resolveWenoLeafPtrs
    /// @param data        Output array of length >= 19, caller-zero-initialized
    template<class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeWenoStencil(
        const NanoLeaf<BuildT>&     leaf,
        uint16_t                    voxelOffset,
        const WenoLeafPtrs<BuildT>& leafPtrs,
        uint64_t*                   data)
    {
        using math::WenoPt;
        const auto lc = leaf.OffsetToLocalCoord(voxelOffset);

        data[WenoPt< 0, 0, 0>::idx] = leaf.getValue(voxelOffset);

        // x-axis: stride per step = 64 = 0100 octal; cross-leaf wrap = ±8*64 = ±0700
        if (leafPtrs.ptrs[0][(lc.x() + 5) >> 3])
            data[WenoPt<-3, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 5) >> 3]->getValue(
                voxelOffset + ((lc[0] < 3) ? 0500 : -0300));
        if (leafPtrs.ptrs[0][(lc.x() + 6) >> 3])
            data[WenoPt<-2, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 6) >> 3]->getValue(
                voxelOffset + ((lc[0] < 2) ? 0600 : -0200));
        if (leafPtrs.ptrs[0][(lc.x() + 7) >> 3])
            data[WenoPt<-1, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 7) >> 3]->getValue(
                voxelOffset + ((lc[0] < 1) ? 0700 : -0100));
        if (leafPtrs.ptrs[0][(lc.x() + 9) >> 3])
            data[WenoPt< 1, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 9) >> 3]->getValue(
                voxelOffset + ((lc[0] < 7) ? 0100 : -0700));
        if (leafPtrs.ptrs[0][(lc.x() + 10) >> 3])
            data[WenoPt< 2, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 10) >> 3]->getValue(
                voxelOffset + ((lc[0] < 6) ? 0200 : -0600));
        if (leafPtrs.ptrs[0][(lc.x() + 11) >> 3])
            data[WenoPt< 3, 0, 0>::idx] = leafPtrs.ptrs[0][(lc.x() + 11) >> 3]->getValue(
                voxelOffset + ((lc[0] < 5) ? 0300 : -0500));

        // y-axis: stride per step = 8 = 010 octal; cross-leaf wrap = ±8*8 = ±070
        if (leafPtrs.ptrs[1][(lc.y() + 5) >> 3])
            data[WenoPt< 0,-3, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 5) >> 3]->getValue(
                voxelOffset + ((lc[1] < 3) ? 0050 : -0030));
        if (leafPtrs.ptrs[1][(lc.y() + 6) >> 3])
            data[WenoPt< 0,-2, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 6) >> 3]->getValue(
                voxelOffset + ((lc[1] < 2) ? 0060 : -0020));
        if (leafPtrs.ptrs[1][(lc.y() + 7) >> 3])
            data[WenoPt< 0,-1, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 7) >> 3]->getValue(
                voxelOffset + ((lc[1] < 1) ? 0070 : -0010));
        if (leafPtrs.ptrs[1][(lc.y() + 9) >> 3])
            data[WenoPt< 0, 1, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 9) >> 3]->getValue(
                voxelOffset + ((lc[1] < 7) ? 0010 : -0070));
        if (leafPtrs.ptrs[1][(lc.y() + 10) >> 3])
            data[WenoPt< 0, 2, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 10) >> 3]->getValue(
                voxelOffset + ((lc[1] < 6) ? 0020 : -0060));
        if (leafPtrs.ptrs[1][(lc.y() + 11) >> 3])
            data[WenoPt< 0, 3, 0>::idx] = leafPtrs.ptrs[1][(lc.y() + 11) >> 3]->getValue(
                voxelOffset + ((lc[1] < 5) ? 0030 : -0050));

        // z-axis: stride per step = 1; cross-leaf wrap = ±8
        if (leafPtrs.ptrs[2][(lc.z() + 5) >> 3])
            data[WenoPt< 0, 0,-3>::idx] = leafPtrs.ptrs[2][(lc.z() + 5) >> 3]->getValue(
                voxelOffset + ((lc[2] < 3) ? 0005 : -0003));
        if (leafPtrs.ptrs[2][(lc.z() + 6) >> 3])
            data[WenoPt< 0, 0,-2>::idx] = leafPtrs.ptrs[2][(lc.z() + 6) >> 3]->getValue(
                voxelOffset + ((lc[2] < 2) ? 0006 : -0002));
        if (leafPtrs.ptrs[2][(lc.z() + 7) >> 3])
            data[WenoPt< 0, 0,-1>::idx] = leafPtrs.ptrs[2][(lc.z() + 7) >> 3]->getValue(
                voxelOffset + ((lc[2] < 1) ? 0007 : -0001));
        if (leafPtrs.ptrs[2][(lc.z() + 9) >> 3])
            data[WenoPt< 0, 0, 1>::idx] = leafPtrs.ptrs[2][(lc.z() + 9) >> 3]->getValue(
                voxelOffset + ((lc[2] < 7) ? 0001 : -0007));
        if (leafPtrs.ptrs[2][(lc.z() + 10) >> 3])
            data[WenoPt< 0, 0, 2>::idx] = leafPtrs.ptrs[2][(lc.z() + 10) >> 3]->getValue(
                voxelOffset + ((lc[2] < 6) ? 0002 : -0006));
        if (leafPtrs.ptrs[2][(lc.z() + 11) >> 3])
            data[WenoPt< 0, 0, 3>::idx] = leafPtrs.ptrs[2][(lc.z() + 11) >> 3]->getValue(
                voxelOffset + ((lc[2] < 5) ? 0003 : -0005));
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
