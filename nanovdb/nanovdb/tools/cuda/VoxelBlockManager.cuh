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
        executed cooperatively across a CUDA thread block.
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

    // Stencil resolver naming. Each stencil shape comes in two forms:
    //   <name>        - resolves each tap by a root-down tree traversal. Uses no shared
    //                   memory and does not synchronize, so it is safe to call from
    //                   divergent threads within a block.
    //   <name>Cached  - the block cooperatively stages a per-leaf neighbor table in shared
    //                   memory, turning each tap into a direct in-leaf lookup. Substantially
    //                   faster, but must be called by all threads in the block (it uses
    //                   __syncthreads) and costs shared memory.
    // Independently, compute<Name> materializes the taps into a 27-slot array while
    // forEach<Name> streams (tap, index) to a device-inlined callback, which avoids the
    // per-thread array entirely for consumers that can take the taps in order.

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

        int tID = threadIdx.x;

        // Count how many additional leaves (following the one indicated by firstLeafID)
        // overlap with this voxel block
        int nExtraLeaves = 0;
        for (int i = 0; i < JumpMapLength; i++)
            nExtraLeaves += util::countOn(jumpMap[i]);

        // Initialize leafIndex & voxelOffset to sentinel values
        // for blocks that extend beyond the last active voxel in the grid
        if (tID < BlockWidth)
            #pragma unroll
            for (int i = 0; i < BlockWidth; i += blockDim.x) {
                smem_leafIndex[i+tID] = UnusedLeafIndex;
                smem_voxelOffset[i+tID] = UnusedVoxelOffset;
            }
        __syncthreads();

        NANOVDB_ASSERT(blockDim.x <= 512);
        const auto& tree = grid->tree();
        // Loop through all leafNodes overlapping the voxel block
        // with all threads in threadblock working collaboratively within each leafNode
        for (int leafID = firstLeafID; leafID <= firstLeafID + nExtraLeaves; leafID++) {
            const auto& leaf = tree.template getFirstNode<0>()[leafID];
            if (leaf.data()->firstOffset() >= blockFirstOffset + BlockWidth) break;
            const Coord origin = leaf.origin();
            for (int threadOffset = 0; threadOffset < 512; threadOffset += blockDim.x) {
                int localOffset = threadOffset + tID;
                auto index = leaf.data()->getValue(localOffset);
                if ((index >= blockFirstOffset) && (index < blockFirstOffset + BlockWidth)) {
                    int blockOffset = index - blockFirstOffset;
                    // Write inverse map to shared memory; no collisions
                    smem_leafIndex[blockOffset] = leafID;
                    smem_voxelOffset[blockOffset] = localOffset;
                }
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

    /// @brief Visit each 3x3x3 stencil index without materializing a 27-element array.
    /// Consumers that reduce or accumulate taps should prefer this streaming form over
    /// computeBoxStencil: the callback is device-inlined and receives (tap, index) in the
    /// same deterministic tap order, so lookup and arithmetic stay fused (no 27-element
    /// per-thread array, no stack spill). Keep computeBoxStencil for callers that need
    /// random access to all taps. Like computeBoxStencil this accesses shared memory but
    /// does not synchronize, so it may be called from divergent threads within a block -
    /// which is the reason to use this rather than the faster forEachBoxStencilCached,
    /// whose cooperative leaf table requires all threads to participate.
    /// @tparam BuildT Build type of the grid
    /// @tparam OpT    Device-callable with signature op(int tap, uint64_t index)
    template <class BuildT, class OpT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    forEachBoxStencil(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        OpT op)
    {
        NANOVDB_ASSERT(grid->isSequential());
        const int tID = threadIdx.x;
        const auto& tree = grid->tree();
        if (smem_leafIndex[tID] == UnusedLeafIndex) return;
        const auto& leaf = tree.template getFirstNode<0>()[smem_leafIndex[tID]];
        const Coord coord = leaf.offsetToGlobalCoord(smem_voxelOffset[tID]);
        for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj)
        for (int dk = -1; dk <= 1; ++dk) {
            const int tap = (di + 1) * 9 + (dj + 1) * 3 + dk + 1;
            op(tap, tree.getValue(coord.offsetBy(di, dj, dk)));
        }
    }

    /// @brief Like computeBoxStencil, but the block cooperatively stages a leaf-pointer table
    /// in shared memory: the block's voxels span few distinct (consecutive) leaves, so the
    /// spannedLeaves x 27 leaf-neighborhood resolutions are distributed across ALL threads and
    /// each stencil tap becomes a direct in-leaf lookup instead of a root-down tree traversal.
    /// Blocks spanning more than MaxCachedLeaves leaves fall back to per-tap traversal for the
    /// uncached leaves. Must be called by all threads in the block (uses __syncthreads).
    ///
    /// Use this for consumers that need all 27 taps materialized (random or repeated access):
    /// measured 1.1-1.6x over the naive per-tap resolution on both sparse and dense topology,
    /// with the margin widening at larger block widths and narrowing as the per-tap payload
    /// gather grows. Consumers that can consume taps in order should prefer
    /// forEachBoxStencilCached, which adds the streaming output for a larger win. For
    /// partial-tap consumers that
    /// read only a subset of the 27, prefer the naive computeBoxStencil: tap-level dead-code
    /// elimination already skips resolving the taps that are never read, so a right-sized table
    /// has less to save (a 19-tap variant measured slower than the DCE'd naive path).
    /// @tparam BuildT Build type of the grid
    /// @param stencilIndices Output stencil indices, length >= 27 (the 3x3x3 stencil)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeBoxStencilCached(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        uint64_t *stencilIndices)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        using LeafT = typename NanoTree<BuildT>::LeafNodeType;
        constexpr int MaxCachedLeaves = 16;
        __shared__ const LeafT* sNeighborLeaves[MaxCachedLeaves][27];

        int tID = threadIdx.x;
        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();

        // Sequential voxels are grouped by (consecutive) leaves; distribute the
        // spannedLeaves x 27 neighbor resolutions across ALL threads so no
        // single thread serializes 27 root walks behind the barrier.
        const uint32_t firstSpanned = smem_leafIndex[0];
        const uint32_t myLeaf = smem_leafIndex[tID];
        __shared__ int sSpannedCount;
        if (tID == 0) sSpannedCount = 0;
        __syncthreads();
        if (myLeaf != UnusedLeafIndex && (tID == 0 || smem_leafIndex[tID-1] != myLeaf))
            atomicMax(&sSpannedCount, int(myLeaf - firstSpanned) + 1);
        __syncthreads();
        const int nSpanned = sSpannedCount < MaxCachedLeaves ? sSpannedCount : MaxCachedLeaves;
        for (int c = tID; c < nSpanned * 27; c += blockDim.x) {
            const int slot = c / 27, n = c % 27;
            if (n == 13) { sNeighborLeaves[slot][13] = leaf0 + (firstSpanned + slot); continue; }
            const Coord origin = leaf0[firstSpanned + slot].origin();
            const int di = n/9 - 1, dj = (n/3)%3 - 1, dk = n%3 - 1;
            sNeighborLeaves[slot][n] = tree.root().probeLeaf(origin.offsetBy(di*8, dj*8, dk*8));
        }
        __syncthreads();

        if (myLeaf != UnusedLeafIndex) {
            const auto& leaf = leaf0[myLeaf];
            const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
            const uint32_t slot = myLeaf - firstSpanned;
            const bool cached = slot < MaxCachedLeaves;
            const int vi = coord[0] & 7, vj = coord[1] & 7, vk = coord[2] & 7;
            for (int di = -1; di <= 1; di++)
            for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                const int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
                const Coord neighbor = coord.offsetBy( di, dj, dk );
                if (cached) {
                    // Which of the 27 leaf-neighborhood slots this tap lands in
                    const int li = ((vi+di)>>3)+1, lj = ((vj+dj)>>3)+1, lk = ((vk+dk)>>3)+1;
                    const LeafT* nl = sNeighborLeaves[slot][li*9+lj*3+lk];
                    stencilIndices[spokeID] = nl ? nl->getValue(LeafT::CoordToOffset(neighbor)) : 0;
                } else {
                    stencilIndices[spokeID] = tree.getValue( neighbor );
                }
            }
        }
    }

    /// @brief Like computeBoxStencil, but resolves only the 7-point cross (center + 6 face
    /// neighbors) - the right-sized shape for Laplacian/upwind-class stencils. Writes the
    /// face+center slots of the 27-slot array (4, 22, 10, 16, 12, 14 and 13); the rest are
    /// left untouched. Like computeBoxStencil this uses no shared memory and does not
    /// synchronize, so it may be called from divergent threads within a block; see
    /// computeCrossStencilCached for the cooperative, leaf-table-accelerated form.
    ///
    /// Prefer this over calling computeBoxStencil and reading only the cross slots: resolving
    /// the 7 taps directly measured 1.6-1.8x faster than relying on the compiler to eliminate
    /// the 20 unread taps of the full box.
    /// @tparam BuildT Build type of the grid
    /// @param stencilIndices Output stencil indices, length >= 27 (the cross slots are written)
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeCrossStencil(
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
            const auto& leaf = tree.template getFirstNode<0>()[ smem_leafIndex[tID] ];
            const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
            stencilIndices[13] = leaf.getValue( smem_voxelOffset[tID] );
            for (int n = 0; n < 6; ++n) {
                const int axis = n >> 1, dir = (n & 1) ? 1 : -1;
                Coord neighbor = coord;
                neighbor[axis] += dir;
                // Standard 27-slot spoke id for this face tap
                const int spokeID = 13 + dir * (axis == 0 ? 9 : axis == 1 ? 3 : 1);
                stencilIndices[spokeID] = tree.getValue( neighbor );
            }
        }
    }

    /// @brief Like computeCrossStencil, but stages a 7-entry leaf table per spanned leaf (own
    /// leaf + 6 face-adjacent leaves) so each of the 7 taps becomes a direct in-leaf lookup
    /// instead of a root-down tree traversal - by a wide margin the fastest way to resolve a
    /// cross-shaped stencil (measured 4.1-6.8x over the naive form at one sidecar channel,
    /// still 1.5-2.0x at sixteen). Writes the same slots as computeCrossStencil.
    ///
    /// @warning Stages a shared-memory table cooperatively, so it must be called by all threads
    /// in the block (uses __syncthreads internally). Use computeCrossStencil when the call site
    /// may be divergent, or when the block cannot spare the shared memory.
    ///
    /// @note There is deliberately no streaming (forEach) cross variant: a 7-element stencil is
    /// cheap enough to materialize that removing the array measured as a wash.
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeCrossStencilCached(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        uint64_t *stencilIndices)
    {
        NANOVDB_ASSERT(grid->isSequential());

        using LeafT = typename NanoTree<BuildT>::LeafNodeType;
        constexpr int MaxCachedLeaves = 16;
        // Slots 0..5: -x,+x,-y,+y,-z,+z face neighbors; slot 6: own leaf
        __shared__ const LeafT* sFaceLeaves[MaxCachedLeaves][7];

        int tID = threadIdx.x;
        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();

        const uint32_t firstSpanned = smem_leafIndex[0];
        const uint32_t myLeaf = smem_leafIndex[tID];
        __shared__ int sSpannedCount;
        if (tID == 0) sSpannedCount = 0;
        __syncthreads();
        if (myLeaf != UnusedLeafIndex && (tID == 0 || smem_leafIndex[tID-1] != myLeaf))
            atomicMax(&sSpannedCount, int(myLeaf - firstSpanned) + 1);
        __syncthreads();
        const int nSpanned = sSpannedCount < MaxCachedLeaves ? sSpannedCount : MaxCachedLeaves;
        for (int c = tID; c < nSpanned * 7; c += blockDim.x) {
            const int slot = c / 7, n = c % 7;
            if (n == 6) { sFaceLeaves[slot][6] = leaf0 + (firstSpanned + slot); continue; }
            Coord nOrigin = leaf0[firstSpanned + slot].origin();
            nOrigin[n >> 1] += (n & 1) ? 8 : -8;
            sFaceLeaves[slot][n] = tree.root().probeLeaf(nOrigin);
        }
        __syncthreads();

        if (myLeaf != UnusedLeafIndex) {
            const auto& leaf = leaf0[myLeaf];
            const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
            const uint32_t slot = myLeaf - firstSpanned;
            const bool cached = slot < MaxCachedLeaves;
            stencilIndices[13] = leaf.getValue( smem_voxelOffset[tID] );
            for (int n = 0; n < 6; ++n) {
                const int axis = n >> 1, dir = (n & 1) ? 1 : -1;
                Coord neighbor = coord;
                neighbor[axis] += dir;
                // Standard 27-slot spoke id for this face tap
                const int spokeID = 13 + dir * (axis == 0 ? 9 : axis == 1 ? 3 : 1);
                if (cached) {
                    const int v = coord[axis] & 7;
                    const LeafT* nl = ((v + dir) & ~7) ? sFaceLeaves[slot][n] : sFaceLeaves[slot][6];
                    stencilIndices[spokeID] = nl ? nl->getValue(LeafT::CoordToOffset(neighbor)) : 0;
                } else {
                    stencilIndices[spokeID] = tree.getValue( neighbor );
                }
            }
        }
    }

    /// @brief The cached leaf table of computeBoxStencilCached with the streaming output of
    /// forEachBoxStencil: taps are resolved through the staged per-leaf neighbor table AND
    /// handed to a device-inlined callback, so neither the 27 root-down traversals nor the
    /// 27-element per-thread array is paid. This is the fastest path for consumers that read
    /// all 27 taps and can consume them in tap order - measured ~1.8x (block width 128) to
    /// ~2.6x (block width 512) over the naive computeBoxStencil, versus ~1.3-1.9x for either
    /// technique alone. Blocks spanning more than MaxCachedLeaves leaves fall back to per-tap
    /// traversal for the uncached leaves.
    ///
    /// @warning Unlike forEachBoxStencil, this stages a shared-memory table cooperatively, so
    /// it must be called by all threads in the block (uses __syncthreads internally). Use
    /// forEachBoxStencil when the call site may be divergent.
    ///
    /// @tparam BuildT Build type of the grid
    /// @tparam OpT    Device-callable with signature op(int tap, uint64_t index)
    template <class BuildT, class OpT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    forEachBoxStencilCached(
        const NanoGrid<BuildT> *grid,
        const uint32_t *smem_leafIndex,
        const uint16_t *smem_voxelOffset,
        OpT op)
    {
        NANOVDB_ASSERT(grid->isSequential());

        using LeafT = typename NanoTree<BuildT>::LeafNodeType;
        constexpr int MaxCachedLeaves = 16;
        __shared__ const LeafT* sNeighborLeaves[MaxCachedLeaves][27];
        __shared__ int sSpannedCount;

        const int tID = threadIdx.x;
        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();
        const uint32_t firstSpanned = smem_leafIndex[0];
        const uint32_t myLeaf = smem_leafIndex[tID];

        // Stage the spannedLeaves x 27 leaf-neighborhood table across ALL threads
        if (tID == 0) sSpannedCount = 0;
        __syncthreads();
        if (myLeaf != UnusedLeafIndex && (tID == 0 || smem_leafIndex[tID-1] != myLeaf))
            atomicMax(&sSpannedCount, int(myLeaf - firstSpanned) + 1);
        __syncthreads();
        const int nSpanned = sSpannedCount < MaxCachedLeaves ? sSpannedCount : MaxCachedLeaves;
        for (int c = tID; c < nSpanned * 27; c += blockDim.x) {
            const int slot = c / 27, n = c % 27;
            if (n == 13) { sNeighborLeaves[slot][13] = leaf0 + (firstSpanned + slot); continue; }
            const Coord origin = leaf0[firstSpanned + slot].origin();
            const int di = n/9 - 1, dj = (n/3)%3 - 1, dk = n%3 - 1;
            sNeighborLeaves[slot][n] = tree.root().probeLeaf(origin.offsetBy(di*8, dj*8, dk*8));
        }
        __syncthreads();

        if (myLeaf == UnusedLeafIndex) return;
        const auto& leaf = leaf0[myLeaf];
        const Coord coord = leaf.offsetToGlobalCoord( smem_voxelOffset[tID] );
        const uint32_t slot = myLeaf - firstSpanned;
        const bool cached = slot < MaxCachedLeaves;
        const int vi = coord[0] & 7, vj = coord[1] & 7, vk = coord[2] & 7;
        for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
        for (int dk = -1; dk <= 1; dk++) {
            const int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
            const Coord neighbor = coord.offsetBy( di, dj, dk );
            if (cached) {
                const int li = ((vi+di)>>3)+1, lj = ((vj+dj)>>3)+1, lk = ((vk+dk)>>3)+1;
                const LeafT* nl = sNeighborLeaves[slot][li*9+lj*3+lk];
                op(spokeID, nl ? nl->getValue(LeafT::CoordToOffset(neighbor)) : uint64_t(0));
            } else {
                op(spokeID, tree.getValue( neighbor ));
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
