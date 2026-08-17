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

    // The decode is a rank + select over the VBM's bit-vectors and is
    // thread-local per output slot: decodeInverseMap requires no threadblock
    // coordination. Consumers that need cross-slot information derive it from
    // the VBM metadata (firstLeafID + jumpMap) rather than from a materialized
    // per-block map.

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

    // Stencil resolver naming. Each stencil shape comes in two forms:
    //   <name>        - resolves each tap by a root-down tree traversal. Thread-local
    //                   like decodeInverseMap itself: no shared memory, no
    //                   synchronization, safe from divergent threads - so the whole
    //                   decode+resolve pipeline is.
    //   <name>Cached  - the block cooperatively stages a per-leaf neighbor table in shared
    //                   memory, turning each tap into a direct in-leaf lookup. Substantially
    //                   faster, but must be called by all threads in the block (it uses
    //                   __syncthreads) and costs shared memory.
    // Independently, compute<Name> materializes the taps into a 27-slot array while
    // forEach<Name> streams (tap, index) to a device-inlined callback, which avoids the
    // per-thread array entirely for consumers that can take the taps in order.
    //
    // Every resolver resolves the taps of ONE decoded slot: the (leafIndex, voxelOffset)
    // pair its calling thread passes in, typically decodeInverseMap(..., threadIdx.x, ...).
    // The cached forms additionally take the block's firstLeafID and jumpMap, from which
    // they derive everything the staging needs - the block's first leaf IS firstLeafID,
    // and the spanned-leaf count is 1 + the jumpMap popcount - so no materialized
    // per-block maps are required anywhere.

    /// @brief Number of distinct, consecutive leaves the block's slots span, clamped to
    /// MaxCachedLeaves: one (the block's first leaf) plus one per leaf-start bit in the
    /// block's jumpMap. Shared preamble of the cached resolvers, which stage one table
    /// entry group per spanned leaf. Thread-local: derived from the VBM metadata alone,
    /// so it needs no leader election, atomics, or barriers.
    template <int MaxCachedLeaves>
    __device__
    static int cachedLeafSpan(const uint64_t *jumpMap)
    {
        int spanned = 1;
        #pragma unroll
        for (int i = 0; i < JumpMapLength; ++i)
            spanned += util::countOn(jumpMap[i]);
        return spanned < MaxCachedLeaves ? spanned : MaxCachedLeaves;
    }

    /// @brief Stage the per-spanned-leaf neighbor table shared by the cached resolvers.
    ///
    /// Fills table[leafSlot*N + neighborID] with the leaf lying at that neighbor position of
    /// the leafSlot-th spanned leaf, or nullptr if absent; entry SelfIndex stores the spanned
    /// leaf itself rather than probing for it. Each resolver supplies the shift from a leaf
    /// origin to its n-th neighbor's origin, so it keeps whatever indexing its lookup expects.
    /// The spannedLeafCount x N entries are distributed across ALL threads, so no thread serializes
    /// the probes.
    ///
    /// Must be called by all threads in the block (ends in __syncthreads).
    ///
    /// @tparam N          Table entries per spanned leaf
    /// @tparam SelfIndex  Entry index holding the spanned leaf itself (not probed for)
    /// @tparam OffsetT    Device-callable (Coord& origin, int neighborID) shifting a leaf origin
    ///                    to the origin of the leaf at that neighbor position
    /// @param table            Flattened [MaxCachedLeaves][N] leaf-pointer table in shared memory
    /// @param spannedLeafCount Number of spanned leaves (from cachedLeafSpan)
    /// @param firstLeaf        Leaf index of the block's first slot
    template <int N, int SelfIndex, class BuildT, class LeafT, class OffsetT>
    __device__
    static void stageLeafTable(const NanoGrid<BuildT> *grid, const LeafT **table,
                               int spannedLeafCount, uint32_t firstLeaf, OffsetT shiftToNeighbor)
    {
        const int tID = threadIdx.x;
        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();
        for (int entry = tID; entry < spannedLeafCount * N; entry += blockDim.x) {
            const int leafSlot = entry / N, neighborID = entry % N;
            if (neighborID == SelfIndex) {// this spanned leaf itself; no probe needed
                table[leafSlot*N + neighborID] = leaf0 + (firstLeaf + leafSlot);
                continue;
            }
            Coord neighborOrigin = leaf0[firstLeaf + leafSlot].origin();
            shiftToNeighbor(neighborOrigin, neighborID);
            table[leafSlot*N + neighborID] = tree.root().probeLeaf(neighborOrigin);
        }
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

    /// @brief Visit each 3x3x3 stencil index without materializing a 27-element array.
    /// Consumers that reduce or accumulate taps should prefer this streaming form over
    /// computeBoxStencil: the callback is device-inlined and receives (tap, index) in the
    /// same deterministic tap order, so lookup and arithmetic stay fused (no 27-element
    /// per-thread array, no stack spill). Keep computeBoxStencil for callers that need
    /// random access to all taps. Thread-local like computeBoxStencil: no shared memory,
    /// no synchronization, safe from divergent threads - which is the reason to use this
    /// rather than the faster forEachBoxStencilCached, whose cooperative leaf table
    /// requires all threads to participate.
    /// @tparam BuildT Build type of the grid
    /// @tparam OpT    Device-callable with signature op(int tap, uint64_t index)
    template <class BuildT, class OpT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    forEachBoxStencil(
        const NanoGrid<BuildT> *grid,
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        OpT op)
    {
        NANOVDB_ASSERT(grid->isSequential());
        const auto& tree = grid->tree();
        if (leafIndex == UnusedLeafIndex) return;
        const auto& leaf = tree.template getFirstNode<0>()[leafIndex];
        const Coord coord = leaf.offsetToGlobalCoord(voxelOffset);
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
    /// uncached leaves. Must be called by all threads in the block (uses __syncthreads), each
    /// passing its own decoded slot.
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
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        uint64_t *stencilIndices)
    {
        // Identical resolution to forEachBoxStencilCached; the only difference is that the
        // taps are stored rather than streamed, so express it as that traversal with a
        // storing callback instead of duplicating the table staging and tap loop.
        forEachBoxStencilCached<BuildT>(grid, firstLeafID, jumpMap, leafIndex, voxelOffset,
            [stencilIndices](int tap, uint64_t index) { stencilIndices[tap] = index; });
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
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        uint64_t *stencilIndices)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        const auto& tree = grid->tree();
        if (leafIndex != UnusedLeafIndex) {
            const auto& leaf = tree.template getFirstNode<0>()[ leafIndex ];
            const Coord coord = leaf.offsetToGlobalCoord( voxelOffset );
            stencilIndices[13] = leaf.getValue( voxelOffset );
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
    /// in the block (uses __syncthreads internally), each passing its own decoded slot. Use
    /// computeCrossStencil when the call site may be divergent, or when the block cannot spare
    /// the shared memory.
    ///
    /// @note There is deliberately no streaming (forEach) cross variant: a 7-element stencil is
    /// cheap enough to materialize that removing the array measured as a wash.
    template <class BuildT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    computeCrossStencilCached(
        const NanoGrid<BuildT> *grid,
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        uint64_t *stencilIndices)
    {
        NANOVDB_ASSERT(grid->isSequential());

        using LeafT = typename NanoTree<BuildT>::LeafNodeType;
        constexpr int MaxCachedLeaves = 16;
        // Slots 0..5: -x,+x,-y,+y,-z,+z face neighbors; slot 6: own leaf
        __shared__ const LeafT* sFaceLeaves[MaxCachedLeaves][7];

        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();

        // Spanned-leaf count straight from the VBM metadata; no election or barriers
        const int spannedLeafCount = cachedLeafSpan<MaxCachedLeaves>(jumpMap);
        // Entry order 0..5 is -x,+x,-y,+y,-z,+z (axis*2 + dir), which the lookup below assumes.
        stageLeafTable<7, 6>(grid, &sFaceLeaves[0][0], spannedLeafCount, firstLeafID,
            [](Coord &o, int n) { o[n >> 1] += (n & 1) ? 8 : -8; });

        if (leafIndex != UnusedLeafIndex) {
            const auto& leaf = leaf0[leafIndex];
            const Coord coord = leaf.offsetToGlobalCoord( voxelOffset );
            const uint32_t leafSlot = leafIndex - firstLeafID;
            const bool isCached = leafSlot < MaxCachedLeaves;
            stencilIndices[13] = leaf.getValue( voxelOffset );
            for (int n = 0; n < 6; ++n) {
                const int axis = n >> 1, dir = (n & 1) ? 1 : -1;
                Coord neighbor = coord;
                neighbor[axis] += dir;
                // Standard 27-slot spoke id for this face tap
                const int spokeID = 13 + dir * (axis == 0 ? 9 : axis == 1 ? 3 : 1);
                if (isCached) {
                    const int voxelOnAxis = coord[axis] & 7;// 0..7 within the leaf
                    // stepping off the leaf on this axis? then use the face neighbor, else self
                    const LeafT* neighborLeaf = ((voxelOnAxis + dir) & ~7) ? sFaceLeaves[leafSlot][n]
                                                                          : sFaceLeaves[leafSlot][6];
                    stencilIndices[spokeID] = neighborLeaf ? neighborLeaf->getValue(LeafT::CoordToOffset(neighbor)) : 0;
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
    /// it must be called by all threads in the block (uses __syncthreads internally), each
    /// passing its own decoded slot. Use forEachBoxStencil when the call site may be divergent.
    ///
    /// @tparam BuildT Build type of the grid
    /// @tparam OpT    Device-callable with signature op(int tap, uint64_t index)
    template <class BuildT, class OpT>
    __device__
    static typename util::enable_if<BuildTraits<BuildT>::is_index, void>::type
    forEachBoxStencilCached(
        const NanoGrid<BuildT> *grid,
        const uint32_t firstLeafID,
        const uint64_t *jumpMap,
        const uint32_t leafIndex,
        const uint16_t voxelOffset,
        OpT op)
    {
        NANOVDB_ASSERT(grid->isSequential());

        using LeafT = typename NanoTree<BuildT>::LeafNodeType;
        constexpr int MaxCachedLeaves = 16;
        __shared__ const LeafT* sNeighborLeaves[MaxCachedLeaves][27];

        const auto& tree = grid->tree();
        const LeafT* leaf0 = tree.template getFirstNode<0>();

        // Spanned-leaf count straight from the VBM metadata; no election or barriers.
        // Stage the spannedLeaves x 27 leaf-neighborhood table across ALL threads.
        const int spannedLeafCount = cachedLeafSpan<MaxCachedLeaves>(jumpMap);
        // The full box neighborhood: entry n is the 3x3x3 spoke n.
        stageLeafTable<27, 13>(grid, &sNeighborLeaves[0][0], spannedLeafCount, firstLeafID,
            [](Coord &o, int n) { o = o.offsetBy((n/9-1)*8, ((n/3)%3-1)*8, (n%3-1)*8); });

        if (leafIndex == UnusedLeafIndex) return;
        const auto& leaf = leaf0[leafIndex];
        const Coord coord = leaf.offsetToGlobalCoord( voxelOffset );
        const uint32_t leafSlot = leafIndex - firstLeafID;
        const bool isCached = leafSlot < MaxCachedLeaves;
        const int voxelX = coord[0] & 7, voxelY = coord[1] & 7, voxelZ = coord[2] & 7;
        for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
        for (int dk = -1; dk <= 1; dk++) {
            const int spokeID = ( di + 1 ) * 9 + ( dj + 1 ) * 3 + dk + 1;
            const Coord neighbor = coord.offsetBy( di, dj, dk );
            if (isCached) {
                // which of the 3x3x3 neighboring leaves this tap falls into
                const int leafX = ((voxelX+di)>>3)+1, leafY = ((voxelY+dj)>>3)+1, leafZ = ((voxelZ+dk)>>3)+1;
                const LeafT* neighborLeaf = sNeighborLeaves[leafSlot][leafX*9 + leafY*3 + leafZ];
                op(spokeID, neighborLeaf ? neighborLeaf->getValue(LeafT::CoordToOffset(neighbor))
                                         : uint64_t(0));
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
