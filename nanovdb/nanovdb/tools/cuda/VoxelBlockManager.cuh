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
      - buildVoxelBlockManagerBatch (device): builds the VBMs of a batch of grids
        into shared arrays with one kernel launch and no host synchronization.
      - VoxelBlockManagerBatchView (host/device): each grid's slice of a batched
        build, and the grid that owns a batch-wide block.
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
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/util/cuda/Util.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/cuda/HandleStorage.h>
#include <atomic>
#include <cstddef> // for std::byte

#include <type_traits>

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
    ///                   that index is beyond the last active voxel, or if
    ///                   firstLeafID is UnusedLeafIndex (a block with no voxel).
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

        if (firstLeafID == UnusedLeafIndex) { // the block covers no voxel
            leafIndex = UnusedLeafIndex;
            voxelOffset = UnusedVoxelOffset;
            return;
        }
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

    /// @brief Visit each 3x3x3 stencil index without materializing a 27-element array.
    /// Consumers that reduce or accumulate taps should prefer this streaming form over
    /// computeBoxStencil: the callback is device-inlined and receives (tap, index) in the
    /// same deterministic tap order, so lookup and arithmetic stay fused (no 27-element
    /// per-thread array, no stack spill). Keep computeBoxStencil for callers that need
    /// random access to all taps. Like computeBoxStencil this accesses shared memory but
    /// does not synchronize, so it may be called from divergent threads within a block.
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
};

namespace detail {

/// @brief Number of CTAs of @c BlockSize threads that fill the current device with @c Kernel:
///        resident CTAs per SM times the SM count. Computed once per device and cached for
///        each kernel and block size, so launches of a grid-stride kernel need nothing from
///        the data they process.
template<auto Kernel, int BlockSize>
inline unsigned deviceFillingGridSize()
{
    constexpr int MaxCachedDevices = 64;
    static std::atomic<unsigned> cache[MaxCachedDevices]; // zero-initialized: not yet computed
    int device = 0;
    cudaCheck(cudaGetDevice(&device));
    if (device < MaxCachedDevices) {
        if (const unsigned n = cache[device].load(std::memory_order_relaxed)) return n;
    }
    int smCount = 0, perSM = 0;
    cudaCheck(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, Kernel, BlockSize, 0));
    const unsigned n = unsigned(smCount > 0 ? smCount : 1) * unsigned(perSM > 0 ? perSM : 1);
    if (device < MaxCachedDevices) cache[device].store(n, std::memory_order_relaxed);
    return n;
}

} // namespace detail

/// @brief This functor calculates the firstLeafID and jumpMap for the
/// VoxelBlockManager over the subset of the Tree nodes specified by
/// firstOffset, lastOffset, and nBlocks. One leaf per loop iteration; since
/// the grid is sequential the leaf array is dense and indexed directly. The
/// leaf count is read from the device-resident tree, and the grid-stride loops
/// cover all leaves and all blocks past the last voxel for any launch size.
template<int Log2BlockWidth>
struct BuildVoxelBlockManagerFunctor
{
    using Base = nanovdb::tools::VoxelBlockManagerBase<Log2BlockWidth>;
    static constexpr int BlockWidth = 1 << Log2BlockWidth;
    static constexpr int JumpMapLength = BlockWidth/64;

    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    void __device__
    operator()(
        uint64_t firstOffset,
        uint64_t lastOffset,
        uint64_t nBlocks,
        const NanoGrid<ValueOnIndex> *grid,
        uint32_t *firstLeafID,
        uint64_t *jumpMap)
    {
        // Verify that the nodes can be accessed linearly
        NANOVDB_ASSERT(grid->isSequential());

        const uint32_t leafCount = grid->tree().nodeCount(0);
        const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
        const uint64_t tID0 = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        for (uint64_t tID = tID0; tID < leafCount; tID += stride)
        {
            Base::buildLeaf(grid, uint32_t(tID), leafCount, firstOffset, lastOffset, nBlocks,
                            firstLeafID, jumpMap);
        }

        // Blocks past the last voxel
        for (uint64_t b = Base::usedBlockCount(firstOffset, lastOffset, nBlocks) + tID0;
             b < nBlocks; b += stride)
        {
            firstLeafID[b] = Base::UnusedLeafIndex;
        }
    }

};

/// @brief One grid of a batched VoxelBlockManager build (see buildVoxelBlockManagerBatch).
///        Each grid's VBM covers its full active range [1, voxelCount] and occupies blocks
///        [blockBase, blockBase + ceil(voxelCount / BlockWidth)) of the batch's shared
///        firstLeafID and jumpMap arrays. Its leaves are batch-wide leaves
///        [leafBase, leafBase + leafCount).
struct VoxelBlockManagerBatchEntry
{
    const NanoGrid<ValueOnIndex> *grid; // device pointer; never read when the grid has no leaves
    uint64_t voxelCount;                // active voxels in the grid (its VBM's lastOffset)
    uint64_t blockBase;                 // first block of the grid's VBM in the shared arrays
    uint32_t leafBase;                  // exclusive prefix sum of the grids' leaf counts
};

/// @brief Entry source over an array of VoxelBlockManagerBatchEntry records, in device memory
///        for the build and in either memory space for a VoxelBlockManagerBatchView.
struct VoxelBlockManagerBatchEntryArray
{
    const VoxelBlockManagerBatchEntry *entries;

    __hostdev__ VoxelBlockManagerBatchEntry operator()(uint32_t g) const { return entries[g]; }
};

namespace detail {

/// @brief Number of the n ascending keys key(0), ..., key(n - 1) that do not exceed x.
///        The last of them, at that count minus one, is the owner of x in a list of bases.
NANOVDB_HOSTDEV_DISABLE_WARNING
template<typename KeyT>
__hostdev__ inline uint32_t countAtOrBelow(uint32_t n, uint64_t x, const KeyT &key)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (key(mid) <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/// @brief Detects an entry source's optional leafBase(g), blockBase(g) and voxelCount(g). A
///        source that can produce those fields more cheaply than a whole entry, e.g. from one
///        compact array, provides them so that searches and per-grid lookups read only what
///        they need.
template<typename T, typename = void>
struct HasLeafBase : std::false_type {};
template<typename T>
struct HasLeafBase<T, std::void_t<decltype(std::declval<const T&>().leafBase(uint32_t(0)))>> : std::true_type {};
template<typename T, typename = void>
struct HasBlockBase : std::false_type {};
template<typename T>
struct HasBlockBase<T, std::void_t<decltype(std::declval<const T&>().blockBase(uint32_t(0)))>> : std::true_type {};
template<typename T, typename = void>
struct HasVoxelCount : std::false_type {};
template<typename T>
struct HasVoxelCount<T, std::void_t<decltype(std::declval<const T&>().voxelCount(uint32_t(0)))>> : std::true_type {};

NANOVDB_HOSTDEV_DISABLE_WARNING
template<typename EntrySource>
__hostdev__ inline uint64_t entryLeafBase(const EntrySource &entries, uint32_t g)
{
    if constexpr (HasLeafBase<EntrySource>::value) return uint64_t(entries.leafBase(g));
    else return entries(g).leafBase;
}

NANOVDB_HOSTDEV_DISABLE_WARNING
template<typename EntrySource>
__hostdev__ inline uint64_t entryBlockBase(const EntrySource &entries, uint32_t g)
{
    if constexpr (HasBlockBase<EntrySource>::value) return uint64_t(entries.blockBase(g));
    else return entries(g).blockBase;
}

NANOVDB_HOSTDEV_DISABLE_WARNING
template<typename EntrySource>
__hostdev__ inline uint64_t entryVoxelCount(const EntrySource &entries, uint32_t g)
{
    if constexpr (HasVoxelCount<EntrySource>::value) return uint64_t(entries.voxelCount(g));
    else return entries(g).voxelCount;
}

} // namespace detail

/// @brief Read-only view of a batched VoxelBlockManager build: the shared firstLeafID and
///        jumpMap arrays plus the entry source that placed the grids in them. Gives each
///        grid's slice of the arrays, and maps a batch-wide block to its grid. Grid g's slice
///        decodes with the usual per-grid calls, at block offsets relative to the slice.
///        Callable on the host and the device, wherever the entry source and arrays live.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam EntrySource     Callable with VoxelBlockManagerBatchEntry operator()(uint32_t g)
///                         const, as for buildVoxelBlockManagerBatch. It may also provide
///                         blockBase(g) and voxelCount(g); the view then reads those instead
///                         of whole entries.
template<int Log2BlockWidth, typename EntrySource = VoxelBlockManagerBatchEntryArray>
class VoxelBlockManagerBatchView
{
public:
    using Base = nanovdb::tools::VoxelBlockManagerBase<Log2BlockWidth>;
    static constexpr int BlockWidth = Base::BlockWidth;
    static constexpr int JumpMapLength = Base::JumpMapLength;
    /// Returned by gridOfBlock for a block that belongs to no grid
    static constexpr uint32_t UnusedGridIndex = ~uint32_t{0};

    /// @param entries      Entry source the batch was built from
    /// @param gridCount    Number of grids in the batch
    /// @param blockCount   Total blocks in the batch, gaps included
    /// @param firstLeafID  The batch's firstLeafID array of blockCount entries
    /// @param jumpMap      The batch's jumpMap array of blockCount * JumpMapLength words
    __hostdev__ VoxelBlockManagerBatchView(const EntrySource &entries, uint32_t gridCount, uint64_t blockCount,
                                           const uint32_t *firstLeafID, const uint64_t *jumpMap)
        : mEntries(entries), mGridCount(gridCount), mBlockCount(blockCount)
        , mFirstLeafID(firstLeafID), mJumpMap(jumpMap) {}

    __hostdev__ uint32_t gridCount() const { return mGridCount; }

    /// @brief Total blocks in the batch, gaps included
    __hostdev__ uint64_t blockCount() const { return mBlockCount; }

    /// @brief The batch's whole firstLeafID array; gap blocks hold UnusedLeafIndex
    __hostdev__ const uint32_t *firstLeafID() const { return mFirstLeafID; }

    /// @brief The batch's whole jumpMap array; gap blocks are zero
    __hostdev__ const uint64_t *jumpMap() const { return mJumpMap; }

    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ VoxelBlockManagerBatchEntry entry(uint32_t g) const { return mEntries(g); }

    /// @brief First block of grid g in the batch's arrays
    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint64_t blockBase(uint32_t g) const { return detail::entryBlockBase(mEntries, g); }

    /// @brief Active voxels in grid g
    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint64_t voxelCount(uint32_t g) const { return detail::entryVoxelCount(mEntries, g); }

    /// @brief Number of blocks of grid g: ceil(voxelCount / BlockWidth), 0 for an empty grid
    __hostdev__ uint64_t blockCount(uint32_t g) const
    {
        return (this->voxelCount(g) + BlockWidth - 1) >> Log2BlockWidth;
    }

    /// @brief Grid g's slice of the firstLeafID array, blockCount(g) entries long
    __hostdev__ const uint32_t *firstLeafID(uint32_t g) const { return mFirstLeafID + this->blockBase(g); }

    /// @brief Grid g's slice of the jumpMap array, blockCount(g) * JumpMapLength words long
    __hostdev__ const uint64_t *jumpMap(uint32_t g) const
    {
        return mJumpMap + this->blockBase(g) * JumpMapLength;
    }

    /// @brief The last grid whose first block is at or before batch-wide block b, or
    ///        UnusedGridIndex if b precedes the first grid. A binary search over the grids'
    ///        block bases, without gridOfBlock's check that b lies within that grid: b may
    ///        be a gap block after it. For callers that check the result against the grid's
    ///        extent anyway, e.g. a slot's voxel index against voxelCount(g).
    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint32_t lastGridAtOrBefore(uint64_t b) const
    {
        return detail::countAtOrBelow(mGridCount, b,
            [this](uint32_t g) { return this->blockBase(g); }) - 1; // 0 - 1 is UnusedGridIndex
    }

    /// @brief The grid that owns batch-wide block b, or UnusedGridIndex for a gap block
    ///        (before, between or after the grids). The owner's slice holds b at local
    ///        block b - blockBase(owner). A binary search over the grids' block bases.
    NANOVDB_HOSTDEV_DISABLE_WARNING
    __hostdev__ uint32_t gridOfBlock(uint64_t b) const
    {
        const uint32_t g = this->lastGridAtOrBefore(b);
        if (g == UnusedGridIndex) return UnusedGridIndex;
        return b - this->blockBase(g) < this->blockCount(g) ? g : UnusedGridIndex;
    }

private:
    EntrySource mEntries;
    uint32_t mGridCount;
    uint64_t mBlockCount;
    const uint32_t *mFirstLeafID;
    const uint64_t *mJumpMap;
};

/// @brief Builds the VBMs of a batch of grids, one thread per leaf across the batch. Each
///        thread finds its grid from the leaf bases, then records its leaf exactly as the
///        single-grid build does, into that grid's slice of the shared arrays. Thread g also
///        sets the firstLeafID of the gap after grid g (and thread 0 the gap before grid 0)
///        to UnusedLeafIndex.
/// @tparam EntrySource  Device-callable returning grid g's VoxelBlockManagerBatchEntry
template<int Log2BlockWidth, typename EntrySource = VoxelBlockManagerBatchEntryArray>
struct BuildVoxelBlockManagerBatchFunctor
{
    using Single = BuildVoxelBlockManagerFunctor<Log2BlockWidth>;

    static constexpr int MaxThreadsPerBlock = Single::MaxThreadsPerBlock;
    static constexpr int MinBlocksPerMultiprocessor = Single::MinBlocksPerMultiprocessor;

    void __device__
    operator()(
        const EntrySource entries,
        uint32_t gridCount,
        uint32_t leafCount,
        uint64_t blockCount,
        uint32_t *firstLeafID,
        uint64_t *jumpMap)
    {
        const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
        const uint64_t tID0 = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        for (uint64_t tID = tID0; tID < leafCount; tID += stride)
        {
            // The last grid whose leaf base does not exceed tID; grids without leaves share
            // their successor's base and are skipped that way. The first leaf base is 0, so
            // some grid always qualifies; a source that breaks that contract skips the leaf.
            const uint32_t n = detail::countAtOrBelow(gridCount, tID,
                [&entries](uint32_t g) { return detail::entryLeafBase(entries, g); });
            NANOVDB_ASSERT(n > 0);
            if (!n) continue;
            const VoxelBlockManagerBatchEntry entry = entries(n - 1);
            const uint32_t leafIndex = uint32_t(tID) - entry.leafBase;
            if (!entry.voxelCount) continue;
            const uint32_t gridLeafCount = entry.grid->tree().nodeCount(0);
            if (leafIndex >= gridLeafCount) continue;
            NANOVDB_ASSERT(entry.grid->isSequential());

            const uint64_t nBlocks = (entry.voxelCount + Single::BlockWidth - 1) >> Log2BlockWidth;
            Single::Base::buildLeaf(entry.grid, leafIndex, gridLeafCount, 1, entry.voxelCount, nBlocks,
                                    firstLeafID + entry.blockBase,
                                    jumpMap + entry.blockBase * Single::JumpMapLength);
        }

        // Gap blocks before, between and after the grids
        for (uint64_t g = tID0; g < gridCount; g += stride) {
            const VoxelBlockManagerBatchEntry entry = entries(uint32_t(g));
            if (g == 0) {
                for (uint64_t b = 0; b < entry.blockBase; ++b) firstLeafID[b] = Single::Base::UnusedLeafIndex;
            }
            const uint64_t end = g + 1 < gridCount ? entries(uint32_t(g + 1)).blockBase : blockCount;
            for (uint64_t b = entry.blockBase + ((entry.voxelCount + Single::BlockWidth - 1) >> Log2BlockWidth);
                 b < end; ++b)
            {
                firstLeafID[b] = Single::Base::UnusedLeafIndex;
            }
        }
    }
};

/// @brief Build a VoxelBlockManager into a pre-allocated handle: zeros the jumpMap
///        on-stream and launches the build kernel, with no memory allocation. This is
///        the fill step of the allocating overload, and the entry point for callers
///        that own the firstLeafID and jumpMap memory themselves. It can also be
///        called repeatedly on the same handle, e.g. for benchmarking. Blocks past the
///        last voxel get firstLeafID UnusedLeafIndex.
///        Fully stream-asynchronous: nothing is read back from the device. The kernel
///        loops grid-stride over the grid's device-resident leaves, so the launch is
///        sized to fill the device rather than to the grid.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam BufferT         Device buffer type (deduced from handle)
/// @param d_grid  Device-side grid pointer passed to the build kernel; must meet the
///                topology preconditions in nanovdb/tools/VoxelBlockManager.h
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

    // the buffers are allocated uninitialized (noInit); jumpMap must be zeroed each build
    cudaCheck(cudaMemsetAsync(handle.deviceJumpMap(), 0,
        handle.blockCount() * JumpMapLength * sizeof(uint64_t), stream));

    using Op = BuildVoxelBlockManagerFunctor<Log2BlockWidth>;
    constexpr auto kernel = util::cuda::operatorKernel<Op, uint64_t, uint64_t, uint64_t,
        NanoGrid<ValueOnIndex>*, uint32_t*, uint64_t*>;
    kernel<<<detail::deviceFillingGridSize<kernel, Op::MaxThreadsPerBlock>(), Op::MaxThreadsPerBlock, 0, stream>>>(
        handle.firstOffset(), handle.lastOffset(), handle.blockCount(),
        d_grid, handle.deviceFirstLeafID(), handle.deviceJumpMap());
}

/// @brief Allocate device buffers and build a VoxelBlockManager on the device.
///        Returns a fully-constructed VoxelBlockManagerHandle backed by device memory.
///        Grid dimensions (when not supplied) are read from device memory via DeviceGridTraits,
///        which synchronizes; with an explicit lastOffset the build is fully asynchronous.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam BufferT         Device buffer type (default: nanovdb::cuda::Buffer<std::byte>)
/// @param d_grid       Device-side grid pointer; must meet the topology preconditions in
///                     nanovdb/tools/VoxelBlockManager.h
/// @param firstOffset  First active-voxel offset covered by this VBM; must satisfy
///                     firstOffset == 1 (mod BlockWidth). Pass 0 (default) to use 1,
///                     which covers the full grid from the first active voxel.
/// @param lastOffset   Last active-voxel offset covered by this VBM. Pass 0 (default)
///                     to read activeVoxelCount from device memory via DeviceGridTraits.
/// @param nBlocks      Allocated capacity in blocks; must be >=
///                     ceil((lastOffset - firstOffset + 1) / BlockWidth). Pass 0
///                     (default) to use the minimum required capacity. Blocks past
///                     the last voxel get firstLeafID UnusedLeafIndex.
/// @param stream       CUDA stream (default 0)
/// @return A fully constructed VoxelBlockManagerHandle backed by device memory
template<int Log2BlockWidth, typename BufferT = nanovdb::cuda::Buffer<std::byte>>
nanovdb::tools::VoxelBlockManagerHandle<BufferT>
buildVoxelBlockManager(
    NanoGrid<ValueOnIndex>* d_grid,
    uint64_t                firstOffset = 0,
    uint64_t                lastOffset  = 0,
    uint64_t                nBlocks     = 0,
    cudaStream_t            stream      = 0,
    const BufferT*          proto       = nullptr)
{
    static constexpr uint64_t BlockWidth    = uint64_t(1) << Log2BlockWidth;
    static constexpr uint64_t JumpMapLength = BlockWidth / 64;

    using Traits = util::cuda::DeviceGridTraits<ValueOnIndex>;
    if (!firstOffset) firstOffset = 1;
    if (!lastOffset)  lastOffset  = Traits::getActiveVoxelCount(d_grid, stream); // synchronizes
    if (lastOffset < firstOffset) {// empty grid: an empty handle, through the pool's resource when one is needed
        if constexpr (BufferIsDefaultConstructible<BufferT>::value) {
            return nanovdb::tools::VoxelBlockManagerHandle<BufferT>{};
        } else {
            if (!proto)
                throw std::runtime_error("buildVoxelBlockManager: an empty handle over a buffer type that is "
                                         "not default-constructible requires a prototype buffer to take the resource from");
            return nanovdb::tools::VoxelBlockManagerHandle<BufferT>(
                nanovdb::cuda::detail::createDeviceStorage<BufferT>(0, proto, stream),
                nanovdb::cuda::detail::createDeviceStorage<BufferT>(0, proto, stream),
                0, firstOffset, lastOffset); // zero-size buffers allocate nothing
        }
    }
    NANOVDB_ASSERT(!((firstOffset - 1) & (BlockWidth - 1))); // firstOffset == 1 (mod BlockWidth)
    if (!nBlocks)     nBlocks     = (lastOffset - firstOffset + BlockWidth) >> Log2BlockWidth;

    auto firstLeafIDBuf = nanovdb::cuda::detail::createDeviceStorage<BufferT>(nBlocks * sizeof(uint32_t),                proto, stream);
    auto jumpMapBuf     = nanovdb::cuda::detail::createDeviceStorage<BufferT>(nBlocks * JumpMapLength * sizeof(uint64_t), proto, stream);

    nanovdb::tools::VoxelBlockManagerHandle<BufferT> handle(
        std::move(firstLeafIDBuf), std::move(jumpMapBuf),
        nBlocks, firstOffset, lastOffset);

    buildVoxelBlockManager<Log2BlockWidth>(d_grid, handle, stream);
    return handle;
}

/// @brief Build the VoxelBlockManagers of a batch of grids with one memset and one kernel
///        launch, into caller-allocated arrays. Fully stream-asynchronous: nothing is read
///        back from the device. Grid g's VBM is identical to what buildVoxelBlockManager
///        produces for it alone (firstOffset 1, lastOffset voxelCount) and sits at blocks
///        [entries(g).blockBase, ...) of the shared arrays; decode it with the usual
///        per-grid calls on firstLeafID + blockBase and jumpMap + blockBase * JumpMapLength.
///
///        Every grid must meet the topology preconditions in nanovdb/tools/VoxelBlockManager.h.
///
///        The entries come from a device-callable source, so a caller whose per-grid data
///        already lives on the device can derive them in the build kernel instead of
///        materializing an array first.
/// @tparam Log2BlockWidth  Log2 of the number of active voxels per VBM block
/// @tparam EntrySource     Device-callable, copyable to the device by value, with
///                         VoxelBlockManagerBatchEntry operator()(uint32_t g) const. It may
///                         also provide leafBase(g), which the kernel's per-leaf grid search
///                         then reads instead of whole entries.
/// @param entries        Source of the gridCount entries. The leaf bases must be exclusive
///                       prefix sums of the grids' exact leaf counts, so empty grids repeat
///                       their successor's leaf base. The block bases must be ascending, with
///                       each grid's blocks [blockBase, blockBase + ceil(voxelCount / BlockWidth))
///                       ending at or before the next grid's blockBase. Gaps before, between
///                       and after grids are allowed: their jumpMap words are zeroed and their
///                       firstLeafID entries set to UnusedLeafIndex, so they decode to sentinels.
/// @param gridCount      Number of grids in the batch
/// @param leafCount      Total leaves in the batch
/// @param blockCount     Total blocks in the batch, gaps included: the size of d_firstLeafID,
///                       and of d_jumpMap in units of JumpMapLength words
/// @param d_firstLeafID  Device array of blockCount entries; every entry is written
/// @param d_jumpMap      Device array of blockCount * JumpMapLength words; zeroed, then built
/// @param stream         CUDA stream (default 0)
template<int Log2BlockWidth, typename EntrySource,
         typename = std::enable_if_t<
             std::is_invocable_r_v<VoxelBlockManagerBatchEntry, const EntrySource&, uint32_t>>>
void buildVoxelBlockManagerBatch(
    const EntrySource& entries,
    uint32_t           gridCount,
    uint32_t           leafCount,
    uint64_t           blockCount,
    uint32_t*          d_firstLeafID,
    uint64_t*          d_jumpMap,
    cudaStream_t       stream = 0)
{
    using Op = BuildVoxelBlockManagerBatchFunctor<Log2BlockWidth, EntrySource>;

    if (!blockCount) return;
    cudaCheck(cudaMemsetAsync(d_jumpMap, 0,
        blockCount * Op::Single::JumpMapLength * sizeof(uint64_t), stream));
    if (!gridCount) { // every block is a gap
        static_assert(Op::Single::Base::UnusedLeafIndex == ~uint32_t{0}, "memset fills one bits");
        cudaCheck(cudaMemsetAsync(d_firstLeafID, 0xff, blockCount * sizeof(uint32_t), stream));
        return;
    }

    const uint32_t threads = leafCount > gridCount ? leafCount : gridCount;
    util::cuda::operatorKernel<Op>
        <<<util::cuda::blocksPerGrid(threads, Op::MaxThreadsPerBlock), Op::MaxThreadsPerBlock, 0, stream>>>(
            entries, gridCount, leafCount, blockCount, d_firstLeafID, d_jumpMap);
}

/// @brief buildVoxelBlockManagerBatch over a device array of gridCount entries.
template<int Log2BlockWidth>
void buildVoxelBlockManagerBatch(
    const VoxelBlockManagerBatchEntry* d_entries,
    uint32_t                           gridCount,
    uint32_t                           leafCount,
    uint64_t                           blockCount,
    uint32_t*                          d_firstLeafID,
    uint64_t*                          d_jumpMap,
    cudaStream_t                       stream = 0)
{
    buildVoxelBlockManagerBatch<Log2BlockWidth>(VoxelBlockManagerBatchEntryArray{d_entries},
        gridCount, leafCount, blockCount, d_firstLeafID, d_jumpMap, stream);
}

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_VOXELBLOCKMANAGER_CUH_HAS_BEEN_INCLUDED
