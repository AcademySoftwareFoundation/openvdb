// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/Morphology.h

    \authors Efty Sifakis

    \brief Host-side implementations of the per-tile morphology operations used by the
           topology operators (DilateGrid, MergeGrids, PruneGrid, RefineGrid, CoarsenGrid).
           Parallel counterparts to the CUDA functors in nanovdb/util/cuda/Morphology.cuh.

    \note  Functions here take ownership of their parallel iteration (via util::forEach).
           They are not designed as per-item functors — the parallelization shape on host
           (e.g., word-granular iteration to expose adequate parallelism on small grids)
           is an implementation detail of each function, distinct from the CUDA-side
           cooperative-block-with-warp-reduce structure.
*/

#ifndef NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED

#include <cstdint>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/ForEach.h>

namespace nanovdb::util::morphology {

/// @brief Enumerate which lower-tile slots are non-empty under each upper tile, and how many
///        active voxels (leaf-mask bits) each non-empty lower contains. Writes results into
///        densified per-tile-per-slot arrays.
///
/// @param upperMasks_         (in)  Array of Mask<5>, one per processed upper tile.
/// @param lowerMasks_         (in)  Array of (Mask<4>[Mask<5>::SIZE]), one set per upper tile.
/// @param lowerCounts         (out) [upperID][slot] = 1 iff upperMask[upperID].isOn(slot), else 0.
/// @param leafCounts          (out) [upperID][slot] = popcount of lowerMasks[upperID][slot]
///                                  iff upperMask[upperID].isOn(slot), else 0.
/// @param processedTileCount  (in)  Number of upper tiles in the speculative root.
///
/// Iteration is over upper-mask words (tileCount × Mask<5>::WORD_COUNT = tileCount × 512).
/// Each task handles one 64-bit word — i.e. 64 consecutive lower slots within one upper.
/// A word-equals-zero fast path skips the popcount work entirely for sparse upper regions.
/// Writes to distinct (upperID, wordInUpper) are to disjoint 64-slot output regions —
/// no atomics needed.
inline void EnumerateNodes(
    const void *upperMasks_,
    const void *lowerMasks_,
    uint32_t (*lowerCounts)[Mask<5>::SIZE],
    uint32_t (*leafCounts)[Mask<5>::SIZE],
    std::size_t processedTileCount)
{
    using UpperMaskArrayT = const Mask<5>*;
    using LowerMaskArrayT = const Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    util::forEach(0, processedTileCount * Mask<5>::WORD_COUNT, /*grain=*/1,
        [=](const util::Range1D &r) {
            for (auto wordIdx = r.begin(); wordIdx != r.end(); ++wordIdx) {
                const auto upperID     = wordIdx / Mask<5>::WORD_COUNT;
                const auto wordInUpper = wordIdx % Mask<5>::WORD_COUNT;
                const auto baseSlot    = wordInUpper * 64;
                const uint64_t word    = upperMasks[upperID].words()[wordInUpper];

                if (word == 0) {
                    // Fast path: all 64 lower slots covered by this word are off.
                    for (uint32_t b = 0; b < 64; ++b) {
                        lowerCounts[upperID][baseSlot + b] = 0;
                        leafCounts[upperID][baseSlot + b]  = 0;
                    }
                    continue;
                }
                for (uint32_t b = 0; b < 64; ++b) {
                    const auto slot = baseSlot + b;
                    if (word & (uint64_t(1) << b)) {
                        const auto& lowerMask = lowerMasks[upperID][slot];
                        uint32_t leafCnt = 0;
                        for (int w = 0; w < 64; ++w)
                            leafCnt += util::countOn(lowerMask.words()[w]);
                        lowerCounts[upperID][slot] = 1;
                        leafCounts[upperID][slot]  = leafCnt;
                    } else {
                        lowerCounts[upperID][slot] = 0;
                        leafCounts[upperID][slot]  = 0;
                    }
                }
            }
        });
}

/// @brief Wire lower nodes and their leaf children into the output grid for all processed upper
///        tiles, using pre-computed offset arrays from countNodes().
///
/// @param upperMasks_        (in)  Array of Mask<5>, one per processed upper tile.
/// @param lowerMasks_        (in)  Array of (Mask<4>[Mask<5>::SIZE]), one set per upper tile.
/// @param upperOffsets       (in)  [processedTileID] -> index of upper node in output grid
///                                 (upperOffsets[i]==upperOffsets[i+1] means speculative tile absent).
/// @param lowerOffsets       (in)  [upperID][slot] -> index of lower node in output grid.
/// @param leafOffsets        (in)  [upperID][slot] -> base index of first leaf under that lower.
/// @param dstGrid            (in/out) Output grid buffer (zero-filled, node arrays already placed).
/// @param lowerParents       (out) [lowerID] -> upperID of its parent.
/// @param leafParents        (out) [leafID]  -> lowerID of its parent.
/// @param processedTileCount (in)  Number of upper tiles in the speculative root.
///
/// Iteration is over processedTileCount * Mask<5>::WORD_COUNT words (one task = 64 upper slots).
/// Zero-word fast path skips empty stretches. Within each active slot, the lower childmask is
/// copied word-by-word and lower→leaf pointers are wired with a sequential prefix sum — the
/// host replacement for the CUDA warp-level ExclusiveSum in ProcessLowerNodesFunctor.
/// No atomics are needed: each task owns a disjoint word of each upper's childmask and a
/// disjoint lower node, so all writes are to non-overlapping memory regions.
template<typename BuildT>
inline void ProcessLowerNodes(
    const void *upperMasks_,
    const void *lowerMasks_,
    const uint32_t *upperOffsets,
    const uint32_t (*lowerOffsets)[Mask<5>::SIZE],
    const uint32_t (*leafOffsets)[Mask<5>::SIZE],
    NanoGrid<BuildT> *dstGrid,
    uint32_t *lowerParents,
    uint32_t *leafParents,
    std::size_t processedTileCount)
{
    using UpperMaskArrayT = const Mask<5>*;
    using LowerMaskArrayT = const Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    const auto& dstTree = dstGrid->tree();

    util::forEach(0, processedTileCount * Mask<5>::WORD_COUNT, /*grain=*/1,
        [=, &dstTree](const util::Range1D &r) {
            for (auto wordIdx = r.begin(); wordIdx != r.end(); ++wordIdx) {
                const auto     processedTileID = wordIdx / Mask<5>::WORD_COUNT;
                const auto     wordInUpper     = wordIdx % Mask<5>::WORD_COUNT;
                const uint32_t upperID         = upperOffsets[processedTileID];

                if (upperOffsets[processedTileID + 1] == upperID) continue; // speculative tile absent

                const uint64_t word = upperMasks[processedTileID].words()[wordInUpper];
                if (word == 0) continue;

                const auto upperOrigin = dstTree.root().tile(upperID)->origin();
                auto& upper = const_cast<NanoUpper<BuildT>&>(dstTree.template getFirstNode<2>()[upperID]);

                // Set all active childmask bits for this word at once (disjoint word per task,
                // no atomics needed), then wire child pointers slot by slot.
                upper.mChildMask.words()[wordInUpper] = word;

                const auto baseSlot = wordInUpper * 64;
                for (uint32_t b = 0; b < 64; ++b) {
                    if (!(word & (uint64_t(1) << b))) continue;
                    const uint32_t jj = baseSlot + b;

                    auto lowerID = lowerOffsets[processedTileID][jj];
                    auto& lower = const_cast<NanoLower<BuildT>&>(dstTree.template getFirstNode<1>()[lowerID]);
                    const auto lowerOrigin = upperOrigin +
                        (NanoUpper<BuildT>::OffsetToLocalCoord(jj) << NanoUpper<BuildT>::ChildNodeType::TOTAL);

                    upper.setChild(jj, &lower);
                    lowerParents[lowerID] = upperID;

                    // Copy the lower mask word by word into lower.mChildMask, and wire
                    // lower→leaf with a running prefix sum (replaces CUDA WarpScan::ExclusiveSum).
                    auto lowerWords = lowerMasks[processedTileID][jj].words();
                    uint32_t prefixSum = 0;
                    for (uint32_t wordID = 0; wordID < Mask<4>::WORD_COUNT; ++wordID) {
                        const uint64_t lw = lowerWords[wordID];
                        lower.mChildMask.words()[wordID] = lw;
                        for (int bitID = 0; bitID < 64; ++bitID) {
                            if (!(lw & (uint64_t(1) << bitID))) continue;
                            const int      kk     = (wordID << 6) + bitID;
                            const uint32_t leafID = leafOffsets[processedTileID][jj] + prefixSum;
                            auto& leaf = const_cast<NanoLeaf<BuildT>&>(dstTree.template getFirstNode<0>()[leafID]);
                            lower.setChild(kk, &leaf);
                            leafParents[leafID] = lowerID;
                            const auto leafOrigin = lowerOrigin +
                                (NanoLower<BuildT>::OffsetToLocalCoord(kk) << NanoLower<BuildT>::ChildNodeType::TOTAL);
                            leaf.mBBoxMin = leafOrigin; // To be further updated after the leaf-level operation is complete
                            leaf.mFlags = (uint64_t)GridFlags::HasBBox;
                            ++prefixSum;
                        }
                    }

                    lower.mBBox  = CoordBBox();
                    lower.mFlags = (uint64_t)GridFlags::HasBBox;
                }
            }
        });
}

/// @brief Union the upper/lower child masks of one source grid into the densified, pre-allocated
///        mask arrays of the speculative merged topology. Host counterpart to the CUDA
///        MergeInternalNodesFunctor; call once per source grid (serially) to accumulate the union.
///
/// @param srcGrid      (in)  Source grid being merged (read host-side; see note on accessibility).
/// @param mergedRoot   (in)  Speculative merged root (host copy), used to locate the output tile.
/// @param upperMasks_  (out) Array of Mask<5>, one per merged upper tile.
/// @param lowerMasks_  (out) Array of (Mask<4>[Mask<5>::SIZE]), one set per merged upper tile.
/// @param lowerCount   (in)  Number of lower nodes in the source grid (nodeCount[1]).
///
/// Iteration is over the source grid's lower nodes. Each lower's child mask is OR'd into the
/// output lower mask for its (tile, upper-slot) location; distinct source lowers map to distinct
/// output lower masks, so that OR needs no atomics. The output *upper* mask bit is set via
/// setOnAtomic because sibling lowers under one upper share a mask word. Across the two source
/// grids the two calls run serially (forEach is blocking), so the union accumulates correctly.
///
/// @note srcGrid is dereferenced on the host. This assumes the grid's storage is host-accessible
///       (UnifiedBuffer / managed memory) and that upstream device work on the same stream has
///       been drained before the call.
template<typename BuildT>
inline void MergeInternalNodes(
    const NanoGrid<BuildT> *srcGrid,
    const NanoRoot<BuildT> *mergedRoot,
    void *upperMasks_,
    void *lowerMasks_,
    std::size_t lowerCount)
{
    using UpperMaskArrayT = Mask<5>*;
    using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    util::forEach(0, lowerCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        for (auto lowerID = r.begin(); lowerID != r.end(); ++lowerID) {
            const auto& lower = srcTree.template getFirstNode<1>()[lowerID];
            const auto& valueMask = lower.childMask();

            auto mergedTile = mergedRoot->probeTile(lower.origin());
            uint64_t tileIndex =
                util::PtrDiff(mergedTile, mergedRoot->tile(0))
                / sizeof(typename NanoRoot<BuildT>::Tile);
            auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(lower.origin());

            auto& outputUpperMask = upperMasks[tileIndex];
            auto& outputLowerMask = lowerMasks[tileIndex][upperChildIndex];
            for (uint32_t w = 0; w < Mask<4>::WORD_COUNT; ++w)
                outputLowerMask.words()[w] |= valueMask.words()[w];
            outputUpperMask.setOnAtomic(upperChildIndex);
        }
    });
}

}// namespace nanovdb::util::morphology

#endif // NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED
