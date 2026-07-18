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
#include <nanovdb/util/MorphologyHelpers.h>

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

/// @brief Union the leaf value masks of one source grid into the corresponding leaves of the
///        output grid. Host counterpart to the CUDA MergeLeafNodesFunctor; call once per source
///        grid (serially) to accumulate the union.
///
/// @param srcGrid    (in)  Source grid being merged (read host-side; see note on accessibility).
/// @param dstGrid    (in/out) Output grid whose leaf value masks receive the union.
/// @param leafCount  (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Iteration is flat over the source grid's leaf array (cf. PruneLeafMasksFunctor), rather than
/// the CUDA per-lower/childMask traversal. Each source leaf maps by origin to a distinct output
/// leaf (probeLeaf), so the value-mask OR needs no atomics; across the two source grids the two
/// calls run serially (forEach is blocking) so the union accumulates correctly.
///
/// @note srcGrid is dereferenced on the host (see MergeInternalNodes' note). Every source leaf
///       origin is present in the merged topology by construction, so probeLeaf is non-null.
template<typename BuildT>
inline void MergeLeafNodes(
    const NanoGrid<BuildT> *srcGrid,
    NanoGrid<BuildT> *dstGrid,
    std::size_t leafCount)
{
    util::forEach(0, leafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        const auto& dstTree = dstGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            auto dstLeafPtr = dstTree.root().probeLeaf(srcLeaf.origin());
            auto& dstMask = const_cast<Mask<3>&>(dstLeafPtr->valueMask());
            for (uint32_t w = 0; w < Mask<3>::WORD_COUNT; ++w)
                dstMask.words()[w] |= srcLeaf.valueMask().words()[w];
        }
    });
}

/// @brief Speculatively dilate the upper/lower child masks of one source grid into the densified,
///        pre-allocated mask arrays of the dilated topology. Host counterpart to the CUDA
///        DilateInternalNodesFunctor; templated on the nearest-neighbor stencil (NN_FACE,
///        NN_FACE_EDGE, NN_FACE_EDGE_VERTEX).
///
/// @param srcGrid      (in)  Source grid being dilated (read host-side; see MergeInternalNodes' note).
/// @param dilatedRoot  (in)  Speculative dilated root (host copy), used to locate output tiles.
/// @param upperMasks_  (out) Array of Mask<5>, one per dilated upper tile.
/// @param lowerMasks_  (out) Array of (Mask<4>[Mask<5>::SIZE]), one set per dilated upper tile.
/// @param lowerCount   (in)  Number of lower nodes in the source grid (nodeCount[1]).
///
/// For each source lower node: (1) build the 27 per-direction "offset masks" over its 4096 leaf
/// slots from each leaf's neighborMaskStencil (the host replaces the CUDA WarpReduce packing with a
/// direct per-slot setOn); (2) turn offset masks into the 27 neighbor-lower-node masks via MaskShift
/// (the CUDA 4-warp split of these calls collapses to one sequential block, kept verbatim); (3)
/// scatter the neighbor masks into the dilated tree's upper/lower masks. Distinct source lowers can
/// dilate into a shared neighbor lower, so the scatter uses setOnAtomic/atomicOr (host-callable).
template<typename BuildT, tools::morphology::NearestNeighbors nnType>
inline void DilateInternalNodes(
    const NanoGrid<BuildT> *srcGrid,
    const NanoRoot<BuildT> *dilatedRoot,
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

            // Per-direction offset masks and per-neighbor-node masks (the CUDA shared-memory
            // sOffsetMasks/sNeighborMasks; default-constructed Mask<4> is zero-filled).
            Mask<4> offsetMasks[3][3][3];
            Mask<4> neighborMasks[3][3][3];

            // For each active leaf slot, OR its 27-bit neighbor stencil into the offset masks.
            // bit = (di+1)*9 + (dj+1)*3 + (dk+1) (matches the CUDA flat sOffsetMasks[0][0][bit]).
            for (uint32_t jj = 0; jj < Mask<4>::SIZE; ++jj) {
                if (lower.childMask().isOn(jj)) {
                    const auto& leaf = *lower.data()->getChild(jj);
                    uint32_t neighborMask = neighborMaskStencil<nnType>(leaf.valueMask());
                    for (int bit = 0; bit < 27; ++bit)
                        if (neighborMask & (1u << bit))
                            offsetMasks[bit/9][(bit/3)%3][bit%3].setOn(jj);
                }
            }

            // Compute neighbor masks from offset masks (verbatim from the CUDA functor; the four
            // per-warp blocks are concatenated into one sequence on the host).
            // Contribution to mask of own lower node
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,1,1)
            MaskShift<  1,  1,  1>( offsetMasks[0][0][0], neighborMasks[1][1][1] );
            MaskShift<  1,  1,  0>( offsetMasks[0][0][1], neighborMasks[1][1][1] );
            MaskShift<  1,  1, -1>( offsetMasks[0][0][2], neighborMasks[1][1][1] );
            MaskShift<  1,  0,  1>( offsetMasks[0][1][0], neighborMasks[1][1][1] );
            MaskShift<  1,  0,  0>( offsetMasks[0][1][1], neighborMasks[1][1][1] );
            MaskShift<  1,  0, -1>( offsetMasks[0][1][2], neighborMasks[1][1][1] );
            MaskShift<  1, -1,  1>( offsetMasks[0][2][0], neighborMasks[1][1][1] );
            MaskShift<  1, -1,  0>( offsetMasks[0][2][1], neighborMasks[1][1][1] );
            MaskShift<  1, -1, -1>( offsetMasks[0][2][2], neighborMasks[1][1][1] );
            MaskShift<  0,  1,  1>( offsetMasks[1][0][0], neighborMasks[1][1][1] );
            MaskShift<  0,  1,  0>( offsetMasks[1][0][1], neighborMasks[1][1][1] );
            MaskShift<  0,  1, -1>( offsetMasks[1][0][2], neighborMasks[1][1][1] );
            MaskShift<  0,  0,  1>( offsetMasks[1][1][0], neighborMasks[1][1][1] );
            MaskShift<  0,  0,  0>( offsetMasks[1][1][1], neighborMasks[1][1][1] );
            MaskShift<  0,  0, -1>( offsetMasks[1][1][2], neighborMasks[1][1][1] );
            MaskShift<  0, -1,  1>( offsetMasks[1][2][0], neighborMasks[1][1][1] );
            MaskShift<  0, -1,  0>( offsetMasks[1][2][1], neighborMasks[1][1][1] );
            MaskShift<  0, -1, -1>( offsetMasks[1][2][2], neighborMasks[1][1][1] );
            MaskShift< -1,  1,  1>( offsetMasks[2][0][0], neighborMasks[1][1][1] );
            MaskShift< -1,  1,  0>( offsetMasks[2][0][1], neighborMasks[1][1][1] );
            MaskShift< -1,  1, -1>( offsetMasks[2][0][2], neighborMasks[1][1][1] );
            MaskShift< -1,  0,  1>( offsetMasks[2][1][0], neighborMasks[1][1][1] );
            MaskShift< -1,  0,  0>( offsetMasks[2][1][1], neighborMasks[1][1][1] );
            MaskShift< -1,  0, -1>( offsetMasks[2][1][2], neighborMasks[1][1][1] );
            MaskShift< -1, -1,  1>( offsetMasks[2][2][0], neighborMasks[1][1][1] );
            MaskShift< -1, -1,  0>( offsetMasks[2][2][1], neighborMasks[1][1][1] );
            MaskShift< -1, -1, -1>( offsetMasks[2][2][2], neighborMasks[1][1][1] );
            // Contribution to mask of lower node at offset (-1,-1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,-15,-15)
            MaskShift<-15,-15,-15>( offsetMasks[0][0][0], neighborMasks[0][0][0] );
            // Contribution to mask of lower node at offset (-1,-1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,-15,17)
            MaskShift<-15,-15, 15>( offsetMasks[0][0][2], neighborMasks[0][0][2] );
            // Contribution to mask of lower node at offset (-1,1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,17,-15)
            MaskShift<-15, 15,-15>( offsetMasks[0][2][0], neighborMasks[0][2][0] );
            // Contribution to mask of lower node at offset (-1,1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,17,17)
            MaskShift<-15, 15, 15>( offsetMasks[0][2][2], neighborMasks[0][2][2] );
            // Contribution to mask of lower node at offset (1,-1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,-15,-15)
            MaskShift< 15,-15,-15>( offsetMasks[2][0][0], neighborMasks[2][0][0] );

            // Contribution to mask of lower node at offset (0,0,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,1,-15)
            MaskShift<  1,  1,-15>( offsetMasks[0][0][0], neighborMasks[1][1][0] );
            MaskShift<  1,  0,-15>( offsetMasks[0][1][0], neighborMasks[1][1][0] );
            MaskShift<  1, -1,-15>( offsetMasks[0][2][0], neighborMasks[1][1][0] );
            MaskShift<  0,  1,-15>( offsetMasks[1][0][0], neighborMasks[1][1][0] );
            MaskShift<  0,  0,-15>( offsetMasks[1][1][0], neighborMasks[1][1][0] );
            MaskShift<  0, -1,-15>( offsetMasks[1][2][0], neighborMasks[1][1][0] );
            MaskShift< -1,  1,-15>( offsetMasks[2][0][0], neighborMasks[1][1][0] );
            MaskShift< -1,  0,-15>( offsetMasks[2][1][0], neighborMasks[1][1][0] );
            MaskShift< -1, -1,-15>( offsetMasks[2][2][0], neighborMasks[1][1][0] );
            // Contribution to mask of lower node at offset (0,0,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,1,17)
            MaskShift<  1,  1, 15>( offsetMasks[0][0][2], neighborMasks[1][1][2] );
            MaskShift<  1,  0, 15>( offsetMasks[0][1][2], neighborMasks[1][1][2] );
            MaskShift<  1, -1, 15>( offsetMasks[0][2][2], neighborMasks[1][1][2] );
            MaskShift<  0,  1, 15>( offsetMasks[1][0][2], neighborMasks[1][1][2] );
            MaskShift<  0,  0, 15>( offsetMasks[1][1][2], neighborMasks[1][1][2] );
            MaskShift<  0, -1, 15>( offsetMasks[1][2][2], neighborMasks[1][1][2] );
            MaskShift< -1,  1, 15>( offsetMasks[2][0][2], neighborMasks[1][1][2] );
            MaskShift< -1,  0, 15>( offsetMasks[2][1][2], neighborMasks[1][1][2] );
            MaskShift< -1, -1, 15>( offsetMasks[2][2][2], neighborMasks[1][1][2] );
            // Contribution to mask of lower node at offset (-1,-1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,-15,1)
            MaskShift<-15,-15,  1>( offsetMasks[0][0][0], neighborMasks[0][0][1] );
            MaskShift<-15,-15,  0>( offsetMasks[0][0][1], neighborMasks[0][0][1] );
            MaskShift<-15,-15, -1>( offsetMasks[0][0][2], neighborMasks[0][0][1] );
            // Contribution to mask of lower node at offset (-1,1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,17,1)
            MaskShift<-15, 15,  1>( offsetMasks[0][2][0], neighborMasks[0][2][1] );
            MaskShift<-15, 15,  0>( offsetMasks[0][2][1], neighborMasks[0][2][1] );
            MaskShift<-15, 15, -1>( offsetMasks[0][2][2], neighborMasks[0][2][1] );
            // Contribution to mask of lower node at offset (1,-1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,-15,1)
            MaskShift< 15,-15,  1>( offsetMasks[2][0][0], neighborMasks[2][0][1] );
            MaskShift< 15,-15,  0>( offsetMasks[2][0][1], neighborMasks[2][0][1] );
            MaskShift< 15,-15, -1>( offsetMasks[2][0][2], neighborMasks[2][0][1] );
            // Contribution to mask of lower node at offset (1,1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,17,1)
            MaskShift< 15, 15,  1>( offsetMasks[2][2][0], neighborMasks[2][2][1] );
            MaskShift< 15, 15,  0>( offsetMasks[2][2][1], neighborMasks[2][2][1] );
            MaskShift< 15, 15, -1>( offsetMasks[2][2][2], neighborMasks[2][2][1] );
            // Contribution to mask of lower node at offset (1,-1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,-15,17)
            MaskShift< 15,-15, 15>( offsetMasks[2][0][2], neighborMasks[2][0][2] );

            // Contribution to mask of lower node at offset (0,-1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,-15,1)
            MaskShift<  1,-15,  1>( offsetMasks[0][0][0], neighborMasks[1][0][1] );
            MaskShift<  1,-15,  0>( offsetMasks[0][0][1], neighborMasks[1][0][1] );
            MaskShift<  1,-15, -1>( offsetMasks[0][0][2], neighborMasks[1][0][1] );
            MaskShift<  0,-15,  1>( offsetMasks[1][0][0], neighborMasks[1][0][1] );
            MaskShift<  0,-15,  0>( offsetMasks[1][0][1], neighborMasks[1][0][1] );
            MaskShift<  0,-15, -1>( offsetMasks[1][0][2], neighborMasks[1][0][1] );
            MaskShift< -1,-15,  1>( offsetMasks[2][0][0], neighborMasks[1][0][1] );
            MaskShift< -1,-15,  0>( offsetMasks[2][0][1], neighborMasks[1][0][1] );
            MaskShift< -1,-15, -1>( offsetMasks[2][0][2], neighborMasks[1][0][1] );
            // Contribution to mask of lower node at offset (0,1,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,17,1)
            MaskShift<  1, 15,  1>( offsetMasks[0][2][0], neighborMasks[1][2][1] );
            MaskShift<  1, 15,  0>( offsetMasks[0][2][1], neighborMasks[1][2][1] );
            MaskShift<  1, 15, -1>( offsetMasks[0][2][2], neighborMasks[1][2][1] );
            MaskShift<  0, 15,  1>( offsetMasks[1][2][0], neighborMasks[1][2][1] );
            MaskShift<  0, 15,  0>( offsetMasks[1][2][1], neighborMasks[1][2][1] );
            MaskShift<  0, 15, -1>( offsetMasks[1][2][2], neighborMasks[1][2][1] );
            MaskShift< -1, 15,  1>( offsetMasks[2][2][0], neighborMasks[1][2][1] );
            MaskShift< -1, 15,  0>( offsetMasks[2][2][1], neighborMasks[1][2][1] );
            MaskShift< -1, 15, -1>( offsetMasks[2][2][2], neighborMasks[1][2][1] );
            // Contribution to mask of lower node at offset (-1,0,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,1,-15)
            MaskShift<-15,  1,-15>( offsetMasks[0][0][0], neighborMasks[0][1][0] );
            MaskShift<-15,  0,-15>( offsetMasks[0][1][0], neighborMasks[0][1][0] );
            MaskShift<-15, -1,-15>( offsetMasks[0][2][0], neighborMasks[0][1][0] );
            // Contribution to mask of lower node at offset (-1,0,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,1,17)
            MaskShift<-15,  1, 15>( offsetMasks[0][0][2], neighborMasks[0][1][2] );
            MaskShift<-15,  0, 15>( offsetMasks[0][1][2], neighborMasks[0][1][2] );
            MaskShift<-15, -1, 15>( offsetMasks[0][2][2], neighborMasks[0][1][2] );
            // Contribution to mask of lower node at offset (1,0,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,1,-15)
            MaskShift< 15,  1,-15>( offsetMasks[2][0][0], neighborMasks[2][1][0] );
            MaskShift< 15,  0,-15>( offsetMasks[2][1][0], neighborMasks[2][1][0] );
            MaskShift< 15, -1,-15>( offsetMasks[2][2][0], neighborMasks[2][1][0] );
            // Contribution to mask of lower node at offset (1,0,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,1,17)
            MaskShift< 15,  1, 15>( offsetMasks[2][0][2], neighborMasks[2][1][2] );
            MaskShift< 15,  0, 15>( offsetMasks[2][1][2], neighborMasks[2][1][2] );
            MaskShift< 15, -1, 15>( offsetMasks[2][2][2], neighborMasks[2][1][2] );
            // Contribution to mask of lower node at offset (1,1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,17,-15)
            MaskShift< 15, 15,-15>( offsetMasks[2][2][0], neighborMasks[2][2][0] );

            // Contribution to mask of lower node at offset (-1,0,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (-15,1,1)
            MaskShift<-15,  1,  1>( offsetMasks[0][0][0], neighborMasks[0][1][1] );
            MaskShift<-15,  1,  0>( offsetMasks[0][0][1], neighborMasks[0][1][1] );
            MaskShift<-15,  1, -1>( offsetMasks[0][0][2], neighborMasks[0][1][1] );
            MaskShift<-15,  0,  1>( offsetMasks[0][1][0], neighborMasks[0][1][1] );
            MaskShift<-15,  0,  0>( offsetMasks[0][1][1], neighborMasks[0][1][1] );
            MaskShift<-15,  0, -1>( offsetMasks[0][1][2], neighborMasks[0][1][1] );
            MaskShift<-15, -1,  1>( offsetMasks[0][2][0], neighborMasks[0][1][1] );
            MaskShift<-15, -1,  0>( offsetMasks[0][2][1], neighborMasks[0][1][1] );
            MaskShift<-15, -1, -1>( offsetMasks[0][2][2], neighborMasks[0][1][1] );
            // Contribution to mask of lower node at offset (1,0,0)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,1,1)
            MaskShift< 15,  1,  1>( offsetMasks[2][0][0], neighborMasks[2][1][1] );
            MaskShift< 15,  1,  0>( offsetMasks[2][0][1], neighborMasks[2][1][1] );
            MaskShift< 15,  1, -1>( offsetMasks[2][0][2], neighborMasks[2][1][1] );
            MaskShift< 15,  0,  1>( offsetMasks[2][1][0], neighborMasks[2][1][1] );
            MaskShift< 15,  0,  0>( offsetMasks[2][1][1], neighborMasks[2][1][1] );
            MaskShift< 15,  0, -1>( offsetMasks[2][1][2], neighborMasks[2][1][1] );
            MaskShift< 15, -1,  1>( offsetMasks[2][2][0], neighborMasks[2][1][1] );
            MaskShift< 15, -1,  0>( offsetMasks[2][2][1], neighborMasks[2][1][1] );
            MaskShift< 15, -1, -1>( offsetMasks[2][2][2], neighborMasks[2][1][1] );
            // Contribution to mask of lower node at offset (0,-1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,-15,-15)
            MaskShift<  1,-15,-15>( offsetMasks[0][0][0], neighborMasks[1][0][0] );
            MaskShift<  0,-15,-15>( offsetMasks[1][0][0], neighborMasks[1][0][0] );
            MaskShift< -1,-15,-15>( offsetMasks[2][0][0], neighborMasks[1][0][0] );
            // Contribution to mask of lower node at offset (0,-1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,-15,17)
            MaskShift<  1,-15, 15>( offsetMasks[0][0][2], neighborMasks[1][0][2] );
            MaskShift<  0,-15, 15>( offsetMasks[1][0][2], neighborMasks[1][0][2] );
            MaskShift< -1,-15, 15>( offsetMasks[2][0][2], neighborMasks[1][0][2] );
            // Contribution to mask of lower node at offset (0,1,-1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,17,-15)
            MaskShift<  1, 15,-15>( offsetMasks[0][2][0], neighborMasks[1][2][0] );
            MaskShift<  0, 15,-15>( offsetMasks[1][2][0], neighborMasks[1][2][0] );
            MaskShift< -1, 15,-15>( offsetMasks[2][2][0], neighborMasks[1][2][0] );
            // Contribution to mask of lower node at offset (0,1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (1,17,17)
            MaskShift<  1, 15, 15>( offsetMasks[0][2][2], neighborMasks[1][2][2] );
            MaskShift<  0, 15, 15>( offsetMasks[1][2][2], neighborMasks[1][2][2] );
            MaskShift< -1, 15, 15>( offsetMasks[2][2][2], neighborMasks[1][2][2] );
            // Contribution to mask of lower node at offset (1,1,1)
            // Arguments to MaskShift plus indices to offsetMasks add up to (17,17,17)
            MaskShift< 15, 15, 15>( offsetMasks[2][2][2], neighborMasks[2][2][2] );

            // Compose contributions to the lower-node masks of the dilated tree. Distinct source
            // lowers may target the same neighbor lower, hence setOnAtomic/atomicOr.
            for (int di = -1; di <= 1; ++di)
            for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk) {
                const auto& neighborMask = neighborMasks[di+1][dj+1][dk+1];
                bool any = false;
                for (uint32_t w = 0; w < Mask<4>::WORD_COUNT; ++w) if (neighborMask.words()[w]) { any = true; break; }
                if (!any) continue;
                auto neighborOrigin = lower.origin().offsetBy(di*128, dj*128, dk*128);
                auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(neighborOrigin);
                auto dilatedTile = dilatedRoot->probeTile(neighborOrigin);
                uint64_t tileChildIndex =
                    util::PtrDiff(dilatedTile, dilatedRoot->tile(0))
                    / sizeof(typename NanoRoot<BuildT>::Tile);
                auto& outputUpperMask = upperMasks[tileChildIndex];
                outputUpperMask.setOnAtomic(upperChildIndex);
                auto& outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                for (uint32_t w = 0; w < Mask<4>::WORD_COUNT; ++w) {
                    const uint64_t computedWord = neighborMask.words()[w];
                    if (computedWord)
                        util::atomicOr(const_cast<uint64_t*>(&outputLowerMask.words()[w]), computedWord);
                }
            }
        }
    });
}

/// @brief Dilate the leaf value masks of the source grid into the (already topologically dilated)
///        output grid. Host counterpart to the CUDA DilateLeafNodesFunctor; templated on the
///        nearest-neighbor stencil (NN_FACE or NN_FACE_EDGE_VERTEX; NN_FACE_EDGE is unsupported at
///        leaf level, matching CUDA, and is guarded out by the operator).
///
/// @param srcGrid    (in)  Source grid being dilated (read host-side; see MergeInternalNodes' note).
/// @param dstGrid    (in/out) Output grid whose leaf value masks receive the dilated result.
/// @param leafCount  (in)  Number of leaf nodes in the output grid (nodeCount[0]).
///
/// Iteration is flat over the output leaves. Each output leaf is computed independently from the
/// source grid's corresponding leaf and its neighbors (probeLeaf) with register/word bit-ops, so
/// the writes are to distinct masks -- no atomics. The per-leaf bodies are copied verbatim from the
/// CUDA functor (which is already thread-centric: no warp/CTA cooperation).
template<typename BuildT, tools::morphology::NearestNeighbors nnType>
inline void DilateLeafNodes(
    const NanoGrid<BuildT> *srcGrid,
    NanoGrid<BuildT> *dstGrid,
    std::size_t leafCount)
{
    util::forEach(0, leafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        auto& dstTree = dstGrid->tree();
        for (auto leafID = r.begin(); leafID != r.end(); ++leafID) {
            auto& dstLeaf = dstTree.template getFirstNode<0>()[leafID];
            const auto leafOrigin = dstLeaf.origin();

            if constexpr (nnType == tools::morphology::NN_FACE) {
                auto originalLeafPtr  = srcTree.root().probeLeaf(leafOrigin);
                uint64_t originalWords[8] = {}, dilatedWords[8]; // Keep these in registers
                if (originalLeafPtr)
                    for (int i = 0; i < 8; i++)
                        originalWords[i] = originalLeafPtr->valueMask().words()[i];

                for (int i = 0; i < 8; i++) {
                    dilatedWords[i] = originalWords[i];
                    // Activate voxel if the neighbor at stencil offset ( 1, 0, 0) is active
                    if (i < 7) dilatedWords[i] |= originalWords[i+1];
                    // Activate voxel if the neighbor at stencil offset (-1, 0, 0) is active
                    if (i > 0) dilatedWords[i] |= originalWords[i-1];

                    // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                    dilatedWords[i] |= (originalWords[i] & 0xffffffffffffff00UL) >> 8;
                    // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                    dilatedWords[i] |= (originalWords[i] & 0x00ffffffffffffffUL) << 8;

                    // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                    dilatedWords[i] |= (originalWords[i] & 0xfefefefefefefefeUL) >> 1;
                    // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                    dilatedWords[i] |= (originalWords[i] & 0x7f7f7f7f7f7f7f7fUL) << 1;
                }

                auto leafPlusXPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 8, 0, 0));
                auto leafMinusXPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy(-8, 0, 0));
                auto leafPlusYPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 8, 0));
                auto leafMinusYPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0,-8, 0));
                auto leafPlusZPtr  = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 0, 8));
                auto leafMinusZPtr = srcTree.root().probeLeaf(leafOrigin.offsetBy( 0, 0,-8));

                // Activate voxel if the neighbor at stencil offset ( 1, 0, 0) is active
                if (leafPlusXPtr) dilatedWords[7] |= leafPlusXPtr->valueMask().words()[0];
                // Activate voxel if the neighbor at stencil offset (-1, 0, 0) is active
                if (leafMinusXPtr) dilatedWords[0] |= leafMinusXPtr->valueMask().words()[7];

                // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                if (leafPlusYPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafPlusYPtr->valueMask().words()[i] & 0x00000000000000ffUL) << 56;
                // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                if (leafMinusYPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafMinusYPtr->valueMask().words()[i] & 0xff00000000000000UL) >> 56;

                // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                if (leafPlusZPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafPlusZPtr->valueMask().words()[i] & 0x0101010101010101UL) << 7;
                // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                if (leafMinusZPtr)
                    for (int i = 0; i < 8; i++)
                        dilatedWords[i] |= (leafMinusZPtr->valueMask().words()[i] & 0x8080808080808080UL) >> 7;

                auto& dilatedMask = const_cast<Mask<3>&>(dstLeaf.valueMask());
                for (int i = 0; i < 8; i++)
                    dilatedMask.words()[i] = dilatedWords[i];
            }

            if constexpr (nnType == tools::morphology::NN_FACE_EDGE_VERTEX) {
                // [x-voxel offset][y-block offset][z-block offset], stored with a +1 bias so the
                // logical ranges [-1,8]x[-1,1]x[-1,1] map to valid [0,10)x[0,3)x[0,3) indices.
                // (The CUDA functor uses a reinterpret_cast + negative indexing; on the host that
                // is out-of-subobject-bounds UB that the optimizer mishandles, so we bias instead.)
                uint64_t originalWordsShifted[10][3][3] = {};
                auto originalWords = [&](int i, int j, int k) -> uint64_t& {
                    return originalWordsShifted[i+1][j+1][k+1]; };

                for (int dBi = -1; dBi <= 1; dBi++)
                for (int dBj = -1; dBj <= 1; dBj++)
                for (int dBk = -1; dBk <= 1; dBk++) {
                    auto neighborOrigin = leafOrigin.offsetBy( dBi*8, dBj*8, dBk*8);
                    if (auto neighborLeafPtr = srcTree.root().probeLeaf(neighborOrigin)) {
                        auto neighborWords = neighborLeafPtr->valueMask().words();
                        if (dBi == -1)
                            originalWords(-1,dBj,dBk) = neighborWords[7];
                        else if (dBi == 1)
                            originalWords(8,dBj,dBk) = neighborWords[0];
                        else
                            for (int i = 0; i < 8; i++)
                                originalWords(i,dBj,dBk) = neighborWords[i]; } }
                // Dilate along z-axis
                for (int i = -1; i <= 8; i++)
                for (int dBj = -1; dBj <= 1; dBj++) {
                    uint64_t dilatedWord = originalWords(i,dBj,0);
                    // Activate voxel if the neighbor at stencil offset ( 0, 0, 1) is active
                    dilatedWord |= (originalWords(i,dBj, 0) & 0xfefefefefefefefeUL) >> 1;
                    dilatedWord |= (originalWords(i,dBj, 1) & 0x0101010101010101UL) << 7;
                    // Activate voxel if the neighbor at stencil offset ( 0, 0,-1) is active
                    dilatedWord |= (originalWords(i,dBj, 0) & 0x7f7f7f7f7f7f7f7fUL) << 1;
                    dilatedWord |= (originalWords(i,dBj,-1) & 0x8080808080808080UL) >> 7;
                    // Replace original with dilation result
                    originalWords(i,dBj,0) = dilatedWord; }

                // Dilate along y-axis
                for (int i = -1; i <= 8; i++) {
                    uint64_t dilatedWord = originalWords(i,0,0);
                    // Activate voxel if the neighbor at stencil offset ( 0, 1, 0) is active
                    dilatedWord |= (originalWords(i, 0,0) & 0xffffffffffffff00UL) >> 8;
                    dilatedWord |= (originalWords(i, 1,0) & 0x00000000000000ffUL) << 56;
                    // Activate voxel if the neighbor at stencil offset ( 0,-1, 0) is active
                    dilatedWord |= (originalWords(i, 0,0) & 0x00ffffffffffffffUL) << 8;
                    dilatedWord |= (originalWords(i,-1,0) & 0xff00000000000000UL) >> 56;
                    // Replace original with dilation result
                    originalWords(i,0,0) = dilatedWord; }

                // Dilate along x-axis
                auto dilatedWords = const_cast<Mask<3>&>(dstLeaf.valueMask()).words();
                for (int i = 0; i <= 7; i++)
                    dilatedWords[i] = originalWords(i-1,0,0) | originalWords(i,0,0) | originalWords(i+1,0,0);
            }
        }
    });
}

/// @brief Set the upper/lower child-mask bits of the pruned topology for every source leaf that
///        retains at least one voxel under the leaf-mask sidecar. Host counterpart to the CUDA
///        PruneInternalNodesFunctor.
///
/// @param srcGrid      (in)  Source grid being pruned (read host-side; see MergeInternalNodes' note).
/// @param prunedRoot   (in)  Speculative pruned root (host copy), used to locate output tiles.
/// @param srcLeafMask  (in)  Per-source-leaf bitmask of voxels to retain (sidecar, host-accessible).
/// @param upperMasks_  (out) Array of Mask<5>, one per pruned upper tile.
/// @param lowerMasks_  (out) Array of (Mask<4>[Mask<5>::SIZE]), one set per pruned upper tile.
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// A source leaf is retained iff its value mask intersects the sidecar mask. Sibling leaves under
/// a shared lower/upper race into the same mask word, so the output bits are set via setOnAtomic.
template<typename BuildT>
inline void PruneInternalNodes(
    const NanoGrid<BuildT> *srcGrid,
    const NanoRoot<BuildT> *prunedRoot,
    const Mask<3> *srcLeafMask,
    void *upperMasks_,
    void *lowerMasks_,
    std::size_t srcLeafCount)
{
    using UpperMaskArrayT = Mask<5>*;
    using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            const auto& leafMask = srcLeafMask[srcLeafID];
            bool retainLeaf = false;
            for (uint32_t w = 0; w < Mask<3>::WORD_COUNT; w++)
                if (srcLeaf.valueMask().words()[w] & leafMask.words()[w])
                    retainLeaf = true;
            if (retainLeaf) {
                auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(srcLeaf.origin());
                auto lowerChildIndex = NanoLower<BuildT>::CoordToOffset(srcLeaf.origin());
                auto prunedTile = prunedRoot->probeTile(srcLeaf.origin());
                uint64_t tileChildIndex =
                    util::PtrDiff(prunedTile, prunedRoot->tile(0))
                    / sizeof(typename NanoRoot<BuildT>::Tile);
                auto& outputUpperMask = upperMasks[tileChildIndex];
                outputUpperMask.setOnAtomic(upperChildIndex);
                auto& outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                outputLowerMask.setOnAtomic(lowerChildIndex);
            }
        }
    });
}

/// @brief Set each retained output leaf's value mask to its source value mask intersected with the
///        leaf-mask sidecar. Host counterpart to the CUDA PruneLeafMasksFunctor.
///
/// @param srcGrid      (in)  Source grid being pruned (read host-side).
/// @param dstGrid      (in/out) Output (pruned) grid whose leaf value masks are written.
/// @param srcLeafMask  (in)  Per-source-leaf bitmask of voxels to retain (sidecar).
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Iteration is flat over source leaves; each maps by origin to a distinct output leaf (probeLeaf,
/// non-null only for retained leaves), so the writes are to distinct masks -- no atomics.
template<typename BuildT>
inline void PruneLeafMasks(
    const NanoGrid<BuildT> *srcGrid,
    NanoGrid<BuildT> *dstGrid,
    const Mask<3> *srcLeafMask,
    std::size_t srcLeafCount)
{
    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        auto& dstTree = dstGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            auto leafPtr = dstTree.root().probeLeaf(srcLeaf.origin());
            if (leafPtr) {
                const_cast<Mask<3>&>(leafPtr->valueMask())  = srcLeaf.valueMask();
                const_cast<Mask<3>&>(leafPtr->valueMask()) &= srcLeafMask[srcLeafID];
            }
        }
    });
}

/// @brief Compute the masks of upper and (densified) lower internal nodes of the coarsened result.
///        Host counterpart to the CUDA CoarsenInternalNodesFunctor.
///
/// @param srcGrid        (in)  Source grid being coarsened (read host-side; see MergeInternalNodes' note).
/// @param coarsenedRoot  (in)  Speculative coarsened root (host copy), used to locate output tiles.
/// @param upperMasks_    (out) Array of Mask<5>, one per coarsened upper tile.
/// @param lowerMasks_    (out) Array of (Mask<4>[Mask<5>::SIZE]), one set per coarsened upper tile.
/// @param srcLeafCount   (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Every non-empty source leaf maps (via coarsenCoord) into the coarsened topology. Multiple source
/// leaves can coarsen into a shared lower/upper, so the output bits are set via setOnAtomic.
template<typename BuildT>
inline void CoarsenInternalNodes(
    const NanoGrid<BuildT> *srcGrid,
    const NanoRoot<BuildT> *coarsenedRoot,
    void *upperMasks_,
    void *lowerMasks_,
    std::size_t srcLeafCount)
{
    using UpperMaskArrayT = Mask<5>*;
    using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            if (!srcLeaf.valueMask().isOff()) { // Gratuitous check; leaf should have at least one active voxel
                auto coarsenedOrigin = coarsenCoord(srcLeaf.origin()); // it's ok if this is not a multiple of 8
                auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(coarsenedOrigin);
                auto lowerChildIndex = NanoLower<BuildT>::CoordToOffset(coarsenedOrigin);
                auto coarsenedTile = coarsenedRoot->probeTile(coarsenedOrigin);
                uint64_t tileChildIndex =
                    util::PtrDiff(coarsenedTile, coarsenedRoot->tile(0))
                    / sizeof(typename NanoRoot<BuildT>::Tile);
                auto& outputUpperMask = upperMasks[tileChildIndex];
                outputUpperMask.setOnAtomic(upperChildIndex);
                auto& outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                outputLowerMask.setOnAtomic(lowerChildIndex);
            }
        }
    });
}

/// @brief In-place 2x downsampling of a leaf value mask: out(x,y,z) = OR of the eight voxels in the
///        2x2x2 block at (2x,2y,2z). The result occupies the low (0..3) range along each axis.
///        Verbatim host copy of the CUDA CoarsenLeafMasksFunctor::coarsenMask word arithmetic.
inline void coarsenMask(Mask<3>& mask)
{
    // Coarsen along x-axis
    mask.words()[0] |= mask.words()[1];
    mask.words()[1] = mask.words()[2] | mask.words()[3];
    mask.words()[2] = mask.words()[4] | mask.words()[5];
    mask.words()[3] = mask.words()[6] | mask.words()[7];
    mask.words()[4] = mask.words()[5] = mask.words()[6] = mask.words()[7] = 0UL;

    for (int w = 0; w < 4; w++) {
        auto& word = mask.words()[w];
        // Coarsen along y-axis
        word |= (word >> 8);
        word &= 0x00ff00ff00ff00ffUL;
        word |= (word >> 8);
        word &= 0x0000ffff0000ffffUL;
        word |= (word >> 16);
        word &= 0x00000000ffffffffUL;
        // Coarsen along z-axis
        word |= (word >> 1);
        word &= 0x0000000055555555UL;
        word |= (word >> 1);
        word &= 0x0000000033333333UL;
        word |= (word >> 2);
        word &= 0x000000000f0f0f0fUL;
    }
}

/// @brief Coarsen each source leaf's active mask into its destination leaf in the coarsened grid.
///        Host counterpart to the CUDA CoarsenLeafMasksFunctor.
///
/// @param srcGrid      (in)  Source grid being coarsened (read host-side).
/// @param dstGrid      (in/out) Output (coarsened) grid whose leaf value masks are written.
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Iteration is flat over source leaves. Up to eight source leaves coarsen into one destination leaf
/// (each contributing one 4x4x4 octant), so the per-octant words are OR'd in via util::atomicOr.
template<typename BuildT>
inline void CoarsenLeafMasks(
    const NanoGrid<BuildT> *srcGrid,
    NanoGrid<BuildT> *dstGrid,
    std::size_t srcLeafCount)
{
    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        auto& dstTree = dstGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            const auto coarsenedOrigin = coarsenCoord(srcLeaf.origin());
            auto coarsenedMask = srcLeaf.valueMask();
            coarsenMask(coarsenedMask);
            int bi = (coarsenedOrigin[0] % 8 != 0);
            int bj = (coarsenedOrigin[1] % 8 != 0);
            int bk = (coarsenedOrigin[2] % 8 != 0);
            auto dstLeafPtr = dstTree.root().probeLeaf(coarsenedOrigin);
            auto& dstMask = const_cast<Mask<3>&>(dstLeafPtr->valueMask());
            for (int w = 0; w < 4; w++)
                util::atomicOr(&dstMask.words()[w+4*bi], coarsenedMask.words()[w] << (4*bk+32*bj));
        }
    });
}

/// @brief Compute the masks of upper and (densified) lower internal nodes of the refined result.
///        Host counterpart to the CUDA RefineInternalNodesFunctor.
///
/// @param srcGrid      (in)  Source grid being refined (read host-side; see MergeInternalNodes' note).
/// @param refinedRoot  (in)  Speculative refined root (host copy), used to locate output tiles.
/// @param upperMasks_  (out) Array of Mask<5>, one per refined upper tile.
/// @param lowerMasks_  (out) Array of (Mask<4>[Mask<5>::SIZE]), one set per refined upper tile.
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Each source leaf upsamples into up to eight destination octants; the present octants are detected
/// from the source value-mask words. Adjacent source leaves can refine into a shared lower/upper, so
/// the output bits are set via setOnAtomic.
template<typename BuildT>
inline void RefineInternalNodes(
    const NanoGrid<BuildT> *srcGrid,
    const NanoRoot<BuildT> *refinedRoot,
    void *upperMasks_,
    void *lowerMasks_,
    std::size_t srcLeafCount)
{
    using UpperMaskArrayT = Mask<5>*;
    using LowerMaskArrayT = Mask<4>(*)[Mask<5>::SIZE];
    auto upperMasks = static_cast<UpperMaskArrayT>(upperMasks_);
    auto lowerMasks = static_cast<LowerMaskArrayT>(lowerMasks_);

    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            uint64_t octantPresent[2][2][2] = {};
            const auto words = srcLeaf.valueMask().words();
            for (int w = 0; w < 8; w++) {
                octantPresent[w>>2][0][0] |= ( words[w] & 0x000000000f0f0f0fUL );
                octantPresent[w>>2][0][1] |= ( words[w] & 0x00000000f0f0f0f0UL );
                octantPresent[w>>2][1][0] |= ( words[w] & 0x0f0f0f0f00000000UL );
                octantPresent[w>>2][1][1] |= ( words[w] & 0xf0f0f0f000000000UL );
            }
            for (int di = 0; di < 2; di++)
            for (int dj = 0; dj < 2; dj++)
            for (int dk = 0; dk < 2; dk++) {
                if (octantPresent[di][dj][dk]) {
                    const auto refinedOrigin = refineCoord(srcLeaf.origin()+nanovdb::Coord(di*4,dj*4,dk*4));
                    auto upperChildIndex = NanoUpper<BuildT>::CoordToOffset(refinedOrigin);
                    auto lowerChildIndex = NanoLower<BuildT>::CoordToOffset(refinedOrigin);
                    auto refinedTile = refinedRoot->probeTile(refinedOrigin);
                    uint64_t tileChildIndex =
                        util::PtrDiff(refinedTile, refinedRoot->tile(0))
                        / sizeof(typename NanoRoot<BuildT>::Tile);
                    auto& outputUpperMask = upperMasks[tileChildIndex];
                    outputUpperMask.setOnAtomic(upperChildIndex);
                    auto& outputLowerMask = lowerMasks[tileChildIndex][upperChildIndex];
                    outputLowerMask.setOnAtomic(lowerChildIndex);
                }
            }
        }
    });
}

/// @brief In-place 2x upsampling of a leaf value mask: spreads the low (0..3) octant of each axis to
///        fill the full 8x8x8, duplicating each source voxel into its 2x2x2 refined block.
///        Verbatim host copy of the CUDA RefineLeafMasksFunctor::refineMask word arithmetic.
inline void refineMask(Mask<3>& mask)
{
    for (int w = 0; w < 4; w++) {
        auto& word = mask.words()[w];
        // Refine and duplicate along z-axis
        word &= 0x000000000f0f0f0fUL;
        word |= (word << 2);
        word &= 0x0000000033333333UL;
        word |= (word << 1);
        word &= 0x0000000055555555UL;
        word |= (word << 1);
        // Refine and duplicate along y-axis
        word |= (word << 16);
        word &= 0x0000ffff0000ffffUL;
        word |= (word << 8);
        word &= 0x00ff00ff00ff00ffUL;
        word |= (word << 8);
    }
    // Refine and duplicate along x-axis
    for (int w = 7; w > 0; w--)
        mask.words()[w] = mask.words()[w>>1];
}

/// @brief Refine each source leaf's active mask into its (up to eight) destination leaves in the
///        refined grid. Host counterpart to the CUDA RefineLeafMasksFunctor.
///
/// @param srcGrid      (in)  Source grid being refined (read host-side).
/// @param dstGrid      (in/out) Output (refined) grid whose leaf value masks are written.
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Iteration is flat over source leaves; each maps to up to eight distinct destination leaves (one per
/// octant of the upsampled 16x16x16 region), so the writes are to distinct masks -- no atomics.
template<typename BuildT>
inline void RefineLeafMasks(
    const NanoGrid<BuildT> *srcGrid,
    NanoGrid<BuildT> *dstGrid,
    std::size_t srcLeafCount)
{
    util::forEach(0, srcLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        auto& dstTree = dstGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            const auto refinedBaseOrigin = srcLeaf.origin()+srcLeaf.origin();
            for (int bi = 0; bi < 2; bi++)
            for (int bj = 0; bj < 2; bj++)
            for (int bk = 0; bk < 2; bk++) {
                auto dstLeafPtr = dstTree.root().probeLeaf(refinedBaseOrigin.offsetBy(8*bi,8*bj,8*bk));
                if (dstLeafPtr != nullptr) {
                    auto& refinedMask = const_cast<Mask<3>&>(dstLeafPtr->valueMask());
                    for (int w = 0; w < 4; w++)
                        refinedMask.words()[w] = srcLeaf.valueMask().words()[w+bi*4] >> (4*bk+32*bj);
                    refineMask(refinedMask);
                }
            }
        }
    });
}

}// namespace nanovdb::util::morphology

#endif // NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED
