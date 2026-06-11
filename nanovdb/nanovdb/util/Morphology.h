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

}// namespace nanovdb::util::morphology

#endif // NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED
