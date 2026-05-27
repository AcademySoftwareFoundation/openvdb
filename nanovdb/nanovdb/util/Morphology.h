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

}// namespace nanovdb::util::morphology

#endif // NANOVDB_UTIL_MORPHOLOGY_H_HAS_BEEN_INCLUDED
