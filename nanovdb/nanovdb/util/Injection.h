// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/util/Injection.h

    \author Efty Sifakis

    \brief Host-side data-injection helpers on sidecars of distinct indexGrid topologies.
           Parallel counterparts to the CUDA functors in nanovdb/util/cuda/Injection.cuh.
*/

#ifndef NANOVDB_UTIL_INJECTION_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_INJECTION_H_HAS_BEEN_INCLUDED

#include <cstdint>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/ForEach.h>

namespace nanovdb::util {

/// @brief Populate a sidecar of leaf-enumerated bitmasks (associated with a destination grid) with
///        the voxels also present in a source grid. Host counterpart of the CUDA
///        InjectGridMaskFunctor.
///
/// @param srcGrid       (in)  Source grid (read host-side).
/// @param dstGrid       (in)  Destination grid whose leaf enumeration indexes the sidecar.
/// @param dstLeafMasks  (out) Per-destination-leaf bitmask = src.valueMask ∩ dst.valueMask, or empty
///                            where the destination leaf has no source counterpart.
/// @param dstLeafCount  (in)  Number of leaf nodes in the destination grid.
///
/// Thread-centric: one task per destination leaf, each writing only its own sidecar entry, so no
/// atomics. If the source topology is not a subset of the destination, the result is the
/// intersection of the two grids.
template<typename BuildT>
inline void injectGridMask(
    const NanoGrid<BuildT> *srcGrid,
    const NanoGrid<BuildT> *dstGrid,
    Mask<3> *dstLeafMasks,
    std::size_t dstLeafCount)
{
    util::forEach(0, dstLeafCount, 1, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        const auto& dstTree = dstGrid->tree();
        for (auto dstLeafID = r.begin(); dstLeafID != r.end(); ++dstLeafID) {
            const auto& dstLeaf = dstTree.template getFirstNode<0>()[dstLeafID];
            auto& resultMask = dstLeafMasks[dstLeafID];
            auto srcLeafPtr = srcTree.root().probeLeaf(dstLeaf.origin());
            if (srcLeafPtr) {
                resultMask  = srcLeafPtr->valueMask();
                resultMask &= dstLeaf.valueMask();
            }
            else
                resultMask.setOff();
        }
    });
}

/// @brief Copy sidecar values from a source grid into the sidecar of an overlapping destination grid,
///        for the voxels active in both. Host counterpart of the CUDA InjectGridDataFunctor.
///
/// @tparam BuildT     Grid build type (e.g. ValueOnIndex).
/// @tparam ValueType  Sidecar element type, copied by assignment.
/// @tparam offset     Compile-time additive offset into both sidecars (multi-channel support).
///
/// @param srcGrid      (in)  Source grid (read host-side).
/// @param dstGrid      (in)  Destination grid (read host-side).
/// @param srcData      (in)  Source sidecar, indexed by srcGrid's ValueOnIndex enumeration.
/// @param dstData      (out) Destination sidecar, indexed by dstGrid's enumeration.
/// @param srcLeafCount (in)  Number of leaf nodes in the source grid (nodeCount[0]).
///
/// Drives the source tree: one task per source leaf, each probing the destination for an overlapping
/// leaf (early-out if none) and copying the intersection of their value masks. Destination voxels with
/// no source counterpart are left unchanged. Source-leaf origins are unique, so each task writes a
/// distinct destination leaf -- no atomics.
///
/// Rather than calling getValue() per voxel (a per-voxel popcount, costly on CPU without AVX-512), each
/// word's base sidecar offset is re-derived from the leaf's packed mPrefixSum via a one-step carry, and
/// the within-word rank (srcCnt/dstCnt) is advanced one increment per active bit. This mirrors the CUDA
/// functor's srcOffset/srcCnt decomposition while iterating words and bits serially.
template<typename BuildT, typename ValueType, int64_t offset = 0>
inline void injectGridData(
    const NanoGrid<BuildT> *srcGrid,
    const NanoGrid<BuildT> *dstGrid,
    const ValueType *srcData,
    ValueType *dstData,
    std::size_t srcLeafCount)
{
    util::forEach(0, srcLeafCount, 8, [=](const util::Range1D &r) {
        const auto& srcTree = srcGrid->tree();
        const auto& dstTree = dstGrid->tree();
        for (auto srcLeafID = r.begin(); srcLeafID != r.end(); ++srcLeafID) {
            const auto& srcLeaf = srcTree.template getFirstNode<0>()[srcLeafID];
            auto dstLeafPtr = dstTree.root().probeLeaf(srcLeaf.origin());
            if (!dstLeafPtr) continue; // no overlapping destination leaf -> nothing to copy

            uint64_t srcOffset = srcLeaf.firstOffset();    // word 0 base (prefix 0)
            uint64_t dstOffset = dstLeafPtr->firstOffset();
            uint64_t srcPS = srcLeaf.mPrefixSum;                   // running copies (one-step carry)
            uint64_t dstPS = dstLeafPtr->mPrefixSum;
            for (int w = 0; w < Mask<3>::WORD_COUNT; ++w) {
                const uint64_t srcWord = srcLeaf.valueMask().words()[w];
                const uint64_t dstWord = dstLeafPtr->valueMask().words()[w];
                const uint64_t both    = srcWord & dstWord;
                if (both) {
                    uint64_t srcCnt = 0, dstCnt = 0; // within-word rank, reset per word
                    for (uint64_t bit = 1; bit; bit <<= 1) {
                        if (bit & both)
                            dstData[int64_t(dstOffset + dstCnt) + offset] =
                                srcData[int64_t(srcOffset + srcCnt) + offset];
                        if (bit & srcWord) ++srcCnt; // advance after copy (rank = count below)
                        if (bit & dstWord) ++dstCnt;
                    }
                }
                // Advance each word's base to the next word via the one-step prefix-sum carry; must
                // run even when the word had no overlap, so later words stay correctly based.
                srcOffset = srcLeaf.firstOffset()    + (srcPS & 511u); srcPS >>= 9u;
                dstOffset = dstLeafPtr->firstOffset() + (dstPS & 511u); dstPS >>= 9u;
            }
        }
    });
}

}// namespace nanovdb::util

#endif // NANOVDB_UTIL_INJECTION_H_HAS_BEEN_INCLUDED
