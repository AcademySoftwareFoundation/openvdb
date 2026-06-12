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

}// namespace nanovdb::util

#endif // NANOVDB_UTIL_INJECTION_H_HAS_BEEN_INCLUDED
