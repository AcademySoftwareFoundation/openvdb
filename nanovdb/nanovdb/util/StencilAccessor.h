// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file StencilAccessor.h

    \brief SIMD stencil-index gatherer built on BatchAccessor.

    Wraps a BatchAccessor and owns the straddling loop, prefetch-hull
    sequencing, and per-tap cachedGetValue calls for one VBM block.
    Its output is a fixed-size array of Simd<uint64_t,W> — one vector
    per stencil tap — containing ValueOnIndex indices for all W lanes.

    Design documented in:
      nanovdb/examples/ex_voxelBlockManager_host_cuda/StencilAccessor.md

    Template parameters
    -------------------
    BuildT      NanoVDB build type (e.g. ValueOnIndex).
    W           SIMD lane width.
    StencilT    Policy class describing the stencil.  Must expose:
                  using Taps = std::tuple<StencilPoint<di,dj,dk>...>;
                  using Hull = std::tuple<StencilPoint<di,dj,dk>...>;
    UnusedLeafIndex
                Sentinel written by decodeInverseMaps for padding slots.
                Defaults to ~uint32_t(0) (VoxelBlockManagerBase::UnusedLeafIndex).

    Usage
    -----
    Construct once per VBM block; call moveTo() for each SIMD batch.
    See StencilAccessor.md §10 for the caller pattern.
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Simd.h>
#include <nanovdb/util/BatchAccessor.h>

#include <cstdint>
#include <cstring>      // std::memset
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>      // std::index_sequence, std::make_index_sequence

namespace nanovdb {

// =============================================================================
// StencilPoint — compile-time stencil tap offset
// =============================================================================

/// Compile-time 3D offset used as a type (not a value) in StencilT::Taps
/// and StencilT::Hull tuples.
template<int DI, int DJ, int DK>
struct StencilPoint {
    static constexpr int di = DI;
    static constexpr int dj = DJ;
    static constexpr int dk = DK;
};

// =============================================================================
// findIndex — compile-time inverse map: (DI,DJ,DK) → slot index in a Taps tuple
// =============================================================================

namespace detail {

/// Returns the first index Is in [0,N) where tuple_element_t<Is,TupsT>
/// matches (DI,DJ,DK), or -1 if not found.
template<typename TupsT, int DI, int DJ, int DK, size_t... Is>
constexpr int findIndex(std::index_sequence<Is...>)
{
    int result = -1;
    // Fold: for each Is, if the tap matches and we haven't found one yet, record it.
    ((std::tuple_element_t<Is, TupsT>::di == DI &&
      std::tuple_element_t<Is, TupsT>::dj == DJ &&
      std::tuple_element_t<Is, TupsT>::dk == DK &&
      result < 0
      ? (result = int(Is)) : 0), ...);
    return result;
}

} // namespace detail

// =============================================================================
// Weno5Stencil — 19-tap axis-aligned WENO5 stencil, radius 3
// =============================================================================

/// Concrete StencilT for the WENO5 3D stencil.
/// Taps:  19 axis-aligned offsets — the center plus {±1,±2,±3} along each of x,y,z.
/// Hull:   6 extremal offsets that cover all 18 non-center tap crossing directions.
///
/// Tap ordering matches WenoPt<i,j,k>::idx in nanovdb/math/Stencils.h:
///    idx  0     : <0,0,0>
///    idx  1.. 6 : x-axis  <-3,0,0> <-2,0,0> <-1,0,0> <+1,0,0> <+2,0,0> <+3,0,0>
///    idx  7..12 : y-axis  <0,-3,0> <0,-2,0> <0,-1,0> <0,+1,0> <0,+2,0> <0,+3,0>
///    idx 13..18 : z-axis  <0,0,-3> <0,0,-2> <0,0,-1> <0,0,+1> <0,0,+2> <0,0,+3>
struct Weno5Stencil {
    using Taps = std::tuple<
        // center
        StencilPoint< 0, 0, 0>,
        // x-axis
        StencilPoint<-3, 0, 0>, StencilPoint<-2, 0, 0>, StencilPoint<-1, 0, 0>,
        StencilPoint<+1, 0, 0>, StencilPoint<+2, 0, 0>, StencilPoint<+3, 0, 0>,
        // y-axis
        StencilPoint< 0,-3, 0>, StencilPoint< 0,-2, 0>, StencilPoint< 0,-1, 0>,
        StencilPoint< 0,+1, 0>, StencilPoint< 0,+2, 0>, StencilPoint< 0,+3, 0>,
        // z-axis
        StencilPoint< 0, 0,-3>, StencilPoint< 0, 0,-2>, StencilPoint< 0, 0,-1>,
        StencilPoint< 0, 0,+1>, StencilPoint< 0, 0,+2>, StencilPoint< 0, 0,+3>
    >;
    // Hull = 6 extremal taps that collectively probe all reachable face-neighbor
    // directions for any combination of voxel position and non-center WENO5 tap.
    // The center tap never crosses a leaf, so it's absent here by design.
    // See StencilAccessor.md §4b for the monotonicity argument.
    using Hull = std::tuple<
        StencilPoint<-3, 0, 0>, StencilPoint<+3, 0, 0>,
        StencilPoint< 0,-3, 0>, StencilPoint< 0,+3, 0>,
        StencilPoint< 0, 0,-3>, StencilPoint< 0, 0,+3>
    >;
};

// =============================================================================
// StencilAccessor
// =============================================================================

template<typename BuildT,
         int      W,
         typename StencilT,
         uint32_t UnusedLeafIndex = ~uint32_t(0)>
class StencilAccessor
{
    using GridT = NanoGrid<BuildT>;

    // -------------------------------------------------------------------------
    // Private type aliases — only used inside moveTo().
    //
    // These are the W-lane SIMD types that carry the input arrays through the
    // straddling loop and the SWAR direction extraction.  They do NOT appear
    // in the public API: callers consume `mIndices` (raw uint64_t[SIZE][W])
    // directly, and `moveTo` returns `void` — active-lane information is read
    // from `leafIndex[]` vs `UnusedLeafIndex` by the caller.
    // -------------------------------------------------------------------------
    using OffsetVec    = std::conditional_t<W == 1, uint16_t, util::Simd<uint16_t, W>>;
    using LeafIdVec    = std::conditional_t<W == 1, uint32_t, util::Simd<uint32_t, W>>;
    using LeafMaskVec  = std::conditional_t<W == 1, bool,     util::SimdMask<uint32_t, W>>;

    using BatchAcc = std::conditional_t<W == 1,
                         BatchAccessor<BuildT, uint64_t,  uint16_t, bool>,
                         BatchAccessor<BuildT, uint64_t,  OffsetVec, LeafMaskVec>>;

    static constexpr int SIZE      = int(std::tuple_size_v<typename StencilT::Taps>);
    static constexpr int HULL_SIZE = int(std::tuple_size_v<typename StencilT::Hull>);

public:
    // -------------------------------------------------------------------------
    // Public API — entirely free of Simd<>/SimdMask<> types.
    //
    // Storage layout: `mIndices[tap][lane]` is a plain uint64_t.  Callers are
    // free to SIMD-load it with whatever backend they choose
    // (e.g. `Simd<uint64_t,W>::load(stencilAcc.mIndices[k], element_aligned)`),
    // iterate scalarly, or pass slices to downstream kernels — we don't
    // impose a choice.
    //
    // Layout is part of the ABI: [SIZE][W] row-major.  Changing it is
    // a breaking change.
    // -------------------------------------------------------------------------
    alignas(64) uint64_t mIndices[SIZE][W];

    // -------------------------------------------------------------------------
    // Construction
    //
    // firstLeafID  -- VBM block's starting leaf ID (vbm.hostFirstLeafID()[blockID]).
    // nExtraLeaves -- number of distinct center-leaf advances possible in this block
    //                 (computed by the caller from the jumpMap).  Debug-only bound
    //                 on the straddling loop; not needed for correctness.
    // -------------------------------------------------------------------------
    StencilAccessor(const GridT& grid, uint32_t firstLeafID, uint32_t nExtraLeaves)
        : mBatch(grid, firstLeafID)
#ifndef NDEBUG
        , mNExtraLeaves(nExtraLeaves)
#endif
    {
        (void)nExtraLeaves;
    }

    // -------------------------------------------------------------------------
    // moveTo -- fill mIndices[0..SIZE-1][0..W-1] for a W-wide batch.
    //
    // leafIndex   -- ptr to leafIndex[batchStart]   (uint32_t from decodeInverseMaps)
    // voxelOffset -- ptr to voxelOffset[batchStart] (uint16_t from decodeInverseMaps)
    //
    // Active-lane semantics: a lane i is "active" iff
    //     leafIndex[i] != UnusedLeafIndex
    // Active lanes receive their 19 tap indices in mIndices[k][i].
    // Inactive lanes are zeroed (NanoVDB background index).
    //
    // Caller pattern:
    //     stencilAcc.moveTo(leafIndex + bs, voxelOffset + bs);
    //     for (int i = 0; i < W; ++i) {
    //         if (leafIndex[bs + i] == UnusedLeafIndex) continue;
    //         ...stencilAcc.mIndices[k][i]...
    //     }
    //
    // See StencilAccessor.md §8 for the full straddling loop design.
    // -------------------------------------------------------------------------
    void moveTo(const uint32_t* leafIndex, const uint16_t* voxelOffset)
    {
        // Zero the whole results buffer — inactive lanes stay 0.
        std::memset(mIndices, 0, sizeof(mIndices));

        // Load the batch into SIMD registers for the SWAR / straddling logic.
        const LeafIdVec leafSlice = loadLeafIdVec(leafIndex);
        const OffsetVec voVec     = loadOffsetVec(voxelOffset);

        // Initial active-lane mask (which lanes have real voxels).
        LeafMaskVec activeMask = (leafSlice != LeafIdVec(UnusedLeafIndex));

        if (util::none_of(activeMask)) return;

#ifndef NDEBUG
        uint32_t nAdvances = 0;
#endif

        // Straddling loop: consume one center leaf's worth of lanes per iteration.
        while (util::any_of(activeMask)) {
            const LeafMaskVec leafMask =
                activeMask & (leafSlice == LeafIdVec(mBatch.centerLeafID()));

            if (util::none_of(leafMask)) {
                // No lanes for this leaf — advance to next.
                mBatch.advance(mBatch.centerLeafID() + 1);
#ifndef NDEBUG
                assert(++nAdvances <= mNExtraLeaves);
#endif
                continue;
            }

            // Prefetch hull — warms all neighbor-leaf directions the full
            // stencil can reach, before any cachedGetValue runs.
            prefetchHull(voVec, leafMask, std::make_index_sequence<HULL_SIZE>{});

            // Fill all SIZE tap entries for the lanes in leafMask.
            calcTaps(voVec, leafMask, std::make_index_sequence<SIZE>{});

            // Remove processed lanes.
            activeMask = activeMask & !leafMask;
        }
    }

    // -------------------------------------------------------------------------
    // moveToInLeaf -- benchmarking variant: identical to moveTo except that
    // each tap is wrapped to the center leaf via (localVoxel + tap) mod 8.
    //
    // Purpose: measure the hybrid pipeline's floor cost with 18 distinct
    // compile-time taps that all access the SAME leaf, preventing both the
    // cross-leaf L1 pressure and the compiler CSE of identical taps.  All
    // StencilT::Taps offsets must be in [0, 7] per axis.
    //
    // NOT for production use -- results have no geometric meaning; they
    // just exercise the hybrid's code path under a controlled cache regime.
    // -------------------------------------------------------------------------
    void moveToInLeaf(const uint32_t* leafIndex, const uint16_t* voxelOffset)
    {
        std::memset(mIndices, 0, sizeof(mIndices));

        const LeafIdVec leafSlice = loadLeafIdVec(leafIndex);
        const OffsetVec voVec     = loadOffsetVec(voxelOffset);

        LeafMaskVec activeMask = (leafSlice != LeafIdVec(UnusedLeafIndex));

        if (util::none_of(activeMask)) return;

#ifndef NDEBUG
        uint32_t nAdvances = 0;
#endif

        while (util::any_of(activeMask)) {
            const LeafMaskVec leafMask =
                activeMask & (leafSlice == LeafIdVec(mBatch.centerLeafID()));

            if (util::none_of(leafMask)) {
                mBatch.advance(mBatch.centerLeafID() + 1);
#ifndef NDEBUG
                assert(++nAdvances <= mNExtraLeaves);
#endif
                continue;
            }

            // No prefetchHull — all targets are the center leaf by construction.
            calcTapsInLeaf(voVec, leafMask, std::make_index_sequence<SIZE>{});

            activeMask = activeMask & !leafMask;
        }
    }

    // -------------------------------------------------------------------------
    // tapIndex<DI,DJ,DK>() -- compile-time tap lookup.
    //
    // Returns the slot in mIndices that corresponds to a named stencil tap,
    // resolved at compile time against StencilT::Taps.  A tap that is not in
    // the stencil produces a static_assert.
    //
    // Usage (reorder-safe, zero runtime cost):
    //     auto& xm3 = stencilAcc.mIndices[SAccT::tapIndex<-3,0,0>()];
    // -------------------------------------------------------------------------
    template<int DI, int DJ, int DK>
    static constexpr int tapIndex()
    {
        constexpr int I = detail::findIndex<typename StencilT::Taps, DI, DJ, DK>(
            std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "StencilAccessor::tapIndex: tap not in stencil");
        return I;
    }

    static constexpr int size() { return SIZE; }

private:
    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    // Load LeafIdVec from a uint32_t pointer (scalar or SIMD).
    static LeafIdVec loadLeafIdVec(const uint32_t* p)
    {
        if constexpr (W == 1) return *p;
        else return LeafIdVec(p, util::element_aligned);
    }

    // Load OffsetVec from a uint16_t pointer (scalar or SIMD).
    static OffsetVec loadOffsetVec(const uint16_t* p)
    {
        if constexpr (W == 1) return *p;
        else return OffsetVec(p, util::element_aligned);
    }

    // Compile-time fold: prefetch all HULL_SIZE hull directions.
    template<size_t... Is>
    void prefetchHull(OffsetVec voVec, LeafMaskVec leafMask, std::index_sequence<Is...>)
    {
        using Hull = typename StencilT::Hull;
        (mBatch.template prefetch<
            std::tuple_element_t<Is, Hull>::di,
            std::tuple_element_t<Is, Hull>::dj,
            std::tuple_element_t<Is, Hull>::dk
         >(voVec, leafMask), ...);
    }

    // Compile-time fold: cachedGetValue for all SIZE taps, write directly into mIndices.
    // No where-blend: cachedGetValue's scalar tail writes only leafMask-active
    // lanes; lanes outside leafMask keep whatever was written by a previous
    // straddling-loop iteration (or zero from the initial memset).
    template<size_t... Is>
    void calcTaps(OffsetVec voVec, LeafMaskVec leafMask, std::index_sequence<Is...>)
    {
        using Taps = typename StencilT::Taps;
        (mBatch.template cachedGetValue<
             std::tuple_element_t<Is, Taps>::di,
             std::tuple_element_t<Is, Taps>::dj,
             std::tuple_element_t<Is, Taps>::dk
         >(mIndices[Is], voVec, leafMask), ...);
    }

    // Benchmark-only counterpart: forces all taps into the center leaf.
    template<size_t... Is>
    void calcTapsInLeaf(OffsetVec voVec, LeafMaskVec leafMask, std::index_sequence<Is...>)
    {
        using Taps = typename StencilT::Taps;
        (mBatch.template cachedGetValueInLeaf<
             std::tuple_element_t<Is, Taps>::di,
             std::tuple_element_t<Is, Taps>::dj,
             std::tuple_element_t<Is, Taps>::dk
         >(mIndices[Is], voVec, leafMask), ...);
    }

    // -------------------------------------------------------------------------
    // Members
    // -------------------------------------------------------------------------

    BatchAcc  mBatch;           // owns neighbor-leaf cache, mCenterLeafID

#ifndef NDEBUG
    uint32_t  mNExtraLeaves;    // removable sanity bound on center-leaf advances
#endif
};

} // namespace nanovdb
