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
// Weno5Stencil — 18-tap axis-aligned WENO5 stencil, radius 3
// =============================================================================

/// Concrete StencilT for the WENO5 3D stencil.
/// Taps:  18 axis-aligned offsets in {±1,±2,±3} × {x,y,z}.
/// Hull:   6 extremal offsets that cover all 18 tap crossing directions.
struct Weno5Stencil {
    using Taps = std::tuple<
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
    // directions for any combination of voxel position and WENO5 tap.
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
    // Type aliases — scalar/SIMD split (§5 of design doc)
    // -------------------------------------------------------------------------

    // Output index type: one Simd<uint64_t,W> per tap.
    using IndexVec  = std::conditional_t<W == 1, uint64_t,  util::Simd<uint64_t, W>>;

    // Voxel offset type: loaded from the voxelOffset[] array (uint16_t).
    using OffsetVec = std::conditional_t<W == 1, uint16_t,  util::Simd<uint16_t, W>>;

    // Leaf index type: loaded from the leafIndex[] array (uint32_t).
    using LeafIdVec = std::conditional_t<W == 1, uint32_t,  util::Simd<uint32_t, W>>;

    // Internal mask — derived from leafIndex[] comparisons (uint32_t domain).
    // Passed to BatchAccessor::prefetch / cachedGetValue.
    using LeafMaskVec  = std::conditional_t<W == 1, bool, util::SimdMask<uint32_t, W>>;

    // External mask — returned by moveTo; semantically over mIndices (uint64_t).
    // Both LeafMaskVec and IndexMaskVec are W-bit masks; conversion is a
    // boolean round-trip (see SimdMask converting constructor in Simd.h).
    using IndexMaskVec = std::conditional_t<W == 1, bool, util::SimdMask<uint64_t, W>>;

    // BatchAccessor parameterised with LeafMaskVec (prefetch/cachedGetValue domain).
    using BatchAcc = std::conditional_t<W == 1,
                         BatchAccessor<BuildT, uint64_t,  uint16_t, bool>,
                         BatchAccessor<BuildT, IndexVec,  OffsetVec, LeafMaskVec>>;

    static constexpr int SIZE      = int(std::tuple_size_v<typename StencilT::Taps>);
    static constexpr int HULL_SIZE = int(std::tuple_size_v<typename StencilT::Hull>);

public:
    // -------------------------------------------------------------------------
    // Construction
    //
    // firstLeafID  -- VBM block's starting leaf ID (vbm.hostFirstLeafID()[blockID]).
    // nExtraLeaves -- number of distinct center-leaf advances possible in this block
    //                 (computed by the caller from the jumpMap).  Used only as a
    //                 debug-mode assert bound; not needed for correctness.
    //                 See StencilAccessor.md §7 for removal instructions.
    // -------------------------------------------------------------------------
    StencilAccessor(const GridT& grid, uint32_t firstLeafID, uint32_t nExtraLeaves)
        : mBatch(grid, firstLeafID)
#ifndef NDEBUG
        , mNExtraLeaves(nExtraLeaves)
#endif
    {
        (void)nExtraLeaves;  // suppress unused-parameter warning in release builds
    }

    // -------------------------------------------------------------------------
    // moveTo -- gather all tap indices for a W-wide batch of center voxels
    //
    // leafIndex   -- ptr to leafIndex[batchStart]   (uint32_t array from decodeInverseMaps)
    // voxelOffset -- ptr to voxelOffset[batchStart] (uint16_t array from decodeInverseMaps)
    //
    // Returns the initial active-lane mask (leafSlice != UnusedLeafIndex), widened
    // to IndexMaskVec.  Active lanes have valid results in mIndices[0..SIZE-1].
    // Inactive lanes hold 0 (NanoVDB background index).
    //
    // See StencilAccessor.md §8 for the full straddling loop design.
    // -------------------------------------------------------------------------
    IndexMaskVec moveTo(const uint32_t* leafIndex, const uint16_t* voxelOffset)
    {
        // Zero all tap slots — inactive lanes will hold 0 (background index).
        zeroIndices(std::make_index_sequence<SIZE>{});

        // Load this batch.
        const LeafIdVec leafSlice = loadLeafIdVec(leafIndex);
        const OffsetVec voVec     = loadOffsetVec(voxelOffset);

        // Initial active-lane mask (which lanes have real voxels).
        LeafMaskVec activeMask = (leafSlice != LeafIdVec(UnusedLeafIndex));

        // Save before the drain loop — this is what we return.
        const IndexMaskVec resultMask = widenMask(activeMask);

        if (util::none_of(activeMask)) return resultMask;

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
            // stencil can reach, before any cachedGetValue is called.
            prefetchHull(voVec, leafMask, std::make_index_sequence<HULL_SIZE>{});

            // Compute all tap indices and blend into mIndices.
            calcTaps(voVec, leafMask, std::make_index_sequence<SIZE>{});

            // Remove processed lanes.
            activeMask = activeMask & !leafMask;
        }

        return resultMask;
    }

    // -------------------------------------------------------------------------
    // getValue<DI,DJ,DK> -- access tap result by compile-time coordinate
    //
    // Resolved entirely at compile time via the findIndex constexpr fold.
    // Returns a const reference valid until the next moveTo() call.
    // -------------------------------------------------------------------------
    template<int DI, int DJ, int DK>
    const IndexVec& getValue() const
    {
        constexpr int I = detail::findIndex<typename StencilT::Taps, DI, DJ, DK>(
            std::make_index_sequence<SIZE>{});
        static_assert(I >= 0, "StencilAccessor::getValue: tap not in stencil");
        return mIndices[I];
    }

    // -------------------------------------------------------------------------
    // operator[] -- indexed tap access (for generic iteration over all taps)
    //
    // No bounds check in release.  Same lifetime as getValue.
    // -------------------------------------------------------------------------
    const IndexVec& operator[](int i) const { return mIndices[i]; }

    static constexpr int size() { return SIZE; }

private:
    // -------------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------------

    // Compile-time zero of all SIZE index slots.
    template<size_t... Is>
    void zeroIndices(std::index_sequence<Is...>)
    {
        ((mIndices[Is] = IndexVec(uint64_t(0))), ...);
    }

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

    // Widen LeafMaskVec (uint32_t domain) → IndexMaskVec (uint64_t domain).
    // Both are W-bit masks; SimdMask<T,W> has a converting constructor from
    // SimdMask<U,W> that copies the bool array element-by-element (Simd.h §B).
    // The stdx backend uses a boolean round-trip (WhereExpression, Simd.h §A).
    static IndexMaskVec widenMask(LeafMaskVec m)
    {
        if constexpr (W == 1) return m;
        else return IndexMaskVec(m);
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

    // Compile-time fold: cachedGetValue for all SIZE taps, where-blend into mIndices.
    template<size_t... Is>
    void calcTaps(OffsetVec voVec, LeafMaskVec leafMask, std::index_sequence<Is...>)
    {
        (blendOneTap<Is>(voVec, leafMask), ...);
    }

    // Fetch one tap and blend its result into mIndices[I] for the active lanes.
    // The where(leafMask, mIndices[I]) = tmp blend uses the heterogeneous
    // where() overload from Simd.h: LeafMaskVec (uint32_t) applied to
    // IndexVec (uint64_t).  Both are W-bit masks; Simd.h handles the conversion.
    template<size_t I>
    void blendOneTap(OffsetVec voVec, LeafMaskVec leafMask)
    {
        using P = std::tuple_element_t<I, typename StencilT::Taps>;
        IndexVec tmp(uint64_t(0));
        mBatch.template cachedGetValue<P::di, P::dj, P::dk>(tmp, voVec, leafMask);
        util::where(leafMask, mIndices[I]) = tmp;
    }

    // -------------------------------------------------------------------------
    // Members
    // -------------------------------------------------------------------------

    BatchAcc  mBatch;           // owns neighbor-leaf cache, mCenterLeafID
    IndexVec  mIndices[SIZE];   // one vector per tap — output store

#ifndef NDEBUG
    uint32_t  mNExtraLeaves;    // removable sanity bound on center-leaf advances
#endif
};

} // namespace nanovdb
