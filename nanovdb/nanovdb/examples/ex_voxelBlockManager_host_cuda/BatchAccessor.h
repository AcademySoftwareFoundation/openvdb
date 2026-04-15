// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file BatchAccessor.h

    \brief SIMD-batch analog of NanoVDB's ValueAccessor.

    Caches the 27-entry 3×3×3 leaf-neighbor pointer table around the current
    center leaf, amortizing probeLeaf calls across all batches that process
    voxels within that leaf.

    Design documented in:
      nanovdb/examples/ex_voxelBlockManager_host_cuda/BatchAccessor.md

    Template parameters
    -------------------
    BuildT        NanoVDB build type (determines tree / leaf types).
    ValueT        Scalar or SIMD result type of cachedGetValue.
                  For NanoGrid<float>:          float  or Simd<float,W>
                  For NanoGrid<ValueOnIndex>:   uint64_t or Simd<uint64_t,W>
    VoxelOffsetT  Compact (9-bit) voxel offset within a leaf.
                  Scalar path: uint16_t.  SIMD path: Simd<uint16_t,W>.
    LeafIDT       Leaf index type — reserved for future use by the caller loop.
                  Scalar: uint32_t.  SIMD: Simd<uint32_t,W>.
    PredicateT    Per-lane active predicate (the leafMask).
                  Scalar: bool.  SIMD: SimdMask<float,W> or similar.

    Usage
    -----
    Scalar defaults allow instantiation without a SIMD library.
    For SIMD use, substitute the concrete Simd<> and SimdMask<> types.

    API (see BatchAccessor.md §5 for the full design):
      - advance(newLeafID)              — move to a new center leaf
      - prefetch<di,dj,dk>(vo, mask)   — warm cache for tap (di,dj,dk)
      - cachedGetValue<di,dj,dk>(result, vo, mask) — fill masked result lanes
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <simd_test/Simd.h>   // simd_traits, scalar_traits, Simd<T,W>, SimdMask<T,W>
#include <cstdint>
#include <cassert>
#include <type_traits>

namespace nanovdb {

// =============================================================================
// BatchAccessor
// =============================================================================

template<typename BuildT,
         typename ValueT       = float,
         typename VoxelOffsetT = uint16_t,
         typename LeafIDT      = uint32_t,
         typename PredicateT   = bool>
class BatchAccessor
{
    using GridT = NanoGrid<BuildT>;
    using TreeT = typename GridT::TreeType;
    using LeafT = typename TreeT::LeafNodeType;

    using VO_traits   = util::simd_traits<VoxelOffsetT>;
    using Pred_traits = util::simd_traits<PredicateT>;
    using Val_traits  = util::simd_traits<ValueT>;

    // Scalar element type of ValueT (e.g. float for Simd<float,W>)
    using ScalarValueT = typename Val_traits::scalar_type;

    static constexpr int LaneWidth = VO_traits::width;

    static_assert(VO_traits::width == Pred_traits::width,
        "BatchAccessor: VoxelOffsetT and PredicateT must have the same lane width");
    static_assert(Val_traits::width == 1 || Val_traits::width == VO_traits::width,
        "BatchAccessor: ValueT lane width must be 1 (scalar) or match VoxelOffsetT");

    // The SWAR packed layout in prefetch occupies bits 0–14 of each element
    // (max packed value 0x71C7, max sum 0xE38E).  The element type must therefore
    // be an unsigned integer of at least 16 bits; signed types produce UB on
    // carry overflow, and 8-bit types cannot hold the packed fields.
    using VoxelOffsetScalarT = util::scalar_traits_t<VoxelOffsetT>;
    static_assert(std::is_unsigned_v<VoxelOffsetScalarT>,
        "BatchAccessor: VoxelOffsetT element type must be unsigned "
        "(SWAR carry detection requires wrap-around, not signed overflow)");
    static_assert(sizeof(VoxelOffsetScalarT) >= 2,
        "BatchAccessor: VoxelOffsetT element type must be at least 16 bits "
        "(SWAR packed layout occupies bits 0-14, max sum 0xE38E)");

public:
    // -------------------------------------------------------------------------
    // Direction encoding
    //
    // bit(dx,dy,dz) = (dx+1)*9 + (dy+1)*3 + (dz+1),   dx,dy,dz ∈ {-1,0,+1}
    //
    // Selected entries:
    //   dir( 0, 0, 0) = 13  — center leaf        (mLeafNeighbors[13])
    //   dir(-1, 0, 0) =  4  — x-minus face
    //   dir(+1, 0, 0) = 22  — x-plus  face
    //   dir( 0,-1, 0) = 10  — y-minus face
    //   dir( 0,+1, 0) = 16  — y-plus  face
    //   dir( 0, 0,-1) = 12  — z-minus face
    //   dir( 0, 0,+1) = 14  — z-plus  face
    // -------------------------------------------------------------------------
    static constexpr int dir(int dx, int dy, int dz)
    {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
    }

    // -------------------------------------------------------------------------
    // Construction
    //
    // Eagerly populates mLeafNeighbors[dir(0,0,0)] (the center pointer) and
    // marks bit 13 in mProbedMask.  The center pointer is O(1) to compute
    // (no probeLeaf needed), so there is no reason to defer it.
    //
    // Consequence: cachedGetValue<0,0,0> is valid immediately after construction
    // without any prefetch call.  The SWAR neededMask in prefetch never sets
    // bit 13 (only crossings fire), so the eager center is never redundantly
    // re-probed.
    // -------------------------------------------------------------------------
    BatchAccessor(const GridT& grid, uint32_t firstLeafID)
        : mGrid(grid)
        , mCenterLeafID(firstLeafID)
        , mCenterOrigin(grid.tree().getFirstLeaf()[firstLeafID].origin())
        , mProbedMask(1u << dir(0, 0, 0))
    {
        for (auto& p : mLeafNeighbors) p = nullptr;
        mLeafNeighbors[dir(0, 0, 0)] = &mGrid.tree().getFirstLeaf()[mCenterLeafID];
    }

    // -------------------------------------------------------------------------
    // advance — move to a new center leaf
    //
    // Call when none_of(leafMask): all active lanes have moved past mCenterLeafID.
    // Repopulates the center pointer eagerly and resets mProbedMask to bit 13,
    // so stale neighbor entries are blocked and the center is immediately valid.
    // -------------------------------------------------------------------------
    void advance(uint32_t newLeafID)
    {
        mCenterLeafID              = newLeafID;
        mCenterOrigin              = mGrid.tree().getFirstLeaf()[newLeafID].origin();
        mLeafNeighbors[dir(0,0,0)] = &mGrid.tree().getFirstLeaf()[newLeafID];
        mProbedMask                = (1u << dir(0, 0, 0));
    }

    // -------------------------------------------------------------------------
    // prefetch<di,dj,dk> — warm the neighbor cache for stencil tap (di,dj,dk)
    //
    // For each active (leafMask) lane, computes which neighbor leaf the tap lands
    // in and probes it into mLeafNeighbors[] if not already cached in mProbedMask.
    //
    // The center direction (dir(0,0,0)) is always pre-populated by the constructor
    // and advance(), so it never appears in neededMask and never needs probeLeaf.
    // Every direction in toProbe is therefore a genuine neighbor: full root-to-leaf
    // traversal via mGrid.tree().root().probeLeaf().
    //
    // A null result from probeLeaf means the neighbor leaf does not exist (outside
    // the narrow band); cachedGetValue returns 0 for those lanes.
    // -------------------------------------------------------------------------
    template<int di, int dj, int dk>
    void prefetch(VoxelOffsetT vo, PredicateT leafMask)
    {
        // -----------------------------------------------------------------------
        // SWAR neededMask computation
        //
        // Replace the scalar per-lane loop with a single SIMD add + two horizontal
        // reductions, using a 15-bit packed coordinate representation.
        //
        // packed_lc layout (one group per axis, zero-guard gaps):
        //   bits  0– 2: lz   carry exits at bit  3  (z-axis crossing)
        //   bits  6– 8: lx   carry exits at bit  9  (x-axis crossing)
        //   bits 12–14: ly   carry exits at bit 15  (y-axis crossing)
        //
        // This is expandVoxelOffset() steps 1+2 only (no step 3), because for a
        // fixed (di,dj,dk) each axis has exactly one possible crossing direction,
        // so we need only one group per axis rather than two.
        //
        // packed_d = 3-bit two's complement of each offset placed in the same groups:
        //   dk & 7  at bits [0:2]  (= 8+dk for dk<0, dk for dk>=0)
        //   di & 7  at bits [6:8]
        //   dj & 7  at bits [12:14]
        //
        // After SIMD add(packed_lc, packed_d):
        //   carry at bit  3 SET   ↔  lz + dk ≥ 8   ↔  hi-z crossing (dk > 0)
        //   carry at bit  3 CLEAR ↔  lz + dk <  0  ↔  lo-z crossing (dk < 0)
        //   (same logic for x@bit9 and y@bit15)
        //
        // Inactive lanes carry the sentinel (lc = 4 per axis), which satisfies
        // |d| ≤ 4: never fires a false hi-carry, never clears a lo-carry.
        //
        // For multi-axis taps (more than one nonzero component), the per-axis
        // may-cross flags are combined conservatively: if two axes can independently
        // cross, the edge/corner direction combining both is also added to neededMask.
        // This may over-probe (extra probeLeaf if no single lane crosses both axes
        // simultaneously) but never misses a direction any lane actually needs.
        // For axis-aligned WENO5 taps (one nonzero component) there is no over-probing.
        // -----------------------------------------------------------------------

        // Use VoxelOffsetT directly for the packed arithmetic: LaneWidth elements
        // of VoxelOffsetScalarT in one register → one vpaddw (16-bit) or vpaddd
        // (32-bit) depending on the instantiation.  All intermediate values fit:
        //   packed_lc ≤ 0x71C7, packed_d ≤ 0x71C7, sum ≤ 0xE38E < 2^16.

        // Compile-time packed stencil offset (3-bit two's complement per axis).
        // d & 7u gives the 3-bit representation; for negative d, d & 7 = 8+d.
        static constexpr auto packed_d =
            static_cast<VoxelOffsetScalarT>(
                 (unsigned(dk) & 7u)
               | ((unsigned(di) & 7u) <<  6)
               | ((unsigned(dj) & 7u) << 12));

        // Sentinel for inactive lanes: lc = (4,4,4) → packed = 4|(4<<6)|(4<<12).
        // Note: expandVoxelOffset(kInactiveVoxelOffset=292) = kSentinel15, so even
        // unconditionally expanded inactive-lane vo values yield the sentinel.
        // However, straddle lanes carry arbitrary vo from the next leaf, so we
        // must apply leafMask before the add to avoid false crossing signals.
        static constexpr auto kSentinel15 =
            static_cast<VoxelOffsetScalarT>(4u | (4u << 6u) | (4u << 12u));
        static constexpr auto kMask15 =
            static_cast<VoxelOffsetScalarT>(0b111'000'111'000'111u);

        // Expand the 9-bit voxel offset into the 15-bit SWAR packed form —
        // one vpor + vpsllw + vpand (no scalar loop).
        //   bits [0:2]  = lz,  bits [6:8]  = lx,  bits [12:14] = ly
        // Then blend: active lanes → expanded form, straddle/inactive → sentinel.
        // util::where accepts SimdMask<U,W> for any U (heterogeneous overload).
        const auto expanded =
              (vo | (vo << VoxelOffsetScalarT(9))) & VoxelOffsetT(kMask15);
        const auto packed_lc =
              util::where(leafMask, expanded, VoxelOffsetT(kSentinel15));

        // One SIMD add across all LaneWidth lanes (one vpaddw/vpaddd instruction).
        const auto packed_sum = packed_lc + VoxelOffsetT(packed_d);

        // Horizontal reductions: widen to uint32_t for the carry-bit checks.
        uint32_t hor_or = 0u, hor_and = ~0u;
        for (int i = 0; i < LaneWidth; ++i) {
            const uint32_t s = static_cast<uint32_t>(VO_traits::get(packed_sum, i));
            hor_or  |= s;
            hor_and &= s;
        }

        // Per-axis may-cross flags: compile-time dispatch on sign of d.
        bool x_cross = false, y_cross = false, z_cross = false;
        if constexpr (di > 0) x_cross = bool(hor_or  & (1u <<  9));
        if constexpr (di < 0) x_cross = !bool(hor_and & (1u <<  9));
        if constexpr (dj > 0) y_cross = bool(hor_or  & (1u << 15));
        if constexpr (dj < 0) y_cross = !bool(hor_and & (1u << 15));
        if constexpr (dk > 0) z_cross = bool(hor_or  & (1u <<  3));
        if constexpr (dk < 0) z_cross = !bool(hor_and & (1u <<  3));

        // Compile-time crossing sign per axis.
        constexpr int sx = (di > 0) ? 1 : -1;  // only used when di != 0
        constexpr int sy = (dj > 0) ? 1 : -1;
        constexpr int sz = (dk > 0) ? 1 : -1;

        // Build neededMask: face neighbors, then edge and corner (conservative).
        uint32_t neededMask = 0u;
        if constexpr (di != 0)               { if (x_cross)                   neededMask |= (1u << dir(sx,  0,  0)); }
        if constexpr (dj != 0)               { if (y_cross)                   neededMask |= (1u << dir( 0, sy,  0)); }
        if constexpr (dk != 0)               { if (z_cross)                   neededMask |= (1u << dir( 0,  0, sz)); }
        if constexpr (di != 0 && dj != 0)    { if (x_cross && y_cross)        neededMask |= (1u << dir(sx, sy,  0)); }
        if constexpr (di != 0 && dk != 0)    { if (x_cross && z_cross)        neededMask |= (1u << dir(sx,  0, sz)); }
        if constexpr (dj != 0 && dk != 0)    { if (y_cross && z_cross)        neededMask |= (1u << dir( 0, sy, sz)); }
        if constexpr (di != 0 && dj != 0 && dk != 0) { if (x_cross && y_cross && z_cross) neededMask |= (1u << dir(sx, sy, sz)); }

        // Probe neighbor directions not already cached.
        // Every direction here requires probeLeaf (center is pre-populated, never in toProbe).
        uint32_t toProbe = neededMask & ~mProbedMask;
        while (toProbe) {
            const int d = __builtin_ctz(toProbe);
            mLeafNeighbors[d] = mGrid.tree().root().probeLeaf(originForDir(d));
            mProbedMask |= (1u << d);
            toProbe     &= toProbe - 1;
        }
    }

    // -------------------------------------------------------------------------
    // cachedGetValue<di,dj,dk> — fill masked result lanes from cached leaf table
    //
    // For each active (leafMask) lane, computes the local voxel offset within the
    // appropriate neighbor leaf and calls leaf->getValue(offset).
    //
    // Requires prefetch<di,dj,dk> (or any prefetch covering the same directions)
    // to have been called first.  Debug builds assert mProbedMask coverage.
    //
    // A null leaf pointer (neighbor outside the narrow band) writes 0 to result.
    // Inactive lanes (leafMask[i] == false) are not touched.
    // -------------------------------------------------------------------------
    template<int di, int dj, int dk>
    void cachedGetValue(ValueT& result, VoxelOffsetT vo, PredicateT leafMask) const
    {
        for (int i = 0; i < LaneWidth; ++i) {
            if (!Pred_traits::get(leafMask, i)) continue;
            const auto vo_i = static_cast<uint16_t>(VO_traits::get(vo, i));
            const int lx = (vo_i >> 6) & 7;
            const int ly = (vo_i >> 3) & 7;
            const int lz =  vo_i       & 7;
            const int nx = lx + di, ny = ly + dj, nz = lz + dk;
            const int dx = (nx < 0) ? -1 : (nx >= 8) ? 1 : 0;
            const int dy = (ny < 0) ? -1 : (ny >= 8) ? 1 : 0;
            const int dz = (nz < 0) ? -1 : (nz >= 8) ? 1 : 0;
            // Wrapped local coordinate within the neighbor leaf.
            const int nx_w = nx - dx * 8;
            const int ny_w = ny - dy * 8;
            const int nz_w = nz - dz * 8;
            // NanoVDB leaf layout: offset = lx*64 + ly*8 + lz.
            const uint32_t offset = uint32_t(nx_w) * 64u
                                  + uint32_t(ny_w) *  8u
                                  + uint32_t(nz_w);
            const int d = dir(dx, dy, dz);
            assert((mProbedMask & (1u << d)) && "cachedGetValue: direction not prefetched");
            const LeafT* leaf = mLeafNeighbors[d];
            const ScalarValueT val = leaf
                ? static_cast<ScalarValueT>(leaf->getValue(offset))
                : ScalarValueT(0);
            Val_traits::set(result, i, val);
        }
    }

private:
    // Compute the world-space origin of the leaf at direction bit d from center.
    // bit(dx,dy,dz) = (dx+1)*9 + (dy+1)*3 + (dz+1); leaf stride = 8 per axis.
    Coord originForDir(int d) const
    {
        const int dx = d / 9 - 1;
        const int dy = (d / 3) % 3 - 1;
        const int dz = d % 3 - 1;
        return mCenterOrigin + Coord(dx * 8, dy * 8, dz * 8);
    }

    const GridT& mGrid;
    uint32_t     mCenterLeafID;
    Coord        mCenterOrigin;
    uint32_t     mProbedMask;
    const LeafT* mLeafNeighbors[27];
};

} // namespace nanovdb
