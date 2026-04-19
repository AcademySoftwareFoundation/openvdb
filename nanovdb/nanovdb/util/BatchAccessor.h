// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file BatchAccessor.h

    \brief SIMD-batch analog of NanoVDB's ValueAccessor.

    Caches the 27-entry 3x3x3 leaf-neighbor pointer table around the current
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
    PredicateT    Per-lane active predicate (the leafMask).
                  Scalar: bool.  SIMD: SimdMask<uint32_t,W>.

    Usage
    -----
    Scalar defaults allow instantiation without a SIMD library.
    For SIMD use, substitute the concrete Simd<> and SimdMask<> types.

    API (see BatchAccessor.md Sec.5 for the full design):
      - advance(newLeafID)              -- move to a new center leaf
      - prefetch<di,dj,dk>(vo, mask)   -- warm cache for tap (di,dj,dk)
      - cachedGetValue<di,dj,dk>(result, vo, mask) -- fill masked result lanes
*/

#pragma once

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Simd.h>
#include <cstdint>
#include <cassert>
#include <functional>
#include <type_traits>

namespace nanovdb {

// =============================================================================
// BatchAccessor
// =============================================================================

template<typename BuildT,
         typename ValueT       = float,
         typename VoxelOffsetT = uint16_t,
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

    // SIMD bundle types for the ingredient gather.
    // Degrade to plain scalar types when LaneWidth == 1.
    using LeafIDVecT   = std::conditional_t<LaneWidth == 1, uint32_t, util::Simd<uint32_t, LaneWidth>>;
    using LeafDataVecT = std::conditional_t<LaneWidth == 1, uint64_t, util::Simd<uint64_t, LaneWidth>>;

    static_assert(VO_traits::width == Pred_traits::width,
        "BatchAccessor: VoxelOffsetT and PredicateT must have the same lane width");
    static_assert(std::is_same_v<PredicateT, bool> ||
                  std::is_same_v<PredicateT, util::SimdMask<uint32_t, LaneWidth>>,
        "BatchAccessor: PredicateT must be bool (scalar) or SimdMask<uint32_t,W> (SIMD)");
    static_assert(Val_traits::width == 1 || Val_traits::width == VO_traits::width,
        "BatchAccessor: ValueT lane width must be 1 (scalar) or match VoxelOffsetT");

    // The SWAR packed layout in prefetch occupies bits 0-14 of each element
    // (max packed value 0x1CE7, max sum 0x4A52).  The element type must therefore
    // be an unsigned integer of at least 16 bits; signed types produce UB on
    // carry overflow, and 8-bit types cannot hold the packed fields.
    using VoxelOffsetScalarT = util::scalar_traits_t<VoxelOffsetT>;
    static_assert(std::is_unsigned_v<VoxelOffsetScalarT>,
        "BatchAccessor: VoxelOffsetT element type must be unsigned "
        "(SWAR carry detection requires wrap-around, not signed overflow)");
    static_assert(sizeof(VoxelOffsetScalarT) >= 2,
        "BatchAccessor: VoxelOffsetT element type must be at least 16 bits "
        "(SWAR packed layout occupies bits 0-14, max sum 0x4A52)");

public:
    // -------------------------------------------------------------------------
    // Direction encoding
    //
    // dir(dx,dy,dz) = (dx+1)*9 + (dy+1)*3 + (dz+1),   dx,dy,dz in {-1,0,+1}
    //
    // Selected entries:
    //   dir( 0, 0, 0) = 13  -- center leaf        (mNeighborLeafIDs[13])
    //   dir(-1, 0, 0) =  4  -- x-minus face
    //   dir(+1, 0, 0) = 22  -- x-plus  face
    //   dir( 0,-1, 0) = 10  -- y-minus face
    //   dir( 0,+1, 0) = 16  -- y-plus  face
    //   dir( 0, 0,-1) = 12  -- z-minus face
    //   dir( 0, 0,+1) = 14  -- z-plus  face
    //
    // Sentinel leaf ID for directions outside the narrow band (no leaf exists).
    // -------------------------------------------------------------------------
    static constexpr int      dir(int dx, int dy, int dz)
    {
        return (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1);
    }
    static constexpr uint32_t kNullLeafID = ~uint32_t(0);

    // -------------------------------------------------------------------------
    // SWAR 15-bit packed encoding constants
    //
    // packed layout:  lx[10:12] | gap[13:14] | ly[5:7] | gap[8:9] | lz[0:2] | gap[3:4]
    //
    // kSwarXZMask  -- keeps lz [0:2] and lx [6:8->10:12] after (vo | vo<<4)
    // kSwarYMask   -- keeps ly [3:5->5:7] after (vo<<2)
    // kSwarSentinel-- inactive-lane value: lx=ly=lz=4, chosen so that
    //                (sentinel + tap) never triggers a false crossing signal
    // -------------------------------------------------------------------------
    static constexpr uint16_t kSwarXZMask   = 0x1C07u;
    static constexpr uint16_t kSwarYMask    = 0x00E0u;
    static constexpr uint16_t kSwarSentinel = 4u | (4u << 5u) | (4u << 10u);

    // -------------------------------------------------------------------------
    // Construction
    //
    // Eagerly populates mNeighborLeafIDs[dir(0,0,0)] (the center leaf ID) and
    // marks bit 13 in mProbedMask.  The center ID is O(1) to compute
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
        , mFirstLeaf   (grid.tree().getFirstLeaf())
        , mOffsetBase  (reinterpret_cast<const uint64_t*>(&grid.tree().getFirstLeaf()[0].data()->mOffset))
        , mPrefixBase  (reinterpret_cast<const uint64_t*>(&grid.tree().getFirstLeaf()[0].data()->mPrefixSum))
        , mMaskWordBase(grid.tree().getFirstLeaf()[0].valueMask().words())
    {
        for (auto& id : mNeighborLeafIDs) id = kNullLeafID;
        mNeighborLeafIDs[dir(0, 0, 0)] = mCenterLeafID;
    }

    // -------------------------------------------------------------------------
    // centerLeafID -- read the current center leaf ID
    //
    // Exposed for StencilAccessor::moveTo, which needs it for the
    // leafSlice == centerLeafID() comparison in the straddling loop.
    // There is no raw setter; advance() is the sole legitimate transition.
    // -------------------------------------------------------------------------
    uint32_t centerLeafID() const { return mCenterLeafID; }

    // -------------------------------------------------------------------------
    // advance -- move to a new center leaf
    //
    // Call when none_of(leafMask): all active lanes have moved past mCenterLeafID.
    // Resets all neighbor IDs to kNullLeafID, repopulates the center eagerly,
    // and resets mProbedMask to bit 13 so the center is immediately valid.
    // Resetting all 27 IDs (108 bytes) ensures mNeighborLeafIDs[d] == kNullLeafID
    // iff bit d is absent from mProbedMask -- a clean invariant for SIMD gather.
    // -------------------------------------------------------------------------
    void advance(uint32_t newLeafID)
    {
        mCenterLeafID = newLeafID;
        mCenterOrigin = mGrid.tree().getFirstLeaf()[newLeafID].origin();
        for (auto& id : mNeighborLeafIDs) id = kNullLeafID;
        mNeighborLeafIDs[dir(0, 0, 0)] = newLeafID;
        mProbedMask                    = (1u << dir(0, 0, 0));
    }

    // -------------------------------------------------------------------------
    // prefetch<di,dj,dk> -- warm the neighbor cache for stencil tap (di,dj,dk)
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
        // packed_lc layout -- 5-bit groups, tightly packed, no inter-group gaps:
        //   bits  0- 2: lz   carry region bits  3-4  (z-axis)
        //   bits  5- 7: ly   carry region bits  8-9  (y-axis)
        //   bits 10-12: lx   carry region bits 13-14 (x-axis)
        //
        // All carry bits land within [0:14], fitting cleanly in uint16_t with
        // bit 15 unused.  The z,y,x ordering matches the weight sequence in
        // dir(): (dz+1)x1 + (dy+1)x3 + (dx+1)x9.
        //
        // packed_tap = stencil offsets biased by +8, placed in the same groups:
        //   (dk+8) at bits [0:...]   dk+8 in [5,11] for dk in [-3,3]
        //   (dj+8) at bits [5:...]
        //   (di+8) at bits [10:...]
        //
        // The +8 bias shifts the zero point so that the per-group sum
        //   s = lc + (d+8),  lc in [0,7], d in [-3,3]  ->  s in [5,18]
        // encodes the neighbor coordinate measured from the (-1,-1,-1) leaf:
        //   s in [ 5, 7]: component + d <  0  -> lo-neighbor (d < 0 case)
        //   s in [ 8,15]: component + d in [0,7] -> center leaf
        //   s in [16,18]: component + d >=  8  -> hi-neighbor (d > 0 case)
        //
        // Carry bits after add:
        //   bit[+3] SET   <=>  s >=  8  (= no lo-crossing)
        //   bit[+4] SET   <=>  s >= 16  (= hi-crossing)
        //
        // For prefetch, only one bit per axis is needed (compile-time dispatch):
        //   dk > 0: z_cross = hor_or  & (1 << 4)   -- any lane has hi-z carry
        //   dk < 0: z_cross = !(hor_and & (1 << 3)) -- any lane lacks lo-z guard
        //   (same at bits [9]/[8] for y, bits [14]/[13] for x)
        //
        // Inactive lanes carry sentinel lc = 4 per axis: s = d+12 in [9,15]
        // -> bit[+3]=1, bit[+4]=0 -> no crossing signal regardless of d. (ok)
        //
        // For multi-axis taps, may-cross flags are combined conservatively.
        // -----------------------------------------------------------------------

        // Use VoxelOffsetT directly for the packed arithmetic: LaneWidth elements
        // of VoxelOffsetScalarT in one register -> one vpaddw (16-bit) or vpaddd
        // (32-bit) depending on the instantiation.  All intermediate values fit:
        //   packed_lc <= 0x1CE7, packed_tap <= 0x2D6B, sum <= 0x4A52 < 2^16.

        // Compile-time packed stencil offset (+8-biased per axis, 5-bit groups).
        static constexpr auto packed_tap =
            static_cast<VoxelOffsetScalarT>(
                 (unsigned(dk) + 8u)
               | ((unsigned(dj) + 8u) <<  5)
               | ((unsigned(di) + 8u) << 10));

        // Expand the 9-bit voxel offset into the 15-bit SWAR packed form.
        // vo = lx[6:8] | ly[3:5] | lz[0:2]  (NanoVDB leaf layout)
        // target: lx[10:12] | ly[5:7] | lz[0:2]
        //
        // (vo | (vo<<4)) & kSwarXZMask places lz (stays at [0:2]) and lx ([6:8]->[10:12])
        //   in one OR+mask; (vo<<2) & kSwarYMask moves ly ([3:5]->[5:7]).
        const auto expanded =
              ((vo | (vo << VoxelOffsetScalarT(4))) & VoxelOffsetT(kSwarXZMask))
            | ((vo << VoxelOffsetScalarT(2))        & VoxelOffsetT(kSwarYMask));

        // Blend: active lanes -> expanded form, straddle/inactive -> sentinel.
        auto packed_lc = VoxelOffsetT(kSwarSentinel);
        util::where(leafMask, packed_lc) = expanded;

        // One SIMD add across all LaneWidth lanes (one vpaddw/vpaddd instruction).
        const auto packed_sum = packed_lc + VoxelOffsetT(packed_tap);

        // Horizontal reductions for the carry-bit checks.
        const auto hor_or  = util::reduce(packed_sum, std::bit_or<>{});
        const auto hor_and = util::reduce(packed_sum, std::bit_and<>{});

        // Per-axis may-cross flags: compile-time dispatch on sign of d.
        // Overflow (d>0): detected by the hi-carry bit (+4 from group base).
        // Underflow (d<0): detected by absence of the lo-guard bit (+3).
        bool x_cross = false, y_cross = false, z_cross = false;
        if constexpr (dk > 0) z_cross =  bool(hor_or  & (1u <<  4));
        if constexpr (dk < 0) z_cross = !bool(hor_and & (1u <<  3));
        if constexpr (dj > 0) y_cross =  bool(hor_or  & (1u <<  9));
        if constexpr (dj < 0) y_cross = !bool(hor_and & (1u <<  8));
        if constexpr (di > 0) x_cross =  bool(hor_or  & (1u << 14));
        if constexpr (di < 0) x_cross = !bool(hor_and & (1u << 13));

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
        if (toProbe) {
            const auto& root = mGrid.tree().root();
            do {
                const int     d       = static_cast<int>(util::countTrailingZeros(toProbe));
                const LeafT*  leafPtr = root.probeLeaf(originForDir(d));
                mNeighborLeafIDs[d]   = leafPtr
                    ? uint32_t(leafPtr - mGrid.tree().getFirstLeaf())
                    : kNullLeafID;
                mProbedMask |= (1u << d);
                toProbe     &= toProbe - 1;
            } while (toProbe);
        }
    }

    // -------------------------------------------------------------------------
    // cachedGetValue<di,dj,dk> -- fill masked result lanes from cached leaf table
    //
    // For each active (leafMask) lane, computes the local voxel offset within the
    // appropriate neighbor leaf and calls leaf->getValue(offset).
    //
    // Requires prefetch<di,dj,dk> (or any prefetch covering the same directions)
    // to have been called first.
    //
    // A null leaf pointer (neighbor outside the narrow band) writes 0 to dst[lane].
    // Inactive lanes (bit lane of leafMask clear) are not touched.
    //
    // Output layout: `ScalarValueT (&dst)[LaneWidth]` — a plain C array, one
    // entry per SIMD lane.  This allows the scalar-tail loop below to write
    // lane results with a single `mov`, avoiding the heterogeneous-mask
    // where-blend that the old `Simd<ScalarValueT,W>&` signature triggered.
    //
    // Hybrid design (BatchAccessor.md §8h / StencilAccessor.md §8.1):
    //   SIMD portion stays in native uint16_t __m256i (no aggregate ABI):
    //     SWAR expansion, packed_sum, base-32 direction extract (d_u16),
    //     local-offset extract (localOffset_u16).
    //   Harvest: two YMM stores to stack C arrays (neighborIdx[], localOffset[]).
    //   Scalar tail: per-lane pointer chase into mNeighborLeafIDs + mFirstLeaf,
    //     one leaf.getValue(offset) call.  The LeafNode handles valueMask /
    //     prefixSum / popcount internally — one popcnt per lookup.
    // -------------------------------------------------------------------------
    template<int di, int dj, int dk>
    void cachedGetValue(ScalarValueT (&dst)[LaneWidth],
                        VoxelOffsetT   vo,
                        PredicateT     leafMask) const
    {
        // ---- SIMD portion (native __m256i uint16_t throughout — no aggregate ABI) ----
        static constexpr auto packed_tap =
            static_cast<VoxelOffsetScalarT>(
                 (unsigned(dk) + 8u)
               | ((unsigned(dj) + 8u) <<  5)
               | ((unsigned(di) + 8u) << 10));
        // SWAR expansion of the (x,y,z) local position of the center voxel.
        // Inactive-lane values are garbage; the scalar tail below filters them
        // out via the leafMask bitmask, so no sentinel / where-blend is needed.
        const auto expanded =
              ((vo | (vo << VoxelOffsetScalarT(4))) & VoxelOffsetT(kSwarXZMask))
            | ((vo << VoxelOffsetScalarT(2))        & VoxelOffsetT(kSwarYMask));
        const auto packed_sum = expanded + VoxelOffsetT(packed_tap);

        // Per-lane direction index (0..26) via the base-32 multiply trick (§8d).
        // Stays in uint16_t — bits [10:14] of (v * 1129) lie entirely below bit 16,
        // so the modular uint16_t product gives the same result as the full-width
        // product for all valid inputs.  No int32 widening → no _Fixed<W> aggregate.
        static constexpr uint16_t kSwarCarryMask = 0x6318u;
        static constexpr uint16_t kDirMul        = 1129u;
        static constexpr uint16_t kDirMask       = 31u;
        const auto d_u16 = (((packed_sum & VoxelOffsetT(kSwarCarryMask))
                                 >> VoxelOffsetScalarT(3))
                             * VoxelOffsetT(kDirMul)
                                 >> VoxelOffsetScalarT(10))
                            & VoxelOffsetT(kDirMask);

        // Per-lane 9-bit local offset in the destination leaf.
        // NanoVDB leaf layout: offset = (destX << 6) | (destY << 3) | destZ.
        //   packed_sum bits: destX=[10:12], destY=[5:7], destZ=[0:2]
        //   output   bits:   destX=[6:8],   destY=[3:5], destZ=[0:2]
        const auto localOffset_u16 =
              ((packed_sum >> VoxelOffsetScalarT(4)) & VoxelOffsetT(0x1C0u))
            | ((packed_sum >> VoxelOffsetScalarT(2)) & VoxelOffsetT(0x38u))
            |  (packed_sum                           & VoxelOffsetT(0x07u));

        // ---- Harvest SIMD → C arrays and scalar tail ----
        if constexpr (LaneWidth == 1) {
            if (!leafMask) return;   // inactive: leave dst[0] alone
            const uint32_t leafID = mNeighborLeafIDs[uint32_t(d_u16)];
            if (leafID == kNullLeafID) { dst[0] = ScalarValueT(0); return; }
            dst[0] = static_cast<ScalarValueT>(
                mFirstLeaf[leafID].getValue(uint32_t(localOffset_u16)));
        } else {
            alignas(32) uint16_t localOffset[LaneWidth];
            alignas(32) uint16_t neighborIdx[LaneWidth];
            util::store(localOffset_u16, localOffset);
            util::store(d_u16,           neighborIdx);

            // Convert SIMD leafMask → uint32_t bitmask once; then a single
            // scalar loop over active lanes with no further SIMD in sight.
            const uint32_t activeBits = util::to_bitmask(leafMask);
            for (int lane = 0; lane < LaneWidth; ++lane) {
                if (!((activeBits >> lane) & 1u)) continue;
                const uint32_t leafID = mNeighborLeafIDs[neighborIdx[lane]];
                if (leafID == kNullLeafID) {
                    dst[lane] = ScalarValueT(0);
                    continue;
                }
                dst[lane] = static_cast<ScalarValueT>(
                    mFirstLeaf[leafID].getValue(localOffset[lane]));
            }
        }
    }

    // -------------------------------------------------------------------------
    // cachedGetValueInLeaf<di,dj,dk> -- benchmarking variant that forces all
    // taps to stay in the center leaf via mod-8 wrap.
    //
    // Purpose: measure the hybrid pipeline's floor cost when all 18 taps
    // access the SAME leaf, with distinct per-tap / per-lane positions (so
    // the compiler can't CSE across taps, and we still exercise different
    // mValueMask words and prefix-sum slots).  The result is semantically
    //   target_local = (voxel_local + (di,dj,dk)) mod 8
    // with target always in the center leaf (direction code 0).
    //
    // Implementation: same SWAR + harvest + scalar-tail pipeline as
    // cachedGetValue, but after `packed_sum = expanded + packed_tap` we mask
    // with kSwarFieldMask = 0x1CE7 to discard all inter-field carry bits,
    // which is exactly `x mod 8 | y mod 8 | z mod 8` in the packed layout.
    //
    // Requires di, dj, dk in [0, 7].  No prefetch call needed; the center
    // leaf is always in mNeighborLeafIDs[13] from construction/advance.
    // -------------------------------------------------------------------------
    template<int di, int dj, int dk>
    void cachedGetValueInLeaf(ScalarValueT (&dst)[LaneWidth],
                              VoxelOffsetT   vo,
                              PredicateT     leafMask) const
    {
        static_assert(di >= 0 && di < 8 && dj >= 0 && dj < 8 && dk >= 0 && dk < 8,
            "cachedGetValueInLeaf: tap offsets must be in [0, 7] per axis");

        static constexpr auto packed_tap =
            static_cast<VoxelOffsetScalarT>(
                 unsigned(dk)
               | (unsigned(dj) <<  5)
               | (unsigned(di) << 10));
        const auto expanded =
              ((vo | (vo << VoxelOffsetScalarT(4))) & VoxelOffsetT(kSwarXZMask))
            | ((vo << VoxelOffsetScalarT(2))        & VoxelOffsetT(kSwarYMask));
        // Mask off inter-field carry bits → per-axis mod-8 wrap; always center.
        static constexpr uint16_t kSwarFieldMask = 0x1CE7u;
        const auto packed_sum =
            (expanded + VoxelOffsetT(packed_tap)) & VoxelOffsetT(kSwarFieldMask);

        // Extract 9-bit local offset (same layout as cachedGetValue).
        const auto localOffset_u16 =
              ((packed_sum >> VoxelOffsetScalarT(4)) & VoxelOffsetT(0x1C0u))
            | ((packed_sum >> VoxelOffsetScalarT(2)) & VoxelOffsetT(0x38u))
            |  (packed_sum                           & VoxelOffsetT(0x07u));

        if constexpr (LaneWidth == 1) {
            if (!leafMask) return;
            dst[0] = static_cast<ScalarValueT>(
                mFirstLeaf[mCenterLeafID].getValue(uint32_t(localOffset_u16)));
        } else {
            alignas(32) uint16_t localOffset[LaneWidth];
            util::store(localOffset_u16, localOffset);
            const uint32_t activeBits = util::to_bitmask(leafMask);
            const LeafT* const leaf = &mFirstLeaf[mCenterLeafID];  // hoisted
            for (int lane = 0; lane < LaneWidth; ++lane) {
                if (!((activeBits >> lane) & 1u)) continue;
                dst[lane] = static_cast<ScalarValueT>(
                    leaf->getValue(localOffset[lane]));
            }
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

    const GridT&          mGrid;
    uint32_t              mCenterLeafID;
    Coord                 mCenterOrigin;
    uint32_t              mProbedMask;
    uint32_t              mNeighborLeafIDs[27]; // kNullLeafID when not probed or outside narrow band
    const LeafT* const    mFirstLeaf;           // getFirstLeaf() — scalar-tail leaf lookup base
    const uint64_t* const mOffsetBase;           // &getFirstLeaf()[0].data()->mOffset
    const uint64_t* const mPrefixBase;           // &getFirstLeaf()[0].data()->mPrefixSum
    const uint64_t* const mMaskWordBase;         // getFirstLeaf()[0].valueMask().words()
};

} // namespace nanovdb
