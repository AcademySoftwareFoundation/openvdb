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
    LeafIDT       Leaf index type -- reserved for future use by the caller loop.
                  Scalar: uint32_t.  SIMD: Simd<uint32_t,W>.
    PredicateT    Per-lane active predicate (the leafMask).
                  Scalar: bool.  SIMD: SimdMask<float,W> or similar.

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
#include <simd_test/Simd.h>   // simd_traits, scalar_traits, Simd<T,W>, SimdMask<T,W>
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
    {
        for (auto& id : mNeighborLeafIDs) id = kNullLeafID;
        mNeighborLeafIDs[dir(0, 0, 0)] = mCenterLeafID;
    }

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
    // to have been called first.  Debug builds assert mProbedMask coverage.
    //
    // A null leaf pointer (neighbor outside the narrow band) writes 0 to result.
    // Inactive lanes (leafMask[i] == false) are not touched.
    // -------------------------------------------------------------------------
    template<int di, int dj, int dk>
    void cachedGetValue(ValueT& result, VoxelOffsetT vo, PredicateT leafMask) const
    {
        // -----------------------------------------------------------------------
        // SIMD ingredient fetch (WIP -- not yet wired to result)
        //
        // Recomputes packed_sum (same SWAR expansion as prefetch) to extract the
        // three per-lane ingredients needed to replace leaf->getValue() with fully
        // SIMD index arithmetic + value gather.  See BatchAccessor.md Sec.8d.
        //
        //   offsets    -- leaf->mOffset:      base value index for the leaf
        //   prefixSums -- leaf->mPrefixSum[w]: prefix popcount up to x-slice w
        //   maskWords  -- leaf->mMask.mWords[w]: uint64_t mask for x-slice w
        //
        // w = dest_x = bits [10:12] of packed_sum (NanoVDB leaf layout: x is
        // the most significant axis, so x-slices index the eight uint64_t words).
        //
        // dir per lane is extracted via the base-32 multiply trick (Sec.8d):
        //   v = (packed_sum & 0x6318u) >> 3
        //   dir = (v * 1129u) >> 10
        //
        // Note: exact field names (mOffset, mPrefixSum, mMask.mWords) need
        // verification against LeafData<ValueOnIndex, 3, false> in NanoVDB.h.
        // -----------------------------------------------------------------------
        {
            static constexpr auto packed_tap =
                static_cast<VoxelOffsetScalarT>(
                     (unsigned(dk) + 8u)
                   | ((unsigned(dj) + 8u) <<  5)
                   | ((unsigned(di) + 8u) << 10));
            const auto expanded =
                  ((vo | (vo << VoxelOffsetScalarT(4))) & VoxelOffsetT(kSwarXZMask))
                | ((vo << VoxelOffsetScalarT(2))        & VoxelOffsetT(kSwarYMask));

            auto packed_lc = VoxelOffsetT(kSwarSentinel);
            util::where(leafMask, packed_lc) = expanded;
            const auto packed_sum = packed_lc + VoxelOffsetT(packed_tap);

            // dest_x per lane: bits [10:12] of packed_sum -> uint64_t mask word index (0..7)
            const auto wordIndex = (packed_sum >> VoxelOffsetScalarT(10)) & VoxelOffsetT(7u);

            // SIMD gather of mOffset, mPrefixSum[w], and maskWords[w] per lane.
            //
            // Step 1 -- d_vec: per-lane dir (0..26) via base-32 multiply trick (Sec.8d).
            //   No widening needed: we extract bits [10:14] of (v * 1129).  Those
            //   bits lie entirely below bit 16, so the modular uint16_t product gives
            //   the same answer as the full-width product for all valid + sentinel inputs.
            //
            // Step 2 -- leaf_id_vec: gather mNeighborLeafIDs[d] for all lanes at once.
            //
            // Step 3 -- raw_idx: leaf_id * (sizeof(LeafT)/sizeof(uint64_t)).
            //   This is the per-lane uint64_t-stride index into the flat leaf array,
            //   viewed as a uint64_t[] through the base pointer of the target field.
            //   Invalid (kNullLeafID) lanes are clamped to index 0 (safe; masked out).
            //
            // Step 4 -- offsets / prefixSums: two gathers with different base pointers
            //   but the same raw_idx; masked to 0 for null lanes.
            //   mPrefixSum is a packed uint64_t: field w lives at bits [9*(w-1)+:9]
            //   (9-bit fields, w=0 -> prefix = 0 by definition).
            //
            // Step 5 -- maskWords: gather from valueMask().words() base.
            //   words()[wi] for leaf[leaf_id] = mask_word_base[leaf_id*kStride + wi].
            //   The per-lane wi is added to raw_idx to form the mask gather index.
            using U32T = util::Simd<uint32_t, LaneWidth>;
            using U64T = util::Simd<uint64_t, LaneWidth>;

            // Direction-extraction constants (base-32 multiply trick, Sec.8d).
            static constexpr uint16_t kSwarCarryMask = 0x6318u; // carry bits [3:4],[8:9],[13:14]
            static constexpr uint16_t kDirMul        = 1129u;   // base-32 multiplier: 1*32^2 + 3*32 + 9
            static constexpr uint16_t kDirMask       = 31u;     // 5-bit digit mask

            // Step 1 -- d_vec: per-lane dir (0..26) via base-32 multiply (Sec.8d).
            // Stay in uint16_t throughout: bits [10:14] of (v * 1129) are entirely
            // within the lower 16 bits, so the modular uint16_t product gives the
            // same result as the full-width product for all valid inputs.
            const auto d_u16 = (((packed_sum & VoxelOffsetT(kSwarCarryMask))
                                     >> VoxelOffsetScalarT(3))
                                 * VoxelOffsetT(kDirMul)
                                     >> VoxelOffsetScalarT(10))
                                & VoxelOffsetT(kDirMask);
            const auto d_i32 = util::simd_cast<int32_t>(d_u16);

            // Step 2 -- leaf IDs
            const auto leaf_id_vec = util::gather(mNeighborLeafIDs, d_i32);  // Simd<uint32_t,W>
            const auto valid_u32   = (leaf_id_vec != U32T(kNullLeafID));     // SimdMask<uint32_t,W>

            // Step 3 -- stride-scaled gather indices (null lanes -> 0)
            static constexpr uint32_t kStride = sizeof(LeafT) / sizeof(uint64_t);
            const auto raw_idx = util::simd_cast<int32_t>(
                util::where(valid_u32, leaf_id_vec * U32T(kStride), U32T(0)));

            // Step 4a -- offsets (mOffset)
            const uint64_t* offset_base = reinterpret_cast<const uint64_t*>(
                &mGrid.tree().getFirstLeaf()[0].data()->mOffset);
            const U64T offsets = util::where(valid_u32,
                util::gather(offset_base, raw_idx), U64T(0));

            // Step 4b -- prefixSums (mPrefixSum packed uint64_t, shift-extract field w)
            const uint64_t* prefix_base = reinterpret_cast<const uint64_t*>(
                &mGrid.tree().getFirstLeaf()[0].data()->mPrefixSum);
            const auto prefix_raw = util::gather(prefix_base, raw_idx);
            const auto w_u64      = util::simd_cast<uint64_t>(wordIndex);
            const auto nonzero_w  = (w_u64 != U64T(0));
            const auto shift      = util::where(nonzero_w, (w_u64 - U64T(1)) * U64T(9), U64T(0));
            const U64T prefixSums = util::where(valid_u32,
                util::where(nonzero_w, (prefix_raw >> shift) & U64T(511u), U64T(0)),
                U64T(0));

            // Step 5 -- maskWords (valueMask().words()[w])
            //   mask_word_base[leaf_id*kStride + w] == leaf[leaf_id].valueMask().words()[w]
            //   because the mask field is at a fixed offsetof within every LeafT.
            const uint64_t* mask_word_base =
                mGrid.tree().getFirstLeaf()[0].valueMask().words();
            const auto w_i32     = util::simd_cast<int32_t>(util::simd_cast<uint32_t>(wordIndex));
            const auto mask_idx  = raw_idx + w_i32;
            const U64T maskWords = util::where(valid_u32,
                util::gather(mask_word_base, mask_idx), U64T(0));
            // -------------------------------------------------------------------
            // Debug cross-check: validate SIMD-path values against scalar ref
            // -------------------------------------------------------------------
#ifndef NDEBUG
            using U64Traits = util::simd_traits<U64T>;
            for (int i = 0; i < LaneWidth; ++i) {
                if (!Pred_traits::get(leafMask, i)) continue;

                // Scalar reference: same arithmetic as the legacy loop below
                const auto vo_i = static_cast<uint16_t>(VO_traits::get(vo, i));
                const int lx = (vo_i >> 6) & 7, ly = (vo_i >> 3) & 7, lz = vo_i & 7;
                const int nx = lx + di, ny = ly + dj, nz = lz + dk;
                const int dx = (nx < 0) ? -1 : (nx >= 8) ? 1 : 0;
                const int dy = (ny < 0) ? -1 : (ny >= 8) ? 1 : 0;
                const int dz = (nz < 0) ? -1 : (nz >= 8) ? 1 : 0;
                const int d_ref  = dir(dx, dy, dz);
                const int nx_w   = nx - dx * 8;    // = dest_x = word index w
                const uint32_t ref_id = mNeighborLeafIDs[d_ref];
                const LeafT*   ref    = (ref_id != kNullLeafID)
                    ? &mGrid.tree().getFirstLeaf()[ref_id] : nullptr;

                // SIMD-path values for this lane
                const uint32_t ps_i   = static_cast<uint32_t>(VO_traits::get(packed_sum, i));
                const int      d_simd = int((((ps_i & 0x6318u) >> 3) * 1129u >> 10) & 31u);
                const int      wi     = int(VO_traits::get(wordIndex, i));

                assert(d_simd == d_ref && "cachedGetValue SIMD: dir mismatch");
                assert(wi == nx_w      && "cachedGetValue SIMD: w (dest_x) mismatch");

                if (ref) {
                    const uint64_t pfx_ref = (uint32_t(nx_w) > 0u)
                        ? (ref->data()->mPrefixSum >> (9u * (uint32_t(nx_w) - 1u))) & 511u
                        : uint64_t(0);
                    assert(U64Traits::get(offsets,    i) == ref->data()->mOffset
                           && "cachedGetValue SIMD: mOffset mismatch");
                    assert(U64Traits::get(prefixSums, i) == pfx_ref
                           && "cachedGetValue SIMD: mPrefixSum mismatch");
                    assert(U64Traits::get(maskWords,  i) == ref->valueMask().words()[nx_w]
                           && "cachedGetValue SIMD: maskWord mismatch");
                } else {
                    assert(U64Traits::get(offsets,    i) == uint64_t(0)
                           && "cachedGetValue SIMD: null leaf offsets should be 0");
                    assert(U64Traits::get(prefixSums, i) == uint64_t(0)
                           && "cachedGetValue SIMD: null leaf prefixSums should be 0");
                    assert(U64Traits::get(maskWords,  i) == uint64_t(0)
                           && "cachedGetValue SIMD: null leaf maskWords should be 0");
                }
            }
#endif
            (void)offsets; (void)prefixSums; (void)maskWords;
        }

        // -----------------------------------------------------------------------
        // Legacy scalar path -- authoritative until SIMD path is wired in
        // -----------------------------------------------------------------------
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
            const int      d       = dir(dx, dy, dz);
            assert((mProbedMask & (1u << d)) && "cachedGetValue: direction not prefetched");
            const uint32_t leaf_id = mNeighborLeafIDs[d];
            const LeafT*   leaf    = (leaf_id != kNullLeafID)
                ? &mGrid.tree().getFirstLeaf()[leaf_id] : nullptr;
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
    uint32_t     mNeighborLeafIDs[27]; // kNullLeafID when not probed or outside narrow band
};

} // namespace nanovdb
