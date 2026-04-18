// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file stencil_gather_cpu.cpp

    \brief Prototype for CPU SIMD stencil gather — Phase 1 only:
           neighbor leaf resolution with lazy probeLeaf and per-leaf probedMask cache.

    Design documented in:
      nanovdb/examples/ex_voxelBlockManager_host_cuda/StencilGather.md

    What this prototype does (and does NOT do):
      - Generates a random Morton-layout domain (same as vbm_host_cuda).
      - Builds a ValueOnIndex NanoVDB grid and a VoxelBlockManager.
      - For every block: calls decodeInverseMaps, then processes SIMD batches of
        SIMDw=16 lanes, running the full probedMask / probeLeaf / batchPtrs population
        pipeline described in StencilGather.md §8d–§8f.
      - Does NOT call computeStencil.  Instead, verifies that batchPtrs[4][SIMDw]
        is correct for every active lane: for each of the 18 non-center WENO5
        stencil offsets, if the offset crosses a leaf boundary the corresponding
        batchPtrs[axis+1][lane] is checked against a direct probeLeaf reference.

    Build:
      Configured via CMakeLists.txt in the parent examples/ directory.
      No CUDA required; CPU-only.

    Usage: stencil_gather_cpu [ambient_voxels [occupancy]]
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/util/ForEach.h>
#include <simd_test/Simd.h>   // SimdMask<T,W>, Simd<T,W>, any_of, none_of, to_bitmask
#include "../ex_voxelBlockManager_host_cuda/BatchAccessor.h"    // BatchAccessor
#include "../ex_voxelBlockManager_host_cuda/StencilAccessor.h"  // StencilAccessor, Weno5Stencil

#include <x86intrin.h>   // __rdtsc, __rdtscp, _mm_lfence

#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cassert>

// ============================================================
// Constants and type aliases
// ============================================================

static constexpr int Log2BlockWidth = 7;
static constexpr int BlockWidth     = 1 << Log2BlockWidth;  // 128
static constexpr int SIMDw          = 16;                   // batch width
static constexpr int R              = 3;                    // WENO5 stencil reach (±3)

using BuildT = nanovdb::ValueOnIndex;
using GridT  = nanovdb::NanoGrid<BuildT>;
using LeafT  = nanovdb::NanoLeaf<BuildT>;
using CPUVBM = nanovdb::tools::VoxelBlockManager<Log2BlockWidth>;
using AccT   = nanovdb::DefaultReadAccessor<BuildT>;

// Lane-predicate types: SIMDw-wide boolean mask and the uint32_t vector it compares.
using LeafIdxVec = nanovdb::util::Simd<uint32_t, SIMDw>;
using LaneMask   = nanovdb::util::SimdMask<uint32_t, SIMDw>;

// BatchAccessor instantiation for correctness cross-validation.
// ValueT = Simd<uint64_t,SIMDw> because ValueOnIndex leaf values are uint64_t active indices.
using BAccT = nanovdb::BatchAccessor<BuildT,
    nanovdb::util::Simd<uint64_t, SIMDw>,   // ValueT
    nanovdb::util::Simd<uint16_t, SIMDw>,   // VoxelOffsetT
    LaneMask>;                               // PredicateT

// StencilAccessor instantiation for WENO5.
using SAccT       = nanovdb::StencilAccessor<BuildT, SIMDw, nanovdb::Weno5Stencil>;
// Return type of StencilAccessor::moveTo (mask over the uint64_t index domain).
using IndexMaskT  = nanovdb::util::SimdMask<uint64_t, SIMDw>;

// Direction bit encoding shared across all stencil types:
//   bit(dx, dy, dz) = (dx+1)*9 + (dy+1)*3 + (dz+1),   dx,dy,dz ∈ {-1,0,+1}
//
// WENO5 face-neighbor bits (the only 6 bits ever set for WENO5):
static constexpr int kLoBit[3] = {4,  10, 12};  // x-lo, y-lo, z-lo
static constexpr int kHiBit[3] = {22, 16, 14};  // x-hi, y-hi, z-hi

// ============================================================
// Test domain generation (mirrors vbm_host_cuda.cpp)
// ============================================================

static uint32_t coordinate_bitpack(uint32_t x)
{
    x &= 0x49249249;
    x |= (x >>  2); x &= 0xc30c30c3;
    x |= (x >>  4); x &= 0x0f00f00f;
    x |= (x >>  8); x &= 0xff0000ff;
    x |= (x >> 16); x &= 0x0000ffff;
    return x;
}

static std::vector<nanovdb::Coord>
generateDomain(int ambient_voxels, float occupancy, uint32_t seed = 42)
{
    const int target = (int)(occupancy * (float)ambient_voxels);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, ambient_voxels - 1);
    std::vector<bool> voxmap(ambient_voxels, false);
    int active = 0;
    while (active < target) {
        int i = dist(rng);
        if (!voxmap[i]) { voxmap[i] = true; ++active; }
    }
    std::vector<nanovdb::Coord> coords;
    coords.reserve(active);
    for (int i = 0; i < ambient_voxels; ++i) {
        if (voxmap[i]) {
            coords.emplace_back(
                (int)coordinate_bitpack( i        & 0x49249249),
                (int)coordinate_bitpack((i >>  1) & 0x49249249),
                (int)coordinate_bitpack((i >>  2) & 0x49249249));
        }
    }
    return coords;
}

// ============================================================
// Neighbor direction utilities (§6a)
// ============================================================

/// @brief Return the origin of the neighbor leaf at direction bit d from center.
/// bit(dx,dy,dz) = (dx+1)*9 + (dy+1)*3 + (dz+1); leaf stride = 8 per axis.
static inline nanovdb::Coord neighborLeafOrigin(const nanovdb::Coord& center, int bit)
{
    const int dx = bit / 9 - 1;
    const int dy = (bit / 3) % 3 - 1;
    const int dz = bit % 3 - 1;
    return center + nanovdb::Coord(dx * 8, dy * 8, dz * 8);
}

/// @brief Extract the local axis coordinate from a voxelOffset.
/// NanoVDB leaf layout: offset = lx*64 + ly*8 + lz.
///   axis 0 (x): bits [8:6] → shift 6
///   axis 1 (y): bits [5:3] → shift 3
///   axis 2 (z): bits [2:0] → shift 0
static inline int localAxisCoord(uint16_t vo, int axis)
{
    return (vo >> (6 - 3 * axis)) & 7;
}

// ============================================================
// Vectorized computeNeededDirs — shift-OR carry trick (§8e)
//
// Determines which of the 6 face-neighbor directions (±x, ±y, ±z) are required
// by any active lane in a SIMDw-wide batch.  For WENO5 (R=3):
//   plus-c  neighbor needed  iff any active lane has local coordinate lc ≥ 8−R = 5
//   minus-c neighbor needed  iff any active lane has local coordinate lc ≤   R−1 = 2
//
// Algorithm — "expand, add, reduce":
//   1. expandVoxelOffset(): pack lz, lx, ly (and second copies) into a 32-bit
//      integer with 3-bit zero guards between groups.  Each group occupies its
//      own 3-bit field; the guards absorb carries so adjacent groups do not bleed.
//   2. Add kExpandCarryK to all SIMDw lane values simultaneously (one SIMD add).
//      Groups 1–3 detect plus-directions; groups 4–6 detect minus-directions.
//   3. Horizontal OR  across all lanes: carry bit SET   → plus-direction needed.
//      Horizontal AND across all lanes: carry bit CLEAR → minus-direction needed.
//      (A minus-direction is needed when at least one lane has NO carry, i.e.,
//       lc < R; the AND bit is clear iff any lane failed to carry.)
//
// ============================================================

/// @brief voxelOffset sentinel for inactive / don't-care SIMD lanes.
///
/// Any lane with laneMask[i] = false needs a voxelOffset value that:
///   • does NOT set carry bits 3, 9, 15  (would wrongly assert plus-directions), AND
///   • DOES set carry bits 19, 25, 31    (a clear bit would wrongly assert minus-directions).
///
/// Local coordinate (4, 4, 4) satisfies both: R ≤ 4 < 8−R for R=3 (strictly interior).
///   voxelOffset(4,4,4) = 4*64 + 4*8 + 4 = 292
///   group 1-3: 4 + 3 = 7  → no carry  → bits 3, 9, 15 stay clear  ✓
///   group 4-6: 4 + 5 = 9  → carry     → bits 19, 25, 31 stay set   ✓
///
static constexpr uint16_t kInactiveVoxelOffset = (4u << 6) | (4u << 3) | 4u;  // = 292

/// @brief Expand a 9-bit voxelOffset into a 32-bit "carry lane" layout.
///
/// NanoVDB voxelOffset bit layout: [8:6] = lx,  [5:3] = ly,  [2:0] = lz.
///
/// Target 32-bit layout — 6 groups of 3 bits with zero-guard separators:
///
///   bits  0– 2 : lz     ← group 1 (plus-z  carry exits at bit  3)
///   bits  3– 5 : 0      (3-bit guard)
///   bits  6– 8 : lx     ← group 2 (plus-x  carry exits at bit  9)
///   bits  9–11 : 0      (3-bit guard)
///   bits 12–14 : ly     ← group 3 (plus-y  carry exits at bit 15)
///   bit  15    : 0      (1-bit guard — sufficient because max carry from a
///                        3-bit field added to a constant < 8 is exactly 1 bit;
///                        at bit 15: input=0, addend=0, carry-in∈{0,1} → no further carry)
///   bits 16–18 : lz     ← group 4 (minus-z carry exits at bit 19)
///   bits 19–21 : 0      (3-bit guard)
///   bits 22–24 : lx     ← group 5 (minus-x carry exits at bit 25)
///   bits 25–27 : 0      (3-bit guard)
///   bits 28–30 : ly     ← group 6 (minus-y carry exits at bit 31)
///   bit  31    : 0      (receives minus-y carry; bit 31 is within uint32_t range)
///
/// Construction — three shift-OR steps, no multiply:
///
///   Step 1: e |= (e << 9)    → 0o xyzxyz  (two 9-bit copies stacked, 18 bits)
///   Step 2: e &= 0x71C7      → keep lz@[0:2], lx@[6:8], ly@[12:14]; zero all others.
///                               0x71C7 = 0b 0111 0001 1100 0111 = 0o 070707
///                               Bits set: {0,1,2, 6,7,8, 12,13,14}
///   Step 3: e |= (e << 16)   → copy the 15-bit pattern to bits [16:18],[22:24],[28:30]
///
static inline constexpr uint32_t expandVoxelOffset(uint16_t vo)
{
    uint32_t e = vo;
    e |= (e <<  9);   // step 1: two packed xyz copies at 9-bit stride
    e &= 0x71C7u;     // step 2: isolate lz@[0:2], lx@[6:8], ly@[12:14] with zero gaps
    e |= (e << 16);   // step 3: second copy to bits [16:18], [22:24], [28:30]
    return e;
}

/// @brief Combined carry-detection constant (added to expandVoxelOffset results).
///
/// Groups 1–3 receive +R   so a 3-bit field ≥ (8−R) produces a carry (plus-direction test).
/// Groups 4–6 receive +(8−R) so a 3-bit field ≥ R   produces a carry (minus-direction test:
///   carry CLEAR ⟺ field < R ⟺ minus-direction needed).
///
///               group1     group2      group3       group4        group5        group6
///   K  =  R   | R<<6    | R<<12    | (8-R)<<16 | (8-R)<<22 | (8-R)<<28
///      =  3   | 192     | 12288    | 327680    | 20971520  | 1342177280
///      =  1,363,488,963  (0x514530C3)  — fits in uint32_t (< 2^32).
///
/// Carry bits produced by expanded + K:
///   bit  3  set   ↔  lz ≥ 8−R  → plus-z  needed
///   bit  9  set   ↔  lx ≥ 8−R  → plus-x  needed
///   bit 15  set   ↔  ly ≥ 8−R  → plus-y  needed
///   bit 19  clear ↔  lz <   R  → minus-z needed
///   bit 25  clear ↔  lx <   R  → minus-x needed
///   bit 31  clear ↔  ly <   R  → minus-y needed
///
static constexpr uint32_t kExpandCarryK =
    ((uint32_t)R      )         |   // bits  0– 2: +R   → lz plus-z  group
    ((uint32_t)R      <<  6)    |   // bits  6– 8: +R   → lx plus-x  group
    ((uint32_t)R      << 12)    |   // bits 12–14: +R   → ly plus-y  group
    ((uint32_t)(8-R)  << 16)    |   // bits 16–18: +5   → lz minus-z group
    ((uint32_t)(8-R)  << 22)    |   // bits 22–24: +5   → lx minus-x group
    ((uint32_t)(8-R)  << 28);       // bits 28–30: +5   → ly minus-y group

/// @brief Pre-expanded sentinel value for inactive / straddle SIMD lanes.
///
/// Caller broadcasts this to all lanes before overwriting the leafMask lanes
/// with the real expandVoxelOffset() values.  Equivalent to
///   expandVoxelOffset(kInactiveVoxelOffset)
/// which, at compile time, is 0x41044104.
static constexpr uint32_t kSentinelExpanded = expandVoxelOffset(kInactiveVoxelOffset);

/// @brief Scalar reference implementation (lane-by-lane loop).
/// Kept alongside the SIMD version so debug builds can cross-check.
static uint32_t computeNeededDirsScalar(const uint16_t* voxelOffset,
                                         int             batchStart,
                                         LaneMask        laneMask)
{
    uint32_t needed = 0;
    for (int i = 0; i < SIMDw; i++) {
        if (!laneMask[i]) continue;
        const uint16_t vo = voxelOffset[batchStart + i];
        for (int axis = 0; axis < 3; axis++) {
            const int lc = localAxisCoord(vo, axis);
            if (lc < R)    needed |= (1u << kLoBit[axis]);
            if (lc >= 8-R) needed |= (1u << kHiBit[axis]);
        }
    }
    return needed;
}

/// @brief Vectorized computeNeededDirs — shift-OR carry trick.
///
/// Returns the 27-bit probedMask subset identifying which of the 6 WENO5
/// face-neighbor directions are required by any active lane.
///
/// For WENO5 (R=3) only the 6 face-direction bits {4,10,12,14,16,22} can
/// ever be set; the 21 edge/corner bits remain zero.
///
/// @param expandedVec  SIMDw pre-expanded voxelOffset values (see expandVoxelOffset).
///   Caller is responsible for:
///     • Broadcasting kSentinelExpanded to all lanes first.
///     • Overwriting leafMask lanes with expandVoxelOffset(voxelOffset[...]).
///   This keeps sentinel / masking logic at the single gather site where leafMask
///   is known, not buried inside this function.
///
/// High-level flow:
///   1. Single SIMD add of kExpandCarryK (caller already expanded each lane).
///   2. Horizontal OR  of all results → carry SET   = plus-direction needed.
///      Horizontal AND of all results → carry CLEAR = minus-direction needed.
///   3. Map carry bits to the 27-bit probedMask encoding.
///
static uint32_t computeNeededDirs(nanovdb::util::Simd<uint32_t, SIMDw> expandedVec)
{
    using VecU32 = nanovdb::util::Simd<uint32_t, SIMDw>;

    // --- Single SIMD add --------------------------------------------------
    // Inject carry-detection thresholds for all 6 groups simultaneously.
    // After this add, each lane's result[i] encodes all six direction tests
    // as carry bits at positions 3, 9, 15 (plus) and 19, 25, 31 (minus).
    const VecU32 result = expandedVec + VecU32(kExpandCarryK);

    // --- Horizontal reductions --------------------------------------------
    //
    // hor_or:  bit k is set iff at least one lane has bit k set in result.
    //   → Check carry bits 3 (z), 9 (x), 15 (y): SET means plus-direction needed.
    //
    // hor_and: bit k is set iff every lane has bit k set in result.
    //   → Check carry bits 19 (z), 25 (x), 31 (y): CLEAR means minus-direction
    //     needed (at least one lane did not carry, i.e., its coordinate < R).
    //
    uint32_t hor_or = 0u, hor_and = ~0u;
    for (int i = 0; i < SIMDw; i++) {
        hor_or  |= result[i];
        hor_and &= result[i];
    }

    // --- Map carry bits → probedMask direction bits -----------------------
    //
    // Plus carries  (bits  3,  9, 15) set   → kHiBit (hi-side neighbor needed).
    // Minus carries (bits 19, 25, 31) clear → kLoBit (lo-side neighbor needed).
    //
    //  carry bit | axis | condition  | probedMask bit
    //  ----------+------+------------+---------------
    //      3     |  z   | lz ≥ 8−R  | kHiBit[2] = 14
    //      9     |  x   | lx ≥ 8−R  | kHiBit[0] = 22
    //     15     |  y   | ly ≥ 8−R  | kHiBit[1] = 16
    //     19 clr |  z   | lz <   R  | kLoBit[2] = 12
    //     25 clr |  x   | lx <   R  | kLoBit[0] =  4
    //     31 clr |  y   | ly <   R  | kLoBit[1] = 10
    //
    uint32_t needed = 0;
    if ( hor_or  & (1u <<  3))   needed |= (1u << kHiBit[2]);  // plus-z
    if ( hor_or  & (1u <<  9))   needed |= (1u << kHiBit[0]);  // plus-x
    if ( hor_or  & (1u << 15))   needed |= (1u << kHiBit[1]);  // plus-y
    if (!(hor_and & (1u << 19))) needed |= (1u << kLoBit[2]);  // minus-z
    if (!(hor_and & (1u << 25))) needed |= (1u << kLoBit[0]);  // minus-x
    if (!(hor_and & (1u << 31))) needed |= (1u << kLoBit[1]);  // minus-y

    return needed;
}

// ============================================================
// Targeted sentinel correctness test (§8e supplement)
//
// Verifies that inactive lanes — including straddle lanes that ARE active
// voxels but belong to a different leaf — do not inject spurious direction
// bits into the SIMD result.
//
// The test is designed so that a broken sentinel (i.e., using the straddle
// lane's real voxelOffset instead of kInactiveVoxelOffset) would produce a
// DIFFERENT result from the scalar reference in BOTH the plus and minus
// directions, making the bug impossible to miss.
//
// Layout (SIMDw = 16 lanes):
//   leafMask lanes (even: 0,2,4,...,14):
//     lx=4 (neutral for x), ly=4 (neutral for y), lz=6 (→ plus-z needed)
//     voxelOffset = 4*64 + 4*8 + 6 = 294
//
//   straddle lanes (odd: 1,3,5,...,15) — active voxels, wrong leaf:
//     lx=0 (→ minus-x if used), ly=7 (→ plus-y if used), lz=1 (→ minus-z if used)
//     voxelOffset = 0*64 + 7*8 + 1 = 57
//
// Expected result (scalar — straddle lanes ignored):
//   plus-z  needed  (bit kHiBit[2]=14): lz=6 ≥ 5 in leafMask lanes   ✓
//   minus-x NOT needed: lx=4 ≥ R=3 for all leafMask lanes             ✓
//   plus-y  NOT needed: ly=4 < 8-R=5 for all leafMask lanes           ✓
//   minus-z NOT needed: lz=6 ≥ R=3 for all leafMask lanes             ✓
//   plus-x  NOT needed: lx=4 < 8-R=5 for all leafMask lanes           ✓
//   minus-y NOT needed: ly=4 ≥ R=3 for all leafMask lanes             ✓
//
// If sentinel fails: straddle lx=0 → minus-x spuriously added;
//                    straddle ly=7 → plus-y spuriously added;
//                    straddle lz=1 → minus-z spuriously added.
// Those discrepancies are caught by the scalar cross-check inside
// computeNeededDirs, which will abort immediately.
// ============================================================

static void verifyComputeNeededDirsSentinel()
{
    // --- Sentinel property: expandVoxelOffset(292) + K must have ---
    // --- plus-carry bits {3,9,15} clear and minus-carry bits {19,25,31} set ---
    {
        const uint32_t expanded = expandVoxelOffset(kInactiveVoxelOffset);
        const uint32_t result   = expanded + kExpandCarryK;
        const bool plus_ok  = !(result & ((1u<<3)|(1u<<9)|(1u<<15)));
        const bool minus_ok =  (result & ((1u<<19)|(1u<<25)|(1u<<31))) ==
                                         ((1u<<19)|(1u<<25)|(1u<<31));
        if (!plus_ok || !minus_ok) {
            std::cerr << "verifyComputeNeededDirsSentinel: sentinel carry property violated"
                      << "  expanded=0x" << std::hex << expanded
                      << "  result=0x"   << result << std::dec << "\n";
            std::abort();
        }
    }

    // --- Straddle scenario: straddle lanes must not pollute the result ---
    alignas(64) uint16_t voxelOffset[BlockWidth] = {};

    // leafMask lanes (even): lx=4, ly=4, lz=6  →  voxelOffset = 4*64+4*8+6 = 294
    // straddle lanes (odd):  lx=0, ly=7, lz=1  →  voxelOffset = 0*64+7*8+1 = 57
    LaneMask laneMask;
    for (int i = 0; i < SIMDw; i++) {
        const bool active  = (i % 2 == 0);
        laneMask[i]        = active;
        voxelOffset[i]     = active ? uint16_t(294) : uint16_t(57);
    }

    // Expected: only plus-z (kHiBit[2] = 14) should be set.
    //
    // Build the pre-expanded vector exactly as the gather site would.
    using VecU32 = nanovdb::util::Simd<uint32_t, SIMDw>;
    VecU32 expandedVec(kSentinelExpanded);
    for (int i = 0; i < SIMDw; i++) {
        if (laneMask[i]) expandedVec[i] = expandVoxelOffset(voxelOffset[i]);
    }
    const uint32_t result   = computeNeededDirs(expandedVec);

    // Explicit cross-check: scalar reference (SIMD cross-check no longer lives inside
    // computeNeededDirs — it is the caller's responsibility at each gather site).
    {
        const uint32_t ref = computeNeededDirsScalar(voxelOffset, 0, laneMask);
        if (result != ref) {
            std::cerr << "verifyComputeNeededDirsSentinel: SIMD/scalar mismatch"
                      << "  simd=0x" << std::hex << result
                      << "  ref=0x"  << ref << std::dec << "\n";
            std::abort();
        }
    }

    const uint32_t expected = (1u << kHiBit[2]);  // plus-z only

    if (result != expected) {
        std::cerr << "verifyComputeNeededDirsSentinel: wrong direction mask"
                  << "  got=0x"      << std::hex << result
                  << "  expected=0x" << expected  << std::dec << "\n";
        std::abort();
    }

    std::cout << "verifyComputeNeededDirsSentinel: PASSED\n";
}

// ============================================================
// Verification
// ============================================================

struct VerifyStats {
    uint64_t laneChecks  = 0;  // stencil-point/lane combinations inspected
    uint64_t errors      = 0;
};

// 18 non-center WENO5 stencil offsets {axis, delta}.
// Each point moves strictly along one axis (axis-aligned stencil).
static constexpr int kWeno5Offsets[18][2] = {
    {0,-3},{0,-2},{0,-1},{0,+1},{0,+2},{0,+3},  // x-axis
    {1,-3},{1,-2},{1,-1},{1,+1},{1,+2},{1,+3},  // y-axis
    {2,-3},{2,-2},{2,-1},{2,+1},{2,+2},{2,+3},  // z-axis
};

/// @brief For every active lane (set in laneMask), walk the 18 non-center WENO5
/// stencil offsets.  For offsets that cross a leaf boundary, confirm that
/// batchPtrs[axis+1][lane] matches a direct probeLeaf reference.
/// Also confirms that batchPtrs[0][lane] == &firstLeaf[leafIndex[batchStart+lane]].
static void verifyBatchPtrs(
    const LeafT* const (&batchPtrs)[4][SIMDw],
    const LeafT*        firstLeaf,
    const uint32_t*     leafIndex,
    const uint16_t*     voxelOffset,
    int                 batchStart,
    LaneMask            laneMask,
    AccT&               refAcc,
    VerifyStats&        stats)
{
    for (int i = 0; i < SIMDw; i++) {
        if (!laneMask[i]) continue;
        const int p = batchStart + i;

        const LeafT* centerLeaf   = &firstLeaf[leafIndex[p]];
        const nanovdb::Coord cOrig = centerLeaf->origin();
        const uint16_t vo          = voxelOffset[p];

        // Center slot must always point to the center leaf.
        stats.laneChecks++;
        if (batchPtrs[0][i] != centerLeaf) {
            ++stats.errors;
            if (stats.errors <= 10)
                std::cerr << "CENTER MISMATCH lane=" << i << "\n";
        }

        // Walk each stencil offset.
        for (const auto& off : kWeno5Offsets) {
            const int axis  = off[0];
            const int delta = off[1];
            const int lc    = localAxisCoord(vo, axis);

            const bool crossesLo = (lc + delta < 0);
            const bool crossesHi = (lc + delta >= 8);
            if (!crossesLo && !crossesHi) continue;  // stays in center leaf

            // Expected: probe the adjacent leaf in the crossing direction.
            const int  dirBit         = crossesLo ? kLoBit[axis] : kHiBit[axis];
            const nanovdb::Coord nOrig = neighborLeafOrigin(cOrig, dirBit);
            const LeafT* expected      = refAcc.probeLeaf(nOrig);
            const LeafT* actual        = batchPtrs[1 + axis][i];

            stats.laneChecks++;
            if (actual != expected) {
                ++stats.errors;
                if (stats.errors <= 10) {
                    std::cerr << "MISMATCH: lane=" << i
                              << " axis=" << axis << " delta=" << delta
                              << " lc=" << lc
                              << " expected=" << static_cast<const void*>(expected)
                              << " actual="   << static_cast<const void*>(actual) << "\n";
                }
            }
        }
    }
}

// ============================================================
// BatchAccessor correctness verification
//
// checkOneTap<di,dj,dk>: calls batchAcc.cachedGetValue for stencil tap (di,dj,dk),
// then for each active lane compares the result against a direct tree reference.
//
// Assumes the caller has already issued the 6 WENO5 extremal prefetches so that
// all directions reachable by ±3 along any axis are in mProbedMask.
// ============================================================

template<int di, int dj, int dk>
static void checkOneTap(
    const BAccT&                          batchAcc,
    nanovdb::util::Simd<uint16_t, SIMDw>  voVec,
    LaneMask                               leafMask,
    nanovdb::Coord                         centerLeafOrigin,
    const LeafT*                           firstLeaf,
    uint32_t                               currentLeafID,
    const uint16_t*                        voxelOffset,
    int                                    batchStart,
    AccT&                                  refAcc,
    VerifyStats&                           stats)
{
    nanovdb::util::Simd<uint64_t, SIMDw> tapResult(uint64_t(0));
    batchAcc.cachedGetValue<di, dj, dk>(tapResult, voVec, leafMask);

    for (int i = 0; i < SIMDw; ++i) {
        if (!leafMask[i]) continue;
        ++stats.laneChecks;

        const uint16_t vo_i = voxelOffset[batchStart + i];
        const int lx = (vo_i >> 6) & 7;
        const int ly = (vo_i >> 3) & 7;
        const int lz =  vo_i       & 7;
        const int nx = lx + di, ny = ly + dj, nz = lz + dk;
        const int dx = (nx < 0) ? -1 : (nx >= 8) ? 1 : 0;
        const int dy = (ny < 0) ? -1 : (ny >= 8) ? 1 : 0;
        const int dz = (nz < 0) ? -1 : (nz >= 8) ? 1 : 0;
        const int nx_w = nx - dx * 8;
        const int ny_w = ny - dy * 8;
        const int nz_w = nz - dz * 8;
        const uint32_t offset = uint32_t(nx_w) * 64u + uint32_t(ny_w) * 8u + uint32_t(nz_w);

        const LeafT* refLeaf;
        if (dx == 0 && dy == 0 && dz == 0) {
            refLeaf = &firstLeaf[currentLeafID];
        } else {
            refLeaf = refAcc.probeLeaf(
                centerLeafOrigin + nanovdb::Coord(dx * 8, dy * 8, dz * 8));
        }

        const uint64_t expected = refLeaf
            ? static_cast<uint64_t>(refLeaf->getValue(offset))
            : uint64_t(0);
        const uint64_t actual = static_cast<uint64_t>(tapResult[i]);

        if (actual != expected) {
            ++stats.errors;
            if (stats.errors <= 10) {
                std::cerr << "BATCHACC MISMATCH"
                          << " tap=(" << di << "," << dj << "," << dk << ")"
                          << " lane=" << i
                          << " expected=" << expected
                          << " actual="   << actual << "\n";
            }
        }
    }
}

/// @brief Cross-validate BatchAccessor::cachedGetValue for all 18 WENO5 non-center taps.
/// Requires the 6 extremal prefetches to have been called first.
static void verifyBatchAccessor(
    const BAccT&                          batchAcc,
    nanovdb::util::Simd<uint16_t, SIMDw>  voVec,
    LaneMask                               leafMask,
    nanovdb::Coord                         centerLeafOrigin,
    const LeafT*                           firstLeaf,
    uint32_t                               currentLeafID,
    const uint16_t*                        voxelOffset,
    int                                    batchStart,
    AccT&                                  refAcc,
    VerifyStats&                           stats)
{
    // x-axis taps (di in {-3,-2,-1,+1,+2,+3})
    checkOneTap<-3, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap<-2, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap<-1, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap<+1, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap<+2, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap<+3, 0, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    // y-axis taps
    checkOneTap< 0,-3, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0,-2, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0,-1, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0,+1, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0,+2, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0,+3, 0>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    // z-axis taps
    checkOneTap< 0, 0,-3>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0, 0,-2>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0, 0,-1>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0, 0,+1>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0, 0,+2>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
    checkOneTap< 0, 0,+3>(batchAcc, voVec, leafMask, centerLeafOrigin, firstLeaf, currentLeafID, voxelOffset, batchStart, refAcc, stats);
}

// ============================================================
// StencilAccessor correctness verification
//
// For every active lane (set in activeMask returned by moveTo):
//   - Reconstruct the global coordinate from (leafIndex, voxelOffset).
//   - For each of the 18 WENO5 taps, add the tap offset, decompose into
//     leaf-local coordinates, probe the neighbor leaf, and compare
//     stencilAcc[k][lane] against refLeaf->getValue(localOffset).
//
// For every inactive lane:
//   - Assert that all tap slots hold 0 (the NanoVDB background index).
// ============================================================

static void verifyStencilAccessor(
    const SAccT&    stencilAcc,
    IndexMaskT      activeMask,   // returned by stencilAcc.moveTo()
    const uint32_t* leafIndex,
    const uint16_t* voxelOffset,
    int             batchStart,
    const LeafT*    firstLeaf,
    AccT&           refAcc,
    VerifyStats&    stats)
{
    // Check inactive lanes: all tap slots must hold 0 (background index).
    for (int i = 0; i < SIMDw; ++i) {
        if (activeMask[i]) continue;
        for (int k = 0; k < stencilAcc.size(); ++k) {
            ++stats.laneChecks;
            const uint64_t got = static_cast<uint64_t>(stencilAcc[k][i]);
            if (got != 0) {
                ++stats.errors;
                if (stats.errors <= 10)
                    std::cerr << "STENCIL inactive lane=" << i
                              << " tap=" << k
                              << ": expected 0, got " << got << "\n";
            }
        }
    }

    // Check active lanes against the scalar tree reference.
    for (int i = 0; i < SIMDw; ++i) {
        if (!activeMask[i]) continue;

        const int      p              = batchStart + i;
        const uint16_t vo             = voxelOffset[p];
        const uint32_t li             = leafIndex[p];
        const nanovdb::Coord cOrigin  = firstLeaf[li].origin();

        // Center voxel local coordinates within the leaf.
        const int lx = (vo >> 6) & 7;
        const int ly = (vo >> 3) & 7;
        const int lz =  vo       & 7;

        for (int k = 0; k < 18; ++k) {
            const int axis  = kWeno5Offsets[k][0];
            const int delta = kWeno5Offsets[k][1];
            const int di    = (axis == 0) ? delta : 0;
            const int dj    = (axis == 1) ? delta : 0;
            const int dk    = (axis == 2) ? delta : 0;

            // Tap destination in leaf-local space (may be outside [0,7]).
            const int nx = lx + di, ny = ly + dj, nz = lz + dk;

            // Leaf-crossing step (−1, 0, or +1 per axis).
            const int dx = (nx < 0) ? -1 : (nx >= 8) ? 1 : 0;
            const int dy = (ny < 0) ? -1 : (ny >= 8) ? 1 : 0;
            const int dz = (nz < 0) ? -1 : (nz >= 8) ? 1 : 0;

            // Wrapped local coordinates within the target leaf.
            const int nx_w = nx - dx * 8;
            const int ny_w = ny - dy * 8;
            const int nz_w = nz - dz * 8;
            const uint32_t offset = uint32_t(nx_w) * 64u + uint32_t(ny_w) * 8u + uint32_t(nz_w);

            // Reference: probe the target leaf and read its value.
            const LeafT* refLeaf = (dx == 0 && dy == 0 && dz == 0)
                ? &firstLeaf[li]
                : refAcc.probeLeaf(cOrigin + nanovdb::Coord(dx * 8, dy * 8, dz * 8));

            const uint64_t expected = refLeaf
                ? static_cast<uint64_t>(refLeaf->getValue(offset))
                : uint64_t(0);
            const uint64_t actual   = static_cast<uint64_t>(stencilAcc[k][i]);

            ++stats.laneChecks;
            if (actual != expected) {
                ++stats.errors;
                if (stats.errors <= 10)
                    std::cerr << "STENCIL MISMATCH"
                              << " tap=(" << di << "," << dj << "," << dk << ")"
                              << " slot=" << k
                              << " lane=" << i
                              << " expected=" << expected
                              << " actual="   << actual << "\n";
            }
        }
    }
}

// ============================================================
// Main prototype: Phase 1 (neighbor leaf resolution) + verification
// ============================================================

static void runPrototype(const GridT*                                                          grid,
                         const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>& vbmHandle)
{
    const auto&    tree        = grid->tree();
    const LeafT*   firstLeaf   = tree.getFirstNode<0>();
    const uint64_t nVoxels     = grid->activeVoxelCount();
    const uint32_t nBlocks     = (uint32_t)vbmHandle.blockCount();

    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();

    // One ReadAccessor per thread, reused across all blocks (§8c).
    AccT acc = grid->getAccessor();

    // Block-local scratch (stack-resident, stays in L1 across batches).
    alignas(64) uint32_t leafIndex[BlockWidth];
    alignas(64) uint16_t voxelOffset[BlockWidth];

    VerifyStats stats;
    uint64_t    nStraddles = 0;

    for (uint32_t bID = 0; bID < nBlocks; bID++) {
        const uint64_t blockFirstOffset =
            vbmHandle.firstOffset() + (uint64_t)bID * BlockWidth;

        // Decode inverse maps.
        CPUVBM::decodeInverseMaps(
            grid,
            firstLeafID[bID],
            &jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength],
            blockFirstOffset,
            leafIndex,
            voxelOffset);

        // Recompute nLeaves from jumpMap; avoids modifying decodeInverseMaps API
        // and keeps CPU/CUDA API symmetric (§9).
        int nExtraLeaves = 0;
        for (int w = 0; w < CPUVBM::JumpMapLength; w++)
            nExtraLeaves += nanovdb::util::countOn(
                jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength + w]);

        // Block-level neighbor-leaf resolution state (§8d, §8f).
        uint32_t     currentLeafID    = firstLeafID[bID];
        uint32_t     probedMask       = 0;
        const LeafT* ptrs[27]         = {};
        nanovdb::Coord centerLeafCoord = firstLeaf[currentLeafID].origin();

        // BatchAccessor: alternate execution path for correctness cross-validation.
        BAccT batchAcc(*grid, currentLeafID);

        // StencilAccessor: constructed once per block, persists across batches.
        SAccT stencilAcc(*grid, firstLeafID[bID], (uint32_t)nExtraLeaves);

        // Process SIMD batches.
        for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {

            // Load the SIMDw leafIndex values for this batch once; reused below.
            const LeafIdxVec leafSlice(&leafIndex[batchStart], nanovdb::util::element_aligned);

            // Active-lane mask: lanes with a valid (non-sentinel) leafIndex.
            LaneMask activeMask = (leafSlice != LeafIdxVec(CPUVBM::UnusedLeafIndex));
            if (nanovdb::util::none_of(activeMask)) continue;

            // StencilAccessor: gather all 18 WENO5 tap indices for this batch.
            // moveTo owns the straddling loop internally; call once per batch.
            {
                const IndexMaskT sActive =
                    stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);
                verifyStencilAccessor(stencilAcc, sActive, leafIndex, voxelOffset,
                                      batchStart, firstLeaf, acc, stats);
            }

            // Track straddle batches for diagnostic output.
            for (int i = 0; i < SIMDw; i++) {
                if (activeMask[i] && leafIndex[batchStart + i] != currentLeafID) {
                    nStraddles++;
                    break;
                }
            }

            // Inner loop: consume one center leaf's worth of lanes per iteration.
            while (nanovdb::util::any_of(activeMask)) {
                // Identify lanes belonging to currentLeafID.
                LaneMask leafMask = activeMask & (leafSlice == LeafIdxVec(currentLeafID));

                if (nanovdb::util::none_of(leafMask)) {
                    // No lanes for currentLeafID: advance to next leaf.
                    assert(currentLeafID < firstLeafID[bID] + (uint32_t)nExtraLeaves);
                    currentLeafID++;
                    probedMask       = 0;
                    centerLeafCoord  = firstLeaf[currentLeafID].origin();
                    batchAcc.advance(currentLeafID);
                    continue;
                }

                // --- Phase 1: probe newly needed neighbor leaves (§8d) ---
                //
                // Build the pre-expanded vector at the gather site — the only
                // place where leafMask is known.  Broadcast the sentinel first
                // (straddle / inactive lanes stay neutral), then overwrite the
                // leafMask lanes with their actual expandVoxelOffset values.
                using VecU32 = nanovdb::util::Simd<uint32_t, SIMDw>;
                VecU32 expandedVec(kSentinelExpanded);
                for (int i = 0; i < SIMDw; i++) {
                    if (leafMask[i])
                        expandedVec[i] = expandVoxelOffset(voxelOffset[batchStart + i]);
                }
                const uint32_t neededMask = computeNeededDirs(expandedVec);

                // Cross-check against scalar reference (always-on; overhead is
                // ~18 scalar ops per batch, negligible vs. the probeLeaf calls).
                {
                    const uint32_t ref = computeNeededDirsScalar(voxelOffset, batchStart, leafMask);
                    if (neededMask != ref) {
                        std::cerr << "computeNeededDirs: SIMD/scalar mismatch"
                                  << "  simd=0x" << std::hex << neededMask
                                  << "  ref=0x"  << ref << std::dec << "\n";
                        std::abort();
                    }
                }

                uint32_t toProbe = neededMask & ~probedMask;

                while (toProbe) {
                    const int d = __builtin_ctz(toProbe);
                    ptrs[d]     = acc.probeLeaf(neighborLeafOrigin(centerLeafCoord, d));
                    probedMask |= (1u << d);
                    toProbe    &= toProbe - 1;
                }

                // --- Phase 2: populate per-lane batchPtrs[4][SIMDw] (§6c) ---
                // batchPtrs[0][i] = center leaf
                // batchPtrs[1][i] = x-axis neighbor (lo, hi, or nullptr)
                // batchPtrs[2][i] = y-axis neighbor
                // batchPtrs[3][i] = z-axis neighbor
                const LeafT* batchPtrs[4][SIMDw] = {};
                for (int i = 0; i < SIMDw; i++) {
                    if (!leafMask[i]) continue;
                    batchPtrs[0][i] = &firstLeaf[currentLeafID];
                    for (int axis = 0; axis < 3; axis++) {
                        const int lc = localAxisCoord(voxelOffset[batchStart + i], axis);
                        if (lc < R)
                            batchPtrs[1 + axis][i] = ptrs[kLoBit[axis]];
                        else if (lc >= 8-R)
                            batchPtrs[1 + axis][i] = ptrs[kHiBit[axis]];
                        // else: nullptr (interior lane for this axis)
                    }
                }

                // --- Verification (Phase 1 pointer check) ---
                verifyBatchPtrs(batchPtrs, firstLeaf, leafIndex, voxelOffset,
                                batchStart, leafMask, acc, stats);

                // --- BatchAccessor alternate path + cross-validation ---
                //
                // 6 extremal WENO5 prefetches cover all face-neighbor directions.
                // The center direction (dir(0,0,0)) is guaranteed populated by at
                // least one of these calls (see BatchAccessor.md §5).
                using VoVecT = nanovdb::util::Simd<uint16_t, SIMDw>;
                const VoVecT voVec(&voxelOffset[batchStart], nanovdb::util::element_aligned);
                batchAcc.prefetch<-3,  0,  0>(voVec, leafMask);
                batchAcc.prefetch<+3,  0,  0>(voVec, leafMask);
                batchAcc.prefetch< 0, -3,  0>(voVec, leafMask);
                batchAcc.prefetch< 0, +3,  0>(voVec, leafMask);
                batchAcc.prefetch< 0,  0, -3>(voVec, leafMask);
                batchAcc.prefetch< 0,  0, +3>(voVec, leafMask);

                verifyBatchAccessor(batchAcc, voVec, leafMask, centerLeafCoord,
                                    firstLeaf, currentLeafID, voxelOffset,
                                    batchStart, acc, stats);

                activeMask = activeMask & !leafMask;
            }
        }
    }

    std::cout << "Prototype (Phase 1 + BatchAccessor verification):\n"
              << "  blocks     = " << nBlocks          << "\n"
              << "  voxels     = " << nVoxels           << "\n"
              << "  straddles  = " << nStraddles        << "\n"
              << "  laneChecks = " << stats.laneChecks  << "\n";

    if (stats.errors == 0)
        std::cout << "  PASSED\n";
    else
        std::cerr << "  FAILED: " << stats.errors << " mismatches\n";
}

// ============================================================
// Performance measurement: StencilAccessor::moveTo throughput
//
// Two-pass design: first pass warms instruction cache, branch predictor,
// and the leaf data accessed by advance()/prefetch().  Second pass is timed.
// decodeInverseMaps is outside the rdtsc fence — we measure moveTo only.
//
// Reports TSC ticks/batch and TSC ticks/voxel (using BlockWidth as denominator;
// slightly over-counts inactive padding slots but is stable across runs).
// TSC ticks ≈ ns × (nominal_GHz); divide by actual turbo frequency for
// CPU cycles if needed.
// ============================================================

static void runPerf(const GridT*                                                          grid,
                    const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>&  vbmHandle)
{
    const LeafT*    firstLeaf   = grid->tree().getFirstNode<0>();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();

    alignas(64) uint32_t leafIndex[BlockWidth];
    alignas(64) uint16_t voxelOffset[BlockWidth];

    static constexpr int kBatchesPerBlock = BlockWidth / SIMDw;

    // Shared decode + moveTo loop, run twice (warmup then timed).
    uint64_t totalTicks = 0;

    for (int pass = 0; pass < 2; ++pass) {
        uint64_t passTicks = 0;

        for (uint32_t bID = 0; bID < nBlocks; ++bID) {
            const uint64_t blockFirstOffset =
                vbmHandle.firstOffset() + (uint64_t)bID * BlockWidth;

            // Decode is outside the timed region.
            CPUVBM::decodeInverseMaps(
                grid, firstLeafID[bID],
                &jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength],
                blockFirstOffset, leafIndex, voxelOffset);

            int nExtraLeaves = 0;
            for (int w = 0; w < CPUVBM::JumpMapLength; ++w)
                nExtraLeaves += nanovdb::util::countOn(
                    jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength + w]);

            SAccT stencilAcc(*grid, firstLeafID[bID], (uint32_t)nExtraLeaves);

            _mm_lfence();
            const uint64_t t0 = __rdtsc();

            for (int b = 0; b < kBatchesPerBlock; ++b)
                stencilAcc.moveTo(leafIndex + b * SIMDw, voxelOffset + b * SIMDw);

            uint32_t aux;
            const uint64_t t1 = __rdtscp(&aux);

            passTicks += (t1 - t0);
        }

        if (pass == 1) totalTicks = passTicks;  // only record the warm pass
    }

    const uint64_t totalBatches = (uint64_t)nBlocks * kBatchesPerBlock;
    const uint64_t totalVoxels  = (uint64_t)nBlocks * BlockWidth;

    std::printf("\nStencilAccessor::moveTo throughput (warm pass, %u blocks):\n", nBlocks);
    std::printf("  total TSC ticks : %lu\n",   totalTicks);
    std::printf("  ticks / batch   : %.1f\n",  double(totalTicks) / double(totalBatches));
    std::printf("  ticks / voxel   : %.2f\n",  double(totalTicks) / double(totalVoxels));
}

// ============================================================
// Entry point
// ============================================================

int main(int argc, char** argv)
{
    try {
        // Targeted sentinel test runs unconditionally before any VBM data is needed.
        verifyComputeNeededDirsSentinel();

        int   ambient_voxels = 1024 * 1024;  // smaller default than the CUDA test
        float occupancy      = 0.5f;

        if (argc > 1) ambient_voxels = std::stoi(argv[1]);
        if (argc > 2) occupancy      = std::stof(argv[2]);
        occupancy = std::max(0.0f, std::min(1.0f, occupancy));

        std::cout << "ambient_voxels = " << ambient_voxels << "\n"
                  << "occupancy      = " << occupancy      << "\n";

        auto coords = generateDomain(ambient_voxels, occupancy);
        std::cout << "Active voxels generated: " << coords.size() << "\n";

        // Build a float build grid from the coordinates.
        nanovdb::tools::build::Grid<float> buildGrid(0.f);
        for (const auto& coord : coords)
            buildGrid.tree().setValue(coord, 1.f);

        // Convert build::Grid<float> → NanoGrid<float> → NanoGrid<ValueOnIndex>.
        // Two-step because createNanoGrid<ValueOnIndex> accepts NanoGrid<float>
        // as its source type (same path as ex_index_grid_cuda).
        auto floatHandle = nanovdb::tools::createNanoGrid(buildGrid);
        auto indexHandle = nanovdb::tools::createNanoGrid<
            nanovdb::NanoGrid<float>,
            nanovdb::ValueOnIndex>(
                *floatHandle.grid<float>(),
                0u,     // channels: no sidecar blind data
                false,  // includeStats
                false); // includeTiles
        auto* grid = indexHandle.grid<nanovdb::ValueOnIndex>();
        if (!grid) throw std::runtime_error("Failed to create ValueOnIndex grid");

        const auto& tree = grid->tree();
        std::cout << "Leaves=" << tree.nodeCount(0)
                  << "  Lower=" << tree.nodeCount(1)
                  << "  Upper=" << tree.nodeCount(2)
                  << "  Active=" << grid->activeVoxelCount() << "\n";

        // Build VBM.
        auto vbmHandle = nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(grid);
        std::cout << "VBM blocks=" << vbmHandle.blockCount()
                  << "  (BlockWidth=" << BlockWidth << ")\n\n";

        runPrototype(grid, vbmHandle);
        runPerf(grid, vbmHandle);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
