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
// computeNeededDirs (§8e)
// ============================================================

/// @brief Return a 27-bit probedMask bitmask of neighbor directions accessed by
/// any lane set in laneMask.  For WENO5 (R=3) only the 6 face-direction bits
/// {4,10,12,14,16,22} can ever be set; the 21 edge/corner bits remain zero.
static uint32_t computeNeededDirs(const uint16_t* voxelOffset,
                                   int             batchStart,
                                   uint32_t        laneMask)
{
    uint32_t needed = 0;
    for (int i = 0; i < SIMDw; i++) {
        if (!(laneMask & (1u << i))) continue;
        const uint16_t vo = voxelOffset[batchStart + i];
        for (int axis = 0; axis < 3; axis++) {
            const int lc = localAxisCoord(vo, axis);
            if (lc < R)     needed |= (1u << kLoBit[axis]);
            if (lc >= 8-R)  needed |= (1u << kHiBit[axis]);
        }
    }
    return needed;
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
    uint32_t            laneMask,
    AccT&               refAcc,
    VerifyStats&        stats)
{
    for (int i = 0; i < SIMDw; i++) {
        if (!(laneMask & (1u << i))) continue;
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

        // Process SIMD batches.
        for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {

            // Build active-lane mask: positions with a valid (non-sentinel) leafIndex.
            uint32_t activeMask = 0;
            for (int i = 0; i < SIMDw; i++) {
                if (leafIndex[batchStart + i] != CPUVBM::UnusedLeafIndex)
                    activeMask |= (1u << i);
            }
            if (!activeMask) continue;

            // Track straddle batches for diagnostic output.
            for (int i = 0; i < SIMDw; i++) {
                if ((activeMask & (1u << i)) &&
                    leafIndex[batchStart + i] != currentLeafID) {
                    nStraddles++;
                    break;
                }
            }

            // Inner loop: consume one center leaf's worth of lanes per iteration.
            while (activeMask) {
                // Identify lanes belonging to currentLeafID.
                uint32_t leafMask = 0;
                for (int i = 0; i < SIMDw; i++) {
                    if ((activeMask & (1u << i)) &&
                        leafIndex[batchStart + i] == currentLeafID)
                        leafMask |= (1u << i);
                }

                if (!leafMask) {
                    // No lanes for currentLeafID: advance to next leaf.
                    assert(currentLeafID < firstLeafID[bID] + (uint32_t)nExtraLeaves);
                    currentLeafID++;
                    probedMask       = 0;
                    centerLeafCoord  = firstLeaf[currentLeafID].origin();
                    continue;
                }

                // --- Phase 1: probe newly needed neighbor leaves (§8d) ---
                const uint32_t neededMask = computeNeededDirs(voxelOffset, batchStart, leafMask);
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
                    if (!(leafMask & (1u << i))) continue;
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

                // --- Verification ---
                verifyBatchPtrs(batchPtrs, firstLeaf, leafIndex, voxelOffset,
                                batchStart, leafMask, acc, stats);

                activeMask &= ~leafMask;
            }
        }
    }

    std::cout << "Prototype (Phase 1 verification):\n"
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
// Entry point
// ============================================================

int main(int argc, char** argv)
{
    try {
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

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
