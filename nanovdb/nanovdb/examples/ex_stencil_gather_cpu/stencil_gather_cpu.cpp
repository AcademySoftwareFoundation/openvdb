// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file stencil_gather_cpu.cpp

    \brief CPU stencil gather: LegacyStencilAccessor vs StencilAccessor.

    Generates a random sparse domain, builds a ValueOnIndex NanoVDB grid and
    a VoxelBlockManager, then runs two stencil-index gather paths side by side:

      LegacyStencilAccessor  --  scalar, one voxel at a time, ReadAccessor-based.
                                 Equivalent to OpenVDB's math/Stencils.h baseline:
                                 path-cached tree walk per tap, per voxel.
                                 The core comparison is the cost of path-cache
                                 eviction: distant WENO5 taps (±3) evict the
                                 center-leaf path, so each moveTo re-traverses
                                 the tree multiple times per voxel.

      StencilAccessor        --  SIMD batch, SIMDw=16 lanes, BatchAccessor-based.
                                 Resolves neighbor leaves once per center-leaf run,
                                 then accesses all taps via direct array indexing.

    runPrototype() cross-validates both paths (LegacyStencilAccessor is the oracle).
    runPerf() measures moveTo throughput for each path (warm pass only, rdtsc).

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
#include <nanovdb/util/Simd.h>
#include <nanovdb/util/BatchAccessor.h>
#include <nanovdb/util/StencilAccessor.h>
#include <nanovdb/util/LegacyStencilAccessor.h>

#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdio>
#include <cassert>
#include <sstream>
#include <numeric>   // std::accumulate (checksum)
#include <nanovdb/util/Timer.h>

// ============================================================
// Constants and type aliases
// ============================================================

static constexpr int Log2BlockWidth = 7;
static constexpr int BlockWidth     = 1 << Log2BlockWidth;  // 128
static constexpr int SIMDw          = 16;                   // StencilAccessor batch width

using BuildT = nanovdb::ValueOnIndex;
using GridT  = nanovdb::NanoGrid<BuildT>;
using LeafT  = nanovdb::NanoLeaf<BuildT>;
using CPUVBM = nanovdb::tools::VoxelBlockManager<Log2BlockWidth>;

using SAccT      = nanovdb::StencilAccessor<BuildT, SIMDw, nanovdb::Weno5Stencil>;
using LegacyAccT = nanovdb::LegacyStencilAccessor<BuildT, nanovdb::Weno5Stencil>;
using IndexMaskT = nanovdb::util::SimdMask<uint64_t, SIMDw>;

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
// Verification
// ============================================================

struct VerifyStats {
    uint64_t laneChecks = 0;
    uint64_t errors     = 0;
};

/// Cross-validate one StencilAccessor batch against LegacyStencilAccessor.
///
/// Active lanes: reconstruct the global coordinate from (leafIndex, voxelOffset),
/// call legacyAcc.moveTo(), and compare all SIZE tap indices element-by-element.
///
/// Inactive lanes: assert all tap slots in stencilAcc hold 0 (background index).
static void verifyStencilAccessor(
    const SAccT&    stencilAcc,
    IndexMaskT      activeMask,
    const uint32_t* leafIndex,
    const uint16_t* voxelOffset,
    int             batchStart,
    const LeafT*    firstLeaf,
    LegacyAccT&     legacyAcc,
    VerifyStats&    stats)
{
    // Inactive lanes: all tap slots must hold 0 (NanoVDB background index).
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

    // Active lanes: compare against the LegacyStencilAccessor oracle.
    for (int i = 0; i < SIMDw; ++i) {
        if (!activeMask[i]) continue;

        const int      p       = batchStart + i;
        const uint16_t vo      = voxelOffset[p];
        const uint32_t li      = leafIndex[p];
        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;

        legacyAcc.moveTo(cOrigin + nanovdb::Coord(lx, ly, lz));

        for (int k = 0; k < stencilAcc.size(); ++k) {
            ++stats.laneChecks;
            const uint64_t expected = legacyAcc[k];
            const uint64_t actual   = static_cast<uint64_t>(stencilAcc[k][i]);
            if (actual != expected) {
                ++stats.errors;
                if (stats.errors <= 10)
                    std::cerr << "STENCIL MISMATCH"
                              << " tap=" << k
                              << " lane=" << i
                              << " expected=" << expected
                              << " actual="   << actual << "\n";
            }
        }
    }
}

// ============================================================
// Correctness run: cross-validate StencilAccessor vs LegacyStencilAccessor
// ============================================================

static void runPrototype(
    const GridT*                                                          grid,
    const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>&  vbmHandle)
{
    const LeafT*   firstLeaf   = grid->tree().getFirstNode<0>();
    const uint64_t nVoxels     = grid->activeVoxelCount();
    const uint32_t nBlocks     = (uint32_t)vbmHandle.blockCount();

    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();

    alignas(64) uint32_t leafIndex[BlockWidth];
    alignas(64) uint16_t voxelOffset[BlockWidth];

    // LegacyStencilAccessor owns its ReadAccessor; one instance per thread.
    LegacyAccT  legacyAcc(*grid);
    VerifyStats stats;

    for (uint32_t bID = 0; bID < nBlocks; ++bID) {
        const uint64_t blockFirstOffset =
            vbmHandle.firstOffset() + (uint64_t)bID * BlockWidth;

        CPUVBM::decodeInverseMaps(
            grid, firstLeafID[bID],
            &jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength],
            blockFirstOffset, leafIndex, voxelOffset);

        int nExtraLeaves = 0;
        for (int w = 0; w < CPUVBM::JumpMapLength; ++w)
            nExtraLeaves += nanovdb::util::countOn(
                jumpMap[(uint64_t)bID * CPUVBM::JumpMapLength + w]);

        SAccT stencilAcc(*grid, firstLeafID[bID], (uint32_t)nExtraLeaves);

        for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
            const IndexMaskT active =
                stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);
            verifyStencilAccessor(stencilAcc, active,
                                  leafIndex, voxelOffset, batchStart,
                                  firstLeaf, legacyAcc, stats);
        }
    }

    std::cout << "Correctness (StencilAccessor vs LegacyStencilAccessor):\n"
              << "  blocks     = " << nBlocks         << "\n"
              << "  voxels     = " << nVoxels          << "\n"
              << "  laneChecks = " << stats.laneChecks << "\n";

    if (stats.errors == 0)
        std::cout << "  PASSED\n";
    else
        std::cerr << "  FAILED: " << stats.errors << " mismatches\n";
}

// ============================================================
// End-to-end performance comparison (multithreaded)
//
// Both paths run the full pipeline inside util::forEach:
//   decodeInverseMaps → coord extraction → stencil gather → sum → store
//
// decodeInverseMaps is deliberately included: its cost is identical for
// both paths (pure cancellation in the comparison) and including it avoids
// fine-grained intra-block timing artifacts.
//
// Anti-DCE artifact: for each active voxel, accumulate the sum of all 18
// tap uint64_t indices and write to sums[bID * BlockWidth + i].  The final
// XOR checksum is printed, forcing the compiler to materialise the stores.
//
// Timing: nanovdb::util::Timer (steady_clock) around each forEach.
// warm pass discards its measurement; only the second pass is reported.
//
// Denominator: grid->activeVoxelCount() — same for both paths.
// ============================================================

static void runPerf(
    const GridT*                                                          grid,
    const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>&  vbmHandle)
{
    const LeafT*    firstLeaf   = grid->tree().getFirstNode<0>();
    const uint64_t  nVoxels     = grid->activeVoxelCount();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();
    const uint64_t  firstOffset = vbmHandle.firstOffset();

    // Anti-DCE output array.  Each thread writes its own non-overlapping
    // range (bID * BlockWidth ... + BlockWidth - 1) — no synchronisation needed.
    std::vector<uint64_t> sums((size_t)nBlocks * BlockWidth, 0);

    std::ostringstream sink;  // absorbs Timer's warm-pass "... " output
    nanovdb::util::Timer timer;

    auto timeForEach = [&](auto&& body) -> double {
        // warm pass
        timer.start("", sink);
        body();
        timer.elapsed();
        // timed pass
        timer.start("", sink);
        body();
        return static_cast<double>(timer.elapsed<std::chrono::microseconds>());
    };

    // ---- decodeInverseMaps-only baseline (both paths pay this cost) ----
    // Anti-DCE: XOR one uint64_t per block derived from leafIndex[] + voxelOffset[]
    // so the compiler can't elide the decode work.
    const double decodeUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    uint64_t acc = 0;
                    for (int i = 0; i < BlockWidth; ++i)
                        acc ^= (uint64_t(leafIndex[i]) << 16) | uint64_t(voxelOffset[i]);
                    sums[bID * BlockWidth] = acc;  // one slot per block as anti-DCE
                }
            });
    });

    // ---- StencilAccessor ----
    std::fill(sums.begin(), sums.end(), uint64_t(0));

    const double stencilUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    int nExtraLeaves = 0;
                    for (int w = 0; w < CPUVBM::JumpMapLength; ++w)
                        nExtraLeaves += nanovdb::util::countOn(
                            jumpMap[bID * CPUVBM::JumpMapLength + w]);

                    SAccT stencilAcc(*grid, firstLeafID[bID], (uint32_t)nExtraLeaves);
                    uint64_t* bs = sums.data() + bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        const IndexMaskT active =
                            stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);
                        for (int i = 0; i < SIMDw; ++i) {
                            if (!active[i]) continue;
                            uint64_t s = 0;
                            for (int k = 0; k < SAccT::size(); ++k)
                                s += static_cast<uint64_t>(stencilAcc[k][i]);
                            bs[batchStart + i] = s;
                        }
                    }
                }
            });
    });

    const uint64_t stencilChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });

    // ---- Legacy cost decomposition variants ----
    // (a) "framing only"  — Legacy loop structure, no accessor call (anti-DCE writes use li+k).
    //     Measures: decodeInverseMaps + Coord compute + 18-iteration inner loop + anti-DCE store.
    // (b) "center-hit only" — Legacy loop + 18× mAcc.getValue(center) instead of tap offsets.
    //     Always hits the ReadAccessor's leaf cache → no tree walk.
    //     Measures: framing + cache-query + leaf-local lookup (mValueMask + mPrefixSum + popcount).
    // (c) "full" — the original LegacyStencilAccessor path.
    //     Measures: framing + cache-query + leaf-local lookup + tree-walk-on-miss.
    //
    // Tree-walk cost per voxel ≈ full − center-hit.
    // Cache + leaf-lookup per voxel ≈ center-hit − framing.
    // Framing per voxel ≈ framing.

    std::fill(sums.begin(), sums.end(), uint64_t(0));
    const double framingUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                uint64_t* bs0 = sums.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    uint64_t* bs = bs0 + bID * BlockWidth;
                    for (int i = 0; i < BlockWidth; ++i) {
                        if (leafIndex[i] == CPUVBM::UnusedLeafIndex) continue;
                        const uint16_t vo = voxelOffset[i];
                        const uint32_t li = leafIndex[i];
                        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                        const nanovdb::Coord center = cOrigin + nanovdb::Coord(lx, ly, lz);
                        // 18 trivial "taps" — no accessor call; anti-DCE via Coord components.
                        uint64_t s = 0;
                        for (int k = 0; k < LegacyAccT::size(); ++k)
                            s += static_cast<uint64_t>(center.x() + center.y() + center.z() + k);
                        bs[i] = s;
                    }
                }
            });
    });

    std::fill(sums.begin(), sums.end(), uint64_t(0));
    const double centerHitUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                auto acc = grid->getAccessor();
                uint64_t* bs0 = sums.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    uint64_t* bs = bs0 + bID * BlockWidth;
                    for (int i = 0; i < BlockWidth; ++i) {
                        if (leafIndex[i] == CPUVBM::UnusedLeafIndex) continue;
                        (void)voxelOffset[i];   // keep decode non-dead
                        const uint32_t li = leafIndex[i];
                        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                        // 18 distinct positions ALL within this leaf's 8^3 footprint
                        // — guarantees leaf-cache hit on every call, but each coord
                        // is unique so the compiler can't CSE the lookups.
                        //   k in [0..17]: local (k&7, (k>>3)&1, 0) sweeps an 8x2x1 slab.
                        uint64_t s = 0;
                        for (int k = 0; k < LegacyAccT::size(); ++k) {
                            const nanovdb::Coord c = cOrigin
                                + nanovdb::Coord(k & 7, (k >> 3) & 1, 0);
                            s += static_cast<uint64_t>(acc.getValue(c));
                        }
                        bs[i] = s;
                    }
                }
            });
    });

    // ---- LegacyStencilAccessor ----
    std::fill(sums.begin(), sums.end(), uint64_t(0));

    const double legacyUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                LegacyAccT legacyAcc(*grid);  // one ReadAccessor per task
                uint64_t* bs0 = sums.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    uint64_t* bs = bs0 + bID * BlockWidth;

                    for (int i = 0; i < BlockWidth; ++i) {
                        if (leafIndex[i] == CPUVBM::UnusedLeafIndex) continue;
                        const uint16_t vo = voxelOffset[i];
                        const uint32_t li = leafIndex[i];
                        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                        legacyAcc.moveTo(cOrigin + nanovdb::Coord(lx, ly, lz));
                        uint64_t s = 0;
                        for (int k = 0; k < LegacyAccT::size(); ++k) s += legacyAcc[k];
                        bs[i] = s;
                    }
                }
            });
    });

    const uint64_t legacyChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });

    std::printf("\nEnd-to-end stencil gather (%u blocks, %lu active voxels):\n",
        nBlocks, nVoxels);
    std::printf("  decodeInverseMaps only: %7.1f ms  (%5.1f ns/voxel)\n",
        decodeUs / 1e3, decodeUs * 1e3 / double(nVoxels));
    std::printf("  StencilAccessor       : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        stencilUs / 1e3, stencilUs * 1e3 / double(nVoxels),
        (stencilUs - decodeUs) / 1e3, stencilChecksum);
    std::printf("  LegacyStencilAccessor : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        legacyUs  / 1e3, legacyUs  * 1e3 / double(nVoxels),
        (legacyUs - decodeUs) / 1e3, legacyChecksum);

    // Decomposition of LegacyStencilAccessor's ns/voxel:
    //   framing       = no accessor call
    //   cache + leaf  = centerHit − framing   (per 18 taps)
    //   tree walk     = legacy    − centerHit (per 18 taps; amortises over ~25% miss rate)
    const double framingNs    = framingUs    * 1e3 / double(nVoxels);
    const double centerHitNs  = centerHitUs  * 1e3 / double(nVoxels);
    const double legacyNs     = legacyUs     * 1e3 / double(nVoxels);
    std::printf("\nLegacy cost decomposition (18 taps/voxel):\n");
    std::printf("  framing only         : %7.1f ms  (%5.1f ns/voxel)\n",
        framingUs / 1e3, framingNs);
    std::printf("  + center-hit × 18    : %7.1f ms  (%5.1f ns/voxel)  [cache+leaf = %5.2f ns/vox = %4.2f ns/tap]\n",
        centerHitUs / 1e3, centerHitNs,
        centerHitNs - framingNs, (centerHitNs - framingNs) / 18.0);
    std::printf("  + stencil × 18 (full): %7.1f ms  (%5.1f ns/voxel)  [tree walk = %5.2f ns/vox = %4.2f ns/tap]\n",
        legacyUs / 1e3, legacyNs,
        legacyNs - centerHitNs, (legacyNs - centerHitNs) / 18.0);

    if (stencilChecksum != legacyChecksum)
        std::cerr << "  WARNING: checksums differ — accessor results disagree!\n";
}

// ============================================================
// Entry point
// ============================================================

int main(int argc, char** argv)
{
    try {
        int   ambient_voxels = 1024 * 1024;
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
