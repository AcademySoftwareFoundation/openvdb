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
#include <memory>       // std::unique_ptr
#include <sstream>
#include <numeric>   // std::accumulate (checksum)
#include <nanovdb/util/Timer.h>
#include <tbb/global_control.h>

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
/// Active lanes (leafIndex[p] != UnusedLeafIndex): reconstruct the global
/// coordinate from (leafIndex, voxelOffset), call legacyAcc.moveTo(), and
/// compare all SIZE tap indices element-by-element.
///
/// Inactive lanes: assert all tap slots in stencilAcc hold 0 (background index).
static void verifyStencilAccessor(
    const SAccT&    stencilAcc,
    const uint32_t* leafIndex,
    const uint16_t* voxelOffset,
    int             batchStart,
    const LeafT*    firstLeaf,
    LegacyAccT&     legacyAcc,
    VerifyStats&    stats)
{
    for (int i = 0; i < SIMDw; ++i) {
        const int      p  = batchStart + i;
        const uint32_t li = leafIndex[p];

        if (li == CPUVBM::UnusedLeafIndex) {
            // Inactive lane: all tap slots must hold 0 (NanoVDB background index).
            for (int k = 0; k < stencilAcc.size(); ++k) {
                ++stats.laneChecks;
                const uint64_t got = stencilAcc.mIndices[k][i];
                if (got != 0) {
                    ++stats.errors;
                    if (stats.errors <= 10)
                        std::cerr << "STENCIL inactive lane=" << i
                                  << " tap=" << k
                                  << ": expected 0, got " << got << "\n";
                }
            }
            continue;
        }

        // Active lane: compare against the LegacyStencilAccessor oracle.
        const uint16_t vo = voxelOffset[p];
        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;

        legacyAcc.moveTo(cOrigin + nanovdb::Coord(lx, ly, lz));

        for (int k = 0; k < stencilAcc.size(); ++k) {
            ++stats.laneChecks;
            const uint64_t expected = legacyAcc[k];
            const uint64_t actual   = stencilAcc.mIndices[k][i];
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
            stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);
            verifyStencilAccessor(stencilAcc,
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
    const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>&  vbmHandle,
    const std::string&                                                    passFilter = "all")
{
    // wantPass(<name>) returns true if this pass should run under the current filter.
    // Supported names: "decode", "stencil", "framing", "legacy",
    //                  "legacy-transposed".  "all" runs everything.
    auto wantPass = [&](const char* name) {
        return passFilter == "all" || passFilter == name;
    };

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
    double decodeUs = 0.0;
    if (wantPass("decode")) decodeUs = timeForEach([&] {
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
    double stencilUs = 0.0;
    uint64_t stencilChecksum = 0;
    if (wantPass("stencil")) {
    std::fill(sums.begin(), sums.end(), uint64_t(0));

    stencilUs = timeForEach([&] {
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
                        stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);
                        for (int i = 0; i < SIMDw; ++i) {
                            if (leafIndex[batchStart + i] == CPUVBM::UnusedLeafIndex) continue;
                            uint64_t s = 0;
                            for (int k = 0; k < SAccT::size(); ++k)
                                s += stencilAcc.mIndices[k][i];
                            bs[batchStart + i] = s;
                        }
                    }
                }
            });
    });

    stencilChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });
    }  // end wantPass("stencil")

    // ---- Legacy framing floor: loop structure + decode, no accessor call ----
    // Anti-DCE writes derive from Coord components.  Subtracted from the legacy
    // pass to expose the 18-tap cost proper.
    double framingUs = 0.0;
    if (wantPass("framing")) {
    std::fill(sums.begin(), sums.end(), uint64_t(0));
    framingUs = timeForEach([&] {
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
    }  // end wantPass("framing")

    // ---- LegacyStencilAccessor ----
    double legacyUs = 0.0;
    uint64_t legacyChecksum = 0;
    if (wantPass("legacy")) {
    std::fill(sums.begin(), sums.end(), uint64_t(0));

    legacyUs = timeForEach([&] {
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

    legacyChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });
    }  // end wantPass("legacy")

    // ---- Legacy transposed: tap-outer, voxel-inner ----
    // Same semantics as `legacy`, reordered.  For each of the 18 WENO5 taps,
    // sweep all BlockWidth voxels — giving long runs of probeLeaf + getValue
    // calls with the SAME compile-time tap offset but varying center voxels.
    double legacyXposedUs = 0.0;
    uint64_t legacyXposedChecksum = 0;
    if (wantPass("legacy-transposed")) {
    std::fill(sums.begin(), sums.end(), uint64_t(0));

    using Weno5Taps = nanovdb::Weno5Stencil::Taps;
    static constexpr int SIZE = int(std::tuple_size_v<Weno5Taps>);

    legacyXposedUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                alignas(64) nanovdb::Coord centers[BlockWidth];
                alignas(64) uint64_t s[BlockWidth];
                nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(grid->tree().root());
                uint64_t* bs0 = sums.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    for (int i = 0; i < BlockWidth; ++i) {
                        s[i] = 0;
                        if (leafIndex[i] == CPUVBM::UnusedLeafIndex) continue;
                        const uint16_t vo = voxelOffset[i];
                        const uint32_t li = leafIndex[i];
                        const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                        centers[i] = cOrigin + nanovdb::Coord(
                            (vo >> 6) & 7, (vo >> 3) & 7, vo & 7);
                    }

                    auto processTap = [&]<int DI, int DJ, int DK>()
                        [[gnu::always_inline]]
                    {
                        for (int i = 0; i < BlockWidth; ++i) {
                            if (leafIndex[i] == CPUVBM::UnusedLeafIndex) continue;
                            const nanovdb::Coord c = centers[i]
                                + nanovdb::Coord(DI, DJ, DK);
                            const LeafT* leaf = acc.probeLeaf(c);
                            if (!leaf) continue;
                            const uint32_t offset = (uint32_t(c[0] & 7) << 6)
                                                  | (uint32_t(c[1] & 7) << 3)
                                                  |  uint32_t(c[2] & 7);
                            s[i] += leaf->data()->getValue(offset);
                        }
                    };

                    [&]<size_t... Is>(std::index_sequence<Is...>) {
                        (processTap.template operator()<
                            std::tuple_element_t<Is, Weno5Taps>::di,
                            std::tuple_element_t<Is, Weno5Taps>::dj,
                            std::tuple_element_t<Is, Weno5Taps>::dk>(), ...);
                    }(std::make_index_sequence<SIZE>{});

                    uint64_t* bs = bs0 + bID * BlockWidth;
                    for (int i = 0; i < BlockWidth; ++i) bs[i] = s[i];
                }
            });
    });

    legacyXposedChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });
    }  // end wantPass("legacy-transposed")

    std::printf("\nEnd-to-end stencil gather (%u blocks, %lu active voxels):\n",
        nBlocks, nVoxels);
    std::printf("  decodeInverseMaps only: %7.1f ms  (%5.1f ns/voxel)\n",
        decodeUs / 1e3, decodeUs * 1e3 / double(nVoxels));
    std::printf("  Framing (no accessor) : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]\n",
        framingUs / 1e3, framingUs * 1e3 / double(nVoxels),
        (framingUs - decodeUs) / 1e3);
    std::printf("  StencilAccessor       : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        stencilUs / 1e3, stencilUs * 1e3 / double(nVoxels),
        (stencilUs - decodeUs) / 1e3, stencilChecksum);
    std::printf("  LegacyStencilAccessor : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        legacyUs  / 1e3, legacyUs  * 1e3 / double(nVoxels),
        (legacyUs - decodeUs) / 1e3, legacyChecksum);
    std::printf("  Legacy transposed     : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        legacyXposedUs / 1e3, legacyXposedUs * 1e3 / double(nVoxels),
        (legacyXposedUs - decodeUs) / 1e3, legacyXposedChecksum);

    if (stencilChecksum != legacyChecksum)
        std::cerr << "  WARNING: stencil/legacy checksums differ — accessor results disagree!\n";
    if (legacyChecksum != legacyXposedChecksum)
        std::cerr << "  WARNING: legacy/legacy-transposed checksums differ — ordering bug!\n";
}

// ============================================================
// Entry point
// ============================================================

int main(int argc, char** argv)
{
    try {
        int         ambient_voxels = 1024 * 1024;
        float       occupancy      = 0.5f;
        std::string passFilter     = "all";   // --pass=<name>
        int         nThreads       = 0;       // --threads=<n>, 0 = TBB default

        if (argc > 1) ambient_voxels = std::stoi(argv[1]);
        if (argc > 2) occupancy      = std::stof(argv[2]);
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--pass=", 0) == 0)    passFilter = a.substr(7);
            else if (a.rfind("--threads=", 0) == 0) nThreads = std::stoi(a.substr(10));
        }
        occupancy = std::max(0.0f, std::min(1.0f, occupancy));

        std::cout << "ambient_voxels = " << ambient_voxels << "\n"
                  << "occupancy      = " << occupancy      << "\n"
                  << "pass           = " << passFilter     << "\n"
                  << "threads        = " << (nThreads > 0 ? std::to_string(nThreads) : std::string("(TBB default)")) << "\n";

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

        // TBB thread-count limit for perf measurements.
        std::unique_ptr<tbb::global_control> tbbLimit;
        if (nThreads > 0) {
            tbbLimit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, (size_t)nThreads);
        }

        if (passFilter == "all" || passFilter == "verify")
            runPrototype(grid, vbmHandle);
        runPerf(grid, vbmHandle, passFilter);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
