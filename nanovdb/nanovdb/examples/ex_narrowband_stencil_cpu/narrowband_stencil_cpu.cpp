// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file narrowband_stencil_cpu.cpp

    \brief CPU stencil gather on a real narrow-band level set loaded from .vdb.

    Counterpart to ex_stencil_gather_cpu, which uses a procedurally generated
    random-occupancy domain.  This example instead loads an openvdb level-set
    FloatGrid from disk, converts it to a NanoVDB ValueOnIndex topology grid,
    and harvests the source float values into a separately-allocated sidecar
    buffer.  Purpose: exercise the same perf-decomposition battery on a
    workload with realistic spatial coherence — narrow-band taps are mostly
    close to the surface, so the valueMask.isOn(offset) branch may be more
    predictable than in the random-occupancy case (see BatchAccessor.md §8j).

    Pipeline:
      openvdb::io::File(path)                              -- disk load
      -> openvdb::GridBase::Ptr                            -- untyped handle
      -> openvdb::FloatGrid                                -- typed, narrow-band
      -> nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> -- builder
           .getHandle<ValueOnIndex, HostBuffer>(channels=0) -- topology only
           .copyValues<ValueOnIndex>(sidecar.data())       -- float sidecar
      -> VBM + runPrototype + runPerf (identical to ex_stencil_gather_cpu)

    The sidecar is captured but not yet consumed by any stencil path -- plumbing
    only, for future "fetch values via the sidecar" work.  A one-time
    validation check at startup compares FloatGrid.getValue(ijk) against
    sidecar[indexGrid.tree().getValue(ijk)] on ~1000 random active voxels.

    Build:
      Configured via CMakeLists.txt in the parent examples/ directory.
      Requires OpenVDB (for .vdb IO).  No CUDA.

    Usage:
      narrowband_stencil_cpu <path.vdb> [--grid=<name>]
                                        [--pass=<name>] [--threads=<n>]
                                        [--skip-validation]
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>      // CreateNanoGrid builder, openToIndexVDB
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/util/Simd.h>
#include <nanovdb/util/BatchAccessor.h>
#include <nanovdb/util/StencilAccessor.h>
#include <nanovdb/util/LegacyStencilAccessor.h>
#include <nanovdb/util/WenoStencil.h>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <random>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>     // std::abs (sidecar-stencil-extrap)
#include <cstdio>
#include <cstring>   // std::memcpy (sidecar-pass checksum)
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
// VDB file loading + sidecar harvest
// ============================================================

/// Picks the first openvdb::FloatGrid from the file (optionally by name).
/// Throws on any failure (file not found, no FloatGrid, etc.).
static openvdb::FloatGrid::Ptr
loadFloatGridFromVdb(const std::string& path, const std::string& gridName)
{
    openvdb::io::File file(path);
    file.open(false);  // delayed loading off

    openvdb::GridBase::Ptr base;
    if (!gridName.empty()) {
        if (!file.hasGrid(gridName))
            throw std::runtime_error(
                "no grid named \"" + gridName + "\" in " + path);
        base = file.readGrid(gridName);
    } else {
        // First FloatGrid wins.
        openvdb::GridPtrVecPtr grids = file.getGrids();
        for (auto& g : *grids) {
            if (g && g->isType<openvdb::FloatGrid>()) {
                base = g;  // already fully loaded by getGrids()
                break;
            }
        }
        if (!base)
            throw std::runtime_error("no openvdb::FloatGrid found in " + path);
    }
    file.close();

    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    if (!floatGrid)
        throw std::runtime_error("grid is not an openvdb::FloatGrid");
    return floatGrid;
}

/// Convert an openvdb::FloatGrid into a NanoVDB ValueOnIndex topology grid
/// plus a separately-allocated std::vector<float> sidecar, using the
/// CreateNanoGrid<SrcGridT> builder path (channels=0, no blind data in grid).
///
/// The builder's internal mValIdx is populated by getHandle(), so the
/// subsequent copyValues() writes the FloatGrid's active voxel values into
/// the sidecar in the same order that leaf.getValue(offset) returns.
struct NarrowBandPayload {
    nanovdb::GridHandle<nanovdb::HostBuffer>  handle;
    std::vector<float>                        sidecar;
};

static NarrowBandPayload
convertToIndexGridWithSidecar(openvdb::FloatGrid& floatGrid)
{
    nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> builder(floatGrid);

    NarrowBandPayload p;
    p.handle = builder.template getHandle<
        nanovdb::ValueOnIndex, nanovdb::HostBuffer>(
            /*channels  =*/ 0u,   // no blind data
            /*incStats  =*/ false,
            /*incTiles  =*/ false);

    // valueCount() is only valid after getHandle with an index DstBuildT.
    p.sidecar.resize(builder.valueCount());
    builder.template copyValues<nanovdb::ValueOnIndex>(p.sidecar.data());

    // NanoVDB convention: index 0 of the sidecar holds the background value.
    // copyValues does not write slot 0 (active voxel indices start at 1);
    // set it explicitly so downstream code can treat sidecar[idx] as valid
    // for both in-band (idx>0) and out-of-band (idx==0) taps without branching.
    if (!p.sidecar.empty()) p.sidecar[0] = floatGrid.background();
    return p;
}

/// One-time consistency check between the source FloatGrid and the
/// IndexGrid + sidecar pair.  Samples N active voxels from the source,
/// verifies: floatGrid.getValue(ijk) == sidecar[indexGrid.tree().getValue(ijk)].
/// Returns number of mismatches (0 == pass).
static uint64_t
validateSidecarOrdering(const openvdb::FloatGrid&          floatGrid,
                        const nanovdb::NanoGrid<nanovdb::ValueOnIndex>& indexGrid,
                        const std::vector<float>&          sidecar,
                        size_t                             maxSamples = 1000)
{
    // Walk the source grid's active voxels; sample up to maxSamples of them.
    const auto totalActive = floatGrid.activeVoxelCount();
    if (totalActive == 0) return 0;

    const size_t step = std::max<size_t>(1, size_t(totalActive / maxSamples));
    auto indexAcc     = indexGrid.getAccessor();

    uint64_t checked = 0, mismatches = 0, firstReports = 0;
    size_t   strideCounter = 0;

    for (auto it = floatGrid.cbeginValueOn(); it; ++it) {
        if ((strideCounter++ % step) != 0) continue;

        const openvdb::Coord& oc = it.getCoord();
        const nanovdb::Coord  nc(oc.x(), oc.y(), oc.z());

        const uint64_t idx = indexAcc.getValue(nc);
        if (idx == 0 || idx >= sidecar.size()) {
            ++mismatches;
            if (firstReports++ < 5)
                std::cerr << "  sidecar OOB at (" << oc.x() << "," << oc.y()
                          << "," << oc.z() << "): idx=" << idx
                          << " sidecar.size=" << sidecar.size() << "\n";
            continue;
        }
        const float expected = it.getValue();
        const float actual   = sidecar[idx];
        if (expected != actual) {
            ++mismatches;
            if (firstReports++ < 5)
                std::cerr << "  sidecar MISMATCH at (" << oc.x() << "," << oc.y()
                          << "," << oc.z() << "): idx=" << idx
                          << " expected=" << expected
                          << " actual="   << actual << "\n";
        }
        ++checked;
        if (checked >= maxSamples) break;
    }

    std::cout << "Sidecar validation: checked=" << checked
              << "  mismatches=" << mismatches
              << (mismatches == 0 ? "  PASSED\n" : "  FAILED\n");
    return mismatches;
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
    const std::vector<float>&                                             sidecar,
    const std::string&                                                    passFilter = "all")
{
    // wantPass(<name>) returns true if this pass should run under the current filter.
    // Supported names: "decode", "stencil", "framing", "legacy",
    //                  "sidecar-legacy", "sidecar-stencil", "sidecar-stencil-extrap",
    //                  "sidecar-stencil-normsqgrad", "sidecar-transposed",
    //                  "legacy-transposed".
    // "all" runs everything.
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

    // Second sidecar for the `sidecar` pass: written at each voxel's
    // VBM-sequential index (firstOffset + bID*BlockWidth + lane), which by
    // construction equals the center voxel's ValueOnIndex.  Sized to match
    // the input sidecar so we can reuse its indexing.
    std::vector<float> outputSidecar(sidecar.size(), 0.f);

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
    // pass to expose the 19-tap cost proper.
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
                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                            const nanovdb::Coord center = cOrigin + nanovdb::Coord(lx, ly, lz);
                            // 18 trivial "taps" — no accessor call; anti-DCE via Coord components.
                            uint64_t s = 0;
                            for (int k = 0; k < LegacyAccT::size(); ++k)
                                s += static_cast<uint64_t>(center.x() + center.y() + center.z() + k);
                            bs[p] = s;
                        }
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

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                            legacyAcc.moveTo(cOrigin + nanovdb::Coord(lx, ly, lz));
                            uint64_t s = 0;
                            for (int k = 0; k < LegacyAccT::size(); ++k) s += legacyAcc[k];
                            bs[p] = s;
                        }
                    }
                }
            });
    });

    legacyChecksum =
        std::accumulate(sums.begin(), sums.end(), uint64_t(0),
                        [](uint64_t a, uint64_t b) { return a ^ b; });
    }  // end wantPass("legacy")

    // ---- sidecar-legacy: float value + bool isActive matrices via LegacyStencilAccessor ----
    // Precursor to the full WENO5 pipeline (§11 of BatchAccessor.md).  Within
    // each SIMDw-lane batch, assembles two per-tap arrays:
    //   float values[SIZE][SIMDw]   -- sidecar[idx]  (idx==0 -> background)
    //   bool  isActive[SIZE][SIMDw] -- (idx != 0)
    // Token op (anti-DCE, stand-in for WENO arithmetic): per active voxel,
    // sum values[k][i] over taps with isActive[k][i]==true, write the result
    // to outputSidecar at the voxel's VBM-sequential index.
    double sidecarLegacyUs = 0.0;
    uint64_t sidecarLegacyChecksum = 0;
    if (wantPass("sidecar-legacy")) {
    std::fill(outputSidecar.begin(), outputSidecar.end(), 0.f);

    sidecarLegacyUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                LegacyAccT legacyAcc(*grid);

                constexpr int SIZE = LegacyAccT::size();
                alignas(64) float values  [SIZE][SIMDw];
                alignas(64) bool  isActive[SIZE][SIMDw];

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputSidecar.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        // Fill values[][] and isActive[][] for this batch.
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) {
                                for (int k = 0; k < SIZE; ++k) {
                                    values[k][i]   = scIn[0];
                                    isActive[k][i] = false;
                                }
                                continue;
                            }
                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                            legacyAcc.moveTo(cOrigin + nanovdb::Coord(lx, ly, lz));
                            for (int k = 0; k < SIZE; ++k) {
                                const uint64_t idx = legacyAcc[k];
                                values[k][i]   = scIn[idx];     // scIn[0] == background
                                isActive[k][i] = (idx != 0);
                            }
                        }

                        // Token op: sum values for Active taps per voxel.
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            float sum = 0.f;
                            for (int k = 0; k < SIZE; ++k)
                                if (isActive[k][i]) sum += values[k][i];
                            scOut[blockBase + p] = sum;
                        }
                    }
                }
            });
    });

    // Anti-DCE checksum: XOR of the float bit patterns across the full
    // output sidecar.  Zero-initialised slots contribute 0 (XOR identity),
    // so inactive voxels don't disturb the result.
    sidecarLegacyChecksum =
        std::accumulate(outputSidecar.begin(), outputSidecar.end(), uint64_t(0),
            [](uint64_t a, float b) {
                uint32_t bits;
                std::memcpy(&bits, &b, sizeof(bits));
                return a ^ uint64_t(bits);
            });
    }  // end wantPass("sidecar-legacy")

    // ---- sidecar-stencil: same matrices via StencilAccessor (hybrid SIMD+scalar) ----
    // Uses StencilAccessor's mIndices[SIZE][SIMDw] — the result of its SIMD
    // direction-decode + scalar leaf.getValue() tail — directly as the
    // uint64 index source for the sidecar lookup.  Inactive lanes have
    // mIndices[k][i]=0 naturally (StencilAccessor zero-fills), so the fill
    // loop has no per-lane UnusedLeafIndex guard.
    double sidecarStencilUs = 0.0;
    uint64_t sidecarStencilChecksum = 0;
    if (wantPass("sidecar-stencil")) {
    std::fill(outputSidecar.begin(), outputSidecar.end(), 0.f);

    sidecarStencilUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                constexpr int SIZE = SAccT::size();
                alignas(64) float values  [SIZE][SIMDw];
                alignas(64) bool  isActive[SIZE][SIMDw];

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputSidecar.data();

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
                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);

                        // Tap-outer fill: StencilAccessor stores mIndices[tap][lane]
                        // contiguously along the lane axis, so iterating k-outer
                        // turns lane-inner into a 16-wide sweep over one row.
                        for (int k = 0; k < SIZE; ++k) {
                            for (int i = 0; i < SIMDw; ++i) {
                                const uint64_t idx = stencilAcc.mIndices[k][i];
                                values[k][i]   = scIn[idx];     // scIn[0] == background
                                isActive[k][i] = (idx != 0);
                            }
                        }

                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            float sum = 0.f;
                            for (int k = 0; k < SIZE; ++k)
                                if (isActive[k][i]) sum += values[k][i];
                            scOut[blockBase + p] = sum;
                        }
                    }
                }
            });
    });

    sidecarStencilChecksum =
        std::accumulate(outputSidecar.begin(), outputSidecar.end(), uint64_t(0),
            [](uint64_t a, float b) {
                uint32_t bits;
                std::memcpy(&bits, &b, sizeof(bits));
                return a ^ uint64_t(bits);
            });
    }  // end wantPass("sidecar-stencil")

    // ---- sidecar-stencil-extrap: sidecar-stencil + WenoStencil::extrapolate ----
    // Same fill as sidecar-stencil, then calls WenoStencil<SIMDw>::extrapolate
    // to repair out-of-band lanes via copysign(|background|, mValues[innerTap]).
    // After extrapolation, isActive is not needed for the downstream op;
    // the token sum over ALL taps (active + extrapolated) is the anti-DCE
    // artifact.  Checksum will differ from sidecar-stencil (which summed
    // active-only) — that's the expected correctness signal.
    double sidecarStencilExtrapUs = 0.0;
    uint64_t sidecarStencilExtrapChecksum = 0;
    if (wantPass("sidecar-stencil-extrap")) {
    std::fill(outputSidecar.begin(), outputSidecar.end(), 0.f);

    const float absBackground = std::abs(sidecar[0]);  // sidecar[0] = floatGrid.background()

    sidecarStencilExtrapUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                // Caller-owned fill-side scratch — scalar scatter writes from
                // the sidecar land here, then a per-tap SIMD load moves the
                // data into the stencil's Simd-typed compute view.
                alignas(64) float raw_values[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];
                alignas(64) bool  raw_active[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];

                nanovdb::WenoStencil<SIMDw> stencil;
                constexpr int SIZE = nanovdb::WenoStencil<SIMDw>::size();
                using FloatV = nanovdb::util::Simd    <float, SIMDw>;
                using MaskV  = nanovdb::util::SimdMask<float, SIMDw>;

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputSidecar.data();

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
                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);

                        // Scalar scatter fill into caller-owned C arrays.
                        for (int k = 0; k < SIZE; ++k) {
                            for (int i = 0; i < SIMDw; ++i) {
                                const uint64_t idx = stencilAcc.mIndices[k][i];
                                raw_values[k][i] = scIn[idx];
                                raw_active[k][i] = (idx != 0);
                            }
                        }

                        // SIMD load-per-tap into the stencil's compute view.
                        for (int k = 0; k < SIZE; ++k) {
                            stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::element_aligned);
                            stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::element_aligned);
                        }

                        // Arithmetic — reads/writes stencil.values[] as Simd in place.
                        stencil.extrapolate(absBackground);

                        // Token sum over all 19 taps, entirely in Simd form.
                        FloatV sum(0.f);
                        for (int k = 0; k < SIZE; ++k) sum = sum + stencil.values[k];

                        // Simd → scalar bridge at the output side, mirroring the
                        // fill-side bridge: SIMD store into a scratch, then per-lane
                        // scalar write to the output sidecar (gated by leafIndex).
                        alignas(64) float sum_lanes[SIMDw];
                        nanovdb::util::store(sum, sum_lanes, nanovdb::util::element_aligned);
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            scOut[blockBase + p] = sum_lanes[i];
                        }
                    }
                }
            });
    });

    sidecarStencilExtrapChecksum =
        std::accumulate(outputSidecar.begin(), outputSidecar.end(), uint64_t(0),
            [](uint64_t a, float b) {
                uint32_t bits;
                std::memcpy(&bits, &b, sizeof(bits));
                return a ^ uint64_t(bits);
            });
    }  // end wantPass("sidecar-stencil-extrap")

    // ---- sidecar-stencil-normsqgrad: full Phase-2+3 pipeline ----
    // load → extrapolate → normSqGrad → store.  Same Phase-2 front end as
    // sidecar-stencil-extrap, but the 19-tap token sum is replaced by the
    // real Phase-3 arithmetic: Godunov's fifth-order WENO upwind
    // norm-square gradient.  The per-voxel `|∇φ|²` goes straight into the
    // output sidecar — no debug intermediate.
    //
    // Grid voxel size from grid->voxelSize()[0] (isotropic assumption for
    // narrow-band SDFs).  iso = 0 (zero-crossing is the surface).
    double sidecarStencilNormSqGradUs = 0.0;
    uint64_t sidecarStencilNormSqGradChecksum = 0;
    if (wantPass("sidecar-stencil-normsqgrad")) {
    std::fill(outputSidecar.begin(), outputSidecar.end(), 0.f);

    const float absBackground = std::abs(sidecar[0]);
    const float dx            = float(grid->voxelSize()[0]);

    sidecarStencilNormSqGradUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                alignas(64) float raw_values[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];
                alignas(64) bool  raw_active[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];

                nanovdb::WenoStencil<SIMDw> stencil(dx);
                constexpr int SIZE = nanovdb::WenoStencil<SIMDw>::size();
                using FloatV = nanovdb::util::Simd    <float, SIMDw>;
                using MaskV  = nanovdb::util::SimdMask<float, SIMDw>;

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputSidecar.data();

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
                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        stencilAcc.moveTo(leafIndex + batchStart, voxelOffset + batchStart);

                        // Fill — scalar scatter from sidecar into caller-owned raw C arrays.
                        for (int k = 0; k < SIZE; ++k) {
                            for (int i = 0; i < SIMDw; ++i) {
                                const uint64_t idx = stencilAcc.mIndices[k][i];
                                raw_values[k][i] = scIn[idx];
                                raw_active[k][i] = (idx != 0);
                            }
                        }

                        // Load — per-tap SIMD load into stencil's compute view.
                        for (int k = 0; k < SIZE; ++k) {
                            stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::element_aligned);
                            stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::element_aligned);
                        }

                        // Phase-3 arithmetic (in-place on stencil.values[], then reduce).
                        stencil.extrapolate(absBackground);
                        const FloatV result = stencil.normSqGrad(/* iso = */ 0.f);

                        // Simd → scalar bridge; per-lane scalar write to output sidecar.
                        alignas(64) float result_lanes[SIMDw];
                        nanovdb::util::store(result, result_lanes, nanovdb::util::element_aligned);
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            scOut[blockBase + p] = result_lanes[i];
                        }
                    }
                }
            });
    });

    sidecarStencilNormSqGradChecksum =
        std::accumulate(outputSidecar.begin(), outputSidecar.end(), uint64_t(0),
            [](uint64_t a, float b) {
                uint32_t bits;
                std::memcpy(&bits, &b, sizeof(bits));
                return a ^ uint64_t(bits);
            });
    }  // end wantPass("sidecar-stencil-normsqgrad")

    // ---- sidecar-transposed: tap-outer fill via direct ReadAccessor ----
    // Mirrors `legacy-transposed`'s loop structure, but instead of summing
    // uint64 indices into a per-voxel accumulator, the tap-outer loop fills
    // values[tap][lane] + isActive[tap][lane].  A second voxel-outer pass
    // performs the same token sum as the other variants.
    double sidecarXposedUs = 0.0;
    uint64_t sidecarXposedChecksum = 0;
    if (wantPass("sidecar-transposed")) {
    std::fill(outputSidecar.begin(), outputSidecar.end(), 0.f);

    using Weno5TapsX = nanovdb::Weno5Stencil::Taps;
    static constexpr int SIZEX = int(std::tuple_size_v<Weno5TapsX>);

    sidecarXposedUs = timeForEach([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];
                alignas(64) nanovdb::Coord centers[SIMDw];
                alignas(64) float values  [SIZEX][SIMDw];
                alignas(64) bool  isActive[SIZEX][SIMDw];
                nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(grid->tree().root());

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputSidecar.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            centers[i] = cOrigin + nanovdb::Coord(
                                (vo >> 6) & 7, (vo >> 3) & 7, vo & 7);
                        }

                        auto processTap = [&]<int K, int DI, int DJ, int DK>()
                            [[gnu::always_inline]]
                        {
                            for (int i = 0; i < SIMDw; ++i) {
                                if (leafIndex[batchStart + i] == CPUVBM::UnusedLeafIndex) {
                                    values  [K][i] = scIn[0];
                                    isActive[K][i] = false;
                                    continue;
                                }
                                const nanovdb::Coord c = centers[i]
                                    + nanovdb::Coord(DI, DJ, DK);
                                const LeafT* leaf = acc.probeLeaf(c);
                                if (!leaf) {
                                    values  [K][i] = scIn[0];
                                    isActive[K][i] = false;
                                    continue;
                                }
                                const uint32_t offset = (uint32_t(c[0] & 7) << 6)
                                                      | (uint32_t(c[1] & 7) << 3)
                                                      |  uint32_t(c[2] & 7);
                                const uint64_t idx = leaf->data()->getValue(offset);
                                values  [K][i] = scIn[idx];
                                isActive[K][i] = (idx != 0);
                            }
                        };

                        [&]<size_t... Is>(std::index_sequence<Is...>) {
                            (processTap.template operator()<
                                int(Is),
                                std::tuple_element_t<Is, Weno5TapsX>::di,
                                std::tuple_element_t<Is, Weno5TapsX>::dj,
                                std::tuple_element_t<Is, Weno5TapsX>::dk>(), ...);
                        }(std::make_index_sequence<SIZEX>{});

                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            float sum = 0.f;
                            for (int k = 0; k < SIZEX; ++k)
                                if (isActive[k][i]) sum += values[k][i];
                            scOut[blockBase + p] = sum;
                        }
                    }
                }
            });
    });

    sidecarXposedChecksum =
        std::accumulate(outputSidecar.begin(), outputSidecar.end(), uint64_t(0),
            [](uint64_t a, float b) {
                uint32_t bits;
                std::memcpy(&bits, &b, sizeof(bits));
                return a ^ uint64_t(bits);
            });
    }  // end wantPass("sidecar-transposed")

    // ---- Legacy transposed: tap-outer, voxel-inner ----
    // Same semantics as `legacy`, reordered.  For each of the 19 WENO5 taps,
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
                alignas(64) nanovdb::Coord centers[SIMDw];
                alignas(64) uint64_t s[SIMDw];
                nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(grid->tree().root());
                uint64_t* bs0 = sums.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        grid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    uint64_t* bs = bs0 + bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        for (int i = 0; i < SIMDw; ++i) {
                            s[i] = 0;
                            const int p = batchStart + i;
                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            centers[i] = cOrigin + nanovdb::Coord(
                                (vo >> 6) & 7, (vo >> 3) & 7, vo & 7);
                        }

                        auto processTap = [&]<int DI, int DJ, int DK>()
                            [[gnu::always_inline]]
                        {
                            for (int i = 0; i < SIMDw; ++i) {
                                if (leafIndex[batchStart + i] == CPUVBM::UnusedLeafIndex) continue;
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

                        for (int i = 0; i < SIMDw; ++i) bs[batchStart + i] = s[i];
                    }
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
    std::printf("  Sidecar (legacy)      : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        sidecarLegacyUs / 1e3, sidecarLegacyUs * 1e3 / double(nVoxels),
        (sidecarLegacyUs - decodeUs) / 1e3, sidecarLegacyChecksum);
    std::printf("  Sidecar (stencil)     : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        sidecarStencilUs / 1e3, sidecarStencilUs * 1e3 / double(nVoxels),
        (sidecarStencilUs - decodeUs) / 1e3, sidecarStencilChecksum);
    std::printf("  Sidecar (stencil+extrap): %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        sidecarStencilExtrapUs / 1e3, sidecarStencilExtrapUs * 1e3 / double(nVoxels),
        (sidecarStencilExtrapUs - decodeUs) / 1e3, sidecarStencilExtrapChecksum);
    std::printf("  Sidecar (+normSqGrad) : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        sidecarStencilNormSqGradUs / 1e3, sidecarStencilNormSqGradUs * 1e3 / double(nVoxels),
        (sidecarStencilNormSqGradUs - decodeUs) / 1e3, sidecarStencilNormSqGradChecksum);
    std::printf("  Sidecar (transposed)  : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        sidecarXposedUs / 1e3, sidecarXposedUs * 1e3 / double(nVoxels),
        (sidecarXposedUs - decodeUs) / 1e3, sidecarXposedChecksum);
    std::printf("  Legacy transposed     : %7.1f ms  (%5.1f ns/voxel)  [%+5.1f ms over decode]  checksum=0x%016lx\n",
        legacyXposedUs / 1e3, legacyXposedUs * 1e3 / double(nVoxels),
        (legacyXposedUs - decodeUs) / 1e3, legacyXposedChecksum);

    if (stencilChecksum != legacyChecksum)
        std::cerr << "  WARNING: stencil/legacy checksums differ — accessor results disagree!\n";
    if (legacyChecksum != legacyXposedChecksum)
        std::cerr << "  WARNING: legacy/legacy-transposed checksums differ — ordering bug!\n";
    if (sidecarLegacyChecksum != sidecarStencilChecksum)
        std::cerr << "  WARNING: sidecar legacy/stencil checksums differ — accessor results disagree!\n";
    if (sidecarLegacyChecksum != sidecarXposedChecksum)
        std::cerr << "  WARNING: sidecar legacy/transposed checksums differ — ordering bug!\n";
}

// ============================================================
// Entry point
// ============================================================

static void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <path.vdb>"
        << " [--grid=<name>]"
        << " [--pass=<name>]"
        << " [--threads=<n>]"
        << " [--skip-validation]\n"
        << "\n"
        << "  <path.vdb>           Input OpenVDB file (single FloatGrid narrow-band)\n"
        << "  --grid=<name>        Select grid by name (default: first FloatGrid)\n"
        << "  --pass=<name>        Run one perf pass:\n"
        << "                         all (default), verify, decode, stencil,\n"
        << "                         framing, legacy, legacy-transposed,\n"
        << "                         sidecar-legacy, sidecar-stencil,\n"
        << "                         sidecar-stencil-extrap,\n"
        << "                         sidecar-stencil-normsqgrad, sidecar-transposed\n"
        << "  --threads=<n>        Limit TBB parallelism (0 = TBB default)\n"
        << "  --skip-validation    Skip the sidecar ordering sanity check\n";
}

int main(int argc, char** argv)
{
    try {
        if (argc < 2 || std::string(argv[1]) == "--help"
                     || std::string(argv[1]) == "-h") {
            printUsage(argv[0]);
            return argc < 2 ? 1 : 0;
        }

        std::string vdbPath        = argv[1];
        std::string gridName       = "";       // --grid=<name>
        std::string passFilter     = "all";    // --pass=<name>
        int         nThreads       = 0;        // --threads=<n>, 0 = TBB default
        bool        skipValidation = false;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--grid=", 0) == 0)                 gridName = a.substr(7);
            else if (a.rfind("--pass=", 0) == 0)            passFilter = a.substr(7);
            else if (a.rfind("--threads=", 0) == 0)         nThreads = std::stoi(a.substr(10));
            else if (a == "--skip-validation")              skipValidation = true;
            else { printUsage(argv[0]); return 1; }
        }

        std::cout << "vdb path       = " << vdbPath << "\n"
                  << "grid name      = " << (gridName.empty() ? "(first FloatGrid)" : gridName) << "\n"
                  << "pass           = " << passFilter << "\n"
                  << "threads        = " << (nThreads > 0 ? std::to_string(nThreads) : std::string("(TBB default)")) << "\n";

        // ---- OpenVDB setup and .vdb load ----
        openvdb::initialize();
        auto floatGrid = loadFloatGridFromVdb(vdbPath, gridName);

        const auto bbox    = floatGrid->evalActiveVoxelBoundingBox();
        const auto dim     = bbox.dim();
        const auto vsize   = floatGrid->voxelSize();
        std::cout << "FloatGrid:\n"
                  << "  name              = \"" << floatGrid->getName() << "\"\n"
                  << "  active voxels     = " << floatGrid->activeVoxelCount() << "\n"
                  << "  bbox              = [" << bbox.min() << " .. " << bbox.max() << "]"
                  << "  dim=" << dim << "\n"
                  << "  voxel size        = " << vsize << "\n"
                  << "  background        = " << floatGrid->background() << "\n";

        // ---- Convert to NanoVDB IndexGrid + separately-allocated float sidecar ----
        auto payload = convertToIndexGridWithSidecar(*floatGrid);
        auto* grid   = payload.handle.grid<nanovdb::ValueOnIndex>();
        if (!grid) throw std::runtime_error("Failed to create ValueOnIndex grid");

        const auto& tree = grid->tree();
        std::cout << "IndexGrid:\n"
                  << "  leaves            = " << tree.nodeCount(0) << "\n"
                  << "  lower nodes       = " << tree.nodeCount(1) << "\n"
                  << "  upper nodes       = " << tree.nodeCount(2) << "\n"
                  << "  active voxels     = " << grid->activeVoxelCount() << "\n"
                  << "  valueCount        = " << grid->valueCount() << "\n"
                  << "  sidecar entries   = " << payload.sidecar.size() << "\n";

        // ---- Sidecar ordering sanity check ----
        if (!skipValidation) {
            if (validateSidecarOrdering(*floatGrid, *grid, payload.sidecar) != 0)
                throw std::runtime_error(
                    "sidecar ordering mismatch -- aborting before benchmarks");
        }

        // ---- VBM ----
        auto vbmHandle = nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(grid);
        std::cout << "VBM:\n"
                  << "  blocks            = " << vbmHandle.blockCount()
                  << "  (BlockWidth=" << BlockWidth << ")\n\n";

        // TBB thread-count limit for perf measurements.
        std::unique_ptr<tbb::global_control> tbbLimit;
        if (nThreads > 0) {
            tbbLimit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, (size_t)nThreads);
        }

        if (passFilter == "all" || passFilter == "verify")
            runPrototype(grid, vbmHandle);
        runPerf(grid, vbmHandle, payload.sidecar, passFilter);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
