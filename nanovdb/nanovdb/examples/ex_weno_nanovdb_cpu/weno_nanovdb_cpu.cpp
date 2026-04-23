// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file weno_nanovdb_cpu.cpp

    \brief End-to-end CPU WENO5 norm-square-gradient on a narrow-band level
           set, with a scalar reference for correctness validation.

    Demonstrates the full Phase-2+3 pipeline that BatchAccessor.md §11 has
    been leading up to:

      VBM decode -> per-batch sidecar value assembly -> out-of-band
      sign-extrapolation -> SIMD Godunov WENO5 -> per-voxel |grad phi|^2
      output sidecar.

    Two passes run over the same .vdb input:

      reference : per-voxel scalar nanovdb::math::WenoStencil<NanoGrid<float>>::normSqGrad.
                  Tile values and in-leaf inactive values preserved through the
                  OpenVDB -> NanoVDB conversion carry correctly-signed
                  extrapolation "for free", matching our explicit extrapolate()
                  semantics on in-the-band-typical topology.

      fast      : LegacyStencilAccessor gather -> WenoStencil<W> load ->
                  extrapolate() -> normSqGrad() -> per-lane scalar store.
                  No hybrid SIMD StencilAccessor; voxel-outer Legacy path
                  for code clarity.

    Both passes write to the same-shape output buffer, keyed by ValueOnIndex
    slot; a histogram of |outputRef - outputFast| follows.

    Usage:
      weno_nanovdb_cpu <path.vdb> [--grid=<name>]
                                  [--threads=<n>]
                                  [--skip-validation]

    Build:
      Configured via CMakeLists.txt in the parent examples/ directory.
      Requires OpenVDB (for .vdb IO).  No CUDA.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/util/Simd.h>
#include <nanovdb/util/LegacyStencilAccessor.h>
#include <nanovdb/util/WenoStencil.h>
#include <nanovdb/math/Stencils.h>      // scalar reference WenoStencil<GridT>

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanovdb/util/Timer.h>
#include <tbb/global_control.h>

// ============================================================
// Constants and type aliases
// ============================================================

static constexpr int Log2BlockWidth = 7;
static constexpr int BlockWidth     = 1 << Log2BlockWidth;  // 128
static constexpr int SIMDw          = 16;                   // float lane width

using BuildT     = nanovdb::ValueOnIndex;
using IndexGridT = nanovdb::NanoGrid<BuildT>;
using LeafT      = nanovdb::NanoLeaf<BuildT>;
using FloatGridT = nanovdb::NanoGrid<float>;
using CPUVBM     = nanovdb::tools::VoxelBlockManager<Log2BlockWidth>;

using LegacyAccT = nanovdb::LegacyStencilAccessor<BuildT, nanovdb::WenoStencil<>>;

// ============================================================
// VDB loading and NanoVDB conversion
// ============================================================

static openvdb::FloatGrid::Ptr
loadFloatGridFromVdb(const std::string& path, const std::string& gridName)
{
    openvdb::io::File file(path);
    file.open(false);  // delayed loading off

    openvdb::GridBase::Ptr base;
    if (!gridName.empty()) {
        if (!file.hasGrid(gridName))
            throw std::runtime_error("no grid named \"" + gridName + "\" in " + path);
        base = file.readGrid(gridName);
    } else {
        openvdb::GridPtrVecPtr grids = file.getGrids();
        for (auto& g : *grids) {
            if (g && g->isType<openvdb::FloatGrid>()) { base = g; break; }
        }
        if (!base) throw std::runtime_error("no openvdb::FloatGrid found in " + path);
    }
    file.close();

    auto floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(base);
    if (!floatGrid) throw std::runtime_error("grid is not an openvdb::FloatGrid");
    return floatGrid;
}

/// NanoVDB conversion products shared across the two passes.
/// - floatHandle  : NanoGrid<float> — tile values + in-leaf inactive values
///                  preserved verbatim, used by the scalar reference stencil.
/// - indexHandle  : NanoGrid<ValueOnIndex> — the topology-only index grid.
/// - sidecar      : float sidecar (slot 0 = background, slots 1..N = active
///                  voxel values in NanoVDB indexing order).
struct ConvertedGrids {
    nanovdb::GridHandle<nanovdb::HostBuffer>  floatHandle;
    nanovdb::GridHandle<nanovdb::HostBuffer>  indexHandle;
    std::vector<float>                        sidecar;
};

static ConvertedGrids
convertFloatGrid(openvdb::FloatGrid& floatGrid)
{
    ConvertedGrids out;

    // Direct OpenVDB -> NanoVDB float conversion.  No flags needed:
    //   - tile values at internal nodes are stored directly (mTable);
    //   - in-leaf inactive voxels share storage with active ones (dense 8^3).
    // Both are preserved verbatim from the source grid.
    out.floatHandle = nanovdb::tools::createNanoGrid<openvdb::FloatGrid>(floatGrid);

    // Sidecar pipeline: build ValueOnIndex + float sidecar via the CreateNanoGrid builder.
    nanovdb::tools::CreateNanoGrid<openvdb::FloatGrid> builder(floatGrid);
    out.indexHandle = builder.template getHandle<
        nanovdb::ValueOnIndex, nanovdb::HostBuffer>(
            /*channels  =*/ 0u,
            /*incStats  =*/ false,
            /*incTiles  =*/ false);
    out.sidecar.resize(builder.valueCount());
    builder.template copyValues<nanovdb::ValueOnIndex>(out.sidecar.data());

    // NanoVDB convention: slot 0 is the "not-found / background" sentinel.
    // copyValues doesn't touch it.  Set it to the grid's background so the
    // fast path can gather unconditionally (no per-lane branch on idx==0 during fill).
    if (!out.sidecar.empty()) out.sidecar[0] = floatGrid.background();

    return out;
}

// ============================================================
// Reference pass — scalar WenoStencil per active voxel
// ============================================================
//
// Uses nanovdb::math::WenoStencil<NanoGrid<float>>.  Its moveTo(ijk)
// populates 19 taps via the float grid's accessor; taps outside the
// narrow band hit either in-leaf inactive slots (stored +-background by
// the OpenVDB narrow-band builder) or tile values at internal nodes
// (same convention).  Both cases yield correctly-signed extrapolated
// values, matching the semantics of our explicit extrapolate().
//
// Cross-path indexing: both outputRef and outputFast are indexed by
// the ValueOnIndex slot of a voxel.  For each active ijk we compute
// normSqGrad on the float grid, then resolve its output slot via the
// index grid's accessor.

static double
runReference(const FloatGridT&    floatGrid,
             const IndexGridT&    indexGrid,
             std::vector<float>&  outputRef)
{
    std::fill(outputRef.begin(), outputRef.end(), 0.f);

    const uint32_t nLeaves = indexGrid.tree().nodeCount(0);

    std::ostringstream sink;
    nanovdb::util::Timer timer;
    auto timeIt = [&](auto&& body) -> double {
        timer.start("", sink);
        body();
        return static_cast<double>(timer.elapsed<std::chrono::microseconds>());
    };

    return timeIt([&] {
        nanovdb::util::forEach(uint32_t(0), nLeaves, uint32_t(1),
            [&](const nanovdb::util::Range1D& range) {
                // One scalar stencil + one index accessor per TBB task.
                nanovdb::math::WenoStencil<FloatGridT> stencil(floatGrid);
                auto indexAcc = indexGrid.getAccessor();

                const auto* firstFloatLeaf =
                    floatGrid.tree().template getFirstNode<0>();
                const auto* firstIndexLeaf =
                    indexGrid.tree().template getFirstNode<0>();

                for (uint32_t lid = range.begin(); lid != range.end(); ++lid) {
                    // The two grids share topology, so leaf LID in the
                    // index grid aligns with leaf LID in the float grid
                    // (same order of insertion).  Iterate the index grid's
                    // active voxels — those are the slots we need to fill.
                    const auto& indexLeaf = firstIndexLeaf[lid];
                    (void)firstFloatLeaf;  // stencil.moveTo uses its own acc

                    for (auto it = indexLeaf.beginValueOn(); it; ++it) {
                        const nanovdb::Coord ijk = it.getCoord();
                        stencil.moveTo(ijk);
                        const float r = stencil.normSqGrad(/*iso=*/0.f);
                        const uint64_t idx = indexAcc.getValue(ijk);
                        outputRef[idx] = r;
                    }
                }
            });
    });
}

// ============================================================
// Fast pass — LegacyStencilAccessor gather + WenoStencil<W> compute
// ============================================================
//
// Structure:
//   for each VBM block:
//     decodeInverseMaps -> leafIndex[128], voxelOffset[128]
//     for each batch of SIMDw voxels:
//       fill: scalar scatter from sidecar into raw_values[SIZE][SIMDw]
//             via LegacyStencilAccessor::moveTo per voxel
//       load: per-tap SIMD load into stencil.values[] / isActive[]
//       extrapolate (sign-fix OOB lanes in-place, Simd)
//       normSqGrad -> FloatV
//       store: per-lane scalar write to outputFast[blockBase + p]

static double
runFast(const IndexGridT&                                                  indexGrid,
        const nanovdb::tools::VoxelBlockManagerHandle<nanovdb::HostBuffer>& vbmHandle,
        const std::vector<float>&                                          sidecar,
        std::vector<float>&                                                outputFast)
{
    std::fill(outputFast.begin(), outputFast.end(), 0.f);

    const LeafT*    firstLeaf   = indexGrid.tree().template getFirstNode<0>();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();
    const uint64_t  firstOffset = vbmHandle.firstOffset();

    const float absBackground = std::abs(sidecar[0]);
    const float dx            = float(indexGrid.voxelSize()[0]);

    std::ostringstream sink;
    nanovdb::util::Timer timer;
    auto timeIt = [&](auto&& body) -> double {
        timer.start("", sink);
        body();
        return static_cast<double>(timer.elapsed<std::chrono::microseconds>());
    };

    return timeIt([&] {
        nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
            [&](const nanovdb::util::Range1D& range) {
                alignas(64) uint32_t leafIndex[BlockWidth];
                alignas(64) uint16_t voxelOffset[BlockWidth];

                // Caller-owned fill-side scratch — scalar scatter from the
                // sidecar lands here, then a per-tap SIMD load moves the
                // data into the stencil's Simd compute view.
                alignas(64) float raw_values[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];
                alignas(64) bool  raw_active[nanovdb::WenoStencil<SIMDw>::size()][SIMDw];

                nanovdb::WenoStencil<SIMDw> stencil(dx);
                constexpr int SIZE = nanovdb::WenoStencil<SIMDw>::size();
                using FloatV = nanovdb::util::Simd    <float, SIMDw>;
                using MaskV  = nanovdb::util::SimdMask<float, SIMDw>;

                // One LegacyStencilAccessor per TBB task (one ReadAccessor).
                LegacyAccT legacyAcc(indexGrid);

                const float* const scIn  = sidecar.data();
                float*       const scOut = outputFast.data();

                for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                    CPUVBM::decodeInverseMaps(
                        &indexGrid, firstLeafID[bID],
                        &jumpMap[bID * CPUVBM::JumpMapLength],
                        firstOffset + bID * BlockWidth,
                        leafIndex, voxelOffset);

                    const uint64_t blockBase =
                        firstOffset + (uint64_t)bID * BlockWidth;

                    for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                        // -------- Fill: LegacyStencilAccessor per voxel --------
                        // Voxel-outer, tap-inner inside the moveTo call
                        // (fillTaps unrolls the 19 tap lookups against the
                        // shared ReadAccessor).  Zero-fill inactive lanes.
                        for (int i = 0; i < SIMDw; ++i) {
                            const int p = batchStart + i;

                            if (leafIndex[p] == CPUVBM::UnusedLeafIndex) {
                                for (int k = 0; k < SIZE; ++k) {
                                    raw_values[k][i] = 0.f;
                                    raw_active[k][i] = false;
                                }
                                continue;
                            }

                            const uint16_t vo = voxelOffset[p];
                            const uint32_t li = leafIndex[p];
                            const nanovdb::Coord cOrigin = firstLeaf[li].origin();
                            const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                            const nanovdb::Coord center =
                                cOrigin + nanovdb::Coord(lx, ly, lz);

                            legacyAcc.moveTo(center);
                            for (int k = 0; k < SIZE; ++k) {
                                const uint64_t idx = legacyAcc[k];
                                raw_values[k][i] = scIn[idx];
                                raw_active[k][i] = (idx != 0);
                            }
                        }

                        // -------- Load: per-tap SIMD load into stencil view --------
                        for (int k = 0; k < SIZE; ++k) {
                            stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::element_aligned);
                            stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::element_aligned);
                        }

                        // -------- Phase-3 arithmetic (in-place on Simd values) --------
                        stencil.extrapolate(absBackground);
                        const FloatV result = stencil.normSqGrad(/*iso=*/0.f);

                        // -------- Simd -> scalar bridge + per-lane store --------
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
}

// ============================================================
// Histogram comparison
// ============================================================
//
// Per-index |outputRef[i] - outputFast[i]| over all active voxels
// (index 1..N; slot 0 is the background/no-op).  Log-decade bins
// from 0 to 1e+1 plus a tail bucket for anything >= 10.
//
// Expected shape: the two leftmost bins ([0,1e-8), [1e-8,1e-7)) hold
// the overwhelming majority — FP-rounding / FMA-fusion differences.
// Anything to the right of [1e-5,1e-4) warrants investigation.

static void
reportHistogram(const std::vector<float>& outputRef,
                const std::vector<float>& outputFast,
                uint64_t                  nActive)
{
    // Bucket edges: 0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e+0, 1e+1
    static constexpr int    nBuckets = 12;
    static constexpr double edges[nBuckets + 1] = {
        0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6,
        1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0
    };
    static const char* labels[nBuckets] = {
        "[0,      1e-10)",
        "[1e-10,  1e-9 )",
        "[1e-9,   1e-8 )",
        "[1e-8,   1e-7 )",
        "[1e-7,   1e-6 )",
        "[1e-6,   1e-5 )",
        "[1e-5,   1e-4 )",
        "[1e-4,   1e-3 )",
        "[1e-3,   1e-2 )",
        "[1e-2,   1e-1 )",
        "[1e-1,   1.0  )",
        "[1.0,    10.0 )"
    };

    std::array<uint64_t, nBuckets + 1> counts{};  // last bucket = [10, inf)
    double   sumDelta   = 0.0;
    float    maxDelta   = 0.f;
    uint64_t worstIdx   = 0;
    uint64_t counted    = 0;

    for (uint64_t i = 1; i <= nActive; ++i) {
        const float d = std::abs(outputRef[i] - outputFast[i]);
        ++counted;
        sumDelta += double(d);
        if (d > maxDelta) { maxDelta = d; worstIdx = i; }

        int b = nBuckets;  // overflow bucket
        for (int k = 0; k < nBuckets; ++k) {
            if (double(d) < edges[k + 1]) { b = k; break; }
        }
        ++counts[b];
    }

    std::printf("\n|Delta| histogram across %lu active voxels"
                " (outputRef vs outputFast):\n", counted);
    for (int k = 0; k < nBuckets; ++k) {
        const double pct = counted ? 100.0 * double(counts[k]) / double(counted) : 0.0;
        std::printf("  %-18s : %12lu   (%6.2f%%)\n",
                    labels[k], counts[k], pct);
    }
    const double pctTail = counted ? 100.0 * double(counts[nBuckets]) / double(counted) : 0.0;
    std::printf("  %-18s : %12lu   (%6.2f%%)\n",
                "[10.0,   inf  )", counts[nBuckets], pctTail);

    std::printf("\n  max  |Delta| = %.6g   (at slot %lu:"
                " ref=%.6g, fast=%.6g)\n",
                double(maxDelta), worstIdx,
                double(outputRef[worstIdx]),
                double(outputFast[worstIdx]));
    std::printf("  mean |Delta| = %.6g\n",
                counted ? sumDelta / double(counted) : 0.0);
}

// ============================================================
// Entry point
// ============================================================

static void printUsage(const char* argv0)
{
    std::cerr
        << "Usage: " << argv0 << " <path.vdb>"
        << " [--grid=<name>] [--threads=<n>]\n"
        << "\n"
        << "  <path.vdb>           Input OpenVDB file (single FloatGrid narrow-band)\n"
        << "  --grid=<name>        Select grid by name (default: first FloatGrid)\n"
        << "  --threads=<n>        Limit TBB parallelism (0 = TBB default)\n";
}

int main(int argc, char** argv)
{
    try {
        if (argc < 2 || std::string(argv[1]) == "--help"
                     || std::string(argv[1]) == "-h") {
            printUsage(argv[0]);
            return argc < 2 ? 1 : 0;
        }

        std::string vdbPath  = argv[1];
        std::string gridName = "";
        int         nThreads = 0;

        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a.rfind("--grid=", 0) == 0)         gridName = a.substr(7);
            else if (a.rfind("--threads=", 0) == 0) nThreads = std::stoi(a.substr(10));
            else { printUsage(argv[0]); return 1; }
        }

        std::cout << "vdb path       = " << vdbPath << "\n"
                  << "grid name      = " << (gridName.empty() ? "(first FloatGrid)" : gridName) << "\n"
                  << "threads        = " << (nThreads > 0 ? std::to_string(nThreads)
                                                          : std::string("(TBB default)")) << "\n";

        // ---- Load the .vdb and convert to both NanoVDB representations ----
        openvdb::initialize();
        auto floatGrid = loadFloatGridFromVdb(vdbPath, gridName);

        const auto bbox  = floatGrid->evalActiveVoxelBoundingBox();
        const auto vsize = floatGrid->voxelSize();
        std::cout << "FloatGrid:\n"
                  << "  active voxels  = " << floatGrid->activeVoxelCount() << "\n"
                  << "  bbox           = [" << bbox.min() << " .. " << bbox.max() << "]\n"
                  << "  voxel size     = " << vsize << "\n"
                  << "  background     = " << floatGrid->background() << "\n";

        auto payload = convertFloatGrid(*floatGrid);
        auto* nanoFloatGrid = payload.floatHandle.grid<float>();
        auto* indexGrid     = payload.indexHandle.grid<nanovdb::ValueOnIndex>();
        if (!nanoFloatGrid || !indexGrid)
            throw std::runtime_error("NanoVDB conversion failed");

        const auto& tree = indexGrid->tree();
        std::cout << "NanoVDB:\n"
                  << "  leaves         = " << tree.nodeCount(0) << "\n"
                  << "  active voxels  = " << indexGrid->activeVoxelCount() << "\n"
                  << "  sidecar size   = " << payload.sidecar.size() << "\n";

        // ---- VBM for the fast path ----
        auto vbmHandle = nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(indexGrid);
        std::cout << "VBM:\n"
                  << "  blocks         = " << vbmHandle.blockCount()
                  << "  (BlockWidth=" << BlockWidth << ")\n\n";

        // ---- TBB thread cap for timings ----
        std::unique_ptr<tbb::global_control> tbbLimit;
        if (nThreads > 0) {
            tbbLimit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, (size_t)nThreads);
        }

        // ---- Output buffers ----
        std::vector<float> outputRef (payload.sidecar.size(), 0.f);
        std::vector<float> outputFast(payload.sidecar.size(), 0.f);

        // ---- Run both passes (warm + timed) ----
        // Warm pass (ignored) for both, then one timed pass each.
        (void)runReference(*nanoFloatGrid, *indexGrid, outputRef);
        (void)runFast(*indexGrid, vbmHandle, payload.sidecar, outputFast);

        const double refUs  = runReference(*nanoFloatGrid, *indexGrid, outputRef);
        const double fastUs = runFast(*indexGrid, vbmHandle, payload.sidecar, outputFast);

        const uint64_t nActive = indexGrid->activeVoxelCount();
        const double   refNs   = refUs  * 1e3 / double(nActive);
        const double   fastNs  = fastUs * 1e3 / double(nActive);

        std::printf("\nEnd-to-end WENO5 |grad phi|^2 (%lu active voxels):\n", nActive);
        std::printf("  reference (scalar): %9.1f ms  (%7.1f ns/voxel)\n",
                    refUs / 1e3, refNs);
        std::printf("  fast  (VBM+SIMD)  : %9.1f ms  (%7.2f ns/voxel)   speedup: %.1fx\n",
                    fastUs / 1e3, fastNs, refUs / std::max(fastUs, 1.0));

        // ---- Histogram of discrepancies ----
        reportHistogram(outputRef, outputFast, nActive);

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
