// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file vbm_host_cuda_kernels.cu

    \brief CUDA implementation for the VoxelBlockManager test harness.

    Builds the VoxelBlockManager from a ValueOnIndex grid, decodes the full
    inverse map (leafIndex, voxelOffset) for all active voxels, downloads the
    result to the host, and validates it against the grid structure.
    Also benchmarks CPU vs GPU VBM construction with repeated timed runs.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/util/Timer.h>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <thrust/universal_vector.h>

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <vector>
#include <iostream>

namespace {

static constexpr int Log2BlockWidth = 7;
static constexpr int BlockWidth     = 1 << Log2BlockWidth; // 128

// Thread-private scratch arrays for CPU decodeInverseMaps benchmarks.
// Declared at namespace scope with static storage so threadprivate is valid;
// aligned to a cache line to avoid false sharing between threads.
alignas(64) static uint32_t g_scratchLeafIndex[BlockWidth];
alignas(64) static uint16_t g_scratchVoxelOffset[BlockWidth];
#pragma omp threadprivate(g_scratchLeafIndex, g_scratchVoxelOffset)

using VBM    = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
using CPUVBM = nanovdb::tools::VoxelBlockManager<Log2BlockWidth>;

/// @brief Software 32-bit popcount (Hamming weight) via the AND/shift/add/multiply path.
/// Unlike hardware POPCNT (which is scalar), this compiles to VPMULLD under AVX2 and
/// vectorizes across all 16 lanes of a uint32_t vertical sweep over a valueMask.
inline uint32_t popcount32(uint32_t x)
{
    x =  x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return (x * 0x01010101u) >> 24;
}

/// @brief For each VBM block, decode the inverse map and store
/// (leafIndex, voxelOffset) for every active voxel into global output arrays
/// indexed by (globalVoxelOffset - firstOffset).
/// Launch configuration: <<<nBlocks, BlockWidth>>>
__global__ void decodeAllBlocksKernel(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex>* grid,
    const uint32_t*                           firstLeafID,
    const uint64_t*                           jumpMap,
    uint64_t                                  firstOffset,
    uint64_t                                  lastOffset,
    uint32_t                                  nBlocks,
    uint32_t*                                 outLeafIndex,
    uint16_t*                                 outVoxelOffset)
{
    __shared__ uint32_t smem_leafIndex[BlockWidth];
    __shared__ uint16_t smem_voxelOffset[BlockWidth];

    uint32_t bID = blockIdx.x;
    if (bID >= nBlocks) return;

    uint64_t blockFirstOffset = firstOffset + (uint64_t)bID * BlockWidth;

    VBM::decodeInverseMaps(
        grid,
        firstLeafID[bID],
        &jumpMap[(uint64_t)bID * VBM::JumpMapLength],
        blockFirstOffset,
        smem_leafIndex,
        smem_voxelOffset);

    int      tID       = threadIdx.x;
    uint64_t globalIdx = blockFirstOffset + tID;

    if (globalIdx <= lastOffset && smem_leafIndex[tID] != VBM::UnusedLeafIndex) {
        uint64_t k          = globalIdx - firstOffset; // 0-based
        outLeafIndex[k]     = smem_leafIndex[tID];
        outVoxelOffset[k]   = smem_voxelOffset[tID];
    }
}

/// @brief For each VBM block, decode the inverse map into shared memory and
/// write a token value per block to dummyOut to prevent dead code elimination.
/// No per-voxel output is written to global memory.
/// Launch configuration: <<<nBlocks, BlockWidth>>>
__global__ void benchDecodeKernel(
    nanovdb::NanoGrid<nanovdb::ValueOnIndex>* grid,
    const uint32_t*                           firstLeafID,
    const uint64_t*                           jumpMap,
    uint64_t                                  firstOffset,
    uint64_t                                  lastOffset,
    uint32_t                                  nBlocks,
    uint32_t*                                 dummyOut)
{
    __shared__ uint32_t smem_leafIndex[BlockWidth];
    __shared__ uint16_t smem_voxelOffset[BlockWidth];

    uint32_t bID = blockIdx.x;
    if (bID >= nBlocks) return;

    uint64_t blockFirstOffset = firstOffset + (uint64_t)bID * BlockWidth;

    VBM::decodeInverseMaps(
        grid,
        firstLeafID[bID],
        &jumpMap[(uint64_t)bID * VBM::JumpMapLength],
        blockFirstOffset,
        smem_leafIndex,
        smem_voxelOffset);

    // Token write from thread 0 only to prevent dead code elimination
    if (threadIdx.x == 0) {
        dummyOut[bID] =
            (smem_leafIndex[0]  != VBM::UnusedLeafIndex   ? 1u : 0u) +
            (smem_voxelOffset[0] != VBM::UnusedVoxelOffset ? 1u : 0u);
    }
}

} // anonymous namespace

void runVBMCudaTest(const std::vector<nanovdb::Coord>& coords)
{
    const uint64_t nCoords = coords.size();

    // --- Build ValueOnIndex grid on GPU ---
    nanovdb::Coord* d_coords = nullptr;
    cudaCheck(cudaMalloc(&d_coords, nCoords * sizeof(nanovdb::Coord)));
    cudaCheck(cudaMemcpy(d_coords, coords.data(),
        nCoords * sizeof(nanovdb::Coord), cudaMemcpyHostToDevice));

    auto handle = nanovdb::tools::cuda::voxelsToGrid<nanovdb::ValueOnIndex>(
        d_coords, nCoords);
    cudaCheck(cudaFree(d_coords));

    auto* d_grid = handle.deviceGrid<nanovdb::ValueOnIndex>();
    if (!d_grid) throw std::runtime_error("Failed to create device grid");

    // Download grid to host for validation and CPU build
    handle.deviceDownload();
    auto* h_grid = handle.grid<nanovdb::ValueOnIndex>();
    if (!h_grid) throw std::runtime_error("Failed to download host grid");

    const auto&    tree    = h_grid->tree();
    const uint64_t nVoxels = h_grid->activeVoxelCount();
    const uint32_t nBlocks = (uint32_t)((nVoxels + BlockWidth - 1) >> Log2BlockWidth);

    std::cout << "Active voxels (unique): " << nVoxels          << "\n"
              << "Leaf nodes            : " << tree.nodeCount(0) << "\n"
              << "Lower nodes           : " << tree.nodeCount(1) << "\n"
              << "Upper nodes           : " << tree.nodeCount(2) << "\n"
              << "VBM blocks            : " << nBlocks
              <<     "  (BlockWidth=" << BlockWidth << ")\n\n";

    // --- Benchmark VBM construction: GPU vs CPU ---
    // Allocate handles once; timing runs reuse the buffers (memset + kernel only,
    // no allocation overhead). First run per device serves as warmup - important
    // for unified-memory buffers where the first access triggers page migration.
    static constexpr int nRuns = 5;

    auto gpuHandle = nanovdb::tools::cuda::buildVoxelBlockManager<Log2BlockWidth>(d_grid);
    auto cpuHandle = nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(h_grid);

    // GPU build (cudaMemsetAsync + kernel, pre-allocated buffers)
    {
        float minMs = std::numeric_limits<float>::max();
        for (int i = 0; i < nRuns; ++i) {
            cudaCheck(cudaDeviceSynchronize()); // ensure stream is idle before timing
            nanovdb::util::cuda::Timer gpuTimer;
            nanovdb::tools::cuda::buildVoxelBlockManager<Log2BlockWidth>(d_grid, gpuHandle);
            float ms = gpuTimer.elapsed(); // records stop event and synchronizes
            if (i > 0) minMs = std::min(minMs, ms);
        }
        std::cout << "GPU buildVoxelBlockManager (memset+kernel): min " << minMs
                  << " ms  (over " << nRuns-1 << " post-warmup runs)\n";
    }

    // CPU build (std::memset + std::for_each(par), pre-allocated buffers)
    {
        float minMs = std::numeric_limits<float>::max();
        for (int i = 0; i < nRuns; ++i) {
            nanovdb::util::Timer cpuTimer;
            cpuTimer.start("");
            nanovdb::tools::buildVoxelBlockManager<Log2BlockWidth>(h_grid, cpuHandle);
            float ms = (float)cpuTimer.elapsed<std::chrono::microseconds>() / 1000.0f;
            if (i > 0) minMs = std::min(minMs, ms);
        }
        std::cout << "CPU buildVoxelBlockManager (memset+forEachPar): min " << minMs
                  << " ms  (over " << nRuns-1 << " post-warmup runs)\n\n";
    }

    // --- Validate CPU build against GPU build ---
    // Download GPU metadata to host and compare byte-for-byte with the CPU handle.
    {
        const uint64_t firstLeafIDBytes = gpuHandle.blockCount() * sizeof(uint32_t);
        const uint64_t jumpMapBytes     = gpuHandle.blockCount() * (BlockWidth / 64) * sizeof(uint64_t);

        std::vector<uint32_t> gpuFirstLeafID(gpuHandle.blockCount());
        std::vector<uint64_t> gpuJumpMap(gpuHandle.blockCount() * (BlockWidth / 64));

        cudaCheck(cudaMemcpy(gpuFirstLeafID.data(), gpuHandle.deviceFirstLeafID(),
            firstLeafIDBytes, cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(gpuJumpMap.data(), gpuHandle.deviceJumpMap(),
            jumpMapBytes, cudaMemcpyDeviceToHost));

        const bool firstLeafIDMatch = std::memcmp(gpuFirstLeafID.data(),
            cpuHandle.hostFirstLeafID(), firstLeafIDBytes) == 0;
        const bool jumpMapMatch = std::memcmp(gpuJumpMap.data(),
            cpuHandle.hostJumpMap(), jumpMapBytes) == 0;

        if (firstLeafIDMatch && jumpMapMatch)
            std::cout << "CPU/GPU metadata match: PASSED\n";
        else
            std::cerr << "CPU/GPU metadata match: FAILED"
                      << (firstLeafIDMatch ? "" : " [firstLeafID mismatch]")
                      << (jumpMapMatch     ? "" : " [jumpMap mismatch]") << "\n";
    }

    // gpuHandle is the last-built GPU VBM; use it for decode/validation below
    auto& vbmHandle = gpuHandle;

    // --- Decode all blocks into unified memory ---
    // Using thrust::universal_vector so the kernel writes directly into host-accessible
    // memory, avoiding a separate cudaMemcpy download step.
    thrust::universal_vector<uint32_t> outLeafIndex(nVoxels);
    thrust::universal_vector<uint16_t> outVoxelOffset(nVoxels);

    decodeAllBlocksKernel<<<nBlocks, BlockWidth>>>(
        d_grid,
        vbmHandle.deviceFirstLeafID(),
        vbmHandle.deviceJumpMap(),
        vbmHandle.firstOffset(), vbmHandle.lastOffset(), nBlocks,
        outLeafIndex.data().get(), outVoxelOffset.data().get());
    cudaCheckError();
    cudaCheck(cudaDeviceSynchronize());

    // --- Validate GPU decodeInverseMaps on host ---
    // For each active voxel index k+1, the decoded (leafIndex, voxelOffset)
    // must satisfy: leaf[leafIndex].getValue(voxelOffset) == k+1
    {
        const auto* firstLeaf = tree.getFirstNode<0>();
        uint64_t    errors    = 0;

        for (uint64_t k = 0; k < nVoxels; ++k) {
            const uint32_t leafIdx     = outLeafIndex[k];
            const uint16_t voxelOff    = outVoxelOffset[k];
            const uint64_t expectedIdx = k + vbmHandle.firstOffset();
            const uint64_t decodedIdx  = firstLeaf[leafIdx].getValue(voxelOff);

            if (decodedIdx != expectedIdx) {
                if (errors < 5)
                    std::cerr << "ERROR at k=" << k
                              << ": expected index " << expectedIdx
                              << ", decoded "        << decodedIdx
                              << "  (leaf=" << leafIdx
                              << " voxelOff=" << voxelOff << ")\n";
                ++errors;
            }
        }

        if (errors == 0)
            std::cout << "GPU decodeInverseMaps: PASSED (all " << nVoxels
                      << " entries validated)\n";
        else
            std::cerr << "GPU decodeInverseMaps: FAILED (" << errors << " / "
                      << nVoxels << " entries incorrect)\n";
    }

    // --- Validate CPU decodeInverseMaps against GPU results ---
    // cpuHandle metadata was already verified to agree with gpuHandle above.
    // Run the CPU decode block-by-block and compare against the unified-memory
    // GPU output (which is now host-accessible after the sync above).
    {
        const uint32_t* firstLeafIDPtr = cpuHandle.hostFirstLeafID();
        const uint64_t* jumpMapPtr     = cpuHandle.hostJumpMap();

        std::vector<uint32_t> cpuLeafIndex(BlockWidth);
        std::vector<uint16_t> cpuVoxelOffset(BlockWidth);
        uint64_t errors = 0;

        for (uint32_t bID = 0; bID < nBlocks; ++bID) {
            const uint64_t blockFirstOffset =
                cpuHandle.firstOffset() + (uint64_t)bID * BlockWidth;

            CPUVBM::decodeInverseMaps(
                h_grid,
                firstLeafIDPtr[bID],
                &jumpMapPtr[(uint64_t)bID * CPUVBM::JumpMapLength],
                blockFirstOffset,
                cpuLeafIndex.data(),
                cpuVoxelOffset.data());

            for (int tID = 0; tID < BlockWidth; ++tID) {
                const uint64_t k = (uint64_t)bID * BlockWidth + tID;
                if (k >= nVoxels) break;
                if (cpuLeafIndex[tID] != outLeafIndex[k] ||
                    cpuVoxelOffset[tID] != outVoxelOffset[k]) {
                    if (errors < 5)
                        std::cerr << "CPU/GPU decodeInverseMaps mismatch at k=" << k
                                  << ": cpu=(" << cpuLeafIndex[tID] << ","
                                  << cpuVoxelOffset[tID] << ") gpu=("
                                  << outLeafIndex[k] << "," << outVoxelOffset[k]
                                  << ")\n";
                    ++errors;
                }
            }
        }

        if (errors == 0)
            std::cout << "CPU decodeInverseMaps vs GPU: PASSED\n";
        else
            std::cerr << "CPU decodeInverseMaps vs GPU: FAILED ("
                      << errors << " mismatches)\n";
    }

    // --- GPU decodeInverseMaps performance benchmark ---
    // benchDecodeKernel decodes into shared memory only; one token uint32_t per
    // block is written to dummyOut to prevent dead code elimination.
    {
        thrust::universal_vector<uint32_t> dummyOut(nBlocks, 0);

        static constexpr int nPerfRuns = 20;

        // Warmup launch (first kernel invocation may incur JIT / page-migration cost)
        benchDecodeKernel<<<nBlocks, BlockWidth>>>(
            d_grid,
            vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
            vbmHandle.firstOffset(), vbmHandle.lastOffset(), nBlocks,
            dummyOut.data().get());
        cudaCheck(cudaDeviceSynchronize());

        std::cout << "\nGPU decodeInverseMaps performance ("
                  << nBlocks << " blocks, " << nVoxels << " voxels):\n";

        for (int run = 0; run < nPerfRuns; ++run) {
            cudaCheck(cudaDeviceSynchronize());
            nanovdb::util::cuda::Timer gpuTimer;
            benchDecodeKernel<<<nBlocks, BlockWidth>>>(
                d_grid,
                vbmHandle.deviceFirstLeafID(), vbmHandle.deviceJumpMap(),
                vbmHandle.firstOffset(), vbmHandle.lastOffset(), nBlocks,
                dummyOut.data().get());
            const float ms = gpuTimer.elapsed();
            std::cout << "  run " << run << ": " << ms << " ms\n";
        }

        // Sum dummyOut on host (unified memory, already sync'd by last gpuTimer.elapsed())
        uint64_t dummy = 0;
        for (uint32_t i = 0; i < nBlocks; ++i) dummy += dummyOut[i];
        std::cout << "  (dummy=" << dummy << ")\n";
    }

    // --- CPU decodeInverseMaps performance benchmark ---
    // Scratch arrays are reused across all blocks so the only memory traffic is
    // reading VBM metadata and leaf data -- not writing nVoxels of output.
    // Token reads of scratch[0] after each block defeat dead code elimination
    // without adding meaningful cost.
    {
        const uint32_t* firstLeafIDPtr = cpuHandle.hostFirstLeafID();
        const uint64_t* jumpMapPtr     = cpuHandle.hostJumpMap();

        static constexpr int nPerfRuns = 20;

        // --- Sanity check: countOn + std::fill only, no tree access ---
        // If OpenMP parallelism is working, this should scale near-linearly.
        {
            std::cout << "\nCPU decodeInverseMaps sanity (countOn only, "
                      << omp_get_max_threads() << " OMP threads):\n";

            // Never true: nExtraLeaves <= JumpMapLength*64, so == JumpMapLength*64+1 is impossible.
            // Used only to prevent dead code elimination without a reduction.
            volatile uint32_t dummy = 0;

            for (int run = 0; run < nPerfRuns; ++run) {
                nanovdb::util::Timer timer;
                timer.start("");

                #pragma omp parallel
                {
                    #pragma omp for schedule(static)
                    for (uint32_t bID = 0; bID < nBlocks; ++bID) {
                        int nExtraLeaves = 0;
                        for (int i = 0; i < CPUVBM::JumpMapLength; i++)
                            nExtraLeaves += nanovdb::util::countOn(
                                jumpMapPtr[(uint64_t)bID * CPUVBM::JumpMapLength + i]);

                        // Reinterpret the first leaf's 8 x uint64_t valueMask as
                        // 16 x uint32_t words, one per group of 32 consecutive voxels.
                        const auto& leaf =
                            h_grid->tree().getFirstNode<0>()[firstLeafIDPtr[bID]];
                        const uint32_t* maskWords =
                            reinterpret_cast<const uint32_t*>(leaf.valueMask().words());

                        // Phase 1: per-word inclusive prefix counts.
                        // prefixCountRealigned[step][lane] = popcount(maskWords[lane] & mask)
                        // where mask covers bits 0..step (inclusive).
                        // At step=31, mask=0xFFFFFFFF so row[31] == wordPopcount[lane].
                        // Safe mask form: (uint32_t(2) << step) - 1u avoids UB at step=31.
                        alignas(32) uint32_t prefixCountRealigned[32][16];
                        for (int step = 0; step < 32; step++) {
                            const uint32_t mask = (uint32_t(2) << step) - 1u;
                            #pragma omp simd
                            for (int lane = 0; lane < 16; lane++)
                                prefixCountRealigned[step][lane] =
                                    popcount32(maskWords[lane] & mask);
                        }

                        // Phase 2: exclusive prefix scan of row[31] -> baseOffset[lane],
                        // then add baseOffset to every row to get global prefix counts.
                        uint32_t baseOffset[16];
                        baseOffset[0] = 0;
                        for (int lane = 1; lane < 16; lane++)
                            baseOffset[lane] = baseOffset[lane-1] +
                                               prefixCountRealigned[31][lane-1];

                        for (int step = 0; step < 32; step++) {
                            #pragma omp simd
                            for (int lane = 0; lane < 16; lane++)
                                prefixCountRealigned[step][lane] += baseOffset[lane];
                        }

                        // Dummy: global prefix count at the last voxel equals the total
                        // active voxel count for this leaf, which is <= 512, never 513.
                        if (prefixCountRealigned[31][15] == 513u)
                            dummy = prefixCountRealigned[31][15];
                    }
                }

                const float ms =
                    (float)timer.elapsed<std::chrono::microseconds>() / 1000.0f;
                std::cout << "  run " << run << ": " << ms << " ms\n";
            }
        }

        uint32_t dummy = 0;

        std::cout << "\nCPU decodeInverseMaps performance ("
                  << nBlocks << " blocks, " << nVoxels << " voxels"
                  << ", " << omp_get_max_threads() << " OMP threads):\n";

        for (int run = 0; run < nPerfRuns; ++run) {
            nanovdb::util::Timer timer;
            timer.start("");

            #pragma omp parallel reduction(+:dummy)
            {
                #pragma omp for schedule(static)
                for (uint32_t bID = 0; bID < nBlocks; ++bID) {
                    const uint64_t blockFirstOffset =
                        cpuHandle.firstOffset() + (uint64_t)bID * BlockWidth;

                    CPUVBM::decodeInverseMaps(
                        h_grid,
                        firstLeafIDPtr[bID],
                        &jumpMapPtr[(uint64_t)bID * CPUVBM::JumpMapLength],
                        blockFirstOffset,
                        g_scratchLeafIndex,
                        g_scratchVoxelOffset);

                    dummy += (g_scratchLeafIndex[0]  != CPUVBM::UnusedLeafIndex)   ? 1u : 0u;
                    dummy += (g_scratchVoxelOffset[0] != CPUVBM::UnusedVoxelOffset) ? 1u : 0u;
                }
            }

            const float ms =
                (float)timer.elapsed<std::chrono::microseconds>() / 1000.0f;
            std::cout << "  run " << run << ": " << ms << " ms\n";
        }
        std::cout << "  (dummy=" << dummy << ")\n";
    }
}
