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

#include <algorithm>
#include <vector>
#include <iostream>

namespace {

static constexpr int BlockWidthLog2 = 7;
static constexpr int BlockWidth     = 1 << BlockWidthLog2; // 128

using VBM = nanovdb::tools::cuda::VoxelBlockManager<BlockWidth>;

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
    const uint32_t nBlocks = (uint32_t)((nVoxels + BlockWidth - 1) >> BlockWidthLog2);

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

    auto gpuHandle = nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(d_grid);
    auto cpuHandle = nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2>(h_grid);

    // GPU build (cudaMemsetAsync + kernel, pre-allocated buffers)
    {
        float minMs = std::numeric_limits<float>::max();
        for (int i = 0; i < nRuns; ++i) {
            cudaCheck(cudaDeviceSynchronize()); // ensure stream is idle before timing
            nanovdb::util::cuda::Timer gpuTimer;
            nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(d_grid, gpuHandle);
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
            nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2>(h_grid, cpuHandle);
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

    // --- Allocate output arrays on GPU ---
    uint32_t* d_outLeafIndex   = nullptr;
    uint16_t* d_outVoxelOffset = nullptr;
    cudaCheck(cudaMalloc(&d_outLeafIndex,   nVoxels * sizeof(uint32_t)));
    cudaCheck(cudaMalloc(&d_outVoxelOffset, nVoxels * sizeof(uint16_t)));

    // --- Decode all blocks ---
    decodeAllBlocksKernel<<<nBlocks, BlockWidth>>>(
        d_grid,
        vbmHandle.deviceFirstLeafID(),
        vbmHandle.deviceJumpMap(),
        vbmHandle.firstOffset(), vbmHandle.lastOffset(), nBlocks,
        d_outLeafIndex, d_outVoxelOffset);
    cudaCheckError();

    // --- Download results ---
    std::vector<uint32_t> outLeafIndex(nVoxels);
    std::vector<uint16_t> outVoxelOffset(nVoxels);
    cudaCheck(cudaMemcpy(outLeafIndex.data(), d_outLeafIndex,
        nVoxels * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(outVoxelOffset.data(), d_outVoxelOffset,
        nVoxels * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // --- Validate on host ---
    // For each active voxel index k+1, the decoded (leafIndex, voxelOffset)
    // must satisfy: leaf[leafIndex].getValue(voxelOffset) == k+1
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
        std::cout << "PASSED: all " << nVoxels
                  << " inverse map entries validated\n";
    else
        std::cerr << "FAILED: " << errors << " / " << nVoxels
                  << " entries incorrect\n";

    cudaCheck(cudaFree(d_outLeafIndex));
    cudaCheck(cudaFree(d_outVoxelOffset));
}
