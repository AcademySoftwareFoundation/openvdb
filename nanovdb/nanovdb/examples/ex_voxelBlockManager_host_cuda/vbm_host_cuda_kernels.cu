// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file vbm_host_cuda_kernels.cu

    \brief CUDA implementation for the VoxelBlockManager test harness.

    Builds the VoxelBlockManager from a ValueOnIndex grid, decodes the full
    inverse map (leafIndex, voxelOffset) for all active voxels, downloads the
    result to the host, and validates it against the grid structure.
*/

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/cuda/PointsToGrid.cuh>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/util/cuda/Util.h>

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

    // Download grid to host for validation
    handle.deviceDownload();
    auto* h_grid = handle.grid<nanovdb::ValueOnIndex>();
    if (!h_grid) throw std::runtime_error("Failed to download host grid");

    const auto&    tree        = h_grid->tree();
    const uint64_t nVoxels     = h_grid->activeVoxelCount();
    const auto     nLowerNodes = tree.nodeCount(1);
    const uint64_t firstOffset = 1;
    const uint64_t lastOffset  = nVoxels;
    const uint32_t nBlocks     = (uint32_t)((nVoxels + BlockWidth - 1) >> BlockWidthLog2);

    std::cout << "Active voxels (unique): " << nVoxels          << "\n"
              << "Leaf nodes            : " << tree.nodeCount(0) << "\n"
              << "Lower nodes           : " << nLowerNodes       << "\n"
              << "Upper nodes           : " << tree.nodeCount(2) << "\n"
              << "VBM blocks            : " << nBlocks
              <<     "  (BlockWidth=" << BlockWidth << ")\n";

    // --- Build VoxelBlockManager on GPU ---
    nanovdb::tools::VoxelBlockManagerHandle<nanovdb::cuda::DeviceBuffer> vbmHandle;
    vbmHandle.deviceResize<BlockWidthLog2>(nBlocks);
    vbmHandle.setOffsets(firstOffset, lastOffset);
    vbmHandle.setLowerCount(nLowerNodes);

    nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2>(
        vbmHandle.firstOffset(), vbmHandle.lastOffset(),
        vbmHandle.blockCount(),  vbmHandle.lowerCount(),
        d_grid,
        vbmHandle.deviceFirstLeafID(),
        vbmHandle.deviceJumpMap());

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
        firstOffset, lastOffset, nBlocks,
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
        const uint64_t expectedIdx = k + firstOffset;
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
