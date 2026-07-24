// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  connected_components_cuda_kernels.cu
///
/// @brief CUDA / NanoVDB side of the connected-components example. Rasterizes a triangle mesh into a
///        ValueOnIndex narrow-band grid (nanovdb::tools::cuda::MeshToGrid), optionally discards the
///        surface/barrier shell (unsigned distance within sqrt(3)/2 voxels of the surface) with
///        nanovdb::tools::cuda::PruneGrid, and runs connected-components labeling with
///        nanovdb::tools::cuda::ConnectedComponents. A CPU union-find oracle independently verifies
///        the GPU labeling (component count + per-voxel partition).

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/DeviceBuffer.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>            // rasterize mesh -> ValueOnIndex + UDF
#include <nanovdb/tools/cuda/PruneGrid.cuh>             // topological prune (barrier removal)
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>  // the connected-components labeling
#include <nanovdb/util/cuda/Util.h>                     // operatorKernel, cudaCheck
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>       // DeviceGridTraits

#include <thrust/universal_vector.h>

#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

using BuildT      = nanovdb::ValueOnIndex;
using GridHandleT = nanovdb::GridHandle<nanovdb::cuda::DeviceBuffer>;
using Traits      = nanovdb::util::cuda::DeviceGridTraits<BuildT>;

constexpr int LEAF_SIZE = 512;  // 8^3

// Per-leaf retain-mask functor: a voxel is kept iff its unsigned distance to the surface exceeds the
// barrier threshold sqrt(3)/2 voxels (i.e. UDF^2 >= 0.75 * voxelSize^2 in world units). Removing the
// barrier shell splits each closed surface's narrow band into disjoint inner/outer shells, which is
// what connected components then labels. One CUDA block per leaf, one thread per voxel offset.
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = LEAF_SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const nanovdb::NanoGrid<BuildT>* d_grid,
                               const float*                     d_udf,          // UDF sidecar, WORLD units
                               float                            barrierSqWorld, // (sqrt(3)/2 * voxelSize)^2
                               nanovdb::Mask<3>*                d_dstLeafMasks)
    {
        const int leafID   = blockIdx.x;
        const int threadID = threadIdx.x;

        const auto& leaf       = d_grid->tree().getFirstNode<0>()[leafID];
        auto&       resultMask = d_dstLeafMasks[leafID];

        // Clear the leaf's mask words in parallel, then set the retain bits.
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();

        if (auto n = leaf.data()->getValue(threadID)) {  // n != 0 => active voxel
            const float udf = d_udf[n];
            if (udf * udf >= barrierSqWorld)              // retain non-barrier voxels
                resultMask.setOnAtomic(threadID);
        }
    }
};

// Pack a voxel coordinate into a sortable/ hashable int64 key (offset so negatives stay positive).
inline int64_t encodeCoord(const nanovdb::Coord& c)
{
    return  (int64_t(c[0]) + (1 << 20))
         | ((int64_t(c[1]) + (1 << 20)) << 21)
         | ((int64_t(c[2]) + (1 << 20)) << 42);
}

// CPU union-find oracle: independently label the derived grid's active voxels by 6-connectivity and
// verify the GPU result (a) has the same component count and (b) induces the same partition (two
// voxels share a GPU label iff the oracle puts them in the same component). Returns true on PASS.
bool validateAgainstOracle(const GridHandleT& derivedHandle, uint32_t leafCount, uint64_t active,
                           const uint32_t* d_labels, uint64_t gpuCount)
{
    // Download the derived grid blob + the per-voxel labels to the host.
    std::vector<char> blob(derivedHandle.bufferSize());
    cudaCheck(cudaMemcpy(blob.data(), derivedHandle.deviceData(), blob.size(), cudaMemcpyDeviceToHost));
    const auto* h_grid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(blob.data());

    if (active == 0) {
        std::cout << "CPU-oracle self-check: PASS (empty grid, 0 components)\n";
        return gpuCount == 0;
    }

    std::vector<uint32_t> labels(active + 1);
    cudaCheck(cudaMemcpy(labels.data(), d_labels, (active + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Gather active voxels: dense id -> {coord, gpu label}, plus a coord->id lookup for neighbours.
    std::unordered_map<int64_t, uint32_t> coordToId;
    std::vector<nanovdb::Coord>           idToCoord;
    std::vector<uint32_t>                 idToGpuLabel;
    coordToId.reserve(active * 2);
    idToCoord.reserve(active);
    idToGpuLabel.reserve(active);

    const auto* leaves = h_grid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < leafCount; ++li) {
        const auto& leaf = leaves[li];
        for (uint32_t n = 0; n < uint32_t(LEAF_SIZE); ++n) {
            if (!leaf.isActive(n)) continue;
            const nanovdb::Coord c = leaf.origin() + nanovdb::NanoLeaf<BuildT>::OffsetToLocalCoord(n);
            const uint64_t       slot = leaf.getValue(n);
            const uint32_t       id   = uint32_t(idToCoord.size());
            coordToId.emplace(encodeCoord(c), id);
            idToCoord.push_back(c);
            idToGpuLabel.push_back(labels[slot]);
        }
    }

    // Union-find over 6-connectivity. Visiting only +X/+Y/+Z reaches every undirected edge once.
    std::vector<uint32_t> parent(idToCoord.size());
    std::iota(parent.begin(), parent.end(), 0u);
    std::function<uint32_t(uint32_t)> find = [&](uint32_t x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        a = find(a); b = find(b);
        if (a != b) parent[a > b ? a : b] = (a < b ? a : b);
    };
    const nanovdb::Coord dirs[3] = { {1,0,0}, {0,1,0}, {0,0,1} };
    for (uint32_t id = 0; id < idToCoord.size(); ++id)
        for (const auto& d : dirs) {
            auto it = coordToId.find(encodeCoord(idToCoord[id] + d));
            if (it != coordToId.end()) unite(id, it->second);
        }

    // Count oracle components and check the GPU labels induce the same partition.
    std::unordered_map<uint32_t, uint32_t> rootToComp;   // oracle root -> dense component id
    for (uint32_t id = 0; id < parent.size(); ++id) {
        const uint32_t r = find(id);
        if (!rootToComp.count(r)) rootToComp.emplace(r, uint32_t(rootToComp.size()));
    }
    const uint64_t oracleCount = rootToComp.size();

    std::unordered_map<uint32_t, uint32_t> gpuToOracle;  // gpu label -> oracle component
    uint64_t partitionViolations = 0;
    for (uint32_t id = 0; id < parent.size(); ++id) {
        const uint32_t gl = idToGpuLabel[id];
        const uint32_t oc = rootToComp[find(id)];
        auto it = gpuToOracle.find(gl);
        if (it == gpuToOracle.end()) gpuToOracle.emplace(gl, oc);
        else if (it->second != oc)   ++partitionViolations;
    }
    const uint64_t gpuDistinctLabels = gpuToOracle.size();

    const bool pass = (gpuCount == oracleCount) &&
                      (gpuDistinctLabels == oracleCount) &&
                      (partitionViolations == 0);

    std::cout << "CPU-oracle self-check: " << (pass ? "PASS" : "FAIL")
              << "  (gpu=" << gpuCount << ", oracle=" << oracleCount
              << ", distinct gpu labels=" << gpuDistinctLabels
              << ", partition violations=" << partitionViolations << ")\n";
    return pass;
}

} // anonymous namespace

uint64_t connectedComponentsFromMesh(const std::vector<nanovdb::Vec3f>& points,
                                     const std::vector<nanovdb::Vec3i>& triangles,
                                     const nanovdb::Map&                map,
                                     float                              bandWidth,
                                     bool                               discardSurfaceVoxels)
{
    const cudaStream_t stream = 0;

    // ---- Step 1: rasterize the mesh -> ValueOnIndex narrow-band grid + UDF sidecar. ----
    thrust::universal_vector<nanovdb::Vec3f> dPoints(points.begin(), points.end());
    thrust::universal_vector<nanovdb::Vec3i> dTriangles(triangles.begin(), triangles.end());

    nanovdb::tools::cuda::MeshToGrid<BuildT> converter(
        dPoints.data().get(),    uint32_t(dPoints.size()),
        dTriangles.data().get(), uint32_t(dTriangles.size()), map);
    converter.setVerbose(1);
    converter.setNarrowBandWidth(bandWidth);
    auto [origHandle, udfSidecar] = converter.getHandleAndUDF();
    const auto* d_orig = origHandle.template deviceGrid<BuildT>();

    // ---- Step 2 (optional): discard the surface/barrier shell -> derived topology. ----
    // With the shell removed, each closed surface's band splits into disjoint inner/outer shells;
    // without it, connected components run on the full narrow band (one component per closed surface).
    GridHandleT                      derivedHandle;   // stays empty unless we prune
    const nanovdb::NanoGrid<BuildT>* d_cc = d_orig;    // grid connected components will label
    if (discardSurfaceVoxels) {
        // World-space voxel size from the map (uniform scale here, but read it generically).
        const nanovdb::Vec3d w0 = map.applyMap(nanovdb::Vec3d(0.0, 0.0, 0.0));
        const nanovdb::Vec3d wx = map.applyMap(nanovdb::Vec3d(1.0, 0.0, 0.0));
        const float    voxelSize      = float(wx[0] - w0[0]);
        const float    barrierSqWorld = 0.75f * voxelSize * voxelSize;  // (sqrt(3)/2 * voxelSize)^2
        const uint32_t srcLeafCount   = Traits::getTreeData(d_orig).mNodeCount[0];

        auto  retainMask   = nanovdb::cuda::DeviceBuffer::create(
            std::size_t(srcLeafCount) * sizeof(nanovdb::Mask<3>), nullptr, false);
        auto* d_retainMask = static_cast<nanovdb::Mask<3>*>(retainMask.deviceData());

        nanovdb::util::cuda::operatorKernel<UDFBarrierPruneMaskFunctor>
            <<<srcLeafCount, UDFBarrierPruneMaskFunctor::MaxThreadsPerBlock, 0, stream>>>(
                d_orig, static_cast<const float*>(udfSidecar.deviceData()), barrierSqWorld, d_retainMask);
        cudaCheckError();

        nanovdb::tools::cuda::PruneGrid<BuildT> pruner(d_orig, d_retainMask, stream);
        derivedHandle = pruner.getHandle();
        d_cc          = derivedHandle.template deviceGrid<BuildT>();
    }

    // ---- Step 3: connected-components labeling on the selected grid. ----
    nanovdb::tools::cuda::ConnectedComponents<BuildT> cc(d_cc, stream);
    auto [d_labels, numComponents] = cc.getVoxelLabelsAndCount();
    cudaCheck(cudaStreamSynchronize(stream));

    // Diagnostics + CPU-oracle self-check (on whichever grid was labeled).
    const GridHandleT& ccHandle = discardSurfaceVoxels ? derivedHandle : origHandle;
    const uint64_t     ccActive = Traits::getActiveVoxelCount(d_cc);
    const uint32_t     ccLeaves = Traits::getTreeData(d_cc).mNodeCount[0];
    std::cout << (discardSurfaceVoxels ? "Derived (barrier-removed) grid: " : "Full narrow-band grid: ")
              << ccActive << " active voxels, " << ccLeaves << " leaves.\n";

    validateAgainstOracle(ccHandle, ccLeaves, ccActive, d_labels, numComponents);

    return numComponents;
}
