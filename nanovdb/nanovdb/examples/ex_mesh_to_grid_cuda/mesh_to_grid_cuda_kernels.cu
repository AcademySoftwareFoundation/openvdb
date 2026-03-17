// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0


#include <nanovdb/NanoVDB.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>

#include <openvdb/openvdb.h>

#include <algorithm>
#include <limits>
#include <vector>


template<typename BuildT>
void mainMeshToGrid(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid)
{
    // Test topology-only path (getHandle)
    {
        nanovdb::tools::cuda::MeshToGrid<BuildT> conv(devicePoints, pointCount, deviceTriangles, triangleCount, map);
        conv.setVerbose(1);
        auto handle = conv.getHandle();
        std::cout << "[getHandle] completed, buffer size: " << handle.bufferSize() << " bytes\n";
    }

    // Initialize mesh-to-grid converter
    nanovdb::tools::cuda::MeshToGrid<BuildT> converter( devicePoints, pointCount, deviceTriangles, triangleCount, map );
    converter.setVerbose(1);
    auto [handle, sidecar] = converter.getHandleAndUDF();

    // --- Comparison against OpenVDB reference ---
    // Download NanoVDB grid to host for CPU-side analysis.
    std::vector<char> hostBuf(handle.bufferSize());
    cudaCheck(cudaMemcpy(hostBuf.data(), handle.deviceData(), handle.bufferSize(), cudaMemcpyDeviceToHost));
    const auto *nanoGrid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBuf.data());

    // Download sidecar.
    const uint64_t sidecarFloats = sidecar.size() / sizeof(float);
    std::vector<float> hostSidecar(sidecarFloats);
    cudaCheck(cudaMemcpy(hostSidecar.data(), sidecar.deviceData(),
                         sidecar.size(), cudaMemcpyDeviceToHost));

    // Encode a voxel coord as a sortable int64_t key.
    auto encodeCoord = [](int x, int y, int z) -> int64_t {
        return (int64_t(x) + (1<<20))
             | ((int64_t(y) + (1<<20)) << 21)
             | ((int64_t(z) + (1<<20)) << 42);
    };

    auto ovdbAcc = refGrid->getConstAccessor();
    const float voxelSize = (float)refGrid->voxelSize()[0];

    uint64_t falsePositives = 0, falseNegatives = 0;
    float minMissedUDF = std::numeric_limits<float>::max();
    uint64_t badUDF = 0;
    float maxErrVoxels = 0.f;

    std::vector<int64_t> ourCoords;
    ourCoords.reserve(12000000);

    const uint32_t nanoLeafCount = nanoGrid->tree().nodeCount(0);
    const auto *leaves = nanoGrid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < nanoLeafCount; ++li) {
        const auto &leaf = leaves[li];
        const auto origin = leaf.origin();
        for (uint32_t vi = 0; vi < 512; ++vi) {
            if (!leaf.isActive(vi)) continue;
            const int lx = vi & 7, ly = (vi >> 3) & 7, lz = (vi >> 6) & 7;
            const int x = origin[0]+lx, y = origin[1]+ly, z = origin[2]+lz;
            ourCoords.push_back(encodeCoord(x, y, z));

            const openvdb::Coord coord(x, y, z);
            if (!ovdbAcc.isValueOn(coord)) {
                ++falsePositives;
            } else {
                const float err = std::abs(hostSidecar[leaf.getValue(vi)] - ovdbAcc.getValue(coord)) / voxelSize;
                maxErrVoxels = std::max(maxErrVoxels, err);
                if (err > 0.1f) ++badUDF;
            }
        }
    }
    std::sort(ourCoords.begin(), ourCoords.end());

    for (auto it = refGrid->tree().cbeginValueOn(); it; ++it) {
        const auto c = it.getCoord();
        if (!std::binary_search(ourCoords.begin(), ourCoords.end(), encodeCoord(c[0], c[1], c[2]))) {
            ++falseNegatives;
            minMissedUDF = std::min(minMissedUDF, *it);
        }
    }

    const uint64_t ourActiveCount = ourCoords.size();
    const uint64_t nTP = ourActiveCount - falsePositives;
    std::cout << "\n--- Comparison against OpenVDB reference ---\n";
    std::cout << "Active voxels: ours=" << ourActiveCount
              << "  OpenVDB=" << refGrid->tree().activeVoxelCount() << "\n";
    std::cout << "False negatives: " << falseNegatives;
    if (falseNegatives > 0)
        std::cout << "  (smallest missed UDF: " << minMissedUDF << ")";
    std::cout << "\n";
    std::cout << "False positives: " << falsePositives
              << " (" << (100.0 * falsePositives / ourActiveCount) << "%)\n";
    std::cout << "UDF vs OpenVDB (true positives=" << nTP << "): "
              << "max error=" << maxErrVoxels << " voxels, "
              << "exceeding 0.1 voxels: " << badUDF << "\n";

}

template
void mainMeshToGrid<nanovdb::ValueOnIndex>(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid);
