// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0


#include <nanovdb/NanoVDB.h>

#include <nanovdb/tools/cuda/MeshToGrid.cuh>

#include <openvdb/openvdb.h>

#include <algorithm>
#include <limits>
#include <array>
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
    // Initialize mesh-to-grid converter
    nanovdb::tools::cuda::MeshToGrid<BuildT> converter( devicePoints, pointCount, deviceTriangles, triangleCount, map );
    converter.setVerbose(1);
    auto [handle, sidecar] = converter.getHandleAndUDF();



    // --- Voxel-level UDF correctness check ---
    // Download NanoVDB grid to host for CPU-side analysis
    std::vector<char> hostBuf(handle.bufferSize());
    cudaCheck(cudaMemcpy(hostBuf.data(), handle.deviceData(), handle.bufferSize(), cudaMemcpyDeviceToHost));
    const auto *nanoGrid = reinterpret_cast<const nanovdb::NanoGrid<BuildT>*>(hostBuf.data());

    std::cout << "\n--- Voxel-level UDF correctness check ---" << std::endl;

    // Encode a voxel coord as a sortable int64_t key.
    // Coords fit well within 20 bits per axis for typical meshes.
    auto encodeCoord = [](int x, int y, int z) -> int64_t {
        return (int64_t(x) + (1<<20))
             | ((int64_t(y) + (1<<20)) << 21)
             | ((int64_t(z) + (1<<20)) << 42);
    };

    // Single pass over our NanoVDB active voxels:
    //   - build sorted coord list (for false-negative lookup)
    //   - count false positives (active in ours, background in OpenVDB)
    auto ovdbAcc = refGrid->getConstAccessor();
    uint64_t falsePositives = 0;
    std::vector<int64_t> ourCoords;
    ourCoords.reserve(12000000);

    const uint32_t nanoLeafCount = nanoGrid->tree().nodeCount(0);
    const auto *leaves = nanoGrid->tree().getFirstLeaf();
    for (uint32_t li = 0; li < nanoLeafCount; ++li) {
        const auto &leaf = leaves[li];
        const auto origin = leaf.origin();
        for (uint32_t vi = 0; vi < 512; ++vi) {
            if (leaf.isActive(vi)) {
                const int lx = vi & 7, ly = (vi >> 3) & 7, lz = (vi >> 6) & 7;
                const int x = origin[0]+lx, y = origin[1]+ly, z = origin[2]+lz;
                ourCoords.push_back(encodeCoord(x, y, z));
                if (!ovdbAcc.isValueOn(openvdb::Coord(x, y, z)))
                    ++falsePositives;
            }
        }
    }
    std::sort(ourCoords.begin(), ourCoords.end());

    // False negatives: active in OpenVDB UDF but absent from our sorted set.
    uint64_t falseNegatives = 0;
    float minMissedUDF = std::numeric_limits<float>::max();
    for (auto it = refGrid->tree().cbeginValueOn(); it; ++it) {
        const auto c = it.getCoord();
        if (!std::binary_search(ourCoords.begin(), ourCoords.end(),
                                encodeCoord(c[0], c[1], c[2]))) {
            ++falseNegatives;
            minMissedUDF = std::min(minMissedUDF, *it);
        }
    }

    const uint64_t ourActiveCount = ourCoords.size();
    std::cout << "Our active voxels:     " << ourActiveCount << "\n";
    std::cout << "OpenVDB active voxels: " << refGrid->tree().activeVoxelCount() << "\n";
    if (falseNegatives == 0)
        std::cout << "False negatives: 0 -- all OpenVDB active voxels present in our output\n";
    else
        std::cerr << "False negatives: " << falseNegatives
                  << " (smallest missed UDF value: " << minMissedUDF << ")\n";
    std::cout << "False positives: " << falsePositives
              << " (" << (100.0 * falsePositives / ourActiveCount) << "% of our active voxels)\n";

    // --- UDF sidecar value check ---
    // Download sidecar and compare per-voxel UDF against OpenVDB reference.
    std::cout << "\n--- UDF sidecar value check ---" << std::endl;

    const uint64_t sidecarFloats = sidecar.size() / sizeof(float);
    std::vector<float> hostSidecar(sidecarFloats);
    cudaCheck(cudaMemcpy(hostSidecar.data(), sidecar.deviceData(),
                         sidecar.size(), cudaMemcpyDeviceToHost));

    // For each active voxel in our grid, compare our UDF against OpenVDB's.
    // Both our sidecar and OpenVDB store world-space distances; error is reported in voxels.
    const float voxelSize = (float)refGrid->voxelSize()[0];
    // Split histogram between true positives (active in both) and false positives
    // (active in ours only). For false positives, refUDF = background = mBandWidth,
    // so error = |ourUDF - mBandWidth|; a large error there is a real concern.
    const float thresholds[] = { 0.01f, 0.05f, 0.1f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f };
    constexpr int nThresh = sizeof(thresholds) / sizeof(thresholds[0]);
    uint64_t badTP[nThresh] = {}, badFP[nThresh] = {};
    float maxErrTP = 0.f, maxErrFP = 0.f;

    // Track worst voxels for CPU ground-truth verification.
    // topNTP is maintained as a min-heap (smallest error at front) capped at N,
    // so we always evict the least-bad entry when a worse one arrives.
    struct WorstVoxel { int x, y, z; float ourUDF, refUDF, err; bool isFP; };
    constexpr int N = 20;
    auto minHeapCmp = [](const WorstVoxel& a, const WorstVoxel& b){ return a.err > b.err; };
    std::vector<WorstVoxel> topNTP;
    topNTP.reserve(N + 1);
    WorstVoxel worstFP{};

    for (uint32_t li = 0; li < nanoLeafCount; ++li) {
        const auto &leaf = leaves[li];
        const auto origin = leaf.origin();
        for (uint32_t vi = 0; vi < 512; ++vi) {
            if (!leaf.isActive(vi)) continue;
            const int lx = vi & 7, ly = (vi >> 3) & 7, lz = (vi >> 6) & 7;
            const int x = origin[0]+lx, y = origin[1]+ly, z = origin[2]+lz;

            const openvdb::Coord coord(x, y, z);
            const uint64_t sidecarIdx = leaf.getValue(vi);
            const float ourUDF_world = hostSidecar[sidecarIdx];
            const float refUDF       = ovdbAcc.getValue(coord);
            const float err = std::abs(ourUDF_world - refUDF) / voxelSize;

            const WorstVoxel candidate{ x, y, z, ourUDF_world / voxelSize,
                                        refUDF / voxelSize, err, !ovdbAcc.isValueOn(coord) };

            if (ovdbAcc.isValueOn(coord)) {
                maxErrTP = std::max(maxErrTP, err);
                for (int t = 0; t < nThresh; ++t)
                    if (err > thresholds[t]) ++badTP[t];
                if ((int)topNTP.size() < N || err > topNTP.front().err) {
                    topNTP.push_back(candidate);
                    std::push_heap(topNTP.begin(), topNTP.end(), minHeapCmp);
                    if ((int)topNTP.size() > N) {
                        std::pop_heap(topNTP.begin(), topNTP.end(), minHeapCmp);
                        topNTP.pop_back();
                    }
                }
            } else {
                if (err > worstFP.err) worstFP = candidate;
                maxErrFP = std::max(maxErrFP, err);
                for (int t = 0; t < nThresh; ++t)
                    if (err > thresholds[t]) ++badFP[t];
            }
        }
    }
    // Sort top-N true positives from worst to best
    std::sort(topNTP.begin(), topNTP.end(), [](const WorstVoxel& a, const WorstVoxel& b){
        return a.err > b.err; });

    const uint64_t nTP = ourCoords.size() - falsePositives;
    std::cout << "True positives (" << nTP << " voxels), max error: " << maxErrTP << " voxels\n";
    for (int t = 0; t < nThresh; ++t)
        std::cout << "  > " << thresholds[t] << " voxels: " << badTP[t] << "\n";
    std::cout << "False positives (" << falsePositives << " voxels), max error: " << maxErrFP << " voxels\n";
    for (int t = 0; t < nThresh; ++t)
        std::cout << "  > " << thresholds[t] << " voxels: " << badFP[t] << "\n";

    // --- CPU ground-truth check: brute-force both NanoVDB and Ericson routines ---
    struct CPUResult {
        float nvdbUDF, ericsonUDF;   // voxel units
        int   nearestTri;            // index into deviceTriangles
        nanovdb::Vec3f voxelCenterWorld;
    };
    auto cpuBruteForce = [&](const WorstVoxel& wv) -> CPUResult {
        const nanovdb::Vec3f wcNvdb = map.applyMap(
            nanovdb::Vec3f(float(wv.x), float(wv.y), float(wv.z)));
        const openvdb::Vec3s wcOvdb(wcNvdb[0], wcNvdb[1], wcNvdb[2]);

        float nvdbMinDistSqr = std::numeric_limits<float>::max();
        float refMinDistSqr  = std::numeric_limits<float>::max();
        int   nearestTri = -1;
        for (int t = 0; t < triangleCount; ++t) {
            const nanovdb::Vec3f nv0 = devicePoints[deviceTriangles[t][0]];
            const nanovdb::Vec3f nv1 = devicePoints[deviceTriangles[t][1]];
            const nanovdb::Vec3f nv2 = devicePoints[deviceTriangles[t][2]];

            const float d = nanovdb::math::pointToTriangleDistSqr(nv0, nv1, nv2, wcNvdb);
            if (d < nvdbMinDistSqr) { nvdbMinDistSqr = d; nearestTri = t; }

            // Reference: Ericson §5.1.5, openvdb::Vec3s arithmetic
            const openvdb::Vec3s a(nv0[0], nv0[1], nv0[2]);
            const openvdb::Vec3s b(nv1[0], nv1[1], nv1[2]);
            const openvdb::Vec3s c(nv2[0], nv2[1], nv2[2]);
            const openvdb::Vec3s p = wcOvdb;
            const openvdb::Vec3s ab = b-a, ac = c-a, ap = p-a;
            const float d1 = ab.dot(ap), d2 = ac.dot(ap);
            if (d1 <= 0.f && d2 <= 0.f) { refMinDistSqr = std::min(refMinDistSqr, (p-a).lengthSqr()); continue; }
            const openvdb::Vec3s bp = p-b;
            const float d3 = ab.dot(bp), d4 = ac.dot(bp);
            if (d3 >= 0.f && d4 <= d3)  { refMinDistSqr = std::min(refMinDistSqr, (p-b).lengthSqr()); continue; }
            const openvdb::Vec3s cp = p-c;
            const float d5 = ab.dot(cp), d6 = ac.dot(cp);
            if (d6 >= 0.f && d5 <= d6)  { refMinDistSqr = std::min(refMinDistSqr, (p-c).lengthSqr()); continue; }
            const float vc = d1*d4 - d3*d2;
            if (vc <= 0.f && d1 >= 0.f && d3 <= 0.f) {
                float v = d1/(d1-d3);
                refMinDistSqr = std::min(refMinDistSqr, (p-(a+v*ab)).lengthSqr()); continue; }
            const float vb = d5*d2 - d1*d6;
            if (vb <= 0.f && d2 >= 0.f && d6 <= 0.f) {
                float w = d2/(d2-d6);
                refMinDistSqr = std::min(refMinDistSqr, (p-(a+w*ac)).lengthSqr()); continue; }
            const float va = d3*d6 - d5*d4;
            if (va <= 0.f && (d4-d3) >= 0.f && (d5-d6) >= 0.f) {
                float w = (d4-d3)/((d4-d3)+(d5-d6));
                refMinDistSqr = std::min(refMinDistSqr, (p-(b+w*(c-b))).lengthSqr()); continue; }
            const float denom = 1.f/(va+vb+vc);
            const float rv = vb*denom, rw = vc*denom;
            refMinDistSqr = std::min(refMinDistSqr, (p-(a+rv*ab+rw*ac)).lengthSqr());
        }
        return { std::sqrt(nvdbMinDistSqr) / voxelSize,
                 std::sqrt(refMinDistSqr)  / voxelSize,
                 nearestTri, wcNvdb };
    };

    auto printVec3 = [](const char* label, float x, float y, float z) {
        std::cout << "    " << label << ": (" << x << ", " << y << ", " << z << ")\n";
    };

    auto printWorst = [&](const char* label, const WorstVoxel& wv) {
        std::cout << "\n--- " << label << " ---\n";
        std::cout << "  Our UDF     : " << wv.ourUDF << " voxels\n";
        std::cout << "  OpenVDB UDF : " << wv.refUDF << " voxels"
                  << (wv.isFP ? " (background — not active in OpenVDB)" : "") << "\n";
        std::cout << "  Error       : " << wv.err << " voxels\n";
        auto res = cpuBruteForce(wv);
        std::cout << "  CPU (NanoVDB routine):   " << res.nvdbUDF    << " voxels\n";
        std::cout << "  CPU (Ericson reference): " << res.ericsonUDF << " voxels\n";
        std::cout << "  Routine discrepancy: " << std::abs(res.nvdbUDF - res.ericsonUDF) << " voxels\n";
        std::cout << "  Error vs CPU: our=" << std::abs(wv.ourUDF - res.nvdbUDF)
                  << "  OpenVDB ref=" << std::abs(wv.refUDF - res.nvdbUDF) << " voxels\n";
        // World-space geometry
        std::cout << "  World-space geometry:\n";
        printVec3("voxel center", res.voxelCenterWorld[0], res.voxelCenterWorld[1], res.voxelCenterWorld[2]);
        if (res.nearestTri >= 0) {
            const nanovdb::Vec3f v0 = devicePoints[deviceTriangles[res.nearestTri][0]];
            const nanovdb::Vec3f v1 = devicePoints[deviceTriangles[res.nearestTri][1]];
            const nanovdb::Vec3f v2 = devicePoints[deviceTriangles[res.nearestTri][2]];
            std::cout << "    nearest tri #" << res.nearestTri << ":\n";
            printVec3("  v0", v0[0], v0[1], v0[2]);
            printVec3("  v1", v1[0], v1[1], v1[2]);
            printVec3("  v2", v2[0], v2[1], v2[2]);
        }
    };

    for (int i = 0; i < (int)topNTP.size(); ++i) {
        std::string label = "True positive #" + std::to_string(i+1)
                          + " (error " + std::to_string(topNTP[i].err) + " voxels)";
        printWorst(label.c_str(), topNTP[i]);
    }
    printWorst("Worst false positive", worstFP);

}

template
void mainMeshToGrid<nanovdb::ValueOnIndex>(
    const nanovdb::Vec3f *devicePoints,
    const int pointCount,
    const nanovdb::Vec3i *deviceTriangles,
    const int triangleCount,
    const nanovdb::Map map,
    const openvdb::FloatGrid::Ptr refGrid);
