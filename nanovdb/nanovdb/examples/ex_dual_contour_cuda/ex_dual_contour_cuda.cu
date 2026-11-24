// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file  ex_dual_contour_cuda.cu
///
/// @brief Extracts triangle meshes from narrow-band level sets with nanovdb::tools::cuda::DualContour,
///        first for one grid and then for a batch of grids meshed together through a batched
///        VoxelBlockManager. The level sets (a torus and a sphere) are built on the host, converted
///        to ValueOnIndex grids whose signed distances are kept in a sidecar indexed by value index,
///        and copied to the device.
///
///        Usage: ex_dual_contour_cuda [output.obj]   (writes the batch's meshes as OBJ groups)

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/cuda/GridHandle.cuh>              // cuda::copyTo
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh> // buildVoxelBlockManagerBatch
#include <nanovdb/tools/cuda/DualContour.cuh>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using BuildT = nanovdb::ValueOnIndex;

/// @brief A level set as a device ValueOnIndex grid and a device sidecar of its values.
struct DeviceLevelSet
{
    nanovdb::GridHandle<nanovdb::cuda::Buffer<std::byte>> grid;
    nanovdb::cuda::Buffer<float> sidecar;
    uint64_t voxelCount;
    uint32_t leafCount;
};

DeviceLevelSet upload(const nanovdb::GridHandle<>& floatHandle)
{
    // index only the active voxels, without stats or tiles, and keep their values as one channel
    auto indexHandle = nanovdb::tools::createNanoGrid<nanovdb::NanoGrid<float>, BuildT>(
        *floatHandle.grid<float>(), 1u, false, false);
    const auto* grid = indexHandle.grid<BuildT>();
    nanovdb::cuda::Buffer<float> sidecar(cudaStream_t(0), grid->valueCount(), nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpy(sidecar.data(), grid->getBlindData<float>(0), grid->valueCount() * sizeof(float),
                         cudaMemcpyHostToDevice));
    return {nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(indexHandle, cudaStream_t(0)), std::move(sidecar),
            grid->activeVoxelCount(), grid->tree().nodeCount(0)};
}

template<typename MeshT>
void report(const std::string& name, const MeshT& mesh, uint32_t g, bool closed = true)
{
    const uint64_t points = mesh.hostPointOffsets[g + 1] - mesh.hostPointOffsets[g];
    const uint64_t triangles = mesh.hostTriangleOffsets[g + 1] - mesh.hostTriangleOffsets[g];
    std::cout << "  " << name << ": " << points << " vertices, " << triangles << " triangles";
    // a closed triangle mesh has 3T/2 edges, so its Euler characteristic is V - T/2 (decimation
    // can leave non-manifold edges, so this is reported only at full resolution)
    if (closed) std::cout << ", Euler characteristic " << int64_t(points) - int64_t(triangles / 2);
    std::cout << "\n";
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        const DeviceLevelSet torus = upload(nanovdb::tools::createLevelSetTorus<float>(
            40.0, 15.0, nanovdb::Vec3d(0.0), 0.5, 3.0));
        const DeviceLevelSet sphere = upload(nanovdb::tools::createLevelSetSphere<float>(
            30.0, nanovdb::Vec3d(100.0, 0.0, 0.0), 0.5, 3.0));

        // One grid: the sidecar is indexed by the grid's value indices.
        nanovdb::tools::cuda::DualContour<BuildT> dualContour;
        {
            const auto mesh = dualContour.getMesh(torus.grid.deviceGrid<BuildT>(), torus.sidecar.data());
            std::cout << "Single grid:\n";
            report("torus", mesh, 0);
            // cluster-collapse decimation: uniform 2x2x2 blocks, then curvature-adaptive blocks
            dualContour.setReduce(2);
            report("torus, reduce 2", dualContour.getMesh(torus.grid.deviceGrid<BuildT>(), torus.sidecar.data()), 0, false);
            dualContour.setReduce(1);
            dualContour.setAdaptivity(0.5f);
            report("torus, adaptivity 0.5", dualContour.getMesh(torus.grid.deviceGrid<BuildT>(), torus.sidecar.data()), 0, false);
            dualContour.setAdaptivity(0.f);
        }

        // A batch: one batched VoxelBlockManager over both grids. Each grid's blocks follow the
        // previous grid's; its field is its own sidecar, at the offsets given below.
        constexpr int Log2BlockWidth = 7;
        using VBM = nanovdb::tools::cuda::VoxelBlockManager<Log2BlockWidth>;
        const DeviceLevelSet* levelSets[2] = {&torus, &sphere};
        std::vector<nanovdb::tools::cuda::VoxelBlockManagerBatchEntry> entries;
        uint64_t blockCount = 0;
        uint32_t leafCount = 0;
        for (const DeviceLevelSet* ls : levelSets) {
            entries.push_back({ls->grid.deviceGrid<BuildT>(), ls->voxelCount, blockCount, leafCount});
            blockCount += (ls->voxelCount + VBM::BlockWidth - 1) / VBM::BlockWidth;
            leafCount += ls->leafCount;
        }
        // the two sidecars back to back, each keeping its entry 0
        nanovdb::cuda::Buffer<float> field(cudaStream_t(0), torus.sidecar.size() + sphere.sidecar.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(field.data(), torus.sidecar.data(), torus.sidecar.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        cudaCheck(cudaMemcpy(field.data() + torus.sidecar.size(), sphere.sidecar.data(), sphere.sidecar.size() * sizeof(float),
                             cudaMemcpyDeviceToDevice));
        const std::vector<int64_t> fieldOffsets = {0, int64_t(torus.sidecar.size())};

        nanovdb::cuda::Buffer<nanovdb::tools::cuda::VoxelBlockManagerBatchEntry> d_entries(cudaStream_t(0), entries.size(), nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<int64_t> d_fieldOffsets(cudaStream_t(0), fieldOffsets.size(), nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<uint32_t> firstLeafID(cudaStream_t(0), blockCount, nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<uint64_t> jumpMap(cudaStream_t(0), blockCount * VBM::JumpMapLength, nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpy(d_entries.data(), entries.data(), entries.size() * sizeof(entries[0]), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_fieldOffsets.data(), fieldOffsets.data(), fieldOffsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        nanovdb::tools::cuda::buildVoxelBlockManagerBatch<Log2BlockWidth>(
            d_entries.data(), uint32_t(entries.size()), leafCount, blockCount, firstLeafID.data(), jumpMap.data());
        const nanovdb::tools::cuda::VoxelBlockManagerBatchView<Log2BlockWidth> view(
            nanovdb::tools::cuda::VoxelBlockManagerBatchEntryArray{d_entries.data()}, uint32_t(entries.size()),
            blockCount, firstLeafID.data(), jumpMap.data());

        const auto mesh = dualContour.getMesh(view, nanovdb::tools::cuda::VoxelBlockManagerBatchEntryArray{entries.data()},
                                              field.data(), d_fieldOffsets.data());
        std::cout << "Batch of " << mesh.meshCount() << " grids:\n";
        report("torus", mesh, 0);
        report("sphere", mesh, 1);

        if (argc > 1) { // write the batch's meshes as two OBJ groups
            std::vector<nanovdb::Vec3f> points(mesh.pointCount());
            std::vector<nanovdb::Vec3i> triangles(mesh.triangleCount());
            cudaCheck(cudaMemcpy(points.data(), mesh.points.data(), points.size() * sizeof(nanovdb::Vec3f), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemcpy(triangles.data(), mesh.triangles.data(), triangles.size() * sizeof(nanovdb::Vec3i), cudaMemcpyDeviceToHost));
            std::ofstream obj(argv[1]);
            for (const auto& p : points) obj << "v " << p[0] << " " << p[1] << " " << p[2] << "\n";
            for (uint32_t g = 0; g < mesh.meshCount(); ++g) {
                obj << "g grid" << g << "\n";
                const uint64_t base = mesh.hostPointOffsets[g] + 1; // OBJ indices are 1-based
                for (uint64_t t = mesh.hostTriangleOffsets[g]; t < mesh.hostTriangleOffsets[g + 1]; ++t)
                    obj << "f " << base + triangles[t][0] << " " << base + triangles[t][1] << " " << base + triangles[t][2] << "\n";
            }
            std::cout << "Wrote " << argv[1] << "\n";
        }
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"\n";
        return 1;
    }
    return 0;
}
