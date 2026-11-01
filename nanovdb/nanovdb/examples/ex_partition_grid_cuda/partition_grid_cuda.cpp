// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/LevelSetUtil.h> // for openvdb::tools::segmentActiveVoxels
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/HandleStorage.h> // host-includable: cuda::copyTo transfers grids without any kernel
#include <nanovdb/tools/CreateNanoGrid.h>

#include <vector>

template<typename BuildT>
void mainPartitionGrid(
    nanovdb::NanoGrid<BuildT> *deviceGrid,
    nanovdb::NanoGrid<BuildT> *indexGrid,
    const std::vector<nanovdb::NanoGrid<BuildT>*> &segments
);

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using GridT = openvdb::FloatGrid;
    using BuildT = nanovdb::ValueOnIndex;

    openvdb::util::CpuTimer cpuTimer;

    try {

        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input.vdb [input.vdb ...]\n");

        // Gather the active voxels of all inputs into one grid, side by side along x with one empty voxel in between,
        // so that inputs never touch
        cpuTimer.start("Read input VDB files");
        openvdb::initialize();
        auto grid = GridT::create();
        auto acc = grid->getAccessor();
        int x = 0;
        for (int i = 1; i < argc; ++i) {
            openvdb::io::File inFile(argv[i]);
            inFile.open();
            auto input = openvdb::gridPtrCast<GridT>(inFile.getGrids()->at(0));
            inFile.close();
            if (!input) OPENVDB_THROW(openvdb::ValueError, std::string(argv[i]) + ": first grid is not a FloatGrid\n");
            const auto bbox = input->evalActiveVoxelBoundingBox();
            const openvdb::Coord offset(x - bbox.min().x(), 0, 0);
            for (auto it = input->cbeginValueOn(); it; ++it) acc.setValueOn(it.getCoord() + offset);
            x += bbox.dim().x() + 1;
        }
        cpuTimer.stop();

        cpuTimer.start("Converting openVDB input to indexGrid");
        auto handle = nanovdb::tools::openToIndexVDB<BuildT>(grid, 0u, false, false);
        auto *indexGrid = handle.grid<BuildT>();
        cpuTimer.stop();

        // Reference: the connected components of the active voxels, one grid each, by OpenVDB
        cpuTimer.start("Segmenting openVDB input (on CPU)");
        std::vector<GridT::Ptr> segments;
        openvdb::tools::segmentActiveVoxels(*grid, segments);
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> segmentHandles;
        for (auto &segment : segments) segmentHandles.push_back(nanovdb::tools::openToIndexVDB<BuildT>(segment, 0u, false, false));
        std::vector<nanovdb::NanoGrid<BuildT>*> segmentGrids;
        for (auto &segmentHandle : segmentHandles) segmentGrids.push_back(segmentHandle.grid<BuildT>());
        cpuTimer.stop();

        // Deep-copy the indexGrid to the device; the returned handle validates it there
        auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(handle);
        auto *deviceGrid = deviceHandle.deviceGrid<BuildT>();
        if (!deviceGrid) OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading indexGrid to GPU");

        mainPartitionGrid(deviceGrid, indexGrid, segmentGrids);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
