// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file propagate_sign_nanovdb_cuda.cpp
///
/// @brief Checks nanovdb::tools::cuda::PropagateSign against an OpenVDB level set: the level set is stripped to its
///        narrow band, and the signs it stores outside the band are compared with the invert masks.

// the following files are from OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/HandleStorage.h> // host-includable: cuda::copyTo transfers grids without any kernel
#include <nanovdb/tools/CreateNanoGrid.h>

#include <algorithm> // std::binary_search
#include <iostream>
#include <vector>

template<typename BuildT>
void mainPropagateSign(
    const nanovdb::NanoGrid<BuildT> *deviceGrid,
    const float *deviceSdf,
    std::vector<nanovdb::Mask<3>> &leafMasks,
    std::vector<nanovdb::Mask<4>> &lowerMasks,
    std::vector<nanovdb::Mask<5>> &upperMasks,
    std::vector<nanovdb::Coord> &rootOrigins
);

/// @brief The narrow band of a level set: only its active leaf voxels are copied, so every value tile and every
///        inactive voxel value is dropped, and a leaf is kept only if it has an active voxel.
static openvdb::FloatGrid::Ptr stripToNarrowBand(const openvdb::FloatGrid& grid)
{
    auto band = openvdb::FloatGrid::create(grid.background());
    for (auto srcLeaf = grid.tree().cbeginLeaf(); srcLeaf; ++srcLeaf) {
        if (srcLeaf->isEmpty()) continue; // no active voxel
        auto *dstLeaf = band->tree().touchLeaf(srcLeaf->origin());
        for (auto it = srcLeaf->cbeginValueOn(); it; ++it) dstLeaf->setValueOn(it.pos(), *it);
    }
    return band;
}

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using BuildT = nanovdb::ValueOnIndex;

    openvdb::util::CpuTimer cpuTimer;

    try {

        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" level_set.vdb\n");

        // Read the level set from file

        cpuTimer.start("Read input VDB file");
        openvdb::initialize();
        openvdb::io::File inFile(argv[1]);
        inFile.open();
        auto baseGrids = inFile.getGrids();
        inFile.close();
        auto grid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrids->at(0));
        if (!grid) OPENVDB_THROW(openvdb::ValueError, "First grid is not a FloatGrid\n");
        cpuTimer.stop();

        cpuTimer.start("Stripping the level set to its narrow band");
        auto band = stripToNarrowBand(*grid);
        cpuTimer.stop();

        // Convert the band to an index grid whose blind data channel 0 is the SDF sidecar
        cpuTimer.start("Converting the narrow band to a ValueOnIndex grid with an SDF sidecar");
        auto handle = nanovdb::tools::openToIndexVDB<BuildT>(
            band,
            1u,    // One sidecar channel with the SDF values
            false, // No stats
            false  // No tiles
        );
        auto *indexGrid = handle.grid<BuildT>();
        const float *hostSdf = indexGrid->getBlindData<float>(0);
        if (!hostSdf) OPENVDB_THROW(openvdb::RuntimeError, "The index grid has no float sidecar\n");
        cpuTimer.stop();

        // Deep-copy the grid to the device; the sidecar lies at the same offset from the grid there as on the host
        auto deviceHandle = nanovdb::cuda::copyTo<nanovdb::cuda::Buffer<std::byte>>(handle);
        auto *deviceGrid = deviceHandle.deviceGrid<BuildT>();
        if (!deviceGrid) OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading the index grid to GPU");
        const float *deviceSdf = nanovdb::util::PtrAdd<float>(deviceGrid, nanovdb::util::PtrDiff(hostSdf, indexGrid));

        const auto &tree = indexGrid->tree();
        std::vector<nanovdb::Mask<3>> leafMasks(tree.nodeCount(0));
        std::vector<nanovdb::Mask<4>> lowerMasks(tree.nodeCount(1));
        std::vector<nanovdb::Mask<5>> upperMasks(tree.nodeCount(2));
        std::vector<nanovdb::Coord> rootOrigins;
        mainPropagateSign(deviceGrid, deviceSdf, leafMasks, lowerMasks, upperMasks, rootOrigins);

        // The invert bit of the element of the band that contains c: descend from the root to the first element
        // without a child (for a region without a root-table entry, the bit is whether its origin was returned)
        auto isInverted = [&](const openvdb::Coord &c) {
            using UpperT = nanovdb::NanoUpper<BuildT>;
            using LowerT = nanovdb::NanoLower<BuildT>;
            using LeafT = nanovdb::NanoLeaf<BuildT>;
            const nanovdb::Coord ijk(c.x(), c.y(), c.z());
            const auto *tile = tree.root().probeTile(ijk);
            if (!tile) return std::binary_search(rootOrigins.begin(), rootOrigins.end(), ijk & ~int(UpperT::DIM - 1));
            const auto *upper = tree.root().getChild(tile);
            const uint32_t u = UpperT::CoordToOffset(ijk);
            if (!upper->isChild(u)) return upperMasks[upper - tree.getFirstUpper()].isOn(u);
            const auto *lower = upper->getChild(u);
            const uint32_t l = LowerT::CoordToOffset(ijk);
            if (!lower->isChild(l)) return lowerMasks[lower - tree.getFirstLower()].isOn(l);
            return leafMasks[lower->getChild(l) - tree.getFirstLeaf()].isOn(LeafT::CoordToOffset(ijk));
        };

        // Every inactive value of the level set (voxels and tiles at every level) is negative exactly where the
        // invert bit is on. The returned root tiles are checked the other way around: they may lie where the level
        // set has no root entry, which its iterator does not visit.
        cpuTimer.start("Checking the invert masks against the level set");
        uint64_t mismatches = 0;
        for (auto it = grid->tree().cbeginValueOff(); it; ++it)
            if ((*it < 0) != isInverted(it.getCoord())) ++mismatches;
        for (const auto &origin : rootOrigins)
            if (!(grid->tree().getValue(openvdb::Coord(origin[0], origin[1], origin[2])) < 0)) ++mismatches;
        cpuTimer.stop();

        if (mismatches == 0)
            std::cout << "Result of PropagateSign check out CORRECT against OpenVDB" << std::endl;
        else
            std::cout << "Result of PropagateSign compares INCORRECT against OpenVDB (" << mismatches << " mismatches)" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
