// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Composite.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

template<class CoordT>
inline CoordT
coarsenCoord(const CoordT& coord)
{
    auto coarsenComponent = [](const typename CoordT::ValueType n) {return (n>=0) ? (n>>1) : -((-n+1)>>1);};
    CoordT result;
    result[0] = coarsenComponent(coord[0]);
    result[1] = coarsenComponent(coord[1]);
    result[2] = coarsenComponent(coord[2]);
    return result;
}

template<typename BuildT>
void mainCoarsenGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridCoarsened,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridCoarsened,
    uint32_t benchmark_iters
);

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using GridT = openvdb::FloatGrid;
    using BuildT = nanovdb::ValueOnIndex;

    openvdb::util::CpuTimer cpuTimer;
    const bool printGridDiagnostics = true;

    try {
        if (argc<2) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input.vdb [<iterations>]\n");
        int benchmark_iters = 10;
        if (argc > 2) sscanf(argv[2], "%d", &benchmark_iters);

        // Read the initial level set from file

        cpuTimer.start("Read input VDB file");
        openvdb::initialize();
        openvdb::io::File inFile(argv[1]);
        inFile.open(false); // disable delayed loading
        auto baseGrids = inFile.getGrids();
        inFile.close();
        auto grid = openvdb::gridPtrCast<GridT>(baseGrids->at(0));
        if (!grid) OPENVDB_THROW(openvdb::ValueError, "First grid is not a FloatGrid\n");
        cpuTimer.stop();

        // Convert to indexGrid (original, un-coarsened)
        cpuTimer.start("Converting openVDB input to indexGrid (original version)");
        auto handleOriginal = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            grid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *indexGridOriginal = handleOriginal.grid<BuildT>();
        cpuTimer.stop();

        if (printGridDiagnostics) {
            std::cout << "============ Original Grid ===========" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridOriginal->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridOriginal->activeVoxelCount() << std::endl;
            auto minCorner = indexGridOriginal->indexBBox().min(), maxCorner = indexGridOriginal->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridOriginal->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridOriginal->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridOriginal->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridOriginal->activeVoxelCount())/(float)(indexGridOriginal->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridOriginal->gridSize() << " bytes" << std::endl;
        }
        // Coarsening (CPU/OpenVDB version)
        cpuTimer.start("Coarsening OpenVDB (on CPU)");
        using TreeT = GridT::TreeType;
        using LeafManagerT = openvdb::tree::LeafManager<const TreeT>;
        LeafManagerT leafMgr(grid->tree());
        auto coarsenedGrid = openvdb::FloatGrid::create(grid->background());
        coarsenedGrid->setTransform(grid->transform().copy());
        coarsenedGrid->setName(grid->getName());
        auto dstAcc = coarsenedGrid->getAccessor();
        for (std::size_t leafID = 0; leafID < leafMgr.leafCount(); ++leafID) {
            const auto& srcLeaf = leafMgr.leaf(leafID);
            for (auto iter = srcLeaf.cbeginValueOn(); iter; ++iter) {
                const auto dstCoord = coarsenCoord(iter.getCoord());
                if (!dstAcc.isValueOn(dstCoord))
                    dstAcc.setValue(dstCoord, iter.getValue());
            }
        }
        cpuTimer.stop();

        // Convert to indexGrid (coarsened)
        cpuTimer.start("Converting openVDB input to indexGrid (coarsened version)");
        auto handleCoarsened = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            coarsenedGrid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        cpuTimer.stop();

        auto *indexGridCoarsened = handleCoarsened.grid<BuildT>();

        if (printGridDiagnostics) {
            std::cout << "=========== Coarsened Grid ===========" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridCoarsened->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridCoarsened->activeVoxelCount() << std::endl;
            auto minCorner = indexGridCoarsened->indexBBox().min(), maxCorner = indexGridCoarsened->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridCoarsened->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridCoarsened->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridCoarsened->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridCoarsened->activeVoxelCount())/(float)(indexGridCoarsened->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridCoarsened->gridSize() << " bytes" << std::endl;
        }

        // Copy both NanoVDB grids to GPU
        handleOriginal.deviceUpload();
        handleCoarsened.deviceUpload();
        auto* deviceGridOriginal = handleOriginal.deviceGrid<BuildT>();
        auto* deviceGridCoarsened = handleCoarsened.deviceGrid<BuildT>();
        if (!deviceGridOriginal || !deviceGridCoarsened)
            OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading indexGrids to GPU");

        // Launch benchmark
        mainCoarsenGrid( deviceGridOriginal, deviceGridCoarsened, indexGridOriginal, indexGridCoarsened, benchmark_iters );
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
