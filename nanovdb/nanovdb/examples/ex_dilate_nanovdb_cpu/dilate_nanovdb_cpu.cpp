// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Morphology.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/CreateNanoGrid.h>

template<typename BuildT>
void mainDilateGrid(
    nanovdb::NanoGrid<BuildT> *srcGrid,         // original (un-dilated) grid; also the prune reference
    nanovdb::NanoGrid<BuildT> *indexGridDilated,
    uint32_t nnType,
    uint32_t benchmark_iters
);

/// @brief This example depends on OpenVDB, NanoVDB, and CUDA
int main(int argc, char *argv[])
{
    using GridT = openvdb::FloatGrid;
    using BuildT = nanovdb::ValueOnIndex;

    // Select the type of dilation here. dilateLeafNodes supports NN_FACE and NN_FACE_EDGE_VERTEX.
    openvdb::tools::NearestNeighbors nnType = openvdb::tools::NN_FACE;

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

        // Convert to indexGrid (original, un-dilated)
        cpuTimer.start("Converting openVDB input to indexGrid (original version)");
        auto handleOriginal = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::HostBuffer>(
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

        // Dilation (CPU/OpenVDB version), used as the correctness reference
        cpuTimer.start("Dilating openVDB (on CPU)");
        openvdb::tools::dilateActiveValues(grid->tree(), 1, nnType);
        cpuTimer.stop();

        // Convert to indexGrid (dilated reference)
        cpuTimer.start("Converting openVDB input to indexGrid (dilated version)");
        auto handleDilated = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::HostBuffer>(
            grid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *indexGridDilated = handleDilated.grid<BuildT>();
        cpuTimer.stop();

        if (printGridDiagnostics) {
            std::cout << "============ Dilated Grid ============" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridDilated->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridDilated->activeVoxelCount() << std::endl;
            auto minCorner = indexGridDilated->indexBBox().min(), maxCorner = indexGridDilated->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridDilated->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridDilated->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridDilated->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridDilated->activeVoxelCount())/(float)(indexGridDilated->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridDilated->gridSize() << " bytes" << std::endl;
        }

        // All grids are host-resident (HostBuffer); the operators read them directly.
        mainDilateGrid( indexGridOriginal, indexGridDilated, nnType, benchmark_iters );

    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
