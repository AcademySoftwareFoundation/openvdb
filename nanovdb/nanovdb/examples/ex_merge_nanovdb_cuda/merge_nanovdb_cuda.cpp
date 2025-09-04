// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

// the following files are from OpenVDB
#include <openvdb/tools/Composite.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

template<typename BuildT>
void mainMergeGrids(
    nanovdb::NanoGrid<BuildT> *deviceSrcGrid1,
    nanovdb::NanoGrid<BuildT> *deviceSrcGrid2,
    nanovdb::NanoGrid<BuildT> *deviceDstReferenceGrid,
    nanovdb::NanoGrid<BuildT> *hostSrcGrid1,
    nanovdb::NanoGrid<BuildT> *hostSrcGrid2,
    nanovdb::NanoGrid<BuildT> *hostDstReferenceGrid,
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

        if (argc < 3) OPENVDB_THROW(openvdb::ValueError, "usage: "+std::string(argv[0])+" input1.vdb input2.vdb [<iterations>]\n");
        int benchmark_iters = 10;
        if (argc > 3) sscanf(argv[3], "%d", &benchmark_iters);

        // Read the openvdb inputs from files

        cpuTimer.start("Read first input VDB file");
        openvdb::initialize();
        openvdb::io::File inFile1(argv[1]);
        inFile1.open(false); // disable delayed loading
        auto baseGrids1 = inFile1.getGrids();
        inFile1.close();
        auto grid1 = openvdb::gridPtrCast<GridT>(baseGrids1->at(0));
        if (!grid1) OPENVDB_THROW(openvdb::ValueError, "First grid is not a FloatGrid\n");
        cpuTimer.stop();

        cpuTimer.start("Read second input VDB file");
        openvdb::initialize();
        openvdb::io::File inFile2(argv[2]);
        inFile2.open(false); // disable delayed loading
        auto baseGrids2 = inFile2.getGrids();
        inFile2.close();
        auto grid2 = openvdb::gridPtrCast<GridT>(baseGrids2->at(0));
        if (!grid2) OPENVDB_THROW(openvdb::ValueError, "Second grid is not a FloatGrid\n");
        cpuTimer.stop();

        // Convert to indexGrid
        cpuTimer.start("Converting openVDB input to indexGrid (first component)");
        auto srcHandle1 = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            grid1,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *srcIndexGrid1 = srcHandle1.grid<BuildT>();
        cpuTimer.stop();

        cpuTimer.start("Converting openVDB input to indexGrid (second component)");
        auto srcHandle2 = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            grid2,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *srcIndexGrid2 = srcHandle2.grid<BuildT>();
        cpuTimer.stop();

        if (printGridDiagnostics) {
            std::cout << "============ Component Grid #1 =======  " << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << srcIndexGrid1->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << srcIndexGrid1->activeVoxelCount() << std::endl;
            auto minCorner = srcIndexGrid1->indexBBox().min(), maxCorner = srcIndexGrid1->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << srcIndexGrid1->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << srcIndexGrid1->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << srcIndexGrid1->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(srcIndexGrid1->activeVoxelCount())/(float)(srcIndexGrid1->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << srcIndexGrid1->gridSize() << " bytes" << std::endl;
            std::cout << "============ Component Grid #2 =======  " << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << srcIndexGrid2->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << srcIndexGrid2->activeVoxelCount() << std::endl;
            minCorner = srcIndexGrid2->indexBBox().min(), maxCorner = srcIndexGrid2->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << srcIndexGrid2->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << srcIndexGrid2->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << srcIndexGrid2->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(srcIndexGrid2->activeVoxelCount())/(float)(srcIndexGrid2->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << srcIndexGrid2->gridSize() << " bytes" << std::endl;
        }

        // Merge (CPU/OpenVDB version)
        cpuTimer.start("Merging openVDBs (on CPU)");
        auto mergedGrid = grid1->deepCopy();
        openvdb::tools::compSum(*mergedGrid, *grid2);
        cpuTimer.stop();

        // Convert to indexGrid
        cpuTimer.start("Converting merged openVDB output to indexGrid");
        auto dstReferenceHandle = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            mergedGrid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        auto *dstReferenceGrid = dstReferenceHandle.grid<BuildT>();
        cpuTimer.stop();

        if (printGridDiagnostics) {
            std::cout << "============ Merged Grid (CPU) =======  " << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << dstReferenceGrid->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << dstReferenceGrid->activeVoxelCount() << std::endl;
            auto minCorner = dstReferenceGrid->indexBBox().min(), maxCorner = dstReferenceGrid->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << dstReferenceGrid->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << dstReferenceGrid->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << dstReferenceGrid->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(dstReferenceGrid->activeVoxelCount())/(float)(dstReferenceGrid->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << dstReferenceGrid->gridSize() << " bytes" << std::endl;
        }

        // Copy both NanoVDB grids to GPU
        srcHandle1.deviceUpload();
        srcHandle2.deviceUpload();
        dstReferenceHandle.deviceUpload();
        auto* deviceSrcGrid1 = srcHandle1.deviceGrid<BuildT>();
        auto* deviceSrcGrid2 = srcHandle2.deviceGrid<BuildT>();
        auto* deviceDstReferenceGrid = dstReferenceHandle.deviceGrid<BuildT>();
        if (!deviceSrcGrid1 || !deviceSrcGrid2 || !deviceDstReferenceGrid)
            OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading indexGrids to GPU");

        // Launch benchmark
        mainMergeGrids(
            deviceSrcGrid1, deviceSrcGrid2, deviceDstReferenceGrid,
            srcIndexGrid1, srcIndexGrid2, dstReferenceGrid, benchmark_iters);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
