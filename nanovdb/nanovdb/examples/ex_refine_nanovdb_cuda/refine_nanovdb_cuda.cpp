// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifdef NANOVDB_USE_TBB
#include <tbb/tbb.h>
#endif

// the following files are from OpenVDB
#include <openvdb/tools/Composite.h>
#include <openvdb/util/CpuTimer.h>

// the following files are from NanoVDB
#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/tools/CreateNanoGrid.h>

template<typename BuildT>
void mainRefineGrid(
    nanovdb::NanoGrid<BuildT> *deviceGridOriginal,
    nanovdb::NanoGrid<BuildT> *deviceGridRefined,
    nanovdb::NanoGrid<BuildT> *indexGridOriginal,
    nanovdb::NanoGrid<BuildT> *indexGridRefined,
    uint32_t benchmark_iters
);

#ifdef NANOVDB_USE_TBB
template<typename GridT>
class VoxelRefiner
{
    using TreeT = typename GridT::TreeType;
    using LeafMgrT = openvdb::tree::LeafManager<const TreeT>;

public:
    VoxelRefiner(typename GridT::Ptr srcGrid, tbb::combinable<typename GridT::Ptr>& destGrids) : mSrcGrid(srcGrid), mDestGrids(destGrids) {}
    VoxelRefiner(const VoxelRefiner& other, tbb::split) : mSrcGrid(other.mSrcGrid), mDestGrids(other.mDestGrids) {}

    void operator()(const typename LeafMgrT::LeafRange& range) const
    {
        // Refine into a thread-local copy, for safety
        typename GridT::Ptr& dstGrid = mDestGrids.local();
        if (!dstGrid) { // Create if it doesn't exist
            dstGrid = GridT::create(mSrcGrid->background());
            dstGrid->setTransform(mSrcGrid->transform().copy());
        }

        auto unitBox = openvdb::CoordBBox::createCube(openvdb::Coord(), 2);
        auto dstAcc = dstGrid->getAccessor();
        for (auto srcLeafIter = range.begin(); srcLeafIter; ++srcLeafIter) {
            for (auto srcIter = srcLeafIter->cbeginValueOn(); srcIter; ++srcIter) {
                const auto& ijk = srcIter.getCoord();
                const auto& value = srcIter.getValue();
                for (auto boxIter = unitBox.begin(); boxIter != unitBox.end(); ++boxIter)
                    dstAcc.setValue(ijk+ijk+*boxIter, value);
            }
        }
    }

private:
    typename GridT::Ptr mSrcGrid;
    tbb::combinable<typename GridT::Ptr>& mDestGrids;
};
#endif

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

        // Convert to indexGrid (original, un-refined)
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

        // Refinement (CPU/OpenVDB version)
        cpuTimer.start("Refining OpenVDB (on CPU)");
        using TreeT = GridT::TreeType;
        using LeafManagerT = openvdb::tree::LeafManager<const TreeT>;
#ifdef NANOVDB_USE_TBB
        // Phase 1: Multithreaded refinement
        tbb::combinable<GridT::Ptr> localGrids;
        LeafManagerT leafMgr(grid->tree());
        VoxelRefiner<GridT> op(grid, localGrids);
        tbb::parallel_for( leafMgr.leafRange(), op);
        // Phase 2: Recombine partial grids into final refined result
        auto refinedGrid = openvdb::FloatGrid::create(grid->background());
        refinedGrid->setTransform(grid->transform().copy());
        refinedGrid->setName(grid->getName());
        localGrids.combine_each(
            [&refinedGrid](const openvdb::FloatGrid::Ptr& localGrid) { if (localGrid) openvdb::tools::compSum(*refinedGrid, *localGrid); } );
#else
        // Single-threaded refinement
        LeafManagerT leafMgr(grid->tree());
        auto refinedGrid = openvdb::FloatGrid::create(grid->background());
        refinedGrid->setTransform(grid->transform().copy());
        refinedGrid->setName(grid->getName());
        auto dstAcc = refinedGrid->getAccessor();
        auto unitBox = openvdb::CoordBBox::createCube(openvdb::Coord(), 2);
        for (std::size_t leafID = 0; leafID < leafMgr.leafCount(); ++leafID) {
            const auto& srcLeaf = leafMgr.leaf(leafID);
            for (auto iter = srcLeaf.cbeginValueOn(); iter; ++iter) {
                const auto& ijk = iter.getCoord();
                const auto& value = iter.getValue();
                for (auto boxIter = unitBox.begin(); boxIter != unitBox.end(); ++boxIter)
                    dstAcc.setValue(ijk+ijk+*boxIter, value);
            }
        }
#endif
        cpuTimer.stop();

        // Convert to indexGrid (refined)
        cpuTimer.start("Converting openVDB input to indexGrid (refineed version)");
        auto handleRefined = nanovdb::tools::openToIndexVDB<BuildT, nanovdb::cuda::DeviceBuffer>(
            refinedGrid,
            0u,    // Don't copy data channel
            false, // No stats
            false, // No tiles
            1      // Verbose mode
        );
        cpuTimer.stop();

        auto *indexGridRefined = handleRefined.grid<BuildT>();

        if (printGridDiagnostics) {
            std::cout << "============ Refined Grid ============" << std::endl;
            std::cout << "Allocated values [valueCount()]       : " << indexGridRefined->valueCount() << std::endl;
            std::cout << "Active voxels    [activeVoxelCount()] : " << indexGridRefined->activeVoxelCount() << std::endl;
            auto minCorner = indexGridRefined->indexBBox().min(), maxCorner = indexGridRefined->indexBBox().max();
            std::cout << "Index-space bounding box              : [" << minCorner.x() << "," << minCorner.y() << "," << minCorner.z()
                      << "] -> [" << maxCorner.x() << "," << maxCorner.y() << "," << maxCorner.z() << "]" << std::endl;
            std::cout << "Leaf nodes                            : " << indexGridRefined->tree().nodeCount(0) << std::endl;
            std::cout << "Lower internal nodes                  : " << indexGridRefined->tree().nodeCount(1) << std::endl;
            std::cout << "Upper internal nodes                  : " << indexGridRefined->tree().nodeCount(2) << std::endl;
            std::cout << "Leaf-level occupancy                  : "
                      << 100.f * (float)(indexGridRefined->activeVoxelCount())/(float)(indexGridRefined->tree().nodeCount(0) * 512)
                      << "%" << std::endl;
            std::cout << "Memory usage                          : " << indexGridRefined->gridSize() << " bytes" << std::endl;
        }

        // Copy both NanoVDB grids to GPU
        handleOriginal.deviceUpload();
        handleRefined.deviceUpload();
        auto* deviceGridOriginal = handleOriginal.deviceGrid<BuildT>();
        auto* deviceGridRefined = handleRefined.deviceGrid<BuildT>();
        if (!deviceGridOriginal || !deviceGridRefined)
            OPENVDB_THROW(openvdb::RuntimeError, "Failure while uploading indexGrids to GPU");

        // Launch benchmark
        mainRefineGrid( deviceGridOriginal, deviceGridRefined, indexGridOriginal, indexGridRefined, benchmark_iters );

    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}
