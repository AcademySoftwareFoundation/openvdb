// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/RefineGrid.cuh

    \authors Efty Sifakis

    \brief 2x Subdivision/Upsampling of NanoVDB indexGrids on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_REFINEGRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_REFINEGRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>


namespace nanovdb {

namespace tools::cuda {

template <typename BuildT>
class RefineGrid
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

public:

    /// @brief Constructor
    /// @param deviceGrid source device grid to be refined
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    RefineGrid(const GridT* d_srcGrid, cudaStream_t stream = 0)
        : mBuilder(stream), mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the refined grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

    void refineRoot();

    void refineInternalNodes();

    void processGridTreeRoot();

    void refineLeafNodes();

private:
    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT> mBuilder;
    cudaStream_t            mStream{0};
    util::cuda::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid;
    TreeData                mSrcTreeData;

public:
    const GridT* deviceSrcGrid() const { return mDeviceSrcGrid; }
};// tools::cuda::RefineGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
RefineGrid<BuildT>::getHandle(const BufferT &pool)
{
    // Copy TreeData from GPU -> CPU
    cudaStreamSynchronize(mStream);
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid);

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively refine root node
    if (mVerbose==1) mTimer.start("\nRefining root node");
    refineRoot();

    // Allocate memory for refined upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);

    // Refine masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Refining internal nodes");
    refineInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count refined tree nodes");
    mBuilder.countNodes(mStream);

    cudaStreamSynchronize(mStream);

    // Allocate new device grid buffer for refined result
    if (mVerbose==1) mTimer.restart("Allocating refined grid buffer");
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process GridData/TreeData/RootData of refined result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of refined result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);

    // Process lower nodes of refined result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);

    // Refine leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Refining leaf nodes");
    refineLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox(mStream);

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// RefineGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void RefineGrid<BuildT>::refineRoot()
{
    // This method conservatively and speculatively refines the root tiles, to accommodate
    // any new root nodes that might be introduced by the upsampling operation.
    // The index-space bounding box of each tile is examined, and if it overlaps any of the 2048^3-sized octants
    // of the tile, a corresponding new tile is preemptively introduced into the root topology.
    // Root tiles that were preemptively introduced, but end up having no active contents will
    // be pruned in later stages of processing.

    int device = 0;
    cudaGetDevice(&device);

    std::map<uint64_t, typename RootT::DataType::Tile> refinedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Make a host copy of the source topology RootNode *and* the Upper Nodes (needed for BBox'es)
        // TODO: Consider avoiding to copy the entire set of upper nodes
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        uint64_t rootAndUpperSize = mSrcTreeData.mNodeOffset[1] - mSrcTreeData.mNodeOffset[3];
        auto srcRootAndUpperBuffer = nanovdb::HostBuffer::create(rootAndUpperSize);
        cudaCheck(cudaMemcpyAsync(srcRootAndUpperBuffer.data(), deviceSrcRoot, rootAndUpperSize, cudaMemcpyDeviceToHost, mStream));
        auto srcRootAndUpper = static_cast<RootT*>(srcRootAndUpperBuffer.data());

        // For each original root tile, consider adding those tiles in its 26-connected neighborhood
        for (uint32_t t = 0; t < srcRootAndUpper->tileCount(); t++) {
            auto srcUpper = srcRootAndUpper->getChild(srcRootAndUpper->tile(t));
            const auto tileBBox = srcUpper->bbox();
            for (int di = 0; di <= 2048; di += 2048)
            for (int dj = 0; dj <= 2048; dj += 2048)
            for (int dk = 0; dk <= 2048; dk += 2048) {
                const auto octantBBox = nanovdb::CoordBBox::createCube(srcUpper->origin().offsetBy(di,dj,dk), 2048);
                if (tileBBox.hasOverlap(octantBBox)) {
                    auto refinedOrigin = octantBBox.min()+octantBBox.min();
                    auto sortKey = coordToKey(refinedOrigin); // key used in the radix sort, in accordance with PointsToGrid
                    auto tileKey = RootT::CoordToKey(refinedOrigin); // encoding used in the NanoVDB tile
                    typename RootT::Tile refinedTile{tileKey}; // Only the key value is needed; child pointer & value will be unused
                    refinedTiles.emplace(sortKey, refinedTile);
               }
            }
        }
    }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    uint64_t rootSize = RootT::memUsage(refinedTiles.size());
    mBuilder.mProcessedRoot = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto refinedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    refinedRootPtr->mTableSize = refinedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : refinedTiles)
        *refinedRootPtr->tile(t++) = tile;
    mBuilder.mProcessedRoot.deviceUpload(device, mStream, false);
}// RefineGrid<BuildT>::refineRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void RefineGrid<BuildT>::refineInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the refinement operation
    // Masks of lower internal nodes are densified in the sense that a serialized array of them is allocated,
    // as if every upper node had a full set of 32^3 lower children
    if (auto srcLeafCount = mSrcTreeData.mNodeCount[0]) { // Unless it's an empty grid
        util::cuda::lambdaKernel<<<numBlocks(srcLeafCount), mNumThreads, 0, mStream>>>(
            srcLeafCount, util::morphology::cuda::RefineInternalNodesFunctor<BuildT>(),
            mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.mUpperMasks.deviceData(), mBuilder.mLowerMasks.deviceData() );
    }
}// RefineGrid<BuildT>::refineInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void RefineGrid<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), deviceSrcGrid()->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), mBuilder.deviceData());
    cudaCheckError();
}// RefineGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void RefineGrid<BuildT>::refineLeafNodes()
{
    // Refines the active masks of the source grid (as indicated at the leaf level), into a new grid that
    // has been already topologically refined to include all necessary leaf nodes.
    auto srcLeafCount = mSrcTreeData.mNodeCount[0];
    if (srcLeafCount) { // Unless grid is empty
        util::cuda::lambdaKernel<<<numBlocks(srcLeafCount), mNumThreads, 0, mStream>>>(
            srcLeafCount, util::morphology::cuda::RefineLeafMasksFunctor<BuildT>(), mDeviceSrcGrid, &mBuilder.data()->getGrid());
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
}// RefineGrid<BuildT>::refineLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_REFINEGRID_CUH_HAS_BEEN_INCLUDED
