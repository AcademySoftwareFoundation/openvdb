// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/RefineGrid.cuh

    \authors Efty Sifakis

    \brief 2x Topological refinement of NanoVDB indexGrids on the device

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
#include <cstddef> // for std::byte
#include <map>
#include <stdexcept>
#include <utility>


namespace nanovdb {

namespace tools::cuda {

template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class RefineGrid
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "RefineGrid allocates stream-ordered scratch and requires an AsyncResource");

    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

public:

    /// @brief Constructor
    /// @param deviceGrid source device grid to be refined
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    RefineGrid(const GridT* d_srcGrid, cudaStream_t stream = 0,
               ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mBuilder(stream, resource), mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the refined grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer prototype buffer whose resource (or create() pool) allocates the output
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DualDeviceBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

    /// @brief Resource form of getHandle: the grid allocates through this
    ///        builder's resource, the one its scratch uses, on its stream.
    /// @return a handle over cuda::Buffer<std::byte, ResourceRef<ResourceT>>,
    ///         which borrows the resource: it must outlive the handle
    GridHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>> build()
    {
        using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;
        return this->getHandle(BufferT(mStream, mBuilder.ref(), 0, nanovdb::cuda::noInit));
    }

private:
    void refineRoot();

    void refineInternalNodes();

    void processGridTreeRoot();

    void refineLeafNodes();

    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT, ResourceT> mBuilder;
    cudaStream_t            mStream{0};
    util::cuda::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid;
    TreeData                mSrcTreeData;
};// tools::cuda::RefineGrid<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename BufferT>
GridHandle<BufferT>
RefineGrid<BuildT, ResourceT>::getHandle(const BufferT &pool)
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
}// RefineGrid<BuildT, ResourceT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void RefineGrid<BuildT, ResourceT>::refineRoot()
{
    // This method conservatively and speculatively refines the root tiles, to accommodate
    // any new root nodes that might be introduced by the upsampling operation.
    // The index-space bounding box of each upper node is refined, and every root tile it
    // overlaps is preemptively introduced into the root topology (each 2048^3 octant of the
    // source tile maps onto one refined tile). Root tiles that were preemptively introduced,
    // but end up having no active contents will be pruned in later stages of processing.

    topology::detail::ProcessedTileMap<RootT> tiles;

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Make a host copy of the source topology RootNode *and* the Upper Nodes (needed for BBox'es)
        // TODO: Consider avoiding to copy the entire set of upper nodes
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        uint64_t rootAndUpperSize = mSrcTreeData.mNodeOffset[1] - mSrcTreeData.mNodeOffset[3];
        auto srcRootAndUpperBuffer = nanovdb::HostBuffer::create(rootAndUpperSize);
        cudaCheck(cudaMemcpyAsync(srcRootAndUpperBuffer.data(), deviceSrcRoot, rootAndUpperSize, cudaMemcpyDeviceToHost, mStream));
        auto srcRootAndUpper = static_cast<RootT*>(srcRootAndUpperBuffer.data());

        for (uint32_t t = 0; t < srcRootAndUpper->tileCount(); t++) {
            auto srcUpper = srcRootAndUpper->getChild(srcRootAndUpper->tile(t));
            const auto tileBBox = srcUpper->bbox();
            const CoordBBox refinedBBox(util::morphology::refineCoord(tileBBox.min()),
                                        util::morphology::refineCoord(tileBBox.max()).offsetBy(1));
            topology::detail::insertProcessedTiles<RootT>(tiles, refinedBBox);
        }
    }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    auto rootPtr = mBuilder.allocateProcessedRoot(RootT::memUsage(tiles.size()));
    topology::detail::packProcessedRoot(tiles, rootPtr);
    mBuilder.uploadProcessedRoot(mStream);
}// RefineGrid<BuildT, ResourceT>::refineRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void RefineGrid<BuildT, ResourceT>::refineInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the refinement operation
    // Masks of lower internal nodes are densified in the sense that a serialized array of them is allocated,
    // as if every upper node had a full set of 32^3 lower children
    if (auto srcLeafCount = mSrcTreeData.mNodeCount[0]) { // Unless it's an empty grid
        util::cuda::lambdaKernel<<<numBlocks(srcLeafCount), mNumThreads, 0, mStream>>>(
            srcLeafCount, util::morphology::cuda::RefineInternalNodesFunctor<BuildT>(),
            mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks() );
    }
}// RefineGrid<BuildT, ResourceT>::refineInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT>
void RefineGrid<BuildT, ResourceT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), mBuilder.deviceData());
    cudaCheckError();
}// RefineGrid<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void RefineGrid<BuildT, ResourceT>::refineLeafNodes()
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
}// RefineGrid<BuildT, ResourceT>::refineLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_REFINEGRID_CUH_HAS_BEEN_INCLUDED
