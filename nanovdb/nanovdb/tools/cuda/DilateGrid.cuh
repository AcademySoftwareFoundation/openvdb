// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/DilateGrid.cuh

    \authors Efty Sifakis

    \brief Morphological dilation of NanoVDB indexGrids on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_DILATEGRID_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_DILATEGRID_CUH_HAS_BEEN_INCLUDED

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
class DilateGrid
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "DilateGrid allocates stream-ordered scratch and requires an AsyncResource");

    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

public:

    /// @brief Constructor
    /// @param deviceGrid source device grid to be dilated
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    DilateGrid(const GridT* d_srcGrid, cudaStream_t stream = 0,
               ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mBuilder(stream, resource), mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Set type of dilation operation
    /// @param op: NN_FACE=face neighbors, NN_FACE_EDGE=face and edge neibhros, NN_FACE_EDGE_VERTEX=26-connected neighbors
    void setOperation(morphology::NearestNeighbors op) { mOp = op; }

    /// @brief Creates a handle to the dilated grid
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
    void dilateRoot();

    void dilateInternalNodes();

    void processGridTreeRoot();

    void dilateLeafNodes();

    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT, ResourceT> mBuilder;
    cudaStream_t                 mStream{0};
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                  *mDeviceSrcGrid;
    morphology::NearestNeighbors mOp{morphology::NN_FACE_EDGE_VERTEX};
    TreeData                     mSrcTreeData;
};// tools::cuda::DilateGrid<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename BufferT>
GridHandle<BufferT>
DilateGrid<BuildT, ResourceT>::getHandle(const BufferT &pool)
{
    // Copy TreeData from GPU -> CPU
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid, mStream);

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively dilate root node
    if (mVerbose==1) mTimer.start("\nDilating root node");
    dilateRoot();

    // Allocate memory for dilated upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);

    // Dilate masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Dilate internal nodes");
    dilateInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count dilated tree nodes");
    mBuilder.countNodes(mStream);

    // Allocate new device grid buffer for dilated result
    if (mVerbose==1) mTimer.restart("Allocating dilated grid buffer");
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process GridData/TreeData/RootData of dilated result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);

    // Process lower nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);

    // Dilate leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Dilating leaf nodes");
    dilateLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox(mStream);

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// DilateGrid<BuildT, ResourceT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void DilateGrid<BuildT, ResourceT>::dilateRoot()
{
    // This method conservatively and speculatively dilates the root tiles, to accommodate
    // any new root nodes that might be introduced by the dilation operation.
    // The index-space bounding box of each upper node is expanded by one voxel, and every
    // root tile it overlaps (at most the 26-connected neighbors) is preemptively introduced
    // into the root topology.
    // (As of the present implementation this presumes a maximum of 1-voxel radius in dilation)
    // Root tiles that were preemptively introduced, but end up having no active contents will
    // be pruned in later stages of processing.

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
            const auto dilatedBBox = srcUpper->bbox().expandBy(1); // TODO: update/specialize if larger dilation neighborhoods are used
            topology::detail::insertProcessedTiles<RootT>(tiles, dilatedBBox);
        }
    }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    auto rootPtr = mBuilder.allocateProcessedRootTiles(static_cast<uint32_t>(tiles.size()));
    topology::detail::packProcessedRoot(tiles, rootPtr);
    mBuilder.uploadProcessedRoot(mStream);
}// DilateGrid<BuildT, ResourceT>::dilateRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void DilateGrid<BuildT, ResourceT>::dilateInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the dilation operation
    // Masks of lower internal nodes are densified in the sense that a serialized array of them is allocated,
    // as if every upper node had a full set of 32^3 lower children
    if (mSrcTreeData.mNodeCount[1]) { // Unless it's an empty grid
        if (mOp == morphology::NN_FACE) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
        else if (mOp == morphology::NN_FACE_EDGE) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE_EDGE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX) {
            using Op = util::morphology::cuda::DilateInternalNodesFunctor<BuildT, morphology::NN_FACE_EDGE_VERTEX>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mSrcTreeData.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks()); }
    }
}// DilateGrid<BuildT, ResourceT>::dilateInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT, typename ResourceT>
void DilateGrid<BuildT, ResourceT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), mBuilder.deviceData());
    cudaCheckError();
}// DilateGrid<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void DilateGrid<BuildT, ResourceT>::dilateLeafNodes()
{
    // Dilates the active masks of the source grid (as indicated at the leaf level), into a new grid that
    // has been already topologically dilated to include all necessary leaf nodes.
    if (mBuilder.data()->nodeCount[1]) { // Unless output grid is empty
        if (mOp == morphology::NN_FACE) {
            using Op = util::morphology::cuda::DilateLeafNodesFunctor<BuildT, morphology::NN_FACE>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr)); }
        else if (mOp == morphology::NN_FACE_EDGE)
            throw std::runtime_error("dilateLeafNodes() not implemented for NN_FACE_EDGE stencil");
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX) {
            using Op = util::morphology::cuda::DilateLeafNodesFunctor<BuildT, morphology::NN_FACE_EDGE_VERTEX>;
            util::cuda::operatorKernel<Op>
                <<<dim3(mBuilder.data()->nodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
                (mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr)); }
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
}// DilateGrid<BuildT, ResourceT>::dilateLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_DILATEGRID_CUH_HAS_BEEN_INCLUDED
