// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MergeGrids.cuh

    \authors Efty Sifakis

    \brief Morphological union of NanoVDB indexGrids on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED

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
class MergeGrids
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

public:

    /// @brief Constructor
    /// @param d_srcGrid1 first source device grid to be merged
    /// @param d_srcGrid2 second source device grid to be merged
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    MergeGrids(const GridT* d_srcGrid1, const GridT* d_srcGrid2, cudaStream_t stream = 0)
        : mBuilder(stream), mStream(stream), mTimer(stream), mDeviceSrcGrid1(d_srcGrid1), mDeviceSrcGrid2(d_srcGrid2) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the merged grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

private:
    void mergeRoot();

    void mergeInternalNodes();

    void processGridTreeRoot();

    void mergeLeafNodes();

    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT> mBuilder;
    cudaStream_t            mStream{0};
    util::cuda::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid1;
    const GridT             *mDeviceSrcGrid2;
    TreeData                mSrcTreeData1;
    TreeData                mSrcTreeData2;
};// tools::cuda::MergeGrids<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
MergeGrids<BuildT>::getHandle(const BufferT &pool)
{
    // Copy TreeData from GPU -> CPU
    cudaStreamSynchronize(mStream);
    mSrcTreeData1 = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid1);
    mSrcTreeData2 = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid2);

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData1.mTileCount[2] || mSrcTreeData1.mTileCount[1] || mSrcTreeData1.mTileCount[0] ||
        mSrcTreeData2.mTileCount[2] || mSrcTreeData2.mTileCount[1] || mSrcTreeData2.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Merge root nodes
    if (mVerbose==1) mTimer.start("\nMerging root nodes");
    mergeRoot();

    // Allocate memory for merged upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);

    // Merge masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Merge internal nodes");
    mergeInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count merged tree nodes");
    mBuilder.countNodes(mStream);

    cudaStreamSynchronize(mStream);

    // Allocate new device grid buffer for merged result
    if (mVerbose==1) mTimer.restart("Allocating merged grid buffer");
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process GridData/TreeData/RootData of merged result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);

    // Process lower nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);

    // Merge leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Merging leaf nodes");
    mergeLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox(mStream);

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// MergeGrids<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeRoot()
{
    // Creates a new merged tree root with the merged tiles of the two input root topologies

    int device = 0;
    cudaGetDevice(&device);

    std::map<uint64_t, typename RootT::DataType::Tile> mergedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    // Make a host copy of the source root topology RootNode for both inputs
    // Then, merge tiles of two sources in a sorted container

    if (mSrcTreeData1.mVoxelCount) { // If the first input is not a null grid
        // Make a host copy of the Root topology
        auto deviceSrcRoot1 = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid1, GridT::memUsage() + mSrcTreeData1.mNodeOffset[3]));
        uint64_t rootSize1 = mSrcTreeData1.mNodeOffset[2] - mSrcTreeData1.mNodeOffset[3];
        auto srcRootBuffer1 = nanovdb::HostBuffer::create(rootSize1);
        cudaCheck(cudaMemcpyAsync(srcRootBuffer1.data(), deviceSrcRoot1, rootSize1, cudaMemcpyDeviceToHost, mStream));
        auto srcRoot1 = static_cast<RootT*>(srcRootBuffer1.data());

        // Add all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot1->tileCount(); t++) {
            auto tile = srcRoot1->tile(t);
            auto sortKey = coordToKey(tile->origin());
            mergedTiles.emplace(sortKey, *tile);
        }
    }

    if (mSrcTreeData2.mVoxelCount) { // If the first input is not a null grid
        // Make a host copy of the Root topology
        auto deviceSrcRoot2 = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid2, GridT::memUsage() + mSrcTreeData2.mNodeOffset[3]));
        uint64_t rootSize2 = mSrcTreeData2.mNodeOffset[2] - mSrcTreeData2.mNodeOffset[3];
        auto srcRootBuffer2 = nanovdb::HostBuffer::create(rootSize2);
        cudaCheck(cudaMemcpyAsync(srcRootBuffer2.data(), deviceSrcRoot2, rootSize2, cudaMemcpyDeviceToHost, mStream));
        auto srcRoot2 = static_cast<RootT*>(srcRootBuffer2.data());

        // Add all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot2->tileCount(); t++) {
            auto tile = srcRoot2->tile(t);
            auto sortKey = coordToKey(tile->origin());
            mergedTiles.emplace(sortKey, *tile);
        }
    }

    // Package the new root topology into a RootNode plus Tile list; upload to the GPU
    uint64_t rootSize = RootT::memUsage(mergedTiles.size());
    mBuilder.mProcessedRoot = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto mergedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    mergedRootPtr->mTableSize = mergedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : mergedTiles)
        *mergedRootPtr->tile(t++) = tile;
    mBuilder.mProcessedRoot.deviceUpload(device, mStream, false);
}// MergeGrids<BuildT>::mergeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeInternalNodes()
{
    // Merges the masks of upper and lower nodes from both input topologies into the
    // densified, pre-allocated mask arrays of the merged result
    using Op = util::morphology::cuda::MergeInternalNodesFunctor<BuildT>;
    if (mSrcTreeData1.mNodeCount[1]) { // Unless the first grid to merge is empty
        util::cuda::operatorKernel<Op>
            <<<mSrcTreeData1.mNodeCount[1], Op::MaxThreadsPerBlock, 0, mStream>>>
            (mDeviceSrcGrid1, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks());
    }
    if (mSrcTreeData2.mNodeCount[1]) { // Unless the second grid to merge is empty
        util::cuda::operatorKernel<Op>
            <<<mSrcTreeData2.mNodeCount[1], Op::MaxThreadsPerBlock, 0, mStream>>>
            (mDeviceSrcGrid2, mBuilder.deviceProcessedRoot(), mBuilder.deviceUpperMasks(), mBuilder.deviceLowerMasks());
    }
}// MergeGrids<BuildT>::mergeInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MergeGrids<BuildT>::processGridTreeRoot()
{
    // Copy GridData from first source grid
    // TODO: Check for instances where extra processing is needed
    // TODO: check that the second grid input has consistent GridData, too
    cudaCheck(cudaMemcpyAsync(&mBuilder.data()->getGrid(), mDeviceSrcGrid1->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), mBuilder.deviceData());
    cudaCheckError();
}// MergeGrids<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeLeafNodes()
{
    using Op = util::morphology::cuda::MergeLeafNodesFunctor<BuildT>;
    if (mSrcTreeData1.mNodeCount[1]) { // Unless first input grid is empty
        util::cuda::operatorKernel<Op>
            <<<dim3(mSrcTreeData1.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
            (mDeviceSrcGrid1, static_cast<GridT*>(mBuilder.data()->d_bufferPtr));
    }
    if (mSrcTreeData2.mNodeCount[1]) { // Unless second input grid is empty
        util::cuda::operatorKernel<Op>
            <<<dim3(mSrcTreeData2.mNodeCount[1],Op::SlicesPerLowerNode,1), Op::MaxThreadsPerBlock, 0, mStream>>>
            (mDeviceSrcGrid2, static_cast<GridT*>(mBuilder.data()->d_bufferPtr));
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
}// MergeGrids<BuildT>::mergeLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MERGEGRIDS_CUH_HAS_BEEN_INCLUDED
