// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PartitionGrid.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Partitions a NanoVDB index grid in two by a leaf-indexed retain mask

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <array>     // std::array
#include <cstddef>   // std::byte
#include <map>       // std::map
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move

namespace nanovdb {

namespace tools::cuda {

/// @brief Splits the active voxels of an index grid in two, by a retain mask with one Mask<3> per source leaf.
/// @details Output 0 holds the active voxels whose retain bit is on, and is identical to what PruneGrid returns
///          for the same mask; output 1 holds the other active voxels, i.e. PruneGrid with the complemented
///          mask. Together they hold every active voxel exactly once. Retain bits of inactive voxels are ignored
///          on both sides. An output that keeps no voxel is a valid empty grid.
///
///          Both outputs are built in one pass: the root table is computed once, and each kernel that reads the
///          retain mask writes both outputs.
/// @note Scratch memory is about twice that of PruneGrid, since each output has its own TopologyBuilder, whose
///       dense lower-node masks take Mask<5>::SIZE Mask<4>s (16 MB) per root tile of the source grid.
/// @tparam BuildT NanoVDB index-grid build type.
/// @tparam ResourceT Stream-ordered resource used for all scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PartitionGrid
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PartitionGrid allocates stream-ordered scratch and requires an AsyncResource");

    using GridT    = NanoGrid<BuildT>;
    using RootT    = NanoRoot<BuildT>;
    using BuilderT = TopologyBuilder<BuildT, ResourceT>;

public:
    /// @param d_srcGrid Source device grid to be partitioned.
    /// @param d_srcLeafMask Retain mask: one Mask<3> per source leaf, in leaf order.
    /// @param stream CUDA stream all work is ordered on.
    /// @param resource Resource all device scratch is allocated from; must outlive this operator.
    PartitionGrid(const GridT* d_srcGrid, const Mask<3>* d_srcLeafMask, cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mBuilders{BuilderT(stream, resource), BuilderT(stream, resource)}
        , mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid), mDeviceSrcLeafMask(d_srcLeafMask) {}

    /// @brief Toggle timing output.
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the checksum mode of both outputs, disabled by default.
    void setChecksum(CheckMode mode = CheckMode::Disable)
    {
        for (auto& builder : mBuilders) builder.mChecksum = mode;
    }

    /// @brief The two outputs: [0] the voxels whose retain bit is on, [1] the rest.
    /// @tparam BufferT Buffer type of the returned grid handles.
    /// @param pool Prototype buffer the outputs take their resource from.
    /// @note The stream is synchronized before this method returns.
    template<typename BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>>
    std::array<GridHandle<BufferT>, 2> getHandles(const BufferT& pool = BufferT());

private:
    void partitionRoot();

    void partitionInternalNodes();

    void processGridTreeRoot(BuilderT& builder);

    void partitionLeafNodes();

    static constexpr unsigned int threadsPerBlock = 128;

    BuilderT          mBuilders[2];
    cudaStream_t      mStream{0};
    util::cuda::Timer mTimer;
    int               mVerbose{0};
    const GridT*      mDeviceSrcGrid;
    const Mask<3>*    mDeviceSrcLeafMask;
    TreeData          mSrcTreeData;
};// tools::cuda::PartitionGrid<BuildT, ResourceT>

//----------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGrid<BuildT, ResourceT>::partitionRoot()
{
    // Like PruneGrid, both outputs start from every source root tile, and the tiles that keep no voxel are
    // dropped once the output nodes are counted. The table is built once on the host and given to each builder.
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Same encoding as PruneGrid and PointsToGrid; it differs from Tile::key
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
               (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
               (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);  // x is the upper 21 bits
    };

    std::map<uint64_t, typename RootT::DataType::Tile> tiles;
    if (mSrcTreeData.mVoxelCount) { // Unless the source grid is empty
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        const uint64_t srcRootSize = mSrcTreeData.mNodeOffset[2] - mSrcTreeData.mNodeOffset[3];
        auto srcRootBuffer = nanovdb::HostBuffer::create(srcRootSize);
        cudaCheck(cudaMemcpyAsync(srcRootBuffer.data(), deviceSrcRoot, srcRootSize, cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        auto srcRoot = static_cast<const RootT*>(srcRootBuffer.data());
        for (uint32_t t = 0; t < srcRoot->tileCount(); t++) {
            const auto tile = srcRoot->tile(t);
            tiles.emplace(coordToKey(tile->origin()), *tile);
        }
    }

    for (auto& builder : mBuilders) {
        auto root = builder.allocateProcessedRoot(RootT::memUsage(tiles.size()));
        root->mTableSize = tiles.size();
        uint32_t t = 0;
        for (const auto& [key, tile] : tiles) *root->tile(t++) = tile;
        builder.uploadProcessedRoot(mStream);
    }
}// PartitionGrid<BuildT, ResourceT>::partitionRoot

//----------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGrid<BuildT, ResourceT>::partitionInternalNodes()
{
    // Marks the upper and lower masks of both outputs in one pass over the source leaves. The masks are dense:
    // each processed root tile has room for a full set of lower nodes.
    if (const uint64_t srcLeafCount = mSrcTreeData.mNodeCount[0]) { // Unless the source grid is empty
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(srcLeafCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            srcLeafCount, util::morphology::cuda::PartitionInternalNodesFunctor<BuildT>(),
            mDeviceSrcGrid, mBuilders[0].deviceProcessedRoot(), mDeviceSrcLeafMask,
            mBuilders[0].deviceUpperMasks(), mBuilders[0].deviceLowerMasks(),
            mBuilders[1].deviceUpperMasks(), mBuilders[1].deviceLowerMasks());
        cudaCheckError();
    }
}// PartitionGrid<BuildT, ResourceT>::partitionInternalNodes

//----------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGrid<BuildT, ResourceT>::processGridTreeRoot(BuilderT& builder)
{
    // Copies the source GridData (name, map, ...), then resets what the builder owns
    cudaCheck(cudaMemcpyAsync(&builder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), builder.deviceData());
    cudaCheckError();
}// PartitionGrid<BuildT, ResourceT>::processGridTreeRoot

//----------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGrid<BuildT, ResourceT>::partitionLeafNodes()
{
    // Writes the leaf value masks of both outputs in one pass over the source leaves, then rebuilds the voxel
    // offsets of each output
    if (const uint64_t srcLeafCount = mSrcTreeData.mNodeCount[0]) { // Unless the source grid is empty
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(srcLeafCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            srcLeafCount, util::morphology::cuda::PartitionLeafMasksFunctor<BuildT>(), mDeviceSrcGrid,
            &mBuilders[0].data()->getGrid(), &mBuilders[1].data()->getGrid(), mDeviceSrcLeafMask);
        cudaCheckError();
    }
    for (auto& builder : mBuilders) builder.processLeafOffsets(mStream);
}// PartitionGrid<BuildT, ResourceT>::partitionLeafNodes

//----------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename BufferT>
std::array<GridHandle<BufferT>, 2> PartitionGrid<BuildT, ResourceT>::getHandles(const BufferT& pool)
{
    // The source TreeData is read with a blocking copy on the default stream, so wait for mStream first
    cudaCheck(cudaStreamSynchronize(mStream));
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid);
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("nanovdb::tools::cuda::PartitionGrid: grids with active tiles are not supported");

    if (mVerbose==1) mTimer.start("\nPartition root node");
    partitionRoot();

    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    for (auto& builder : mBuilders) builder.allocateInternalMaskBuffers(mStream);

    if (mVerbose==1) mTimer.restart("Partition internal nodes");
    partitionInternalNodes();

    if (mVerbose==1) mTimer.restart("Count partitioned tree nodes");
    for (auto& builder : mBuilders) builder.countNodes(mStream);
    cudaCheck(cudaStreamSynchronize(mStream)); // the node counts are read on the host

    if (mVerbose==1) mTimer.restart("Allocating partitioned grid buffers");
    auto buffer0 = mBuilders[0].getBuffer(pool, mStream);
    auto buffer1 = mBuilders[1].getBuffer(pool, mStream);

    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    for (auto& builder : mBuilders) processGridTreeRoot(builder);

    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    for (auto& builder : mBuilders) builder.processUpperNodes(mStream);

    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    for (auto& builder : mBuilders) builder.processLowerNodes(mStream);

    if (mVerbose==1) mTimer.restart("Partitioning leaf nodes");
    partitionLeafNodes();

    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    for (auto& builder : mBuilders) builder.processBBox(mStream);

    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    for (auto& builder : mBuilders) builder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaStreamSynchronize(mStream));
    return {GridHandle<BufferT>(std::move(buffer0)), GridHandle<BufferT>(std::move(buffer1))};
}// PartitionGrid<BuildT, ResourceT>::getHandles

//----------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED
