// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PartitionGrid.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Partitions a NanoVDB index grid in two by a leaf-indexed retain mask, or into one grid per label

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

#include <cub/cub.cuh> // cub::DeviceHistogram

#include <algorithm> // std::upper_bound
#include <array>     // std::array
#include <cstddef>   // std::byte
#include <limits>    // std::numeric_limits
#include <stdexcept> // std::runtime_error
#include <utility>   // std::move
#include <vector>    // std::vector

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
/// @tparam BuildT NanoVDB build type; only ValueOnIndex is supported.
/// @tparam ResourceT Stream-ordered resource used for all scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PartitionGridBinary
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex,
                  "PartitionGridBinary partitions the active voxels of a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PartitionGridBinary allocates stream-ordered scratch and requires an AsyncResource");

    using GridT    = NanoGrid<BuildT>;
    using RootT    = NanoRoot<BuildT>;
    using BuilderT = TopologyBuilder<BuildT, ResourceT>;

public:
    /// @brief Constructor
    /// @param d_srcGrid source device grid to be partitioned
    /// @param d_srcLeafMask retain mask: one Mask<3> per source leaf, in leaf order
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    PartitionGridBinary(const GridT* d_srcGrid, const Mask<3>* d_srcLeafMask, cudaStream_t stream = 0,
                        ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mBuilders{BuilderT(stream, resource), BuilderT(stream, resource)}
        , mStream(stream), mTimer(stream), mDeviceSrcGrid(d_srcGrid), mDeviceSrcLeafMask(d_srcLeafMask) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the checksum mode of both outputs, disabled by default.
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable)
    {
        for (auto& builder : mBuilders) builder.mChecksum = mode;
    }

    /// @brief The two outputs: [0] the voxels whose retain bit is on, [1] the rest.
    /// @tparam BufferT Buffer type of the returned grid handles.
    /// @param pool Prototype buffer the outputs take their resource from.
    /// @note The stream is synchronized before this method returns.
    /// @throw std::runtime_error if the grid has active tiles.
    template<typename BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>>
    std::array<GridHandle<BufferT>, 2> getHandles(const BufferT& pool = BufferT());

    /// @brief Resource form of getHandles: the outputs allocate through this
    ///        builder's resource, the one its scratch uses, on its stream.
    /// @return handles over cuda::Buffer<std::byte, ResourceRef<ResourceT>>,
    ///         which borrow the resource: it must outlive the handles
    std::array<GridHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>>, 2> build()
    {
        using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;
        return this->getHandles(BufferT(mStream, mBuilders[0].ref(), 0, nanovdb::cuda::noInit));
    }

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
};// tools::cuda::PartitionGridBinary<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename BufferT>
std::array<GridHandle<BufferT>, 2> PartitionGridBinary<BuildT, ResourceT>::getHandles(const BufferT& pool)
{
    mSrcTreeData = util::cuda::DeviceGridTraits<BuildT>::getTreeData(mDeviceSrcGrid, mStream);
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    if (mVerbose==1) mTimer.start("\nPartition root node");
    partitionRoot();

    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    for (auto& builder : mBuilders) builder.allocateInternalMaskBuffers(mStream);

    if (mVerbose==1) mTimer.restart("Partition internal nodes");
    partitionInternalNodes();

    if (mVerbose==1) mTimer.restart("Count partitioned tree nodes");
    for (auto& builder : mBuilders) builder.countNodes(mStream);

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
}// PartitionGridBinary<BuildT, ResourceT>::getHandles

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGridBinary<BuildT, ResourceT>::partitionRoot()
{
    // Like PruneGrid, both outputs start from every source root tile, and the tiles that keep no voxel are
    // dropped once the output nodes are counted. The table is built once on the host and given to each builder.
    topology::detail::ProcessedTileMap<RootT> tiles;
    if (mSrcTreeData.mVoxelCount) { // Unless the source grid is empty
        auto deviceSrcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));
        const uint64_t srcRootSize = mSrcTreeData.mNodeOffset[2] - mSrcTreeData.mNodeOffset[3];
        auto srcRootBuffer = nanovdb::HostBuffer::create(srcRootSize);
        cudaCheck(cudaMemcpyAsync(srcRootBuffer.data(), deviceSrcRoot, srcRootSize, cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        auto srcRoot = static_cast<const RootT*>(srcRootBuffer.data());
        for (uint32_t t = 0; t < srcRoot->tileCount(); t++)
            topology::detail::insertProcessedTile<RootT>(tiles, srcRoot->tile(t)->origin());
    }

    for (auto& builder : mBuilders) {
        topology::detail::packProcessedRoot(tiles, builder.allocateProcessedRootTiles(static_cast<uint32_t>(tiles.size())));
        builder.uploadProcessedRoot(mStream);
    }
}// PartitionGridBinary<BuildT, ResourceT>::partitionRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGridBinary<BuildT, ResourceT>::partitionInternalNodes()
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
}// PartitionGridBinary<BuildT, ResourceT>::partitionInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGridBinary<BuildT, ResourceT>::processGridTreeRoot(BuilderT& builder)
{
    // Copies the source GridData (name, map, ...), then resets what the builder owns
    cudaCheck(cudaMemcpyAsync(&builder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage(), cudaMemcpyDeviceToDevice, mStream));
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, topology::detail::BuildGridTreeRootFunctor<BuildT>(), builder.deviceData());
    cudaCheckError();
}// PartitionGridBinary<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
void PartitionGridBinary<BuildT, ResourceT>::partitionLeafNodes()
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
}// PartitionGridBinary<BuildT, ResourceT>::partitionLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief Splits the active voxels of an index grid into one grid per label.
/// @details Output l holds the active voxels whose label is l, and is a valid empty grid if no voxel has label l.
///          The labels are split in two at the label that halves the voxel count, by one PartitionGridBinary, and
///          each side is split again until it holds a single label: labelCount - 1 splits in all, each over the
///          voxels of its side only. A single label still takes one split, which keeps every voxel: the source
///          itself is not returned.
/// @tparam BuildT NanoVDB build type; only ValueOnIndex is supported.
/// @tparam ResourceT Stream-ordered resource used for all scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PartitionGrid
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex,
                  "PartitionGrid partitions the active voxels of a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PartitionGrid allocates stream-ordered scratch and requires an AsyncResource");

    using GridT = NanoGrid<BuildT>;

    /// @brief Stream-ordered, device-only scratch storage borrowed from the injected resource.
    template <typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;

public:
    /// @brief Maximum number of labels: the label histogram takes its level count, labelCount + 1, as an int.
    static constexpr uint32_t MaxLabelCount = uint32_t(std::numeric_limits<int>::max()) - 1;

    /// @brief Constructor
    /// @param d_srcGrid source device grid to be partitioned
    /// @param d_srcLabels label of each active voxel, indexed by the source grid's value index (leaf.getValue(n)),
    ///        e.g. the labels of ConnectedComponents. Every active voxel's label must be in [0, labelCount); the
    ///        other entries do not affect the outputs
    /// @param labelCount number of labels, and of outputs; at most MaxLabelCount
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @param resource resource instance all device scratch is allocated from;
    ///        must outlive this operator (defaults to the per-type default resource)
    PartitionGrid(const GridT* d_srcGrid, const uint32_t* d_srcLabels, uint32_t labelCount, cudaStream_t stream = 0,
                  ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream), mTimer(stream), mResource(&resource), mDeviceSrcGrid(d_srcGrid)
        , mDeviceSrcLabels(d_srcLabels), mLabelCount(labelCount) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the checksum mode of the outputs, disabled by default.
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable) { mChecksum = mode; }

    /// @brief The outputs: [l] the voxels whose label is l.
    /// @tparam BufferT Buffer type of the returned grid handles.
    /// @param pool Prototype buffer the outputs, and the grids split on the way, take their resource from.
    /// @note The stream is synchronized before this method returns.
    /// @throw std::runtime_error if labelCount exceeds MaxLabelCount, or if the grid has active tiles.
    template<typename BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::DeviceResource>>
    std::vector<GridHandle<BufferT>> getHandles(const BufferT& pool = BufferT());

    /// @brief Resource form of getHandles: the outputs, and the grids split on the way, allocate through this
    ///        builder's resource, the one its scratch uses, on its stream.
    /// @return handles over cuda::Buffer<std::byte, ResourceRef<ResourceT>>,
    ///         which borrow the resource: it must outlive the handles
    std::vector<GridHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>>> build()
    {
        using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;
        return this->getHandles(BufferT(mStream, this->ref(), 0, nanovdb::cuda::noInit));
    }

private:

    // --- Pipeline stages (run in order by getHandles). ---

    // Stage 1: the voxel count of each label -> mLabelOffsets.
    void countLabels();

    // Stage 2: the labels [lo, hi), whose voxels the grid of handle holds, split recursively into outputs[lo, hi),
    // which are appended in label order.
    template<typename BufferT>
    void split(GridHandle<BufferT> handle, uint32_t lo, uint32_t hi, const BufferT& pool,
               std::vector<GridHandle<BufferT>>& outputs);

    // The label that splits [lo, hi) in two, by voxel count.
    uint32_t pivotLabel(uint32_t lo, uint32_t hi) const;

    // One split: [0] the voxels of d_grid with a label below pivot, [1] the rest.
    template<typename BufferT>
    std::array<GridHandle<BufferT>, 2> splitAt(const GridT* d_grid, uint32_t pivot, const BufferT& pool);

    /// @brief Borrowed handle to the injected resource, for the scratch allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    cudaStream_t          mStream{0};
    util::cuda::Timer     mTimer;
    int                   mVerbose{0};
    CheckMode             mChecksum{CheckMode::Disable};
    ResourceT*            mResource;     // non-owning; all device scratch routes through this instance
    const GridT*          mDeviceSrcGrid;
    const uint32_t*       mDeviceSrcLabels;
    uint32_t              mLabelCount;
    std::vector<uint64_t> mLabelOffsets; // [l], for l in [0, labelCount]: the number of label entries below l
};// tools::cuda::PartitionGrid<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename BufferT>
std::vector<GridHandle<BufferT>> PartitionGrid<BuildT, ResourceT>::getHandles(const BufferT& pool)
{
    if (mLabelCount > MaxLabelCount)
        throw std::runtime_error("nanovdb::tools::cuda::PartitionGrid: labelCount exceeds MaxLabelCount");
    std::vector<GridHandle<BufferT>> outputs;
    if (mLabelCount == 0) return outputs;
    outputs.reserve(mLabelCount);

    if (mLabelCount == 1) {
        // Every voxel has label 0, but the source is not ours to return: rebuild it as the voxels with a label
        // below labelCount, i.e. all of them
        if (mVerbose==1) mTimer.start("\nSplitting");
        outputs.push_back(std::move(splitAt(mDeviceSrcGrid, mLabelCount, pool)[0]));
    } else {
        if (mVerbose==1) mTimer.start("\nCounting labels");
        countLabels();
        if (mVerbose==1) mTimer.restart("Splitting");
        // The source is not ours to release, so it is split here, and split() only ever gets grids it owns
        const uint32_t pivot = pivotLabel(0, mLabelCount);
        auto halves = splitAt(mDeviceSrcGrid, pivot, pool);
        split(std::move(halves[0]), 0, pivot, pool, outputs);
        split(std::move(halves[1]), pivot, mLabelCount, pool, outputs);
    }

    // Checksums of the outputs only: the splits leave them disabled, so the grids released on the way pay none
    if (mChecksum != CheckMode::Disable) {
        if (mVerbose==1) mTimer.restart("Updating checksums");
        for (auto& output : outputs) updateChecksum(output.template deviceGrid<BuildT>(), mChecksum, mStream);
    }
    if (mVerbose==1) mTimer.stop();

    cudaCheck(cudaStreamSynchronize(mStream));
    return outputs;
}// PartitionGrid<BuildT, ResourceT>::getHandles

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 1: the voxel count of each label.
//
// One histogram of the labels, computed once: a split never changes a voxel's label, only which grid holds it, so
// the counts hold for the whole recursion. Entries outside [0, labelCount), such as the background slot of
// ConnectedComponents labels, are not counted.
template<typename BuildT, typename ResourceT>
void PartitionGrid<BuildT, ResourceT>::countLabels()
{
    // The labels hold activeVoxelCount + 1 entries: slot 0 (the background), then one per active voxel
    const uint64_t activeVoxelCount = util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(
        mDeviceSrcGrid, mStream);
    BufT<uint32_t> d_counts(mStream, this->ref(), mLabelCount, nanovdb::cuda::noInit);
    nanovdb::cuda::TempPool<ResourceT> tempPool(*mResource);
    // mLabelCount <= MaxLabelCount, so the level count mLabelCount + 1 fits in an int
    cudaCheck(cub::DeviceHistogram::HistogramEven(nullptr, tempPool.requestedSize(), mDeviceSrcLabels,
                                                  d_counts.data(), int(mLabelCount) + 1, 0u, mLabelCount,
                                                  activeVoxelCount + 1, mStream));
    tempPool.reallocate(mStream);
    cudaCheck(cub::DeviceHistogram::HistogramEven(tempPool.data(), tempPool.size(), mDeviceSrcLabels,
                                                  d_counts.data(), int(mLabelCount) + 1, 0u, mLabelCount,
                                                  activeVoxelCount + 1, mStream));

    std::vector<uint32_t> counts(mLabelCount);
    cudaCheck(cudaMemcpyAsync(counts.data(), d_counts.data(), d_counts.size_bytes(), cudaMemcpyDeviceToHost, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
    mLabelOffsets.assign(mLabelCount + 1, 0);
    for (uint32_t l = 0; l < mLabelCount; l++) mLabelOffsets[l + 1] = mLabelOffsets[l] + counts[l];
}// PartitionGrid<BuildT, ResourceT>::countLabels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace partition::detail {

// Stage 2: recursive bisection of the labels.
//
// Each split builds a retain mask of the voxels with a label below the pivot and runs PartitionGridBinary with it.
// Depth first, and a grid is released once it is split, so only the grids on the path from the source and their
// siblings are alive at once.

/// @brief Retain mask of one leaf of a grid that was split off from a source grid: a voxel is retained when it is
///        active and its label is below the pivot. The labels are indexed by the source grid's value indices, so a
///        voxel's label is read through the source leaf with the same origin, which holds that voxel at the same n.
///        One warp per leaf, lane i testing voxels i, i + 32, ..., so that a warp reads the labels of 32 consecutive
///        voxels at once.
template <typename BuildT>
struct LabelBelowPivotFunctor
{
    // Intended to be called via nanovdb::util::cuda::lambdaKernel, with one warp (32 threads) per leaf
    __device__
    void operator()(
        size_t tID,
        const NanoGrid<BuildT>* grid,
        const NanoGrid<BuildT>* srcGrid,
        const uint32_t* srcLabels,
        uint32_t pivot,
        Mask<3>* leafMasks) const
    {
        const size_t leafID = tID >> 5;   // tID / 32
        const uint32_t lane = tID & 31;   // tID % 32
        const auto& leaf = grid->tree().template getFirstNode<0>()[leafID];
        // grid and srcGrid number their values differently, but a split only drops voxels and never moves them:
        // every leaf of grid has the origin of a leaf of srcGrid, and its voxel n is that source leaf's voxel n.
        // So the label of voxel n is found through the source leaf with the same origin.
        const auto* srcLeaf = srcGrid->tree().root().probeLeaf(leaf.origin());
        NANOVDB_ASSERT(srcLeaf);
        auto retained = [&](uint32_t n) { return leaf.isActive(n) && srcLabels[srcLeaf->getValue(n)] < pivot; };
        // The Mask<3> words recast as 32-bit words: one ballot fills one, bit i from lane i (voxel h * 32 + i)
        uint32_t* maskWords_u32 = reinterpret_cast<uint32_t*>(leafMasks[leafID].words());
        for (uint32_t h = 0; h < 2 * Mask<3>::WORD_COUNT; h++) {
            const uint32_t ballot = __ballot_sync(0xFFFFFFFF, retained(h * 32 + lane));
            if (lane == 0) maskWords_u32[h] = ballot;
        }
    }
};

}// namespace partition::detail

template<typename BuildT, typename ResourceT>
template<typename BufferT>
void PartitionGrid<BuildT, ResourceT>::split(GridHandle<BufferT> handle, uint32_t lo, uint32_t hi, const BufferT& pool,
                                             std::vector<GridHandle<BufferT>>& outputs)
{
    if (hi - lo == 1) {
        // Appending puts this grid at outputs[lo]: every split, the first one in getHandles included, finishes its
        // left side [lo, pivot) before it starts its right side [pivot, hi), so the labels are completed in
        // increasing order and outputs already holds exactly labels 0, ..., lo - 1
        NANOVDB_ASSERT(outputs.size() == lo);
        outputs.push_back(std::move(handle)); // this grid holds exactly the voxels of label lo
        return;
    }
    const uint32_t pivot = pivotLabel(lo, hi);
    auto halves = splitAt(handle.template deviceGrid<BuildT>(), pivot, pool);
    handle.reset();
    split(std::move(halves[0]), lo, pivot, pool, outputs);
    split(std::move(halves[1]), pivot, hi, pool, outputs);
}// PartitionGrid<BuildT, ResourceT>::split

template<typename BuildT, typename ResourceT>
uint32_t PartitionGrid<BuildT, ResourceT>::pivotLabel(uint32_t lo, uint32_t hi) const
{
    // Splits [lo, hi), hi - lo >= 2, into [lo, pivot) and [pivot, hi) with voxel counts closest to half each,
    // keeping at least one label on each side
    // A range without voxels balances equally at every cut: cut at the middle label, so that a long run of empty
    // labels is halved rather than peeled one label at a time
    if (mLabelOffsets[lo] == mLabelOffsets[hi]) return (hi + lo) / 2;
    const uint64_t half = (mLabelOffsets[lo] + mLabelOffsets[hi]) / 2;
    const auto first = mLabelOffsets.begin();
    // Binary search (mLabelOffsets is sorted): the first of lo + 1, ..., hi - 1 whose offset exceeds half, or hi
    const uint32_t above = uint32_t(std::upper_bound(first + lo + 1, first + hi, half) - first);
    const uint32_t below = above - 1; // the last at or below half
    // The best cut is above or below, whichever is closer to half, if it leaves a label on each side
    if (above == hi) return below; // no candidate exceeds half
    if (below == lo) return above; // lo would leave the left side empty
    return half - mLabelOffsets[below] <= mLabelOffsets[above] - half ? below : above;
}// PartitionGrid<BuildT, ResourceT>::pivotLabel

template<typename BuildT, typename ResourceT>
template<typename BufferT>
std::array<GridHandle<BufferT>, 2>
PartitionGrid<BuildT, ResourceT>::splitAt(const GridT* d_grid, uint32_t pivot, const BufferT& pool)
{
    const uint32_t leafCount = util::cuda::DeviceGridTraits<BuildT>::getTreeData(d_grid, mStream).mNodeCount[0];
    BufT<Mask<3>> d_leafMasks(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
    if (leafCount) {
        using MaskFunctorT = partition::detail::LabelBelowPivotFunctor<BuildT>;
        static constexpr unsigned int threadsPerBlock = 128;
        const size_t threadCount = size_t(leafCount) * 32; // one warp per leaf
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(threadCount, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            threadCount, MaskFunctorT(), d_grid, mDeviceSrcGrid, mDeviceSrcLabels, pivot, d_leafMasks.data());
        cudaCheckError();
    }
    return PartitionGridBinary<BuildT, ResourceT>(d_grid, d_leafMasks.data(), mStream, *mResource).getHandles(pool);
}// PartitionGrid<BuildT, ResourceT>::splitAt

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools::cuda

}// namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_PARTITIONGRID_CUH_HAS_BEEN_INCLUDED
