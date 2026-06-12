// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/PruneGrid.h

    \authors Efty Sifakis

    \brief Morphological pruning of NanoVDB indexGrids using a leaf-indexed mask (host-side port)

    \warning In this initial stage of the host-side port, this header still contains cuda device
             code; include it only from .cu files (or other .cuh files) until CUDA dependencies
             are progressively removed.
*/

#ifndef NANOVDB_TOOLS_PRUNEGRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_PRUNEGRID_H_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>

#include <cstring>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/TopologyBuilder.h>
#include <nanovdb/util/Morphology.h>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Morphology.cuh>
#include <nanovdb/util/Timer.h>
#include <nanovdb/util/cuda/Util.h>


namespace nanovdb {

namespace tools {

template <typename BuildT>
class PruneGrid
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

    // Storage policy for host-visible scratch (end state of the port is HostBuffer).
    using ScratchBufferT = typename TopologyBuilder<BuildT>::ScratchBufferT;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device grid to be pruned
    /// @param d_srcLeafMask sidecar array of leaf masks for voxels to retain
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    PruneGrid(const GridT* d_srcGrid, const Mask<3>* d_srcLeafMask, cudaStream_t stream = 0)
        : mBuilder(stream), mStream(stream), mDeviceSrcGrid(d_srcGrid), mDeviceSrcLeafMask(d_srcLeafMask) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the pruned grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::HostBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

private:
    void pruneRoot();

    void pruneInternalNodes();

    void processGridTreeRoot();

    void pruneLeafNodes();

    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT> mBuilder;
    cudaStream_t            mStream{0};
    util::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid;
    const Mask<3>           *mDeviceSrcLeafMask;
    TreeData                mSrcTreeData;
};// tools::PruneGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
PruneGrid<BuildT>::getHandle(const BufferT &pool)
{
    // Copy TreeData from GPU -> CPU
    cudaStreamSynchronize(mStream);
    mSrcTreeData = mDeviceSrcGrid->tree();  // host-resident source grid (NanoTree is-a TreeData)

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively prune root node
    if (mVerbose==1) mTimer.start("\nPrune root node");
    pruneRoot();

    // Allocate memory for pruned upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);

    // Prune masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Prune internal nodes");
    pruneInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count pruned tree nodes");
    mBuilder.countNodes(mStream);

    cudaStreamSynchronize(mStream);

    // Allocate new device grid buffer for pruned result
    if (mVerbose==1) mTimer.restart("Allocating pruned grid buffer");
    auto buffer = mBuilder.getBuffer(pool, mStream);

    // Process GridData/TreeData/RootData of pruned result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of pruned result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);

    // Process lower nodes of pruned result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);

    // Prune active masks of leaf nodes and rebuild offsets
    if (mVerbose==1) mTimer.restart("Pruning leaf nodes");
    pruneLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox(mStream);

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// PruneGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void PruneGrid<BuildT>::pruneRoot()
{
    // This method conservatively (and trivially) prunes the root tile table.
    // For this simple approximation, it is assumed that all root tiles currently present will persist,
    // and they will be pruned at a later stage if deemed empty.

    std::map<uint64_t, typename RootT::DataType::Tile> prunedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Read the source RootNode directly from the managed source grid -- no D2H copy needed
        // (UnifiedBuffer is host-accessible, drained by the stream-sync at the top of getHandle).
        auto srcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));

        // Carry over all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot->tileCount(); t++) {
            auto tile = srcRoot->tile(t);
            auto sortKey = coordToKey(tile->origin());
            prunedTiles.emplace(sortKey, *tile);
        }
    }

    // Package the duplicated root topology into a host RootNode plus Tile list
    uint64_t rootSize = RootT::memUsage(prunedTiles.size());
    mBuilder.mProcessedRoot = ScratchBufferT::create(rootSize);
    auto prunedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    prunedRootPtr->mTableSize = prunedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : prunedTiles)
        *prunedRootPtr->tile(t++) = tile;
}// PruneGrid<BuildT>::pruneRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void PruneGrid<BuildT>::pruneInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the pruning
    // operation, from the source leaves retained under the leaf-mask sidecar.

    // Drain upstream device work (allocateInternalMaskBuffers zero-fills mUpperMasks/mLowerMasks
    // on the stream) before the host prunes into them and reads the source grid/mask host-side.
    cudaCheck(cudaStreamSynchronize(mStream));

    if (mSrcTreeData.mNodeCount[0]) // Unless it's an empty grid
        util::morphology::PruneInternalNodes<BuildT>(
            mDeviceSrcGrid, mBuilder.hostProcessedRoot(), mDeviceSrcLeafMask,
            mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData.mNodeCount[0]);
}// PruneGrid<BuildT>::pruneInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void PruneGrid<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later

    // Drain upstream device work (getBuffer's cudaMemsetAsync zero-fill of the output buffer)
    // before the host writes the grid header and builds grid/tree/root metadata.
    cudaCheck(cudaStreamSynchronize(mStream));
    std::memcpy(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage());
    topology::detail::BuildGridTreeRootFunctor<BuildT>()(0, mBuilder.data());
}// PruneGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void PruneGrid<BuildT>::pruneLeafNodes()
{
    // Prunes the active masks of the source grid to the intersection with the leaf-mask sidecar,
    // followed by rebuilding the leaf offsets.

    // Drain upstream work on the stream before the host writes the output leaf masks and reads
    // the source grid/mask host-side.
    cudaCheck(cudaStreamSynchronize(mStream));

    if (mSrcTreeData.mNodeCount[0]) // Unless grid is empty
        util::morphology::PruneLeafMasks<BuildT>(
            mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr),
            mDeviceSrcLeafMask, mSrcTreeData.mNodeCount[0]);

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets(mStream);
}// PruneGrid<BuildT>::pruneLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools

}// namespace nanovdb

#endif // NANOVDB_TOOLS_PRUNEGRID_H_HAS_BEEN_INCLUDED
