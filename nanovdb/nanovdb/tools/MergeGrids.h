// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/MergeGrids.h

    \authors Efty Sifakis

    \brief Morphological union of NanoVDB indexGrids (host-side port)
*/

#ifndef NANOVDB_TOOLS_MERGEGRIDS_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_MERGEGRIDS_H_HAS_BEEN_INCLUDED

#include <cstring>
#include <map>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/TopologyBuilder.h>
#include <nanovdb/util/Morphology.h>
#include <nanovdb/util/Timer.h>


namespace nanovdb {

namespace tools {

template <typename BuildT>
class MergeGrids
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using ScratchBufferT = typename TopologyBuilder<BuildT>::ScratchBufferT;

public:

    /// @brief Constructor
    /// @param d_srcGrid1 first source device grid to be merged
    /// @param d_srcGrid2 second source device grid to be merged
    MergeGrids(const GridT* d_srcGrid1, const GridT* d_srcGrid2)
        : mDeviceSrcGrid1(d_srcGrid1), mDeviceSrcGrid2(d_srcGrid2) {}

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
    template<typename BufferT = nanovdb::HostBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

private:
    void mergeRoot();

    void mergeInternalNodes();

    void processGridTreeRoot();

    void mergeLeafNodes();

    TopologyBuilder<BuildT> mBuilder;
    util::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid1;
    const GridT             *mDeviceSrcGrid2;
    TreeData                mSrcTreeData1;
    TreeData                mSrcTreeData2;
};// tools::MergeGrids<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
MergeGrids<BuildT>::getHandle(const BufferT &pool)
{
    // Read TreeData directly from the host-resident source grids (NanoTree is-a TreeData)
    mSrcTreeData1 = mDeviceSrcGrid1->tree();
    mSrcTreeData2 = mDeviceSrcGrid2->tree();

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData1.mTileCount[2] || mSrcTreeData1.mTileCount[1] || mSrcTreeData1.mTileCount[0] ||
        mSrcTreeData2.mTileCount[2] || mSrcTreeData2.mTileCount[1] || mSrcTreeData2.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Merge root nodes
    if (mVerbose==1) mTimer.start("\nMerging root nodes");
    mergeRoot();

    // Allocate memory for merged upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers();

    // Merge masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Merge internal nodes");
    mergeInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count merged tree nodes");
    mBuilder.countNodes();


    // Allocate new device grid buffer for merged result
    if (mVerbose==1) mTimer.restart("Allocating merged grid buffer");
    auto buffer = mBuilder.getBuffer(pool);

    // Process GridData/TreeData/RootData of merged result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes();

    // Process lower nodes of merged result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes();

    // Merge leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Merging leaf nodes");
    mergeLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox();

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree();
    if (mVerbose==1) mTimer.stop();


    return GridHandle<BufferT>(std::move(buffer));
}// MergeGrids<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeRoot()
{
    // Creates a new merged tree root with the merged tiles of the two input root topologies

    std::map<uint64_t, typename RootT::DataType::Tile> mergedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    // Read the source RootNodes directly from the host-resident source grids (no D2H copy),
    // and merge the tiles of both sources into a sorted container.

    if (mSrcTreeData1.mVoxelCount) { // If the first input is not a null grid
        auto srcRoot1 = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid1, GridT::memUsage() + mSrcTreeData1.mNodeOffset[3]));

        // Add all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot1->tileCount(); t++) {
            auto tile = srcRoot1->tile(t);
            auto sortKey = coordToKey(tile->origin());
            mergedTiles.emplace(sortKey, *tile);
        }
    }

    if (mSrcTreeData2.mVoxelCount) { // If the second input is not a null grid
        auto srcRoot2 = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid2, GridT::memUsage() + mSrcTreeData2.mNodeOffset[3]));

        // Add all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot2->tileCount(); t++) {
            auto tile = srcRoot2->tile(t);
            auto sortKey = coordToKey(tile->origin());
            mergedTiles.emplace(sortKey, *tile);
        }
    }

    // Package the new root topology into a host RootNode plus Tile list
    uint64_t rootSize = RootT::memUsage(mergedTiles.size());
    mBuilder.mProcessedRoot = ScratchBufferT::create(rootSize);
    auto mergedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    mergedRootPtr->mTableSize = mergedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : mergedTiles)
        *mergedRootPtr->tile(t++) = tile;
}// MergeGrids<BuildT>::mergeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeInternalNodes()
{
    // Merges the masks of upper and lower nodes from both input topologies into the
    // densified, pre-allocated mask arrays of the merged result.


    if (mSrcTreeData1.mNodeCount[1]) // Unless the first grid to merge is empty
        util::morphology::MergeInternalNodes<BuildT>(
            mDeviceSrcGrid1, mBuilder.hostProcessedRoot(),
            mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData1.mNodeCount[1]);
    if (mSrcTreeData2.mNodeCount[1]) // Unless the second grid to merge is empty
        util::morphology::MergeInternalNodes<BuildT>(
            mDeviceSrcGrid2, mBuilder.hostProcessedRoot(),
            mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData2.mNodeCount[1]);
}// MergeGrids<BuildT>::mergeInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void MergeGrids<BuildT>::processGridTreeRoot()
{
    // Copy GridData from first source grid
    // TODO: Check for instances where extra processing is needed
    // TODO: check that the second grid input has consistent GridData, too

    std::memcpy(&mBuilder.data()->getGrid(), mDeviceSrcGrid1->data(), GridT::memUsage());
    topology::detail::BuildGridTreeRootFunctor<BuildT>()(0, mBuilder.data());
}// MergeGrids<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MergeGrids<BuildT>::mergeLeafNodes()
{

    if (mSrcTreeData1.mNodeCount[0]) // Unless first input grid is empty
        util::morphology::MergeLeafNodes<BuildT>(
            mDeviceSrcGrid1, static_cast<GridT*>(mBuilder.data()->d_bufferPtr), mSrcTreeData1.mNodeCount[0]);
    if (mSrcTreeData2.mNodeCount[0]) // Unless second input grid is empty
        util::morphology::MergeLeafNodes<BuildT>(
            mDeviceSrcGrid2, static_cast<GridT*>(mBuilder.data()->d_bufferPtr), mSrcTreeData2.mNodeCount[0]);

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets();
}// MergeGrids<BuildT>::mergeLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools

}// namespace nanovdb

#endif // NANOVDB_TOOLS_MERGEGRIDS_H_HAS_BEEN_INCLUDED
