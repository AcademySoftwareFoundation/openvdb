// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/CoarsenGrid.h

    \authors Efty Sifakis

    \brief 2x topological coarsening of NanoVDB indexGrids (host-side port)

    \warning In this initial stage of the host-side port, this header still contains cuda device
             code; include it only from .cu files (or other .cuh files) until CUDA dependencies
             are progressively removed.
*/

#ifndef NANOVDB_TOOLS_COARSENGRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_COARSENGRID_H_HAS_BEEN_INCLUDED

#include <cstring>
#include <map>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/TopologyBuilder.h>
#include <nanovdb/util/Morphology.h>
#include <nanovdb/util/MorphologyHelpers.h>
#include <nanovdb/util/Timer.h>


namespace nanovdb {

namespace tools {

template <typename BuildT>
class CoarsenGrid
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

    // Storage policy for host-visible scratch (end state of the port is HostBuffer).
    using ScratchBufferT = typename TopologyBuilder<BuildT>::ScratchBufferT;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device grid to be coarsened
    CoarsenGrid(const GridT* d_srcGrid)
        : mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Creates a handle to the coarsened grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::HostBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

private:
    void coarsenRoot();

    void coarsenInternalNodes();

    void processGridTreeRoot();

    void coarsenLeafNodes();

    TopologyBuilder<BuildT> mBuilder;
    util::Timer       mTimer;
    int                     mVerbose{0};
    const GridT             *mDeviceSrcGrid;
    TreeData                mSrcTreeData;
};// tools::CoarsenGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
CoarsenGrid<BuildT>::getHandle(const BufferT &pool)
{
    // Read TreeData directly from the host-resident source grid (NanoTree is-a TreeData)
    mSrcTreeData = mDeviceSrcGrid->tree();  // host-resident source grid (NanoTree is-a TreeData)

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Coarsen root node
    if (mVerbose==1) mTimer.start("\nCoarsening root node");
    coarsenRoot();

    // Allocate memory for coarsened upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers();

    // Coarsen masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Coarsening internal nodes");
    coarsenInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count coarsened tree nodes");
    mBuilder.countNodes();


    // Allocate new device grid buffer for coarsened result
    if (mVerbose==1) mTimer.restart("Allocating coarsened grid buffer");
    auto buffer = mBuilder.getBuffer(pool);

    // Process GridData/TreeData/RootData of coarsened result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of coarsened result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes();

    // Process lower nodes of coarsened result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes();

    // Coarsen leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Coarsen leaf nodes");
    coarsenLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox();

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree();
    if (mVerbose==1) mTimer.stop();


    return GridHandle<BufferT>(std::move(buffer));
}// CoarsenGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void CoarsenGrid<BuildT>::coarsenRoot()
{
    // This method coarsens the root tiles, to accommodate for the overall downsamping operation.

    std::map<uint64_t, typename RootT::DataType::Tile> coarsenedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Read the source RootNode directly from the host-resident source grid -- no copy needed.
        auto srcRoot = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));

        // Add all root tiles, reordering if necessary
        for (uint32_t t = 0; t < srcRoot->tileCount(); t++) {
            auto coarsenedOrigin = util::morphology::coarsenCoord(srcRoot->tile(t)->origin());
            auto sortKey = coordToKey(coarsenedOrigin); // key used in the radix sort, in accordance with PointsToGrid
            auto tileKey = RootT::CoordToKey(coarsenedOrigin); // encoding used in the NanoVDB tile
            typename RootT::Tile coarsenedTile{tileKey}; // Only the key value is needed; child pointer & value will be unused
            coarsenedTiles.emplace(sortKey, coarsenedTile);
        }
    }

    // Package the new root topology into a host RootNode plus Tile list
    uint64_t rootSize = RootT::memUsage(coarsenedTiles.size());
    mBuilder.mProcessedRoot = ScratchBufferT::create(rootSize);
    auto coarsenedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    coarsenedRootPtr->mTableSize = coarsenedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : coarsenedTiles)
        *coarsenedRootPtr->tile(t++) = tile;
}// CoarsenGrid<BuildT>::coarsenRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void CoarsenGrid<BuildT>::coarsenInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the coarsening
    // operation. Masks of lower internal nodes are densified in the sense that a serialized array of
    // them is allocated, as if every upper node had a full set of 32^3 lower children.

    if (mSrcTreeData.mNodeCount[0]) // Unless it's an empty grid
        util::morphology::CoarsenInternalNodes<BuildT>(
            mDeviceSrcGrid, mBuilder.hostProcessedRoot(),
            mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData.mNodeCount[0]);
}// CoarsenGrid<BuildT>::coarsenInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void CoarsenGrid<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later

    std::memcpy(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage());
    topology::detail::BuildGridTreeRootFunctor<BuildT>()(0, mBuilder.data());
}// CoarsenGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void CoarsenGrid<BuildT>::coarsenLeafNodes()
{
    // Coarsens the active masks of the source grid (as indicated at the leaf level), into a new grid that
    // has been already topologically coarsened to include all necessary leaf nodes.

    if (mSrcTreeData.mNodeCount[0]) // Unless grid is empty
        util::morphology::CoarsenLeafMasks<BuildT>(
            mDeviceSrcGrid, static_cast<GridT*>(mBuilder.data()->d_bufferPtr),
            mSrcTreeData.mNodeCount[0]);

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets();
}// CoarsenGrid<BuildT>::coarsenLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools

}// namespace nanovdb

#endif // NANOVDB_TOOLS_COARSENGRID_H_HAS_BEEN_INCLUDED
