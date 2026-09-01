// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/DilateGrid.h

    \authors Efty Sifakis

    \brief Morphological dilation of NanoVDB indexGrids (host-side port)
*/

#ifndef NANOVDB_TOOLS_DILATEGRID_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_DILATEGRID_H_HAS_BEEN_INCLUDED

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
class DilateGrid
{
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;

    // Storage policy for host-visible scratch. UnifiedBuffer makes device-allocated scratch
    // host-readable without explicit downloads, as in the host TopologyBuilder. End state of
    // the port is HostBuffer.
    using ScratchBufferT = typename TopologyBuilder<BuildT>::ScratchBufferT;

public:

    /// @brief Constructor
    /// @param d_srcGrid source device grid to be dilated
    DilateGrid(const GridT* d_srcGrid)
        : mDeviceSrcGrid(d_srcGrid) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mBuilder.mChecksum = mode;}

    /// @brief Set type of dilation operation
    /// @param op: NN_FACE=face neighbors, NN_FACE_EDGE=face and edge neighbors, NN_FACE_EDGE_VERTEX=26-connected neighbors
    void setOperation(morphology::NearestNeighbors op) { mOp = op; }

    /// @brief Creates a handle to the dilated grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::HostBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

private:
    void dilateRoot();

    void dilateInternalNodes();

    void processGridTreeRoot();

    void dilateLeafNodes();

    TopologyBuilder<BuildT>      mBuilder;
    util::Timer            mTimer;
    int                          mVerbose{0};
    const GridT                  *mDeviceSrcGrid;
    morphology::NearestNeighbors mOp{morphology::NN_FACE_EDGE_VERTEX};
    TreeData                     mSrcTreeData;
};// tools::DilateGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
DilateGrid<BuildT>::getHandle(const BufferT &pool)
{
    // Read TreeData directly from the host-resident source grid (NanoTree is-a TreeData)
    mSrcTreeData = mDeviceSrcGrid->tree();  // host-resident source grid (NanoTree is-a TreeData)

    // Ensure that the input grid contains no tile values
    if (mSrcTreeData.mTileCount[2] || mSrcTreeData.mTileCount[1] || mSrcTreeData.mTileCount[0])
        throw std::runtime_error("Topological operations not supported on grids with value tiles");

    // Speculatively dilate root node
    if (mVerbose==1) mTimer.start("\nDilating root node");
    dilateRoot();

    // Allocate memory for dilated upper/lower masks
    if (mVerbose==1) mTimer.restart("Allocating internal node mask buffers");
    mBuilder.allocateInternalMaskBuffers();

    // Dilate masks of upper/lower nodes
    if (mVerbose==1) mTimer.restart("Dilate internal nodes");
    dilateInternalNodes();

    // Enumerate tree nodes
    if (mVerbose==1) mTimer.restart("Count dilated tree nodes");
    mBuilder.countNodes();


    // Allocate new device grid buffer for dilated result
    if (mVerbose==1) mTimer.restart("Allocating dilated grid buffer");
    auto buffer = mBuilder.getBuffer(pool);

    // Process GridData/TreeData/RootData of dilated result
    if (mVerbose==1) mTimer.restart("Processing grid/tree/root");
    processGridTreeRoot();

    // Process upper nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing upper nodes");
    mBuilder.processUpperNodes();

    // Process lower nodes of dilated result
    if (mVerbose==1) mTimer.restart("Processing lower nodes");
    mBuilder.processLowerNodes();

    // Dilate leaf node active masks into new topology
    if (mVerbose==1) mTimer.restart("Dilating leaf nodes");
    dilateLeafNodes();

    // Process bounding boxes
    if (mVerbose==1) mTimer.restart("Processing bounding boxes");
    mBuilder.processBBox();

    // Post-process Grid/Tree data
    if (mVerbose==1) mTimer.restart("Post-processing grid/tree data");
    mBuilder.postProcessGridTree();
    if (mVerbose==1) mTimer.stop();


    return GridHandle<BufferT>(std::move(buffer));
}// DilateGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void DilateGrid<BuildT>::dilateRoot()
{
    // This method conservatively and speculatively dilates the root tiles, to accommodate
    // any new root nodes that might be introduced by the dilation operation.
    // The index-space bounding box of each tile is examined, and if it is within a 1-pixel of
    // intersecting any of the 26-connected neighboring root tiles, those are preemptively
    // introduced into the root topology.
    // (As of the present implementation this presumes a maximum of 1-voxel radius in dilation)
    // Root tiles that were preemptively introduced, but end up having no active contents will
    // be pruned in later stages of processing.

    std::map<uint64_t, typename RootT::DataType::Tile> dilatedTiles;

    // This encoding scheme mirrors the one used in PointsToGrid; note that it is different from Tile::key
    auto coordToKey = [](const Coord &ijk)->uint64_t{
        // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
        static constexpr int64_t kOffset = 1 << 31;
        return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
            (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
            (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42); //  x is the upper 21 bits
    };// coordToKey lambda functor

    if (mSrcTreeData.mVoxelCount) { // If the input grid is not empty
        // Read the source RootNode and its Upper Nodes (needed for the BBoxes below) directly from
        // the host-resident source grid -- no copy needed. getChild() resolves its relative offsets
        // within the same grid, so the uppers are reachable.
        auto srcRootAndUpper = static_cast<const RootT*>(util::PtrAdd(mDeviceSrcGrid, GridT::memUsage() + mSrcTreeData.mNodeOffset[3]));

        // For each original root tile, consider adding those tiles in its 26-connected neighborhood
        for (uint32_t t = 0; t < srcRootAndUpper->tileCount(); t++) {
            auto srcUpper = srcRootAndUpper->getChild(srcRootAndUpper->tile(t));
            const auto dilatedBBox = srcUpper->bbox().expandBy(1); // TODO: update/specialize if larger dilation neighborhoods are used

            static constexpr int32_t rootTileDim = UpperT::DIM; // 4096
            for (int di = -rootTileDim; di <= rootTileDim; di += rootTileDim)
            for (int dj = -rootTileDim; dj <= rootTileDim; dj += rootTileDim)
            for (int dk = -rootTileDim; dk <= rootTileDim; dk += rootTileDim) {
                auto testBBox = nanovdb::CoordBBox::createCube(srcUpper->origin().offsetBy(di,dj,dk), rootTileDim);
                auto sortKey = coordToKey(testBBox.min()); // key used in the radix sort, in accordance with PointsToGrid
                auto tileKey = RootT::CoordToKey(testBBox.min()); // encoding used in the NanoVDB tile
                if (testBBox.hasOverlap(dilatedBBox) & (dilatedTiles.count(sortKey) == 0)) {
                    typename RootT::Tile neighborTile{tileKey}; // Only the key value is needed; child pointer & value will be unused
                    dilatedTiles.emplace(sortKey, neighborTile);
                }
            }
        }
    }

    // Package the new root topology into a host RootNode plus Tile list
    uint64_t rootSize = RootT::memUsage(dilatedTiles.size());
    mBuilder.mProcessedRoot = ScratchBufferT::create(rootSize);
    auto dilatedRootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    dilatedRootPtr->mTableSize = dilatedTiles.size();
    uint32_t t = 0;
    for (const auto& [key, tile] : dilatedTiles)
        *dilatedRootPtr->tile(t++) = tile;
}// DilateGrid<BuildT>::dilateRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void DilateGrid<BuildT>::dilateInternalNodes()
{
    // Computes the masks of upper and (densified) lower internal nodes, as a result of the dilation operation.
    // Masks of lower internal nodes are densified in the sense that a serialized array of them is allocated,
    // as if every upper node had a full set of 32^3 lower children.


    if (mSrcTreeData.mNodeCount[1]) { // Unless it's an empty grid
        if (mOp == morphology::NN_FACE)
            util::morphology::DilateInternalNodes<BuildT, morphology::NN_FACE>(
                mDeviceSrcGrid, mBuilder.hostProcessedRoot(),
                mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData.mNodeCount[1]);
        else if (mOp == morphology::NN_FACE_EDGE)
            util::morphology::DilateInternalNodes<BuildT, morphology::NN_FACE_EDGE>(
                mDeviceSrcGrid, mBuilder.hostProcessedRoot(),
                mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData.mNodeCount[1]);
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX)
            util::morphology::DilateInternalNodes<BuildT, morphology::NN_FACE_EDGE_VERTEX>(
                mDeviceSrcGrid, mBuilder.hostProcessedRoot(),
                mBuilder.mUpperMasks.data(), mBuilder.mLowerMasks.data(), mSrcTreeData.mNodeCount[1]);
    }
}// DilateGrid<BuildT>::dilateInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template <typename BuildT>
void DilateGrid<BuildT>::processGridTreeRoot()
{
    // Copy GridData from source grid
    // By convention: this will duplicate grid name and map. Others will be reset later

    std::memcpy(&mBuilder.data()->getGrid(), mDeviceSrcGrid->data(), GridT::memUsage());
    topology::detail::BuildGridTreeRootFunctor<BuildT>()(0, mBuilder.data());
}// DilateGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void DilateGrid<BuildT>::dilateLeafNodes()
{
    // Dilates the active masks of the source grid (as indicated at the leaf level), into a new grid that
    // has been already topologically dilated to include all necessary leaf nodes.


    if (mBuilder.data()->nodeCount[0]) { // Unless output grid is empty
        auto dstGrid = static_cast<GridT*>(mBuilder.data()->d_bufferPtr);
        if (mOp == morphology::NN_FACE)
            util::morphology::DilateLeafNodes<BuildT, morphology::NN_FACE>(
                mDeviceSrcGrid, dstGrid, mBuilder.data()->nodeCount[0]);
        else if (mOp == morphology::NN_FACE_EDGE)
            throw std::runtime_error("dilateLeafNodes() not implemented for NN_FACE_EDGE stencil");
        else if (mOp == morphology::NN_FACE_EDGE_VERTEX)
            util::morphology::DilateLeafNodes<BuildT, morphology::NN_FACE_EDGE_VERTEX>(
                mDeviceSrcGrid, dstGrid, mBuilder.data()->nodeCount[0]);
    }

    // Update leaf offsets and prefix sums
    mBuilder.processLeafOffsets();
}// DilateGrid<BuildT>::dilateLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

}// namespace tools

}// namespace nanovdb

#endif // NANOVDB_TOOLS_DILATEGRID_H_HAS_BEEN_INCLUDED
