// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridStats.h

    \author Ken Museth

    \date August 29, 2020

    \brief Re-computes min/max/bbox information for each node in a pre-existing NanoVDB grid.
*/

#ifndef NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"
#include "Range.h"
#include "ForEach.h"
#include "GridBuilder.h"// for Extrema
#include "GridChecksum.h"

#include <atomic>

namespace nanovdb {

/// @brief Re-computes the min/max and bbox information for an existing NaoVDB Grid
///
/// @param grid  Grid whoes stats to update
/// @param mode  Mode of computation for the checksum.
template <typename ValueT = float>
void gridStats(NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Default);

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template <typename ValueT, typename ExtremaOp = Extrema<ValueT> >
class GridStats
{
    using Node0 = LeafNode<ValueT>; // leaf
    using Node1 = InternalNode<Node0>; // lower
    using Node2 = InternalNode<Node1>; // upper
    using RootT = RootNode<Node2>;

    NanoGrid<ValueT>*     mGrid;
    ValueT                mDelta;// skip node if: node.max < -mDelta || node.min > mDelta
    std::atomic<uint64_t> mActiveVoxelCount;
    
    // Below are private methods use to serialize nodes into NanoVDB
    void processLeafs();   
    template <typename NodeT>
    void processNodes();
    void processRoot();
    void processGrid();
    void postProcessGrid(ChecksumMode mode);

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

public:
    GridStats() : mGrid(nullptr) {}

    void operator()(NanoGrid<ValueT> &grid, ChecksumMode mode = ChecksumMode::Default, ValueT delta = ValueT(0));

};// GridStats

//================================================================================================

template <typename ValueT, typename ExtremaOp>
void GridStats<ValueT, ExtremaOp>::operator()(NanoGrid<ValueT> &grid, ChecksumMode mode, ValueT delta)
{
    mGrid = &grid;
    mDelta = delta;// delta = voxel size for level sets, else 0
    mActiveVoxelCount = 0;
    this->processLeafs();
    this->template processNodes<Node1>();
    this->template processNodes<Node2>();
    this->processRoot();
    this->processGrid();
    this->postProcessGrid( mode );
}

//================================================================================================

template<typename ValueT, typename ExtremaOp>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridStats<ValueT, ExtremaOp>::
setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridStats<ValueT, ExtremaOp>::
processLeafs()
{
    auto &tree = mGrid->tree();
    auto kernel = [&](const Range1D &r) {
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i) {
            Node0 *leaf = tree.template getNode<Node0>(i);
            auto *data = leaf->data();
            auto iter = data->mValueMask.beginOn();
            if (!iter) throw std::runtime_error("Expected at least one active voxel in every leaf node!");
            sum += data->mValueMask.countOn();
            const ValueT* src = data->mValues;
            ExtremaOp extrema( src[*iter] );
            CoordBBox bbox;// empty
            bbox.expand(Node0::OffsetToLocalCoord(*iter));// initially use local coord for speed
            for (++iter; iter; ++iter) {
                bbox.expand(Node0::OffsetToLocalCoord(*iter));
                extrema( src[*iter] );
            }
            assert(!bbox.empty());
            leaf->localToGlobalCoord(bbox[0]);
            leaf->localToGlobalCoord(bbox[1]);
            data->mBBoxDif[0] = uint8_t(bbox[1][0] - bbox[0][0]);
            data->mBBoxDif[1] = uint8_t(bbox[1][1] - bbox[0][1]);
            data->mBBoxDif[2] = uint8_t(bbox[1][2] - bbox[0][2]);
            data->mBBoxMin = bbox[0];
            data->mValueMin = extrema.min();
            data->mValueMax = extrema.max();
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
        }
        mActiveVoxelCount += sum;
    };
    forEach(0, tree.nodeCount(0), 8, kernel);
} // GridStats::processLeafs

//================================================================================================
template<typename ValueT, typename ExtremaOp>
template<typename NodeT>
void GridStats<ValueT, ExtremaOp>::
processNodes()
{
    using ChildT = typename NodeT::ChildNodeType;
    auto &tree = mGrid->tree();
    auto kernel = [&](const Range1D &r) 
    {
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i) {
            NodeT *node = tree.template getNode<NodeT>(i);
            auto *data = node->data();
            sum += ChildT::NUM_VALUES * data->mValueMask.countOn();// active tiles
            auto onValIter = data->mValueMask.beginOn();
            auto childIter = data->mChildMask.beginOn();
            ExtremaOp extrema;
            if (onValIter) {
                extrema = ExtremaOp(data->mTable[*onValIter].value);
                const Coord ijk = node->offsetToGlobalCoord(*onValIter);
                data->mBBox[0] = ijk;
                data->mBBox[1] = ijk + Coord(int32_t(ChildT::DIM) - 1);
                ++onValIter;
            } else if (childIter) {
                auto* child = data->child(*childIter);
                extrema = ExtremaOp(child->valueMin(), child->valueMax());
                data->mBBox = child->bbox();
                ++childIter;
            } else {
                throw std::runtime_error("Internal node with no children or active values! Hint: try pruneInactive.");
            }
            for (; onValIter; ++onValIter) { // typically there are few active tiles
                extrema( data->mTable[*onValIter].value );
                const Coord ijk = node->offsetToGlobalCoord(*onValIter);
                data->mBBox[0].minComponent(ijk);
                data->mBBox[1].maxComponent(ijk + Coord(int32_t(ChildT::DIM) - 1));
            }
            for (; childIter; ++childIter) {
                auto* child = data->child(*childIter);
                extrema.min( child->valueMin() );
                extrema.max( child->valueMax() );
                const auto& bbox = child->bbox();
                data->mBBox[0].minComponent(bbox[0]);
                data->mBBox[1].maxComponent(bbox[1]);
            }
            data->mValueMin = extrema.min();
            data->mValueMax = extrema.max();
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
        }
        mActiveVoxelCount += sum;
    };
    forEach(0, tree.template nodeCount<NodeT>(), 4, kernel);
} // GridStats::processNodes

//================================================================================================
template<typename ValueT, typename ExtremaOp>
void GridStats<ValueT, ExtremaOp>::
processRoot()
{
    using ChildT = Node2;
    RootT &root = mGrid->tree().root();
    //auto &root = const_cast<RootT&>(mGrid->tree().root());
    auto &data = *root.data();
    if (data.mTileCount == 0) { // empty root node
        data.mValueMin = data.mValueMax = data.mBackground;
        data.mBBox[0] = Coord::max(); // set to an empty bounding box
        data.mBBox[1] = Coord::min();
        data.mActiveVoxelCount = 0;
    } else {
        ExtremaOp extrema;// invalid
        for (uint32_t i = 0; i<data.mTileCount; ++i) {
            auto& tile = data.tile(i);
            if (tile.isChild()) {// process child node
                auto& child = data.child(tile);
                if (!extrema) {
                    extrema = ExtremaOp( child.valueMin(), child.valueMax() );
                    assert(extrema);
                    data.mBBox = child.bbox();
                } else {
                    extrema.min( child.valueMin() );
                    extrema.max( child.valueMax() );
                    data.mBBox[0].minComponent(child.bbox()[0]);
                    data.mBBox[1].maxComponent(child.bbox()[1]);
                }
            } else {// process tile
                if (tile.state) {// active tile
                    mActiveVoxelCount += ChildT::NUM_VALUES;
                    const Coord ijk = tile.origin();
                    if (!extrema) {
                        extrema = ExtremaOp(tile.value);
                        assert(extrema);
                        data.mBBox[0] = ijk;
                        data.mBBox[1] = ijk + Coord(ChildT::DIM - 1);
                    } else {
                        extrema( tile.value );
                        data.mBBox[0].minComponent(ijk);
                        data.mBBox[1].maxComponent(ijk + Coord(ChildT::DIM - 1)); 
                    }
                }
            }
        }
        data.mValueMin = extrema.min();
        data.mValueMax = extrema.max();
        data.mActiveVoxelCount = mActiveVoxelCount;
        if (!extrema) std::cerr << "\nWarning: input tree only contained inactive root tiles! While not strictly an error it's suspecious." << std::endl;
    }
}// GridStats::processRoot

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridStats<ValueT, ExtremaOp>::
processGrid()
{
    // set world space AABB
    const auto& indexBBox = mGrid->tree().root().bbox();
    const auto &map = mGrid->map();
    auto &data = *mGrid->data();
    auto &worldBBox = data.mWorldBBox;
    worldBBox[0] = worldBBox[1] = map.applyMap(Vec3d(indexBBox[0][0], indexBBox[0][1], indexBBox[0][2]));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[0][1], indexBBox[1][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[1][1], indexBBox[0][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[0][1], indexBBox[0][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[1][1], indexBBox[0][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[0][1], indexBBox[1][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[1][1], indexBBox[1][2])));
    worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[1][1], indexBBox[1][2])));
    
    // set bit flags
    data.setMinMax(true);
    data.setBBox(true);
}// GridStats::processGrid

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridStats<ValueT, ExtremaOp>::
postProcessGrid(ChecksumMode mode)
{
    auto &data = *mGrid->data();
    data.mChecksum = checksum(*mGrid, mode);
}// GridStats::postProcessGrid

//================================================================================================

template <typename ValueT>
void gridStats(NanoGrid<ValueT> &grid, ChecksumMode mode)
{
    GridStats<ValueT> stats;
    stats( grid, mode );
}

} // namespace nanovdb

#endif // NANOVDB_GRIDSTATS_H_HAS_BEEN_INCLUDED
