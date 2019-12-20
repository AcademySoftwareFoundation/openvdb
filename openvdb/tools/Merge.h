// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file Merge.h
///
/// @brief Functions to efficiently merge grids
///
/// @author Dan Bailey

#ifndef OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED

#include <openvdb/Platform.h>
#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/Grid.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/tools/Composite.h>

#include <sstream>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Merge const and non-const arrays of grids into a single output grid.
/// Non-const grids are destroyed as they are merged.
template <typename GridT>
inline typename GridT::Ptr
mergeSum(const std::vector<typename GridT::Ptr>& gridsToSteal,
         const std::vector<typename GridT::ConstPtr>& gridsToCopy)
{
    using GridPtr = typename GridT::Ptr;

    // hard-coded to root -> internal1 -> internal2 -> leaf node for now

    using TreeType = typename GridT::TreeType;
    using RootNodeType = typename TreeType::RootNodeType;
    using Internal1 = typename RootNodeType::ChildNodeType;
    using Internal2 = typename Internal1::ChildNodeType;
    using LeafNode = typename Internal2::ChildNodeType;

    GridPtr result;

    // set result grid to the first grid if there are non-const grids

    std::stringstream ss;
    ss << "Steal " << gridsToSteal.size() << " VDB Grids";
    util::CpuTimer timer(ss.str());

    auto gridIter = gridsToSteal.begin();
    if (!gridsToSteal.empty()) {
        // first grid is stolen
        result = gridsToSteal.front();
        ++gridIter;
    } else if (!gridsToCopy.empty()) {
        // first grid is to be deep-copied, create a new (empty) grid using this transform
        GridBase::Ptr gridBase = gridsToCopy.front()->copyGridWithNewTree();
        result = gridPtrCast<FloatGrid>(gridBase);
    }

    // steal and insert grids from non-const array (except first grid if used for result)

    std::vector<GridPtr> grids;
    grids.reserve(gridsToSteal.size()+gridsToCopy.size());
    grids.insert(grids.begin(), gridIter, gridsToSteal.end());

    // deep-copy and insert grids from const array

    ss.str(""); ss << "Deep Copy " << gridsToCopy.size() << " VDB Grids";
    timer.restart(ss.str());

    for (auto constGridIter = gridsToCopy.begin();
        constGridIter != gridsToCopy.end(); ++constGridIter) {
        assert(*constGridIter);
        grids.emplace_back((*constGridIter)->deepCopy());
    }

    timer.restart("Steal Internal1 Nodes in Serial");

    auto& root = result->tree().root();

    // insert all existing keys into an unordered set

    std::unordered_set<Coord> keys;
    keys.reserve(root.getTableSize());
    for (auto iter = root.cbeginChildAll(); iter; ++iter) {
        const Coord& key = iter.getCoord();
        keys.insert(key);
    }

    // steal internal1 nodes that don't exist in the output grid
    // this is done in serial because the root node does not support concurrent insertion,
    // (usually not a bottleneck due to the high fan-out factor)

    for (GridPtr& grid : grids) {
        auto& otherRoot = grid->tree().root();
        auto background = otherRoot.background();
        for (auto iter = otherRoot.cbeginChildAll(); iter; ++iter) {
            const Coord& key = iter.getCoord();
            // TODO: test if iter is an active child
            if (!keys.count(key)) {
                auto* child = otherRoot.template stealNode<Internal1>(
                    key, background, false);
                assert(child);
                root.addChild(child);
                keys.insert(key);
            }
        }
    }

    timer.restart("Steal Internal2 Nodes in Parallel");

    // steal internal2 nodes that don't exist in the output grid
    // this is parallelized across internal1 nodes so is deterministic

    tbb::parallel_for(tbb::blocked_range<size_t>(0, root.getTableSize()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (auto r = range.begin(); r != range.end(); ++r) {
                auto rootIter = root.beginChildOn();
                rootIter.increment(static_cast<Index>(r));
                for (GridPtr& grid : grids) {
                    auto& otherRoot = grid->tree().root();
                    auto background = otherRoot.background();
                    auto* otherNode = otherRoot.template probeNode<Internal1>(rootIter.getCoord());
                    if (!otherNode)    continue;
                    for (auto iter = otherNode->cbeginChildOn(); iter; ++iter) {
                        Index pos = iter.pos();
                        if (otherNode->isChildMaskOn(pos) && rootIter->isChildMaskOff(pos)) {
                            auto* otherChild = otherNode->template stealNode<Internal2>(iter.getCoord(), background, false);
                            rootIter->addChild(otherChild);
                        }
                    }
                }
            }
        }
    );

    timer.restart("Steal or Merge Leaf Nodes in Parallel");

    // get a list of pointers to internal2 nodes

    std::deque<Internal2*> internalNodes;
    root.getNodes(internalNodes);

    // steal leaf nodes that don't exist in the output grid
    // combine using a sum with existing leaf nodes if they do
    // this is parallelized across internal2 nodes so is deterministic

    struct Local {
        static inline void op(CombineArgs<typename TreeType::ValueType>& args) {
            args.setResult(args.a() + args.b());
        }
    };

    tbb::parallel_for(tbb::blocked_range<size_t>(0, internalNodes.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (auto r = range.begin(); r != range.end(); ++r) {
                Internal2& node = *internalNodes[r];
                for (GridPtr& grid : grids) {
                    auto& otherRoot = grid->tree().root();
                    auto background = otherRoot.background();
                    Internal2* otherNode = otherRoot.template probeNode<Internal2>(node.origin());
                    if (!otherNode)     continue;
                    for (auto iter = otherNode->beginChildOn(); iter; ++iter) {
                        Index pos = iter.pos();
                        if (!otherNode->isChildMaskOn(pos))    continue;
                        if (node.isChildMaskOff(pos)) {
                            auto* otherLeaf = otherNode->template stealNode<LeafNode>(iter.getCoord(), background, false);
                            node.addLeaf(otherLeaf);
                        } else {
                            auto* leaf = node.probeLeaf(iter.getCoord());
                            leaf->template combine(*iter, Local::op);
                        }
                    }
                }
            }
        }
    );

    timer.stop();

    return result;
}


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TOOLS_MERGE_HAS_BEEN_INCLUDED
