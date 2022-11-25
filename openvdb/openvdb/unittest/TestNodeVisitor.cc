// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetPlatonic.h>
#include <openvdb/tools/NodeVisitor.h>
#include <openvdb/tools/Prune.h>
#include <gtest/gtest.h>

class TestNodeVisitor: public ::testing::Test
{
};


struct NodeCountOp
{
    template <typename NodeT>
    void operator()(const NodeT&, size_t)
    {
        const openvdb::Index level = NodeT::LEVEL;
        while (level >= counts.size())     counts.emplace_back(0);
        counts[level]++;
    }

    std::vector<openvdb::Index32> counts;
}; // struct NodeCountOp


TEST_F(TestNodeVisitor, testNodeCount)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    NodeCountOp nodeCountOp;
    tools::visitNodesDepthFirst(grid->tree(), nodeCountOp);

    std::vector<Index32> nodeCount1 = nodeCountOp.counts;
    std::vector<Index32> nodeCount2 = grid->tree().nodeCount();

    EXPECT_EQ(nodeCount1.size(), nodeCount2.size());

    for (size_t i = 0; i < nodeCount1.size(); i++) {
        EXPECT_EQ(nodeCount1[i], nodeCount2[i]);
    }
}


template <typename TreeT>
struct LeafCountOp
{
    using LeafT = typename TreeT::LeafNodeType;

    template <typename NodeT>
    void operator()(const NodeT&, size_t) { }
    void operator()(const LeafT&, size_t) { count++; }

    openvdb::Index32 count{0};
}; // struct LeafCountOp


TEST_F(TestNodeVisitor, testLeafCount)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    { // non-const tree
        LeafCountOp<FloatTree> leafCountOp;
        tools::visitNodesDepthFirst(grid->tree(), leafCountOp);

        EXPECT_EQ(grid->tree().leafCount(), leafCountOp.count);
    }

    { // const tree
        LeafCountOp<FloatTree> leafCountOp;
        tools::visitNodesDepthFirst(grid->constTree(), leafCountOp);

        EXPECT_EQ(grid->tree().leafCount(), leafCountOp.count);
    }
}


struct DescendOp
{
    template <typename NodeT>
    void operator()(const NodeT&, size_t)
    {
        const openvdb::Index level = NodeT::LEVEL;
        // count the number of times the operator descends
        // from a higher-level node to a lower-level node
        if (NodeT::LEVEL < previousLevel)   count++;
        previousLevel = level;
    }

    openvdb::Index32 previousLevel{0};
    openvdb::Index32 count{0};
}; // struct DescendOp


TEST_F(TestNodeVisitor, testDepthFirst)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    DescendOp descendOp;
    tools::visitNodesDepthFirst(grid->tree(), descendOp);

    // this confirms that the visit pattern is depth-first
    EXPECT_EQ(descendOp.count, grid->tree().nonLeafCount());
}


template <typename TreeT>
struct StoreOriginsOp
{
    using RootT = typename TreeT::RootNodeType;

    StoreOriginsOp(std::vector<openvdb::Coord>& _origins)
        : origins(_origins) { }

    void operator()(const RootT&, size_t idx)
    {
        // root node has no origin
        origins[idx] = openvdb::Coord::max();
    }

    template <typename NodeT>
    void operator()(const NodeT& node, size_t idx)
    {
        origins[idx] = node.origin();
    }

    std::vector<openvdb::Coord>& origins;
}; // struct StoreOriginsOp


TEST_F(TestNodeVisitor, testOriginArray)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    std::vector<Index32> nodeCount = grid->tree().nodeCount();
    Index32 totalNodeCount(0);
    for (Index32 count : nodeCount)     totalNodeCount += count;

    // use an offset
    size_t offset = 10;

    std::vector<Coord> origins;
    origins.resize(totalNodeCount + offset);

    StoreOriginsOp<FloatTree> storeOriginsOp(origins);
    tools::visitNodesDepthFirst(grid->tree(), storeOriginsOp, offset);

    size_t idx = offset;

    // root node
    EXPECT_EQ(origins[idx++], Coord::max());

    const auto& root = grid->tree().root();
    for (auto internal1Iter = root.cbeginChildOn(); internal1Iter; ++internal1Iter) {
        EXPECT_EQ(origins[idx++], internal1Iter->origin());
        for (auto internal2Iter = internal1Iter->cbeginChildOn(); internal2Iter; ++internal2Iter) {
            EXPECT_EQ(origins[idx++], internal2Iter->origin());
            for (auto leafIter = internal2Iter->cbeginChildOn(); leafIter; ++leafIter) {
                EXPECT_EQ(origins[idx++], leafIter->origin());
            }
        }
    }

    EXPECT_EQ(idx, origins.size());
}


template <typename TreeType>
struct DeactivateOp
{
    using LeafT = typename TreeType::LeafNodeType;

    template <typename NodeT>
    void operator()(NodeT&, size_t) { }

    void operator()(LeafT& leaf, size_t)
    {
        leaf.setValuesOff();
    }
}; // struct DeactivateOp


TEST_F(TestNodeVisitor, testPartialDeactivate)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    using NodeT = FloatTree::RootNodeType::ChildNodeType::ChildNodeType;

    auto iter = grid->tree().root().beginChildOn()->beginChildOn();

    DeactivateOp<FloatTree> deactivateOp;
    tools::DepthFirstNodeVisitor<NodeT>::visit(*iter, deactivateOp);

    EXPECT_EQ(Index32(1413), grid->tree().leafCount());

    tools::pruneInactive(grid->tree());

    // a subset of the leaf nodes have now been deactivated and removed
    EXPECT_EQ(Index32(1195), grid->tree().leafCount());
}


// Functor to offset all the active values of a tree.
struct OffsetOp
{
    OffsetOp(float v): mOffset(v) { }

    template<typename NodeT>
    void operator()(NodeT& node, size_t) const
    {
        for (auto iter = node.beginValueOn(); iter; ++iter) {
            iter.setValue(iter.getValue() + mOffset);
        }
    }
private:
    const float mOffset;
};

// Functor to offset all the active values of a tree. Note
// this implementation also illustrates how different
// computation can be applied to the different node types.
template<typename TreeT>
struct OffsetByLevelOp
{
    using ValueT = typename TreeT::ValueType;
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;
    OffsetByLevelOp(const ValueT& v) : mOffset(v) {}
    // Processes the root node.
    void operator()(RootT& root, size_t) const
    {
        for (auto iter = root.beginValueOn(); iter; ++iter) {
            iter.setValue(iter.getValue() + mOffset);
        }
    }
    // Processes the leaf nodes.
    void operator()(LeafT& leaf, size_t) const
    {
        for (auto iter = leaf.beginValueOn(); iter; ++iter) {
            iter.setValue(iter.getValue() + mOffset);
        }
    }
    // Processes the internal nodes.
    template<typename NodeT>
    void operator()(NodeT& node, size_t) const
    {
        for (auto iter = node.beginValueOn(); iter; ++iter) {
            iter.setValue(iter.getValue() + mOffset);
        }
    }
private:
    const ValueT mOffset;
};

// this is the example from the documentation
TEST_F(TestNodeVisitor, testOffset)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = tools::createLevelSetCube<FloatGrid>(/*scale=*/10.0f);

    OffsetOp op(3.0f);
    tools::visitNodesDepthFirst(grid->tree(), op);

    OffsetByLevelOp<FloatTree> byLevelOp(3.0f);
    tools::visitNodesDepthFirst(grid->tree(), byLevelOp);
}
