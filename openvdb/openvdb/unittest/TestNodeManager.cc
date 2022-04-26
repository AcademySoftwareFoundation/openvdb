// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/NodeManager.h>
#include <openvdb/tree/LeafManager.h>
#include "util.h" // for unittest_util::makeSphere()
#include <gtest/gtest.h>


class TestNodeManager: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


namespace {

template<typename TreeT>
struct NodeCountOp {
    NodeCountOp() : nodeCount(TreeT::DEPTH, 0), totalCount(0)
    {
    }
    NodeCountOp(const NodeCountOp&, tbb::split)
        : nodeCount(TreeT::DEPTH, 0), totalCount(0)
    {
    }
    void join(const NodeCountOp& other)
    {
        for (size_t i = 0; i < nodeCount.size(); ++i) {
            nodeCount[i] += other.nodeCount[i];
        }
        totalCount += other.totalCount;
    }
    // do nothing for the root node
    bool operator()(const typename TreeT::RootNodeType&, size_t = 0)
    {
        return true;
    }
    // count the internal and leaf nodes
    template<typename NodeT>
    bool operator()(const NodeT&, size_t = 0)
    {
        ++(nodeCount[NodeT::LEVEL]);
        ++totalCount;
        return true;
    }
    std::vector<openvdb::Index64> nodeCount;
    openvdb::Index64 totalCount;
};// NodeCountOp

}//unnamed namespace

TEST_F(TestNodeManager, testAll)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;
    using openvdb::Vec3f;
    using openvdb::Index64;
    using openvdb::FloatGrid;
    using openvdb::FloatTree;

    const Vec3f center(0.35f, 0.35f, 0.35f);
    const float radius = 0.15f;
    const int dim = 128, half_width = 5;
    const float voxel_size = 1.0f/dim;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/half_width*voxel_size);
    FloatTree& tree = grid->tree();
    grid->setTransform(openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size));

    unittest_util::makeSphere<FloatGrid>(Coord(dim), center,
                                         radius, *grid, unittest_util::SPHERE_SPARSE_NARROW_BAND);

    EXPECT_EQ(4, int(FloatTree::DEPTH));
    EXPECT_EQ(3, int(openvdb::tree::NodeManager<FloatTree>::LEVELS));

    std::vector<Index64> nodeCount;
    for (openvdb::Index i=0; i<FloatTree::DEPTH; ++i) nodeCount.push_back(0);
    for (FloatTree::NodeCIter it = tree.cbeginNode(); it; ++it) ++(nodeCount[it.getLevel()]);

    //for (size_t i=0; i<nodeCount.size(); ++i) {//includes the root node
    //    std::cerr << "Level=" << i << " nodes=" << nodeCount[i] << std::endl;
    //}

    {// test tree constructor
        openvdb::tree::NodeManager<FloatTree> manager(tree);

        //for (openvdb::Index i=0; i<openvdb::tree::NodeManager<FloatTree>::LEVELS; ++i) {
        //    std::cerr << "Level=" << i << " nodes=" << manager.nodeCount(i) << std::endl;
        //}

        Index64 totalCount = 0;
        for (openvdb::Index i=0; i<FloatTree::RootNodeType::LEVEL; ++i) {//exclude root in nodeCount
            //std::cerr << "Level=" << i << " expected=" << nodeCount[i]
            //          << " cached=" << manager.nodeCount(i) << std::endl;
            EXPECT_EQ(nodeCount[i], manager.nodeCount(i));
            totalCount += nodeCount[i];
        }
        EXPECT_EQ(totalCount, manager.nodeCount());

        // test the map reduce functionality
        NodeCountOp<FloatTree> bottomUpOp;
        NodeCountOp<FloatTree> topDownOp;
        manager.reduceBottomUp(bottomUpOp);
        manager.reduceTopDown(topDownOp);
        for (openvdb::Index i=0; i<FloatTree::RootNodeType::LEVEL; ++i) {//exclude root in nodeCount
            EXPECT_EQ(bottomUpOp.nodeCount[i], manager.nodeCount(i));
            EXPECT_EQ(topDownOp.nodeCount[i], manager.nodeCount(i));
        }
        EXPECT_EQ(bottomUpOp.totalCount, manager.nodeCount());
        EXPECT_EQ(topDownOp.totalCount, manager.nodeCount());
    }

    {// test LeafManager constructor
        typedef openvdb::tree::LeafManager<FloatTree> LeafManagerT;
        LeafManagerT manager1(tree);
        EXPECT_EQ(nodeCount[0], Index64(manager1.leafCount()));
        openvdb::tree::NodeManager<LeafManagerT> manager2(manager1);
        Index64 totalCount = 0;
        for (openvdb::Index i=0; i<FloatTree::RootNodeType::LEVEL; ++i) {//exclude root in nodeCount
            //std::cerr << "Level=" << i << " expected=" << nodeCount[i]
            //          << " cached=" << manager2.nodeCount(i) << std::endl;
            EXPECT_EQ(nodeCount[i], Index64(manager2.nodeCount(i)));
            totalCount += nodeCount[i];
        }
        EXPECT_EQ(totalCount, Index64(manager2.nodeCount()));

        // test the map reduce functionality
        NodeCountOp<FloatTree> bottomUpOp;
        NodeCountOp<FloatTree> topDownOp;
        manager2.reduceBottomUp(bottomUpOp);
        manager2.reduceTopDown(topDownOp);
        for (openvdb::Index i=0; i<FloatTree::RootNodeType::LEVEL; ++i) {//exclude root in nodeCount
            EXPECT_EQ(bottomUpOp.nodeCount[i], manager2.nodeCount(i));
            EXPECT_EQ(topDownOp.nodeCount[i], manager2.nodeCount(i));
        }
        EXPECT_EQ(bottomUpOp.totalCount, manager2.nodeCount());
        EXPECT_EQ(topDownOp.totalCount, manager2.nodeCount());
    }

}


TEST_F(TestNodeManager, testConst)
{
    using namespace openvdb;

    const Vec3f center(0.35f, 0.35f, 0.35f);
    const int dim = 128, half_width = 5;
    const float voxel_size = 1.0f/dim;

    FloatGrid::Ptr grid = FloatGrid::create(/*background=*/half_width*voxel_size);
    const FloatTree& tree = grid->constTree();

    tree::NodeManager<const FloatTree> nodeManager(tree);

    NodeCountOp<const FloatTree> topDownOp;
    nodeManager.reduceTopDown(topDownOp);

    std::vector<Index64> nodeCount;
    for (openvdb::Index i=0; i<FloatTree::DEPTH; ++i) nodeCount.push_back(0);
    for (FloatTree::NodeCIter it = tree.cbeginNode(); it; ++it) ++(nodeCount[it.getLevel()]);

    Index64 totalCount = 0;
    for (openvdb::Index i=0; i<FloatTree::RootNodeType::LEVEL; ++i) {//exclude root in nodeCount
        EXPECT_EQ(nodeCount[i], nodeManager.nodeCount(i));
        totalCount += nodeCount[i];
    }
    EXPECT_EQ(totalCount, nodeManager.nodeCount());
    EXPECT_EQ(totalCount, topDownOp.totalCount);

    // test DynamicNodeManager also works with a const tree

    tree::DynamicNodeManager<const FloatTree> dynamicNodeManager(tree);
    dynamicNodeManager.reduceTopDown(topDownOp);
    EXPECT_EQ(totalCount, topDownOp.totalCount);
}


namespace {

template<typename TreeT>
struct ExpandOp
{
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    explicit ExpandOp(bool zeroOnly = false) : mZeroOnly(zeroOnly) { }

    // do nothing for the root node
    bool operator()(RootT&, size_t = 1) const { return true; }

    // count the internal and leaf nodes
    template<typename NodeT>
    bool operator()(NodeT& node, size_t idx = 1) const
    {
        for (auto iter = node.cbeginValueAll(); iter; ++iter) {
            const openvdb::Coord ijk = iter.getCoord();
            if (ijk.x() < 256 && ijk.y() < 256 && ijk.z() < 256) {
                node.addChild(new typename NodeT::ChildNodeType(iter.getCoord(), NodeT::LEVEL, true));
            }
        }

        if (mZeroOnly)  return idx == 0;
        return true;
    }

    bool operator()(LeafT& leaf, size_t /*idx*/ = 1) const
    {
        for (auto iter = leaf.beginValueAll(); iter; ++iter) {
            iter.setValue(iter.pos());
        }

        return true;
    }

    bool mZeroOnly = false;
};// ExpandOp

template<typename TreeT>
struct RootOnlyOp
{
    using RootT = typename TreeT::RootNodeType;

    RootOnlyOp() = default;
    RootOnlyOp(const RootOnlyOp&, tbb::split) { }
    void join(const RootOnlyOp&) { }

    // do nothing for the root node but return false
    bool operator()(RootT&, size_t) const { return false; }

    // throw on internal or leaf nodes
    template<typename NodeOrLeafT>
    bool operator()(NodeOrLeafT&, size_t) const
    {
        OPENVDB_THROW(openvdb::RuntimeError, "Should not process nodes below root.");
    }
};// RootOnlyOp

template<typename TreeT>
struct SumOp {
    using RootT = typename TreeT::RootNodeType;
    using LeafT = typename TreeT::LeafNodeType;

    explicit SumOp(bool zeroOnly = false) : mZeroOnly(zeroOnly) { }
    SumOp(const SumOp& other, tbb::split): totalCount(0), mZeroOnly(other.mZeroOnly) { }
    void join(const SumOp& other)
    {
        totalCount += other.totalCount;
    }
    // do nothing for the root node
    bool operator()(const typename TreeT::RootNodeType&, size_t /*idx*/ = 0) { return true; }
    // count the internal nodes
    template<typename NodeT>
    bool operator()(const NodeT& node, size_t idx = 0)
    {
        for (auto iter = node.cbeginValueAll(); iter; ++iter) {
            totalCount += *iter;
        }
        if (mZeroOnly)  return idx == 0;
        return true;
    }
    // count the leaf nodes
    bool operator()(const LeafT& leaf, size_t /*idx*/ = 0)
    {
        for (auto iter = leaf.cbeginValueAll(); iter; ++iter) {
            totalCount += *iter;
        }
        return true;
    }
    openvdb::Index64 totalCount = openvdb::Index64(0);
    bool mZeroOnly = false;
};// SumOp

}//unnamed namespace

TEST_F(TestNodeManager, testDynamic)
{
    using openvdb::Coord;
    using openvdb::Index32;
    using openvdb::Index64;
    using openvdb::Int32Tree;

    using RootNodeType = Int32Tree::RootNodeType;
    using Internal1NodeType = RootNodeType::ChildNodeType;

    Int32Tree sourceTree(0);

    auto child =
        std::make_unique<Internal1NodeType>(Coord(0, 0, 0), /*value=*/1.0f);

    EXPECT_TRUE(sourceTree.root().addChild(child.release()));
    EXPECT_EQ(Index32(0), sourceTree.leafCount());
    EXPECT_EQ(Index32(2), sourceTree.nonLeafCount());

    ExpandOp<Int32Tree> expandOp;

    { // use NodeManager::foreachTopDown
        Int32Tree tree(sourceTree);
        openvdb::tree::NodeManager<Int32Tree> manager(tree);
        EXPECT_EQ(Index64(1), manager.nodeCount());
        manager.foreachTopDown(expandOp);
        EXPECT_EQ(Index32(0), tree.leafCount());

        // first level has been expanded, but node manager cache does not include the new nodes
        SumOp<Int32Tree> sumOp;
        manager.reduceBottomUp(sumOp);
        EXPECT_EQ(Index64(32760), sumOp.totalCount);
    }

    { // use DynamicNodeManager::foreachTopDown and filter out nodes below root
        Int32Tree tree(sourceTree);
        openvdb::tree::DynamicNodeManager<Int32Tree> manager(tree);
        RootOnlyOp<Int32Tree> rootOnlyOp;
        EXPECT_NO_THROW(manager.foreachTopDown(rootOnlyOp));
        EXPECT_NO_THROW(manager.reduceTopDown(rootOnlyOp));
    }

    { // use DynamicNodeManager::foreachTopDown
        Int32Tree tree(sourceTree);
        openvdb::tree::DynamicNodeManager<Int32Tree> manager(tree);
        manager.foreachTopDown(expandOp, /*threaded=*/true, /*leafGrainSize=*/32, /*nonLeafGrainSize=*/8);
        EXPECT_EQ(Index32(32768), tree.leafCount());

        SumOp<Int32Tree> sumOp;
        manager.reduceTopDown(sumOp);
        EXPECT_EQ(Index64(4286611448), sumOp.totalCount);

        SumOp<Int32Tree> zeroSumOp(true);
        manager.reduceTopDown(zeroSumOp);
        EXPECT_EQ(Index64(535855096), zeroSumOp.totalCount);
    }

    { // use DynamicNodeManager::foreachTopDown but filter nodes with non-zero index
        Int32Tree tree(sourceTree);
        openvdb::tree::DynamicNodeManager<Int32Tree> manager(tree);
        ExpandOp<Int32Tree> zeroExpandOp(true);
        manager.foreachTopDown(zeroExpandOp);
        EXPECT_EQ(Index32(32768), tree.leafCount());

        SumOp<Int32Tree> sumOp;
        manager.reduceTopDown(sumOp);
        EXPECT_EQ(Index64(550535160), sumOp.totalCount);
    }
}
