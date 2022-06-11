// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Merge.h>

#include <gtest/gtest.h>


using namespace openvdb;

class TestMerge: public ::testing::Test
{
};

namespace
{

auto getTileCount = [](const auto& node) -> Index
{
    Index sum = 0;
    for (auto iter = node.cbeginValueAll(); iter; ++iter)   sum++;
    return sum;
};

auto getActiveTileCount = [](const auto& node) -> Index
{
    Index sum = 0;
    for (auto iter = node.cbeginValueOn(); iter; ++iter)   sum++;
    return sum;
};

auto getInactiveTileCount = [](const auto& node) -> Index
{
    Index sum = 0;
    for (auto iter = node.cbeginValueOff(); iter; ++iter)   sum++;
    return sum;
};

auto getInsideTileCount = [](const auto& node) -> Index
{
    using ValueT = typename std::remove_reference<decltype(node)>::type::ValueType;
    Index sum = 0;
    for (auto iter = node.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() < zeroVal<ValueT>())     sum++;
    }
    return sum;
};

auto getOutsideTileCount = [](const auto& node) -> Index
{
    using ValueT = typename std::remove_reference<decltype(node)>::type::ValueType;
    Index sum = 0;
    for (auto iter = node.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() > zeroVal<ValueT>())     sum++;
    }
    return sum;
};

auto getChildCount = [](const auto& node) -> Index
{
    return node.childCount();
};

auto hasOnlyInactiveNegativeBackgroundTiles = [](const auto& node) -> bool
{
    if (getActiveTileCount(node) > Index(0))       return false;
    for (auto iter = node.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() != -node.background())  return false;
    }
    return true;
};

} // namespace


TEST_F(TestMerge, testTreeToMerge)
{
    using RootChildNode = FloatTree::RootNodeType::ChildNodeType;
    using LeafNode = FloatTree::LeafNodeType;

    { // non-const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        EXPECT_EQ(Index(1), grid->tree().leafCount());

        tools::TreeToMerge<FloatTree> treeToMerge{grid->tree(), Steal()};
        EXPECT_EQ(&grid->constTree().root(), treeToMerge.rootPtr());

        // probe root child

        const RootChildNode* nodePtr = treeToMerge.probeConstNode<RootChildNode>(Coord(8));
        EXPECT_TRUE(nodePtr);
        EXPECT_EQ(grid->constTree().probeConstNode<RootChildNode>(Coord(8)), nodePtr);

        // probe leaf node

        const LeafNode* leafNode = treeToMerge.probeConstNode<LeafNode>(Coord(8));
        EXPECT_TRUE(leafNode);
        EXPECT_EQ(grid->constTree().probeConstLeaf(Coord(8)), leafNode);
        EXPECT_EQ(Index(1), grid->tree().leafCount());
        EXPECT_EQ(Index(1), grid->tree().root().childCount());

        // steal leaf node

        std::unique_ptr<LeafNode> leafNodePtr = treeToMerge.stealOrDeepCopyNode<LeafNode>(Coord(8));
        EXPECT_TRUE(leafNodePtr);
        EXPECT_EQ(Index(0), grid->tree().leafCount());
        EXPECT_EQ(leafNodePtr->origin(), Coord(8));
        EXPECT_EQ(Index(1), grid->tree().root().childCount());

        // steal root child

        grid->tree().touchLeaf(Coord(8));
        std::unique_ptr<RootChildNode> node2Ptr = treeToMerge.stealOrDeepCopyNode<RootChildNode>(Coord(8));
        EXPECT_TRUE(node2Ptr);
        EXPECT_EQ(Index(0), grid->tree().root().childCount());

        // attempt to add leaf node tile (set value)

        grid->tree().touchLeaf(Coord(8));
        EXPECT_EQ(Index64(0), grid->tree().activeTileCount());
        treeToMerge.addTile<LeafNode>(Coord(8), 1.6f, true);
        // value has not been set
        EXPECT_EQ(3.0f, grid->tree().probeConstLeaf(Coord(8))->getFirstValue());

        // add root child tile

        treeToMerge.addTile<RootChildNode>(Coord(8), 1.7f, true);
        EXPECT_EQ(Index64(1), grid->tree().activeTileCount());

        // tile in node that does not exist

        grid->tree().clear();
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.8f, true);
        EXPECT_EQ(Index64(0), grid->tree().activeTileCount());
    }

    { // const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        EXPECT_EQ(Index(1), grid->tree().leafCount());

        tools::TreeToMerge<FloatTree> treeToMerge{grid->constTree(), DeepCopy(), /*initialize=*/false};
        EXPECT_TRUE(!treeToMerge.hasMask());
        treeToMerge.initializeMask();
        EXPECT_TRUE(treeToMerge.hasMask());
        EXPECT_EQ(&grid->constTree().root(), treeToMerge.rootPtr());

        // probe root child

        const RootChildNode* nodePtr = treeToMerge.probeConstNode<RootChildNode>(Coord(8));
        EXPECT_TRUE(nodePtr);
        EXPECT_EQ(grid->constTree().probeConstNode<RootChildNode>(Coord(8)), nodePtr);

        // probe leaf node

        const LeafNode* leafNode = treeToMerge.probeConstNode<LeafNode>(Coord(8));
        EXPECT_TRUE(leafNode);
        EXPECT_EQ(grid->constTree().probeConstLeaf(Coord(8)), leafNode);
        EXPECT_EQ(Index(1), grid->tree().leafCount());
        EXPECT_EQ(Index(1), grid->tree().root().childCount());

        { // deep copy leaf node
            tools::TreeToMerge<FloatTree> treeToMerge2{grid->constTree(), DeepCopy()};
            std::unique_ptr<LeafNode> leafNodePtr = treeToMerge2.stealOrDeepCopyNode<LeafNode>(Coord(8));
            EXPECT_TRUE(leafNodePtr);
            EXPECT_EQ(Index(1), grid->tree().leafCount()); // leaf has not been stolen
            EXPECT_EQ(leafNodePtr->origin(), Coord(8));
            EXPECT_EQ(Index(1), grid->tree().root().childCount());
        }

        { // deep copy root child
            tools::TreeToMerge<FloatTree> treeToMerge2{grid->constTree(), DeepCopy()};
            grid->tree().touchLeaf(Coord(8));
            std::unique_ptr<RootChildNode> node2Ptr = treeToMerge2.stealOrDeepCopyNode<RootChildNode>(Coord(8));
            EXPECT_TRUE(node2Ptr);
            EXPECT_EQ(Index(1), grid->tree().root().childCount());
        }

        { // add root child tile
            tools::TreeToMerge<FloatTree> treeToMerge2{grid->constTree(), DeepCopy()};
            EXPECT_TRUE(treeToMerge2.probeConstNode<RootChildNode>(Coord(8)));
            treeToMerge2.addTile<RootChildNode>(Coord(8), 1.7f, true);
            EXPECT_TRUE(!treeToMerge2.probeConstNode<RootChildNode>(Coord(8))); // tile has been added to mask
            EXPECT_EQ(Index64(0), grid->tree().activeTileCount());
        }

        // tile in node that does not exist

        grid->tree().clear();
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.8f, true);
        EXPECT_EQ(Index64(0), grid->tree().activeTileCount());
    }

    { // non-const tree shared pointer
        { // shared pointer constructor
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().touchLeaf(Coord(8));
            tools::TreeToMerge<FloatTree> treeToMerge(grid->treePtr(), Steal());

            // verify tree shared ownership

            EXPECT_TRUE(treeToMerge.treeToSteal());
            EXPECT_TRUE(!treeToMerge.treeToDeepCopy());
            EXPECT_TRUE(treeToMerge.rootPtr());
            EXPECT_TRUE(treeToMerge.probeConstNode<FloatTree::LeafNodeType>(Coord(8)));
        }

        // empty tree
        FloatTree tree;
        tools::TreeToMerge<FloatTree> treeToMerge(tree, DeepCopy());
        EXPECT_TRUE(!treeToMerge.treeToSteal());
        EXPECT_TRUE(treeToMerge.treeToDeepCopy());
        EXPECT_TRUE(treeToMerge.rootPtr());
        EXPECT_TRUE(!treeToMerge.probeConstNode<FloatTree::LeafNodeType>(Coord(8)));

        {
            FloatTree::Ptr emptyPtr;
            EXPECT_THROW(treeToMerge.reset(emptyPtr, Steal()), RuntimeError);

            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().touchLeaf(Coord(8));
            EXPECT_EQ(Index(1), grid->tree().leafCount());

            treeToMerge.reset(grid->treePtr(), Steal());
        }

        // verify tree shared ownership

        EXPECT_TRUE(treeToMerge.treeToSteal());
        EXPECT_TRUE(!treeToMerge.treeToDeepCopy());
        EXPECT_TRUE(treeToMerge.rootPtr());
        EXPECT_TRUE(treeToMerge.probeConstNode<FloatTree::LeafNodeType>(Coord(8)));

        // verify tree pointers are updated on reset()

        const FloatTree tree2;
        tools::TreeToMerge<FloatTree> treeToMerge2(tree2, DeepCopy());
        treeToMerge2.initializeMask(); // no-op

        EXPECT_TRUE(!treeToMerge2.treeToSteal());
        EXPECT_TRUE(treeToMerge2.treeToDeepCopy());
        EXPECT_EQ(Index(0), treeToMerge2.treeToDeepCopy()->leafCount());

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        treeToMerge2.reset(grid->treePtr(), Steal());

        EXPECT_TRUE(treeToMerge2.treeToSteal());
        EXPECT_TRUE(!treeToMerge2.treeToDeepCopy());
        EXPECT_EQ(Index(1), treeToMerge2.treeToSteal()->leafCount());
    }
}

TEST_F(TestMerge, testCsgUnion)
{
    using RootChildType = FloatTree::RootNodeType::ChildNodeType;
    using LeafParentType = RootChildType::ChildNodeType;
    using LeafT = FloatTree::LeafNodeType;

    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        { // one non-const tree (steal)
            tools::CsgUnionOp<FloatTree> mergeOp(tree1, Steal());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgUnionOp<FloatTree> mergeOp(tree1, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgUnionOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // vector of tree pointers
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // deque of tree pointers
            std::deque<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, DeepCopy());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // vector of TreesToMerge (to mix const and non-const trees)
            std::vector<tools::TreeToMerge<FloatTree>> trees;
            trees.emplace_back(tree1, Steal());
            trees.emplace_back(tree3, DeepCopy()); // const tree
            trees.emplace_back(tree2, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp(trees);
            EXPECT_EQ(size_t(3), mergeOp.size());
        }
        { // implicit copy constructor
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp2(mergeOp);
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
        { // implicit assignment operator
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp2 = mergeOp;
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // empty merge trees
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        std::vector<FloatTree*> trees;
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());

        EXPECT_EQ(size_t(0), mergeOp.size());

        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(0), root.getTableSize());
    }

    /////////////////////////////////////////////////////////////////////////

    { // test one tile or one child

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getTileCount(root));
            EXPECT_EQ(-grid->background(), grid->cbeginValueAll().getValue());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getTileCount(root));
            EXPECT_EQ(-grid->background(), grid->cbeginValueAll().getValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(1.0, root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(1.0, root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(1.0, root.cbeginChildOn()->getFirstValue());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test two tiles

        { // test outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test inside vs outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test inside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test inside vs outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test inside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test one tile, one child

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(Index(1), getChildCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test two children

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), true));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(false, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(true, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), true));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(false, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(false, root.cbeginChildOn()->isValueOn(0));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test multiple root node elements

        { // merge a child node into a grid with an existing child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root.addTile(Coord(8192, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), grid2->background(), false);
            root2.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(root));
            EXPECT_TRUE(root.cbeginChildOn()->cbeginValueAll());
            EXPECT_EQ(1.0f, root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(2.0f, (++root.cbeginChildOn())->getFirstValue());
        }

        { // merge a child node into a grid with an existing child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), grid->background(), false);
            root.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root2.addTile(Coord(8192, 0, 0), grid2->background(), false);

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(root));
            EXPECT_TRUE(root.cbeginChildOn()->cbeginValueAll());
            EXPECT_EQ(1.0f, root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(2.0f, (++root.cbeginChildOn())->getFirstValue());
        }

        { // merge background tiles and child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root.addTile(Coord(8192, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), -grid2->background(), false);
            root2.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
            EXPECT_EQ(-grid->background(), *(++root.cbeginValueOff()));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test merging internal node children

        { // merge two internal nodes into a grid with an inside tile and an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), -123.0f, false);
            rootChild->addTile(0, grid->background(), false);
            root.addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

            rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
            rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
            rootChild2->addTile(2, -grid->background(), false);
            root2.addChild(rootChild2.release());

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(1), getChildCount(*root.cbeginChildOn()));
            EXPECT_EQ(Index(0), getOutsideTileCount(*root.cbeginChildOn()));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(0));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(1));
            EXPECT_EQ(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
            EXPECT_EQ(-123.0f, root.cbeginChildOn()->cbeginValueAll().getValue());
        }

        { // merge two internal nodes into a grid with an inside tile and an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
            root.addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), -55.0f, false);

            rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
            rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
            rootChild2->addTile(2, -140.0f, false);
            rootChild2->addTile(3, grid2->background(), false);
            root2.addChild(rootChild2.release());

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(*root.cbeginChildOn()));
            EXPECT_EQ(Index(1), getOutsideTileCount(*root.cbeginChildOn()));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(0));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(1));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(2));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(3));
            EXPECT_EQ(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->cbeginValueAll().getItem(2));
            EXPECT_EQ(123.0f, root.cbeginChildOn()->cbeginValueAll().getItem(3));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test merging leaf nodes

        { // merge a leaf node into an empty grid
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), grid->tree().leafCount());
        }

        { // merge a leaf node into a grid with an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -10.0f, false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // merge a leaf node into a grid with an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().touchLeaf(Coord(0, 0, 0));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -10.0f, false);

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // merge a leaf node into a grid with an internal node inside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), grid->background(), false);
            grid->tree().root().addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));

            leaf->setValueOnly(11, grid2->background());
            leaf->setValueOnly(12, -grid2->background());

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), grid->tree().leafCount());
            EXPECT_EQ(Index32(0), grid2->tree().leafCount());

            // test background values are remapped

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(grid->background(), testLeaf->getValue(11));
            EXPECT_EQ(-grid->background(), testLeaf->getValue(12));
        }

        { // merge a leaf node into a grid with a partially constructed leaf node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

            grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
            auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);

            tools::CsgUnionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(-2.3f, testLeaf->getValue(10));
        }

        { // merge three leaf nodes from different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);

            auto* leaf = grid->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf2 = grid2->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf3 = grid3->tree().touchLeaf(Coord(0, 0, 0));

            // active state from the voxel with the minimum value preserved

            leaf->setValueOnly(5, 4.0f);
            leaf2->setValueOnly(5, 2.0f);
            leaf2->setValueOn(5);
            leaf3->setValueOnly(5, 3.0f);

            leaf->setValueOnly(7, 2.0f);
            leaf->setValueOn(7);
            leaf2->setValueOnly(7, 3.0f);
            leaf3->setValueOnly(7, 4.0f);

            leaf->setValueOnly(9, 4.0f);
            leaf->setValueOn(9);
            leaf2->setValueOnly(9, 3.0f);
            leaf3->setValueOnly(9, 2.0f);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(2.0f, testLeaf->getValue(5));
            EXPECT_TRUE(testLeaf->isValueOn(5));
            EXPECT_EQ(2.0f, testLeaf->getValue(7));
            EXPECT_TRUE(testLeaf->isValueOn(7));
            EXPECT_EQ(2.0f, testLeaf->getValue(9));
            EXPECT_TRUE(!testLeaf->isValueOn(9));
        }

        { // merge a leaf node into an empty grid from a const grid
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), 1.0f, false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            EXPECT_EQ(Index32(0), grid->tree().leafCount());
            EXPECT_EQ(Index32(1), grid2->tree().leafCount());

            // merge from a const tree

            std::vector<tools::TreeToMerge<FloatTree>> treesToMerge;
            treesToMerge.emplace_back(grid2->constTree(), DeepCopy());
            tools::CsgUnionOp<FloatTree> mergeOp(treesToMerge);
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), grid->tree().leafCount());
            // leaf has been deep copied not stolen
            EXPECT_EQ(Index32(1), grid2->tree().leafCount());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // merge multiple grids

        { // merge two background root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/grid2->background(), false);
            root2.addTile(Coord(8192, 0, 0), /*background=*/-grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/-grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/grid3->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getTileCount(root));

            EXPECT_EQ(-grid->background(), root.cbeginValueAll().getValue());
            EXPECT_EQ(-grid->background(), (++root.cbeginValueAll()).getValue());
        }

        { // merge two outside root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/-10.0f, false);
            root2.addTile(Coord(8192, 0, 0), /*background=*/grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/-11.0f, false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getTileCount(root));

            EXPECT_EQ(-grid->background(), root.cbeginValueAll().getValue());
            EXPECT_EQ(-grid->background(), (++root.cbeginValueAll()).getValue());
        }

        { // merge two active, outside root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), /*background=*/grid->background(), false);
            root.addTile(Coord(8192, 0, 0), /*background=*/grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/-10.0f, true);
            root2.addTile(Coord(8192, 0, 0), /*background=*/grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/-11.0f, true);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getTileCount(root));

            EXPECT_EQ(-grid->background(), root.cbeginValueAll().getValue());
            EXPECT_EQ(-grid->background(), (++root.cbeginValueAll()).getValue());
        }

        { // merge three root tiles, one of which is a background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), grid->background(), true);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), -grid2->background(), true);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), -grid3->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(1), getTileCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginValueOn().getValue());
        }
    }
}

TEST_F(TestMerge, testCsgIntersection)
{
    using RootChildType = FloatTree::RootNodeType::ChildNodeType;
    using LeafParentType = RootChildType::ChildNodeType;
    using LeafT = FloatTree::LeafNodeType;

    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        { // one non-const tree (steal)
            tools::CsgIntersectionOp<FloatTree> mergeOp(tree1, Steal());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgIntersectionOp<FloatTree> mergeOp(tree1, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgIntersectionOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // vector of tree pointers
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // deque of tree pointers
            std::deque<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // vector of TreesToMerge (to mix const and non-const trees)
            std::vector<tools::TreeToMerge<FloatTree>> trees;
            trees.emplace_back(tree1, Steal());
            trees.emplace_back(tree3, DeepCopy()); // const tree
            trees.emplace_back(tree2, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees);
            EXPECT_EQ(size_t(3), mergeOp.size());
        }
        { // implicit copy constructor
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp2(mergeOp);
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
        { // implicit assignment operator
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp2 = mergeOp;
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // empty merge trees
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        std::vector<FloatTree*> trees;
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());

        EXPECT_EQ(size_t(0), mergeOp.size());

        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(0), root.getTableSize());
    }

    /////////////////////////////////////////////////////////////////////////

    { // test one tile or one child

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test one child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), 1.0, false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test two tiles

        { // test outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside background tiles
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }

        { // test outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside vs outside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test inside background tiles (different background values)
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), -grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_TRUE(hasOnlyInactiveNegativeBackgroundTiles(root));
            EXPECT_EQ(Index(1), getInactiveTileCount(root));
            EXPECT_EQ(-grid->background(), *root.cbeginValueOff());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test one tile, one child

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(-grid->background(), root.cbeginChildOn()->getFirstValue());
        }

        { // test background tiles vs child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), grid2->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test two children

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), true));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(false, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(true, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid->background(), false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid2->background(), true));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(true, root.cbeginChildOn()->isValueOn(0));
        }

        { // test two child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addChild(new RootChildType(Coord(0, 0, 0), grid->background(), true));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addChild(new RootChildType(Coord(0, 0, 0), -grid2->background(), false));

            std::vector<FloatTree*> trees{&grid2->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto& root = grid->tree().root();
            EXPECT_EQ(Index(0), getTileCount(root));
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(true, root.cbeginChildOn()->isValueOn(0));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test multiple root node elements

        { // merge a child node into a grid with an existing child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root.addTile(Coord(8192, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), -grid2->background(), false);
            root2.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(root));
            EXPECT_TRUE(root.cbeginChildOn()->cbeginValueAll());
            EXPECT_EQ(1.0f, root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(2.0f, (++root.cbeginChildOn())->getFirstValue());
        }

        { // merge a child node into a grid with an existing child node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), -grid->background(), false);
            root.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root2.addTile(Coord(8192, 0, 0), -grid2->background(), false);

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(root));
            EXPECT_TRUE(root.cbeginChildOn()->cbeginValueAll());
            EXPECT_EQ(1.0f, root.cbeginChildOn()->getFirstValue());
            EXPECT_EQ(2.0f, (++root.cbeginChildOn())->getFirstValue());
        }

        { // merge background tiles and child nodes
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
            root.addTile(Coord(8192, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), grid2->background(), false);
            root2.addChild(new RootChildType(Coord(8192, 0, 0), 2.0f, false));

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), getTileCount(root));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test merging internal node children

        { // merge two internal nodes into a grid with an inside tile and an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
            rootChild->addTile(0, -grid->background(), false);
            root.addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

            rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
            rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
            rootChild2->addTile(2, -grid->background(), false);
            root2.addChild(rootChild2.release());

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(1), getChildCount(*root.cbeginChildOn()));
            EXPECT_EQ(Index(0), getInsideTileCount(*root.cbeginChildOn()));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(0));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(1));
            EXPECT_EQ(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
            EXPECT_EQ(123.0f, root.cbeginChildOn()->cbeginValueAll().getValue());
        }

        { // merge two internal nodes into a grid with an inside tile and an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), -123.0f, false);
            root.addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

            rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
            rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
            rootChild2->addTile(2, 140.0f, false);
            rootChild2->addTile(3, -grid2->background(), false);
            root2.addChild(rootChild2.release());

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getChildCount(*root.cbeginChildOn()));
            EXPECT_EQ(Index(1), getInsideTileCount(*root.cbeginChildOn()));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(0));
            EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(1));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(2));
            EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(3));
            EXPECT_EQ(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
            EXPECT_EQ(grid->background(), root.cbeginChildOn()->cbeginValueAll().getItem(2));
            EXPECT_EQ(-123.0f, root.cbeginChildOn()->cbeginValueAll().getItem(3));
        }
    }

    /////////////////////////////////////////////////////////////////////////

    { // test merging leaf nodes

        { // merge a leaf node into an empty grid
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(0), grid->tree().leafCount());
        }

        { // merge a leaf node into a grid with a background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(0), grid->tree().leafCount());
        }

        { // merge a leaf node into a grid with an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), 10.0f, false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // merge a leaf node into a grid with an outside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().touchLeaf(Coord(0, 0, 0));
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            grid2->tree().root().addTile(Coord(0, 0, 0), 10.0f, false);

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

        { // merge a leaf node into a grid with an internal node inside tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), -grid->background(), false);
            grid->tree().root().addChild(rootChild.release());
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));

            leaf->setValueOnly(11, grid2->background());
            leaf->setValueOnly(12, -grid2->background());

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), grid->tree().leafCount());
            EXPECT_EQ(Index32(0), grid2->tree().leafCount());

            // test background values are remapped

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(grid->background(), testLeaf->getValue(11));
            EXPECT_EQ(-grid->background(), testLeaf->getValue(12));
        }

        { // merge a leaf node into a grid with a partially constructed leaf node
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

            grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
            auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, 6.4f);

            tools::CsgIntersectionOp<FloatTree> mergeOp{grid2->tree(), Steal()};
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(6.4f, testLeaf->getValue(10));
        }

        { // merge three leaf nodes from different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);

            auto* leaf = grid->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf2 = grid2->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf3 = grid3->tree().touchLeaf(Coord(0, 0, 0));

            // active state from the voxel with the maximum value preserved

            leaf->setValueOnly(5, 4.0f);
            leaf2->setValueOnly(5, 2.0f);
            leaf2->setValueOn(5);
            leaf3->setValueOnly(5, 3.0f);

            leaf->setValueOnly(7, 2.0f);
            leaf->setValueOn(7);
            leaf2->setValueOnly(7, 3.0f);
            leaf3->setValueOnly(7, 4.0f);

            leaf->setValueOnly(9, 4.0f);
            leaf->setValueOn(9);
            leaf2->setValueOnly(9, 3.0f);
            leaf3->setValueOnly(9, 2.0f);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(4.0f, testLeaf->getValue(5));
            EXPECT_TRUE(!testLeaf->isValueOn(5));
            EXPECT_EQ(4.0f, testLeaf->getValue(7));
            EXPECT_TRUE(!testLeaf->isValueOn(7));
            EXPECT_EQ(4.0f, testLeaf->getValue(9));
            EXPECT_TRUE(testLeaf->isValueOn(9));
        }

        { // merge a leaf node into an empty grid from a const grid
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().root().addTile(Coord(0, 0, 0), -1.0f, false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            grid2->tree().touchLeaf(Coord(0, 0, 0));

            EXPECT_EQ(Index32(0), grid->tree().leafCount());
            EXPECT_EQ(Index32(1), grid2->tree().leafCount());

            // merge from a const tree

            std::vector<tools::TreeToMerge<FloatTree>> treesToMerge;
            treesToMerge.emplace_back(grid2->constTree(), DeepCopy());
            tools::CsgIntersectionOp<FloatTree> mergeOp(treesToMerge);
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), grid->tree().leafCount());
            // leaf has been deep copied not stolen
            EXPECT_EQ(Index32(1), grid2->tree().leafCount());
        }

        { // merge three leaf nodes from four grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
            FloatGrid::Ptr grid4 = createLevelSet<FloatGrid>();

            auto* leaf = grid->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf2 = grid2->tree().touchLeaf(Coord(0, 0, 0));
            auto* leaf3 = grid3->tree().touchLeaf(Coord(0, 0, 0));

            // active state from the voxel with the maximum value preserved

            leaf->setValueOnly(5, 4.0f);
            leaf2->setValueOnly(5, 2.0f);
            leaf2->setValueOn(5);
            leaf3->setValueOnly(5, 3.0f);

            leaf->setValueOnly(7, 2.0f);
            leaf->setValueOn(7);
            leaf2->setValueOnly(7, 3.0f);
            leaf3->setValueOnly(7, 4.0f);

            leaf->setValueOnly(9, 4.0f);
            leaf->setValueOn(9);
            leaf2->setValueOnly(9, 3.0f);
            leaf3->setValueOnly(9, 2.0f);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree(), &grid4->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), grid->tree().root().getTableSize());
        }

    }

    /////////////////////////////////////////////////////////////////////////

    { // merge multiple grids

        { // merge two background root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), /*background=*/-grid->background(), false);
            root.addTile(Coord(8192, 0, 0), /*background=*/-grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/grid2->background(), false);
            root2.addTile(Coord(8192, 0, 0), /*background=*/-grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/-grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/grid3->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), root.getTableSize());
        }

        { // merge two outside root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), /*background=*/-grid->background(), false);
            root.addTile(Coord(8192, 0, 0), /*background=*/-grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/10.0f, false);
            root2.addTile(Coord(8192, 0, 0), /*background=*/-grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/-grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/11.0f, false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), root.getTableSize());
        }

        { // merge two active, outside root tiles from two different grids
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), /*background=*/-grid->background(), false);
            root.addTile(Coord(8192, 0, 0), /*background=*/-grid->background(), false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/5);
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), /*background=*/10.0f, true);
            root2.addTile(Coord(8192, 0, 0), /*background=*/-grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*narrowBandWidth=*/7);
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), /*background=*/-grid3->background(), false);
            root3.addTile(Coord(8192, 0, 0), /*background=*/11.0f, true);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(2), getTileCount(root));

            EXPECT_EQ(grid->background(), root.cbeginValueAll().getValue());
            EXPECT_EQ(grid->background(), (++root.cbeginValueAll()).getValue());
        }

        { // merge three root tiles, one of which is a background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), -grid->background(), true);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), grid2->background(), true);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), grid3->background(), false);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(1), root.getTableSize());
            EXPECT_EQ(Index(1), getTileCount(root));
        }

        { // merge three root tiles, one of which is a background tile
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            auto& root = grid->tree().root();
            root.addTile(Coord(0, 0, 0), -grid->background(), true);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            auto& root2 = grid2->tree().root();
            root2.addTile(Coord(0, 0, 0), grid2->background(), false);
            FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
            auto& root3 = grid3->tree().root();
            root3.addTile(Coord(0, 0, 0), grid3->background(), true);

            std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index(0), root.getTableSize());
        }

    }
}

TEST_F(TestMerge, testCsgDifference)
{
    using RootChildType = FloatTree::RootNodeType::ChildNodeType;
    using LeafParentType = RootChildType::ChildNodeType;
    using LeafT = FloatTree::LeafNodeType;

    { // construction
        FloatTree tree1;
        const FloatTree tree2;

        { // one non-const tree (steal)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree1, Steal());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree1, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one non-const tree wrapped in TreeToMerge
            tools::TreeToMerge<FloatTree> tree3(tree1, Steal());
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree3);
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one const tree wrapped in TreeToMerge
            tools::TreeToMerge<FloatTree> tree4(tree2, DeepCopy());
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree4);
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // implicit copy constructor
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
            tools::CsgDifferenceOp<FloatTree> mergeOp2(mergeOp);
            EXPECT_EQ(size_t(1), mergeOp2.size());
        }
        { // implicit assignment operator
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
            tools::CsgDifferenceOp<FloatTree> mergeOp2 = mergeOp;
            EXPECT_EQ(size_t(1), mergeOp2.size());
        }
    }

    { // merge two different outside root tiles from one grid into an empty grid (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), false);
        root2.addTile(Coord(8192, 0, 0), grid->background(), true);

        EXPECT_EQ(Index(2), root2.getTableSize());
        EXPECT_EQ(Index(2), getTileCount(root2));
        EXPECT_EQ(Index(1), getActiveTileCount(root2));
        EXPECT_EQ(Index(1), getInactiveTileCount(root2));

        // test container constructor here
        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(0), root.getTableSize());
    }

    { // merge an outside root tile to a grid which already has this tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(0), root.getTableSize());
    }

    { // merge an outside root tile to a grid which already has this tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), false);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        EXPECT_EQ(Index(1), getActiveTileCount(root));
        EXPECT_EQ(Index(0), getInactiveTileCount(root));
        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));
    }

    { // merge an outside root tile to a grid which has an inside tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 123.0f, true);

        EXPECT_EQ(Index(1), getInsideTileCount(root));
        EXPECT_EQ(Index(0), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        EXPECT_EQ(Index(0), getActiveTileCount(root));
        EXPECT_EQ(Index(1), getInactiveTileCount(root));
        EXPECT_EQ(Index(1), getInsideTileCount(root));
        EXPECT_EQ(Index(0), getOutsideTileCount(root));
    }

    { // merge an outside root tile to a grid which has a child (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 123.0f, true);

        EXPECT_EQ(Index(1), getChildCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(0), getTileCount(root));
        EXPECT_EQ(Index(1), getChildCount(root));
    }

    { // merge a child to a grid which has an outside root tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 123.0f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));

        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        EXPECT_EQ(Index(0), getChildCount(root));
        EXPECT_EQ(Index(1), getActiveTileCount(root));
        EXPECT_EQ(Index(0), getInactiveTileCount(root));
        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));
    }

    { // merge an inside root tile to a grid which has an outside tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -123.0f, true);

        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        EXPECT_EQ(Index(1), getActiveTileCount(root));
        EXPECT_EQ(Index(0), getInactiveTileCount(root));
        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));
    }

    { // merge two grids with outside tiles, active state should be carried across
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 0.1f, false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 0.2f, true);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        // outside tile should now be inactive
        EXPECT_EQ(Index(0), getActiveTileCount(root));
        EXPECT_EQ(Index(1), getInactiveTileCount(root));
        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));

        EXPECT_EQ(0.1f, root.cbeginValueAll().getValue());
    }

    { // merge two grids with outside tiles, active state should be carried across
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -0.1f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -0.2f, false);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(0), root.getTableSize());
    }

    { // merge an inside root tile to a grid which has a child, inside tile has precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -123.0f, true);

        EXPECT_EQ(Index(1), getChildCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(1), getTileCount(root));
        EXPECT_EQ(Index(0), getChildCount(root));
        EXPECT_EQ(Index(1), getActiveTileCount(root));
        EXPECT_EQ(Index(0), getInactiveTileCount(root));
        EXPECT_EQ(Index(0), getInsideTileCount(root));
        EXPECT_EQ(Index(1), getOutsideTileCount(root));

        EXPECT_EQ(grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge a child to a grid which has an inside root tile, child should be stolen
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -123.0f, true);
        // use a different background value
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>(/*voxelSize=*/1.0, /*halfWidth=*/5);
        auto& root2 = grid2->tree().root();
        auto childPtr = std::make_unique<RootChildType>(Coord(0, 0, 0), 5.0f, false);
        childPtr->addTile(Index(1), 1.3f, true);
        root2.addChild(childPtr.release());

        EXPECT_EQ(Index(1), getInsideTileCount(root));
        EXPECT_EQ(Index(0), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), root.getTableSize());
        EXPECT_EQ(Index(0), getTileCount(root));
        EXPECT_EQ(Index(1), getChildCount(root));
        EXPECT_EQ(Index(0), getChildCount(root2));

        EXPECT_TRUE(!root.cbeginChildOn()->isValueOn(Index(0)));
        EXPECT_TRUE(root.cbeginChildOn()->isValueOn(Index(1)));

        auto iter = root.cbeginChildOn()->cbeginValueAll();
        EXPECT_EQ(-3.0f, iter.getValue());
        ++iter;
        EXPECT_EQ(-1.3f, iter.getValue());
    }

    { // merge two child nodes into a grid with two inside tiles
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -2.0f, false);
        root.addTile(Coord(8192, 0, 0), -4.0f, false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), -123.0f, true));

        EXPECT_EQ(Index(2), root2.getTableSize());
        EXPECT_EQ(Index(0), getTileCount(root2));
        EXPECT_EQ(Index(2), getChildCount(root2));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(2), root.getTableSize());
        EXPECT_EQ(Index(0), getTileCount(root));
        EXPECT_EQ(Index(2), getChildCount(root));
    }

    { // merge an inside tile and an outside tile into a grid with two child nodes
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 15.0f, true); // should not replace child
        root2.addTile(Coord(8192, 0, 0), -25.0f, true); // should replace child

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), getChildCount(root));
        EXPECT_EQ(Index(1), getTileCount(root));
        EXPECT_EQ(123.0f, root.cbeginChildOn()->getFirstValue());
        EXPECT_TRUE(root.cbeginChildAll().isChildNode());
        EXPECT_TRUE(!(++root.cbeginChildAll()).isChildNode());
        EXPECT_EQ(grid->background(), root.cbeginValueOn().getValue());
    }

    { // merge an inside tile and an outside tile into a grid with two child nodes
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 15.0f, false); // should not replace child
        root2.addTile(Coord(8192, 0, 0), -25.0f, false); // should replace child

        EXPECT_EQ(Index(2), getChildCount(root));
        EXPECT_EQ(Index(2), getTileCount(root2));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), getChildCount(root));
        EXPECT_EQ(Index(0), getTileCount(root));
        EXPECT_EQ(123.0f, root.cbeginChildOn()->getFirstValue());
    }

    { // merge two internal nodes into a grid with an inside tile and an outside tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
        rootChild->addTile(0, -14.0f, false);
        rootChild->addTile(1, 15.0f, false);
        rootChild->addTile(2, -13.0f, false);
        root.addChild(rootChild.release());

        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);
        rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
        rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
        rootChild2->addTile(2, -17.0f, true);
        rootChild2->addTile(9, 19.0f, true);
        root2.addChild(rootChild2.release());

        EXPECT_EQ(Index(2), getInsideTileCount(*root.cbeginChildOn()));
        EXPECT_EQ(Index(0), getActiveTileCount(*root.cbeginChildOn()));

        EXPECT_EQ(Index(2), getChildCount(*root2.cbeginChildOn()));
        EXPECT_EQ(Index(1), getInsideTileCount(*root2.cbeginChildOn()));
        EXPECT_EQ(Index(2), getActiveTileCount(*root2.cbeginChildOn()));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(1), getChildCount(*root.cbeginChildOn()));
        EXPECT_EQ(Index(0), getInsideTileCount(*root.cbeginChildOn()));
        EXPECT_EQ(Index(1), getActiveTileCount(*root.cbeginChildOn()));
        EXPECT_TRUE(root.cbeginChildOn()->isChildMaskOn(0));
        EXPECT_TRUE(!root.cbeginChildOn()->isChildMaskOn(1));
        EXPECT_EQ(-29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
        auto iter = root.cbeginChildOn()->cbeginValueAll();
        EXPECT_EQ(15.0f, iter.getValue());
        ++iter;
        EXPECT_EQ(3.0f, iter.getValue());

        EXPECT_EQ(Index(1), getChildCount(*root2.cbeginChildOn()));
    }

    { // merge a leaf node into a grid with an inside tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().addTile(1, Coord(0, 0, 0), -1.3f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        EXPECT_EQ(Index32(0), grid->tree().leafCount());
        EXPECT_EQ(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index32(1), grid->tree().leafCount());
        EXPECT_EQ(Index32(0), grid2->tree().leafCount());
    }

    { // merge two leaf nodes into a grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(0, 0, 0));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        EXPECT_EQ(Index32(1), grid->tree().leafCount());
        EXPECT_EQ(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* leaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        EXPECT_TRUE(leaf);
    }

    { // merge a leaf node into a grid with a partially constructed leaf node
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
        auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, 6.4f);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        EXPECT_EQ(3.0f, testLeaf->getValue(10));
    }

    { // merge two leaf nodes from different grids
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        auto* leaf = grid->tree().touchLeaf(Coord(0, 0, 0));
        auto* leaf2 = grid2->tree().touchLeaf(Coord(0, 0, 0));

        // active state from the voxel with the maximum value preserved

        leaf->setValueOnly(5, 98.0f);
        leaf2->setValueOnly(5, 2.0f);
        leaf2->setValueOn(5);

        leaf->setValueOnly(7, 2.0f);
        leaf->setValueOn(7);
        leaf2->setValueOnly(7, 100.0f);

        leaf->setValueOnly(9, 4.0f);
        leaf->setValueOn(9);
        leaf2->setValueOnly(9, -100.0f);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        EXPECT_EQ(98.0f, testLeaf->getValue(5));
        EXPECT_TRUE(!testLeaf->isValueOn(5));
        EXPECT_EQ(2.0f, testLeaf->getValue(7));
        EXPECT_TRUE(testLeaf->isValueOn(7));
        EXPECT_EQ(100.0f, testLeaf->getValue(9));
        EXPECT_TRUE(!testLeaf->isValueOn(9));
    }

    { // merge a leaf node into a grid with an inside tile from a const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().addTile(1, Coord(0, 0, 0), -1.3f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        EXPECT_EQ(Index32(0), grid->tree().leafCount());
        EXPECT_EQ(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->constTree(), DeepCopy());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index32(1), grid->tree().leafCount());
        EXPECT_EQ(Index32(1), grid2->tree().leafCount());
    }
}


TEST_F(TestMerge, testSum)
{
    using RootChildType = FloatTree::RootNodeType::ChildNodeType;
    using LeafT = FloatTree::LeafNodeType;

    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        { // one non-const tree (steal)
            tools::SumMergeOp<FloatTree> mergeOp(tree1, Steal());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::SumMergeOp<FloatTree> mergeOp(tree1, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::SumMergeOp<FloatTree> mergeOp(tree2, DeepCopy());
            EXPECT_EQ(size_t(1), mergeOp.size());
        }
        { // vector of tree pointers
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::SumMergeOp<FloatTree> mergeOp(trees, Steal());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // deque of tree pointers
            std::deque<FloatTree*> trees{&tree1, &tree2};
            tools::SumMergeOp<FloatTree> mergeOp(trees, DeepCopy());
            EXPECT_EQ(size_t(2), mergeOp.size());
        }
        { // vector of TreesToMerge (to mix const and non-const trees)
            std::vector<tools::TreeToMerge<FloatTree>> trees;
            trees.emplace_back(tree1, Steal());
            trees.emplace_back(tree3, DeepCopy()); // const tree
            trees.emplace_back(tree2, Steal());
            tools::SumMergeOp<FloatTree> mergeOp(trees);
            EXPECT_EQ(size_t(3), mergeOp.size());
        }
        { // implicit copy constructor
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::SumMergeOp<FloatTree> mergeOp(trees, Steal());
            tools::SumMergeOp<FloatTree> mergeOp2(mergeOp);
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
        { // implicit assignment operator
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::SumMergeOp<FloatTree> mergeOp(trees, Steal());
            tools::SumMergeOp<FloatTree> mergeOp2 = mergeOp;
            EXPECT_EQ(size_t(2), mergeOp2.size());
        }
    }

    { // merge two different background root tiles from one tree into an empty tree
        FloatTree tree, tree2;
        tree2.root().addTile(Coord(0, 0, 0), tree2.background(), false);
        tree2.root().addTile(Coord(8192, 0, 0), tree2.background(), true);

        const auto& root2 = tree2.root();
        EXPECT_EQ(Index(2), root2.getTableSize());
        EXPECT_EQ(Index(2), getTileCount(root2));
        EXPECT_EQ(Index(1), getActiveTileCount(root2));
        EXPECT_EQ(Index(1), getInactiveTileCount(root2));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        // background tiles are not erased
        EXPECT_EQ(Index(2), tree.root().getTableSize());
    }

    { // merge two different root tiles from one tree into an empty tree
        FloatTree tree, tree2;
        tree2.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addTile(Coord(8192, 0, 0), 2.2f, true);
        EXPECT_EQ(Index(2), getTileCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index(2), tree.root().getTableSize());
        EXPECT_EQ(Index(1), getActiveTileCount(tree.root()));
        EXPECT_EQ(Index(1), getInactiveTileCount(tree.root()));
    }

    { // merge two different root tiles from one tree into a tree with one root tile
        FloatTree tree(100.0f), tree2(200.0f);
        tree.root().addTile(Coord(-8192, 0, 0), -3.3f, true);
        tree2.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addTile(Coord(8192, 0, 0), 2.2f, true);

        EXPECT_EQ(Index(1), getTileCount(tree.root()));
        EXPECT_EQ(Index(2), getTileCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(Index(3), root.getTableSize());
        EXPECT_EQ(Index(2), getActiveTileCount(root));
        EXPECT_EQ(Index(1), getInactiveTileCount(root));
        auto iter = root.cbeginValueAll();
        EXPECT_EQ(200.0f-3.3f, *iter);
        EXPECT_TRUE(iter.isValueOn());
        ++iter;
        EXPECT_EQ(100.0f+1.1f, *iter);
        EXPECT_TRUE(iter.isValueOff());
        ++iter;
        EXPECT_EQ(100.0f+2.2f, *iter);
        EXPECT_TRUE(iter.isValueOn());
    }

    { // merge root tiles with the same active state
        FloatTree tree(100.0f), tree2(200.0f);
        tree.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addTile(Coord(0, 0, 0), 2.2f, false);
        tree.root().addTile(Coord(8192, 0, 0), 1.1f, true);
        tree2.root().addTile(Coord(8192, 0, 0), 2.2f, true);

        EXPECT_EQ(Index(2), getTileCount(tree.root()));
        EXPECT_EQ(Index(2), getTileCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(root.background(), 300.0f);
        EXPECT_EQ(Index(1), getActiveTileCount(root));
        auto iter = root.cbeginValueAll();
        EXPECT_EQ(1.1f+2.2f, *iter);
        EXPECT_TRUE(iter.isValueOff());
        ++iter;
        EXPECT_EQ(1.1f+2.2f, *iter);
        EXPECT_TRUE(iter.isValueOn());
    }

    { // merge root tiles with different active state
        FloatTree tree(100.0f), tree2(200.0f);
        tree.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addTile(Coord(0, 0, 0), 2.2f, true);
        tree.root().addTile(Coord(8192, 0, 0), 1.1f, true);
        tree2.root().addTile(Coord(8192, 0, 0), 2.2f, false);

        EXPECT_EQ(Index(2), getTileCount(tree.root()));
        EXPECT_EQ(Index(2), getTileCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(Index(2), getActiveTileCount(root));
        auto iter = root.cbeginValueAll();
        EXPECT_EQ(1.1f+2.2f, *iter);
        EXPECT_TRUE(iter.isValueOn());
        ++iter;
        EXPECT_EQ(1.1f+2.2f, *iter);
        EXPECT_TRUE(iter.isValueOn());
    }

    { // merge root tiles from three trees
        FloatTree tree(100.0f), tree2(200.0f), tree3(300.0f);
        tree.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addTile(Coord(0, 0, 0), 2.2f, false);
        tree3.root().addTile(Coord(0, 0, 0), 3.3f, false);
        tree2.root().addTile(Coord(8192, 0, 0), 2.2f, false);
        tree3.root().addTile(Coord(8192, 0, 0), 3.3f, true);
        tree3.root().addTile(Coord(-8192, 0, 0), -9.9f, false);

        EXPECT_EQ(Index(1), getTileCount(tree.root()));
        EXPECT_EQ(Index(2), getTileCount(tree2.root()));
        EXPECT_EQ(Index(3), getTileCount(tree3.root()));

        std::vector<FloatTree*> trees{&tree2, &tree3};
        tools::SumMergeOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        auto iter = root.cbeginValueAll();
        EXPECT_EQ(100.0f+200.0f-9.9f, *iter);
        EXPECT_TRUE(iter.isValueOff());
        ++iter;
        EXPECT_EQ(1.1f+2.2f+3.3f, *iter);
        EXPECT_TRUE(iter.isValueOff());
        ++iter;
        EXPECT_EQ(100.0f+2.2f+3.3f, *iter);
        EXPECT_TRUE(iter.isValueOn());
    }

    { // merge root tiles and root children
        FloatTree tree, tree2;
        tree.root().addTile(Coord(0, 0, 0), 1.1f, false);
        tree2.root().addChild(new RootChildType(Coord(0, 0, 0), 2.2f, false));
        tree.root().addTile(Coord(8192, 0, 0), 1.1f, false);
        tree2.root().addChild(new RootChildType(Coord(8192, 0, 0), 2.2f, true));
        tree.root().addTile(Coord(16384, 0, 0), 1.1f, true);
        tree2.root().addChild(new RootChildType(Coord(16384, 0, 0), 2.2f, false));
        tree.root().addChild(new RootChildType(Coord(24576, 0, 0), 1.1f, false));
        tree2.root().addTile(Coord(24576, 0, 0), 2.2f, false);
        tree.root().addChild(new RootChildType(Coord(32768, 0, 0), 1.1f, true));
        tree2.root().addTile(Coord(32768, 0, 0), 2.2f, false);
        tree.root().addChild(new RootChildType(Coord(40960, 0, 0), 1.1f, false));
        tree2.root().addTile(Coord(40960, 0, 0), 2.2f, true);

        EXPECT_EQ(Index(3), getTileCount(tree.root()));
        EXPECT_EQ(Index(3), getTileCount(tree2.root()));
        EXPECT_EQ(Index(3), getChildCount(tree.root()));
        EXPECT_EQ(Index(3), getChildCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(Index(6), getChildCount(root));
        EXPECT_EQ(Index(0), getTileCount(root));
        auto iter = root.cbeginChildOn();
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_FALSE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_FALSE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
    }

    { // merge root children
        FloatTree tree, tree2;
        tree.root().addChild(new RootChildType(Coord(0, 0, 0), 1.1f, false));
        tree2.root().addChild(new RootChildType(Coord(0, 0, 0), 2.2f, false));
        tree.root().addChild(new RootChildType(Coord(8192, 0, 0), 1.1f, false));
        tree2.root().addChild(new RootChildType(Coord(8192, 0, 0), 2.2f, true));
        tree.root().addChild(new RootChildType(Coord(16384, 0, 0), 1.1f, true));
        tree2.root().addChild(new RootChildType(Coord(16384, 0, 0), 2.2f, false));

        EXPECT_EQ(Index(0), getTileCount(tree.root()));
        EXPECT_EQ(Index(0), getTileCount(tree2.root()));
        EXPECT_EQ(Index(3), getChildCount(tree.root()));
        EXPECT_EQ(Index(3), getChildCount(tree2.root()));

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(Index(3), getChildCount(root));
        EXPECT_EQ(Index(0), getTileCount(root));
        auto iter = root.cbeginChildOn();
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_FALSE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
        ++iter;
        EXPECT_EQ(1.1f+2.2f, iter->getFirstValue());
        EXPECT_TRUE(iter->isValueOn(0));
    }

    { // merge root children tiles
        FloatTree tree, tree2;
        auto* child = new RootChildType(Coord(0, 0, 0), 0.0f, false);
        child->addTile(0, 1.1f, false);
        child->addTile(1, 2.2f, true);
        child->addTile(2, 3.3f, false);
        tree.root().addChild(child);
        child = new RootChildType(Coord(0, 0, 0), 2.2f, false);
        child->addTile(0, 4.4f, false);
        child->addTile(1, 5.5f, false);
        child->addTile(2, 6.6f, true);
        tree2.root().addChild(child);

        tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        const auto& root = tree.root();
        EXPECT_EQ(Index(1), getChildCount(root));
        EXPECT_EQ(Index(0), getTileCount(root));
        auto iter = root.cbeginChildOn()->cbeginValueAll();
        EXPECT_EQ(1.1f+4.4f, *iter);
        EXPECT_FALSE(iter.isValueOn());
        ++iter;
        EXPECT_EQ(2.2f+5.5f, *iter);
        EXPECT_TRUE(iter.isValueOn());
        ++iter;
        EXPECT_EQ(3.3f+6.6f, *iter);
        EXPECT_TRUE(iter.isValueOn());
    }

    /////////////////////////////////////////////////////////////////////////

    { // test merging leaf nodes

        { // merge a leaf node into an empty tree
            FloatTree tree, tree2;
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(tree.cbeginLeaf()->getFirstValue(), 0.0f);
        }

        { // merge a leaf node into a tree with a tile
            FloatTree tree, tree2;
            tree.root().addTile(Coord(0, 0, 0), 10.0f, false);
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 10.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge a leaf node into a tree with a tile (deep-copy)
            FloatTree tree(100.0f), tree2(200.0f);
            tree.root().addTile(Coord(0, 0, 0), 10.0f, false);
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, DeepCopy());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 210.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
            // test tile value
            EXPECT_EQ(tree.getValue(Coord(0, 0, 8)), 210.0f);
        }

        { // merge a leaf node into a tree with a tile and non-zero background values (deep-copy)
            FloatTree tree(100.0f), tree2(200.0f);
            tree.root().addTile(Coord(0, 0, 0), 10.0f, false);
            // add internal node 1 tile
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, DeepCopy());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 210.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge a leaf node and a tile into a tree with a tile (deep-copy)
            FloatTree tree, tree2, tree3;
            tree.root().addTile(Coord(0, 0, 0), 10.0f, false);
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);
            tree3.root().addTile(Coord(0, 0, 0), 5.0f, false);

            std::vector<FloatTree*> trees{&tree2, &tree3};
            tools::SumMergeOp<FloatTree> mergeOp(trees, DeepCopy());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 15.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 15.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 15.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge a tile into a tree with a leaf node
            FloatTree tree, tree2;
            auto* leaf = tree.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);
            tree2.root().addTile(Coord(0, 0, 0), 10.0f, false);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, DeepCopy());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 10.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge a root child tile into a tree with a leaf node
            FloatTree tree, tree2;
            auto* leaf = tree.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);
            tree2.root().addChild(new RootChildType(Coord(0, 0, 0), 10.0f, true));

            tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 10.0f);
            EXPECT_TRUE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_TRUE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge an empty tree with non-zero background value into a tree with a leaf node
            FloatTree tree, tree2;
            auto* leaf = tree.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);
            leaf->setValueOn(11, 1.5f);
            tree2.root().setBackground(10.0f, /*updateChildNodes=*/false);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            EXPECT_EQ(Index(0), getTileCount(tree.root()));
            auto iter = tree.cbeginLeaf();
            EXPECT_EQ(iter->getValue(0), 10.0f);
            EXPECT_FALSE(iter->isValueOn(0));
            EXPECT_EQ(iter->getValue(10), 10.0f-2.3f);
            EXPECT_FALSE(iter->isValueOn(10));
            EXPECT_EQ(iter->getValue(11), 10.0f+1.5f);
            EXPECT_TRUE(iter->isValueOn(11));
        }

        { // merge a leaf node into a grid with a partially constructed leaf node
            FloatTree tree, tree2;

            tree.addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
            auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
            leaf->setValueOnly(10, -2.3f);

            tools::SumMergeOp<FloatTree> mergeOp(tree2, Steal());
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            const auto* testLeaf = tree.probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(-2.3f, testLeaf->getValue(10));
        }

        { // merge three leaf nodes from different grids
            DoubleTree tree, tree2, tree3;

            auto* leaf = tree.touchLeaf(Coord(0, 0, 0));
            auto* leaf2 = tree2.touchLeaf(Coord(0, 0, 0));
            auto* leaf3 = tree3.touchLeaf(Coord(0, 0, 0));

            // active state from the voxel with the minimum value preserved

            leaf->setValueOnly(5, 0.7);
            leaf2->setValueOnly(5, 0.2);
            leaf2->setValueOn(5);
            leaf3->setValueOnly(5, 0.1);

            leaf->setValueOnly(7, 0.2);
            leaf->setValueOn(7);
            leaf2->setValueOnly(7, 0.1);
            leaf3->setValueOnly(7, 0.7);

            leaf->setValueOnly(9, 0.7);
            leaf2->setValueOnly(9, 0.1);
            leaf3->setValueOnly(9, 0.2);

            std::vector<DoubleTree*> trees{&tree2, &tree3};
            tools::SumMergeOp<DoubleTree> mergeOp(trees, Steal());
            tree::DynamicNodeManager<DoubleTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            // non-associativity of floating-point addition
            EXPECT_NE(0.7 + 0.2 + 0.1, 0.7 + 0.1 + 0.2);

            // order of additions must be preserved

            const auto* testLeaf = tree.probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(0.7 + 0.2 + 0.1, testLeaf->getValue(5));
            EXPECT_TRUE(testLeaf->isValueOn(5));
            EXPECT_EQ(0.2 + 0.1 + 0.7, testLeaf->getValue(7));
            EXPECT_TRUE(testLeaf->isValueOn(7));
            EXPECT_EQ(0.7 + 0.1 + 0.2, testLeaf->getValue(9));
            EXPECT_FALSE(testLeaf->isValueOn(9));
        }

        { // merge three leaf nodes from different grids
            DoubleTree tree, tree2, tree3;

            auto* leaf = tree.touchLeaf(Coord(0, 0, 0));
            auto* leaf2 = tree2.touchLeaf(Coord(0, 0, 0));
            auto* leaf3 = tree3.touchLeaf(Coord(0, 0, 0));

            // active state from the voxel with the minimum value preserved

            leaf->setValueOnly(5, 0.7);
            leaf2->setValueOnly(5, 0.2);
            leaf2->setValueOn(5);
            leaf3->setValueOnly(5, 0.1);

            leaf->setValueOnly(7, 0.2);
            leaf->setValueOn(7);
            leaf2->setValueOnly(7, 0.1);
            leaf3->setValueOnly(7, 0.7);

            leaf->setValueOnly(9, 0.7);
            leaf2->setValueOnly(9, 0.1);
            leaf3->setValueOnly(9, 0.2);

            std::vector<DoubleTree*> trees{&tree2, &tree3};
            tools::SumMergeOp<DoubleTree> mergeOp(trees, DeepCopy());
            tree::DynamicNodeManager<DoubleTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            // non-associativity of floating-point addition
            EXPECT_NE(0.7 + 0.2 + 0.1, 0.7 + 0.1 + 0.2);

            // order of additions must be preserved

            const auto* testLeaf = tree.probeConstLeaf(Coord(0, 0, 0));
            EXPECT_EQ(0.7 + 0.2 + 0.1, testLeaf->getValue(5));
            EXPECT_TRUE(testLeaf->isValueOn(5));
            EXPECT_EQ(0.2 + 0.1 + 0.7, testLeaf->getValue(7));
            EXPECT_TRUE(testLeaf->isValueOn(7));
            EXPECT_EQ(0.7 + 0.1 + 0.2, testLeaf->getValue(9));
            EXPECT_FALSE(testLeaf->isValueOn(9));
        }

        { // merge a leaf node into an empty grid from a const grid
            FloatTree tree, tree2;
            tree.root().addTile(Coord(0, 0, 0), 1.0f, false);
            FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
            tree2.touchLeaf(Coord(0, 0, 0));

            EXPECT_EQ(Index32(0), tree.leafCount());
            EXPECT_EQ(Index32(1), tree2.leafCount());

            // merge from a const tree

            const FloatTree& constTree2 = tree2;

            std::vector<tools::TreeToMerge<FloatTree>> treesToMerge;
            treesToMerge.emplace_back(constTree2, DeepCopy());
            tools::SumMergeOp<FloatTree> mergeOp(treesToMerge);
            tree::DynamicNodeManager<FloatTree, 3> nodeManager(tree);
            nodeManager.foreachTopDown(mergeOp);

            EXPECT_EQ(Index32(1), tree.leafCount());
            // leaf has been deep copied not stolen
            EXPECT_EQ(Index32(1), tree2.leafCount());
        }
    }

    { // test a Vec3STree
        Vec3STree tree, tree2;
        tree.root().addTile(Coord(0, 0, 0), Vec3s(1.0f, 2.0f, 3.0f), false);
        auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, Vec3s(0.1f, 0.2f, 0.3f));
        leaf->setValueOn(11, Vec3s(0.4f, 0.5f, 0.6f));

        tools::SumMergeOp<Vec3STree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<Vec3STree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index32(1), tree.leafCount());
        EXPECT_EQ(Index(0), getTileCount(tree.root()));
        auto iter = tree.cbeginLeaf();
        EXPECT_EQ(iter->getValue(0), Vec3s(1.0f, 2.0f, 3.0f));
        EXPECT_FALSE(iter->isValueOn(0));
        EXPECT_EQ(iter->getValue(10), Vec3s(1.0f+0.1f, 2.0f+0.2f, 3.0f+0.3f));
        EXPECT_FALSE(iter->isValueOn(10));
        EXPECT_EQ(iter->getValue(11), Vec3s(1.0f+0.4f, 2.0f+0.5f, 3.0f+0.6f));
        EXPECT_TRUE(iter->isValueOn(11));
    }

    { // test a MaskTree
        MaskTree tree, tree2;
        tree.root().addTile(Coord(0, 0, 0), false, false);
        auto* leaf = tree2.touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, true);

        tools::SumMergeOp<MaskTree> mergeOp(tree2, Steal());
        tree::DynamicNodeManager<MaskTree, 3> nodeManager(tree);
        nodeManager.foreachTopDown(mergeOp);

        EXPECT_EQ(Index32(1), tree.leafCount());
        EXPECT_EQ(Index(0), getTileCount(tree.root()));
        auto iter = tree.cbeginLeaf();
        EXPECT_FALSE(iter->isValueOn(0));
        EXPECT_TRUE(iter->isValueOn(10));
    }
}
