// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <cppunit/extensions/HelperMacros.h>

#include <openvdb/openvdb.h>

#include <openvdb/tools/Merge.h>

using namespace openvdb;

class TestMerge: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestMerge);
    CPPUNIT_TEST(testTreeToMerge);
    CPPUNIT_TEST(testCsgUnion);
    CPPUNIT_TEST(testCsgIntersection);
    CPPUNIT_TEST(testCsgDifference);
    CPPUNIT_TEST_SUITE_END();

    void testTreeToMerge();
    void testCsgUnion();
    void testCsgIntersection();
    void testCsgDifference();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMerge);

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

} // namespace


void
TestMerge::testTreeToMerge()
{
    using RootChildNode = FloatTree::RootNodeType::ChildNodeType;
    using LeafNode = FloatTree::LeafNodeType;

    { // non-const nullptr
        FloatTree* tree = nullptr;
        tools::TreeToMerge<FloatTree> treeToMerge(tree, Steal());
        CPPUNIT_ASSERT(!treeToMerge.rootPtr());
        CPPUNIT_ASSERT(!treeToMerge.probeConstNode<RootChildNode>(Coord(0)));
        std::unique_ptr<RootChildNode> nodePtr = treeToMerge.stealOrDeepCopyNode<RootChildNode>(Coord(0));
        CPPUNIT_ASSERT(!nodePtr);
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.3f, true);
        CPPUNIT_ASSERT(!treeToMerge.rootPtr());
    }

    { // const nullptr
        const FloatTree* tree = nullptr;
        tools::TreeToMerge<FloatTree> treeToMerge(tree, DeepCopy());
        CPPUNIT_ASSERT(!treeToMerge.rootPtr());
        CPPUNIT_ASSERT(!treeToMerge.probeConstNode<RootChildNode>(Coord(0)));
        std::unique_ptr<RootChildNode> nodePtr = treeToMerge.stealOrDeepCopyNode<RootChildNode>(Coord(0));
        CPPUNIT_ASSERT(!nodePtr);
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.3f, true);
        CPPUNIT_ASSERT(!treeToMerge.rootPtr());
    }

    { // non-const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount());

        tools::TreeToMerge<FloatTree> treeToMerge{&grid->tree(), Steal()};
        CPPUNIT_ASSERT_EQUAL(&grid->constTree().root(), treeToMerge.rootPtr());

        // probe root child

        const RootChildNode* nodePtr = treeToMerge.probeConstNode<RootChildNode>(Coord(8));
        CPPUNIT_ASSERT(nodePtr);
        CPPUNIT_ASSERT_EQUAL(grid->constTree().probeConstNode<RootChildNode>(Coord(8)), nodePtr);

        // probe leaf node

        const LeafNode* leafNode = treeToMerge.probeConstNode<LeafNode>(Coord(8));
        CPPUNIT_ASSERT(leafNode);
        CPPUNIT_ASSERT_EQUAL(grid->constTree().probeConstLeaf(Coord(8)), leafNode);
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().root().childCount());

        // steal leaf node

        std::unique_ptr<LeafNode> leafNodePtr = treeToMerge.stealOrDeepCopyNode<LeafNode>(Coord(8));
        CPPUNIT_ASSERT(leafNodePtr);
        CPPUNIT_ASSERT_EQUAL(Index(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(leafNodePtr->origin(), Coord(8));
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().root().childCount());

        // steal root child

        grid->tree().touchLeaf(Coord(8));
        std::unique_ptr<RootChildNode> node2Ptr = treeToMerge.stealOrDeepCopyNode<RootChildNode>(Coord(8));
        CPPUNIT_ASSERT(node2Ptr);
        CPPUNIT_ASSERT_EQUAL(Index(0), grid->tree().root().childCount());

        // attempt to add leaf node tile (set value)

        grid->tree().touchLeaf(Coord(8));
        CPPUNIT_ASSERT_EQUAL(Index64(0), grid->tree().activeTileCount());
        treeToMerge.addTile<LeafNode>(Coord(8), 1.6f, true);
        // value has not been set
        CPPUNIT_ASSERT_EQUAL(3.0f, grid->tree().probeConstLeaf(Coord(8))->getFirstValue());

        // add root child tile

        treeToMerge.addTile<RootChildNode>(Coord(8), 1.7f, true);
        CPPUNIT_ASSERT_EQUAL(Index64(1), grid->tree().activeTileCount());

        // tile in node that does not exist

        grid->tree().clear();
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.8f, true);
        CPPUNIT_ASSERT_EQUAL(Index64(0), grid->tree().activeTileCount());
    }

    { // const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount());

        tools::TreeToMerge<FloatTree> treeToMerge{&grid->constTree(), DeepCopy(), /*initialize=*/false};
        CPPUNIT_ASSERT(!treeToMerge.hasMask());
        treeToMerge.initializeMask();
        CPPUNIT_ASSERT(treeToMerge.hasMask());
        CPPUNIT_ASSERT_EQUAL(&grid->constTree().root(), treeToMerge.rootPtr());

        // probe root child

        const RootChildNode* nodePtr = treeToMerge.probeConstNode<RootChildNode>(Coord(8));
        CPPUNIT_ASSERT(nodePtr);
        CPPUNIT_ASSERT_EQUAL(grid->constTree().probeConstNode<RootChildNode>(Coord(8)), nodePtr);

        // probe leaf node

        const LeafNode* leafNode = treeToMerge.probeConstNode<LeafNode>(Coord(8));
        CPPUNIT_ASSERT(leafNode);
        CPPUNIT_ASSERT_EQUAL(grid->constTree().probeConstLeaf(Coord(8)), leafNode);
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().root().childCount());

        { // deep copy leaf node
            tools::TreeToMerge<FloatTree> treeToMerge2{&grid->constTree(), DeepCopy()};
            std::unique_ptr<LeafNode> leafNodePtr = treeToMerge2.stealOrDeepCopyNode<LeafNode>(Coord(8));
            CPPUNIT_ASSERT(leafNodePtr);
            CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount()); // leaf has not been stolen
            CPPUNIT_ASSERT_EQUAL(leafNodePtr->origin(), Coord(8));
            CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().root().childCount());
        }

        { // deep copy root child
            tools::TreeToMerge<FloatTree> treeToMerge2{&grid->constTree(), DeepCopy()};
            grid->tree().touchLeaf(Coord(8));
            std::unique_ptr<RootChildNode> node2Ptr = treeToMerge2.stealOrDeepCopyNode<RootChildNode>(Coord(8));
            CPPUNIT_ASSERT(node2Ptr);
            CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().root().childCount());
        }

        { // add root child tile
            tools::TreeToMerge<FloatTree> treeToMerge2{&grid->constTree(), DeepCopy()};
            CPPUNIT_ASSERT(treeToMerge2.probeConstNode<RootChildNode>(Coord(8)));
            treeToMerge2.addTile<RootChildNode>(Coord(8), 1.7f, true);
            CPPUNIT_ASSERT(!treeToMerge2.probeConstNode<RootChildNode>(Coord(8))); // tile has been added to mask
            CPPUNIT_ASSERT_EQUAL(Index64(0), grid->tree().activeTileCount());
        }

        // tile in node that does not exist

        grid->tree().clear();
        treeToMerge.addTile<RootChildNode>(Coord(0), 1.8f, true);
        CPPUNIT_ASSERT_EQUAL(Index64(0), grid->tree().activeTileCount());
    }

    { // non-const tree shared pointer
        // no tree or const tree
        tools::TreeToMerge<FloatTree> treeToMerge;
        CPPUNIT_ASSERT(!treeToMerge.tree());
        CPPUNIT_ASSERT(!treeToMerge.constTree());
        CPPUNIT_ASSERT(!treeToMerge.rootPtr());
        CPPUNIT_ASSERT(!treeToMerge.probeConstNode<FloatTree::LeafNodeType>(Coord(8)));

        {
            FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
            grid->tree().touchLeaf(Coord(8));
            CPPUNIT_ASSERT_EQUAL(Index(1), grid->tree().leafCount());

            treeToMerge.reset(grid->treePtr());
        }

        // verify tree shared ownership

        CPPUNIT_ASSERT(treeToMerge.tree());
        CPPUNIT_ASSERT(!treeToMerge.constTree());
        CPPUNIT_ASSERT(treeToMerge.rootPtr());
        CPPUNIT_ASSERT(treeToMerge.probeConstNode<FloatTree::LeafNodeType>(Coord(8)));

        // verify tree pointers are updated on reset()

        const FloatTree tree;
        tools::TreeToMerge<FloatTree> treeToMerge2(&tree, DeepCopy());
        treeToMerge2.initializeMask(); // no-op

        CPPUNIT_ASSERT(!treeToMerge2.tree());
        CPPUNIT_ASSERT(treeToMerge2.constTree());
        CPPUNIT_ASSERT_EQUAL(Index(0), treeToMerge2.constTree()->leafCount());

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(8));
        treeToMerge2.reset(grid->treePtr());

        CPPUNIT_ASSERT(treeToMerge2.tree());
        CPPUNIT_ASSERT(!treeToMerge2.constTree());
        CPPUNIT_ASSERT_EQUAL(Index(1), treeToMerge2.tree()->leafCount());
    }
}

void
TestMerge::testCsgUnion()
{
    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        { // one non-const tree (steal)
            tools::CsgUnionOp<FloatTree> mergeOp(&tree1, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgUnionOp<FloatTree> mergeOp(&tree1, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgUnionOp<FloatTree> mergeOp(&tree2, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // vector of tree pointers
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp.size());
        }
        { // deque of tree pointers
            std::deque<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp.size());
        }
        { // vector of TreesToMerge (to mix const and non-const trees)
            std::vector<tools::TreeToMerge<FloatTree>> trees;
            trees.emplace_back(&tree1, Steal());
            trees.emplace_back(&tree3, DeepCopy()); // const tree
            trees.emplace_back(&tree2, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp(trees);
            CPPUNIT_ASSERT_EQUAL(size_t(3), mergeOp.size());
        }
        { // implicit copy constructor
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp2(mergeOp);
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp2.size());
        }
        { // implicit assignment operator
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgUnionOp<FloatTree> mergeOp2 = mergeOp;
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp2.size());
        }
    }

    { // empty merge trees
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        std::vector<FloatTree*> trees;
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());

        CPPUNIT_ASSERT_EQUAL(size_t(0), mergeOp.size());

        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(0), root.getTableSize());
    }

    { // merge two different outside root tiles from one grid into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), false);
        root2.addTile(Coord(8192, 0, 0), grid->background(), true);

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root2));

        // test container constructor here
        std::vector<FloatTree*> trees{&grid2->tree()};
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getOutsideTileCount(root));
    }

    { // merge two different outside root tiles from two grids into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root2.addTile(Coord(0, 0, 0), /*background=*/123.0f, false);
        root3.addTile(Coord(8192, 0, 0), /*background=*/0.1f, true);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getOutsideTileCount(root));

        // background values of merge trees should be ignored, only important
        // that the values are greater than zero and thus an outside tile
        for (auto iter = root.cbeginValueAll(); iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(grid->background(), iter.getValue());
        }
    }

    { // merge the same outside root tiles from two grids into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);
        root3.addTile(Coord(0, 0, 0), grid->background(), false);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // merge order is important - tile should be active
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));

        root.clear();
        // reverse tree order
        std::vector<FloatTree*> trees2{&grid3->tree(), &grid2->tree()};
        tools::CsgUnionOp<FloatTree> mergeOp2(trees2, Steal());
        nodeManager.foreachTopDown(mergeOp2);
        // merge order is important - tile should now be inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
    }

    { // merge an outside root tile to a grid which already has this tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
    }

    { // merge an inside root tile to a grid which has an outside tile, inside takes precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid replace existing tile - tile should now be active and inside
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));
    }

    { // merge two grids with an outside and an inside tile, inside takes precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root3.addTile(Coord(0, 0, 0), /*inside*/-0.1f, false);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgUnionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));
    }

    { // merge two child nodes into an empty grid
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), -123.0f, true));

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root2));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
    }

    { // merge a child node into a grid with an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 123.0f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));

        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root2));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
    }

    { // merge a child node into a grid with an existing child node
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root2));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
        CPPUNIT_ASSERT(root.cbeginChildOn()->cbeginValueAll());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->cbeginValueAll().getItem(0));
        CPPUNIT_ASSERT_EQUAL(1.9f, (++root.cbeginChildOn())->cbeginValueAll().getItem(0));
    }

    { // merge an inside tile and an outside tile into a grid with two child nodes
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 15.0f, false); // should not replace child
        root2.addTile(Coord(8192, 0, 0), -25.0f, false); // should replace child

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT(root.cbeginChildAll().isChildNode());
        CPPUNIT_ASSERT(!(++root.cbeginChildAll()).isChildNode());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->getFirstValue());
        // inside tile value replaced with negative background
        CPPUNIT_ASSERT_EQUAL(-grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge two child nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 15.0f, false); // should be replaced by child
        root.addTile(Coord(8192, 0, 0), -25.0f, false); // should not be replaced by child
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));

        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root2));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT(root.cbeginChildAll().isChildNode());
        CPPUNIT_ASSERT(!(++root.cbeginChildAll()).isChildNode());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->getFirstValue());
        CPPUNIT_ASSERT_EQUAL(-grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge two internal nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;
        using LeafParentType = RootChildType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
        rootChild->addTile(1, -14.0f, false);
        root.addChild(rootChild.release());
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

        rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
        rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
        rootChild2->addTile(2, 17.0f, true);
        rootChild2->addTile(9, -19.0f, true);
        root2.addChild(rootChild2.release());

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(*root.cbeginChildOn()));

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getActiveTileCount(*root2.cbeginChildOn()));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT(root.cbeginChildOn()->isChildMaskOn(0));
        CPPUNIT_ASSERT(!root.cbeginChildOn()->isChildMaskOn(1));
        CPPUNIT_ASSERT_EQUAL(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
        CPPUNIT_ASSERT_EQUAL(-14.0f, root.cbeginChildOn()->cbeginValueAll().getValue());

        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(*root2.cbeginChildOn()));
    }

    { // merge two internal nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;
        using LeafParentType = RootChildType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
        rootChild->addTile(1, -14.0f, false);
        root.addChild(rootChild.release());
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

        rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
        rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
        rootChild2->addTile(2, 17.0f, true);
        rootChild2->addTile(9, -19.0f, true);
        root2.addChild(rootChild2.release());

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(*root.cbeginChildOn()));

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getActiveTileCount(*root2.cbeginChildOn()));

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT(root.cbeginChildOn()->isChildMaskOn(0));
        CPPUNIT_ASSERT(!root.cbeginChildOn()->isChildMaskOn(1));
        CPPUNIT_ASSERT_EQUAL(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
        CPPUNIT_ASSERT_EQUAL(-14.0f, root.cbeginChildOn()->cbeginValueAll().getValue());

        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(*root2.cbeginChildOn()));
    }

    { // merge a leaf node into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(0), grid2->tree().leafCount());
    }

    { // merge a leaf node into a grid with a partially constructed leaf node
        using LeafT = FloatTree::LeafNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
        auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, -2.3f);

        tools::CsgUnionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        CPPUNIT_ASSERT_EQUAL(-2.3f, testLeaf->getValue(10));
    }

    { // merge three leaf nodes from different grids
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();

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
        CPPUNIT_ASSERT_EQUAL(2.0f, testLeaf->getValue(5));
        CPPUNIT_ASSERT(testLeaf->isValueOn(5));
        CPPUNIT_ASSERT_EQUAL(2.0f, testLeaf->getValue(7));
        CPPUNIT_ASSERT(testLeaf->isValueOn(7));
        CPPUNIT_ASSERT_EQUAL(2.0f, testLeaf->getValue(9));
        CPPUNIT_ASSERT(!testLeaf->isValueOn(9));
    }

    { // merge a leaf node into an empty grid from a const grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        // merge from a const tree

        std::vector<tools::TreeToMerge<FloatTree>> treesToMerge;
        treesToMerge.emplace_back(&grid2->constTree(), DeepCopy());

        tools::CsgUnionOp<FloatTree> mergeOp(treesToMerge);
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        // leaf has been deep copied not stolen
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());
    }
}

void
TestMerge::testCsgIntersection()
{
    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        { // one non-const tree (steal)
            tools::CsgIntersectionOp<FloatTree> mergeOp(&tree1, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgIntersectionOp<FloatTree> mergeOp(&tree1, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgIntersectionOp<FloatTree> mergeOp(&tree2, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // vector of tree pointers
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp.size());
        }
        { // deque of tree pointers
            std::deque<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp.size());
        }
        { // vector of TreesToMerge (to mix const and non-const trees)
            std::vector<tools::TreeToMerge<FloatTree>> trees;
            trees.emplace_back(&tree1, Steal());
            trees.emplace_back(&tree3, DeepCopy()); // const tree
            trees.emplace_back(&tree2, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees);
            CPPUNIT_ASSERT_EQUAL(size_t(3), mergeOp.size());
        }
        { // implicit copy constructor
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp2(mergeOp);
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp2.size());
        }
        { // implicit assignment operator
            std::vector<FloatTree*> trees{&tree1, &tree2};
            tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
            tools::CsgIntersectionOp<FloatTree> mergeOp2 = mergeOp;
            CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp2.size());
        }
    }

    { // empty merge trees
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        std::vector<FloatTree*> trees;
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());

        CPPUNIT_ASSERT_EQUAL(size_t(0), mergeOp.size());

        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(0), root.getTableSize());
    }

    { // merge two different outside root tiles from one grid into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), false);
        root2.addTile(Coord(8192, 0, 0), grid->background(), true);

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root2));

        // test container constructor here
        std::vector<FloatTree*> trees{&grid2->tree()};
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getOutsideTileCount(root));
    }

    { // merge two different outside root tiles from two grids into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root2.addTile(Coord(0, 0, 0), /*background=*/123.0f, false);
        root3.addTile(Coord(8192, 0, 0), /*background=*/0.1f, true);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getOutsideTileCount(root));

        // background values of merge trees should be ignored, only important
        // that the values are greater than zero and thus an outside tile
        for (auto iter = root.cbeginValueAll(); iter; ++iter) {
            CPPUNIT_ASSERT_EQUAL(grid->background(), iter.getValue());
        }
    }

    { // merge the same outside root tiles from two grids into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);
        root3.addTile(Coord(0, 0, 0), grid->background(), false);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // merge order is important - tile should be active
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));

        root.clear();
        // reverse tree order
        std::vector<FloatTree*> trees2{&grid3->tree(), &grid2->tree()};
        tools::CsgIntersectionOp<FloatTree> mergeOp2(trees2, Steal());
        nodeManager.foreachTopDown(mergeOp2);
        // merge order is important - tile should now be inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
    }

    { // merge an outside root tile to a grid which already has this tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), true);

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
    }

    { // merge an outside root tile to a grid which has an inside tile, outside takes precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid replace existing tile - tile should now be active and outside
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
    }

    { // merge two grids with an outside and an inside tile, outside takes precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -grid->background(), true);
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();
        auto& root3 = grid3->tree().root();
        root3.addTile(Coord(0, 0, 0), /*outside*/0.1f, false);

        std::vector<FloatTree*> trees{&grid2->tree(), &grid3->tree()};
        tools::CsgIntersectionOp<FloatTree> mergeOp(trees, Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
    }

    { // merge two child nodes into an empty grid
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), -123.0f, true));

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root2));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
    }

    { // merge a child node into a grid with an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 123.0f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));

        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root2));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
    }

    { // merge a child node into a grid with an existing child node
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root2));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
        CPPUNIT_ASSERT(root.cbeginChildOn()->cbeginValueAll());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->cbeginValueAll().getItem(0));
        CPPUNIT_ASSERT_EQUAL(1.9f, (++root.cbeginChildOn())->cbeginValueAll().getItem(0));
    }

    { // merge an inside tile and an outside tile into a grid with two child nodes
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -15.0f, false); // should not replace child
        root2.addTile(Coord(8192, 0, 0), 25.0f, false); // should replace child

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT(root.cbeginChildAll().isChildNode());
        CPPUNIT_ASSERT(!(++root.cbeginChildAll()).isChildNode());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->getFirstValue());
        // outside tile value replaced with background
        CPPUNIT_ASSERT_EQUAL(grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge two child nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -15.0f, false); // should be replaced by child
        root.addTile(Coord(8192, 0, 0), 25.0f, false); // should not be replaced by child
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));

        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root2));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT(root.cbeginChildAll().isChildNode());
        CPPUNIT_ASSERT(!(++root.cbeginChildAll()).isChildNode());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->getFirstValue());
        CPPUNIT_ASSERT_EQUAL(grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge two internal nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;
        using LeafParentType = RootChildType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        auto rootChild = std::make_unique<RootChildType>(Coord(0, 0, 0), 123.0f, false);
        rootChild->addTile(0, -14.0f, false);
        rootChild->addTile(1, 15.0f, false);
        root.addChild(rootChild.release());
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        auto rootChild2 = std::make_unique<RootChildType>(Coord(0, 0, 0), 55.0f, false);

        rootChild2->addChild(new LeafParentType(Coord(0, 0, 0), 29.0f, false));
        rootChild2->addChild(new LeafParentType(Coord(0, 0, 128), 31.0f, false));
        rootChild2->addTile(2, -17.0f, true);
        rootChild2->addTile(9, 19.0f, true);
        root2.addChild(rootChild2.release());

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(*root.cbeginChildOn()));

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getActiveTileCount(*root2.cbeginChildOn()));

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT(root.cbeginChildOn()->isChildMaskOn(0));
        CPPUNIT_ASSERT(!root.cbeginChildOn()->isChildMaskOn(1));
        CPPUNIT_ASSERT_EQUAL(29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
        CPPUNIT_ASSERT_EQUAL(15.0f, root.cbeginChildOn()->cbeginValueAll().getValue());

        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(*root2.cbeginChildOn()));
    }

    { // merge a leaf node into an empty grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(0), grid2->tree().leafCount());
    }

    { // merge a leaf node into a grid with a partially constructed leaf node
        using LeafT = FloatTree::LeafNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
        auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, 6.4f);

        tools::CsgIntersectionOp<FloatTree> mergeOp{&grid2->tree(), Steal()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        CPPUNIT_ASSERT_EQUAL(6.4f, testLeaf->getValue(10));
    }

    { // merge three leaf nodes from different grids
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid3 = createLevelSet<FloatGrid>();

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
        CPPUNIT_ASSERT_EQUAL(4.0f, testLeaf->getValue(5));
        CPPUNIT_ASSERT(!testLeaf->isValueOn(5));
        CPPUNIT_ASSERT_EQUAL(4.0f, testLeaf->getValue(7));
        CPPUNIT_ASSERT(!testLeaf->isValueOn(7));
        CPPUNIT_ASSERT_EQUAL(4.0f, testLeaf->getValue(9));
        CPPUNIT_ASSERT(testLeaf->isValueOn(9));
    }

    { // merge a leaf node into an empty grid from a const grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        // merge from a const tree

        std::vector<tools::TreeToMerge<FloatTree>> treesToMerge;
        treesToMerge.emplace_back(&grid2->constTree(), DeepCopy());

        tools::CsgIntersectionOp<FloatTree> mergeOp(treesToMerge);
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        // leaf has been deep copied not stolen
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());
    }
}

void
TestMerge::testCsgDifference()
{
    using RootChildType = FloatTree::RootNodeType::ChildNodeType;

    { // construction
        FloatTree tree1;
        const FloatTree tree2;

        { // one non-const tree (steal)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree1, Steal());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one non-const tree (deep-copy)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree1, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one const tree (deep-copy)
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one non-const tree wrapped in TreeToMerge
            tools::TreeToMerge<FloatTree> tree3(&tree1, Steal());
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree3);
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // one const tree wrapped in TreeToMerge
            tools::TreeToMerge<FloatTree> tree4(&tree2, DeepCopy());
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree4);
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
        }
        { // implicit copy constructor
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
            tools::CsgDifferenceOp<FloatTree> mergeOp2(mergeOp);
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp2.size());
        }
        { // implicit assignment operator
            tools::CsgDifferenceOp<FloatTree> mergeOp(tree2, DeepCopy());
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp.size());
            tools::CsgDifferenceOp<FloatTree> mergeOp2 = mergeOp;
            CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp2.size());
        }
    }

    { // merge two different outside root tiles from one grid into an empty grid (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), grid->background(), false);
        root2.addTile(Coord(8192, 0, 0), grid->background(), true);

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root2));

        // test container constructor here
        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(0), root.getTableSize());
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

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // tile in merge grid should not replace existing tile - tile should remain inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
    }

    { // merge an outside root tile to a grid which has an inside tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));
    }

    { // merge an outside root tile to a grid which has a child (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
    }

    { // merge a child to a grid which has an outside root tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), 123.0f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));

        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
    }

    { // merge an inside root tile to a grid which has an outside tile (noop)
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), grid->background(), false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
    }

    { // merge two grids with inside tiles, active state should be carried across
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -0.1f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -0.2f, false);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // inside tile should now be inactive
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));

        CPPUNIT_ASSERT_EQUAL(grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge an inside root tile to a grid which has a child, inside tile has precedence
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), -123.0f, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));

        CPPUNIT_ASSERT_EQUAL(grid->background(), root.cbeginValueAll().getValue());
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

        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getOutsideTileCount(root));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp, true);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getChildCount(root2));

        CPPUNIT_ASSERT(!root.cbeginChildOn()->isValueOn(Index(0)));
        CPPUNIT_ASSERT(root.cbeginChildOn()->isValueOn(Index(1)));

        auto iter = root.cbeginChildOn()->cbeginValueAll();
        CPPUNIT_ASSERT_EQUAL(-3.0f, iter.getValue());
        ++iter;
        CPPUNIT_ASSERT_EQUAL(-1.3f, iter.getValue());
    }

    { // merge two child nodes into a grid with two inside tiles
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addTile(Coord(0, 0, 0), -2.0f, false);
        root.addTile(Coord(8192, 0, 0), -4.0f, false);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addChild(new RootChildType(Coord(0, 0, 0), 1.0f, false));
        root2.addChild(new RootChildType(Coord(8192, 0, 0), -123.0f, true));

        CPPUNIT_ASSERT_EQUAL(Index(2), root2.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root2));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root2));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(0), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
    }

    { // merge an inside tile and an outside tile into a grid with two child nodes
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        root.addChild(new RootChildType(Coord(0, 0, 0), 123.0f, false));
        root.addChild(new RootChildType(Coord(8192, 0, 0), 1.9f, false));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        auto& root2 = grid2->tree().root();
        root2.addTile(Coord(0, 0, 0), 15.0f, false); // should not replace child
        root2.addTile(Coord(8192, 0, 0), -25.0f, false); // should replace child

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(2), getTileCount(root2));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(2), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(1), getOutsideTileCount(root));
        CPPUNIT_ASSERT(root.cbeginChildAll().isChildNode());
        CPPUNIT_ASSERT(!(++root.cbeginChildAll()).isChildNode());
        CPPUNIT_ASSERT_EQUAL(123.0f, root.cbeginChildOn()->getFirstValue());
        // outside tile value replaced with negative background
        CPPUNIT_ASSERT_EQUAL(grid->background(), root.cbeginValueAll().getValue());
    }

    { // merge two internal nodes into a grid with an inside tile and an outside tile
        using RootChildType = FloatTree::RootNodeType::ChildNodeType;
        using LeafParentType = RootChildType::ChildNodeType;

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

        CPPUNIT_ASSERT_EQUAL(Index(2), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getActiveTileCount(*root.cbeginChildOn()));

        CPPUNIT_ASSERT_EQUAL(Index(2), getChildCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getInsideTileCount(*root2.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(2), getActiveTileCount(*root2.cbeginChildOn()));

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInsideTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(*root.cbeginChildOn()));
        CPPUNIT_ASSERT(root.cbeginChildOn()->isChildMaskOn(0));
        CPPUNIT_ASSERT(!root.cbeginChildOn()->isChildMaskOn(1));
        CPPUNIT_ASSERT_EQUAL(-29.0f, root.cbeginChildOn()->cbeginChildOn()->getFirstValue());
        auto iter = root.cbeginChildOn()->cbeginValueAll();
        CPPUNIT_ASSERT_EQUAL(15.0f, iter.getValue());
        ++iter;
        CPPUNIT_ASSERT_EQUAL(3.0f, iter.getValue());

        CPPUNIT_ASSERT_EQUAL(Index(1), getChildCount(*root2.cbeginChildOn()));
    }

    { // merge a leaf node into a grid with an inside tile
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().addTile(1, Coord(0, 0, 0), -1.3f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(0), grid2->tree().leafCount());
    }

    { // merge two leaf nodes into a grid
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().touchLeaf(Coord(0, 0, 0));
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* leaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        CPPUNIT_ASSERT(leaf);
    }

    { // merge a leaf node into a grid with a partially constructed leaf node
        using LeafT = FloatTree::LeafNodeType;

        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();

        grid->tree().addLeaf(new LeafT(PartialCreate(), Coord(0, 0, 0)));
        auto* leaf = grid2->tree().touchLeaf(Coord(0, 0, 0));
        leaf->setValueOnly(10, 6.4f);

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->tree(), Steal());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        const auto* testLeaf = grid->tree().probeConstLeaf(Coord(0, 0, 0));
        CPPUNIT_ASSERT_EQUAL(3.0f, testLeaf->getValue(10));
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
        CPPUNIT_ASSERT_EQUAL(98.0f, testLeaf->getValue(5));
        CPPUNIT_ASSERT(!testLeaf->isValueOn(5));
        CPPUNIT_ASSERT_EQUAL(2.0f, testLeaf->getValue(7));
        CPPUNIT_ASSERT(testLeaf->isValueOn(7));
        CPPUNIT_ASSERT_EQUAL(100.0f, testLeaf->getValue(9));
        CPPUNIT_ASSERT(!testLeaf->isValueOn(9));
    }

    { // merge a leaf node into a grid with an inside tile from a const tree
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        grid->tree().addTile(1, Coord(0, 0, 0), -1.3f, true);
        FloatGrid::Ptr grid2 = createLevelSet<FloatGrid>();
        grid2->tree().touchLeaf(Coord(0, 0, 0));

        CPPUNIT_ASSERT_EQUAL(Index32(0), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());

        tools::CsgDifferenceOp<FloatTree> mergeOp(grid2->constTree(), DeepCopy());
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());
    }
}
