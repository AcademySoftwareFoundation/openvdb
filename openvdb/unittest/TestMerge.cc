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
    CPPUNIT_TEST(testCsgUnion);
    CPPUNIT_TEST_SUITE_END();

    void testCsgUnion();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestMerge);

namespace
{

auto getTileCount = [](const auto& root) -> Index
{
    Index sum = 0;
    for (auto iter = root.cbeginValueAll(); iter; ++iter)   sum++;
    return sum;
};

auto getActiveTileCount = [](const auto& root) -> Index
{
    Index sum = 0;
    for (auto iter = root.cbeginValueOn(); iter; ++iter)   sum++;
    return sum;
};

auto getInactiveTileCount = [](const auto& root) -> Index
{
    Index sum = 0;
    for (auto iter = root.cbeginValueOff(); iter; ++iter)   sum++;
    return sum;
};

auto getInsideTileCount = [](const auto& root) -> Index
{
    using ValueT = typename std::remove_reference<decltype(root)>::type::ValueType;
    Index sum = 0;
    for (auto iter = root.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() < zeroVal<ValueT>())     sum++;
    }
    return sum;
};

auto getOutsideTileCount = [](const auto& root) -> Index
{
    using ValueT = typename std::remove_reference<decltype(root)>::type::ValueType;
    Index sum = 0;
    for (auto iter = root.cbeginValueAll(); iter; ++iter) {
        if (iter.getValue() > zeroVal<ValueT>())     sum++;
    }
    return sum;
};

auto getChildCount = [](const auto& root) -> Index
{
    Index sum = 0;
    for (auto iter = root.cbeginChildOn(); iter; ++iter)   sum++;
    return sum;
};

} // namespace

void
TestMerge::testCsgUnion()
{
    std::cerr << std::endl;

    { // construction
        FloatTree tree1;
        FloatTree tree2;
        const FloatTree tree3;

        // empty
        tools::CsgUnionMergeOp<FloatTree> mergeOp1{};
        CPPUNIT_ASSERT_EQUAL(size_t(0), mergeOp1.size());
        // one item, initializer list
        tools::CsgUnionMergeOp<FloatTree> mergeOp2{&tree1};
        CPPUNIT_ASSERT_EQUAL(size_t(1), mergeOp2.size());
        // two items, initializer list
        tools::CsgUnionMergeOp<FloatTree> mergeOp3{&tree1, &tree2};
        CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp3.size());
        // vector of tree pointers
        std::vector<FloatTree*> trees4{&tree1, &tree2};
        tools::CsgUnionMergeOp<FloatTree> mergeOp4(trees4);
        CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp4.size());
        // deque of tree pointers
        std::deque<FloatTree*> trees5{&tree1, &tree2};
        tools::CsgUnionMergeOp<FloatTree> mergeOp5(trees5);
        CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp5.size());
        // vector of TreesToMerge (to mix const and non-const trees)
        std::vector<tools::TreeToMerge<FloatTree>> trees6;
        trees6.emplace_back(&tree1);
        trees6.emplace_back(&tree3); // const tree
        trees6.emplace_back(&tree2);
        tools::CsgUnionMergeOp<FloatTree> mergeOp6(trees6);
        CPPUNIT_ASSERT_EQUAL(size_t(3), mergeOp6.size());
        // implicit copy constructor
        tools::CsgUnionMergeOp<FloatTree> mergeOp7(mergeOp3);
        CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp7.size());
        // implicit assignment operator
        tools::CsgUnionMergeOp<FloatTree> mergeOp8 = mergeOp3;
        CPPUNIT_ASSERT_EQUAL(size_t(2), mergeOp8.size());
    }

    { // empty merge trees
        FloatGrid::Ptr grid = createLevelSet<FloatGrid>();
        auto& root = grid->tree().root();
        tools::CsgUnionMergeOp<FloatTree> mergeOp{};

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
        std::vector<FloatTree*> treesToMerge{&grid2->tree()};
        tools::CsgUnionMergeOp<FloatTree> mergeOp(treesToMerge);
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree(), &grid3->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree(), &grid3->tree()};
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index(1), root.getTableSize());
        CPPUNIT_ASSERT_EQUAL(Index(1), getTileCount(root));
        // merge order is important - tile should be active
        CPPUNIT_ASSERT_EQUAL(Index(1), getActiveTileCount(root));
        CPPUNIT_ASSERT_EQUAL(Index(0), getInactiveTileCount(root));

        root.clear();
        // reverse tree order
        tools::CsgUnionMergeOp<FloatTree> mergeOp2{&grid3->tree(), &grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree(), &grid3->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree()};
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

        tools::CsgUnionMergeOp<FloatTree> mergeOp{&grid2->tree(), &grid3->tree()};
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

        std::vector<tools::TreeToMerge<FloatTree>> trees;
        trees.emplace_back(&grid2->constTree());

        tools::CsgUnionMergeOp<FloatTree> mergeOp(trees);
        tree::DynamicNodeManager<FloatTree, 3> nodeManager(grid->tree());
        nodeManager.foreachTopDown(mergeOp);

        CPPUNIT_ASSERT_EQUAL(Index32(1), grid->tree().leafCount());
        // leaf has been deep copied not stolen
        CPPUNIT_ASSERT_EQUAL(Index32(1), grid2->tree().leafCount());
    }
}
