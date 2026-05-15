// Copyright Contributors to the OpenVDB Project
// SDPX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/VectorFromScalar.h>

#include <gtest/gtest.h>

using namespace openvdb;

class TestVectorFromScalar: public ::testing::Test
{
};

TEST_F(TestVectorFromScalar, testEmptyGrids)
{
    auto xGrid = createGrid<FloatGrid>(1.1f);
    auto yGrid = createGrid<FloatGrid>(2.2f);
    auto zGrid = createGrid<FloatGrid>(3.3f);

    auto vectorGrid = tools::vectorFromScalar(*xGrid, *yGrid, *zGrid);

    EXPECT_EQ(vectorGrid->background(), Vec3f(1.1f, 2.2f, 3.3f));
}

TEST_F(TestVectorFromScalar, testMergeVoxels)
{
    auto xGrid = createGrid<FloatGrid>(-0.1f);
    auto yGrid = createGrid<FloatGrid>(-0.2f);
    auto zGrid = createGrid<FloatGrid>(-0.3f);

    auto& xTree = xGrid->tree();
    auto& yTree = yGrid->tree();
    auto& zTree = zGrid->tree();

    // Create various overlapping and non-overlapping voxels
    // voxel   |
    // index   | 0    1    2    3    4    5    6    7
    // --------|---------------------------------------
    // x grid  |    [1.1]     [3.1]     [5.1]     [7.1]
    // y grid  |         [2.2][3.2]          [6.2][7.2]
    // z grid  |                   [4.3][5.3][6.3][7.3]

    xTree.setValue(Coord(1, 0, 0), 1.1f);
    xTree.setValue(Coord(3, 0, 0), 3.1f);
    xTree.setValue(Coord(5, 0, 0), 5.1f);
    xTree.setValue(Coord(7, 0, 0), 7.1f);

    yTree.setValue(Coord(2, 0, 0), 2.2f);
    yTree.setValue(Coord(3, 0, 0), 3.2f);
    yTree.setValue(Coord(6, 0, 0), 6.2f);
    yTree.setValue(Coord(7, 0, 0), 7.2f);

    zTree.setValue(Coord(4, 0, 0), 4.3f);
    zTree.setValue(Coord(5, 0, 0), 5.3f);
    zTree.setValue(Coord(6, 0, 0), 6.3f);
    zTree.setValue(Coord(7, 0, 0), 7.3f);

    auto vectorGrid = tools::vectorFromScalar(*xGrid, *yGrid, *zGrid);
    auto& vectorTree = vectorGrid->tree();

    EXPECT_EQ(vectorGrid->background(), Vec3f(-0.1f, -0.2f, -0.3f));

    EXPECT_EQ(vectorTree.activeTileCount(), 0);
    EXPECT_EQ(vectorTree.leafCount(), 1);
    EXPECT_EQ(vectorTree.activeVoxelCount(), 7);

    EXPECT_EQ(vectorTree.getValue(Coord(-1, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 0, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 1, 0, 0)), Vec3f( 1.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 2, 0, 0)), Vec3f(-0.1f,  2.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 3, 0, 0)), Vec3f( 3.1f,  3.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 4, 0, 0)), Vec3f(-0.1f, -0.2f,  4.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 5, 0, 0)), Vec3f( 5.1f, -0.2f,  5.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 6, 0, 0)), Vec3f(-0.1f,  6.2f,  6.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 7, 0, 0)), Vec3f( 7.1f,  7.2f,  7.3f));
    EXPECT_EQ(vectorTree.getValue(Coord( 8, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    const Index NOT_FOUND_DEPTH = -1;
    const Index L2_DEPTH = 2;
    const Index LEAF_DEPTH = 3;

    EXPECT_EQ(vectorTree.getValueDepth(Coord(-1, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 0, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 1, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 2, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 3, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 4, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 5, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 6, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 7, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord( 8, 0, 0)), L2_DEPTH);
}

TEST_F(TestVectorFromScalar, testMergeRootTiles)
{
    auto xGrid = createGrid<FloatGrid>(-0.1f);
    auto yGrid = createGrid<FloatGrid>(-0.2f);
    auto zGrid = createGrid<FloatGrid>(-0.3f);

    auto& xTree = xGrid->tree();
    auto& yTree = yGrid->tree();
    auto& zTree = zGrid->tree();

    const Index ROOT_LEVEL = FloatTree::RootNodeType::getLevel();
    const Index ROOT_STRIDE = FloatTree::RootNodeType::getChildDim();

    // Create various overlapping and non-overlapping tiles
    // index at  |
    // L0 stride | 0    1    2    3    4    5    6    7
    // ----------|---------------------------------------
    // x grid    |    [1.1]     [3.1]     [5.1]     [7.1]
    // y grid    |         [2.2][3.2]          [6.2][7.2]
    // z grid    |                   [4.3][5.3][6.3][7.3]

    xTree.addTile(ROOT_LEVEL, Coord(1 * ROOT_STRIDE, 0, 0), 1.1f, true);
    xTree.addTile(ROOT_LEVEL, Coord(3 * ROOT_STRIDE, 0, 0), 3.1f, true);
    xTree.addTile(ROOT_LEVEL, Coord(5 * ROOT_STRIDE, 0, 0), 5.1f, true);
    xTree.addTile(ROOT_LEVEL, Coord(7 * ROOT_STRIDE, 0, 0), 7.1f, true);

    yTree.addTile(ROOT_LEVEL, Coord(2 * ROOT_STRIDE, 0, 0), 2.2f, true);
    yTree.addTile(ROOT_LEVEL, Coord(3 * ROOT_STRIDE, 0, 0), 3.2f, true);
    yTree.addTile(ROOT_LEVEL, Coord(6 * ROOT_STRIDE, 0, 0), 6.2f, true);
    yTree.addTile(ROOT_LEVEL, Coord(7 * ROOT_STRIDE, 0, 0), 7.2f, true);

    zTree.addTile(ROOT_LEVEL, Coord(4 * ROOT_STRIDE, 0, 0), 4.3f, true);
    zTree.addTile(ROOT_LEVEL, Coord(5 * ROOT_STRIDE, 0, 0), 5.3f, true);
    zTree.addTile(ROOT_LEVEL, Coord(6 * ROOT_STRIDE, 0, 0), 6.3f, true);
    zTree.addTile(ROOT_LEVEL, Coord(7 * ROOT_STRIDE, 0, 0), 7.3f, true);

    auto vectorGrid = tools::vectorFromScalar(*xGrid, *yGrid, *zGrid);
    auto& vectorTree = vectorGrid->tree();

    EXPECT_EQ(vectorGrid->background(), Vec3f(-0.1f, -0.2f, -0.3f));

    EXPECT_EQ(vectorTree.activeTileCount(), 7);
    EXPECT_EQ(vectorTree.leafCount(), 0);

    EXPECT_EQ(vectorTree.getValue(Coord(0 * ROOT_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE, 0, 0)), Vec3f( 1.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE, 0, 0)), Vec3f(-0.1f,  2.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE, 0, 0)), Vec3f( 3.1f,  3.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(4 * ROOT_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f,  4.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(5 * ROOT_STRIDE, 0, 0)), Vec3f( 5.1f, -0.2f,  5.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(6 * ROOT_STRIDE, 0, 0)), Vec3f(-0.1f,  6.2f,  6.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(7 * ROOT_STRIDE, 0, 0)), Vec3f( 7.1f,  7.2f,  7.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(8 * ROOT_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    const Index NOT_FOUND_DEPTH = -1;
    const Index ROOT_DEPTH = 0;

    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(5 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(6 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(7 * ROOT_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(8 * ROOT_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
}

TEST_F(TestVectorFromScalar, testMergeMixedLevelTiles)
{
    auto xGrid = createGrid<FloatGrid>(-0.1f);
    auto yGrid = createGrid<FloatGrid>(-0.2f);
    auto zGrid = createGrid<FloatGrid>(-0.3f);

    auto& xTree = xGrid->tree();
    auto& yTree = yGrid->tree();
    auto& zTree = zGrid->tree();

    using RootNodeType = typename FloatTree::RootNodeType;
    using L1NodeType = typename RootNodeType::ChildNodeType;

    const Index ROOT_LEVEL = RootNodeType::getLevel();
    const Index ROOT_STRIDE = RootNodeType::getChildDim();
    const Index L1_LEVEL = L1NodeType::getLevel();
    const Index L1_STRIDE = L1NodeType::getChildDim();

    // x grid    |                       [---------1.1--------][---------2.1--------][-------3.1--------]
    // y grid    | [[0.2][1.2]      ... ]                      [[4.2][5.2]       ...]
    // z grid    | [--------0.3---------][[2.3][3.3]      ... ][[4.3][5.3]       ...][-------6.3--------]
    // ----------|---------------------------------------------------------------------------------------
    // expected  | [[---][---][---][...]][[---][---][---][...]][[---][---][---][...]][------------------]
    // topology  |
    //           | <--- L1 tiles ---------------------------------------------------><--- root tile ---->

    xTree.addTile(ROOT_LEVEL, Coord(1 * ROOT_STRIDE, 0, 0), 1.1f, true);
    xTree.addTile(ROOT_LEVEL, Coord(2 * ROOT_STRIDE, 0, 0), 2.1f, true);
    xTree.addTile(ROOT_LEVEL, Coord(3 * ROOT_STRIDE, 0, 0), 3.1f, true);

    yTree.addTile(L1_LEVEL, Coord(0 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0), 0.2f, true);
    yTree.addTile(L1_LEVEL, Coord(0 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0), 1.2f, true);
    yTree.addTile(L1_LEVEL, Coord(2 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0), 4.2f, true);
    yTree.addTile(L1_LEVEL, Coord(2 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0), 5.2f, true);

    zTree.addTile(ROOT_LEVEL, Coord(0 * ROOT_STRIDE, 0, 0), 0.3f, true);
    zTree.addTile(L1_LEVEL, Coord(1 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0), 2.3f, true);
    zTree.addTile(L1_LEVEL, Coord(1 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0), 3.3f, true);
    zTree.addTile(L1_LEVEL, Coord(2 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0), 4.3f, true);
    zTree.addTile(L1_LEVEL, Coord(2 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0), 5.3f, true);
    zTree.addTile(ROOT_LEVEL, Coord(3 * ROOT_STRIDE, 0, 0), 6.3f, true);

    auto vectorGrid = tools::vectorFromScalar(*xGrid, *yGrid, *zGrid);
    auto& vectorTree = vectorGrid->tree();

    EXPECT_EQ(vectorGrid->background(), Vec3f(-0.1f, -0.2f, -0.3f));

    // Footprint of root tile -1
    EXPECT_EQ(vectorTree.getValue(Coord(-1 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    // Footprint of root tile 0
    EXPECT_EQ(vectorTree.getValue(Coord(0 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, 0.2f, 0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, 1.2f, 0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, 0.3f));
    // ...
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, 0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, 0.3f));

    // Footprint of root tile 1
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(1.1f, -0.2f, 2.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), Vec3f(1.1f, -0.2f, 3.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), Vec3f(1.1f, -0.2f, -0.3f));
    // ...
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), Vec3f(1.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(1.1f, -0.2f, -0.3f));

    // Footprint of root tile 2
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(2.1f, 4.2f, 4.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), Vec3f(2.1f, 5.2f, 5.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(2 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), Vec3f(2.1f, -0.2f, -0.3f));
    // ...
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), Vec3f(2.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(2.1f, -0.2f, -0.3f));

    // Footprint of root tile 3
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(3.1f, -0.2f, 6.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), Vec3f(3.1f, -0.2f, 6.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(3 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), Vec3f(3.1f, -0.2f, 6.3f));
    // ...
    EXPECT_EQ(vectorTree.getValue(Coord(4 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), Vec3f(3.1f, -0.2f, 6.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(4 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(3.1f, -0.2f, 6.3f));

    // Footprint of root tile 4
    EXPECT_EQ(vectorTree.getValue(Coord(4 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(5 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    const Index NOT_FOUND_DEPTH = -1;
    const Index ROOT_DEPTH = 0;
    const Index L1_DEPTH = 1;

    // Footprint of root tile -1
    EXPECT_EQ(vectorTree.getValueDepth(Coord(-1 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(-1 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(-1 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);

    // // Footprint of root tile 0
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), L1_DEPTH);

    // Footprint of root tile 1
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), L1_DEPTH);

    // Footprint of root tile 2
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), L1_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), L1_DEPTH);

    // Footprint of root tile 3
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), ROOT_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), ROOT_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), ROOT_DEPTH);

    // Footprint of root tile 4
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE + 0 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE + 1 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE + 2 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    // ...
    EXPECT_EQ(vectorTree.getValueDepth(Coord(5 * ROOT_STRIDE - 2 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(5 * ROOT_STRIDE - 1 * L1_STRIDE, 0, 0)), NOT_FOUND_DEPTH);
}

TEST_F(TestVectorFromScalar, testMergeTilesAndVoxels)
{
    auto xGrid = createGrid<FloatGrid>(-0.1f);
    auto yGrid = createGrid<FloatGrid>(-0.2f);
    auto zGrid = createGrid<FloatGrid>(-0.3f);

    auto& xTree = xGrid->tree();
    auto& yTree = yGrid->tree();
    auto& zTree = zGrid->tree();

    using RootNodeType = typename FloatTree::RootNodeType;
    using L1NodeType = typename RootNodeType::ChildNodeType;
    using L2NodeType = typename L1NodeType::ChildNodeType;
    using LeafNodeType = typename FloatTree::LeafNodeType;

    const Index L1_LEVEL = L1NodeType::getLevel();
    const Index L1_STRIDE = L1NodeType::getChildDim();
    const Index L2_LEVEL = L2NodeType::getLevel();
    const Index L2_STRIDE = L2NodeType::getChildDim();
    const Index LEAF_LEVEL = LeafNodeType::getLevel();

    // x grid (voxels)   | [0][1][...][6][7]                 [0][1][...][6][7] ...  [0][1][...][6][7][0][1][...][6][7]
    // y grid (l2 tiles  | [       3       ][       9       ]                  ...                   [       4       ]
    // z grid (l1 tiles) | [                           5                       ... ]

    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0), 0.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0), 1.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0), 2.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0), 3.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0), 4.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0), 5.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0), 6.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0), 7.0f);

    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 0, 0, 0), 0.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 1, 0, 0), 1.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 2, 0, 0), 2.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 3, 0, 0), 3.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 4, 0, 0), 4.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 5, 0, 0), 5.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 6, 0, 0), 6.0f);
    xTree.setValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 7, 0, 0), 7.0f);

    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0), 0.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0), 1.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0), 2.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0), 3.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0), 4.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0), 5.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0), 6.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0), 7.0f);

    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 0, 0, 0), 0.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 1, 0, 0), 1.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 2, 0, 0), 2.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 3, 0, 0), 3.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 4, 0, 0), 4.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 5, 0, 0), 5.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 6, 0, 0), 6.0f);
    xTree.setValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 7, 0, 0), 7.0f);

    yTree.addTile(L2_LEVEL, Coord(0 * L1_STRIDE + 0 * L2_STRIDE, 0, 0), 3.0f, true);
    yTree.addTile(L2_LEVEL, Coord(0 * L1_STRIDE + 1 * L2_STRIDE, 0, 0), 9.0f, true);
    yTree.addTile(L2_LEVEL, Coord(1 * L1_STRIDE + 1 * L2_STRIDE, 0, 0), 4.0f, true);

    zTree.addTile(L1_LEVEL, Coord(0 * L1_STRIDE, 0, 0), 5.0f, true);

    auto vectorGrid = tools::vectorFromScalar(*xGrid, *yGrid, *zGrid);
    auto& vectorTree = vectorGrid->tree();

    EXPECT_EQ(vectorGrid->background(), Vec3f(-0.1f, -0.2f, -0.3f));

    // Footprint of L1 tile -1, L2 tile -1
    EXPECT_EQ(vectorTree.getValue(Coord(-1 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE - 1 * L2_STRIDE + 0, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    // Footprint of L1 tile 0, L2 tile +0
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), Vec3f(0.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0)), Vec3f(1.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0)), Vec3f(2.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0)), Vec3f(3.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0)), Vec3f(4.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0)), Vec3f(5.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0)), Vec3f(6.0f, 3.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0)), Vec3f(7.0f, 3.0f, 5.0f));

    // Footprint of L1 tile 0, L2 tile +1
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 0, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 1, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 2, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 3, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 4, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 5, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 6, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 7, 0, 0)), Vec3f(-0.1f, 9.0f, 5.0f));

    // Footprint of L1 tile 0, L2 tile +2
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 0, 0, 0)), Vec3f(0.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 1, 0, 0)), Vec3f(1.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 2, 0, 0)), Vec3f(2.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 3, 0, 0)), Vec3f(3.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 4, 0, 0)), Vec3f(4.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 5, 0, 0)), Vec3f(5.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 6, 0, 0)), Vec3f(6.0f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 7, 0, 0)), Vec3f(7.0f, -0.2f, 5.0f));

    // Footprint of L1 tile 0, L2 tile +3
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 0, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 1, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 2, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 3, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 4, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 5, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 6, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));
    EXPECT_EQ(vectorTree.getValue(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 7, 0, 0)), Vec3f(-0.1f, -0.2f, 5.0f));

    // Footprint of L1 tile 1, L2 tile +0
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), Vec3f(0.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0)), Vec3f(1.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0)), Vec3f(2.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0)), Vec3f(3.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0)), Vec3f(4.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0)), Vec3f(5.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0)), Vec3f(6.0f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0)), Vec3f(7.0f, -0.2f, -0.3f));

    // Footprint of L1 tile 1, L2 tile +1
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 0, 0, 0)), Vec3f(0.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 1, 0, 0)), Vec3f(1.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 2, 0, 0)), Vec3f(2.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 3, 0, 0)), Vec3f(3.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 4, 0, 0)), Vec3f(4.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 5, 0, 0)), Vec3f(5.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 6, 0, 0)), Vec3f(6.0f, 4.0f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 7, 0, 0)), Vec3f(7.0f, 4.0f, -0.3f));

    // Footprint of L1 tile 1, L2 tile +2
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 2 * L2_STRIDE + 0, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));
    EXPECT_EQ(vectorTree.getValue(Coord(1 * L1_STRIDE + 2 * L2_STRIDE + 1, 0, 0)), Vec3f(-0.1f, -0.2f, -0.3f));

    const Index NOT_FOUND_DEPTH = -1;
    const Index L2_DEPTH = 2;
    const Index LEAF_DEPTH = 3;

    // Footprint of L1 tile -1, L2 tile -1
    EXPECT_EQ(vectorTree.getValueDepth(Coord(-1 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), NOT_FOUND_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE - 1 * L2_STRIDE + 0, 0, 0)), NOT_FOUND_DEPTH);

    // Footprint of L1 tile 0, L2 tile +0
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0)), LEAF_DEPTH);

    // Footprint of L1 tile 0, L2 tile +1
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 0, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 1, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 2, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 3, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 4, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 5, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 6, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 1 * L2_STRIDE + 7, 0, 0)), L2_DEPTH);

    // Footprint of L1 tile 0, L2 tile +2
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 0, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 1, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 2, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 3, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 4, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 5, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 6, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 2 * L2_STRIDE + 7, 0, 0)), LEAF_DEPTH);

    // Footprint of L1 tile 0, L2 tile +3
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 0, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 1, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 2, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 3, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 4, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 5, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 6, 0, 0)), L2_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * L1_STRIDE + 3 * L2_STRIDE + 7, 0, 0)), L2_DEPTH);

    // Footprint of L1 tile 1, L2 tile +0
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 0, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 1, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 2, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 3, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 4, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 5, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 6, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 0 * L2_STRIDE + 7, 0, 0)), LEAF_DEPTH);

    // Footprint of L1 tile 1, L2 tile +1
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 0, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 1, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 2, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 3, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 4, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 5, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 6, 0, 0)), LEAF_DEPTH);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * L1_STRIDE + 1 * L2_STRIDE + 7, 0, 0)), LEAF_DEPTH);
}

TEST_F(TestVectorFromScalar, testMergeIntGrids)
{
    // TODO
}
