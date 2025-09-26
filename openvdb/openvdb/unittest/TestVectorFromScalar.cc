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
    // index  | 0    1    2    3    4    5    6    7
    // -------|---------------------------------------
    // x grid |    [1.1]     [3.1]     [5.1]     [7.1]
    // y grid |         [2.2][3.2]          [6.2][7.2]
    // z grid |                   [4.3][5.3][6.3][7.3]

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

    EXPECT_EQ(vectorTree.getValueDepth(Coord(0 * ROOT_STRIDE, 0, 0)), -1);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(1 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(2 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(3 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(4 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(5 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(6 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(7 * ROOT_STRIDE, 0, 0)), 0);
    EXPECT_EQ(vectorTree.getValueDepth(Coord(8 * ROOT_STRIDE, 0, 0)), -1);
}
