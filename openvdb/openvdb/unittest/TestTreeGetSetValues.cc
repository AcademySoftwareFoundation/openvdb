// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Types.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tools/ValueTransformer.h> // for tools::setValueOnMin() et al.
#include <openvdb/tools/Prune.h>

#include <gtest/gtest.h>


#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);


class TestTreeGetSetValues: public ::testing::Test
{
};


namespace {
typedef openvdb::tree::Tree4<float, 3, 2, 3>::Type Tree323f; // 8^3 x 4^3 x 8^3
}


TEST_F(TestTreeGetSetValues, testGetBackground)
{
    const float background = 256.0f;
    Tree323f tree(background);

    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.background());
}


TEST_F(TestTreeGetSetValues, testGetValues)
{
    Tree323f tree(/*background=*/256.0f);

    tree.setValue(openvdb::Coord(0, 0,  0), 1.0);
    tree.setValue(openvdb::Coord(1, 0,  0), 1.5);
    tree.setValue(openvdb::Coord(0, 0,  8), 2.0);
    tree.setValue(openvdb::Coord(1, 0,  8), 2.5);
    tree.setValue(openvdb::Coord(0, 0, 16), 3.0);
    tree.setValue(openvdb::Coord(1, 0, 16), 3.5);
    tree.setValue(openvdb::Coord(0, 0, 24), 4.0);
    tree.setValue(openvdb::Coord(1, 0, 24), 4.5);

    ASSERT_DOUBLES_EXACTLY_EQUAL(1.0, tree.getValue(openvdb::Coord(0, 0,  0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(1.5, tree.getValue(openvdb::Coord(1, 0,  0)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(2.0, tree.getValue(openvdb::Coord(0, 0,  8)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(2.5, tree.getValue(openvdb::Coord(1, 0,  8)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(3.0, tree.getValue(openvdb::Coord(0, 0, 16)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(3.5, tree.getValue(openvdb::Coord(1, 0, 16)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(4.0, tree.getValue(openvdb::Coord(0, 0, 24)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(4.5, tree.getValue(openvdb::Coord(1, 0, 24)));
}


TEST_F(TestTreeGetSetValues, testSetValues)
{
    using namespace openvdb;

    const float background = 256.0;
    Tree323f tree(background);

    for (int activeTile = 0; activeTile < 2; ++activeTile) {
        if (activeTile) tree.fill(CoordBBox(Coord(0), Coord(31)), background, /*active=*/true);

        tree.setValue(openvdb::Coord(0, 0,  0), 1.0);
        tree.setValue(openvdb::Coord(1, 0,  0), 1.5);
        tree.setValue(openvdb::Coord(0, 0,  8), 2.0);
        tree.setValue(openvdb::Coord(1, 0,  8), 2.5);
        tree.setValue(openvdb::Coord(0, 0, 16), 3.0);
        tree.setValue(openvdb::Coord(1, 0, 16), 3.5);
        tree.setValue(openvdb::Coord(0, 0, 24), 4.0);
        tree.setValue(openvdb::Coord(1, 0, 24), 4.5);

        const int expectedActiveCount = (!activeTile ? 8 : 32 * 32 * 32);
        EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));

        float val = 1.f;
        for (Tree323f::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
            ASSERT_DOUBLES_EXACTLY_EQUAL(val, iter->getValue(openvdb::Coord(0, 0, 0)));
            ASSERT_DOUBLES_EXACTLY_EQUAL(val+0.5, iter->getValue(openvdb::Coord(1, 0, 0)));
            val = val + 1.f;
        }
    }
}


TEST_F(TestTreeGetSetValues, testUnsetValues)
{
    using namespace openvdb;

    const float background = 256.0;
    Tree323f tree(background);

    for (int activeTile = 0; activeTile < 2; ++activeTile) {
        if (activeTile) tree.fill(CoordBBox(Coord(0), Coord(31)), background, /*active=*/true);

        Coord setCoords[8] = {
            Coord(0, 0, 0),
            Coord(1, 0, 0),
            Coord(0, 0, 8),
            Coord(1, 0, 8),
            Coord(0, 0, 16),
            Coord(1, 0, 16),
            Coord(0, 0, 24),
            Coord(1, 0, 24)
        };

        for (int i = 0; i < 8; ++i) {
            tree.setValue(setCoords[i], 1.0);
        }
        const int expectedActiveCount = (!activeTile ? 8 : 32 * 32 * 32);
        EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));

        // Unset some voxels.
        for (int i = 0; i < 8; i += 2) {
            tree.setValueOff(setCoords[i]);
        }
        EXPECT_EQ(expectedActiveCount - 4, int(tree.activeVoxelCount()));

        // Unset some voxels, but change their values.
        for (int i = 0; i < 8; i += 2) {
            tree.setValueOff(setCoords[i], background);
        }
        EXPECT_EQ(expectedActiveCount - 4, int(tree.activeVoxelCount()));
        for (int i = 0; i < 8; i += 2) {
            ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(setCoords[i]));
        }
    }
}


TEST_F(TestTreeGetSetValues, testFill)
{
    using openvdb::CoordBBox;
    using openvdb::Coord;

    const float background = 256.0;
    Tree323f tree(background);

    // Fill from (-2,-2,-2) to (2,2,2) with active value 2.
    tree.fill(CoordBBox(Coord(-2), Coord(2)), 2.0);
    Coord xyz, xyzMin = Coord::max(), xyzMax = Coord::min();
    for (Tree323f::ValueOnCIter iter = tree.cbeginValueOn(); iter; ++iter) {
        xyz = iter.getCoord();
        xyzMin = std::min(xyzMin, xyz);
        xyzMax = std::max(xyz, xyzMax);
        ASSERT_DOUBLES_EXACTLY_EQUAL(2.0, *iter);
    }
    EXPECT_EQ(openvdb::Index64(5*5*5), tree.activeVoxelCount());
    EXPECT_EQ(Coord(-2), xyzMin);
    EXPECT_EQ(Coord( 2), xyzMax);

    // Fill from (1,1,1) to (3,3,3) with active value 3.
    tree.fill(CoordBBox(Coord(1), Coord(3)), 3.0);
    xyzMin = Coord::max(); xyzMax = Coord::min();
    for (Tree323f::ValueOnCIter iter = tree.cbeginValueOn(); iter; ++iter) {
        xyz = iter.getCoord();
        xyzMin = std::min(xyzMin, xyz);
        xyzMax = std::max(xyz, xyzMax);
        const float expectedValue = (xyz[0] >= 1 && xyz[1] >= 1 && xyz[2] >= 1
            && xyz[0] <= 3 && xyz[1] <= 3 && xyz[2] <= 3) ? 3.0 : 2.0;
        ASSERT_DOUBLES_EXACTLY_EQUAL(expectedValue, *iter);
    }
    openvdb::Index64 expectedCount =
          5*5*5  // (-2,-2,-2) to (2,2,2)
        + 3*3*3  // (1,1,1) to (3,3,3)
        - 2*2*2; // (1,1,1) to (2,2,2) overlap
    EXPECT_EQ(expectedCount, tree.activeVoxelCount());
    EXPECT_EQ(Coord(-2), xyzMin);
    EXPECT_EQ(Coord( 3), xyzMax);

    // Fill from (10,10,10) to (20,20,20) with active value 10.
    tree.fill(CoordBBox(Coord(10), Coord(20)), 10.0);
    xyzMin = Coord::max(); xyzMax = Coord::min();
    for (Tree323f::ValueOnCIter iter = tree.cbeginValueOn(); iter; ++iter) {
        xyz = iter.getCoord();
        xyzMin = std::min(xyzMin, xyz);
        xyzMax = std::max(xyz, xyzMax);
        float expectedValue = 2.0;
        if (xyz[0] >= 1 && xyz[1] >= 1 && xyz[2] >= 1
            && xyz[0] <= 3 && xyz[1] <= 3 && xyz[2] <= 3)
        {
            expectedValue = 3.0;
        } else if (xyz[0] >= 10 && xyz[1] >= 10 && xyz[2] >= 10
            && xyz[0] <= 20 && xyz[1] <= 20 && xyz[2] <= 20)
        {
            expectedValue = 10.0;
        }
        ASSERT_DOUBLES_EXACTLY_EQUAL(expectedValue, *iter);
    }
    expectedCount =
          5*5*5     // (-2,-2,-2) to (2,2,2)
        + 3*3*3     // (1,1,1) to (3,3,3)
        - 2*2*2     // (1,1,1) to (2,2,2) overlap
        + 11*11*11; // (10,10,10) to (20,20,20)
    EXPECT_EQ(expectedCount, tree.activeVoxelCount());
    EXPECT_EQ(Coord(-2), xyzMin);
    EXPECT_EQ(Coord(20), xyzMax);

    // "Undo" previous fill from (10,10,10) to (20,20,20).
    tree.fill(CoordBBox(Coord(10), Coord(20)), background, /*active=*/false);
    xyzMin = Coord::max(); xyzMax = Coord::min();
    for (Tree323f::ValueOnCIter iter = tree.cbeginValueOn(); iter; ++iter) {
        xyz = iter.getCoord();
        xyzMin = std::min(xyzMin, xyz);
        xyzMax = std::max(xyz, xyzMax);
        const float expectedValue = (xyz[0] >= 1 && xyz[1] >= 1 && xyz[2] >= 1
            && xyz[0] <= 3 && xyz[1] <= 3 && xyz[2] <= 3) ? 3.0 : 2.0;
        ASSERT_DOUBLES_EXACTLY_EQUAL(expectedValue, *iter);
    }
    expectedCount =
          5*5*5  // (-2,-2,-2) to (2,2,2)
        + 3*3*3  // (1,1,1) to (3,3,3)
        - 2*2*2; // (1,1,1) to (2,2,2) overlap
    EXPECT_EQ(expectedCount, tree.activeVoxelCount());
    EXPECT_EQ(Coord(-2), xyzMin);
    EXPECT_EQ(Coord( 3), xyzMax);

    // The following tests assume a [3,2,3] tree configuration.

    tree.clear();
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount()); // root node

    // Partially fill a single leaf node.
    tree.fill(CoordBBox(Coord(8), Coord(14)), 0.0);
    EXPECT_EQ(openvdb::Index32(1), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    // Completely fill the leaf node, replacing it with a tile.
    tree.fill(CoordBBox(Coord(8), Coord(15)), 0.0);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    {
        const int activeVoxelCount = int(tree.activeVoxelCount());

        // Fill a single voxel of the tile with a different (active) value.
        tree.fill(CoordBBox(Coord(10), Coord(10)), 1.0);
        EXPECT_EQ(openvdb::Index32(1), tree.leafCount());
        EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());
        EXPECT_EQ(activeVoxelCount, int(tree.activeVoxelCount()));
        // Fill the voxel with an inactive value.
        tree.fill(CoordBBox(Coord(10), Coord(10)), 1.0, /*active=*/false);
        EXPECT_EQ(openvdb::Index32(1), tree.leafCount());
        EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());
        EXPECT_EQ(activeVoxelCount - 1, int(tree.activeVoxelCount()));

        // Completely fill the leaf node, replacing it with a tile again.
        tree.fill(CoordBBox(Coord(8), Coord(15)), 0.0);
        EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
        EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());
    }

    // Expand by one voxel, creating seven neighboring leaf nodes.
    tree.fill(CoordBBox(Coord(8), Coord(16)), 0.0);
    EXPECT_EQ(openvdb::Index32(7), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    // Completely fill the internal node containing the tile, replacing it with
    // a tile at the next level of the tree.
    tree.fill(CoordBBox(Coord(0), Coord(31)), 0.0);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(2), tree.nonLeafCount());

    // Expand by one voxel, creating a layer of leaf nodes on three faces.
    tree.fill(CoordBBox(Coord(0), Coord(32)), 0.0);
    EXPECT_EQ(openvdb::Index32(5*5 + 4*5 + 4*4), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(2 + 7), tree.nonLeafCount()); // +7 internal nodes

    // Completely fill the second-level internal node, replacing it with a root-level tile.
    tree.fill(CoordBBox(Coord(0), Coord(255)), 0.0);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount());

    // Repeat, filling with an inactive value.

    tree.clear();
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount()); // root node

    // Partially fill a single leaf node.
    tree.fill(CoordBBox(Coord(8), Coord(14)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(1), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    // Completely fill the leaf node, replacing it with a tile.
    tree.fill(CoordBBox(Coord(8), Coord(15)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    // Expand by one voxel, creating seven neighboring leaf nodes.
    tree.fill(CoordBBox(Coord(8), Coord(16)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(7), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(3), tree.nonLeafCount());

    // Completely fill the internal node containing the tile, replacing it with
    // a tile at the next level of the tree.
    tree.fill(CoordBBox(Coord(0), Coord(31)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(2), tree.nonLeafCount());

    // Expand by one voxel, creating a layer of leaf nodes on three faces.
    tree.fill(CoordBBox(Coord(0), Coord(32)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(5*5 + 4*5 + 4*4), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(2 + 7), tree.nonLeafCount()); // +7 internal nodes

    // Completely fill the second-level internal node, replacing it with a root-level tile.
    tree.fill(CoordBBox(Coord(0), Coord(255)), 0.0, /*active=*/false);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount());

    tree.clear();
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount()); // root node
    EXPECT_TRUE(tree.empty());

    // Partially fill a region with inactive background values.
    tree.fill(CoordBBox(Coord(27), Coord(254)), background, /*active=*/false);
    // Confirm that after pruning, the tree is empty.
    openvdb::tools::prune(tree);
    EXPECT_EQ(openvdb::Index32(0), tree.leafCount());
    EXPECT_EQ(openvdb::Index32(1), tree.nonLeafCount()); // root node
    EXPECT_TRUE(tree.empty());
}


// Verify that setting voxels inside active tiles works correctly.
// In particular, it should preserve the active states of surrounding voxels.
TEST_F(TestTreeGetSetValues, testSetActiveStates)
{
    using namespace openvdb;

    const float background = 256.0;
    Tree323f tree(background);

    const Coord xyz(10);
    const float val = 42.0;
    const int expectedActiveCount = 32 * 32 * 32;

#define RESET_TREE() \
    tree.fill(CoordBBox(Coord(0), Coord(31)), background, /*active=*/true) // create an active tile

    RESET_TREE();
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));

    tree.setValueOff(xyz);
    EXPECT_EQ(expectedActiveCount - 1, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(xyz));

    RESET_TREE();
    tree.setValueOn(xyz);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(xyz));

    RESET_TREE();
    tree.setValueOff(xyz, val);
    EXPECT_EQ(expectedActiveCount - 1, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(val, tree.getValue(xyz));

    RESET_TREE();
    tree.setActiveState(xyz, true);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(xyz));

    RESET_TREE();
    tree.setActiveState(xyz, false);
    EXPECT_EQ(expectedActiveCount - 1, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(background, tree.getValue(xyz));

    RESET_TREE();
    tree.setValueOn(xyz, val);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(val, tree.getValue(xyz));

    RESET_TREE();
    tools::setValueOnMin(tree, xyz, val);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(std::min(val, background), tree.getValue(xyz));

    RESET_TREE();
    tools::setValueOnMax(tree, xyz, val);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(std::max(val, background), tree.getValue(xyz));

    RESET_TREE();
    tools::setValueOnSum(tree, xyz, val);
    EXPECT_EQ(expectedActiveCount, int(tree.activeVoxelCount()));
    ASSERT_DOUBLES_EXACTLY_EQUAL(val + background, tree.getValue(xyz));

#undef RESET_TREE
}


TEST_F(TestTreeGetSetValues, testHasActiveTiles)
{
    Tree323f tree(/*background=*/256.0f);

    EXPECT_TRUE(!tree.hasActiveTiles());

    // Fill from (-2,-2,-2) to (2,2,2) with active value 2.
    tree.fill(openvdb::CoordBBox(openvdb::Coord(-2), openvdb::Coord(2)), 2.0f);
    EXPECT_TRUE(!tree.hasActiveTiles());

    // Fill from (-200,-200,-200) to (-4,-4,-4) with active value 3.
    tree.fill(openvdb::CoordBBox(openvdb::Coord(-200), openvdb::Coord(-4)), 3.0f);
    EXPECT_TRUE(tree.hasActiveTiles());
}
