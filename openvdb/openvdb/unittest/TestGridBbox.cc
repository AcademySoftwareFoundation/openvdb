// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/Types.h>
#include <openvdb/Exceptions.h>

#include <gtest/gtest.h>


class TestGridBbox: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestGridBbox, testLeafBbox)
{
    openvdb::FloatTree tree(/*fillValue=*/256.0f);

    openvdb::CoordBBox bbox;
    EXPECT_TRUE(!tree.evalLeafBoundingBox(bbox));

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(  0,  9,   9), 2.0);
    tree.setValue(openvdb::Coord(100, 35, 800), 2.5);

    // Coordinates in CoordBBox are inclusive!
    EXPECT_TRUE(tree.evalLeafBoundingBox(bbox));
    EXPECT_EQ(openvdb::Coord(0,        8,     8), bbox.min());
    EXPECT_EQ(openvdb::Coord(104-1, 40-1, 808-1), bbox.max());

    // Test negative coordinates.
    tree.setValue(openvdb::Coord(-100, -35, -800), 2.5);

    EXPECT_TRUE(tree.evalLeafBoundingBox(bbox));
    EXPECT_EQ(openvdb::Coord(-104,   -40,  -800), bbox.min());
    EXPECT_EQ(openvdb::Coord(104-1, 40-1, 808-1), bbox.max());

    // Clear the tree without trimming.
    tree.setValueOff(openvdb::Coord(  0,  9,   9));
    tree.setValueOff(openvdb::Coord(100, 35, 800));
    tree.setValueOff(openvdb::Coord(-100, -35, -800));
    EXPECT_TRUE(!tree.evalLeafBoundingBox(bbox));
}


TEST_F(TestGridBbox, testGridBbox)
{
    openvdb::FloatTree tree(/*fillValue=*/256.0f);

    openvdb::CoordBBox bbox;
    EXPECT_TRUE(!tree.evalActiveVoxelBoundingBox(bbox));

    // Add values to buffer zero.
    tree.setValue(openvdb::Coord(  1,  0,   0), 1.5);
    tree.setValue(openvdb::Coord(  0, 12,   8), 2.0);
    tree.setValue(openvdb::Coord(  1, 35, 800), 2.5);
    tree.setValue(openvdb::Coord(100,  0,  16), 3.0);
    tree.setValue(openvdb::Coord(  1,  0,  16), 3.5);

    // Coordinates in CoordBBox are inclusive!
    EXPECT_TRUE(tree.evalActiveVoxelBoundingBox(bbox));
    EXPECT_EQ(openvdb::Coord(  0,  0,   0), bbox.min());
    EXPECT_EQ(openvdb::Coord(100, 35, 800), bbox.max());

    // Test negative coordinates.
    tree.setValue(openvdb::Coord(-100, -35, -800), 2.5);

    EXPECT_TRUE(tree.evalActiveVoxelBoundingBox(bbox));
    EXPECT_EQ(openvdb::Coord(-100,   -35,  -800), bbox.min());
    EXPECT_EQ(openvdb::Coord(100, 35, 800), bbox.max());

    // Clear the tree without trimming.
    tree.setValueOff(openvdb::Coord(  1,  0,   0));
    tree.setValueOff(openvdb::Coord(  0, 12,   8));
    tree.setValueOff(openvdb::Coord(  1, 35, 800));
    tree.setValueOff(openvdb::Coord(100,  0,  16));
    tree.setValueOff(openvdb::Coord(  1,  0,  16));
    tree.setValueOff(openvdb::Coord(-100, -35, -800));
    EXPECT_TRUE(!tree.evalActiveVoxelBoundingBox(bbox));
}
