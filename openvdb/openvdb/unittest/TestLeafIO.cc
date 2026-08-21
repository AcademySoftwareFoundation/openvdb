// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/Types.h>
#include <openvdb/io/io.h>
#include <gtest/gtest.h>

#include <cctype> // for toupper()
#include <iostream>
#include <sstream>

template<typename T>
class TestLeafIO
{
public:
    static void testBuffer();
    static void testTreeIO();
};

template<typename T>
void
TestLeafIO<T>::testBuffer()
{
    using LeafT = openvdb::tree::LeafNode<T, 3>;
    LeafT leaf(openvdb::Coord(0, 0, 0));
    const openvdb::Coord origin = leaf.origin();

    leaf.setValueOn(openvdb::Coord(0, 1, 0), T(1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), T(1));

    // read and write topology to disk

    {
        // create a grid with the leaf for topology testing
        typedef openvdb::Grid<openvdb::tree::Tree<openvdb::tree::RootNode<
            openvdb::tree::InternalNode<openvdb::tree::InternalNode<LeafT, 4>, 5>>>> GridType;
        if (!GridType::isRegistered()) GridType::registerGrid();

        typename GridType::Ptr grid = GridType::create();
        grid->setName("leaf_io");
        grid->tree().addLeaf(new LeafT(leaf));

        openvdb::GridCPtrVec grids;
        grids.push_back(grid);

        // write to file
        {
            openvdb::io::File file("leaf_io.vdb");
            file.write(grids);
            file.close();
        }

        // read grid from file
        typename GridType::Ptr gridFromDisk;
        {
            openvdb::io::File file("leaf_io.vdb");
            file.open();
            openvdb::GridBase::Ptr baseGrid = file.readGrid("leaf_io");
            file.close();

            gridFromDisk = openvdb::gridPtrCast<GridType>(baseGrid);
        }

        LeafT* leaf2 = gridFromDisk->tree().probeLeaf(origin);
        EXPECT_TRUE(leaf2);

        // check topology and values match

        EXPECT_NEAR(T(1), leaf2->getValue(openvdb::Coord(0, 1, 0)), /*tolerance=*/0);
        EXPECT_NEAR(T(1), leaf2->getValue(openvdb::Coord(1, 0, 0)), /*tolerance=*/0);
        EXPECT_TRUE(leaf2->onVoxelCount() == 2);

        remove("leaf_io.vdb");
    }
}


template<typename T>
void
TestLeafIO<T>::testTreeIO()
{
    using LeafT = openvdb::tree::LeafNode<T, 3>;
    LeafT leaf(openvdb::Coord(0, 0, 0));

    leaf.setValueOn(openvdb::Coord(0, 1, 0), T(1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), T(1));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOn(openvdb::Coord(0, 1, 0), T(0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1), T(1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    EXPECT_NEAR(T(1), leaf.getValue(openvdb::Coord(0, 1, 0)), /*tolerance=*/0);
    EXPECT_NEAR(T(1), leaf.getValue(openvdb::Coord(1, 0, 0)), /*tolerance=*/0);

    EXPECT_TRUE(leaf.onVoxelCount() == 2);
}


class TestLeafIOTest: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


TEST_F(TestLeafIOTest, testBufferInt) { TestLeafIO<int>::testBuffer(); }
TEST_F(TestLeafIOTest, testBufferFloat) { TestLeafIO<float>::testBuffer(); }
TEST_F(TestLeafIOTest, testBufferDouble) { TestLeafIO<double>::testBuffer(); }
TEST_F(TestLeafIOTest, testBufferBool) { TestLeafIO<bool>::testBuffer(); }
TEST_F(TestLeafIOTest, testBufferByte) { TestLeafIO<openvdb::Byte>::testBuffer(); }


TEST_F(TestLeafIOTest, testBufferVec3R)
{
    using LeafT = openvdb::tree::LeafNode<openvdb::Vec3R, 3>;
    LeafT leaf(openvdb::Coord(0, 0, 0));
    const openvdb::Coord origin = leaf.origin();

    leaf.setValueOn(openvdb::Coord(0, 1, 0), openvdb::Vec3R(1, 1, 1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), openvdb::Vec3R(1, 1, 1));

    // read and write topology to disk

    {
        // create a grid with the leaf for topology testing
        typedef openvdb::Grid<openvdb::tree::Tree<openvdb::tree::RootNode<
            openvdb::tree::InternalNode<openvdb::tree::InternalNode<LeafT, 4>, 5>>>> GridType;
        GridType::Ptr grid = GridType::create();
        grid->setName("leaf_vec3r");
        grid->tree().addLeaf(new LeafT(leaf));

        openvdb::GridCPtrVec grids;
        grids.push_back(grid);

        // write to file
        {
            openvdb::io::File file("leaf_vec3r.vdb");
            file.write(grids);
            file.close();
        }

        // read grid from file
        GridType::Ptr gridFromDisk;
        {
            openvdb::io::File file("leaf_vec3r.vdb");
            file.open();
            openvdb::GridBase::Ptr baseGrid = file.readGrid("leaf_vec3r");
            file.close();

            gridFromDisk = openvdb::gridPtrCast<GridType>(baseGrid);
        }

        LeafT* leaf2 = gridFromDisk->tree().probeLeaf(origin);
        EXPECT_TRUE(leaf2);

        // check topology and values match

        EXPECT_TRUE(leaf2->getValue(openvdb::Coord(0, 1, 0)) == openvdb::Vec3R(1, 1, 1));
        EXPECT_TRUE(leaf2->getValue(openvdb::Coord(1, 0, 0)) == openvdb::Vec3R(1, 1, 1));
        EXPECT_TRUE(leaf2->onVoxelCount() == 2);

        remove("leaf_vec3r.vdb");
    }
}

TEST_F(TestLeafIOTest, testTreeIOInt) { TestLeafIO<int>::testTreeIO(); }
TEST_F(TestLeafIOTest, testTreeIOFloat) { TestLeafIO<float>::testTreeIO(); }
TEST_F(TestLeafIOTest, testTreeIODouble) { TestLeafIO<double>::testTreeIO(); }
TEST_F(TestLeafIOTest, testTreeIOBool) { TestLeafIO<bool>::testTreeIO(); }
TEST_F(TestLeafIOTest, testTreeIOByte) { TestLeafIO<openvdb::Byte>::testTreeIO(); }


TEST_F(TestLeafIOTest, testTreeIOVec3R)
{
    using LeafT = openvdb::tree::LeafNode<openvdb::Vec3R, 3>;
    LeafT leaf(openvdb::Coord(0, 0, 0));

    leaf.setValueOn(openvdb::Coord(0, 1, 0), openvdb::Vec3R(1, 1, 1));
    leaf.setValueOn(openvdb::Coord(1, 0, 0), openvdb::Vec3R(1, 1, 1));

    std::ostringstream ostr(std::ios_base::binary);

    leaf.writeBuffers(ostr);

    leaf.setValueOn(openvdb::Coord(0, 1, 0), openvdb::Vec3R(0, 0, 0));
    leaf.setValueOn(openvdb::Coord(0, 1, 1), openvdb::Vec3R(1, 1, 1));

    std::istringstream istr(ostr.str(), std::ios_base::binary);
    openvdb::io::setCurrentVersion(istr);

    leaf.readBuffers(istr);

    EXPECT_TRUE(leaf.getValue(openvdb::Coord(0, 1, 0)) == openvdb::Vec3R(1, 1, 1));
    EXPECT_TRUE(leaf.getValue(openvdb::Coord(1, 0, 0)) == openvdb::Vec3R(1, 1, 1));

    EXPECT_TRUE(leaf.onVoxelCount() == 2);
}
