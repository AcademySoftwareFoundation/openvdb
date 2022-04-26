// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/math/Transform.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>

#include <gtest/gtest.h>

#include <set>


class TestLeafOrigin: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
};


////////////////////////////////////////


TEST_F(TestLeafOrigin, test)
{
    using namespace openvdb;

    std::set<Coord> indices;
    indices.insert(Coord( 0,   0,  0));
    indices.insert(Coord( 1,   0,  0));
    indices.insert(Coord( 0, 100,  8));
    indices.insert(Coord(-9,   0,  8));
    indices.insert(Coord(32,   0, 16));
    indices.insert(Coord(33,  -5, 16));
    indices.insert(Coord(42,  17, 35));
    indices.insert(Coord(43,  17, 64));

    FloatTree tree(/*bg=*/256.0);
    std::set<Coord>::iterator iter = indices.begin();
    for ( ; iter != indices.end(); ++iter) tree.setValue(*iter, 1.0);

    for (FloatTree::LeafCIter leafIter = tree.cbeginLeaf(); leafIter; ++leafIter) {
        const Int32 mask = ~((1 << leafIter->log2dim()) - 1);
        const Coord leafOrigin = leafIter->origin();
        for (FloatTree::LeafNodeType::ValueOnCIter valIter = leafIter->cbeginValueOn();
            valIter; ++valIter)
        {
            Coord xyz = valIter.getCoord();
            EXPECT_EQ(leafOrigin, xyz & mask);

            iter = indices.find(xyz);
            EXPECT_TRUE(iter != indices.end());
            indices.erase(iter);
        }
    }
    EXPECT_TRUE(indices.empty());
}


TEST_F(TestLeafOrigin, test2Values)
{
    using namespace openvdb;

    FloatGrid::Ptr grid = createGrid<FloatGrid>(/*bg=*/1.0f);
    FloatTree& tree = grid->tree();

    tree.setValue(Coord(0, 0, 0), 5);
    tree.setValue(Coord(100, 0, 0), 6);

    grid->setTransform(math::Transform::createLinearTransform(0.1));

    FloatTree::LeafCIter iter = tree.cbeginLeaf();
    EXPECT_EQ(Coord(0, 0, 0), iter->origin());
    ++iter;
    EXPECT_EQ(Coord(96, 0, 0), iter->origin());
}

TEST_F(TestLeafOrigin, testGetValue)
{
    const openvdb::Coord c0(0,-10,0), c1(100,13,0);
    const float v0=5.0f, v1=6.0f, v2=1.0f;
    openvdb::FloatTree::Ptr tree(new openvdb::FloatTree(v2));

    tree->setValue(c0, v0);
    tree->setValue(c1, v1);

    openvdb::FloatTree::LeafCIter iter = tree->cbeginLeaf();
    EXPECT_EQ(v0, iter->getValue(c0));
    ++iter;
    EXPECT_EQ(v1, iter->getValue(c1));
}
