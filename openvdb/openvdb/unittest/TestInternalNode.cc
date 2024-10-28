
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tree/InternalNode.h>
#include <openvdb/openvdb.h>
#include <gtest/gtest.h>

using namespace openvdb;
using namespace openvdb::tree;


class TestInternalNode: public ::testing::Test
{
};


TEST_F(TestInternalNode, test)
{
    const Coord c0(1000, 1000, 1000);
    const Coord c1(896, 896, 896);

    using LeafNodeType = openvdb::tree::LeafNode<float,3>;
    using InternalNodeType = openvdb::tree::InternalNode<LeafNodeType,4>;
    using ChildType = LeafNodeType;

    { // test inserting child nodes directly and indirectly
        Coord c2 = c1.offsetBy(8,0,0);
        Coord c3 = c1.offsetBy(16,16,16);

        InternalNodeType internalNode(c1, 0.0f);
        internalNode.touchLeaf(c2);
        internalNode.touchLeaf(c3);

        EXPECT_EQ(Index(2), internalNode.leafCount());
        EXPECT_EQ(Index32(2), internalNode.childCount());
        EXPECT_TRUE(!internalNode.hasActiveTiles());

        { // verify c0 and c1 are the root node coordinates
            auto childIter = internalNode.cbeginChildOn();
            EXPECT_EQ(c2, childIter.getCoord());
            ++childIter;
            EXPECT_EQ(c3, childIter.getCoord());
        }

        // copy the internal node
        InternalNodeType internalNodeCopy(internalNode);

        // steal the internal node children leaving it empty again
        std::vector<ChildType*> children;
        internalNode.stealNodes(children, 0.0f, false);
        EXPECT_EQ(Index(0), internalNode.leafCount());
        EXPECT_EQ(Index32(0), internalNode.childCount());

        // insert the root node children directly
        for (ChildType* child : children) {
            internalNode.addChild(child);
        }
        EXPECT_EQ(Index(2), internalNode.leafCount());
        EXPECT_EQ(Index32(2), internalNode.childCount());

        { // verify the coordinates of the root node children
            auto childIter = internalNode.cbeginChildOn();
            EXPECT_EQ(c2, childIter.getCoord());
            ++childIter;
            EXPECT_EQ(c3, childIter.getCoord());
        }
    }

    { // test inserting a tile and replacing with a child node
        InternalNodeType internalNode(c1, 0.0f);
        EXPECT_TRUE(!internalNode.hasActiveTiles());
        EXPECT_EQ(Index(0), internalNode.leafCount());
        EXPECT_EQ(Index32(0), internalNode.childCount());

        // add a tile
        internalNode.addTile(Index(0), /*value=*/1.0f, /*state=*/true);
        EXPECT_TRUE(internalNode.hasActiveTiles());
        EXPECT_EQ(Index(0), internalNode.leafCount());
        EXPECT_EQ(Index32(0), internalNode.childCount());

        // replace the tile with a child node
        EXPECT_TRUE(internalNode.addChild(new ChildType(c1, 2.0f)));
        EXPECT_TRUE(!internalNode.hasActiveTiles());
        EXPECT_EQ(Index(1), internalNode.leafCount());
        EXPECT_EQ(Index32(1), internalNode.childCount());
        EXPECT_EQ(c1, internalNode.cbeginChildOn().getCoord());
        EXPECT_NEAR(internalNode.cbeginChildOn()->getValue(0), 2.0f, /*tolerance=*/0.0);

        // replace the child node with another child node
        EXPECT_TRUE(internalNode.addChild(new ChildType(c1, 3.0f)));
        EXPECT_NEAR(internalNode.cbeginChildOn()->getValue(0), 3.0f, /*tolerance=*/0.0);
    }

    { // test inserting child nodes that do and do not belong to the internal node
        InternalNodeType internalNode(c1, 0.0f);

        // succeed if child belongs to this internal node
        EXPECT_TRUE(internalNode.addChild(new ChildType(c0.offsetBy(8,0,0))));
        EXPECT_TRUE(internalNode.probeLeaf(c0.offsetBy(8,0,0)));
        Index index1 = internalNode.coordToOffset(c0);
        Index index2 = internalNode.coordToOffset(c0.offsetBy(8,0,0));
        EXPECT_TRUE(!internalNode.isChildMaskOn(index1));
        EXPECT_TRUE(internalNode.isChildMaskOn(index2));

        // fail otherwise
        auto* child = new ChildType(c0.offsetBy(8000,0,0));
        EXPECT_TRUE(!internalNode.addChild(child));
        delete child;
    }

    { // test transient data
        InternalNodeType internalNode(c1, 0.0f);
        EXPECT_EQ(Index32(0), internalNode.transientData());
        internalNode.setTransientData(Index32(5));
        EXPECT_EQ(Index32(5), internalNode.transientData());
        InternalNodeType internalNode2(internalNode);
        EXPECT_EQ(Index32(5), internalNode2.transientData());
        InternalNodeType internalNode3 = internalNode;
        EXPECT_EQ(Index32(5), internalNode3.transientData());
    }
}
