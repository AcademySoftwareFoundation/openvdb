
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

TEST_F(TestInternalNode, testProbe)
{
    using RootNode = FloatTree::RootNodeType;
    using InternalNode = RootNode::ChildNodeType;

    const Coord ijk(0, 0, 4096);
    InternalNode internalNode(ijk, 1.0f);

    internalNode.addTile(32, 3.0f, true); // (0, 128, 4096)
    internalNode.addTile(33, 4.0f, true); // (0, 128, 4224)

    auto* child = new InternalNode::ChildNodeType(Coord(0, 256, 4096), 5.0f, true);
    EXPECT_TRUE(internalNode.addChild(child)); // always returns true

    { // probeNode, probeConstNode
        auto* node1 = internalNode.probeNode<InternalNode::ChildNodeType>(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node1));
        auto* node2 = internalNode.probeNode<InternalNode::ChildNodeType>(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node2));
        const InternalNode& constInternalNode = internalNode;
        auto* node3 = constInternalNode.probeNode<InternalNode::ChildNodeType>(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node3));
        auto* node4 = constInternalNode.probeNode<InternalNode::ChildNodeType>(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node4));
        auto* node5 = internalNode.probeConstNode<InternalNode::ChildNodeType>(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node5));
        auto* node6 = internalNode.probeConstNode<InternalNode::ChildNodeType>(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node6));
    }

    { // probeChild, probeConstChild
        auto* node1 = internalNode.probeChild(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node1));
        auto* node2 = internalNode.probeChild(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node2));
        const InternalNode& constInternalNode = internalNode;
        auto* node3 = constInternalNode.probeChild(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node3));
        auto* node4 = constInternalNode.probeChild(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node4));
        auto* node5 = internalNode.probeConstChild(Coord(0, 256, 4096));
        EXPECT_TRUE(bool(node5));
        auto* node6 = internalNode.probeConstChild(Coord(0, 128, 4096));
        EXPECT_FALSE(bool(node6));
    }

    { // probeChildUnsafe, probeConstChildUnsafe
        auto* node1 = internalNode.probeChildUnsafe(64);
        EXPECT_TRUE(bool(node1));
        auto* node2 = internalNode.probeChildUnsafe(33);
        EXPECT_FALSE(bool(node2));
        const InternalNode& constInternalNode = internalNode;
        auto* node3 = constInternalNode.probeChildUnsafe(64);
        EXPECT_TRUE(bool(node3));
        auto* node4 = constInternalNode.probeChildUnsafe(33);
        EXPECT_FALSE(bool(node4));
        auto* node5 = internalNode.probeConstChildUnsafe(64);
        EXPECT_TRUE(bool(node5));
        auto* node6 = internalNode.probeConstChildUnsafe(33);
        EXPECT_FALSE(bool(node6));
    }

    float value = -1.0f;
    bool active = false;

    { // probeChild, probeConstChild with value and active status
        auto* node1 = internalNode.probeChild(Coord(0, 256, 4096), value, active);
        EXPECT_TRUE(bool(node1));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node2 = internalNode.probeChild(Coord(0, 128, 4096), value, active);
        EXPECT_FALSE(bool(node2));
        EXPECT_EQ(value, 3.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
        const InternalNode& constInternalNode = internalNode;
        auto* node3 = constInternalNode.probeChild(Coord(0, 256, 4096), value, active);
        EXPECT_TRUE(bool(node3));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node4 = constInternalNode.probeChild(Coord(0, 128, 4096), value, active);
        EXPECT_FALSE(bool(node4));
        EXPECT_EQ(value, 3.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
        auto* node5 = internalNode.probeConstChild(Coord(0, 256, 4096), value, active);
        EXPECT_TRUE(bool(node5));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node6 = internalNode.probeConstChild(Coord(0, 128, 4096), value, active);
        EXPECT_FALSE(bool(node6));
        EXPECT_EQ(value, 3.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
    }

    { // probeChildUnsafe, probeConstChildUnsafe with value and active status
        auto* node1 = internalNode.probeChildUnsafe(64, value, active);
        EXPECT_TRUE(bool(node1));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node2 = internalNode.probeChildUnsafe(33, value, active);
        EXPECT_FALSE(bool(node2));
        EXPECT_EQ(value, 4.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
        const InternalNode& constInternalNode = internalNode;
        auto* node3 = constInternalNode.probeChildUnsafe(64, value, active);
        EXPECT_TRUE(bool(node3));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node4 = constInternalNode.probeChildUnsafe(33, value, active);
        EXPECT_FALSE(bool(node4));
        EXPECT_EQ(value, 4.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
        auto* node5 = internalNode.probeConstChildUnsafe(64, value, active);
        EXPECT_TRUE(bool(node5));
        EXPECT_EQ(value, -1.0f);
        EXPECT_FALSE(active);
        auto* node6 = internalNode.probeConstChildUnsafe(33, value, active);
        EXPECT_FALSE(bool(node6));
        EXPECT_EQ(value, 4.0f); value = -1.0f;
        EXPECT_TRUE(active); active = false;
    }
}
