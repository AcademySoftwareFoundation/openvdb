// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/tree/RootNode.h>
#include <openvdb/openvdb.h>

#include <gtest/gtest.h>

using namespace openvdb;
using namespace tree;

class TestRoot: public ::testing::Test
{
};


TEST_F(TestRoot, test)
{
    using RootNodeType = FloatTree::RootNodeType;
    using ChildType = RootNodeType::ChildNodeType;
    const Coord c0(0,0,0), c1(49152, 16384, 28672);

    { // test inserting child nodes directly and indirectly
        RootNodeType root(0.0f);
        EXPECT_TRUE(root.empty());
        EXPECT_EQ(Index32(0), root.childCount());

        // populate the tree by inserting the two leaf nodes containing c0 and c1
        root.touchLeaf(c0);
        root.touchLeaf(c1);
        EXPECT_EQ(Index(2), root.getTableSize());
        EXPECT_EQ(Index32(2), root.childCount());
        EXPECT_TRUE(!root.hasActiveTiles());

        { // verify c0 and c1 are the root node coordinates
            auto rootIter = root.cbeginChildOn();
            EXPECT_EQ(c0, rootIter.getCoord());
            ++rootIter;
            EXPECT_EQ(c1, rootIter.getCoord());
        }

        // copy the root node
        RootNodeType rootCopy(root);

        // steal the root node children leaving the root node empty again
        std::vector<ChildType*> children;
        root.stealNodes(children);
        EXPECT_TRUE(root.empty());

        // insert the root node children directly
        for (ChildType* child : children) {
            root.addChild(child);
        }
        EXPECT_EQ(Index(2), root.getTableSize());
        EXPECT_EQ(Index32(2), root.childCount());

        { // verify the coordinates of the root node children
            auto rootIter = root.cbeginChildOn();
            EXPECT_EQ(c0, rootIter.getCoord());
            ++rootIter;
            EXPECT_EQ(c1, rootIter.getCoord());
        }
    }

    { // test inserting tiles and replacing them with child nodes
        RootNodeType root(0.0f);
        EXPECT_TRUE(root.empty());

        // no-op
        root.addChild(nullptr);

        // populate the root node by inserting tiles
        root.addTile(c0, /*value=*/1.0f, /*state=*/true);
        root.addTile(c1, /*value=*/2.0f, /*state=*/true);
        EXPECT_EQ(Index(2), root.getTableSize());
        EXPECT_EQ(Index32(0), root.childCount());
        EXPECT_TRUE(root.hasActiveTiles());
        EXPECT_NEAR(root.getValue(c0), 1.0f, /*tolerance=*/0.0);
        EXPECT_NEAR(root.getValue(c1), 2.0f, /*tolerance=*/0.0);

        // insert child nodes with the same coordinates
        root.addChild(new ChildType(c0, 3.0f));
        root.addChild(new ChildType(c1, 4.0f));

        // insert a new child at c0
        root.addChild(new ChildType(c0, 5.0f));

        // verify active tiles have been replaced by child nodes
        EXPECT_EQ(Index(2), root.getTableSize());
        EXPECT_EQ(Index32(2), root.childCount());
        EXPECT_TRUE(!root.hasActiveTiles());

        { // verify the coordinates of the root node children
            auto rootIter = root.cbeginChildOn();
            EXPECT_EQ(c0, rootIter.getCoord());
            EXPECT_NEAR(root.getValue(c0), 5.0f, /*tolerance=*/0.0);
            ++rootIter;
            EXPECT_EQ(c1, rootIter.getCoord());
        }
    }

    { // test transient data
        RootNodeType rootNode(0.0f);
        EXPECT_EQ(Index32(0), rootNode.transientData());
        rootNode.setTransientData(Index32(5));
        EXPECT_EQ(Index32(5), rootNode.transientData());
        RootNodeType rootNode2(rootNode);
        EXPECT_EQ(Index32(5), rootNode2.transientData());
        RootNodeType rootNode3 = rootNode;
        EXPECT_EQ(Index32(5), rootNode3.transientData());
    }
}

TEST_F(TestRoot, testProbe)
{
    using RootNode = FloatTree::RootNodeType;

    RootNode root(1.0f);

    root.addTile(Coord(1, 2, 3), 2.0f, true);
    root.addTile(Coord(4096, 2, 3), 3.0f, false);

    auto* child = new RootNode::ChildNodeType(Coord(0, 0, 4096), 5.0f, true);
    EXPECT_TRUE(root.addChild(child)); // always returns true

    { // probeNode, probeConstNode
        auto* node1 = root.probeNode<RootNode::ChildNodeType>(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node1));
        auto* node2 = root.probeNode<RootNode::ChildNodeType>(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node2));
        const RootNode& constRoot = root;
        auto* node3 = constRoot.probeNode<RootNode::ChildNodeType>(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node3));
        auto* node4 = constRoot.probeNode<RootNode::ChildNodeType>(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node4));
        auto* node5 = root.probeConstNode<RootNode::ChildNodeType>(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node5));
        auto* node6 = root.probeConstNode<RootNode::ChildNodeType>(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node6));
    }

    { // probeChild, probeConstChild
        auto* node1 = root.probeChild(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node1));
        auto* node2 = root.probeChild(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node2));
        const RootNode& constRoot = root;
        auto* node3 = constRoot.probeChild(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node3));
        auto* node4 = constRoot.probeChild(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node4));
        auto* node5 = root.probeConstChild(Coord(0, 0, 4096));
        EXPECT_TRUE(bool(node5));
        auto* node6 = root.probeConstChild(Coord(4096, 0, 0));
        EXPECT_FALSE(bool(node6));
    }

    RootNode::ChildNodeType* childPtr = nullptr;
    const RootNode::ChildNodeType* constChildPtr = nullptr;
    float value = -1.0f;
    bool active = false;

    { // probe, probeConst - child
        bool keyExists = root.probe(Coord(0, 0, 4096), childPtr, value, active);
        EXPECT_TRUE(keyExists);
        EXPECT_TRUE(bool(childPtr));
        childPtr = nullptr;
        keyExists = root.probe(Coord(0, 10, 4096), childPtr, value, active);
        EXPECT_TRUE(keyExists);
        EXPECT_TRUE(bool(childPtr));
        childPtr = nullptr;
        EXPECT_FALSE(root.probe(Coord(4096, 4096, 4096), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));

        const RootNode& constRoot = root;
        keyExists = constRoot.probe(Coord(0, 0, 4096), constChildPtr, value, active);
        EXPECT_TRUE(keyExists);
        EXPECT_TRUE(bool(constChildPtr));
        constChildPtr = nullptr;
        EXPECT_FALSE(root.probe(Coord(4096, 4096, 4096), constChildPtr, value, active));
        EXPECT_FALSE(bool(childPtr));

        keyExists = root.probeConst(Coord(0, 0, 4096), constChildPtr, value, active);
        EXPECT_TRUE(keyExists);
        EXPECT_TRUE(bool(constChildPtr));
        constChildPtr = nullptr;
        EXPECT_FALSE(root.probeConst(Coord(4096, 4096, 4096), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
    }

    { // probe, probeConst - tile
        EXPECT_TRUE(root.probe(Coord(0, 0, 0), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));
        EXPECT_EQ(value, 2.0f);
        EXPECT_EQ(active, true);
        value = -1.0f;
        EXPECT_TRUE(root.probe(Coord(4096, 0, 0), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));
        EXPECT_EQ(value, 3.0f);
        EXPECT_EQ(active, false);
        EXPECT_FALSE(root.probe(Coord(4096, 4096, 4096), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));

        const RootNode& constRoot = root;
        EXPECT_TRUE(root.probe(Coord(0, 0, 0), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));
        EXPECT_EQ(value, 2.0f);
        EXPECT_EQ(active, true);
        value = -1.0f;
        EXPECT_TRUE(root.probe(Coord(4096, 0, 0), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));
        EXPECT_EQ(value, 3.0f);
        EXPECT_EQ(active, false);
        EXPECT_FALSE(root.probe(Coord(4096, 4096, 4096), childPtr, value, active));
        EXPECT_FALSE(bool(childPtr));

        EXPECT_TRUE(root.probeConst(Coord(0, 0, 0), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
        EXPECT_EQ(value, 2.0f);
        EXPECT_EQ(active, true);
        value = -1.0f;
        EXPECT_TRUE(root.probeConst(Coord(4096, 0, 0), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
        EXPECT_EQ(value, 3.0f);
        EXPECT_EQ(active, false);
        EXPECT_FALSE(root.probeConst(Coord(4096, 4096, 4096), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
    }
}
