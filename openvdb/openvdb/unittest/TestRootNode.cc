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

TEST_F(TestRoot, testMap)
{
    using RootNode = FloatTree::RootNodeType;

    { // empty root node
        RootNode root(1.0f);

        // background checks

        EXPECT_EQ(root.background(), 1.0f);
        root.setBackground(2.0f, false);
        EXPECT_EQ(root.background(), 2.0f);
        EXPECT_EQ(root.numBackgroundTiles(), 0);

        // count checks

        EXPECT_TRUE(root.empty());
        EXPECT_FALSE(root.hasActiveTiles());
        EXPECT_EQ(root.getTableSize(), 0);
        EXPECT_EQ(root.leafCount(), 0);
        EXPECT_EQ(root.nonLeafCount(), 1); // root counts as a node
        EXPECT_EQ(root.childCount(), 0);
        EXPECT_EQ(root.tileCount(), 0);
        EXPECT_EQ(root.activeTileCount(), 0);
        EXPECT_EQ(root.inactiveTileCount(), 0);

        EXPECT_EQ(root.onVoxelCount(), 0);
        EXPECT_EQ(root.offVoxelCount(), 0);
        EXPECT_EQ(root.onLeafVoxelCount(), 0);
        EXPECT_EQ(root.offLeafVoxelCount(), 0);
        EXPECT_EQ(root.onTileCount(), 0);

        // bounding box checks

        EXPECT_EQ(root.getMinIndex(), Coord());
        EXPECT_EQ(root.getMaxIndex(), Coord());
        EXPECT_EQ(root.getWidth(), 0);
        EXPECT_EQ(root.getHeight(), 0);
        EXPECT_EQ(root.getDepth(), 0);
        EXPECT_EQ(root.getNodeBoundingBox(), CoordBBox::inf()); // always infinite

        CoordBBox bbox;
        root.evalActiveBoundingBox(bbox);
        EXPECT_EQ(bbox, CoordBBox()); // empty bbox

        root.getIndexRange(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(0))); // zero bbox

        // origin checks

        root.setOrigin(Coord(0, 0, 0));
        EXPECT_THROW(root.setOrigin(Coord(1, 2, 3)), ValueError); // non-zero origins not supported

        // key checks

        EXPECT_EQ(root.getValueDepth(Coord(0, 0, 0)), -1);

        EXPECT_EQ(root.coordToKey(Coord(0, 0, 0)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(1, 2, 3)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(5000, 6000, 7000)), Coord(4096, 4096, 4096));

        EXPECT_FALSE(root.hasKey(Coord(0, 0, 0)));
    }

    { // one active, non-background root node tile
        RootNode root(1.0f);
        root.addTile(Coord(1, 2, 3), 2.0f, true);

        // background checks

        EXPECT_EQ(root.background(), 1.0f);
        EXPECT_EQ(root.numBackgroundTiles(), 0);

        // count checks

        EXPECT_FALSE(root.empty());
        EXPECT_TRUE(root.hasActiveTiles());
        EXPECT_EQ(root.getTableSize(), 1);
        EXPECT_EQ(root.leafCount(), 0);
        EXPECT_EQ(root.nonLeafCount(), 1);
        EXPECT_EQ(root.childCount(), 0);
        EXPECT_EQ(root.tileCount(), 1);
        EXPECT_EQ(root.activeTileCount(), 1);
        EXPECT_EQ(root.inactiveTileCount(), 0);

        Index64 voxels = Index64(4096) * 4096 * 4096;
        EXPECT_EQ(root.onVoxelCount(), voxels);
        EXPECT_EQ(root.offVoxelCount(), 0);
        EXPECT_EQ(root.onLeafVoxelCount(), 0);
        EXPECT_EQ(root.offLeafVoxelCount(), 0);
        EXPECT_EQ(root.onTileCount(), 1);

        // bounding box checks

        EXPECT_EQ(root.getMinIndex(), Coord(0));
        EXPECT_EQ(root.getMaxIndex(), Coord(4095));
        EXPECT_EQ(root.getWidth(), 4095);
        EXPECT_EQ(root.getHeight(), 4095);
        EXPECT_EQ(root.getDepth(), 4095);
        EXPECT_EQ(root.getNodeBoundingBox(), CoordBBox::inf()); // always infinite

        CoordBBox bbox;
        root.evalActiveBoundingBox(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        root.getIndexRange(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        // key checks

        EXPECT_EQ(root.getValueDepth(Coord(0, 0, 0)), 0);

        EXPECT_EQ(root.coordToKey(Coord(0, 0, 0)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(1, 2, 3)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(5000, 6000, 7000)), Coord(4096, 4096, 4096));

        EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
        EXPECT_FALSE(root.hasKey(Coord(1, 2, 3)));

        // erase background tiles

        root.eraseBackgroundTiles();
        EXPECT_EQ(root.getTableSize(), 1);
        EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));

        // clear root

        root.clear();
        EXPECT_EQ(root.getTableSize(), 0);
        EXPECT_FALSE(root.hasKey(Coord(0, 0, 0)));
    }

    { // one inactive, background root node tile
        RootNode root(1.0f);
        root.addTile(Coord(1, 2, 3), 1.0f, false);

        // background checks

        EXPECT_EQ(root.background(), 1.0f);
        EXPECT_EQ(root.numBackgroundTiles(), 1);

        // count checks

        EXPECT_TRUE(root.empty()); // root is empty if it only has inactive background tiles
        EXPECT_FALSE(root.hasActiveTiles());
        EXPECT_EQ(root.getTableSize(), 1);
        EXPECT_EQ(root.leafCount(), 0);
        EXPECT_EQ(root.nonLeafCount(), 1);
        EXPECT_EQ(root.childCount(), 0);
        EXPECT_EQ(root.tileCount(), 1);
        EXPECT_EQ(root.activeTileCount(), 0);
        EXPECT_EQ(root.inactiveTileCount(), 1);

        EXPECT_EQ(root.onVoxelCount(), 0);
        EXPECT_EQ(root.offVoxelCount(), 0);
        EXPECT_EQ(root.onLeafVoxelCount(), 0);
        EXPECT_EQ(root.offLeafVoxelCount(), 0);
        EXPECT_EQ(root.onTileCount(), 0);

        // bounding box checks

        EXPECT_EQ(root.getMinIndex(), Coord(0));
        EXPECT_EQ(root.getMaxIndex(), Coord(4095));
        EXPECT_EQ(root.getWidth(), 4095);
        EXPECT_EQ(root.getHeight(), 4095);
        EXPECT_EQ(root.getDepth(), 4095);
        EXPECT_EQ(root.getNodeBoundingBox(), CoordBBox::inf()); // always infinite

        CoordBBox bbox;
        root.evalActiveBoundingBox(bbox);
        EXPECT_EQ(bbox, CoordBBox()); // empty bbox

        root.getIndexRange(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        // key checks

        EXPECT_EQ(root.getValueDepth(Coord(0, 0, 0)), 0);

        EXPECT_EQ(root.coordToKey(Coord(0, 0, 0)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(1, 2, 3)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(5000, 6000, 7000)), Coord(4096, 4096, 4096));

        EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
        EXPECT_FALSE(root.hasKey(Coord(1, 2, 3)));

        // erase background tiles

        root.eraseBackgroundTiles();
        EXPECT_EQ(root.getTableSize(), 0);
        EXPECT_FALSE(root.hasKey(Coord(0, 0, 0)));
    }

    { // one active, background root node tile
        RootNode root(1.0f);
        root.addTile(Coord(1, 2, 3), 1.0f, true);

        // background checks

        EXPECT_EQ(root.background(), 1.0f);
        EXPECT_EQ(root.numBackgroundTiles(), 0);

        // count checks

        EXPECT_FALSE(root.empty());
        EXPECT_TRUE(root.hasActiveTiles());
        EXPECT_EQ(root.getTableSize(), 1);
        EXPECT_EQ(root.leafCount(), 0);
        EXPECT_EQ(root.nonLeafCount(), 1);
        EXPECT_EQ(root.childCount(), 0);
        EXPECT_EQ(root.tileCount(), 1);
        EXPECT_EQ(root.activeTileCount(), 1);
        EXPECT_EQ(root.inactiveTileCount(), 0);

        Index64 voxels = Index64(4096) * 4096 * 4096;
        EXPECT_EQ(root.onVoxelCount(), voxels);
        EXPECT_EQ(root.offVoxelCount(), 0);
        EXPECT_EQ(root.onLeafVoxelCount(), 0);
        EXPECT_EQ(root.offLeafVoxelCount(), 0);
        EXPECT_EQ(root.onTileCount(), 1);

        // bounding box checks

        EXPECT_EQ(root.getMinIndex(), Coord(0));
        EXPECT_EQ(root.getMaxIndex(), Coord(4095));
        EXPECT_EQ(root.getWidth(), 4095);
        EXPECT_EQ(root.getHeight(), 4095);
        EXPECT_EQ(root.getDepth(), 4095);
        EXPECT_EQ(root.getNodeBoundingBox(), CoordBBox::inf()); // always infinite

        CoordBBox bbox;
        root.evalActiveBoundingBox(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        root.getIndexRange(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        // key checks

        EXPECT_EQ(root.getValueDepth(Coord(0, 0, 0)), 0);

        EXPECT_EQ(root.coordToKey(Coord(0, 0, 0)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(1, 2, 3)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(5000, 6000, 7000)), Coord(4096, 4096, 4096));

        EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
        EXPECT_FALSE(root.hasKey(Coord(1, 2, 3)));
    }

    { // one internal node tile (which implicitly adds a root node child)
        RootNode root(1.0f);
        root.addTile(2, Coord(1, 2, 3), 2.0f, true);

        // count checks

        EXPECT_FALSE(root.empty());
        EXPECT_TRUE(root.hasActiveTiles()); // this method recurses down the tree
        EXPECT_EQ(root.getTableSize(), 1);
        EXPECT_EQ(root.leafCount(), 0);
        EXPECT_EQ(root.nonLeafCount(), 2);
        EXPECT_EQ(root.childCount(), 1);
        EXPECT_EQ(root.tileCount(), 0);
        EXPECT_EQ(root.activeTileCount(), 0);
        EXPECT_EQ(root.inactiveTileCount(), 0);

        Index64 totalVoxels = Index64(4096) * 4096 * 4096;
        Index64 onVoxels = Index64(128) * 128 * 128;
        EXPECT_EQ(root.onVoxelCount(), onVoxels);
        EXPECT_EQ(root.offVoxelCount(), totalVoxels - onVoxels);
        EXPECT_EQ(root.onLeafVoxelCount(), 0);
        EXPECT_EQ(root.offLeafVoxelCount(), 0);
        EXPECT_EQ(root.onTileCount(), 1);

        // bounding box checks

        EXPECT_EQ(root.getMinIndex(), Coord(0));
        EXPECT_EQ(root.getMaxIndex(), Coord(4095));
        EXPECT_EQ(root.getWidth(), 4095);
        EXPECT_EQ(root.getHeight(), 4095);
        EXPECT_EQ(root.getDepth(), 4095);
        EXPECT_EQ(root.getNodeBoundingBox(), CoordBBox::inf()); // always infinite

        CoordBBox bbox;
        root.evalActiveBoundingBox(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(127)));

        root.getIndexRange(bbox);
        EXPECT_EQ(bbox, CoordBBox(Coord(0), Coord(4095)));

        // key checks

        EXPECT_EQ(root.getValueDepth(Coord(0, 0, 0)), 1); // InternalNode 1

        EXPECT_EQ(root.coordToKey(Coord(0, 0, 0)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(1, 2, 3)), Coord(0, 0, 0));
        EXPECT_EQ(root.coordToKey(Coord(5000, 6000, 7000)), Coord(4096, 4096, 4096));

        EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
        EXPECT_FALSE(root.hasKey(Coord(1, 2, 3)));
    }
}

TEST_F(TestRoot, testDelete)
{
    using RootNode = FloatTree::RootNodeType;

    RootNode root(1.0f);

    root.addTile(Coord(1, 2, 3), 2.0f, true);
    root.addTile(Coord(4096, 2, 3), 3.0f, true);

    auto* child = new RootNode::ChildNodeType(Coord(0, 0, 4096), 5.0f, true);
    EXPECT_TRUE(root.addChild(child)); // always returns true

    EXPECT_EQ(root.getTableSize(), 3);
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
    EXPECT_TRUE(root.hasKey(Coord(4096, 0, 0)));
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 4096)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 4096, 0)));

    EXPECT_FALSE(root.deleteChildOrTile(Coord(4096, 4096, 0)));

    EXPECT_EQ(root.getTableSize(), 3);
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
    EXPECT_TRUE(root.hasKey(Coord(4096, 0, 0)));
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 4096)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 4096, 0)));

    EXPECT_TRUE(root.deleteChildOrTile(Coord(4096, 5, 6)));

    EXPECT_EQ(root.getTableSize(), 2);
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 0, 0)));
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 4096)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 4096, 0)));

    EXPECT_TRUE(root.deleteChildOrTile(Coord(1, 5, 4097)));

    EXPECT_EQ(root.getTableSize(), 1);
    EXPECT_TRUE(root.hasKey(Coord(0, 0, 0)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 0, 0)));
    EXPECT_FALSE(root.hasKey(Coord(0, 0, 4096)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 4096, 0)));

    EXPECT_TRUE(root.deleteChildOrTile(Coord(1, 5, 7)));

    EXPECT_EQ(root.getTableSize(), 0);
    EXPECT_FALSE(root.hasKey(Coord(0, 0, 0)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 0, 0)));
    EXPECT_FALSE(root.hasKey(Coord(0, 0, 4096)));
    EXPECT_FALSE(root.hasKey(Coord(4096, 4096, 0)));
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
        EXPECT_TRUE(constRoot.probe(Coord(0, 0, 0), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
        EXPECT_EQ(value, 2.0f);
        EXPECT_EQ(active, true);
        value = -1.0f;
        EXPECT_TRUE(root.probe(Coord(4096, 0, 0), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));
        EXPECT_EQ(value, 3.0f);
        EXPECT_EQ(active, false);
        EXPECT_FALSE(root.probe(Coord(4096, 4096, 4096), constChildPtr, value, active));
        EXPECT_FALSE(bool(constChildPtr));

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

TEST_F(TestRoot, testUnsafe)
{
    using RootNode = FloatTree::RootNodeType;

    RootNode root(1.0f);

    root.addTile(Coord(1, 2, 3), 2.0f, true);
    root.addTile(Coord(4096, 2, 3), 3.0f, false);

    auto* child = new RootNode::ChildNodeType(Coord(0, 0, 4096), 5.0f, true);
    EXPECT_TRUE(root.addChild(child)); // always returns true

    { // get value
        EXPECT_EQ(root.getTileValueUnsafe(Coord(1, 2, 3)), 2.0f);
        EXPECT_EQ(root.getTileValueUnsafe(Coord(4096, 2, 3)), 3.0f);
        float value = -1.0f;
        EXPECT_TRUE(root.getTileValueUnsafe(Coord(1, 2, 3), value));
        EXPECT_EQ(value, 2.0f); value = -1.0f;
        EXPECT_FALSE(root.getTileValueUnsafe(Coord(4096, 2, 3), value));
        EXPECT_EQ(value, 3.0f); value = -1.0f;
    }

    { // get child
        auto* node1 = root.getChildUnsafe(Coord(0, 0, 4096));
        EXPECT_TRUE(node1);
        const RootNode& constRoot = root;
        auto* node2 = constRoot.getChildUnsafe(Coord(0, 0, 4096));
        EXPECT_TRUE(node2);
        auto* node3 = root.getConstChildUnsafe(Coord(0, 0, 4096));
        EXPECT_TRUE(node3);
    }
}
