// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Activate.h>

#include <gtest/gtest.h>

// #define BENCHMARK


class TestActivate: public ::testing::Test
{
};


////////////////////////////////////////


// migrated from TestTools::testActivate()
TEST_F(TestActivate, testActivate)
{
    using namespace openvdb;

    const Vec3s background(0.0, -1.0, 1.0), foreground(42.0);

    Vec3STree tree(background);

#ifndef BENCHMARK
    const CoordBBox bbox1(Coord(-200), Coord(-181));
    const CoordBBox bbox2(Coord(51), Coord(373));
#else
    const CoordBBox bbox1(Coord(-200*20), Coord(-181*20));
    const CoordBBox bbox2(Coord(51*20), Coord(373*20));
#endif

    // Set some non-background active voxels.
    tree.fill(bbox1, Vec3s(0.0), /*active=*/true);

    // Mark some background voxels as active.
    tree.fill(bbox2, background, /*active=*/true);
    EXPECT_EQ(bbox2.volume() + bbox1.volume(), tree.activeVoxelCount());

    // Deactivate all voxels with the background value.
    tools::deactivate(tree, background, /*tolerance=*/Vec3s(1.0e-6f));
    // Verify that there are no longer any active voxels with the background value.
    EXPECT_EQ(bbox1.volume(), tree.activeVoxelCount());

    // Set some voxels to the foreground value but leave them inactive.
    tree.fill(bbox2, foreground, /*active=*/false);
    // Verify that there are no active voxels with the background value.
    EXPECT_EQ(bbox1.volume(), tree.activeVoxelCount());

    // Activate all voxels with the foreground value.
    tools::activate(tree, foreground);
    // Verify that the expected number of voxels are active.
    EXPECT_EQ(bbox1.volume() + bbox2.volume(), tree.activeVoxelCount());
}

TEST_F(TestActivate, testActivateLeafValues)
{
    using namespace openvdb;

    { // activate leaf with a single inactive voxel
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));
        EXPECT_TRUE(leaf->isEmpty());

        // all values are 0.0f so activate with value = 1.0f is a noop

        tools::activate(tree, 1.0f);
        EXPECT_TRUE(leaf->isEmpty());

        // set leaf[0] to 1.0f and activate this one voxel

        leaf->setValueOff(0, 1.0f);
        tools::activate(tree, 1.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
    }

    { // activate leaf with a single inactive voxel within the tolerance
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        // set leaf[0] to a small tolerance above 1.0f

        leaf->setValueOff(0, 1.0f + 1e-4f);
        tools::activate(tree, 1.0f); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));

        // activate using explicit tolerance

        tools::activate(tree, 1.0f, 1e-6f); // tolerance is too small
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
        tools::activate(tree, 1.0f, 1e-3f); // tolerance is now large enough
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
    }

    { // activate leaf with a single inactive voxel with 0.1f value
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        // set leaf[0] to 0.1f (which cannot be represented exactly in floating-point)

        leaf->setValueOff(0, 0.1f);
        tools::activate(tree, 0.1f); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
    }

    { // activate leaf with a few active and inactive voxels
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOff(0, 1.0f);
        leaf->setValueOff(1, 3.0f);
        leaf->setValueOff(2, -3.0f);
        leaf->setValueOn(3, 1.0f);
        leaf->setValueOn(4, 3.0f);
        leaf->setValueOn(5, -3.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(3));
        tools::activate(tree, 1.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(4));
    }

    { // activate an integer leaf
        Int32Tree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOff(0, 10);
        leaf->setValueOff(1, 9);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
        tools::activate(tree, 9); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
        tools::activate(tree, 9, /*tolerance=*/2);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
    }

    { // activate a Vec3s leaf
        Vec3STree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOff(0, Vec3s(10));
        leaf->setValueOff(1, Vec3s(2, 3, 4));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
        tools::activate(tree, Vec3s(2, 3, 5)); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
        tools::activate(tree, Vec3s(2, 3, 5), /*tolerance=*/Vec3s(0, 0, 2));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
        tools::activate(tree, Vec3s(10), Vec3s(0.1f));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
    }

    { // activate a mask leaf
        MaskTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOnly(0, true);
        leaf->setValueOnly(1, true);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::activate(tree, true); // noop
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::activate(tree, false); // all inactive values become active
        EXPECT_EQ(leaf->onVoxelCount(), Index64(512));
    }
}

TEST_F(TestActivate, testActivateTiles)
{
    using namespace openvdb;

    auto getActiveTiles = [](const auto& node) -> Index
    {
        Index count(0);
        for (auto iter = node.cbeginValueOn(); iter; ++iter)   count++;
        return count;
    };

    auto getInactiveTiles = [](const auto& node) -> Index
    {
        Index count(0);
        for (auto iter = node.cbeginValueOff(); iter; ++iter)   count++;
        return count;
    };

    { // activate a single inactive tile
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), 1.0f, false);

        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
        tools::activate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
    }

    { // activate a single inactive tile with tolerance
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), 1.0f + 1e-4f, false);

        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
        tools::activate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
        tools::activate(tree, 1.0f, 1e-6f);
        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
        tools::activate(tree, 1.0f, 1e-3f);
        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
    }

    { // activate a single inactive tile from an internal node
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add an internal node tile
        tree.addTile(/*level=*/1, Coord(0), 1.0f, false);

        const auto& child = *(root.cbeginChildOn()->cbeginChildOn());

        EXPECT_EQ(getInactiveTiles(child), Index(4096));
        EXPECT_EQ(getActiveTiles(child), Index(0));
        tools::activate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(child), Index(4095));
        EXPECT_EQ(getActiveTiles(child), Index(1));
    }

    { // activate a single inactive tile in a Vec3s
        Vec3STree tree;
        const Vec3STree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), Vec3s(1), false);

        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
        tools::activate(tree, Vec3s(1));
        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
    }
}

TEST_F(TestActivate, testDeactivateLeafValues)
{
    using namespace openvdb;

    { // deactivate leaf with a single active voxel
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));
        EXPECT_TRUE(leaf->isEmpty());

        // all values are 0.0f so deactivate with value = 1.0f is a noop

        tools::deactivate(tree, 1.0f);
        EXPECT_TRUE(leaf->isEmpty());

        // set leaf[0] to 1.0f and deactivate this one voxel

        leaf->setValueOn(0, 1.0f);
        tools::deactivate(tree, 1.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }

    { // deactivate leaf with a single active voxel within the tolerance
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        // set leaf[0] to a small tolerance above 1.0f

        leaf->setValueOn(0, 1.0f + 1e-4f);
        tools::deactivate(tree, 1.0f); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));

        // deactivate using explicit tolerance

        tools::deactivate(tree, 1.0f, 1e-6f); // tolerance is too small
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
        tools::deactivate(tree, 1.0f, 1e-3f); // tolerance is now large enough
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }

    { // deactivate leaf with a single active voxel with 0.1f value
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        // set leaf[0] to 0.1f (which cannot be represented exactly in floating-point)

        leaf->setValueOn(0, 0.1f);
        tools::deactivate(tree, 0.1f); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }

    { // deactivate leaf with a few active and inactive voxels
        FloatTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOff(0, 1.0f);
        leaf->setValueOff(1, 3.0f);
        leaf->setValueOff(2, -3.0f);
        leaf->setValueOn(3, 1.0f);
        leaf->setValueOn(4, 3.0f);
        leaf->setValueOn(5, -3.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(3));
        tools::deactivate(tree, 1.0f);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
    }

    { // deactivate an integer leaf
        Int32Tree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOn(0, 10);
        leaf->setValueOn(1, 9);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::deactivate(tree, 9); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
        tools::deactivate(tree, 9, /*tolerance=*/2);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }

    { // deactivate a Vec3s leaf
        Vec3STree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOn(0, Vec3s(10));
        leaf->setValueOn(1, Vec3s(2, 3, 4));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::deactivate(tree, Vec3s(2, 3, 5)); // default tolerance is zero
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::deactivate(tree, Vec3s(2, 3, 5), /*tolerance=*/Vec3s(0, 0, 2));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(1));
        tools::deactivate(tree, Vec3s(10), Vec3s(0.1f));
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }

    { // deactivate a mask leaf
        MaskTree tree;
        auto* leaf = tree.touchLeaf(openvdb::Coord(0));

        leaf->setValueOnly(0, true);
        leaf->setValueOnly(1, true);
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::deactivate(tree, false); // noop
        EXPECT_EQ(leaf->onVoxelCount(), Index64(2));
        tools::deactivate(tree, true); // all active values become inactive
        EXPECT_EQ(leaf->onVoxelCount(), Index64(0));
    }
}

TEST_F(TestActivate, testDeactivateTiles)
{
    using namespace openvdb;

    auto getActiveTiles = [](const auto& node) -> Index
    {
        Index count(0);
        for (auto iter = node.cbeginValueOn(); iter; ++iter)   count++;
        return count;
    };

    auto getInactiveTiles = [](const auto& node) -> Index
    {
        Index count(0);
        for (auto iter = node.cbeginValueOff(); iter; ++iter)   count++;
        return count;
    };

    { // deactivate a single active tile
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), 1.0f, true);

        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
        tools::deactivate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
    }

    { // deactivate a single active tile with tolerance
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), 1.0f + 1e-4f, true);

        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
        tools::deactivate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
        tools::deactivate(tree, 1.0f, 1e-6f);
        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
        tools::deactivate(tree, 1.0f, 1e-3f);
        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
    }

    { // deactivate a single active tile from an internal node
        FloatTree tree;
        const FloatTree::RootNodeType& root = tree.root();

        // add an internal node tile
        tree.addTile(/*level=*/1, Coord(0), 1.0f, true);

        const auto& child = *(root.cbeginChildOn()->cbeginChildOn());

        EXPECT_EQ(getInactiveTiles(child), Index(4095));
        EXPECT_EQ(getActiveTiles(child), Index(1));
        tools::deactivate(tree, 1.0f);
        EXPECT_EQ(getInactiveTiles(child), Index(4096));
        EXPECT_EQ(getActiveTiles(child), Index(0));
    }

    { // deactivate a single active tile in a Vec3s
        Vec3STree tree;
        const Vec3STree::RootNodeType& root = tree.root();

        // add a root node tile
        tree.addTile(/*level=*/3, Coord(0), Vec3s(1), true);

        EXPECT_EQ(getInactiveTiles(root), Index(0));
        EXPECT_EQ(getActiveTiles(root), Index(1));
        tools::deactivate(tree, Vec3s(1));
        EXPECT_EQ(getInactiveTiles(root), Index(1));
        EXPECT_EQ(getActiveTiles(root), Index(0));
    }
}
