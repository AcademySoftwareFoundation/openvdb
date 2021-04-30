// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "gtest/gtest.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetSphere.h> // tools::createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // tools::sdfToFogVolume
#include <openvdb/tree/ValueAccessor.h>


class TestCount: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestCount, testActiveVoxelCount)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(25.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // count the number of active voxels by hand in active tiles and leaf nodes

    using Internal1NodeT = FloatTree::RootNodeType::ChildNodeType;
    using Internal2NodeT = Internal1NodeT::ChildNodeType;
    using LeafNodeT = Internal2NodeT::ChildNodeType;

    Index64 activeVoxelCount1(0);

    const auto& tree = grid->tree();

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(tree.activeTileCount() > 0);

    const auto& root = tree.root();

    for (auto valueIter = root.cbeginValueOn(); valueIter; ++valueIter) {
        activeVoxelCount1 += Internal1NodeT::NUM_VOXELS;
    }

    for (auto internal1Iter = root.cbeginChildOn(); internal1Iter; ++internal1Iter) {
        for (auto valueIter = internal1Iter->cbeginValueOn(); valueIter; ++valueIter) {
            activeVoxelCount1 += Internal2NodeT::NUM_VOXELS;
        }

        for (auto internal2Iter = internal1Iter->cbeginChildOn(); internal2Iter; ++internal2Iter) {
            for (auto valueIter = internal2Iter->cbeginValueOn(); valueIter; ++valueIter) {
                activeVoxelCount1 += LeafNodeT::NUM_VOXELS;
            }

            for (auto leafIter = internal2Iter->cbeginChildOn(); leafIter; ++leafIter) {
                activeVoxelCount1 += leafIter->onVoxelCount();
            }
        }
    }

    Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree());

    EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
}


TEST_F(TestCount, testActiveVoxelCountBBox)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(grid->tree().activeTileCount() > 0);

    { // entire bbox
        const CoordBBox bbox(Coord(-110), Coord(110));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter))   activeVoxelCount1++;
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox, false);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
    }

    { // tiny bbox
        const CoordBBox bbox(Coord(-2), Coord(2));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter))   activeVoxelCount1++;
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
    }

    { // subset bbox
        const CoordBBox bbox(Coord(-80, -110, -80), Coord(80, 110, 80));

        // count manually - iterate over all Coords in bbox and test each one

        Index64 activeVoxelCount1(0);
        tree::ValueAccessor<const FloatTree> acc(grid->constTree());
        for (auto iter = bbox.begin(); iter; ++iter) {
            if (acc.isValueOn(*iter))   activeVoxelCount1++;
        }

        Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree(), bbox);

        EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
    }
}


TEST_F(TestCount, testMemUsage)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);
    tools::sdfToFogVolume(*grid); // convert to fog volume to generate active tiles

    // count the memory usage manually across all nodes

    using Internal1NodeT = FloatTree::RootNodeType::ChildNodeType;
    using Internal2NodeT = Internal1NodeT::ChildNodeType;

    const auto& tree = grid->tree();

    // ensure there are active tiles in this example grid

    EXPECT_TRUE(tree.activeTileCount() > 0);

    const auto& root = tree.root();

    Index64 memUsage1(sizeof(tree) + sizeof(root));

    for (auto internal1Iter = root.cbeginChildOn(); internal1Iter; ++internal1Iter) {
        memUsage1 += Internal1NodeT::NUM_VALUES * sizeof(Internal1NodeT::UnionType);
        memUsage1 += internal1Iter->getChildMask().memUsage();
        memUsage1 += internal1Iter->getValueMask().memUsage();
        memUsage1 += sizeof(Coord);

        for (auto internal2Iter = internal1Iter->cbeginChildOn(); internal2Iter; ++internal2Iter) {
            memUsage1 += Internal2NodeT::NUM_VALUES * sizeof(Internal2NodeT::UnionType);
            memUsage1 += internal2Iter->getChildMask().memUsage();
            memUsage1 += internal2Iter->getValueMask().memUsage();
            memUsage1 += sizeof(Coord);

            for (auto leafIter = internal2Iter->cbeginChildOn(); leafIter; ++leafIter) {
                memUsage1 += leafIter->memUsage();
            }
        }
    }

    Index64 memUsage2 = tools::memUsage(grid->tree());

    EXPECT_EQ(memUsage1, memUsage2);
}
