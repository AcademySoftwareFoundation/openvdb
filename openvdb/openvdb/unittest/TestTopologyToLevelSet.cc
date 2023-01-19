// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/tools/TopologyToLevelSet.h>

#include <gtest/gtest.h>


class TopologyToLevelSet: public ::testing::Test
{
};


TEST_F(TopologyToLevelSet, testConversion)
{
    typedef openvdb::tree::Tree4<bool, 5, 4, 3>::Type   Tree543b;
    typedef openvdb::Grid<Tree543b>                     BoolGrid;

    typedef openvdb::tree::Tree4<float, 5, 4, 3>::Type  Tree543f;
    typedef openvdb::Grid<Tree543f>                     FloatGrid;

    /////

    const float voxelSize = 0.1f;
    const openvdb::math::Transform::Ptr transform =
            openvdb::math::Transform::createLinearTransform(voxelSize);

    BoolGrid maskGrid(false);
    maskGrid.setTransform(transform);

    // Define active region
    maskGrid.fill(openvdb::CoordBBox(openvdb::Coord(0), openvdb::Coord(7)), true);
    maskGrid.tree().voxelizeActiveTiles();

    FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(maskGrid);

    EXPECT_TRUE(sdfGrid.get() != NULL);
    EXPECT_TRUE(!sdfGrid->empty());
    EXPECT_EQ(int(openvdb::GRID_LEVEL_SET), int(sdfGrid->getGridClass()));

    // test inside coord value
    EXPECT_TRUE(sdfGrid->tree().getValue(openvdb::Coord(3,3,3)) < 0.0f);

    // test outside coord value
    EXPECT_TRUE(sdfGrid->tree().getValue(openvdb::Coord(10,10,10)) > 0.0f);
}

TEST_F(TopologyToLevelSet, testInputTiles)
{
    // We create a mask with a single active tile with the size of TILE_DIM.
    // The zero crossing should be initialized across this coord boundary.
    static constexpr size_t TILE_DIM = openvdb::MaskGrid::TreeType::LeafNodeType::DIM;

    openvdb::MaskGrid::Ptr maskGrid(new openvdb::MaskGrid());
    maskGrid->tree().addTile(1, openvdb::Coord(0), true, true);
    EXPECT_TRUE(maskGrid->tree().hasActiveTiles());

    openvdb::FloatGrid::Ptr sdfGrid = openvdb::tools::topologyToLevelSet(*maskGrid);
    EXPECT_TRUE(sdfGrid);
    EXPECT_TRUE(!sdfGrid->empty());
    EXPECT_TRUE(sdfGrid->activeVoxelCount() > 0);
    EXPECT_EQ(int(openvdb::GRID_LEVEL_SET), int(sdfGrid->getGridClass()));

    // test expected zero crossing
    EXPECT_TRUE(sdfGrid->tree().getValue(openvdb::Coord(TILE_DIM-1)) < 0.0f);
    EXPECT_TRUE(sdfGrid->tree().getValue(openvdb::Coord(TILE_DIM))   > 0.0f);
}

