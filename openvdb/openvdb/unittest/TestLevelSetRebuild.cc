// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <vector>
#include "gtest/gtest.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetRebuild.h>

class TestLevelSetRebuild: public ::testing::Test
{
};


////////////////////////////////////////

TEST_F(TestLevelSetRebuild, test)
{
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(10.0);

    grid->fill(openvdb::CoordBBox(openvdb::Coord(-100), openvdb::Coord(100)), 9.0);
    grid->fill(openvdb::CoordBBox(openvdb::Coord(-50), openvdb::Coord(50)), -9.0);

    openvdb::StringMetadata meta;
    meta.setValue("bar");

    grid->insertMeta("foo", meta);

    openvdb::FloatGrid::Ptr newGrid = openvdb::tools::levelSetRebuild(
        *grid, /*isovalue=*/0.0f, /*exBandWidth=*/3.0f, /*inBandWidth=*/3.0f);

    // check the metadata has been copied from the input grid

    auto newMeta = (*newGrid)["foo"];
    EXPECT_TRUE(newMeta);
    if (newMeta) EXPECT_EQ(newMeta->str(), "bar");
}
