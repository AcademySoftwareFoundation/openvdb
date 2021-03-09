// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "gtest/gtest.h"

#include <openvdb/openvdb.h>
#include <openvdb/tools/Count.h>
#include <openvdb/tools/LevelSetSphere.h>


class TestCount: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestCount, testActiveVoxelCount)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);

    Index64 activeVoxelCount1 = grid->tree().activeVoxelCount();

    Index64 activeVoxelCount2 = tools::activeVoxelCount(grid->tree());

    EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
}
