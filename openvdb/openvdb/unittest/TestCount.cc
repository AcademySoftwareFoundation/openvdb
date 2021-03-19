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

    Index64 activeVoxelCount2 = tools::countActiveVoxels(grid->tree());

    EXPECT_EQ(activeVoxelCount1, activeVoxelCount2);
}


TEST_F(TestCount, testMemUsage)
{
    using namespace openvdb;

    auto grid = tools::createLevelSetSphere<FloatGrid>(10.0f, Vec3f(0), 0.1f);

    Index64 memUsage1 = grid->tree().memUsage();

    Index64 memUsage2 = tools::memUsage(grid->tree());

    EXPECT_EQ(memUsage1, memUsage2);
}
