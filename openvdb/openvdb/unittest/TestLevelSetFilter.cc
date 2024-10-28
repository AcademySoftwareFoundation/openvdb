// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/Composite.h>        // for csgUnion()

#include <gtest/gtest.h>

using namespace openvdb;

class TestLevelSetFilter: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }

    void testLevelSetFillet();
}; // class TestLevelSetFilter


////////////////////////////////////////


TEST_F(TestLevelSetFilter, testLevelSetFillet)
{
    using GridT = FloatGrid;
    using FilterT = tools::LevelSetFilter<GridT>;

    const float radius = 5.0f;
    const float voxelSize = 1.0f;
    const float halfWidth = 3.0f;
    typename GridT::Ptr sdfGrid = tools::createLevelSetSphere<GridT>(/*radius=*/radius,
            /*center=*/Vec3f(-radius, 0.0f, 0.0f),
            /*dx=*/voxelSize, /*halfWidth*/ halfWidth);
    typename GridT::Ptr sdfGridB = tools::createLevelSetSphere<GridT>(/*radius=*/radius,
            /*center=*/Vec3f(radius, 0.0f, 0.0f),
            /*dx=*/voxelSize, /*halfWidth*/ halfWidth);
    typename GridT::Accessor acc = sdfGrid->getAccessor();

    EXPECT_TRUE(sdfGrid);
    EXPECT_TRUE(sdfGridB);

    tools::csgUnion(*sdfGrid, *sdfGridB);

    {
        EXPECT_TRUE(sdfGrid);

        Coord ijk(0, 3, 0);

        // We expect that the intersection between the two spheres are at (0, 0, 0)
        // so the SDF value of the union in these offsets locations should be > 0
        EXPECT_TRUE(acc.getValue(ijk) > 0.f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0, 1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0,-1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 0)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 1)) > 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1,-1)) > 0.0f);
    }

    FilterT filter(*sdfGrid);
    filter.fillet();

    {
        EXPECT_TRUE(sdfGrid);

        Coord ijk(0, 3, 0);

        // After the fillet operation, we expect that the zero-isocontour is
        // pushed outward.
        EXPECT_TRUE(acc.getValue(ijk) < 0.f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0, 1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0, 0,-1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 0)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1, 1)) < 0.0f);
        EXPECT_TRUE(acc.getValue(ijk.offsetBy(0,-1,-1)) < 0.0f);
    }
}
