// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h> // for createLevelSetSphere
#include <openvdb/tools/LevelSetUtil.h> // for sdfToFogVolume
#include <openvdb/tools/VolumeToSpheres.h> // for fillWithSpheres

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>


class TestVolumeToSpheres: public ::testing::Test
{
};


////////////////////////////////////////


TEST_F(TestVolumeToSpheres, testFromLevelSet)
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    openvdb::FloatGrid::ConstPtr grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);

    const bool overlapping = false;
    const int instanceCount = 10000;
    const float
        isovalue = 0.0f,
        minRadius = 5.0f,
        maxRadius = std::numeric_limits<float>::max();
    const openvdb::Vec2i sphereCount(1, 100);

    {
        std::vector<openvdb::Vec4s> spheres;

        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        EXPECT_EQ(1, int(spheres.size()));

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        const auto tolerance = 2.0 * voxelSize;
        EXPECT_NEAR(center[0], spheres[0][0], tolerance);
        EXPECT_NEAR(center[1], spheres[0][1], tolerance);
        EXPECT_NEAR(center[2], spheres[0][2], tolerance);
        EXPECT_NEAR(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces a valid sphere.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount,
            overlapping, minRadius, maxRadius, 1.5f * halfWidth, instanceCount);
        EXPECT_EQ(1, int(spheres.size()));
    }
    {
        // Verify that an isovalue inside the narrow band produces no spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount,
            overlapping, minRadius, maxRadius, -1.5f * halfWidth, instanceCount);
        EXPECT_EQ(0, int(spheres.size()));
    }
}


TEST_F(TestVolumeToSpheres, testFromFog)
{
    const float
        radius = 20.0f,
        voxelSize = 1.0f,
        halfWidth = 3.0f;
    const openvdb::Vec3f center(15.0f, 13.0f, 16.0f);

    auto grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, center, voxelSize, halfWidth);
    openvdb::tools::sdfToFogVolume(*grid);

    const bool overlapping = false;
    const int instanceCount = 10000;
    const float
        isovalue = 0.01f,
        minRadius = 5.0f,
        maxRadius = std::numeric_limits<float>::max();
    const openvdb::Vec2i sphereCount(1, 100);

    {
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, isovalue, instanceCount);

        //for (size_t i=0; i< spheres.size(); ++i) {
        //    std::cout << "\nSphere #" << i << ": " << spheres[i] << std::endl;
        //}

        EXPECT_EQ(1, int(spheres.size()));

        const auto tolerance = 2.0 * voxelSize;
        EXPECT_NEAR(center[0], spheres[0][0], tolerance);
        EXPECT_NEAR(center[1], spheres[0][1], tolerance);
        EXPECT_NEAR(center[2], spheres[0][2], tolerance);
        EXPECT_NEAR(radius,    spheres[0][3], tolerance);
    }
    {
        // Verify that an isovalue outside the narrow band still produces valid spheres.
        std::vector<openvdb::Vec4s> spheres;
        openvdb::tools::fillWithSpheres(*grid, spheres, sphereCount, overlapping,
            minRadius, maxRadius, 10.0f, instanceCount);
        EXPECT_TRUE(!spheres.empty());
    }
}


TEST_F(TestVolumeToSpheres, testMinimumSphereCount)
{
    using namespace openvdb;
    {
        auto grid = tools::createLevelSetSphere<FloatGrid>(/*radius=*/5.0f,
            /*center=*/Vec3f(15.0f, 13.0f, 16.0f), /*voxelSize=*/1.0f, /*halfWidth=*/3.0f);

        // Verify that the requested minimum number of spheres is generated, for various minima.
        const int maxSphereCount = 100;
        for (int minSphereCount = 1; minSphereCount < 20; minSphereCount += 5) {
            std::vector<Vec4s> spheres;
            tools::fillWithSpheres(*grid, spheres, Vec2i(minSphereCount, maxSphereCount),
                /*overlapping=*/true, /*minRadius=*/2.0f);

            // Given the relatively large minimum radius, the actual sphere count
            // should be no larger than the requested mimimum count.
            EXPECT_EQ(minSphereCount, int(spheres.size()));
            //EXPECT_TRUE(int(spheres.size()) >= minSphereCount);
            EXPECT_TRUE(int(spheres.size()) <= maxSphereCount);
        }
    }
    {
        // One step in the sphere packing algorithm is to erode the active voxel mask
        // of the input grid.  Previously, for very small grids this sometimes resulted in
        // an empty mask and therefore no spheres.  Verify that that no longer happens
        // (as long as the minimum sphere count is nonzero).

        FloatGrid grid;
        CoordBBox bbox(Coord(1), Coord(2));
        grid.fill(bbox, 1.0f);

        const float minRadius = 1.0f;
        const Vec2i sphereCount(5, 100);

        std::vector<Vec4s> spheres;
        tools::fillWithSpheres(grid, spheres, sphereCount, /*overlapping=*/true, minRadius);

        EXPECT_TRUE(int(spheres.size()) >= sphereCount[0]);
    }
}


TEST_F(TestVolumeToSpheres, testClosestSurfacePoint)
{
    using namespace openvdb;

    const float voxelSize = 1.0f;
    const Vec3f center{0.0f}; // ensure multiple internal nodes

    for (const float radius: { 8.0f, 50.0f }) {
        // Construct a spherical level set.
        const auto sphere = tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize);
        EXPECT_TRUE(sphere);

        // Construct the corners of a cube that exactly encloses the sphere.
        const std::vector<Vec3R> corners{
            { -radius, -radius, -radius },
            { -radius, -radius,  radius },
            { -radius,  radius, -radius },
            { -radius,  radius,  radius },
            {  radius, -radius, -radius },
            {  radius, -radius,  radius },
            {  radius,  radius, -radius },
            {  radius,  radius,  radius },
        };
        // Compute the distance from a corner of the cube to the surface of the sphere.
        const auto distToSurface = Vec3d{radius}.length() - radius;

        auto csp = tools::ClosestSurfacePoint<FloatGrid>::create(*sphere);
        EXPECT_TRUE(csp);

        // Move each corner point to the closest surface point.
        auto points = corners;
        std::vector<float> distances;
        bool ok = csp->searchAndReplace(points, distances);
        EXPECT_TRUE(ok);
        EXPECT_EQ(8, int(points.size()));
        EXPECT_EQ(8, int(distances.size()));

        for (auto d: distances) {
            EXPECT_TRUE((std::abs(d - distToSurface) / distToSurface) < 0.01); // rel err < 1%
        }
        for (int i = 0; i < 8; ++i) {
            const auto intersection = corners[i] + distToSurface * (center - corners[i]).unit();
            EXPECT_TRUE(points[i].eq(intersection, /*tolerance=*/0.1));
        }

        // Place a point inside the sphere.
        points.clear();
        distances.clear();
        points.emplace_back(1, 0, 0);
        ok = csp->searchAndReplace(points, distances);
        EXPECT_TRUE(ok);
        EXPECT_EQ(1, int(points.size()));
        EXPECT_EQ(1, int(distances.size()));
        EXPECT_TRUE((std::abs(radius - 1 - distances[0]) / (radius - 1)) < 0.01);
        EXPECT_TRUE(points[0].eq(Vec3R{radius, 0, 0}, /*tolerance=*/0.5));
            ///< @todo off by half a voxel in y and z
    }
}
