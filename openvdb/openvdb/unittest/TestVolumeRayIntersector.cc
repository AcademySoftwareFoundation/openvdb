// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file unittest/TestVolumeRayIntersector.cc
/// @author Ken Museth

#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/RayIntersector.h>

#include "gtest/gtest.h"

#include <cassert>
#include <deque>
#include <iostream>
#include <vector>


#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/1.e-6);


class TestVolumeRayIntersector : public ::testing::Test
{
};


TEST_F(TestVolumeRayIntersector, testAll)
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;

    {//one single leaf node
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0,0,0), 1.0f);
        grid.tree().setValue(Coord(7,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
    {//same as above but with dilation
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0,0,0), 1.0f);
        grid.tree().setValue(Coord(7,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid, 1);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
    {//one single leaf node
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(1,1,1), 1.0f);
        grid.tree().setValue(Coord(7,3,3), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL( 9.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
     {//same as above but with dilation
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(1,1,1), 1.0f);
        grid.tree().setValue(Coord(7,3,3), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid, 1);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
    {//two adjacent leaf nodes
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0,0,0), 1.0f);
        grid.tree().setValue(Coord(8,0,0), 1.0f);
        grid.tree().setValue(Coord(15,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
    {//two adjacent leafs followed by a gab and leaf
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8+7,7,7), 1.0f);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(33.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }
    {//two adjacent leafs followed by a gab, a leaf and an active tile
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.fill(CoordBBox(Coord(4*8,0,0), Coord(4*8+7,7,7)), 2.0f, true);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, t1);
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(41.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }

    {//two adjacent leafs followed by a gab, a leaf and an active tile
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.fill(CoordBBox(Coord(4*8,0,0), Coord(4*8+7,7,7)), 2.0f, true);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));

        std::vector<RayT::TimeSpan> list;
        inter.hits(list);
        EXPECT_TRUE(list.size() == 2);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, list[0].t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, list[0].t1);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, list[1].t0);
        ASSERT_DOUBLES_APPROX_EQUAL(41.0, list[1].t1);
    }

    {//same as above but now with std::deque instead of std::vector
        FloatGrid grid(0.0f);

        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        grid.fill(CoordBBox(Coord(4*8,0,0), Coord(4*8+7,7,7)), 2.0f, true);

        const Vec3T dir( 1.0, 0.0, 0.0);
        const Vec3T eye(-1.0, 0.0, 0.0);
        const RayT ray(eye, dir);//ray in index space
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        EXPECT_TRUE(inter.setIndexRay(ray));

        std::deque<RayT::TimeSpan> list;
        inter.hits(list);
        EXPECT_TRUE(list.size() == 2);
        ASSERT_DOUBLES_APPROX_EQUAL( 1.0, list[0].t0);
        ASSERT_DOUBLES_APPROX_EQUAL(17.0, list[0].t1);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, list[1].t0);
        ASSERT_DOUBLES_APPROX_EQUAL(41.0, list[1].t1);
    }

    {// Test submitted by "Jan" @ GitHub
        FloatGrid grid(0.0f);
        grid.tree().setValue(Coord(0*8,0,0), 1.0f);
        grid.tree().setValue(Coord(1*8,0,0), 1.0f);
        grid.tree().setValue(Coord(3*8,0,0), 1.0f);
        tools::VolumeRayIntersector<FloatGrid> inter(grid);

        const Vec3T dir(-1.0, 0.0, 0.0);
        const Vec3T eye(50.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        EXPECT_TRUE(inter.setIndexRay(ray));
        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(18.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(26.0, t1);
        EXPECT_TRUE(inter.march(t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(34.0, t0);
        ASSERT_DOUBLES_APPROX_EQUAL(50.0, t1);
        EXPECT_TRUE(!inter.march(t0, t1));
    }

    {// Test submitted by "Trevor" @ GitHub

        FloatGrid::Ptr grid = createGrid<FloatGrid>(0.0f);
        grid->tree().setValue(Coord(0,0,0), 1.0f);
        tools::dilateActiveValues(grid->tree(), 1, tools::NN_FACE, tools::IGNORE_TILES);
        tools::VolumeRayIntersector<FloatGrid> inter(*grid);

        //std::cerr << "BBox = " << inter.bbox() << std::endl;

        const Vec3T eye(-0.25, -0.25, 10.0);
        const Vec3T dir( 0.00,  0.00, -1.0);
        const RayT ray(eye, dir);
        EXPECT_TRUE(inter.setIndexRay(ray));// hits bbox

        double t0=0, t1=0;
        EXPECT_TRUE(!inter.march(t0, t1));// misses leafs
    }

    {// Test submitted by "Trevor" @ GitHub

        FloatGrid::Ptr grid = createGrid<FloatGrid>(0.0f);
        grid->tree().setValue(Coord(0,0,0), 1.0f);
        tools::dilateActiveValues(grid->tree(), 1, tools::NN_FACE, tools::IGNORE_TILES);
        tools::VolumeRayIntersector<FloatGrid> inter(*grid);

        //GridPtrVec grids;
        //grids.push_back(grid);
        //io::File vdbfile("trevor_v1.vdb");
        //vdbfile.write(grids);

        //std::cerr << "BBox = " << inter.bbox() << std::endl;

        const Vec3T eye(0.75, 0.75, 10.0);
        const Vec3T dir( 0.00,  0.00, -1.0);
        const RayT ray(eye, dir);
        EXPECT_TRUE(inter.setIndexRay(ray));// hits bbox

        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));// misses leafs
        //std::cerr << "t0=" << t0 << " t1=" << t1 << std::endl;
    }

    {// Test derived from the test submitted by "Trevor" @ GitHub

        FloatGrid grid(0.0f);
        grid.fill(math::CoordBBox(Coord(-1,-1,-1),Coord(1,1,1)), 1.0f);
        tools::VolumeRayIntersector<FloatGrid> inter(grid);
        //std::cerr << "BBox = " << inter.bbox() << std::endl;

        const Vec3T eye(-0.25, -0.25, 10.0);
        const Vec3T dir( 0.00,  0.00, -1.0);
        const RayT ray(eye, dir);
        EXPECT_TRUE(inter.setIndexRay(ray));// hits bbox

        double t0=0, t1=0;
        EXPECT_TRUE(inter.march(t0, t1));// hits leafs
        //std::cerr << "t0=" << t0 << " t1=" << t1 << std::endl;
    }
}
