// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file unittest/TestLevelSetRayIntersector.cc
/// @author Ken Museth

// Uncomment to enable statistics of ray-intersections
//#define STATS_TEST

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/Ray.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/RayIntersector.h>
#include <openvdb/tools/RayTracer.h>// for Film
#ifdef STATS_TEST
//only needed for statistics
#include <openvdb/math/Stats.h>
#include <openvdb/util/CpuTimer.h>
#include <iostream>
#endif

#include <gtest/gtest.h>


#define ASSERT_DOUBLES_APPROX_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/1.e-6);


class TestLevelSetRayIntersector : public ::testing::Test
{
};


template<typename GridT>
void
testLevelSetRayIntersectorImpl()
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        tools::LevelSetRayIntersector<GridT> lsri(*ls);

        const Vec3T dir(1.0, 0.0, 0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        //std::cerr << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(13.0, time);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << " time = " << time << std::endl;
        EXPECT_TRUE(ray(t0) == xyz);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(20.0f, 0.0f, 0.0f);
        const float s = 0.5f, w = 2.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        tools::LevelSetRayIntersector<GridT> lsri(*ls);

        const Vec3T dir(1.0,-0.0,-0.0);
        const Vec3T eye(2.0, 0.0, 0.0);
        const RayT ray(eye, dir);
        //std::cerr << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(13.0, time);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        EXPECT_TRUE(ray(t0) == xyz);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        tools::LevelSetRayIntersector<GridT> lsri(*ls);

        const Vec3T dir(0.0, 1.0, 0.0);
        const Vec3T eye(0.0,-2.0, 0.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        constexpr double tolerance = std::is_floating_point_v<typename GridT::ValueType> ? 1e-6 : 1e-3;

        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        EXPECT_NEAR( 0.0, xyz[0], tolerance);
        EXPECT_NEAR(15.0, xyz[1], tolerance);
        EXPECT_NEAR( 0.0, xyz[2], tolerance);
        EXPECT_NEAR(17.0, time, tolerance);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        EXPECT_NEAR(t0, time, tolerance);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        EXPECT_NEAR( 0.0, ray(t0)[0], tolerance);
        EXPECT_NEAR(15.0, ray(t0)[1], tolerance);
        EXPECT_NEAR( 0.0, ray(t0)[2], tolerance);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 20.0f, 0.0f);
        const float s = 1.5f, w = 2.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        tools::LevelSetRayIntersector<GridT> lsri(*ls);

        const Vec3T dir(-0.0, 1.0,-0.0);
        const Vec3T eye( 0.0,-2.0, 0.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        constexpr double tolerance = std::is_floating_point_v<typename GridT::ValueType> ? 1e-6 : 1e-3;

        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        EXPECT_NEAR( 0.0, xyz[0], tolerance);
        EXPECT_NEAR(15.0, xyz[1], tolerance);
        EXPECT_NEAR( 0.0, xyz[2], tolerance);
        EXPECT_NEAR(17.0, time, tolerance);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        EXPECT_NEAR(t0, time, tolerance);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        EXPECT_NEAR( 0.0, ray(t0)[0], tolerance);
        EXPECT_NEAR(15.0, ray(t0)[1], tolerance);
        EXPECT_NEAR( 0.0, ray(t0)[2], tolerance);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        typedef tools::LinearSearchImpl<GridT> SearchImplT;
        tools::LevelSetRayIntersector<GridT, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(0.0, 0.0, 1.0);
        const Vec3T eye(0.0, 0.0, 4.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;

        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(11.0, time);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        typedef tools::LinearSearchImpl<GridT> SearchImplT;
        tools::LevelSetRayIntersector<GridT, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(-0.0,-0.0, 1.0);
        const Vec3T eye( 0.0, 0.0, 4.0);
        RayT ray(eye, dir);
        Vec3T xyz(0);
        Real time = 0;
        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(11.0, time);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        ASSERT_DOUBLES_APPROX_EQUAL(t0, time);
        //std::cerr << "t0 = " << t0 << " t1 = " << t1 << std::endl;
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(15.0, ray(t0)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(0.0f, 0.0f, 20.0f);
        const float s = 1.0f, w = 3.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        typedef tools::LinearSearchImpl<GridT> SearchImplT;
        tools::LevelSetRayIntersector<GridT, SearchImplT, -1> lsri(*ls);

        const Vec3T dir(-0.0,-0.0, 1.0);
        const Vec3T eye( 0.0, 0.0, 4.0);
        RayT ray(eye, dir, 16.0);
        Vec3T xyz(0);
        Real time = 0;
        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, xyz[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, xyz[2]);
        ASSERT_DOUBLES_APPROX_EQUAL(21.0, time);
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        //std::cerr << "t0 = " << t0 << " t1 = " << t1 << std::endl;
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "Intersection at  xyz = " << xyz << std::endl;
        ASSERT_DOUBLES_APPROX_EQUAL(t1, time);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[0]);
        ASSERT_DOUBLES_APPROX_EQUAL( 0.0, ray(t0)[1]);
        ASSERT_DOUBLES_APPROX_EQUAL(25.0, ray(t1)[2]);
    }

    {// voxel intersection against a level set sphere
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 10.0f);
        const float s = 1.0f, w = 3.0f;

        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        tools::LevelSetRayIntersector<GridT> lsri(*ls);

        Vec3T dir(1.0, 1.0, 1.0); dir.normalize();
        const Vec3T eye(0.0, 0.0, 0.0);
        RayT ray(eye, dir);
        //std::cerr << "ray: " << ray << std::endl;
        Vec3T xyz(0);
        Real time = 0;
        constexpr double tolerance = std::is_floating_point_v<typename GridT::ValueType> ? 1e-6 : 2e-5;

        EXPECT_TRUE(lsri.intersectsWS(ray, xyz, time));
        //std::cerr << "\nIntersection at  xyz = " << xyz << std::endl;
        //analytical intersection test
        double t0=0, t1=0;
        EXPECT_TRUE(ray.intersects(c, r, t0, t1));
        EXPECT_NEAR(t0, time, tolerance);
        EXPECT_NEAR((ray(t0)-c).length()-r, 0, tolerance);
        EXPECT_NEAR((ray(t1)-c).length()-r, 0, tolerance);
        //std::cerr << "\nray("<<t0<<")="<<ray(t0)<<std::endl;
        //std::cerr << "\nray("<<t1<<")="<<ray(t1)<<std::endl;
        const Vec3T delta = xyz - ray(t0);
        //std::cerr << "delta = " << delta << std::endl;
        //std::cerr << "|delta|/dx=" << (delta.length()/ls->voxelSize()[0]) << std::endl;
        EXPECT_NEAR(0, delta.length(), tolerance);
    }

    {// test intersections against a high-resolution level set sphere @1024^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        double t0=0, t1=0;
        typename GridT::Ptr ls = tools::createLevelSetSphere<GridT>(r, c, s, w);

        typedef tools::LinearSearchImpl<GridT, /*iterations=*/2> SearchImplT;
        tools::LevelSetRayIntersector<GridT, SearchImplT> lsri(*ls);

        Vec3T xyz(0);
        Real time = 0;
        const size_t width = 1024;
        const double dx = 20.0/width;
        const Vec3T dir(0.0, 0.0, 1.0);

        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*double(i), dx*double(j), 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, xyz, time)){
                    EXPECT_TRUE(ray.intersects(c, r, t0, t1));
                    EXPECT_NEAR(0, 100*(t0-time)/t0, /*tolerance=*/0.1);//percent
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    EXPECT_TRUE(delta < 0.06);
                }
            }
        }
    }
}

TEST_F(TestLevelSetRayIntersector, testLevelSetRayIntersectorFloat)
{
    testLevelSetRayIntersectorImpl<openvdb::FloatGrid>();
}

TEST_F(TestLevelSetRayIntersector, testLevelSetRayIntersectorHalf)
{
    testLevelSetRayIntersectorImpl<openvdb::HalfGrid>();
}

TEST_F(TestLevelSetRayIntersector, testMissedIntersections)
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;

    // Create a level set sphere
    const float r = 10.0f;
    const Vec3f c(0.0f, 0.0f, 0.0f);
    const float s = 1.0f, w = 3.0f;
    FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);
    tools::LevelSetRayIntersector<FloatGrid> lsri(*ls);

    // Create a ray that misses the sphere, but intersects the bounding box
    const Vec3T dir(-2.0, -2.0, 3.0); // Ray crossing the sphere
    const Vec3T eye(12.0, 12.0, 12.0); // Starting point
    const RayT ray(eye, dir);

    // Initial values
    Vec3T world(1.0, 2.0, 3.0);
    Vec3T normal(4.0, 5.0, 6.0);
    Real time = 42.0;

    // These tests all have zero intersections and it is expected
    // that the variables being passed in are not altered as a result.

    // Test with time
    EXPECT_FALSE(lsri.intersectsIS(ray, time));
    EXPECT_EQ(42.0, time);

    // Test with world position
    EXPECT_FALSE(lsri.intersectsIS(ray, world));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);

    // Test with world position and time
    EXPECT_FALSE(lsri.intersectsIS(ray, world, time));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);
    EXPECT_EQ(42.0, time);

    // Test with time
    EXPECT_FALSE(lsri.intersectsWS(ray, time));
    EXPECT_EQ(42.0, time);

    // Test with world position
    EXPECT_FALSE(lsri.intersectsWS(ray, world));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);

    // Test with world position and time
    EXPECT_FALSE(lsri.intersectsWS(ray, world, time));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);
    EXPECT_EQ(42.0, time);

    // Test with world position and normal
    EXPECT_FALSE(lsri.intersectsWS(ray, world, normal));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);
    EXPECT_EQ(4.0, normal[0]);
    EXPECT_EQ(5.0, normal[1]);
    EXPECT_EQ(6.0, normal[2]);

    // Test with world position, normal and time
    EXPECT_FALSE(lsri.intersectsWS(ray, world, normal, time));
    EXPECT_EQ(1.0, world[0]);
    EXPECT_EQ(2.0, world[1]);
    EXPECT_EQ(3.0, world[2]);
    EXPECT_EQ(4.0, normal[0]);
    EXPECT_EQ(5.0, normal[1]);
    EXPECT_EQ(6.0, normal[2]);
    EXPECT_EQ(42.0, time);
}

#ifdef STATS_TEST
TEST_F(TestLevelSetRayIntersector, stats)
{
    using namespace openvdb;
    typedef math::Ray<double>  RayT;
    typedef RayT::Vec3Type     Vec3T;
    util::CpuTimer timer;

    {// generate an image, benchmarks and statistics

        // Generate a high-resolution level set sphere @1024^3
        const float r = 5.0f;
        const Vec3f c(10.0f, 10.0f, 20.0f);
        const float s = 0.01f, w = 2.0f;
        double t0=0, t1=0;
        FloatGrid::Ptr ls = tools::createLevelSetSphere<FloatGrid>(r, c, s, w);

        typedef tools::LinearSearchImpl<FloatGrid, /*iterations=*/2> SearchImplT;
        tools::LevelSetRayIntersector<FloatGrid, SearchImplT> lsri(*ls);

        Vec3T xyz(0);
        const size_t width = 1024;
        const double dx = 20.0/width;
        const Vec3T dir(0.0, 0.0, 1.0);

        tools::Film film(width, width);
        math::Stats stats;
        math::Histogram hist(0.0, 0.1, 20);

        timer.start("\nSerial ray-intersections of sphere");
        for (size_t i=0; i<width; ++i) {
            for (size_t j=0; j<width; ++j) {
                const Vec3T eye(dx*i, dx*j, 0.0);
                const RayT ray(eye, dir);
                if (lsri.intersectsWS(ray, xyz)){
                    EXPECT_TRUE(ray.intersects(c, r, t0, t1));
                    double delta = (ray(t0)-xyz).length()/s;//in voxel units
                    stats.add(delta);
                    hist.add(delta);
                    if (delta > 0.01) {
                        film.pixel(i, j) = tools::Film::RGBA(1.0f, 0.0f, 0.0f);
                    } else {
                        film.pixel(i, j) = tools::Film::RGBA(0.0f, 1.0f, 0.0f);
                    }
                }
            }
        }
        timer.stop();

        film.savePPM("sphere_serial");
        stats.print("First hit");
        hist.print("First hit");
    }
}
#endif // STATS_TEST

#undef STATS_TEST
