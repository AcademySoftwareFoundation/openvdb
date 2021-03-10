// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <gtest/gtest.h>

#include <openvdb/openvdb.h>
#include <openvdb/points/PointStatistics.h>
#include "PointBuilder.h"

using namespace openvdb;

class TestPointStatistics: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestPointStatistics


TEST_F(TestPointStatistics, testEvalMinMax)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        float min=-1.0f, max=-2.0f;
        const bool success = points::evalMinMax<float>(points->tree(), "noop", min, max);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, min);
        EXPECT_EQ(-2.0f, max);
    }

    // Test no attribute
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        float min=-1.0f, max=-2.0f;
        const bool success = points::evalMinMax<float>(points->tree(), "noop", min, max);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, min);
        EXPECT_EQ(-2.0f, max);
    }

    // Test invalid attribute Type
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        int32_t min=-1, max=-2;
        EXPECT_THROW(points::evalMinMax<int32_t>(points->tree(), "test", min, max), TypeError);
        EXPECT_EQ(int32_t(-1), min);
        EXPECT_EQ(int32_t(-2), max);
    }

    // Test min max
    {
        auto points = PointBuilder(getBoxPoints()) // 8 points
            .attribute<int32_t>({-3,2,1,0,3,-2,-1,0}, "inttest1") // zero crossing
            .attribute<int32_t>({-10,-5,-9,-1,-2,-2,-1,-2}, "inttest2") // all under 0
            .attribute<float>({-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f}, "floattest")
            .attribute<Vec3f>({ Vec3f(0.0f), Vec3f(-0.0f),
                Vec3f(0.3f), Vec3f(1.0f,-0.5f,-0.2f),
                Vec3f(0.2f), Vec3f(0.2f, 0.5f, 0.1f),
                Vec3f(-0.1f), Vec3f(0.1f) }, "vectest")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .voxelsize(1.0)
            .get();

        // int32_t
        {
            int32_t min=0, max=0;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-3), min);
            EXPECT_EQ(int32_t(3), max);

            success = points::evalMinMax<int32_t>(points->tree(), "inttest2", min, max);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-10), min);
            EXPECT_EQ(int32_t(-1), max);
        }

        // float
        {
            float min=0, max=0;
            bool success = points::evalMinMax<float>(points->tree(), "floattest", min, max);
            EXPECT_TRUE(success);
            EXPECT_EQ(float(-10.2f), min);
            EXPECT_EQ(float(9.5f), max);
        }

        // Vec3f
        {
            Vec3f min(0), max(0);
            bool success = points::evalMinMax<Vec3f>(points->tree(), "vectest", min, max);
            EXPECT_TRUE(success);
            EXPECT_EQ(-0.1f, min.x());
            EXPECT_EQ(-0.5f, min.y());
            EXPECT_EQ(-0.2f, min.z());
            EXPECT_EQ(1.0f, max.x());
            EXPECT_EQ(0.5f, max.y());
            EXPECT_EQ(0.3f, max.z());
        }

        // Test min max filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            int32_t min=0, max=0;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-1), min);
            EXPECT_EQ(int32_t(2), max);

            success = points::evalMinMax<int32_t>(points->tree(), "inttest2", min, max, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-5), min);
            EXPECT_EQ(int32_t(-1), max);
        }

        // float
        {
            float min=0, max=0;
            bool success = points::evalMinMax<float>(points->tree(), "floattest", min, max, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(float(0.0f), min);
            EXPECT_EQ(float(6.2f), max);
        }

        // Vec3f
        {
            Vec3f min(0), max(0);
            bool success = points::evalMinMax<Vec3f>(points->tree(), "vectest", min, max, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(-0.1f, min.x());
            EXPECT_EQ(-0.5f, min.y());
            EXPECT_EQ(-0.2f, min.z());
            EXPECT_EQ(1.0f, max.x());
            EXPECT_EQ(0.1f, max.y());
            EXPECT_EQ(0.1f, max.z());
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            int32_t min=100, max=100;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, empty);
            EXPECT_TRUE(!success);
            EXPECT_EQ(min, 100);
            EXPECT_EQ(max, 100);
        }

        // Test min max trees

        // int32_t
        {
            int32_t min=0, max=0;
            Int32Tree mint, maxt;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, points::NullFilter(), &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-3), min);
            EXPECT_EQ(int32_t(3), max);

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 8);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 8);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({-1, -1, -1}));
            EXPECT_TRUE(mint.isValueOn({-1, -1, 0}));
            EXPECT_TRUE(mint.isValueOn({-1, 0, -1}));
            EXPECT_TRUE(mint.isValueOn({-1, 0, 0}));
            EXPECT_TRUE(mint.isValueOn({0, -1, -1}));
            EXPECT_TRUE(mint.isValueOn({0, -1, 1}));
            EXPECT_TRUE(mint.isValueOn({0, 0, -1}));
            EXPECT_TRUE(mint.isValueOn({0, 0, 0}));

            EXPECT_TRUE(maxt.isValueOn({-1, -1, -1}));
            EXPECT_TRUE(maxt.isValueOn({-1, -1, 0}));
            EXPECT_TRUE(maxt.isValueOn({-1, 0, -1}));
            EXPECT_TRUE(maxt.isValueOn({-1, 0, 0}));
            EXPECT_TRUE(maxt.isValueOn({0, -1, -1}));
            EXPECT_TRUE(maxt.isValueOn({0, -1, 1}));
            EXPECT_TRUE(maxt.isValueOn({0, 0, -1}));
            EXPECT_TRUE(maxt.isValueOn({0, 0, 0}));

            // only 1 point per leaf so values in both trees should match

            EXPECT_EQ(mint.getValue({-1, -1, -1}), -3);
            EXPECT_EQ(mint.getValue({-1, -1, 0}), 2);
            EXPECT_EQ(mint.getValue({-1, 0, -1}), 1);
            EXPECT_EQ(mint.getValue({-1, 0, 0}), 0);
            EXPECT_EQ(mint.getValue({0, -1, -1}), 3);
            EXPECT_EQ(mint.getValue({0, -1, 1}),-2);
            EXPECT_EQ(mint.getValue({0, 0, -1}), -1);
            EXPECT_EQ(mint.getValue({0, 0, 0}), 0);

            EXPECT_EQ(maxt.getValue({-1, -1, -1}), -3);
            EXPECT_EQ(maxt.getValue({-1, -1, 0}), 2);
            EXPECT_EQ(maxt.getValue({-1, 0, -1}), 1);
            EXPECT_EQ(maxt.getValue({-1, 0, 0}), 0);
            EXPECT_EQ(maxt.getValue({0, -1, -1}), 3);
            EXPECT_EQ(maxt.getValue({0, -1, 1}),-2);
            EXPECT_EQ(maxt.getValue({0, 0, -1}), -1);
            EXPECT_EQ(maxt.getValue({0, 0, 0}), 0);
        }
    }

    // Test min max trees (multiple points in leaf)
    {
        auto points = PointBuilder(getBoxPoints(/*scale*/0.0f)) // 8 points at origin
            .attribute<int32_t>({-3,2,1,0,3,-2,-1,0}, "inttest1")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .get();

        // int32_t
        {
            int32_t min=0, max=0;
            Int32Tree mint, maxt;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, points::NullFilter(), &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-3), min);
            EXPECT_EQ(int32_t(3), max);

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 1);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 1);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({0,0,0}));
            EXPECT_TRUE(maxt.isValueOn({0,0,0}));
            EXPECT_EQ(mint.getValue({0,0,0}), -3);
            EXPECT_EQ(maxt.getValue({0,0,0}), 3);
        }

        // test min max trees filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            int32_t min=0, max=0;
            Int32Tree mint, maxt;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, filter, &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-1), min);
            EXPECT_EQ(int32_t(2), max);

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 1);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 1);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({0,0,0}));
            EXPECT_TRUE(maxt.isValueOn({0,0,0}));
            EXPECT_EQ(mint.getValue({0,0,0}), -1);
            EXPECT_EQ(maxt.getValue({0,0,0}), 2);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            int32_t min=100, max=100;
            Int32Tree mint, maxt;
            bool success = points::evalMinMax<int32_t>(points->tree(), "inttest1", min, max, empty, &mint, &maxt);
            EXPECT_TRUE(!success);
            EXPECT_EQ(min, 100);
            EXPECT_EQ(max, 100);
            EXPECT_TRUE(mint.empty());
            EXPECT_TRUE(maxt.empty());
        }
    }
}


TEST_F(TestPointStatistics, testEvalAverage)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        float avg=-1.0f;
        const bool success = points::evalAverage<float>(points->tree(), "noop", avg);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, avg);
    }

    // Test no attribute
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        float avg=-1.0f;
        const bool success = points::evalAverage<float>(points->tree(), "noop", avg);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, avg);
    }

    // Test invalid attribute Type
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        int32_t avg=-1;
        EXPECT_THROW(points::evalAverage<int32_t>(points->tree(), "test", avg), TypeError);
        EXPECT_EQ(int32_t(-1), avg);
    }

    // Test average
    {
        auto points = PointBuilder(getBoxPoints()) // 8 points
            .attribute<int32_t>({-3,2,1,0,3,-2,-1,0}, "inttest1") // zero crossing
            .attribute<int32_t>({-10,-5,-9,-1,-2,-2,-1,-2}, "inttest2") // all under 0
            .attribute<float>({-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f}, "floattest")
            .attribute<Vec3f>({ Vec3f(0.0f), Vec3f(-0.0f),
                Vec3f(0.3f), Vec3f(1.0f,-0.5f,-0.2f),
                Vec3f(0.2f), Vec3f(0.2f, 0.5f, 0.1f),
                Vec3f(-0.1f), Vec3f(0.1f) }, "vectest")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .voxelsize(1.0)
            .get();

        // int32_t
        {
            int32_t avgi = 0;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avgi);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(0), avgi);

            success = points::evalAverage<int32_t>(points->tree(), "inttest2", avgi);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-4), avgi);

            float avgf = 0.0f;
            success = points::evalAverage<int32_t, float>(points->tree(), "inttest1", avgf);
            EXPECT_TRUE(success);
            EXPECT_EQ(0.0f, avgf);

            success = points::evalAverage<int32_t, float>(points->tree(), "inttest2", avgf);
            EXPECT_TRUE(success);
            EXPECT_EQ(-4.0f, avgf);
        }

        // float
        {
            float avg=0;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.075f, avg);
        }

        // Vec3f
        {
            Vec3f avg(0);
            bool success = points::evalAverage<Vec3f>(points->tree(), "vectest", avg);
            EXPECT_TRUE(success);
            EXPECT_EQ(0.2125f, avg.x());
            EXPECT_EQ(0.0625f, avg.y());
            EXPECT_EQ(0.05f, avg.z());
        }

        // Test avg filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            float avg=0;
            bool success = points::evalAverage<int32_t, float>(points->tree(), "inttest1", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0f/4.0f, avg);

            success = points::evalAverage<int32_t, float>(points->tree(), "inttest2", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(-9.0f/4.0f, avg);
        }

        // float
        {
            float avg=0;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ((5.1f+3.4f+6.2f)/4.0f, avg);
        }

        // Vec3f
        {
            Vec3f avg;
            bool success = points::evalAverage<Vec3f>(points->tree(), "vectest", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_NEAR(1.0f/4.0f, avg.x(), 1e-6);
            EXPECT_NEAR(-0.5f/4.0, avg.y(), 1e-6);
            EXPECT_NEAR(-0.2f/4.0f, avg.z(), 1e-6);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            int32_t avg=100;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avg, empty);
            EXPECT_TRUE(!success);
            EXPECT_EQ(avg, 100);
        }

        // Test avg trees

        // int32_t
        {
            int32_t avg=0;
            Int32Tree avgt;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avg, points::NullFilter(), &avgt);
            EXPECT_TRUE(success);
            EXPECT_EQ(0, avg);

            EXPECT_EQ(avgt.leafCount(), 0);
            EXPECT_EQ(avgt.activeTileCount(), 8);

            EXPECT_TRUE(avgt.isValueOn({-1, -1, -1}));
            EXPECT_TRUE(avgt.isValueOn({-1, -1, 0}));
            EXPECT_TRUE(avgt.isValueOn({-1, 0, -1}));
            EXPECT_TRUE(avgt.isValueOn({-1, 0, 0}));
            EXPECT_TRUE(avgt.isValueOn({0, -1, -1}));
            EXPECT_TRUE(avgt.isValueOn({0, -1, 1}));
            EXPECT_TRUE(avgt.isValueOn({0, 0, -1}));
            EXPECT_TRUE(avgt.isValueOn({0, 0, 0}));

            // only 1 point per leaf so avg = points value

            EXPECT_EQ(avgt.getValue({-1, -1, -1}), -3);
            EXPECT_EQ(avgt.getValue({-1, -1, 0}), 2);
            EXPECT_EQ(avgt.getValue({-1, 0, -1}), 1);
            EXPECT_EQ(avgt.getValue({-1, 0, 0}), 0);
            EXPECT_EQ(avgt.getValue({0, -1, -1}), 3);
            EXPECT_EQ(avgt.getValue({0, -1, 1}),-2);
            EXPECT_EQ(avgt.getValue({0, 0, -1}), -1);
            EXPECT_EQ(avgt.getValue({0, 0, 0}), 0);
        }
    }

    // Test avg trees (multiple points in leaf)
    {
        auto points = PointBuilder(getBoxPoints(/*scale*/0.0f)) // 8 points at origin
            .attribute<float>({-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f}, "floattest")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .get();

        // float
        {
            float avg=0;
            FloatTree avgt;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, points::NullFilter(), &avgt);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.075f, avg);

            EXPECT_EQ(avgt.leafCount(), 0);
            EXPECT_EQ(avgt.activeTileCount(), 1);
            EXPECT_TRUE(avgt.isValueOn({0,0,0}));
            EXPECT_EQ(avgt.getValue({0,0,0}), 1.075f);
        }

        // test avg trees filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // float
        {
            float avg=0;
            FloatTree avgt;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, filter, &avgt);
            EXPECT_TRUE(success);
            EXPECT_EQ((5.1f+3.4f+6.2f)/4.0f, avg);

            EXPECT_EQ(avgt.leafCount(), 0);
            EXPECT_EQ(avgt.activeTileCount(), 1);
            EXPECT_TRUE(avgt.isValueOn({0,0,0}));
            EXPECT_EQ(avgt.getValue({0,0,0}), (5.1f+3.4f+6.2f)/4.0f);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            float avg=100;
            FloatTree avgt;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, empty, &avgt);
            EXPECT_TRUE(!success);
            EXPECT_EQ(avg, 100);
            EXPECT_TRUE(avgt.empty());
        }
    }
}


TEST_F(TestPointStatistics, testAccumulate)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        float total=-1.0f;
        const bool success = points::accumulate<float>(points->tree(), "noop", total);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, total);
    }

    // Test no attribute
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        float total=-1.0f;
        const bool success = points::accumulate<float>(points->tree(), "noop", total);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, total);
    }

    // Test invalid attribute Type
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        int32_t total=-1;
        EXPECT_THROW(points::accumulate<int32_t>(points->tree(), "test", total), TypeError);
        EXPECT_EQ(int32_t(-1), total);
    }

    // Test accumulate
    {
        auto points = PointBuilder(getBoxPoints()) // 8 points
            .attribute<int32_t>({-3,2,1,0,3,-2,-1,0}, "inttest1") // zero crossing
            .attribute<int32_t>({-10,-5,-9,-1,-2,-2,-1,-2}, "inttest2") // all under 0
            .attribute<float>({-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f}, "floattest")
            .attribute<Vec3f>({ Vec3f(0.0f), Vec3f(-0.0f),
                Vec3f(0.3f), Vec3f(1.0f,-0.5f,-0.2f),
                Vec3f(0.2f), Vec3f(0.2f, 0.5f, 0.1f),
                Vec3f(-0.1f), Vec3f(0.1f) }, "vectest")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .voxelsize(1.0)
            .get();

        // int32_t
        {
            int32_t totali = 1;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", totali);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(0), totali);

            success = points::accumulate<int32_t>(points->tree(), "inttest2", totali);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-32), totali);

            float totalf = 1.0f;
            success = points::accumulate<int32_t, float>(points->tree(), "inttest1", totalf);
            EXPECT_TRUE(success);
            EXPECT_EQ(0.0f, totalf);

            success = points::accumulate<int32_t, float>(points->tree(), "inttest2", totalf);
            EXPECT_TRUE(success);
            EXPECT_EQ(-32.0f, totalf);
        }

        // float
        {
            float total=0;
            bool success = points::accumulate<float>(points->tree(), "floattest", total);
            EXPECT_TRUE(success);
            EXPECT_EQ(-4.3f+5.1f+-1.1f+0.0f+9.5f+-10.2f+3.4f+6.2f, total);
        }

        // Vec3f
        {
            Vec3f total(0);
            bool success = points::accumulate<Vec3f>(points->tree(), "vectest", total);
            EXPECT_TRUE(success);
            Vec3f r = Vec3f(0.3f) + Vec3f(1.0f,-0.5f,-0.2f) + Vec3f(0.2f) + Vec3f(0.2f, 0.5f, 0.1f) + Vec3f(-0.1f) + Vec3f(0.1f);
            EXPECT_EQ(r, total);
        }

        // Test accumulate filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            float total=0;
            bool success = points::accumulate<int32_t, float>(points->tree(), "inttest1", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0f, total);

            success = points::accumulate<int32_t, float>(points->tree(), "inttest2", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(-9.0f, total);
        }

        // float
        {
            float total=0;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(5.1f+3.4f+6.2f, total);
        }

        // Vec3f
        {
            Vec3f total;
            bool success = points::accumulate<Vec3f>(points->tree(), "vectest", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0f, total.x());
            EXPECT_NEAR(-0.5f, total.y(), 1e-6);
            EXPECT_NEAR(-0.2f, total.z(), 1e-6);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            int32_t total=100;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", total, empty);
            EXPECT_TRUE(!success);
            EXPECT_EQ(total, 100);
        }

        // Test total trees

        // int32_t
        {
            int32_t total=0;
            Int32Tree totalt;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", total, points::NullFilter(), &totalt);
            EXPECT_TRUE(success);
            EXPECT_EQ(0, total);

            EXPECT_EQ(totalt.leafCount(), 0);
            EXPECT_EQ(totalt.activeTileCount(), 8);

            EXPECT_TRUE(totalt.isValueOn({-1, -1, -1}));
            EXPECT_TRUE(totalt.isValueOn({-1, -1, 0}));
            EXPECT_TRUE(totalt.isValueOn({-1, 0, -1}));
            EXPECT_TRUE(totalt.isValueOn({-1, 0, 0}));
            EXPECT_TRUE(totalt.isValueOn({0, -1, -1}));
            EXPECT_TRUE(totalt.isValueOn({0, -1, 1}));
            EXPECT_TRUE(totalt.isValueOn({0, 0, -1}));
            EXPECT_TRUE(totalt.isValueOn({0, 0, 0}));

            // only 1 point per leaf so total = points value

            EXPECT_EQ(totalt.getValue({-1, -1, -1}), -3);
            EXPECT_EQ(totalt.getValue({-1, -1, 0}), 2);
            EXPECT_EQ(totalt.getValue({-1, 0, -1}), 1);
            EXPECT_EQ(totalt.getValue({-1, 0, 0}), 0);
            EXPECT_EQ(totalt.getValue({0, -1, -1}), 3);
            EXPECT_EQ(totalt.getValue({0, -1, 1}),-2);
            EXPECT_EQ(totalt.getValue({0, 0, -1}), -1);
            EXPECT_EQ(totalt.getValue({0, 0, 0}), 0);
        }
    }

    // Test total trees (multiple points in leaf)
    {
        auto points = PointBuilder(getBoxPoints(/*scale*/0.0f)) // 8 points at origin
            .attribute<float>({-4.3f,5.1f,-1.1f,0.0f,9.5f,-10.2f,3.4f,6.2f}, "floattest")
            .group({0,1,0,1,0,0,1,1}, "group1")
            .group({0,0,0,0,0,0,0,0}, "empty")
            .get();

        // float
        {
            float total=0;
            FloatTree totalt;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, points::NullFilter(), &totalt);
            EXPECT_TRUE(success);
            EXPECT_EQ(-4.3f+5.1f+-1.1f+0.0f+9.5f+-10.2f+3.4f+6.2f, total);

            EXPECT_EQ(totalt.leafCount(), 0);
            EXPECT_EQ(totalt.activeTileCount(), 1);
            EXPECT_TRUE(totalt.isValueOn({0,0,0}));
            EXPECT_EQ(totalt.getValue({0,0,0}), -4.3f+5.1f+-1.1f+0.0f+9.5f+-10.2f+3.4f+6.2f);
        }

        // test total trees filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // float
        {
            float total=0;
            FloatTree totalt;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, filter, &totalt);
            EXPECT_TRUE(success);
            EXPECT_EQ(5.1f+3.4f+6.2f, total);

            EXPECT_EQ(totalt.leafCount(), 0);
            EXPECT_EQ(totalt.activeTileCount(), 1);
            EXPECT_TRUE(totalt.isValueOn({0,0,0}));
            EXPECT_EQ(totalt.getValue({0,0,0}), 5.1f+3.4f+6.2f);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            float total=100;
            FloatTree totalt;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, empty, &totalt);
            EXPECT_TRUE(!success);
            EXPECT_EQ(total, 100);
            EXPECT_TRUE(totalt.empty());
        }
    }
}

template <typename VecT>
struct AbsExtent
{
    using ExtentT = std::pair<VecT, VecT>;
    AbsExtent(const VecT& init) : mMinMax(math::Abs(init), math::Abs(init)) {}
    AbsExtent(const ExtentT& init) : mMinMax(init) {}
    inline const ExtentT& get() const { return mMinMax; }
    inline void operator()(const VecT& b)
    {
        auto in = math::Abs(b);
        mMinMax.first = math::minComponent(mMinMax.first, in);
        mMinMax.second =  math::maxComponent(mMinMax.second, in);
    }
    inline void operator()(const ExtentT& b)
    {
        mMinMax.first = math::minComponent(mMinMax.first, b.first);
        mMinMax.second = math::maxComponent(mMinMax.second, b.second);
    }
    ExtentT mMinMax;
};


TEST_F(TestPointStatistics, testEvalExtents)
{
    // Test direct invocation of evalExtents

    // Test Vec3f->float magnitude
    {
        auto points = PointBuilder(getBoxPoints(/*scale=*/0.0)) // 8 points
            .attribute<Vec3f>({
                    Vec3f(0.0f), Vec3f(-0.0f), Vec3f(0.3f),
                    Vec3f(1.0f,-0.5f,-0.2f), Vec3f(0.2f),
                    Vec3f(0.2f, 0.5f, 0.1f), Vec3f(-0.1f),
                    Vec3f(0.1f),
                }, "vectest")
            .get();


        // Abs of vector componentwise

        {
            AbsExtent<Vec3f>::ExtentT p;
            Vec3fTree mint, maxt;
            const bool success = points::statistics_internal::evalExtents
                <Vec3f, points::NullCodec, points::NullFilter, AbsExtent<Vec3f>, points::PointDataTree>
                    (points->tree(), "vectest", p, points::NullFilter(), &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(p.first.x(), 0.0f);
            EXPECT_EQ(p.first.y(), 0.0f);
            EXPECT_EQ(p.first.z(), 0.0f);
            EXPECT_EQ(p.second.x(), 1.0f);
            EXPECT_EQ(p.second.y(), 0.5f);
            EXPECT_EQ(p.second.z(), 0.3f);

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 1);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 1);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({0,0,0}));
            EXPECT_TRUE(maxt.isValueOn({0,0,0}));
            EXPECT_EQ(mint.getValue({0,0,0}), Vec3f(0.0f));
            EXPECT_EQ(maxt.getValue({0,0,0}), Vec3f(1.0f, 0.5f, 0.3f));
        }

        // Vector magnitudes as length sqr

        {
            using OpT = points::statistics_internal::MagnitudeExtent<Vec3f>;

            OpT::ExtentT p;
            FloatTree mint, maxt;
            const bool success = points::statistics_internal::evalExtents
                <Vec3f, points::NullCodec, points::NullFilter, OpT, points::PointDataTree>
                    (points->tree(), "vectest", p, points::NullFilter(), &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(p.first, 0.0f);
            EXPECT_EQ(p.second, Vec3f(1.0f,-0.5f,-0.2f).lengthSqr());

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 1);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 1);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({0,0,0}));
            EXPECT_TRUE(maxt.isValueOn({0,0,0}));
            EXPECT_EQ(mint.getValue({0,0,0}), 0.0f);
            EXPECT_EQ(maxt.getValue({0,0,0}), Vec3f(1.0f,-0.5f,-0.2f).lengthSqr());
        }

        // Vector magnitude as component values

        {
            using OpT = points::statistics_internal::MagnitudeExtent<Vec3f, false>;

            OpT::ExtentT p;
            Vec3fTree mint, maxt;
            const bool success = points::statistics_internal::evalExtents
                <Vec3f, points::NullCodec, points::NullFilter, OpT, points::PointDataTree>
                    (points->tree(), "vectest", p, points::NullFilter(),  &mint, &maxt);
            EXPECT_TRUE(success);
            EXPECT_EQ(p.first.x(), 0.0f);
            EXPECT_EQ(p.first.y(), 0.0f);
            EXPECT_EQ(p.first.z(), 0.0f);
            EXPECT_EQ(p.second.x(), 1.0f);
            EXPECT_EQ(p.second.y(),-0.5f);
            EXPECT_EQ(p.second.z(),-0.2f);

            EXPECT_EQ(mint.leafCount(), 0);
            EXPECT_EQ(mint.activeTileCount(), 1);
            EXPECT_EQ(maxt.leafCount(), 0);
            EXPECT_EQ(maxt.activeTileCount(), 1);
            EXPECT_TRUE(mint.hasSameTopology(maxt));

            EXPECT_TRUE(mint.isValueOn({0,0,0}));
            EXPECT_TRUE(maxt.isValueOn({0,0,0}));
            EXPECT_EQ(mint.getValue({0,0,0}), Vec3f(0.0f));
            EXPECT_EQ(maxt.getValue({0,0,0}), Vec3f(1.0f,-0.5f,-0.2f));
        }

    }
}
