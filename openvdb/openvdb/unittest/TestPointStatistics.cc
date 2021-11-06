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

        // basic signature returns zeroVal
        auto int_result = points::evalMinMax<int32_t>(points->tree(), "noop");
        EXPECT_EQ(int32_t(0), int_result.first);
        EXPECT_EQ(int32_t(0), int_result.second);
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
        EXPECT_ANY_THROW(points::evalMinMax<int32_t>(points->tree(), "test", min, max));
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

        // test basic signature (int32_t, float, Ve3f)
        {
            auto int_result = points::evalMinMax<int32_t>(points->tree(), "inttest1");
            EXPECT_EQ(int32_t(-3), int_result.first);
            EXPECT_EQ(int32_t(3), int_result.second);

            auto flt_result = points::evalMinMax<float>(points->tree(), "floattest");
            EXPECT_EQ(float(-10.2f), flt_result.first);
            EXPECT_EQ(float(9.5f), flt_result.second);

            auto vec_result = points::evalMinMax<Vec3f>(points->tree(), "vectest");
            EXPECT_EQ(-0.1f, vec_result.first.x());
            EXPECT_EQ(-0.5f, vec_result.first.y());
            EXPECT_EQ(-0.2f, vec_result.first.z());
            EXPECT_EQ(1.0f, vec_result.second.x());
            EXPECT_EQ(0.5f, vec_result.second.y());
            EXPECT_EQ(0.3f, vec_result.second.z());
        }

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

            // basic signature returns zeroVal
            auto int_result = points::evalMinMax<int32_t>(points->tree(), "inttest1", empty);
            EXPECT_EQ(int32_t(0), int_result.first);
            EXPECT_EQ(int32_t(0), int_result.second);
        }
    }
}


template <typename PointsT, typename ValueT>
using ResultTree =
    typename std::decay<PointsT>::type::TreeType::template ValueConverter<ValueT>::Type;


TEST_F(TestPointStatistics, testEvalAverage)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        ConvertElementType<float, double>::Type avg = -1.0;
        const bool success = points::evalAverage<float>(points->tree(), "noop", avg);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0, avg);

        // basic signature returns zeroVal
        auto float_result = points::evalAverage<float>(points->tree(), "noop");
        EXPECT_EQ(0.0, float_result);
    }

    // Test no attribute
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        ConvertElementType<float, double>::Type avg=-1.0;
        const bool success = points::evalAverage<float>(points->tree(), "noop", avg);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0, avg);
    }

    // Test invalid attribute Type
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        ConvertElementType<int32_t, double>::Type avg=-1;
        EXPECT_ANY_THROW(points::evalAverage<int32_t>(points->tree(), "test", avg));
        EXPECT_EQ(-1.0, avg);
    }

    // Test one point
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(100.0f, "test1")
            .attribute<int>(200, "test2")
            .get();
        ConvertElementType<float, double>::Type avgf = -1.0f;
        bool success = points::evalAverage<float>(points->tree(), "test1", avgf);
        EXPECT_TRUE(success);
        EXPECT_EQ(100.0, avgf);

        ConvertElementType<int, double>::Type avgi=-1;
        success = points::evalAverage<int>(points->tree(), "test2", avgi);
        EXPECT_TRUE(success);
        EXPECT_EQ(200, avgi);
    }

    // Test average
    {
        // different point counts in different nodes
        // with a voxel size of 1.0, creates 3 leaf nodes with 4,3,1 points
        std::vector<openvdb::Vec3f> boxPoints {
            { 1,1,1}, { 2,2,2}, { 3,3,3}, {4,4,4},
            {-1,1,1}, {-2,1,1}, {-3,1,1},
            {1,-1,1}
        };

        auto points = PointBuilder(boxPoints) // 8 points
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

        // test basic signature (int32_t, float, Ve3f)
        {
            auto int_result = points::evalAverage<int32_t>(points->tree(), "inttest1");
            EXPECT_EQ(0.0, int_result);

            auto flt_result = points::evalAverage<float>(points->tree(), "floattest");
            EXPECT_NEAR(1.075, flt_result, 1e-6);

            auto vec_result = points::evalAverage<Vec3f>(points->tree(), "vectest");
            EXPECT_NEAR(0.2125, vec_result.x(), 1e-6);
            EXPECT_NEAR(0.0625, vec_result.y(), 1e-6);
            EXPECT_NEAR(0.05,   vec_result.z(), 1e-6);
        }

        // int32_t
        {
            ConvertElementType<int32_t, double>::Type avgi = 0;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avgi);
            EXPECT_TRUE(success);
            EXPECT_EQ(0.0, avgi);

            success = points::evalAverage<int32_t>(points->tree(), "inttest2", avgi);
            EXPECT_TRUE(success);
            EXPECT_EQ(-4.0, avgi);
        }

        // float
        {
            ConvertElementType<float, double>::Type avg = 0;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg);
            EXPECT_TRUE(success);
            EXPECT_NEAR(1.075, avg, 1e-6);
        }

        // Vec3f
        {
            ConvertElementType<Vec3f, double>::Type avg(0);
            bool success = points::evalAverage<Vec3f>(points->tree(), "vectest", avg);
            EXPECT_TRUE(success);
            EXPECT_NEAR(0.2125, avg.x(), 1e-6);
            EXPECT_NEAR(0.0625, avg.y(), 1e-6);
            EXPECT_NEAR(0.05,   avg.z(), 1e-6);
        }

        // Test avg filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            ConvertElementType<int32_t, double>::Type avg=0;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0/4.0, avg);

            success = points::evalAverage<int32_t>(points->tree(), "inttest2", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(-9.0/4.0, avg);
        }

        // float
        {
            ConvertElementType<float, double>::Type avg=0;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ((5.1f+3.4f+6.2f)/4.0f, avg);
        }

        // Vec3f
        {
            ConvertElementType<Vec3f, double>::Type avg;
            bool success = points::evalAverage<Vec3f>(points->tree(), "vectest", avg, filter);
            EXPECT_TRUE(success);
            EXPECT_NEAR(1.0f/4.0f, avg.x(), 1e-6);
            EXPECT_NEAR(-0.5f/4.0, avg.y(), 1e-6);
            EXPECT_NEAR(-0.2f/4.0f, avg.z(), 1e-6);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            ConvertElementType<int32_t, double>::Type avg=100;
            bool success = points::evalAverage<int32_t>(points->tree(), "inttest1", avg, empty);
            EXPECT_TRUE(!success);
            EXPECT_EQ(avg, 100);
        }

        // Test avg trees

        // int32_t
        {
            ConvertElementType<int32_t, double>::Type avg=0;
            ResultTree<decltype(*points), decltype(avg)> avgt;

            bool success = points::evalAverage<int32_t>(points->tree(), "inttest2", avg, points::NullFilter(), &avgt);
            EXPECT_TRUE(success);
            EXPECT_EQ(-4.0, avg);

            EXPECT_EQ(avgt.leafCount(), 0);
            EXPECT_EQ(avgt.activeTileCount(), 3);

            EXPECT_TRUE(avgt.isValueOn({-8, 0, 0}));
            EXPECT_TRUE(avgt.isValueOn({ 0,-8, 0}));
            EXPECT_TRUE(avgt.isValueOn({ 0, 0, 0}));

            EXPECT_NEAR(avgt.getValue({-8, 0, 0}), -1.66667, 1e-4);
            EXPECT_EQ(avgt.getValue({ 0,-8, 0}), -2);
            EXPECT_EQ(avgt.getValue({ 0, 0, 0}), -6.25);
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
            ConvertElementType<float, double>::Type avg=0;
            ResultTree<decltype(*points), decltype(avg)> avgt;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, points::NullFilter(), &avgt);
            EXPECT_TRUE(success);
            EXPECT_NEAR(1.075, avg, 1e-6);

            EXPECT_EQ(avgt.leafCount(), 0);
            EXPECT_EQ(avgt.activeTileCount(), 1);
            EXPECT_TRUE(avgt.isValueOn({0,0,0}));
            EXPECT_NEAR(avgt.getValue({0,0,0}), 1.075f, 1e-6);
        }

        // test avg trees filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // float
        {
            ConvertElementType<float, double>::Type avg=0;
            ResultTree<decltype(*points), decltype(avg)> avgt;
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
            ConvertElementType<float, double>::Type avg=100;
            ResultTree<decltype(*points), decltype(avg)> avgt;
            bool success = points::evalAverage<float>(points->tree(), "floattest", avg, empty, &avgt);
            EXPECT_TRUE(!success);
            EXPECT_EQ(avg, 100);
            EXPECT_TRUE(avgt.empty());

            // basic signature returns zeroVal
            auto float_result = points::evalAverage<float>(points->tree(), "floattest", empty);
            EXPECT_EQ(0.0, float_result);
        }
    }
}


TEST_F(TestPointStatistics, testAccumulate)
{
    // Test no points
    {
        const auto points = PointBuilder({}).get();
        PromoteType<float>::Highest total=-1.0f;
        const bool success = points::accumulate<float>(points->tree(), "noop", total);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, total);

        // basic signature returns zeroVal
        auto float_result = points::accumulate<float>(points->tree(), "noop");
        EXPECT_EQ(0.0, float_result);
    }

    // Test no attribute
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        PromoteType<float>::Highest total=-1.0f;
        const bool success = points::accumulate<float>(points->tree(), "noop", total);
        EXPECT_TRUE(!success);
        EXPECT_EQ(-1.0f, total);
    }

    // Test invalid attribute Type
    {
        auto points = PointBuilder({Vec3f(0)})
            .attribute<float>(zeroVal<float>(), "test")
            .get();
        PromoteType<int32_t>::Highest total=-1;
        EXPECT_ANY_THROW(points::accumulate<int32_t>(points->tree(), "test", total));
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

        // test basic signature (int32_t, float, Ve3f)
        {
            auto int_result = points::accumulate<int32_t>(points->tree(), "inttest1");
            EXPECT_EQ(0.0, int_result);

            auto flt_result = points::accumulate<float>(points->tree(), "floattest");
            EXPECT_NEAR(-4.3+5.1+-1.1+0.0+9.5+-10.2+3.4+6.2, flt_result, 1e-6);

            auto vec_result = points::accumulate<Vec3f>(points->tree(), "vectest");
            Vec3d r = Vec3d(0.3) + Vec3d(1.0,-0.5,-0.2) + Vec3d(0.2) + Vec3d(0.2, 0.5, 0.1) + Vec3d(-0.1) + Vec3d(0.1);
            EXPECT_NEAR(r.x(), vec_result.x(), 1e-6);
            EXPECT_NEAR(r.y(), vec_result.y(), 1e-6);
            EXPECT_NEAR(r.z(), vec_result.z(), 1e-6);
        }

        // int32_t
        {
            PromoteType<int32_t>::Highest totali = 1;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", totali);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(0), totali);

            success = points::accumulate<int32_t>(points->tree(), "inttest2", totali);
            EXPECT_TRUE(success);
            EXPECT_EQ(int32_t(-32), totali);

            PromoteType<int32_t>::Highest totalf = 1.0f;
            success = points::accumulate<int32_t>(points->tree(), "inttest1", totalf);
            EXPECT_TRUE(success);
            EXPECT_EQ(0.0f, totalf);

            success = points::accumulate<int32_t>(points->tree(), "inttest2", totalf);
            EXPECT_TRUE(success);
            EXPECT_EQ(-32.0f, totalf);
        }

        // float
        {
            PromoteType<float>::Highest total=0;
            bool success = points::accumulate<float>(points->tree(), "floattest", total);
            EXPECT_TRUE(success);
            EXPECT_NEAR(-4.3+5.1+-1.1+0.0+9.5+-10.2+3.4+6.2, total, 1e-6);
        }

        // Vec3f
        {
            PromoteType<Vec3f>::Highest total(0);
            bool success = points::accumulate<Vec3f>(points->tree(), "vectest", total);
            EXPECT_TRUE(success);
            Vec3d r = Vec3d(0.3) + Vec3d(1.0,-0.5,-0.2) + Vec3d(0.2) + Vec3d(0.2, 0.5, 0.1) + Vec3d(-0.1) + Vec3d(0.1);
            EXPECT_NEAR(r.x(), total.x(), 1e-6);
            EXPECT_NEAR(r.y(), total.y(), 1e-6);
            EXPECT_NEAR(r.z(), total.z(), 1e-6);
        }

        // Test accumulate filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // int32_t
        {
            PromoteType<int32_t>::Highest total=0;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0f, total);

            success = points::accumulate<int32_t>(points->tree(), "inttest2", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(-9.0f, total);
        }

        // float
        {
            PromoteType<float>::Highest total=0;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(5.1f+3.4f+6.2f, total);
        }

        // Vec3f
        {
            PromoteType<Vec3f>::Highest total;
            bool success = points::accumulate<Vec3f>(points->tree(), "vectest", total, filter);
            EXPECT_TRUE(success);
            EXPECT_EQ(1.0f, total.x());
            EXPECT_NEAR(-0.5f, total.y(), 1e-6);
            EXPECT_NEAR(-0.2f, total.z(), 1e-6);
        }

        // test no valid points in filter
        {
            points::GroupFilter empty("empty", points->tree().cbeginLeaf()->attributeSet());
            PromoteType<int32_t>::Highest total=100;
            bool success = points::accumulate<int32_t>(points->tree(), "inttest1", total, empty);
            EXPECT_TRUE(!success);
            EXPECT_EQ(total, 100);
        }

        // Test total trees

        // int32_t
        {
            PromoteType<int32_t>::Highest total=0;
            ResultTree<decltype(*points), decltype(total)> totalt;
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
            PromoteType<float>::Highest total=0;
            ResultTree<decltype(*points), decltype(total)> totalt;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, points::NullFilter(), &totalt);
            EXPECT_TRUE(success);
            EXPECT_NEAR(-4.3+5.1+-1.1+0.0+9.5+-10.2+3.4+6.2, total, 1e-6);

            EXPECT_EQ(totalt.leafCount(), 0);
            EXPECT_EQ(totalt.activeTileCount(), 1);
            EXPECT_TRUE(totalt.isValueOn({0,0,0}));
            EXPECT_NEAR(totalt.getValue({0,0,0}), -4.3+5.1+-1.1+0.0+9.5+-10.2+3.4+6.2, 1e-6);
        }

        // test total trees filter

        points::GroupFilter filter("group1", points->tree().cbeginLeaf()->attributeSet());

        // float
        {
            PromoteType<float>::Highest total=0;
            ResultTree<decltype(*points), decltype(total)> totalt;
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
            PromoteType<float>::Highest total=100;
            ResultTree<decltype(*points), decltype(total)> totalt;
            bool success = points::accumulate<float>(points->tree(), "floattest", total, empty, &totalt);
            EXPECT_TRUE(!success);
            EXPECT_EQ(total, 100);
            EXPECT_TRUE(totalt.empty());

            // basic signature returns zeroVal
            auto float_result = points::accumulate<float>(points->tree(), "noop", empty);
            EXPECT_EQ(0.0, float_result);
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
