// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/math/BBox.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>

#include <gtest/gtest.h>

typedef float Real;

class TestBBox: public ::testing::Test
{
};


TEST_F(TestBBox, testBBox)
{
    typedef openvdb::Vec3R                     Vec3R;
    typedef openvdb::math::BBox<Vec3R>         BBoxType;

    {
        BBoxType B(Vec3R(1,1,1),Vec3R(2,2,2));

        EXPECT_TRUE(B.isSorted());
        EXPECT_TRUE(B.isInside(Vec3R(1.5,2,2)));
        EXPECT_TRUE(!B.isInside(Vec3R(2,3,2)));
        B.expand(Vec3R(3,3,3));
        EXPECT_TRUE(B.isInside(Vec3R(3,3,3)));
    }

    {
        BBoxType B;
        EXPECT_TRUE(B.empty());
        const Vec3R expected(1);
        B.expand(expected);
        EXPECT_EQ(expected, B.min());
        EXPECT_EQ(expected, B.max());
    }
}


TEST_F(TestBBox, testCenter)
{
    using namespace openvdb::math;

    const Vec3<double> expected(1.5);

    BBox<openvdb::Vec3R> fbox(openvdb::Vec3R(1.0), openvdb::Vec3R(2.0));
    EXPECT_EQ(expected, fbox.getCenter());

    BBox<openvdb::Vec3i> ibox(openvdb::Vec3i(1), openvdb::Vec3i(2));
    EXPECT_EQ(expected, ibox.getCenter());

    openvdb::CoordBBox cbox(openvdb::Coord(1), openvdb::Coord(2));
    EXPECT_EQ(expected, cbox.getCenter());
}

TEST_F(TestBBox, testExtent)
{
    typedef openvdb::Vec3R                     Vec3R;
    typedef openvdb::math::BBox<Vec3R>         BBoxType;

    {
        BBoxType B(Vec3R(-20,0,1),Vec3R(2,2,2));
        EXPECT_EQ(size_t(2), B.minExtent());
        EXPECT_EQ(size_t(0), B.maxExtent());
    }
    {
        BBoxType B(Vec3R(1,0,1),Vec3R(2,21,20));
        EXPECT_EQ(size_t(0), B.minExtent());
        EXPECT_EQ(size_t(1), B.maxExtent());
    }
    {
        BBoxType B(Vec3R(1,0,1),Vec3R(3,1.5,20));
        EXPECT_EQ(size_t(1), B.minExtent());
        EXPECT_EQ(size_t(2), B.maxExtent());
    }
}
