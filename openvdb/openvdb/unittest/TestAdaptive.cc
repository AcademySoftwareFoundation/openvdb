// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/PointAdvect.h>
#include <openvdb/adaptive/AdaptiveGrid.h>

#include <gtest/gtest.h>

class TestAdaptive: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::initialize(); }
};


////////////////////////////////////////


TEST_F(TestAdaptive, test)
{
    openvdb::adaptive::AdaptiveGrid<float> adaptiveGrid(5.0f);

    EXPECT_EQ(adaptiveGrid.background(), 5.0f);
}

TEST_F(TestAdaptive, testSample)
{
    const float background = 5.0f;
    openvdb::adaptive::AdaptiveGrid<float> adaptiveGrid(background);

    float result = openvdb::tools::BoxSampler::sample(adaptiveGrid.tree(), openvdb::Vec3R(1.3, 1.6, 1.8));

    EXPECT_EQ(result, background);
}

TEST_F(TestAdaptive, testAdvect)
{
    using AdaptiveGridT = openvdb::adaptive::AdaptiveGrid<openvdb::Vec3s>;
    using PointAdvectT = openvdb::tools::PointAdvect<AdaptiveGridT>;
    using PointListT = PointAdvectT::PointListType;

    const openvdb::Vec3s background(0.0f, 1.0f, 0.0f);
    AdaptiveGridT adaptiveGrid(background);

    openvdb::tools::PointAdvect<AdaptiveGridT> pointAdvect(adaptiveGrid);

    PointListT points;
    points.push_back(openvdb::Vec3s(0.0f, 0.0f, 0.0f));
    points.push_back(openvdb::Vec3s(1.0f, 2.0f, 3.0f));

    float dt = 1/24.0f;
    pointAdvect.advect(points, dt);

    EXPECT_EQ(points[0], openvdb::Vec3s(0.0f, dt, 0.0f));
    EXPECT_EQ(points[1], openvdb::Vec3s(1.0f, 2.0f + dt, 3.0f));
}
