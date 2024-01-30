// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
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
