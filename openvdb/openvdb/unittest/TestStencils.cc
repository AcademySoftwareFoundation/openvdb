// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>

#include <gtest/gtest.h>


class TestStencils : public ::testing::Test
{
};

TEST_F(TestStencils, testMinMax)
{
    using namespace openvdb;

    Int32Grid grid;
    grid.tree().setValue(math::Coord(0,0,0), -3);
    grid.tree().setValue(math::Coord(0,0,1), -2);
    grid.tree().setValue(math::Coord(0,1,0), -1);
    grid.tree().setValue(math::Coord(1,0,0),  0);
    grid.tree().setValue(math::Coord(1,1,0),  1);
    grid.tree().setValue(math::Coord(0,1,1),  2);
    grid.tree().setValue(math::Coord(1,0,1),  3);
    grid.tree().setValue(math::Coord(1,1,1),  4);
    math::BoxStencil<Int32Grid> stencil(grid);

    stencil.moveTo(Coord(0,0,0));
    EXPECT_EQ(stencil.min(), -3);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(1,1,1));
    EXPECT_EQ(stencil.min(), 0);
    EXPECT_EQ(stencil.max(), 4);

    stencil.moveTo(Coord(0,0,1));
    EXPECT_EQ(stencil.min(), -2);
    EXPECT_EQ(stencil.max(),  4);
}
