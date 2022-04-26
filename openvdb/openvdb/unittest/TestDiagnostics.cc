// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>
#include <openvdb/math/Stats.h>
#include <openvdb/tools/Diagnostics.h>
#include <openvdb/tools/Statistics.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/LevelSetUtil.h>

#include <gtest/gtest.h>

#include <limits>

class TestDiagnostics: public ::testing::Test
{
};


////////////////////////////////////////

TEST_F(TestDiagnostics, testCheck)
{
    const float val = 1.0f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf1= std::numeric_limits<float>::infinity();
    const openvdb::math::Vec3<float> inf2(val, inf1, val);

    {//test CheckNan
        openvdb::tools::CheckNan<openvdb::FloatGrid> c;
        EXPECT_TRUE(!c(val));
        EXPECT_TRUE( c(nan));
        EXPECT_TRUE( c(nan));
        EXPECT_TRUE(!c(inf1));
        EXPECT_TRUE(!c(inf2));
    }
    {//test CheckInf
        openvdb::tools::CheckInf<openvdb::FloatGrid> c;
        EXPECT_TRUE(!c(val));
        EXPECT_TRUE(!c(nan));
        EXPECT_TRUE(!c(nan));
        EXPECT_TRUE( c(inf1));
        EXPECT_TRUE( c(inf2));
    }
    {//test CheckFinite
        openvdb::tools::CheckFinite<openvdb::FloatGrid> c;
        EXPECT_TRUE(!c(val));
        EXPECT_TRUE( c(nan));
        EXPECT_TRUE( c(nan));
        EXPECT_TRUE( c(inf1));
        EXPECT_TRUE( c(inf2));
    }
    {//test CheckMin
        openvdb::tools::CheckMin<openvdb::FloatGrid> c(0.0f);
        EXPECT_TRUE(!c( 0.5f));
        EXPECT_TRUE(!c( 0.0f));
        EXPECT_TRUE(!c( 1.0f));
        EXPECT_TRUE(!c( 1.1f));
        EXPECT_TRUE( c(-0.1f));
    }
    {//test CheckMax
        openvdb::tools::CheckMax<openvdb::FloatGrid> c(0.0f);
        EXPECT_TRUE( c( 0.5f));
        EXPECT_TRUE(!c( 0.0f));
        EXPECT_TRUE( c( 1.0f));
        EXPECT_TRUE( c( 1.1f));
        EXPECT_TRUE(!c(-0.1f));
    }
    {//test CheckRange
        // first check throw on construction from an invalid range
        EXPECT_THROW(openvdb::tools::CheckRange<openvdb::FloatGrid> c(1.0f, 0.0f),
                             openvdb::ValueError);
        openvdb::tools::CheckRange<openvdb::FloatGrid> c(0.0f, 1.0f);
        EXPECT_TRUE(!c(0.5f));
        EXPECT_TRUE(!c(0.0f));
        EXPECT_TRUE(!c(1.0f));
        EXPECT_TRUE( c(1.1f));
        EXPECT_TRUE(c(-0.1f));
    }
}//testCheck

TEST_F(TestDiagnostics, testDiagnose)
{
    using namespace openvdb;
    const float val = 1.0f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    {//empty grid
        FloatGrid grid;
        tools::Diagnose<FloatGrid> d(grid);
        tools::CheckNan<FloatGrid> c;
        std::string str = d.check(c);
        //std::cerr << "Empty grid:\n" << str;
        EXPECT_EQ(std::string(), str);
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {//non-empty grid
        FloatGrid grid;
        grid.tree().setValue(Coord(-1,3,6), val);
        tools::Diagnose<FloatGrid> d(grid);
        tools::CheckNan<FloatGrid> c;
        std::string str = d.check(c);
        //std::cerr << "Non-Empty grid:\n" << str;
        EXPECT_EQ(std::string(), str);
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {//nan grid
        FloatGrid grid;
        grid.tree().setValue(Coord(-1,3,6), nan);
        tools::Diagnose<FloatGrid> d(grid);
        tools::CheckNan<FloatGrid> c;
        std::string str = d.check(c);
        //std::cerr << "NaN grid:\n" << str;
        EXPECT_TRUE(!str.empty());
        EXPECT_EQ(1, int(d.failureCount()));
    }

    {//nan and infinite grid
        FloatGrid grid;
        grid.tree().setValue(Coord(-1,3,6), nan);
        grid.tree().setValue(Coord(10,30,60), inf);
        tools::Diagnose<FloatGrid> d(grid);
        tools::CheckFinite<FloatGrid> c;
        std::string str = d.check(c);
        //std::cerr << "Not Finite grid:\n" << str;
        EXPECT_TRUE(!str.empty());
        EXPECT_EQ(2, int(d.failureCount()));
    }
    {//out-of-range grid
        FloatGrid grid(10.0f);
        grid.tree().setValue(Coord(-1,3,6), 1.0f);
        grid.tree().setValue(Coord(10,30,60), 1.5);
        grid.tree().fill(math::CoordBBox::createCube(math::Coord(0),8), 20.0f, true);
        tools::Diagnose<FloatGrid> d(grid);
        tools::CheckRange<FloatGrid> c(0.0f, 1.0f);
        std::string str = d.check(c);
        //std::cerr << "out-of-range grid:\n" << str;
        EXPECT_TRUE(!str.empty());
        EXPECT_EQ(3, int(d.failureCount()));
    }

    const float radius = 4.3f;
    const openvdb::Vec3f center(15.8f, 13.2f, 16.7f);
    const float voxelSize = 0.1f, width = 2.0f, gamma=voxelSize*width;

    FloatGrid::Ptr gridSphere =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);

    //gridSphere->print(std::cerr, 2);

    {// Check min/max of active values
        math::Extrema ex = tools::extrema(gridSphere->cbeginValueOn());
        //std::cerr << "Min = " << ex.min() << " max = " << ex.max() << std::endl;
        EXPECT_TRUE(ex.min() > -voxelSize*width);
        EXPECT_TRUE(ex.max() <  voxelSize*width);

    }
    {// Check min/max of all values
        math::Extrema ex = tools::extrema(gridSphere->cbeginValueAll());
        //std::cerr << "Min = " << ex.min() << " max = " << ex.max() << std::endl;
        EXPECT_TRUE(ex.min() >= -voxelSize*width);
        EXPECT_TRUE(ex.max() <=  voxelSize*width);

    }
    {// check range of all values in a sphere w/o mask
        tools::CheckRange<FloatGrid, true, true, FloatGrid::ValueAllCIter> c(-gamma, gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "Values out of range:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check range of on values in a sphere w/o mask
        tools::CheckRange<FloatGrid, true, true, FloatGrid::ValueOnCIter> c(-gamma, gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "Values out of range:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check range of off tiles in a sphere w/o mask
        tools::CheckRange<FloatGrid, true, true, FloatGrid::ValueOffCIter> c(-gamma, gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        {// check off tile iterator
            FloatGrid::ValueOffCIter i(gridSphere->tree());
            i.setMaxDepth(FloatGrid::ValueOffCIter::LEAF_DEPTH - 1);
            for (; i; ++i) EXPECT_TRUE( math::Abs(*i) <= gamma);
        }
        std::string str = d.check(c);
        //std::cerr << "Values out of range:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check range of sphere w/o mask
        tools::CheckRange<FloatGrid> c(0.0f, gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "Values out of range:\n" << str;
        EXPECT_TRUE(!str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_TRUE(d.failureCount() <  gridSphere->activeVoxelCount());
    }
    {// check range of sphere w mask
        tools::CheckRange<FloatGrid> c(0.0f, gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c, true);
        //std::cerr << "Values out of range:\n" << str;
        EXPECT_TRUE(!str.empty());
        EXPECT_EQ(d.valueCount(), d.valueCount());
        EXPECT_TRUE(d.failureCount() <  gridSphere->activeVoxelCount());
    }
    {// check min of sphere w/o mask
        tools::CheckMin<FloatGrid> c(-gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "Min values:\n" << str;
        EXPECT_EQ(std::string(), str);
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check max of sphere w/o mask
        tools::CheckMax<FloatGrid> c(gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "MAX values:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check norm of gradient of sphere w/o mask
        tools::CheckEikonal<FloatGrid> c(*gridSphere, 0.97f, 1.03f);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c, false, true, false, false);
        //std::cerr << "NormGrad:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check norm of gradient of sphere w/o mask
        tools::CheckNormGrad<FloatGrid> c(*gridSphere, 0.75f, 1.25f);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c, false, true, false, false);
        //std::cerr << "NormGrad:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
    {// check inactive values
        tools::CheckMagnitude<FloatGrid, FloatGrid::ValueOffCIter> c(gamma);
        tools::Diagnose<FloatGrid> d(*gridSphere);
        std::string str = d.check(c);
        //std::cerr << "Magnitude:\n" << str;
        EXPECT_TRUE(str.empty());
        EXPECT_EQ(0, int(d.valueCount()));
        EXPECT_EQ(0, int(d.failureCount()));
    }
}// testDiagnose

TEST_F(TestDiagnostics, testCheckLevelSet)
{
    using namespace openvdb;
    const float radius = 4.3f;
    const Vec3f center(15.8f, 13.2f, 16.7f);
    const float voxelSize = 0.1f, width = LEVEL_SET_HALF_WIDTH;

    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);

    //tools::CheckLevelSet<FloatGrid> c(*grid);
    //std::string str = c.check();
    std::string str = tools::checkLevelSet(*grid);
    EXPECT_TRUE(str.empty());
    //std::cerr << "\n" << str << std::endl;

    grid->tree().setValue(Coord(0,0,0), voxelSize*(width+0.5f));
    //str = c.check();
    str = tools::checkLevelSet(*grid);
    EXPECT_TRUE(!str.empty());
    //std::cerr << "\n" << str << std::endl;

    //str = c.check(6);
    str = tools::checkLevelSet(*grid, 6);
    EXPECT_TRUE(str.empty());

}// testCheckLevelSet

TEST_F(TestDiagnostics, testCheckFogVolume)
{
    using namespace openvdb;
    const float radius = 4.3f;
    const Vec3f center(15.8f, 13.2f, 16.7f);
    const float voxelSize = 0.1f, width = LEVEL_SET_HALF_WIDTH;

    FloatGrid::Ptr grid =
        tools::createLevelSetSphere<FloatGrid>(radius, center, voxelSize, width);
    tools::sdfToFogVolume(*grid);

    //tools::CheckFogVolume<FloatGrid> c(*grid);
    //std::string str = c.check();
    std::string str = tools::checkFogVolume(*grid);
    EXPECT_TRUE(str.empty());
    //std::cerr << "\n" << str << std::endl;

    grid->tree().setValue(Coord(0,0,0), 1.5f);
    //str = c.check();
    str = tools::checkFogVolume(*grid);
    EXPECT_TRUE(!str.empty());
    //std::cerr << "\n" << str << std::endl;

    str = tools::checkFogVolume(*grid, 5);
    //str = c.check(5);
    EXPECT_TRUE(str.empty());

}// testCheckFogVolume

TEST_F(TestDiagnostics, testUniqueInactiveValues)
{
    openvdb::FloatGrid grid;

    grid.tree().setValueOff(openvdb::Coord(0,0,0), -1);
    grid.tree().setValueOff(openvdb::Coord(0,0,1), -2);
    grid.tree().setValueOff(openvdb::Coord(0,1,0), -3);
    grid.tree().setValue(openvdb::Coord(1,0,0),  1);

    std::vector<float> values;

    EXPECT_TRUE(openvdb::tools::uniqueInactiveValues(grid, values, 4));

    EXPECT_EQ(4, int(values.size()));

    EXPECT_TRUE(openvdb::math::isApproxEqual(values[0], -3.0f));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[1], -2.0f));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[2], -1.0f));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[3], 0.0f));


    // test with level set sphere
    const float radius = 4.3f;
    const openvdb::Vec3f center(15.8f, 13.2f, 16.7f);
    const float voxelSize = 0.5f, width = 2.0f;

    openvdb::FloatGrid::Ptr gridSphere =
        openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center, voxelSize, width);

    EXPECT_TRUE(openvdb::tools::uniqueInactiveValues(*gridSphere.get(), values, 2));

    EXPECT_EQ(2, int(values.size()));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[0], -voxelSize * width));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[1],  voxelSize * width));

    // test with fog volume
    openvdb::tools::sdfToFogVolume(*gridSphere);

    EXPECT_TRUE(openvdb::tools::uniqueInactiveValues(*gridSphere.get(), values, 1));

    EXPECT_EQ(1, int(values.size()));
    EXPECT_TRUE(openvdb::math::isApproxEqual(values[0], 0.0f));
}
