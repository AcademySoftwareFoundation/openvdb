// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/openvdb.h>
#include <openvdb/math/Operators.h> // for ISGradient
#include <openvdb/math/Stats.h>
#include <openvdb/tools/Statistics.h>
#include <gtest/gtest.h>

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);


class TestStats: public ::testing::Test
{
};


TEST_F(TestStats, testMinMax)
{
    {// test Coord which uses lexicographic less than
        openvdb::math::MinMax<openvdb::Coord> s(openvdb::Coord::max(), openvdb::Coord::min());
        //openvdb::math::MinMax<openvdb::Coord> s;// will not compile since Coord is not a POD type
        EXPECT_EQ(openvdb::Coord::max(), s.min());
        EXPECT_EQ(openvdb::Coord::min(), s.max());
        s.add( openvdb::Coord(1,2,3) );
        EXPECT_EQ(openvdb::Coord(1,2,3), s.min());
        EXPECT_EQ(openvdb::Coord(1,2,3), s.max());
        s.add( openvdb::Coord(0,2,3) );
        EXPECT_EQ(openvdb::Coord(0,2,3), s.min());
        EXPECT_EQ(openvdb::Coord(1,2,3), s.max());
        s.add( openvdb::Coord(1,2,4) );
        EXPECT_EQ(openvdb::Coord(0,2,3), s.min());
        EXPECT_EQ(openvdb::Coord(1,2,4), s.max());
    }
    {// test double
        openvdb::math::MinMax<double> s;
        EXPECT_EQ( std::numeric_limits<double>::max(), s.min());
        EXPECT_EQ(-std::numeric_limits<double>::max(), s.max());
        s.add( 1.0 );
        EXPECT_EQ(1.0, s.min());
        EXPECT_EQ(1.0, s.max());
        s.add( 2.5 );
        EXPECT_EQ(1.0, s.min());
        EXPECT_EQ(2.5, s.max());
        s.add( -0.5 );
        EXPECT_EQ(-0.5, s.min());
        EXPECT_EQ( 2.5, s.max());
    }
    {// test int
        openvdb::math::MinMax<int> s;
        EXPECT_EQ(std::numeric_limits<int>::max(), s.min());
        EXPECT_EQ(std::numeric_limits<int>::min(), s.max());
        s.add( 1 );
        EXPECT_EQ(1, s.min());
        EXPECT_EQ(1, s.max());
        s.add( 2 );
        EXPECT_EQ(1, s.min());
        EXPECT_EQ(2, s.max());
        s.add( -5 );
        EXPECT_EQ(-5, s.min());
        EXPECT_EQ( 2, s.max());
    }
    {// test unsigned
        openvdb::math::MinMax<uint32_t> s;
        EXPECT_EQ(std::numeric_limits<uint32_t>::max(), s.min());
        EXPECT_EQ(uint32_t(0), s.max());
        s.add( 1 );
        EXPECT_EQ(uint32_t(1), s.min());
        EXPECT_EQ(uint32_t(1), s.max());
        s.add( 2 );
        EXPECT_EQ(uint32_t(1), s.min());
        EXPECT_EQ(uint32_t(2), s.max());
        s.add( 0 );
        EXPECT_EQ( uint32_t(0), s.min());
        EXPECT_EQ( uint32_t(2), s.max());
    }
}


TEST_F(TestStats, testExtrema)
{
    {// trivial test
        openvdb::math::Extrema s;
        s.add(0);
        s.add(1);
        EXPECT_EQ(2, int(s.size()));
        EXPECT_NEAR(0.0, s.min(), 0.000001);
        EXPECT_NEAR(1.0, s.max(), 0.000001);
        EXPECT_NEAR(1.0, s.range(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test
        openvdb::math::Extrema s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        EXPECT_EQ(5, int(s.size()));
        EXPECT_NEAR(data[2], s.min(), 0.000001);
        EXPECT_NEAR(data[0], s.max(), 0.000001);
        EXPECT_NEAR(data[0]-data[2], s.range(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test of Extrema::add(Extrema)
        openvdb::math::Extrema s, t;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<3; ++i) s.add(data[i]);
        for (int i=3; i<5; ++i) t.add(data[i]);
        s.add(t);
        EXPECT_EQ(5, int(s.size()));
        EXPECT_NEAR(data[2], s.min(), 0.000001);
        EXPECT_NEAR(data[0], s.max(), 0.000001);
        EXPECT_NEAR(data[0]-data[2], s.range(), 0.000001);
        //s.print("test");
    }
    {// Trivial test of Extrema::add(value, n)
        openvdb::math::Extrema s;
        const double val = 3.45;
        const uint64_t n = 57;
        s.add(val, 57);
        EXPECT_EQ(n, s.size());
        EXPECT_NEAR(val, s.min(), 0.000001);
        EXPECT_NEAR(val, s.max(), 0.000001);
        EXPECT_NEAR(0.0, s.range(), 0.000001);
    }
    {// Test 1 of Extrema::add(value), Extrema::add(value, n) and Extrema::add(Extrema)
        openvdb::math::Extrema s, t;
        const double val1 = 1.0, val2 = 3.0;
        const uint64_t n1 = 1, n2 =1;
        s.add(val1,  n1);
        EXPECT_EQ(uint64_t(n1), s.size());
        EXPECT_NEAR(val1, s.min(),      0.000001);
        EXPECT_NEAR(val1, s.max(),      0.000001);
        for (uint64_t i=0; i<n2; ++i) t.add(val2);
        s.add(t);
        EXPECT_EQ(uint64_t(n2), t.size());
        EXPECT_NEAR(val2, t.min(),      0.000001);
        EXPECT_NEAR(val2, t.max(),      0.000001);

        EXPECT_EQ(uint64_t(n1+n2), s.size());
        EXPECT_NEAR(val1,    s.min(),  0.000001);
        EXPECT_NEAR(val2,    s.max(),  0.000001);
    }
    {// Non-trivial test of Extrema::add(value, n)
        openvdb::math::Extrema s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Extrema t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        EXPECT_EQ(s.size(), t.size());
        EXPECT_NEAR(s.min(), t.min(),  0.000001);
        EXPECT_NEAR(s.max(), t.max(),  0.000001);
    }
}

TEST_F(TestStats, testStats)
{
    {// trivial test
        openvdb::math::Stats s;
        s.add(0);
        s.add(1);
        EXPECT_EQ(2, int(s.size()));
        EXPECT_NEAR(0.0, s.min(), 0.000001);
        EXPECT_NEAR(1.0, s.max(), 0.000001);
        EXPECT_NEAR(0.5, s.mean(), 0.000001);
        EXPECT_NEAR(0.25, s.variance(), 0.000001);
        EXPECT_NEAR(0.5, s.stdDev(), 0.000001);
        //s.print("test");
    }
    {// non-trivial test
        openvdb::math::Stats s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        double sum = 0.0;
        for (int i=0; i<5; ++i) sum += data[i];
        const double mean = sum/5.0;
        sum = 0.0;
        for (int i=0; i<5; ++i) sum += (data[i]-mean)*(data[i]-mean);
        const double var = sum/5.0;
        EXPECT_EQ(5, int(s.size()));
        EXPECT_NEAR(data[2], s.min(), 0.000001);
        EXPECT_NEAR(data[0], s.max(), 0.000001);
        EXPECT_NEAR(mean, s.mean(), 0.000001);
        EXPECT_NEAR(var, s.variance(), 0.000001);
        EXPECT_NEAR(sqrt(var), s.stdDev(),  0.000001);
        //s.print("test");
    }
    {// non-trivial test of Stats::add(Stats)
        openvdb::math::Stats s, t;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<3; ++i) s.add(data[i]);
        for (int i=3; i<5; ++i) t.add(data[i]);
        s.add(t);
        double sum = 0.0;
        for (int i=0; i<5; ++i) sum += data[i];
        const double mean = sum/5.0;
        sum = 0.0;
        for (int i=0; i<5; ++i) sum += (data[i]-mean)*(data[i]-mean);
        const double var = sum/5.0;
        EXPECT_EQ(5, int(s.size()));
        EXPECT_NEAR(data[2], s.min(), 0.000001);
        EXPECT_NEAR(data[0], s.max(), 0.000001);
        EXPECT_NEAR(mean, s.mean(), 0.000001);
        EXPECT_NEAR(var, s.variance(), 0.000001);
        EXPECT_NEAR(sqrt(var), s.stdDev(),  0.000001);
        //s.print("test");
    }
    {// Trivial test of Stats::add(value, n)
        openvdb::math::Stats s;
        const double val = 3.45;
        const uint64_t n = 57;
        s.add(val, 57);
        EXPECT_EQ(n, s.size());
        EXPECT_NEAR(val, s.min(), 0.000001);
        EXPECT_NEAR(val, s.max(), 0.000001);
        EXPECT_NEAR(val, s.mean(), 0.000001);
        EXPECT_NEAR(0.0, s.variance(), 0.000001);
        EXPECT_NEAR(0.0, s.stdDev(),  0.000001);
    }
    {// Test 1 of Stats::add(value), Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s, t;
        const double val1 = 1.0, val2 = 3.0, sum = val1 + val2;
        const uint64_t n1 = 1, n2 =1;
        s.add(val1,  n1);
        EXPECT_EQ(uint64_t(n1), s.size());
        EXPECT_NEAR(val1, s.min(),      0.000001);
        EXPECT_NEAR(val1, s.max(),      0.000001);
        EXPECT_NEAR(val1, s.mean(),     0.000001);
        EXPECT_NEAR(0.0,  s.variance(), 0.000001);
        EXPECT_NEAR(0.0,  s.stdDev(),   0.000001);
        for (uint64_t i=0; i<n2; ++i) t.add(val2);
        s.add(t);
        EXPECT_EQ(uint64_t(n2), t.size());
        EXPECT_NEAR(val2, t.min(),      0.000001);
        EXPECT_NEAR(val2, t.max(),      0.000001);
        EXPECT_NEAR(val2, t.mean(),     0.000001);
        EXPECT_NEAR(0.0,  t.variance(), 0.000001);
        EXPECT_NEAR(0.0,  t.stdDev(),   0.000001);
        EXPECT_EQ(uint64_t(n1+n2), s.size());
        EXPECT_NEAR(val1,    s.min(),  0.000001);
        EXPECT_NEAR(val2,    s.max(),  0.000001);
        const double mean = sum/double(n1+n2);
        EXPECT_NEAR(mean,    s.mean(), 0.000001);
        double var = 0.0;
        for (uint64_t i=0; i<n1; ++i) var += openvdb::math::Pow2(val1-mean);
        for (uint64_t i=0; i<n2; ++i) var += openvdb::math::Pow2(val2-mean);
        var /= double(n1+n2);
        EXPECT_NEAR(var, s.variance(), 0.000001);
    }
    {// Test 2 of Stats::add(value), Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s, t;
        const double val1 = 1.0, val2 = 3.0, sum = val1 + val2;
        const uint64_t n1 = 1, n2 =1;
        for (uint64_t i=0; i<n1; ++i) s.add(val1);
        EXPECT_EQ(uint64_t(n1), s.size());
        EXPECT_NEAR(val1, s.min(),      0.000001);
        EXPECT_NEAR(val1, s.max(),      0.000001);
        EXPECT_NEAR(val1, s.mean(),     0.000001);
        EXPECT_NEAR(0.0,  s.variance(), 0.000001);
        EXPECT_NEAR(0.0,  s.stdDev(),   0.000001);
        t.add(val2,  n2);
        EXPECT_EQ(uint64_t(n2), t.size());
        EXPECT_NEAR(val2, t.min(),      0.000001);
        EXPECT_NEAR(val2, t.max(),      0.000001);
        EXPECT_NEAR(val2, t.mean(),     0.000001);
        EXPECT_NEAR(0.0,  t.variance(), 0.000001);
        EXPECT_NEAR(0.0,  t.stdDev(),   0.000001);
        s.add(t);
        EXPECT_EQ(uint64_t(n1+n2), s.size());
        EXPECT_NEAR(val1,    s.min(),  0.000001);
        EXPECT_NEAR(val2,    s.max(),  0.000001);
        const double mean = sum/double(n1+n2);
        EXPECT_NEAR(mean,    s.mean(), 0.000001);
        double var = 0.0;
        for (uint64_t i=0; i<n1; ++i) var += openvdb::math::Pow2(val1-mean);
        for (uint64_t i=0; i<n2; ++i) var += openvdb::math::Pow2(val2-mean);
        var /= double(n1+n2);
        EXPECT_NEAR(var, s.variance(), 0.000001);
    }
    {// Non-trivial test of Stats::add(value, n) and Stats::add(Stats)
        openvdb::math::Stats s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Stats t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        EXPECT_EQ(s.size(), t.size());
        EXPECT_NEAR(s.min(), t.min(),  0.000001);
        EXPECT_NEAR(s.max(), t.max(),  0.000001);
        EXPECT_NEAR(s.mean(),t.mean(), 0.000001);
        EXPECT_NEAR(s.variance(), t.variance(), 0.000001);
    }
    {// Non-trivial test of Stats::add(value, n)
        openvdb::math::Stats s;
        s.add(3.45,  6);
        s.add(1.39,  2);
        s.add(2.56, 13);
        s.add(0.03);
        openvdb::math::Stats t;
        for (int i=0; i< 6; ++i) t.add(3.45);
        for (int i=0; i< 2; ++i) t.add(1.39);
        for (int i=0; i<13; ++i) t.add(2.56);
        t.add(0.03);
        EXPECT_EQ(s.size(), t.size());
        EXPECT_NEAR(s.min(), t.min(),  0.000001);
        EXPECT_NEAR(s.max(), t.max(),  0.000001);
        EXPECT_NEAR(s.mean(),t.mean(), 0.000001);
        EXPECT_NEAR(s.variance(), t.variance(), 0.000001);
    }

    //std::cerr << "\nCompleted TestStats::testStats!\n" << std::endl;
}

TEST_F(TestStats, testHistogram)
{
     {// Histogram test
        openvdb::math::Stats s;
        const int data[5]={600, 470, 170, 430, 300};
        for (int i=0; i<5; ++i) s.add(data[i]);
        openvdb::math::Histogram h(s, 10);
        for (int i=0; i<5; ++i) EXPECT_TRUE(h.add(data[i]));
        int bin[10]={0};
        for (int i=0; i<5; ++i) {
            for (int j=0; j<10; ++j) if (data[i] >= h.min(j) && data[i] < h.max(j)) bin[j]++;
        }
        for (int i=0; i<5; ++i)  EXPECT_EQ(bin[i],int(h.count(i)));
        //h.print("test");
    }
    {//Test print of Histogram
        openvdb::math::Stats s;
        const int N=500000;
        for (int i=0; i<N; ++i) s.add(N/2-i);
        //s.print("print-test");
        openvdb::math::Histogram h(s, 25);
        for (int i=0; i<N; ++i) EXPECT_TRUE(h.add(N/2-i));
        //h.print("print-test");
    }
}

namespace {

struct GradOp
{
    typedef openvdb::FloatGrid GridT;

    GridT::ConstAccessor acc;

    GradOp(const GridT& grid): acc(grid.getConstAccessor()) {}

    template <typename StatsT>
    void operator()(const GridT::ValueOnCIter& it, StatsT& stats) const
    {
        typedef openvdb::math::ISGradient<openvdb::math::FD_1ST> GradT;
        if (it.isVoxelValue()) {
            stats.add(GradT::result(acc, it.getCoord()).length());
        } else {
            openvdb::CoordBBox bbox = it.getBoundingBox();
            openvdb::Coord xyz;
            int &x = xyz[0], &y = xyz[1], &z = xyz[2];
            for (x = bbox.min()[0]; x <= bbox.max()[0]; ++x) {
                for (y = bbox.min()[1]; y <= bbox.max()[1]; ++y) {
                    for (z = bbox.min()[2]; z <= bbox.max()[2]; ++z) {
                        stats.add(GradT::result(acc, xyz).length());
                    }
                }
            }
        }
    }
};

} // unnamed namespace

TEST_F(TestStats, testGridExtrema)
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            // Compute active value statistics for a grid with a single active voxel.
            grid.tree().setValue(Coord(0), /*value=*/42.0);
            math::Extrema ex = tools::extrema(grid.cbeginValueOn());

            EXPECT_NEAR(42.0, ex.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(42.0, ex.max(),  /*tolerance=*/0.0);

            // Compute inactive value statistics for a grid with only background voxels.
            grid.tree().setValueOff(Coord(0), background);
            ex = tools::extrema(grid.cbeginValueOff());

            EXPECT_NEAR(background, ex.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(background, ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for a grid with two active voxel populations
        // of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        EXPECT_EQ(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), threaded);

            EXPECT_NEAR(double(-3.0), ex.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(1.0),  ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for just the positive values.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            struct Local {
                static void addIfPositive(const FloatGrid::ValueOnCIter& it, math::Extrema& ex)
                {
                    const float f = *it;
                    if (f > 0.0) {
                        if (it.isVoxelValue()) ex.add(f);
                        else ex.add(f, it.getVoxelCount());
                    }
                }
            };
            math::Extrema ex =
                tools::extrema(grid.cbeginValueOn(), &Local::addIfPositive, threaded);

            EXPECT_NEAR(double(1.0), ex.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(1.0), ex.max(),  /*tolerance=*/0.0);
        }

        // Compute active value statistics for the first-order gradient.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            // First, using a custom ValueOp...
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), GradOp(grid), threaded);
            EXPECT_NEAR(double(0.0), ex.min(), /*tolerance=*/0.0);
            EXPECT_NEAR(
                double(9.0 + 9.0 + 9.0), ex.max() * ex.max(), /*tol=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)

            // ...then using tools::opStatistics().
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > MathOp;
            ex = tools::opExtrema(grid.cbeginValueOn(), MathOp(), threaded);
            EXPECT_NEAR(double(0.0), ex.min(), /*tolerance=*/0.0);
            EXPECT_NEAR(
                double(9.0 + 9.0 + 9.0), ex.max() * ex.max(), /*tolerance=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute active vector magnitude statistics for a vector-valued grid
        // with two active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        EXPECT_EQ(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Extrema ex = tools::extrema(grid.cbeginValueOn(), threaded);

            EXPECT_NEAR(double(3.0), ex.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(5.0), ex.max(),  /*tolerance=*/0.0);
        }
    }
}

TEST_F(TestStats, testGridStats)
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            // Compute active value statistics for a grid with a single active voxel.
            grid.tree().setValue(Coord(0), /*value=*/42.0);
            math::Stats stats = tools::statistics(grid.cbeginValueOn());

            EXPECT_NEAR(42.0, stats.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(42.0, stats.max(),  /*tolerance=*/0.0);
            EXPECT_NEAR(42.0, stats.mean(), /*tolerance=*/1.0e-8);
            EXPECT_NEAR(0.0,  stats.variance(), /*tolerance=*/1.0e-8);

            // Compute inactive value statistics for a grid with only background voxels.
            grid.tree().setValueOff(Coord(0), background);
            stats = tools::statistics(grid.cbeginValueOff());

            EXPECT_NEAR(background, stats.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(background, stats.max(),  /*tolerance=*/0.0);
            EXPECT_NEAR(background, stats.mean(), /*tolerance=*/1.0e-8);
            EXPECT_NEAR(0.0,        stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for a grid with two active voxel populations
        // of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        EXPECT_EQ(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), threaded);

            EXPECT_NEAR(double(-3.0), stats.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(1.0),  stats.max(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(-1.0), stats.mean(), /*tolerance=*/1.0e-8);
            EXPECT_NEAR(double(4.0),  stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for just the positive values.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            struct Local {
                static void addIfPositive(const FloatGrid::ValueOnCIter& it, math::Stats& stats)
                {
                    const float f = *it;
                    if (f > 0.0) {
                        if (it.isVoxelValue()) stats.add(f);
                        else stats.add(f, it.getVoxelCount());
                    }
                }
            };
            math::Stats stats =
                tools::statistics(grid.cbeginValueOn(), &Local::addIfPositive, threaded);

            EXPECT_NEAR(double(1.0), stats.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(1.0), stats.max(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(1.0), stats.mean(), /*tolerance=*/1.0e-8);
            EXPECT_NEAR(double(0.0), stats.variance(), /*tolerance=*/1.0e-8);
        }

        // Compute active value statistics for the first-order gradient.
        for (int threaded = 0; threaded <= 1; ++threaded) {
            // First, using a custom ValueOp...
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), GradOp(grid), threaded);
            EXPECT_NEAR(double(0.0), stats.min(), /*tolerance=*/0.0);
            EXPECT_NEAR(
                double(9.0 + 9.0 + 9.0), stats.max() * stats.max(), /*tol=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)

            // ...then using tools::opStatistics().
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > MathOp;
            stats = tools::opStatistics(grid.cbeginValueOn(), MathOp(), threaded);
            EXPECT_NEAR(double(0.0), stats.min(), /*tolerance=*/0.0);
            EXPECT_NEAR(
                double(9.0 + 9.0 + 9.0), stats.max() * stats.max(), /*tolerance=*/1.0e-3);
                // max gradient is (dx, dy, dz) = (-3 - 0, -3 - 0, -3 - 0)
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute active vector magnitude statistics for a vector-valued grid
        // with two active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        EXPECT_EQ(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Stats stats = tools::statistics(grid.cbeginValueOn(), threaded);

            EXPECT_NEAR(double(3.0), stats.min(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(5.0), stats.max(),  /*tolerance=*/0.0);
            EXPECT_NEAR(double(4.0), stats.mean(), /*tolerance=*/1.0e-8);
            EXPECT_NEAR(double(1.0),  stats.variance(), /*tolerance=*/1.0e-8);
        }
    }
}


namespace {

template<typename OpT, typename GridT>
inline void
doTestGridOperatorStats(const GridT& grid, const OpT& op)
{
    openvdb::math::Stats serialStats =
        openvdb::tools::opStatistics(grid.cbeginValueOn(), op, /*threaded=*/false);

    openvdb::math::Stats parallelStats =
        openvdb::tools::opStatistics(grid.cbeginValueOn(), op, /*threaded=*/true);

    // Verify that the results from threaded and serial runs are equivalent.
    EXPECT_EQ(serialStats.size(), parallelStats.size());
    ASSERT_DOUBLES_EXACTLY_EQUAL(serialStats.min(), parallelStats.min());
    ASSERT_DOUBLES_EXACTLY_EQUAL(serialStats.max(), parallelStats.max());
    EXPECT_NEAR(serialStats.mean(), parallelStats.mean(), /*tolerance=*/1.0e-6);
    EXPECT_NEAR(serialStats.variance(), parallelStats.variance(), 1.0e-6);
}

}

TEST_F(TestStats, testGridOperatorStats)
{
    using namespace openvdb;

    typedef math::UniformScaleMap MapType;
    MapType map;

    const int DIM = 109;
    {
        // Test operations on a scalar grid.
        const float background = 0.0;
        FloatGrid grid(background);
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/-3.0);

        {   // Magnitude of gradient computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::OpMagnitude<math::Gradient<MapType, math::FD_1ST>, MapType>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of index-space gradient computed via first-order differencing
            typedef math::ISOpMagnitude<math::ISGradient<math::FD_1ST> > OpT;
            doTestGridOperatorStats(grid, OpT());
        }
        {   // Laplacian of index-space gradient computed via second-order central differencing
            typedef math::ISLaplacian<math::CD_SECOND> OpT;
            doTestGridOperatorStats(grid, OpT());
        }
    }
    {
        // Test operations on a vector grid.
        const Vec3s background(0.0);
        Vec3SGrid grid(background);
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        {   // Divergence computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::Divergence<MapType, math::FD_1ST>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of curl computed via first-order differencing
            typedef math::MapAdapter<MapType,
                math::OpMagnitude<math::Curl<MapType, math::FD_1ST>, MapType>, double> OpT;
            doTestGridOperatorStats(grid, OpT(map));
        }
        {   // Magnitude of index-space curl computed via first-order differencing
            typedef math::ISOpMagnitude<math::ISCurl<math::FD_1ST> > OpT;
            doTestGridOperatorStats(grid, OpT());
        }
    }
}


TEST_F(TestStats, testGridHistogram)
{
    using namespace openvdb;

    const int DIM = 109;
    {
        const float background = 0.0;
        FloatGrid grid(background);
        {
            const double value = 42.0;

            // Compute a histogram of the active values of a grid with a single active voxel.
            grid.tree().setValue(Coord(0), value);
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/100.0);

            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                uint64_t expected = ((hist.min(i) <= value && value <= hist.max(i)) ? 1 : 0);
                EXPECT_EQ(expected, hist.count(i));
            }
        }

        // Compute a histogram of the active values of a grid with two
        // active voxel populations of the same size but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM), /*value=*/1.0);
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), /*value=*/3.0);

        EXPECT_EQ(uint64_t(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/10.0, /*numBins=*/9, threaded);

            EXPECT_EQ(Index64(2 * DIM * DIM * DIM), hist.size());
            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                if (i == 0 || i == 2) {
                    EXPECT_EQ(uint64_t(DIM * DIM * DIM), hist.count(i));
                } else {
                    EXPECT_EQ(uint64_t(0), hist.count(i));
                }
            }
        }
    }
    {
        const Vec3s background(0.0);
        Vec3SGrid grid(background);

        // Compute a histogram of vector magnitudes of the active values of a
        // vector-valued grid with two active voxel populations of the same size
        // but two different values.
        grid.fill(CoordBBox::createCube(Coord(0), DIM),    Vec3s(3.0, 0.0, 4.0)); // length = 5
        grid.fill(CoordBBox::createCube(Coord(-300), DIM), Vec3s(1.0, 2.0, 2.0)); // length = 3

        EXPECT_EQ(Index64(2 * DIM * DIM * DIM), grid.activeVoxelCount());

        for (int threaded = 0; threaded <= 1; ++threaded) {
            math::Histogram hist = tools::histogram(grid.cbeginValueOn(),
                /*min=*/0.0, /*max=*/10.0, /*numBins=*/9, threaded);

            EXPECT_EQ(Index64(2 * DIM * DIM * DIM), hist.size());
            for (int i = 0, N = int(hist.numBins()); i < N; ++i) {
                if (i == 2 || i == 4) {
                    EXPECT_EQ(uint64_t(DIM * DIM * DIM), hist.count(i));
                } else {
                    EXPECT_EQ(uint64_t(0), hist.count(i));
                }
            }
        }
    }
}
