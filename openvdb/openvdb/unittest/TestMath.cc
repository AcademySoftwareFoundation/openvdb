// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <gtest/gtest.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>
#include <openvdb/Types.h>
#include <type_traits>
#include <vector>


class TestMath: public ::testing::Test
{
};


// This suite of tests obviously needs to be expanded!
TEST_F(TestMath, testAll)
{
    using namespace openvdb;

    {// Sign
        EXPECT_EQ(math::Sign( 3   ), 1);
        EXPECT_EQ(math::Sign(-1.0 ),-1);
        EXPECT_EQ(math::Sign( 0.0f), 0);
    }
    {// SignChange
        EXPECT_TRUE( math::SignChange( -1, 1));
        EXPECT_TRUE(!math::SignChange( 0.0f, 0.5f));
        EXPECT_TRUE( math::SignChange( 0.0f,-0.5f));
        EXPECT_TRUE( math::SignChange(-0.1, 0.0001));

    }
    {// isApproxZero
        EXPECT_TRUE( math::isApproxZero( 0.0f));
        EXPECT_TRUE(!math::isApproxZero( 9.0e-6f));
        EXPECT_TRUE(!math::isApproxZero(-9.0e-6f));
        EXPECT_TRUE( math::isApproxZero( 9.0e-9f));
        EXPECT_TRUE( math::isApproxZero(-9.0e-9f));
        EXPECT_TRUE( math::isApproxZero( 0.01, 0.1));
    }
    {// Cbrt
        const double a = math::Cbrt(3.0);
        EXPECT_TRUE(math::isApproxEqual(a*a*a, 3.0, 1e-6));
    }
    {// isNegative
        EXPECT_TRUE(!std::is_signed<unsigned int>::value);
        EXPECT_TRUE(std::is_signed<int>::value);
        EXPECT_TRUE(!std::is_signed<bool>::value);
        //EXPECT_TRUE(std::is_signed<double>::value);//fails!
        //EXPECT_TRUE(std::is_signed<float>::value);//fails!

        EXPECT_TRUE( math::isNegative(-1.0f));
        EXPECT_TRUE(!math::isNegative( 1.0f));
        EXPECT_TRUE( math::isNegative(-1.0));
        EXPECT_TRUE(!math::isNegative( 1.0));
        EXPECT_TRUE(!math::isNegative(true));
        EXPECT_TRUE(!math::isNegative(false));
        EXPECT_TRUE(!math::isNegative(1u));
        EXPECT_TRUE( math::isNegative(-1));
        EXPECT_TRUE(!math::isNegative( 1));
    }
    {// zeroVal
        EXPECT_EQ(zeroVal<bool>(), false);
        EXPECT_EQ(zeroVal<int>(), int(0));
        EXPECT_EQ(zeroVal<float>(), 0.0f);
        EXPECT_EQ(zeroVal<double>(), 0.0);
        EXPECT_EQ(zeroVal<Vec3i>(), Vec3i(0,0,0));
        EXPECT_EQ(zeroVal<Vec3s>(), Vec3s(0,0,0));
        EXPECT_EQ(zeroVal<Vec3d>(), Vec3d(0,0,0));
        EXPECT_EQ(zeroVal<Quats>(), Quats::zero());
        EXPECT_EQ(zeroVal<Quatd>(), Quatd::zero());
        EXPECT_EQ(zeroVal<Mat3s>(), Mat3s::zero());
        EXPECT_EQ(zeroVal<Mat3d>(), Mat3d::zero());
        EXPECT_EQ(zeroVal<Mat4s>(), Mat4s::zero());
        EXPECT_EQ(zeroVal<Mat4d>(), Mat4d::zero());
    }
}


TEST_F(TestMath, testRandomInt)
{
    using openvdb::math::RandomInt;

    int imin = -3, imax = 11;
    RandomInt rnd(/*seed=*/42, imin, imax);

    // Generate a sequence of random integers and verify that they all fall
    // in the interval [imin, imax].
    std::vector<int> seq(100);
    for (int i = 0; i < 100; ++i) {
        seq[i] = rnd();
        EXPECT_TRUE(seq[i] >= imin);
        EXPECT_TRUE(seq[i] <= imax);
    }

    // Verify that generators with the same seed produce the same sequence.
    rnd = RandomInt(42, imin, imax);
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        EXPECT_EQ(seq[i], r);
    }

    // Verify that generators with different seeds produce different sequences.
    rnd = RandomInt(101, imin, imax);
    std::vector<int> newSeq(100);
    for (int i = 0; i < 100; ++i) newSeq[i] = rnd();
    EXPECT_TRUE(newSeq != seq);

    // Temporarily change the range.
    imin = -5; imax = 6;
    for (int i = 0; i < 100; ++i) {
        int r = rnd(imin, imax);
        EXPECT_TRUE(r >= imin);
        EXPECT_TRUE(r <= imax);
    }
    // Verify that the range change was temporary.
    imin = -3; imax = 11;
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        EXPECT_TRUE(r >= imin);
        EXPECT_TRUE(r <= imax);
    }

    // Permanently change the range.
    imin = -5; imax = 6;
    rnd.setRange(imin, imax);
    for (int i = 0; i < 100; ++i) {
        int r = rnd();
        EXPECT_TRUE(r >= imin);
        EXPECT_TRUE(r <= imax);
    }

    // Verify that it is OK to specify imin > imax (they are automatically swapped).
    imin = 5; imax = -6;
    rnd.setRange(imin, imax);

    rnd = RandomInt(42, imin, imax);
}


TEST_F(TestMath, testRandom01)
{
    using openvdb::math::Random01;
    using openvdb::math::isApproxEqual;

    Random01 rnd(/*seed=*/42);

    // Generate a sequence of random numbers and verify that they all fall
    // in the interval [0, 1).
    std::vector<Random01::ValueType> seq(100);
    for (int i = 0; i < 100; ++i) {
        seq[i] = rnd();
        EXPECT_TRUE(seq[i] >= 0.0);
        EXPECT_TRUE(seq[i] < 1.0);
    }

    // Verify that generators with the same seed produce the same sequence.
    rnd = Random01(42);
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(seq[i], rnd(), /*tolerance=*/1.0e-6);
    }

    // Verify that generators with different seeds produce different sequences.
    rnd = Random01(101);
    bool allEqual = true;
    for (int i = 0; allEqual && i < 100; ++i) {
        if (!isApproxEqual(rnd(), seq[i])) allEqual = false;
    }
    EXPECT_TRUE(!allEqual);
}

TEST_F(TestMath, testMinMaxIndex)
{
    const openvdb::Vec3R a(-1, 2, 0);
    EXPECT_EQ(size_t(0), openvdb::math::MinIndex(a));
    EXPECT_EQ(size_t(1), openvdb::math::MaxIndex(a));
    const openvdb::Vec3R b(-1, -2, 0);
    EXPECT_EQ(size_t(1), openvdb::math::MinIndex(b));
    EXPECT_EQ(size_t(2), openvdb::math::MaxIndex(b));
    const openvdb::Vec3R c(5, 2, 1);
    EXPECT_EQ(size_t(2), openvdb::math::MinIndex(c));
    EXPECT_EQ(size_t(0), openvdb::math::MaxIndex(c));
    const openvdb::Vec3R d(0, 0, 1);
    EXPECT_EQ(size_t(1), openvdb::math::MinIndex(d));
    EXPECT_EQ(size_t(2), openvdb::math::MaxIndex(d));
    const openvdb::Vec3R e(1, 0, 0);
    EXPECT_EQ(size_t(2), openvdb::math::MinIndex(e));
    EXPECT_EQ(size_t(0), openvdb::math::MaxIndex(e));
    const openvdb::Vec3R f(0, 1, 0);
    EXPECT_EQ(size_t(2), openvdb::math::MinIndex(f));
    EXPECT_EQ(size_t(1), openvdb::math::MaxIndex(f));
    const openvdb::Vec3R g(1, 1, 0);
    EXPECT_EQ(size_t(2), openvdb::math::MinIndex(g));
    EXPECT_EQ(size_t(1), openvdb::math::MaxIndex(g));
    const openvdb::Vec3R h(1, 0, 1);
    EXPECT_EQ(size_t(1), openvdb::math::MinIndex(h));
    EXPECT_EQ(size_t(2), openvdb::math::MaxIndex(h));
    const openvdb::Vec3R i(0, 1, 1);
    EXPECT_EQ(size_t(0), openvdb::math::MinIndex(i));
    EXPECT_EQ(size_t(2), openvdb::math::MaxIndex(i));
    const openvdb::Vec3R j(1, 1, 1);
    EXPECT_EQ(size_t(2), openvdb::math::MinIndex(j));
    EXPECT_EQ(size_t(2), openvdb::math::MaxIndex(j));
}
