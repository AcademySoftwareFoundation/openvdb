// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <openvdb/Exceptions.h>
#include <openvdb/math/Math.h>
#include <openvdb/Types.h>
#include <type_traits>
#include <vector>


class TestMath: public ::testing::Test
{
};

// Testing the operators within the member functions of Vec3
template<typename ValueT>
void testMemberOperatorsImpl()
{
    using namespace openvdb;
    using Vec3T = math::Vec3<ValueT>;

    {
        Vec3T vecA(ValueT(3.14), ValueT(2.18), ValueT(-299792458.f));

        // Alternative indexed the elements
        EXPECT_EQ(vecA(0), ValueT(3.14));
        EXPECT_EQ(vecA(1), ValueT(2.18));
        EXPECT_EQ(vecA(2), ValueT(-299792458.f));

        // Assignment operator
        Vec3T vecB = vecA;
        EXPECT_EQ(vecB(0), vecA(0));
        EXPECT_EQ(vecB(1), vecA(1));
        EXPECT_EQ(vecB(2), vecA(2));

        // Negation operator
        Vec3T vecC = -vecA;
        EXPECT_EQ(vecC(0), -vecA(0));
        EXPECT_EQ(vecC(1), -vecA(1));
        EXPECT_EQ(vecC(2), -vecA(2));

        // Multiply each element of the vector by a scalar
        Vec3T vecD = vecA;
        const ValueT gr = ValueT(1.6180339887); // golden ratio
        vecD *= gr;
        EXPECT_EQ(vecD(0), ValueT(gr * vecA(0)));
        EXPECT_EQ(vecD(1), ValueT(gr * vecA(1)));
        EXPECT_EQ(vecD(2), ValueT(gr * vecA(2)));

        // Multiply each element of the vector by the corresponding element
        Vec3T vecE = vecA;
        Vec3T vecF(ValueT(-2.5), ValueT(1.2), ValueT(3.14159));
        vecE *= vecF;
        EXPECT_EQ(vecE(0), ValueT(vecA(0) * vecF(0)));
        EXPECT_EQ(vecE(1), ValueT(vecA(1) * vecF(1)));
        EXPECT_EQ(vecE(2), ValueT(vecA(2) * vecF(2)));

        // Divide each element of the vector by a scalar
        Vec3T vecG = vecA;
        vecG /= gr;
        EXPECT_EQ(vecG(0), ValueT(vecA(0) / gr));
        EXPECT_EQ(vecG(1), ValueT(vecA(1) / gr));
        EXPECT_EQ(vecG(2), ValueT(vecA(2) / gr));

        // Divide each element of the vector by the corresponding element of the given vector
        Vec3T vecH = vecA;
        vecH /= vecF;
        EXPECT_EQ(vecH(0), ValueT(vecA(0) / vecF(0)));
        EXPECT_EQ(vecH(1), ValueT(vecA(1) / vecF(1)));
        EXPECT_EQ(vecH(2), ValueT(vecA(2) / vecF(2)));

        // Add a scalar to each element of the vector
        Vec3T vecI = vecA;
        vecI += gr;
        EXPECT_EQ(vecI(0), ValueT(vecA(0) + gr));
        EXPECT_EQ(vecI(1), ValueT(vecA(1) + gr));
        EXPECT_EQ(vecI(2), ValueT(vecA(2) + gr));

        // Add each element of the given vector to the corresponding element of this vector
        Vec3T vecJ = vecA;
        vecJ += vecF;
        EXPECT_EQ(vecJ(0), ValueT(vecA(0) + vecF(0)));
        EXPECT_EQ(vecJ(1), ValueT(vecA(1) + vecF(1)));
        EXPECT_EQ(vecJ(2), ValueT(vecA(2) + vecF(2)));

        // Subtract a scalar from each element of this vector
        Vec3T vecK = vecA;
        vecK -= gr;
        EXPECT_EQ(vecK(0), ValueT(vecA(0) - gr));
        EXPECT_EQ(vecK(1), ValueT(vecA(1) - gr));
        EXPECT_EQ(vecK(2), ValueT(vecA(2) - gr));

        // Subtract each element of the given vector from the corresponding element of this vector
        Vec3T vecL = vecA;
        vecL -= vecF;
        EXPECT_EQ(vecL(0), ValueT(vecA(0) - vecF(0)));
        EXPECT_EQ(vecL(1), ValueT(vecA(1) - vecF(1)));
        EXPECT_EQ(vecL(2), ValueT(vecA(2) - vecF(2)));
    }
}

TEST_F(TestMath, testMemberOperators)
{
    using namespace openvdb;

    testMemberOperatorsImpl<math::half>();
    testMemberOperatorsImpl<float>();
    testMemberOperatorsImpl<double>();
}


template<typename ValueT>
void testFreeFunctionsOperatorsImpl()
{
    using namespace openvdb;
    using Vec3T = math::Vec3<ValueT>;

    {
        Vec3T vecA(ValueT(1),ValueT(2),ValueT(3));
        Vec3T vecB(ValueT(3),ValueT(4),ValueT(5));
        const ValueT gr = ValueT(1.6180339887); // golden ratio

        /// Check equality operator, does exact floating point comparisons ==
        bool eqRes = vecA == vecB;
        EXPECT_FALSE(eqRes);

        /// Check inequality operator, does exact floating point comparisons !=
        bool ineqRes = vecA != vecB;
        EXPECT_TRUE(ineqRes);

        /// Check multiplication: scalar * vec
        Vec3T sclrMultA = gr * vecA;
        EXPECT_EQ(sclrMultA(0), ValueT(vecA(0) * gr));
        EXPECT_EQ(sclrMultA(1), ValueT(vecA(1) * gr));
        EXPECT_EQ(sclrMultA(2), ValueT(vecA(2) * gr));

        /// Check multiplication: vec * scalar
        Vec3T sclrMultB = vecA * gr;
        EXPECT_EQ(sclrMultB(0), ValueT(vecA(0) * gr));
        EXPECT_EQ(sclrMultB(1), ValueT(vecA(1) * gr));
        EXPECT_EQ(sclrMultB(2), ValueT(vecA(2) * gr));

        /// Check multiplication vec0 * vec1
        Vec3T multRes = vecA * vecB;
        EXPECT_EQ(multRes, Vec3T(ValueT(3), ValueT(8), ValueT(15)));

        /// Check: scalar / vec
        Vec3T sclrDivA = gr / vecA;
        EXPECT_EQ(sclrDivA(0), ValueT(gr / vecA(0)));
        EXPECT_EQ(sclrDivA(1), ValueT(gr / vecA(1)));
        EXPECT_EQ(sclrDivA(2), ValueT(gr / vecA(2)));

        /// Check: vec / scalar
        Vec3T scalarDivB = vecA / gr;
        EXPECT_EQ(scalarDivB(0), ValueT(vecA(0) / gr));
        EXPECT_EQ(scalarDivB(1), ValueT(vecA(1) / gr));
        EXPECT_EQ(scalarDivB(2), ValueT(vecA(2) / gr));

        /// Check element-wise div: vec0 / vec1
        Vec3T divRes = vecA / vecB;
        EXPECT_EQ(divRes(0), ValueT(ValueT(vecA(0)) / ValueT(vecB(0))));
        EXPECT_EQ(divRes(1), ValueT(ValueT(vecA(1)) / ValueT(vecB(1))));
        EXPECT_EQ(divRes(2), ValueT(ValueT(vecA(2)) / ValueT(vecB(2))));

        /// Check addition: vec0 + vec1
        Vec3T addRes = vecA + vecB;
        EXPECT_EQ(addRes, Vec3T(ValueT(4), ValueT(6), ValueT(8)));

        /// Check scalar addition: a + vec
        Vec3T addSclrRes = vecA + gr;
        EXPECT_EQ(addSclrRes(0), ValueT(vecA(0) + gr));
        EXPECT_EQ(addSclrRes(1), ValueT(vecA(1) + gr));
        EXPECT_EQ(addSclrRes(2), ValueT(vecA(2) + gr));

        /// Check element-wise subtraction: vec0 - vec1
        Vec3T subtractRes = vecA - vecB;
        EXPECT_EQ(subtractRes, Vec3T(ValueT(-2), ValueT(-2), ValueT(-2)));

        /// Check scalar subtraction: vec0 - a
        Vec3T subSclrRes = vecA - gr;
        EXPECT_EQ(subSclrRes(0), ValueT(vecA(0) - gr));
        EXPECT_EQ(subSclrRes(1), ValueT(vecA(1) - gr));
        EXPECT_EQ(subSclrRes(2), ValueT(vecA(2) - gr));
    }
}

TEST_F(TestMath, testFreeFunctionOperators)
{
    using namespace openvdb;
    testFreeFunctionsOperatorsImpl<math::half>();
    testFreeFunctionsOperatorsImpl<float>();
    testFreeFunctionsOperatorsImpl<double>();
}


// This suite of tests obviously needs to be expanded!
TEST_F(TestMath, testAll)
{
    using namespace openvdb;

    {// Sign
        EXPECT_EQ(math::Sign( 3   ), 1);
        EXPECT_EQ(math::Sign(-1.0 ),-1);
        EXPECT_EQ(math::Sign( 0.0f), 0);
        EXPECT_EQ(math::Sign(math::half(0.)), 0);
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
