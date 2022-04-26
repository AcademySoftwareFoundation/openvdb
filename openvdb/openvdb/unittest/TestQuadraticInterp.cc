// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
//
/// @file TestQuadraticInterp.cc

#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

#include <gtest/gtest.h>

#include <sstream>


namespace {
// Absolute tolerance for floating-point equality comparisons
const double TOLERANCE = 1.0e-5;
}


////////////////////////////////////////


template<typename GridType>
class TestQuadraticInterp
{
public:
    typedef typename GridType::ValueType ValueT;
    typedef typename GridType::Ptr GridPtr;
    struct TestVal { float x, y, z; ValueT expected; };

    static void test();
    static void testConstantValues();
    static void testFillValues();
    static void testNegativeIndices();

protected:
    static void executeTest(const GridPtr&, const TestVal*, size_t numVals);

    /// Initialize an arbitrary ValueType from a scalar.
    static inline ValueT constValue(double d) { return ValueT(d); }

    /// Compare two numeric values for equality within an absolute tolerance.
    static inline bool relEq(const ValueT& v1, const ValueT& v2)
        { return fabs(v1 - v2) <= TOLERANCE; }
};


class TestQuadraticInterpTest: public ::testing::Test
{
};


////////////////////////////////////////


/// Specialization for Vec3s grids
template<>
inline openvdb::Vec3s
TestQuadraticInterp<openvdb::Vec3SGrid>::constValue(double d)
{
    return openvdb::Vec3s(float(d), float(d), float(d));
}

/// Specialization for Vec3s grids
template<>
inline bool
TestQuadraticInterp<openvdb::Vec3SGrid>::relEq(
    const openvdb::Vec3s& v1, const openvdb::Vec3s& v2)
{
    return v1.eq(v2, float(TOLERANCE));
}


/// Sample the given tree at various locations and assert if
/// any of the sampled values don't match the expected values.
template<typename GridType>
void
TestQuadraticInterp<GridType>::executeTest(const GridPtr& grid,
    const TestVal* testVals, size_t numVals)
{
    openvdb::tools::GridSampler<GridType, openvdb::tools::QuadraticSampler> interpolator(*grid);
    //openvdb::tools::QuadraticInterp<GridType> interpolator(*tree);

    for (size_t i = 0; i < numVals; ++i) {
        const TestVal& val = testVals[i];
        const ValueT actual = interpolator.sampleVoxel(val.x, val.y, val.z);
        if (!relEq(val.expected, actual)) {
            std::ostringstream ostr;
            ostr << std::setprecision(10)
                << "sampleVoxel(" << val.x << ", " << val.y << ", " << val.z
                << "): expected " << val.expected << ", got " << actual;
            FAIL() << ostr.str();
        }
    }
}


template<typename GridType>
void
TestQuadraticInterp<GridType>::test()
{
    const ValueT
        one = constValue(1),
        two = constValue(2),
        three = constValue(3),
        four = constValue(4),
        fillValue = constValue(256);

    GridPtr grid(new GridType(fillValue));
    typename GridType::TreeType& tree = grid->tree();

    tree.setValue(openvdb::Coord(10, 10, 10), one);

    tree.setValue(openvdb::Coord(11, 10, 10), two);
    tree.setValue(openvdb::Coord(11, 11, 10), two);
    tree.setValue(openvdb::Coord(10, 11, 10), two);
    tree.setValue(openvdb::Coord( 9, 11, 10), two);
    tree.setValue(openvdb::Coord( 9, 10, 10), two);
    tree.setValue(openvdb::Coord( 9,  9, 10), two);
    tree.setValue(openvdb::Coord(10,  9, 10), two);
    tree.setValue(openvdb::Coord(11,  9, 10), two);

    tree.setValue(openvdb::Coord(10, 10, 11), three);
    tree.setValue(openvdb::Coord(11, 10, 11), three);
    tree.setValue(openvdb::Coord(11, 11, 11), three);
    tree.setValue(openvdb::Coord(10, 11, 11), three);
    tree.setValue(openvdb::Coord( 9, 11, 11), three);
    tree.setValue(openvdb::Coord( 9, 10, 11), three);
    tree.setValue(openvdb::Coord( 9,  9, 11), three);
    tree.setValue(openvdb::Coord(10,  9, 11), three);
    tree.setValue(openvdb::Coord(11,  9, 11), three);

    tree.setValue(openvdb::Coord(10, 10, 9), four);
    tree.setValue(openvdb::Coord(11, 10, 9), four);
    tree.setValue(openvdb::Coord(11, 11, 9), four);
    tree.setValue(openvdb::Coord(10, 11, 9), four);
    tree.setValue(openvdb::Coord( 9, 11, 9), four);
    tree.setValue(openvdb::Coord( 9, 10, 9), four);
    tree.setValue(openvdb::Coord( 9,  9, 9), four);
    tree.setValue(openvdb::Coord(10,  9, 9), four);
    tree.setValue(openvdb::Coord(11,  9, 9), four);

    const TestVal testVals[] = {
        { 10.5f, 10.5f, 10.5f, constValue(1.703125) },
        { 10.0f, 10.0f, 10.0f, one },
        { 11.0f, 10.0f, 10.0f, two },
        { 11.0f, 11.0f, 10.0f, two },
        { 11.0f, 11.0f, 11.0f, three },
        {  9.0f, 11.0f,  9.0f, four },
        {  9.0f, 10.0f,  9.0f, four },
        { 10.1f, 10.0f, 10.0f, constValue(1.01) },
        { 10.8f, 10.8f, 10.8f, constValue(2.513344) },
        { 10.1f, 10.8f, 10.5f, constValue(1.8577) },
        { 10.8f, 10.1f, 10.5f, constValue(1.8577) },
        { 10.5f, 10.1f, 10.8f, constValue(2.2927) },
        { 10.5f, 10.8f, 10.1f, constValue(1.6977) },
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(grid, testVals, numVals);
}
TEST_F(TestQuadraticInterpTest, testFloat) { TestQuadraticInterp<openvdb::FloatGrid>::test(); }
TEST_F(TestQuadraticInterpTest, testDouble) { TestQuadraticInterp<openvdb::DoubleGrid>::test(); }
TEST_F(TestQuadraticInterpTest, testVec3S) { TestQuadraticInterp<openvdb::Vec3SGrid>::test(); }


template<typename GridType>
void
TestQuadraticInterp<GridType>::testConstantValues()
{
    const ValueT
        two = constValue(2),
        fillValue = constValue(256);

    GridPtr grid(new GridType(fillValue));
    typename GridType::TreeType& tree = grid->tree();

    tree.setValue(openvdb::Coord(10, 10, 10), two);

    tree.setValue(openvdb::Coord(11, 10, 10), two);
    tree.setValue(openvdb::Coord(11, 11, 10), two);
    tree.setValue(openvdb::Coord(10, 11, 10), two);
    tree.setValue(openvdb::Coord( 9, 11, 10), two);
    tree.setValue(openvdb::Coord( 9, 10, 10), two);
    tree.setValue(openvdb::Coord( 9,  9, 10), two);
    tree.setValue(openvdb::Coord(10,  9, 10), two);
    tree.setValue(openvdb::Coord(11,  9, 10), two);

    tree.setValue(openvdb::Coord(10, 10, 11), two);
    tree.setValue(openvdb::Coord(11, 10, 11), two);
    tree.setValue(openvdb::Coord(11, 11, 11), two);
    tree.setValue(openvdb::Coord(10, 11, 11), two);
    tree.setValue(openvdb::Coord( 9, 11, 11), two);
    tree.setValue(openvdb::Coord( 9, 10, 11), two);
    tree.setValue(openvdb::Coord( 9,  9, 11), two);
    tree.setValue(openvdb::Coord(10,  9, 11), two);
    tree.setValue(openvdb::Coord(11,  9, 11), two);

    tree.setValue(openvdb::Coord(10, 10, 9), two);
    tree.setValue(openvdb::Coord(11, 10, 9), two);
    tree.setValue(openvdb::Coord(11, 11, 9), two);
    tree.setValue(openvdb::Coord(10, 11, 9), two);
    tree.setValue(openvdb::Coord( 9, 11, 9), two);
    tree.setValue(openvdb::Coord( 9, 10, 9), two);
    tree.setValue(openvdb::Coord( 9,  9, 9), two);
    tree.setValue(openvdb::Coord(10,  9, 9), two);
    tree.setValue(openvdb::Coord(11,  9, 9), two);

    const TestVal testVals[] = {
        { 10.5f, 10.5f, 10.5f, two },
        { 10.0f, 10.0f, 10.0f, two },
        { 10.1f, 10.0f, 10.0f, two },
        { 10.8f, 10.8f, 10.8f, two },
        { 10.1f, 10.8f, 10.5f, two },
        { 10.8f, 10.1f, 10.5f, two },
        { 10.5f, 10.1f, 10.8f, two },
        { 10.5f, 10.8f, 10.1f, two }
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(grid, testVals, numVals);
}
TEST_F(TestQuadraticInterpTest, testConstantValuesFloat) { TestQuadraticInterp<openvdb::FloatGrid>::testConstantValues(); }
TEST_F(TestQuadraticInterpTest, testConstantValuesDouble) { TestQuadraticInterp<openvdb::DoubleGrid>::testConstantValues(); }
TEST_F(TestQuadraticInterpTest, testConstantValuesVec3S) { TestQuadraticInterp<openvdb::Vec3SGrid>::testConstantValues(); }


template<typename GridType>
void
TestQuadraticInterp<GridType>::testFillValues()
{
    const ValueT fillValue = constValue(256);

    GridPtr grid(new GridType(fillValue));

    const TestVal testVals[] = {
        { 10.5f, 10.5f, 10.5f, fillValue },
        { 10.0f, 10.0f, 10.0f, fillValue },
        { 10.1f, 10.0f, 10.0f, fillValue },
        { 10.8f, 10.8f, 10.8f, fillValue },
        { 10.1f, 10.8f, 10.5f, fillValue },
        { 10.8f, 10.1f, 10.5f, fillValue },
        { 10.5f, 10.1f, 10.8f, fillValue },
        { 10.5f, 10.8f, 10.1f, fillValue }
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(grid, testVals, numVals);
}
TEST_F(TestQuadraticInterpTest, testFillValuesFloat) { TestQuadraticInterp<openvdb::FloatGrid>::testFillValues(); }
TEST_F(TestQuadraticInterpTest, testFillValuesDouble) { TestQuadraticInterp<openvdb::DoubleGrid>::testFillValues(); }
TEST_F(TestQuadraticInterpTest, testFillValuesVec3S) { TestQuadraticInterp<openvdb::Vec3SGrid>::testFillValues(); }


template<typename GridType>
void
TestQuadraticInterp<GridType>::testNegativeIndices()
{
    const ValueT
        one = constValue(1),
        two = constValue(2),
        three = constValue(3),
        four = constValue(4),
        fillValue = constValue(256);

    GridPtr grid(new GridType(fillValue));
    typename GridType::TreeType& tree = grid->tree();

    tree.setValue(openvdb::Coord(-10, -10, -10), one);

    tree.setValue(openvdb::Coord(-11, -10, -10), two);
    tree.setValue(openvdb::Coord(-11, -11, -10), two);
    tree.setValue(openvdb::Coord(-10, -11, -10), two);
    tree.setValue(openvdb::Coord( -9, -11, -10), two);
    tree.setValue(openvdb::Coord( -9, -10, -10), two);
    tree.setValue(openvdb::Coord( -9,  -9, -10), two);
    tree.setValue(openvdb::Coord(-10,  -9, -10), two);
    tree.setValue(openvdb::Coord(-11,  -9, -10), two);

    tree.setValue(openvdb::Coord(-10, -10, -11), three);
    tree.setValue(openvdb::Coord(-11, -10, -11), three);
    tree.setValue(openvdb::Coord(-11, -11, -11), three);
    tree.setValue(openvdb::Coord(-10, -11, -11), three);
    tree.setValue(openvdb::Coord( -9, -11, -11), three);
    tree.setValue(openvdb::Coord( -9, -10, -11), three);
    tree.setValue(openvdb::Coord( -9,  -9, -11), three);
    tree.setValue(openvdb::Coord(-10,  -9, -11), three);
    tree.setValue(openvdb::Coord(-11,  -9, -11), three);

    tree.setValue(openvdb::Coord(-10, -10, -9), four);
    tree.setValue(openvdb::Coord(-11, -10, -9), four);
    tree.setValue(openvdb::Coord(-11, -11, -9), four);
    tree.setValue(openvdb::Coord(-10, -11, -9), four);
    tree.setValue(openvdb::Coord( -9, -11, -9), four);
    tree.setValue(openvdb::Coord( -9, -10, -9), four);
    tree.setValue(openvdb::Coord( -9,  -9, -9), four);
    tree.setValue(openvdb::Coord(-10,  -9, -9), four);
    tree.setValue(openvdb::Coord(-11,  -9, -9), four);

    const TestVal testVals[] = {
        { -10.5f, -10.5f, -10.5f, constValue(-104.75586) },
        { -10.0f, -10.0f, -10.0f, one },
        { -11.0f, -10.0f, -10.0f, two },
        { -11.0f, -11.0f, -10.0f, two },
        { -11.0f, -11.0f, -11.0f, three },
        {  -9.0f, -11.0f,  -9.0f, four },
        {  -9.0f, -10.0f,  -9.0f, four },
        { -10.1f, -10.0f, -10.0f, constValue(-10.28504) },
        { -10.8f, -10.8f, -10.8f, constValue(-62.84878) },
        { -10.1f, -10.8f, -10.5f, constValue(-65.68951) },
        { -10.8f, -10.1f, -10.5f, constValue(-65.68951) },
        { -10.5f, -10.1f, -10.8f, constValue(-65.40736) },
        { -10.5f, -10.8f, -10.1f, constValue(-66.30510) },
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(grid, testVals, numVals);
}
TEST_F(TestQuadraticInterpTest, testNegativeIndicesFloat) { TestQuadraticInterp<openvdb::FloatGrid>::testNegativeIndices(); }
TEST_F(TestQuadraticInterpTest, testNegativeIndicesDouble) { TestQuadraticInterp<openvdb::DoubleGrid>::testNegativeIndices(); }
TEST_F(TestQuadraticInterpTest, testNegativeIndicesVec3S) { TestQuadraticInterp<openvdb::Vec3SGrid>::testNegativeIndices(); }
