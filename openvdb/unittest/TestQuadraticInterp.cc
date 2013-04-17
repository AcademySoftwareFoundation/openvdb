///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @file TestQuadraticInterp.cc
///
/// @author Peter Cucka

#include <sstream>
#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>

// CPPUNIT_TEST_SUITE() invokes CPPUNIT_TESTNAMER_DECL() to generate a suite name
// from the FixtureType.  But if FixtureType is a templated type, the generated name
// can become long and messy.  This macro overrides the normal naming logic,
// instead invoking FixtureType::testSuiteName(), which should be a static member
// function that returns a std::string containing the suite name for the specific
// template instantiation.
#undef CPPUNIT_TESTNAMER_DECL
#define CPPUNIT_TESTNAMER_DECL( variableName, FixtureType ) \
    CPPUNIT_NS::TestNamer variableName( FixtureType::testSuiteName() )


namespace {
// Absolute tolerance for floating-point equality comparisons
const double TOLERANCE = 1.0e-5;
}


////////////////////////////////////////


template<typename TreeType>
class TestQuadraticInterp: public CppUnit::TestCase
{
public:
    typedef typename TreeType::ValueType ValueT;
    typedef typename TreeType::Ptr TreePtr;
    struct TestVal { float x, y, z; ValueT expected; };

    static std::string testSuiteName()
    {
        std::string name = openvdb::typeNameAsString<ValueT>();
        if (!name.empty()) name[0] = ::toupper(name[0]);
        // alternatively, "std::string name = TreeType::treeType();"
        return "TestQuadraticInterp" + name;
    }

    CPPUNIT_TEST_SUITE(TestQuadraticInterp);
    CPPUNIT_TEST(test);
    CPPUNIT_TEST(testConstantValues);
    CPPUNIT_TEST(testFillValues);
    CPPUNIT_TEST(testNegativeIndices);
    CPPUNIT_TEST_SUITE_END();

    void test();
    void testConstantValues();
    void testFillValues();
    void testNegativeIndices();

private:
    void executeTest(const TreePtr&, const TestVal*, size_t numVals) const;

    /// Initialize an arbitrary ValueType from a scalar.
    static inline ValueT constValue(double d) { return ValueT(d); }

    /// Compare two numeric values for equality within an absolute tolerance.
    static inline bool relEq(const ValueT& v1, const ValueT& v2)
        { return fabs(v1 - v2) <= TOLERANCE; }
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestQuadraticInterp<openvdb::FloatTree>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestQuadraticInterp<openvdb::DoubleTree>);
CPPUNIT_TEST_SUITE_REGISTRATION(TestQuadraticInterp<openvdb::Vec3STree>);


////////////////////////////////////////


/// Specialization for Vec3s trees
template<>
inline openvdb::Vec3s
TestQuadraticInterp<openvdb::Vec3STree>::constValue(double d)
{
    return openvdb::Vec3s(d, d, d);
}

/// Specialization for Vec3s trees
template<>
inline bool
TestQuadraticInterp<openvdb::Vec3STree>::relEq(
    const openvdb::Vec3s& v1, const openvdb::Vec3s& v2)
{
    return v1.eq(v2, TOLERANCE);
}


/// Sample the given tree at various locations and assert if
/// any of the sampled values don't match the expected values.
template<typename TreeType>
void
TestQuadraticInterp<TreeType>::executeTest(const TreePtr& tree,
    const TestVal* testVals, size_t numVals) const
{
    openvdb::tools::GridSampler<TreeType, openvdb::tools::QuadraticSampler> interpolator(*tree);
    //openvdb::tools::QuadraticInterp<TreeType> interpolator(*tree);

    for (size_t i = 0; i < numVals; ++i) {
        const TestVal& val = testVals[i];
        const ValueT actual = interpolator.sampleVoxel(val.x, val.y, val.z);
        if (!relEq(val.expected, actual)) {
            std::ostringstream ostr;
            ostr << std::setprecision(10)
                << "sampleVoxel(" << val.x << ", " << val.y << ", " << val.z
                << "): expected " << val.expected << ", got " << actual;
            CPPUNIT_FAIL(ostr.str());
        }
    }
}


template<typename TreeType>
void
TestQuadraticInterp<TreeType>::test()
{
    const ValueT
        one = constValue(1),
        two = constValue(2),
        three = constValue(3),
        four = constValue(4),
        fillValue = constValue(256);

    TreePtr tree(new TreeType(fillValue));

    tree->setValue(openvdb::Coord(10, 10, 10), one);

    tree->setValue(openvdb::Coord(11, 10, 10), two);
    tree->setValue(openvdb::Coord(11, 11, 10), two);
    tree->setValue(openvdb::Coord(10, 11, 10), two);
    tree->setValue(openvdb::Coord( 9, 11, 10), two);
    tree->setValue(openvdb::Coord( 9, 10, 10), two);
    tree->setValue(openvdb::Coord( 9,  9, 10), two);
    tree->setValue(openvdb::Coord(10,  9, 10), two);
    tree->setValue(openvdb::Coord(11,  9, 10), two);

    tree->setValue(openvdb::Coord(10, 10, 11), three);
    tree->setValue(openvdb::Coord(11, 10, 11), three);
    tree->setValue(openvdb::Coord(11, 11, 11), three);
    tree->setValue(openvdb::Coord(10, 11, 11), three);
    tree->setValue(openvdb::Coord( 9, 11, 11), three);
    tree->setValue(openvdb::Coord( 9, 10, 11), three);
    tree->setValue(openvdb::Coord( 9,  9, 11), three);
    tree->setValue(openvdb::Coord(10,  9, 11), three);
    tree->setValue(openvdb::Coord(11,  9, 11), three);

    tree->setValue(openvdb::Coord(10, 10, 9), four);
    tree->setValue(openvdb::Coord(11, 10, 9), four);
    tree->setValue(openvdb::Coord(11, 11, 9), four);
    tree->setValue(openvdb::Coord(10, 11, 9), four);
    tree->setValue(openvdb::Coord( 9, 11, 9), four);
    tree->setValue(openvdb::Coord( 9, 10, 9), four);
    tree->setValue(openvdb::Coord( 9,  9, 9), four);
    tree->setValue(openvdb::Coord(10,  9, 9), four);
    tree->setValue(openvdb::Coord(11,  9, 9), four);

    const TestVal testVals[] = {
        { 10.5, 10.5, 10.5, constValue(1.703125) },
        { 10.0, 10.0, 10.0, one },
        { 11.0, 10.0, 10.0, two },
        { 11.0, 11.0, 10.0, two },
        { 11.0, 11.0, 11.0, three },
        {  9.0, 11.0,  9.0, four },
        {  9.0, 10.0,  9.0, four },
        { 10.1, 10.0, 10.0, constValue(1.01) },
        { 10.8, 10.8, 10.8, constValue(2.513344) },
        { 10.1, 10.8, 10.5, constValue(1.8577) },
        { 10.8, 10.1, 10.5, constValue(1.8577) },
        { 10.5, 10.1, 10.8, constValue(2.2927) },
        { 10.5, 10.8, 10.1, constValue(1.6977) },
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(tree, testVals, numVals);
}


template<typename TreeType>
void
TestQuadraticInterp<TreeType>::testConstantValues()
{
    const ValueT
        two = constValue(2),
        fillValue = constValue(256);

    TreePtr tree(new TreeType(fillValue));

    tree->setValue(openvdb::Coord(10, 10, 10), two);

    tree->setValue(openvdb::Coord(11, 10, 10), two);
    tree->setValue(openvdb::Coord(11, 11, 10), two);
    tree->setValue(openvdb::Coord(10, 11, 10), two);
    tree->setValue(openvdb::Coord( 9, 11, 10), two);
    tree->setValue(openvdb::Coord( 9, 10, 10), two);
    tree->setValue(openvdb::Coord( 9,  9, 10), two);
    tree->setValue(openvdb::Coord(10,  9, 10), two);
    tree->setValue(openvdb::Coord(11,  9, 10), two);

    tree->setValue(openvdb::Coord(10, 10, 11), two);
    tree->setValue(openvdb::Coord(11, 10, 11), two);
    tree->setValue(openvdb::Coord(11, 11, 11), two);
    tree->setValue(openvdb::Coord(10, 11, 11), two);
    tree->setValue(openvdb::Coord( 9, 11, 11), two);
    tree->setValue(openvdb::Coord( 9, 10, 11), two);
    tree->setValue(openvdb::Coord( 9,  9, 11), two);
    tree->setValue(openvdb::Coord(10,  9, 11), two);
    tree->setValue(openvdb::Coord(11,  9, 11), two);

    tree->setValue(openvdb::Coord(10, 10, 9), two);
    tree->setValue(openvdb::Coord(11, 10, 9), two);
    tree->setValue(openvdb::Coord(11, 11, 9), two);
    tree->setValue(openvdb::Coord(10, 11, 9), two);
    tree->setValue(openvdb::Coord( 9, 11, 9), two);
    tree->setValue(openvdb::Coord( 9, 10, 9), two);
    tree->setValue(openvdb::Coord( 9,  9, 9), two);
    tree->setValue(openvdb::Coord(10,  9, 9), two);
    tree->setValue(openvdb::Coord(11,  9, 9), two);

    const TestVal testVals[] = {
        { 10.5, 10.5, 10.5, two },
        { 10.0, 10.0, 10.0, two },
        { 10.1, 10.0, 10.0, two },
        { 10.8, 10.8, 10.8, two },
        { 10.1, 10.8, 10.5, two },
        { 10.8, 10.1, 10.5, two },
        { 10.5, 10.1, 10.8, two },
        { 10.5, 10.8, 10.1, two }
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(tree, testVals, numVals);
}


template<typename TreeType>
void
TestQuadraticInterp<TreeType>::testFillValues()
{
    const ValueT fillValue = constValue(256);

    TreePtr tree(new TreeType(fillValue));

    const TestVal testVals[] = {
        { 10.5, 10.5, 10.5, fillValue },
        { 10.0, 10.0, 10.0, fillValue },
        { 10.1, 10.0, 10.0, fillValue },
        { 10.8, 10.8, 10.8, fillValue },
        { 10.1, 10.8, 10.5, fillValue },
        { 10.8, 10.1, 10.5, fillValue },
        { 10.5, 10.1, 10.8, fillValue },
        { 10.5, 10.8, 10.1, fillValue }
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(tree, testVals, numVals);
}


template<typename TreeType>
void
TestQuadraticInterp<TreeType>::testNegativeIndices()
{
    const ValueT
        one = constValue(1),
        two = constValue(2),
        three = constValue(3),
        four = constValue(4),
        fillValue = constValue(256);

    TreePtr tree(new TreeType(fillValue));

    tree->setValue(openvdb::Coord(-10, -10, -10), one);

    tree->setValue(openvdb::Coord(-11, -10, -10), two);
    tree->setValue(openvdb::Coord(-11, -11, -10), two);
    tree->setValue(openvdb::Coord(-10, -11, -10), two);
    tree->setValue(openvdb::Coord( -9, -11, -10), two);
    tree->setValue(openvdb::Coord( -9, -10, -10), two);
    tree->setValue(openvdb::Coord( -9,  -9, -10), two);
    tree->setValue(openvdb::Coord(-10,  -9, -10), two);
    tree->setValue(openvdb::Coord(-11,  -9, -10), two);

    tree->setValue(openvdb::Coord(-10, -10, -11), three);
    tree->setValue(openvdb::Coord(-11, -10, -11), three);
    tree->setValue(openvdb::Coord(-11, -11, -11), three);
    tree->setValue(openvdb::Coord(-10, -11, -11), three);
    tree->setValue(openvdb::Coord( -9, -11, -11), three);
    tree->setValue(openvdb::Coord( -9, -10, -11), three);
    tree->setValue(openvdb::Coord( -9,  -9, -11), three);
    tree->setValue(openvdb::Coord(-10,  -9, -11), three);
    tree->setValue(openvdb::Coord(-11,  -9, -11), three);

    tree->setValue(openvdb::Coord(-10, -10, -9), four);
    tree->setValue(openvdb::Coord(-11, -10, -9), four);
    tree->setValue(openvdb::Coord(-11, -11, -9), four);
    tree->setValue(openvdb::Coord(-10, -11, -9), four);
    tree->setValue(openvdb::Coord( -9, -11, -9), four);
    tree->setValue(openvdb::Coord( -9, -10, -9), four);
    tree->setValue(openvdb::Coord( -9,  -9, -9), four);
    tree->setValue(openvdb::Coord(-10,  -9, -9), four);
    tree->setValue(openvdb::Coord(-11,  -9, -9), four);

    const TestVal testVals[] = {
        { -10.5, -10.5, -10.5, constValue(-104.75586) },
        { -10.0, -10.0, -10.0, one },
        { -11.0, -10.0, -10.0, two },
        { -11.0, -11.0, -10.0, two },
        { -11.0, -11.0, -11.0, three },
        {  -9.0, -11.0,  -9.0, four },
        {  -9.0, -10.0,  -9.0, four },
        { -10.1, -10.0, -10.0, constValue(-10.28504) },
        { -10.8, -10.8, -10.8, constValue(-62.84878) },
        { -10.1, -10.8, -10.5, constValue(-65.68951) },
        { -10.8, -10.1, -10.5, constValue(-65.68951) },
        { -10.5, -10.1, -10.8, constValue(-65.40736) },
        { -10.5, -10.8, -10.1, constValue(-66.30510) },
    };
    const size_t numVals = sizeof(testVals) / sizeof(TestVal);

    executeTest(tree, testVals, numVals);
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
