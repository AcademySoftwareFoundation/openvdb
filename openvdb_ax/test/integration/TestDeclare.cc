///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
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

#include "TestHarness.h"

#include <openvdb_ax/test/util.h>

#include <openvdb/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestDeclare : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestDeclare);
    CPPUNIT_TEST(testLocalVariables);
    CPPUNIT_TEST(testLocalVectorVariables);
    CPPUNIT_TEST(testAttributes);
    CPPUNIT_TEST(testVectorAttributes);
    CPPUNIT_TEST(testNewAttributes);
    CPPUNIT_TEST(testNewVectorAttributes);
    CPPUNIT_TEST(testVectorAttributeImplicit);
    CPPUNIT_TEST(testAmbiguousScalarAttributes);
    CPPUNIT_TEST(testAmbiguousVectorAttributes);
    CPPUNIT_TEST(testAmbiguousScalarExternals);
    CPPUNIT_TEST(testAmbiguousVectorExternals);
    CPPUNIT_TEST(testAttributesVolume);
    CPPUNIT_TEST_SUITE_END();

    void testLocalVariables();
    void testAttributes();
    void testNewAttributes();
    void testNewVectorAttributes();
    void testLocalVectorVariables();
    void testVectorAttributes();
    void testVectorAttributeImplicit();
    void testAmbiguousScalarAttributes();
    void testAmbiguousVectorAttributes();
    void testAmbiguousScalarExternals();
    void testAmbiguousVectorExternals();
    void testAttributesVolume();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDeclare);

void
TestDeclare::testLocalVariables()
{
    mHarness.executeCode("test/snippets/declare/declareLocalVariables");

    // input data should not have changed
    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testLocalVectorVariables()
{
    mHarness.executeCode("test/snippets/declare/declareLocalVectorVariables");

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testAttributes()
{
    mHarness.addAttributes<float>(unittest_util::nameSequence("float_test", 4),
        {0.0f, 0.2f, 10.0f, 10.0f});
    mHarness.addAttributes<int>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});

    mHarness.addAttribute("short_test", int16_t(1));
    mHarness.addAttribute("long_test", int64_t(3));
    mHarness.addAttribute("double_test", 0.3);

    mHarness.executeCode("test/snippets/declare/declareAttributes");

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testAttributesVolume()
{
    mHarness.addAttributes<float>(unittest_util::nameSequence("float_test", 4),
        {0.0f, 0.2f, 10.0f, 10.0f});
    mHarness.addAttributes<int>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});

    mHarness.addAttribute("long_test", int64_t(3));
    mHarness.addAttribute("double_test", 0.3);

    mHarness.executeCode("test/snippets/declare/declareAttributesVolume");

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testNewAttributes()
{
    mHarness.addExpectedAttributes<float>(unittest_util::nameSequence("float_test", 4),
        {0.0f, 0.2f, 10.0f, 10.0f});
    mHarness.addExpectedAttributes<int>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});

    mHarness.addExpectedAttribute("short_test", int16_t(1));
    mHarness.addExpectedAttribute("long_test", int64_t(3));
    mHarness.addExpectedAttribute("double_test", 0.3);

    // Volume data needs to exist to be tested
    mHarness.addInputVolumes<float>(unittest_util::nameSequence("float_test", 4),
        {0.0f, 0.2f, 10.0f, 10.0f});
    mHarness.addInputVolumes<int>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});
    mHarness.addInputVolumes<int16_t>({"short_test"}, {int16_t(1)});
    mHarness.addInputVolumes<int64_t>({"long_test"}, {int64_t(3)});
    mHarness.addInputVolumes<double>({"double_test"}, {0.3});

    mHarness.executeCode("test/snippets/declare/declareAttributes", nullptr, nullptr, true);

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testNewVectorAttributes()
{
    mHarness.addExpectedAttributes<openvdb::Vec3f>({"vec_float_test", "vec_float_test2"},
        {openvdb::Vec3f::zero(), openvdb::Vec3f(0.2f, 0.3f, 0.4f)});
    mHarness.addExpectedAttributes<openvdb::Vec3i>({"vec_int_test", "vec_int_test2"},
        {openvdb::Vec3i::zero(), openvdb::Vec3i(5, 6, 7)});
    mHarness.addExpectedAttribute<openvdb::Vec3d>("vec_double_test", openvdb::Vec3d(0.3, 0.4, 0.5));

    // Volume data needs to exist to be tested
    mHarness.addInputVolumes<openvdb::Vec3f>({"vec_float_test", "vec_float_test2"},
        {openvdb::Vec3f::zero(), openvdb::Vec3f(0.2f, 0.3f, 0.4f)});
    mHarness.addInputVolumes<openvdb::Vec3i>({"vec_int_test", "vec_int_test2"},
        {openvdb::Vec3i::zero(), openvdb::Vec3i(5, 6, 7)});
    mHarness.addInputVolumes<openvdb::Vec3d>({"vec_double_test"}, {openvdb::Vec3d(0.3, 0.4, 0.5)});

    mHarness.executeCode("test/snippets/declare/declareNewVectorAttributes", nullptr, nullptr, true);

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testVectorAttributes()
{
    mHarness.addAttribute<openvdb::Vec3d>("vec_double_test", openvdb::Vec3d(0.3, 0.4, 0.5));
    mHarness.addAttributes<openvdb::Vec3f>({"vec_float_test", "vec_float_test2"},
        {openvdb::Vec3f::zero(), openvdb::Vec3f(0.2f, 0.3f, 0.4f)});
    mHarness.addAttributes<openvdb::Vec3i>({"vec_int_test", "vec_int_test2"},
        {openvdb::Vec3i::zero(), openvdb::Vec3i(5, 6, 7)});

    mHarness.executeCode("test/snippets/declare/declareVectorAttributes");

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testVectorAttributeImplicit()
{
    mHarness.addAttribute<openvdb::Vec3d>("vec_double_test", openvdb::Vec3d(1.0, 0.3, 0.4));
    mHarness.executeCode("test/snippets/declare/declareVectorAttributeImplicit");

    AXTESTS_STANDARD_ASSERT();
}

void
TestDeclare::testAmbiguousScalarAttributes()
{
    CPPUNIT_ASSERT_THROW(mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarAttributes"),
        openvdb::AXCompilerError);
}

void
TestDeclare::testAmbiguousVectorAttributes()
{
    CPPUNIT_ASSERT_THROW(mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarAttributes"),
        openvdb::AXCompilerError);
}

void
TestDeclare::testAmbiguousScalarExternals()
{
    CPPUNIT_ASSERT_THROW(mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarExternals"),
        openvdb::AXCompilerError);
}

void
TestDeclare::testAmbiguousVectorExternals()
{
    CPPUNIT_ASSERT_THROW(mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarExternals"),
        openvdb::AXCompilerError);
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
