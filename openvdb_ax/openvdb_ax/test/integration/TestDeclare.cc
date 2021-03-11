// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include "../util.h"

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
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("int_test", 3),
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
    mHarness.addAttributes<int32_t>(unittest_util::nameSequence("int_test", 3),
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
    mHarness.addExpectedAttributes<int32_t>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});

    mHarness.addExpectedAttribute("short_test", int16_t(1));
    mHarness.addExpectedAttribute("long_test", int64_t(3));
    mHarness.addExpectedAttribute("double_test", 0.3);

    // Volume data needs to exist to be tested
    mHarness.addInputVolumes<float>(unittest_util::nameSequence("float_test", 4),
        {0.0f, 0.2f, 10.0f, 10.0f});
    mHarness.addInputVolumes<int32_t>(unittest_util::nameSequence("int_test", 3),
        {0, 5, 10});
    mHarness.addInputVolumes<int16_t>({"short_test"}, {int16_t(1)});
    mHarness.addInputVolumes<int64_t>({"long_test"}, {int64_t(3)});
    mHarness.addInputVolumes<double>({"double_test"}, {0.3});

    mHarness.executeCode("test/snippets/declare/declareAttributes", nullptr, true);

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

    mHarness.executeCode("test/snippets/declare/declareNewVectorAttributes", nullptr, true);

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
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarAttributes");
    CPPUNIT_ASSERT(!success);
}

void
TestDeclare::testAmbiguousVectorAttributes()
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousVectorAttributes");
    CPPUNIT_ASSERT(!success);
}

void
TestDeclare::testAmbiguousScalarExternals()
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarExternals");
    CPPUNIT_ASSERT(!success);
}

void
TestDeclare::testAmbiguousVectorExternals()
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousVectorExternals");
    CPPUNIT_ASSERT(!success);
}

