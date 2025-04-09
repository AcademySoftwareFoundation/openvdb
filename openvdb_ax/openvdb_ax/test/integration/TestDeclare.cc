// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

#include "../util.h"

#include <openvdb/Exceptions.h>

using namespace openvdb::points;

class TestDeclare : public unittest_util::AXTestCase
{
};

TEST_F(TestDeclare, testLocalVariables)
{
    mHarness.executeCode("test/snippets/declare/declareLocalVariables");

    // input data should not have changed
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestDeclare, testLocalVectorVariables)
{
    mHarness.executeCode("test/snippets/declare/declareLocalVectorVariables");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestDeclare, testAttributes)
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

TEST_F(TestDeclare, testAttributesVolume)
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

TEST_F(TestDeclare, testNewAttributes)
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

TEST_F(TestDeclare, testNewVectorAttributes)
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

TEST_F(TestDeclare, testVectorAttributes)
{
    mHarness.addAttribute<openvdb::Vec3d>("vec_double_test", openvdb::Vec3d(0.3, 0.4, 0.5));
    mHarness.addAttributes<openvdb::Vec3f>({"vec_float_test", "vec_float_test2"},
        {openvdb::Vec3f::zero(), openvdb::Vec3f(0.2f, 0.3f, 0.4f)});
    mHarness.addAttributes<openvdb::Vec3i>({"vec_int_test", "vec_int_test2"},
        {openvdb::Vec3i::zero(), openvdb::Vec3i(5, 6, 7)});

    mHarness.executeCode("test/snippets/declare/declareVectorAttributes");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestDeclare, testVectorAttributeImplicit)
{
    mHarness.addAttribute<openvdb::Vec3d>("vec_double_test", openvdb::Vec3d(1.0, 0.3, 0.4));
    mHarness.executeCode("test/snippets/declare/declareVectorAttributeImplicit");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestDeclare, testAmbiguousScalarAttributes)
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarAttributes");
    ASSERT_TRUE(!success);
}

TEST_F(TestDeclare, testAmbiguousVectorAttributes)
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousVectorAttributes");
    ASSERT_TRUE(!success);
}

TEST_F(TestDeclare, testAmbiguousScalarExternals)
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousScalarExternals");
    ASSERT_TRUE(!success);
}

TEST_F(TestDeclare, testAmbiguousVectorExternals)
{
    const bool success = mHarness.executeCode("test/snippets/declare/declareAmbiguousVectorExternals");
    ASSERT_TRUE(!success);
}

