// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestConditional : public unittest_util::AXTestCase
{
public:
    CPPUNIT_TEST_SUITE(TestConditional);
    CPPUNIT_TEST(testConditionalIfWithinElse);
    CPPUNIT_TEST(testConditionalScopingStatement);
    CPPUNIT_TEST(testConditionalSimpleStatement);
    CPPUNIT_TEST(testConditionalSimpleElseIf);
    CPPUNIT_TEST(testConditionalErrors);
    CPPUNIT_TEST_SUITE_END();

    void testConditionalIfWithinElse();
    void testConditionalSimpleStatement();
    void testConditionalScopingStatement();
    void testConditionalSimpleElseIf();
    void testConditionalErrors();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestConditional);

void
TestConditional::testConditionalIfWithinElse()
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.executeCode("test/snippets/conditional/conditionalIfWithinElse");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalSimpleStatement()
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.addAttribute<float>("float_test", 1.0f);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleStatement");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalScopingStatement()
{
    mHarness.addAttribute<int32_t>("int_test", 1);
    mHarness.executeCode("test/snippets/conditional/conditionalScopingStatement");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalSimpleElseIf()
{
    mHarness.addAttribute("bool_test", true);
    mHarness.addAttribute("int_test", 2);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleElseIf");

    AXTESTS_STANDARD_ASSERT();
}

void
TestConditional::testConditionalErrors()
{
    const bool success = mHarness.executeCode("test/snippets/conditional/conditionalErrors");
    CPPUNIT_ASSERT(!success);
}


