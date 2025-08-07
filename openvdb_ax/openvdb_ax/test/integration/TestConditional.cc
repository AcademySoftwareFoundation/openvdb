// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

using namespace openvdb::points;

class TestConditional : public unittest_util::AXTestCase
{
};

TEST_F(TestConditional, testConditionalIfWithinElse)
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.executeCode("test/snippets/conditional/conditionalIfWithinElse");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestConditional, testConditionalSimpleStatement)
{
    mHarness.addAttribute<bool>("bool_test", true);
    mHarness.addAttribute<float>("float_test", 1.0f);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleStatement");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestConditional, testConditionalScopingStatement)
{
    mHarness.addAttribute<int32_t>("int_test", 1);
    mHarness.executeCode("test/snippets/conditional/conditionalScopingStatement");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestConditional, testConditionalSimpleElseIf)
{
    mHarness.addAttribute("bool_test", true);
    mHarness.addAttribute("int_test", 2);

    mHarness.executeCode("test/snippets/conditional/conditionalSimpleElseIf");

    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestConditional, testConditionalErrors)
{
    const bool success = mHarness.executeCode("test/snippets/conditional/conditionalErrors");
    ASSERT_TRUE(!success);
}


