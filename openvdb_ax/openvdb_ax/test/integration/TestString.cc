// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

#include "../util.h"

using namespace openvdb::points;

class TestString : public unittest_util::AXTestCase
{
};

TEST_F(TestString, testAssignCompound)
{
    mHarness.addAttributes<std::string>(unittest_util::nameSequence("test", 3),
        {"foo", "foobar", "aaaaaaaaaa"});
    mHarness.executeCode("test/snippets/string/assignCompound");
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestString, testAssignFromAttributes)
{
    mHarness.addInputPtAttributes<std::string>({"string_test1"}, {"test"});
    mHarness.addExpectedAttributes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"new value", "test", "new value", "new value", "", ""});
    // Volume data needs to exist
    mHarness.addInputVolumes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"test", "test", "new value", "new value", "", ""});

    mHarness.executeCode("test/snippets/string/assignFromAttributes", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestString, testAssignFromLocals)
{
    mHarness.addAttributes<std::string>(unittest_util::nameSequence("string_test", 4),
        {"test", "test", "new string size", ""});
    mHarness.executeCode("test/snippets/string/assignFromLocals");
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestString, testAssignNewOverwrite)
{
    mHarness.addExpectedAttributes<std::string>({"string_test1", "string_test2"},
        {"next_value", "new_value"});
    // Volume data needs to exist
    mHarness.addInputVolumes<std::string>({"string_test1", "string_test2"},
        {"next_value", "new_value"});

    mHarness.executeCode("test/snippets/string/assignNewOverwrite", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestString, testBinaryConcat)
{
    mHarness.addExpectedAttributes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"test new value", "test new value", "test new value", "test new value", "", "test new value"});
    // Volume data needs to exist
    mHarness.addInputVolumes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"test new value", "test new value", "test new value", "test new value", "", "test new value"});

    mHarness.executeCode("test/snippets/string/binaryConcat", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

TEST_F(TestString, testDeclare)
{
    mHarness.addAttribute<std::string>("string_test", "test");
    mHarness.executeCode("test/snippets/string/declare");
    AXTESTS_STANDARD_ASSERT();
}

