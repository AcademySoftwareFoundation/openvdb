// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include "../util.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestString : public unittest_util::AXTestCase
{
public:
    void setUp() override {
        unittest_util::AXTestCase::setUp();
    }

    CPPUNIT_TEST_SUITE(TestString);
    CPPUNIT_TEST(testAssignCompound);
    CPPUNIT_TEST(testAssignFromAttributes);
    CPPUNIT_TEST(testAssignFromLocals);
    CPPUNIT_TEST(testAssignNewOverwrite);
    CPPUNIT_TEST(testBinaryConcat);
    CPPUNIT_TEST(testDeclare);
    CPPUNIT_TEST_SUITE_END();

    void testAssignCompound();
    void testAssignFromAttributes();
    void testAssignFromLocals();
    void testAssignNewOverwrite();
    void testBinaryConcat();
    void testDeclare();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestString);

void
TestString::testAssignCompound()
{
    mHarness.addAttributes<std::string>(unittest_util::nameSequence("test", 3),
        {"foo", "foobar", "aaaaaaaaaa"});
    mHarness.executeCode("test/snippets/string/assignCompound");
    AXTESTS_STANDARD_ASSERT();
}

void
TestString::testAssignFromAttributes()
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

void
TestString::testAssignFromLocals()
{
    mHarness.addAttributes<std::string>(unittest_util::nameSequence("string_test", 4),
        {"test", "test", "new string size", ""});
    mHarness.executeCode("test/snippets/string/assignFromLocals");
    AXTESTS_STANDARD_ASSERT();
}

void
TestString::testAssignNewOverwrite()
{
    mHarness.addExpectedAttributes<std::string>({"string_test1", "string_test2"},
        {"next_value", "new_value"});
    // Volume data needs to exist
    mHarness.addInputVolumes<std::string>({"string_test1", "string_test2"},
        {"next_value", "new_value"});

    mHarness.executeCode("test/snippets/string/assignNewOverwrite", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

void
TestString::testBinaryConcat()
{
    mHarness.addExpectedAttributes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"test new value", "test new value", "test new value", "test new value", "", "test new value"});
    // Volume data needs to exist
    mHarness.addInputVolumes<std::string>(unittest_util::nameSequence("string_test", 6),
        {"test new value", "test new value", "test new value", "test new value", "", "test new value"});

    mHarness.executeCode("test/snippets/string/binaryConcat", nullptr, true);
    AXTESTS_STANDARD_ASSERT();
}

void
TestString::testDeclare()
{
    mHarness.addAttribute<std::string>("string_test", "test");
    mHarness.executeCode("test/snippets/string/declare");
    AXTESTS_STANDARD_ASSERT();
}

