// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "TestHarness.h"

#include <openvdb_ax/Exceptions.h>

#include <cppunit/extensions/HelperMacros.h>

using namespace openvdb::points;

class TestEmpty : public unittest_util::AXTestCase
{

public:
    CPPUNIT_TEST_SUITE(TestEmpty);
    CPPUNIT_TEST(testEmpty);
    CPPUNIT_TEST_SUITE_END();
    void testEmpty();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestEmpty);

void
TestEmpty::testEmpty()
{
    unittest_util::AXTestHarness harness;
    harness.executeCode("test/snippets/empty/empty");
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
}

