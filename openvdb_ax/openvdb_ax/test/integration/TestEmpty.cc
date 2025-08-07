// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "TestHarness.h"

#include <openvdb_ax/Exceptions.h>

using namespace openvdb::points;

class TestEmpty : public unittest_util::AXTestCase
{
};

TEST_F(TestEmpty, testEmpty)
{
    unittest_util::AXTestHarness harness;
    harness.executeCode("test/snippets/empty/empty");
    AXTESTS_STANDARD_ASSERT_HARNESS(harness);
}

