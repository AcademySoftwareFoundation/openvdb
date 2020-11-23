// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>

#include "util.h"

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class TestFunctionRegistry : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestFunctionRegistry);
    CPPUNIT_TEST(testCreateAllVerify);
    CPPUNIT_TEST_SUITE_END();

    void testCreateAllVerify();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFunctionRegistry);

void
TestFunctionRegistry::testCreateAllVerify()
{
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry();
    openvdb::ax::FunctionOptions opts;

    // check that no warnings are printed during registration
    // @todo  Replace this with a better logger once AX has one!

    std::streambuf* sbuf = std::cerr.rdbuf();

    try {
        // Redirect cerr
        std::stringstream buffer;
        std::cerr.rdbuf(buffer.rdbuf());
        reg->createAll(opts, true);
        const std::string& result = buffer.str();
        CPPUNIT_ASSERT_MESSAGE(result, result.empty());
    }
    catch (...) {
        std::cerr.rdbuf(sbuf);
        throw;
    }

    std::cerr.rdbuf(sbuf);
}

