// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>

#include "util.h"

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>

#include <gtest/gtest.h>

class TestFunctionRegistry : public ::testing::Test
{
};

TEST_F(TestFunctionRegistry, testCreateAllVerify)
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
        ASSERT_TRUE(result.empty()) << result;
    }
    catch (...) {
        std::cerr.rdbuf(sbuf);
        throw;
    }

    std::cerr.rdbuf(sbuf);
}

