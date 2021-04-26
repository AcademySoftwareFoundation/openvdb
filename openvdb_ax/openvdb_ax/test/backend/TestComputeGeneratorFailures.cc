// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"
#include "../util.h"

#include <openvdb_ax/compiler/CompilerOptions.h>
#include <openvdb_ax/compiler/Logger.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/ComputeGenerator.h>
#include <openvdb_ax/ast/AST.h>

#include <cppunit/extensions/HelperMacros.h>

static const std::vector<std::string> tests {
    //  codegen errors

    /// implicit narrow
    "int a; vec2i b; a=b;",
    "int a; vec3i b; a=b;",
    "int a; vec4i b; a=b;",
    "float a; vec2f b; a=b;",
    "float a; vec3f b; a=b;",
    "float a; vec4f b; a=b;",
    "float a; mat3f b; a=b;",
    "float a; mat4f b; a=b;",
    "double a; vec2d b; a=b;",
    "double a; vec3d b; a=b;",
    "double a; vec4d b; a=b;",
    "double a; mat3f b; a=b;",
    "double a; mat4f b; a=b;",
    /// unsupported bin ops
    "float a; a << 1;",
    "float a; a >> 1;",
    "float a; a | 1;",
    "float a; a ^ 1;",
    "float a; a & 1;",
    "double a; a << 1;",
    "double a; a >> 1;",
    "double a; a | 1;",
    "double a; a ^ 1;",
    "double a; a & 1;",
    "mat3d a,b; a % b;",
    "mat3d a,b; a & b;",
    "mat3d a,b; a && b;",
    "mat3d a,b; a << b;",
    "mat3d a,b; a >> b;",
    "mat3d a,b; a ^ b;",
    "mat3d a,b; a || b;",
    "mat3f a,b; a % b;",
    "mat3f a,b; a & b;",
    "mat3f a,b; a && b;",
    "mat3f a,b; a << b;",
    "mat3f a,b; a >> b;",
    "mat3f a,b; a ^ b;",
    "mat3f a,b; a || b;",
    "mat4d a,b; a % b;",
    "mat4d a,b; a & b;",
    "mat4d a,b; a && b;",
    "mat4d a,b; a << b;",
    "mat4d a,b; a >> b;",
    "mat4d a,b; a ^ b;",
    "mat4d a,b; a || b;",
    "string a,b; a & b;",
    "string a,b; a && b;",
    "string a,b; a - b;",
    "string a,b; a << b;",
    "string a,b; a >> b;",
    "string a,b; a ^ b;",
    "string a,b; a | b;",
    "string a,b; a || b;",
    "string a,b; a * b;",
    "string a,b; a - b;",
    "string a,b; a / b;",
    "vec2d a,b; a & b;",
    "vec2d a,b; a && b;",
    "vec2d a,b; a << b;",
    "vec2d a,b; a >> b;",
    "vec2d a,b; a ^ b;",
    "vec2d a,b; a | b;",
    "vec2d a,b; a || b;",
    "vec2f a,b; a & b;",
    "vec2f a,b; a && b;",
    "vec2f a,b; a << b;",
    "vec2f a,b; a >> b;",
    "vec2f a,b; a ^ b;",
    "vec2f a,b; a | b;",
    "vec2f a,b; a || b;",
    "vec2i a,b; a & b;",
    "vec2i a,b; a && b;",
    "vec2i a,b; a << b;",
    "vec2i a,b; a >> b;",
    "vec2i a,b; a ^ b;",
    "vec2i a,b; a | b;",
    "vec2i a,b; a || b;",
    "vec3d a,b; a & b;",
    "vec3d a,b; a && b;",
    "vec3d a,b; a << b;",
    "vec3d a,b; a >> b;",
    "vec3d a,b; a ^ b;",
    "vec3d a,b; a | b;",
    "vec3d a,b; a || b;",
    "vec3f a,b; a & b;",
    "vec3f a,b; a && b;",
    "vec3f a,b; a << b;",
    "vec3f a,b; a >> b;",
    "vec3f a,b; a ^ b;",
    "vec3f a,b; a | b;",
    "vec3f a,b; a || b;",
    "vec3i a,b; a & b;",
    "vec3i a,b; a && b;",
    "vec3i a,b; a << b;",
    "vec3i a,b; a >> b;",
    "vec3i a,b; a ^ b;",
    "vec3i a,b; a | b;",
    "vec3i a,b; a || b;",
    "vec4d a,b; a & b;",
    "vec4d a,b; a && b;",
    "vec4d a,b; a << b;",
    "vec4d a,b; a >> b;",
    "vec4d a,b; a ^ b;",
    "vec4d a,b; a ^ b;",
    "vec4d a,b; a || b;",
    "vec4f a,b; a & b;",
    "vec4f a,b; a && b;",
    "vec4f a,b; a << b;",
    "vec4f a,b; a >> b;",
    "vec4f a,b; a ^ b;",
    "vec4f a,b; a | b;",
    "vec4f a,b; a || b;",
    "vec4i a,b; a & b;",
    "vec4i a,b; a && b;",
    "vec4i a,b; a << b;",
    "vec4i a,b; a >> b;",
    "vec4i a,b; a ^ b;",
    "vec4i a,b; a | b;",
    "vec4i a,b; a || b;",
    /// invalid unary ops
    "vec2f a; !a;",
    "vec2d a; !a;",
    "vec3d a; !a;",
    "vec3f a; !a;",
    "vec4f a; !a;",
    "vec4d a; !a;",
    "mat3f a; !a;",
    "mat3d a; !a;",
    "mat3f a; !a;",
    "mat4d a; !a;",
    "vec2f a; ~a;",
    "vec2d a; ~a;",
    "vec3d a; ~a;",
    "vec3f a; ~a;",
    "vec4f a; ~a;",
    "vec4d a; ~a;",
    "mat3f a; ~a;",
    "mat3d a; ~a;",
    "mat3f a; ~a;",
    "mat4d a; ~a;",
    /// missing function
    "nonexistent();",
    /// non/re declared
    "a;",
    "int a; int a;",
    "{ int a; int a; }",
    "int a, a;",
    "a ? b : c;",
    "a ? true : false;",
    "true ? a : c;",
    "true ? a : false;",
    "true ? true : c;",
    "a && b;",
    "a && true;",
    "true && b;",
    /// invalid crement
    "string a; ++a;",
    "vec2f a; ++a;",
    "vec3f a; ++a;",
    "vec4f a; ++a;",
    "mat3f a; ++a;",
    "mat4f a; ++a;",
    /// array size assignments
    "vec2f a; vec3f b; a=b;",
    "vec3f a; vec2f b; a=b;",
    "vec4f a; vec3f b; a=b;",
    "vec2d a; vec3d b; a=b;",
    "vec3d a; vec2d b; a=b;",
    "vec4d a; vec3d b; a=b;",
    "vec2i a; vec3i b; a=b;",
    "vec3i a; vec2i b; a=b;",
    "vec4i a; vec3i b; a=b;",
    "mat4f a; mat3f b; a=b;",
    "mat4d a; mat3d b; a=b;",
    "vec2f a = {1,2,3};",
    "vec3f a = {1,2};",
    "vec4f a = {1,2};",
    "mat3f a = {1,2};",
    "mat4f a = {1,2};",
    /// string assignments
    "string a = 1;",
    "int a; string b; b=a;",
    "float a; string b; b=a;",
    "double a; string b; b=a;",
    "vec3f a; string b; b=a;",
    "mat3f a; string b; b=a;",
    /// array index
    "int a; a[0];",
    "vec3f a; string b; a[b];",
    "vec3f a; vec3f b; a[b];",
    "vec3f a; mat3f b; a[b];",
    "vec3f a; a[1,1];",
    "mat3f a; vec3f b; a[b,1];",
    "mat3f a; vec3f b; a[1,b];",
    /// array packs
    "{1, {2,3}};",
    "{{1,2}, 3};",
    "{1, 2, \"a\"};",
    "vec2f a; {1, 2, a};",
    "vec3f a; {1, 2, a};",
    "mat3f a; {1, 2, a};",
    "mat4f a; {1, 2, a};",
    "string a; {1, 2, a};",
    /// unsupported implicit casts/ops
    "vec2f a; vec3f b; a*b;",
    "vec3f a; vec4f b; a*b;",
    "vec3f a; vec2f b; a*b;",
    "vec2i a; vec3f b; a*b;",
    "mat3f a; mat4f b; a*b;",
    "string a; mat4f b; a*b;",
    "int a; string b; a*b;",
    "{0,0,0,0,0,0,0,0,0} * 0;"
    "0 * {0,0,0,0,0,0,0,0,0};"
    "{0,0,0,0,0,0,0,0,0} + 0;"
    "{0,0,0,0,0,0,0,0,0} - 0;"
    "{0,0,0,0,0,0,0,0,0} / 0;"
    "0 * {.0f,0,0,0,0,0,0,0,0};"
    "{.0f,0,0,0,0,0,0,0,0} * 0;"
    "{0.0,0,0,0,0,0,0,0,0} * 0;"
    "{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} * 0;"
    "0 * {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};"
    "{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} + 0;"
    "{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} - 0;"
    "{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} / 0;"
    "0 * {.0f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};"
    "{.0f,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} * 0;"
    "{0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0} * 0;"
    "~0.0f;",
    "vec3f a; ~a;"
    /// loops
    "break;",
    "continue;",
    // ternary
    "int a = true ? print(1) : print(2);",
    "true ? print(1) : 1;",
    "mat4d a; a ? 0 : 1;",
    "mat4f a; a ? 0 : 1;",
    "string a; a ? 0 : 1;",
    "vec2d a; a ? 0 : 1;",
    "vec2f a; a ? 0 : 1;",
    "vec2i a; a ? 0 : 1;",
    "vec3d a; a ? 0 : 1;",
    "vec3f a; a ? 0 : 1;",
    "vec3i a; a ? 0 : 1;",
    "vec4d a; a ? 0 : 1;",
    "vec4f a; a ? 0 : 1;",
    "vec4i a; a ? 0 : 1;",
    // "int a, b; (a ? b : 2) = 1;",
    "true ? {1,2} : {1,2,3};",
    "true ? \"foo\" : 1;",
    "true ? 1.0f : \"foo\";",
    "{1,1} && 1 ? true : false;",
    "{1,1} ? true : false;",
    "{1,1} && 1 ? \"foo\" : false;",
    "\"foo\" ? true : false;",
    "true ? {1,1} && 1: {1,1};",
    "true ? {1,1} : {1,1} && 1;",
    "string a; true ? a : 1;",
    "string a; true ? 1.0f : a;",
    // conditional
    "mat4d a; if (a) 1;",
    "mat4f a; if (a) 1;",
    "string a; if (a) 1;",
    "vec2d a; if (a) 1;",
    "vec2f a; if (a) 1;",
    "vec2i a; if (a) 1;",
    "vec3d a; if (a) 1;",
    "vec3f a; if (a) 1;",
    "vec3i a; if (a) 1;",
    "vec4d a; if (a) 1;",
    "vec4f a; if (a) 1;",
    "vec4i a; if (a) 1;",
    "if ({1,1} && 1) 1;",
    "if (true) {1,1} && 1;",
    // loops
    "mat4d a; for (;a;) 1;",
    "mat4f a; for (;a;) 1;",
    "string a; for (;a;) 1;",
    "vec2d a; for (;a;) 1;",
    "vec2f a; for (;a;) 1;",
    "vec2i a; for (;a;) 1;",
    "vec3d a; for (;a;) 1;",
    "vec3f a; for (;a;) 1;",
    "vec3i a; for (;a;) 1;",
    "vec4d a; for (;a;) 1;",
    "vec4f a; for (;a;) 1;",
    "vec4i a; for (;a;) 1;",
    "mat4d a; while (a) 1;",
    "mat4f a; while (a) 1;",
    "string a; while (a) 1;",
    "vec2d a; while (a) 1;",
    "vec2f a; while (a) 1;",
    "vec2i a; while (a) 1;",
    "vec3d a; while (a) 1;",
    "vec3f a; while (a) 1;",
    "vec3i a; while (a) 1;",
    "vec4d a; while (a) 1;",
    "vec4f a; while (a) 1;",
    "vec4i a; while (a) 1;",
    "mat4d a; do { 1; } while(a);",
    "mat4f a; do { 1; } while(a);",
    "string a; do { 1; } while(a);",
    "vec2d a; do { 1; } while(a);",
    "vec2f a; do { 1; } while(a);",
    "vec2i a; do { 1; } while(a);",
    "vec3d a; do { 1; } while(a);",
    "vec3f a; do { 1; } while(a);",
    "vec3i a; do { 1; } while(a);",
    "vec4d a; do { 1; } while(a);",
    "vec4f a; do { 1; } while(a);",
    "vec4i a; do { 1; } while(a);",
    // comma
    "vec2i v; v++, 1;",
    "vec2i v; 1, v++;"
};

class TestComputeGeneratorFailures : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestComputeGeneratorFailures);
    CPPUNIT_TEST(testFailures);
    CPPUNIT_TEST_SUITE_END();

    void testFailures();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestComputeGeneratorFailures);


void
TestComputeGeneratorFailures::testFailures()
{
    openvdb::ax::FunctionOptions opts;
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // create logger that suppresses all messages, but still logs number of errors/warnings
    openvdb::ax::Logger logger([](const std::string&) {});
    logger.setMaxErrors(1);

    for (const auto& code : tests) {
        const openvdb::ax::ast::Tree::ConstPtr ast =
            openvdb::ax::ast::parse(code.c_str(), logger);
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Unable to parse", code), ast.get());
        CPPUNIT_ASSERT(!logger.hasError());

        unittest_util::LLVMState state;
        openvdb::ax::codegen::codegen_internal::ComputeGenerator gen(state.module(), opts, *reg, logger);
        gen.generate(*ast);

        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Expected Compiler Error", code), logger.hasError());
        logger.clear();
    }
}

