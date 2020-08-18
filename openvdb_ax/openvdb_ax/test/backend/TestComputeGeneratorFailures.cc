// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <iostream>

#include "util.h"
#include "../util.h"

#include "../compiler/CompilerOptions.h"
#include "../codegen/Functions.h"
#include "../codegen/FunctionRegistry.h"
#include "../codegen/ComputeGenerator.h"
#include "../ast/AST.h"
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
    "mat3f a,b; a % b;",
    "mat3d a,b; a % b;",
    "mat3f a,b; a % b;",
    "mat4d a,b; a % b;",
    "vec2i a,b; a & b;",
    "vec2f a,b; a & b;",
    "vec2d a,b; a & b;",
    "vec3i a,b; a & b;",
    "vec3d a,b; a & b;",
    "vec3f a,b; a & b;",
    "vec4i a,b; a & b;",
    "vec4f a,b; a & b;",
    "vec4d a,b; a & b;",
    "mat3f a,b; a & b;",
    "mat3d a,b; a & b;",
    "mat3f a,b; a & b;",
    "mat4d a,b; a & b;",
    "vec2i a,b; a | b;",
    "vec2f a,b; a | b;",
    "vec2d a,b; a | b;",
    "vec3i a,b; a | b;",
    "vec3d a,b; a | b;",
    "vec3f a,b; a | b;",
    "vec4i a,b; a | b;",
    "vec4f a,b; a | b;",
    "vec4d a,b; a ^ b;",
    "mat3f a,b; a ^ b;",
    "mat3d a,b; a ^ b;",
    "mat3f a,b; a ^ b;",
    "mat4d a,b; a ^ b;",
    "vec2i a,b; a ^ b;",
    "vec2f a,b; a ^ b;",
    "vec2d a,b; a ^ b;",
    "vec3i a,b; a ^ b;",
    "vec3d a,b; a ^ b;",
    "vec3f a,b; a ^ b;",
    "vec4i a,b; a ^ b;",
    "vec4f a,b; a ^ b;",
    "vec4d a,b; a ^ b;",
    "mat3f a,b; a ^ b;",
    "mat3d a,b; a ^ b;",
    "mat3f a,b; a ^ b;",
    "mat4d a,b; a ^ b;",
    "vec2i a,b; a << b;",
    "vec2f a,b; a << b;",
    "vec2d a,b; a << b;",
    "vec3i a,b; a << b;",
    "vec3d a,b; a << b;",
    "vec3f a,b; a << b;",
    "vec4i a,b; a << b;",
    "vec4f a,b; a << b;",
    "vec4d a,b; a << b;",
    "mat3f a,b; a << b;",
    "mat3d a,b; a << b;",
    "mat3f a,b; a << b;",
    "mat4d a,b; a << b;",
    "vec2i a,b; a >> b;",
    "vec2f a,b; a >> b;",
    "vec2d a,b; a >> b;",
    "vec3i a,b; a >> b;",
    "vec3d a,b; a >> b;",
    "vec3f a,b; a >> b;",
    "vec4i a,b; a >> b;",
    "vec4f a,b; a >> b;",
    "vec4d a,b; a >> b;",
    "mat3f a,b; a >> b;",
    "mat3d a,b; a >> b;",
    "mat3f a,b; a >> b;",
    "mat4d a,b; a >> b;",
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
    /// unsupported implicit casts/ops
    "vec2f a; vec3f b; a*b;",
    "vec3f a; vec4f b; a*b;",
    "vec3f a; vec2f b; a*b;",
    "vec2i a; vec3f b; a*b;",
    "mat3f a; mat4f b; a*b;",
    "string a; mat4f b; a*b;",
    "int a; string b; a*b;",
    "string a; string b; a*b;",
    "string a; string b; a-b;",
    "string a; string b; a/b;",
    "~0.0f;",
    "vec3f a; ~a;"
    /// loops
    "break;",
    "continue;",
    // ternary
    "int a = true ? print(1) : print(2);",
    "true ? print(1) : 1;",
    // "int a, b; (a ? b : 2) = 1;",
    "true ? {1,2} : {1,2,3};",
    "true ? \"foo\" : 1;",
    "true ? 1.0f : \"foo\";",
    "string a; true ? a : 1;"
    "string a; true ? 1.0f : a;"
    /// attrib access - this test only tests the base
    // generator which has no attribute access support
    "@a;"
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
    // empty reg
    openvdb::ax::codegen::FunctionRegistry reg;

    for (const auto& code : tests) {
        const openvdb::ax::ast::Tree::Ptr ast =
            openvdb::ax::ast::parse(code.c_str());
        CPPUNIT_ASSERT(ast.get());

        unittest_util::LLVMState state;
        openvdb::ax::codegen::ComputeGenerator gen(state.module(), opts, reg);
        CPPUNIT_ASSERT_THROW_MESSAGE(ERROR_MSG("Expected Compiler Error", code),
            gen.generate(*ast), openvdb::Exception);
    }
}

