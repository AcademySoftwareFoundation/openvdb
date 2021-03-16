// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>
#include <openvdb_ax/ast/PrintTree.h>
#include <openvdb_ax/Exceptions.h>

#include "../util.h"

#include <cppunit/extensions/HelperMacros.h>

#include <string>

using namespace openvdb::ax::ast;
using namespace openvdb::ax::ast::tokens;

namespace {

static const unittest_util::CodeTests tests =
{
    { "bool a_;",    Node::Ptr(new DeclareLocal(CoreType::BOOL, new Local("a_"))) },
    { "int32 _;",    Node::Ptr(new DeclareLocal(CoreType::INT32, new Local("_"))) },
    { "int64 aa;",   Node::Ptr(new DeclareLocal(CoreType::INT64, new Local("aa"))) },
    { "float A;",    Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("A"))) },
    { "double _A;",  Node::Ptr(new DeclareLocal(CoreType::DOUBLE, new Local("_A"))) },
    { "vec2i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC2I, new Local("a1"))) },
    { "vec2f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC2F, new Local("_1"))) },
    { "vec2d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC2D, new Local("abc"))) },
    { "vec3i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC3I, new Local("a1"))) },
    { "vec3f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC3F, new Local("_1"))) },
    { "vec3d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC3D, new Local("abc"))) },
    { "vec4i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC4I, new Local("a1"))) },
    { "vec4f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC4F, new Local("_1"))) },
    { "vec4d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC4D, new Local("abc"))) },
    { "mat3f _1;",   Node::Ptr(new DeclareLocal(CoreType::MAT3F, new Local("_1"))) },
    { "mat3d abc;",  Node::Ptr(new DeclareLocal(CoreType::MAT3D, new Local("abc"))) },
    { "mat4f _1;",   Node::Ptr(new DeclareLocal(CoreType::MAT4F, new Local("_1"))) },
    { "mat4d abc;",  Node::Ptr(new DeclareLocal(CoreType::MAT4D, new Local("abc"))) },
    { "string D1f;", Node::Ptr(new DeclareLocal(CoreType::STRING, new Local("D1f"))) },
    { "float a = 1.0f;",  Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"), new Value<float>(1.0f))) },
    { "float a = 1;",     Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"), new Value<int32_t>(1))) },
    { "float a = a + 1;", Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"),
                            new BinaryOperator(new Local("a"), new Value<int32_t>(1), OperatorToken::PLUS)))
    },
    { "float a = v.x;",   Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"),
                            new ArrayUnpack(new Local("v"), new Value<int32_t>(0))))
    },
    { "vec3f v = {1, 2, 3};", Node::Ptr(new DeclareLocal(CoreType::VEC3F, new Local("v"),
                                new ArrayPack({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3),
                                })))
    },
    { "mat3f m = 1;", Node::Ptr(new DeclareLocal(CoreType::MAT3F,
                            new Local("m"),
                            new Value<int32_t>(1)))
    },
    { "string s = \"foo\";", Node::Ptr(new DeclareLocal(CoreType::STRING,
                                new Local("s"),
                                new Value<std::string>("foo")))
    },
    { "float a = b = c;", Node::Ptr(new DeclareLocal(CoreType::FLOAT,
                            new Local("a"),
                            new AssignExpression(new Local("b"), new Local("c"))))
    },
};

}

class TestDeclareLocalNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestDeclareLocalNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDeclareLocalNode);

void TestDeclareLocalNode::testASTNode()
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("No AST returned", code), static_cast<bool>(tree));

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid AST node", code),
            Node::DeclareLocalNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Declaration code", code) + os.str());
        }
    }
}


