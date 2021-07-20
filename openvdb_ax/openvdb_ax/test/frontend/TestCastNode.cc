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
    { "bool(a);",           Node::Ptr(new Cast(new Local("a"), CoreType::BOOL)) },
    { "int(a);",            Node::Ptr(new Cast(new Local("a"), CoreType::INT32)) },
    { "int32(a);",          Node::Ptr(new Cast(new Local("a"), CoreType::INT32)) },
    { "int64(a);",          Node::Ptr(new Cast(new Local("a"), CoreType::INT64)) },
    { "float(a);",          Node::Ptr(new Cast(new Local("a"), CoreType::FLOAT)) },
    { "double(a);",         Node::Ptr(new Cast(new Local("a"), CoreType::DOUBLE)) },
    { "int32((a));",        Node::Ptr(new Cast(new Local("a"), CoreType::INT32)) },
    { "int32(1l);",         Node::Ptr(new Cast(new Value<int64_t>(1), CoreType::INT32)) },
    { "int32(1);",          Node::Ptr(new Cast(new Value<int32_t>(1), CoreType::INT32)) },
    { "int32(0);",          Node::Ptr(new Cast(new Value<int32_t>(0), CoreType::INT32)) },
    { "int32(@a);",         Node::Ptr(new Cast(new Attribute("a", CoreType::FLOAT, true), CoreType::INT32)) },
    { "double(true);",      Node::Ptr(new Cast(new Value<bool>(true), CoreType::DOUBLE)) },
    { "double(false);",     Node::Ptr(new Cast(new Value<bool>(false), CoreType::DOUBLE)) },
    { "int32(1.0f);",       Node::Ptr(new Cast(new Value<float>(1.0f), CoreType::INT32)) },
    { "int64(1.0);",        Node::Ptr(new Cast(new Value<double>(1.0), CoreType::INT64)) },
    { "float(true);",       Node::Ptr(new Cast(new Value<bool>(true), CoreType::FLOAT)) },
    { "int32(func());",     Node::Ptr(new Cast(new FunctionCall("func"), CoreType::INT32)) },
    { "bool(a+b);",         Node::Ptr(new Cast(new BinaryOperator(new Local("a"), new Local("b"), OperatorToken::PLUS), CoreType::BOOL)) },
    { "int32(~a);",         Node::Ptr(new Cast(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), CoreType::INT32)) },
    { "int64(~a);",         Node::Ptr(new Cast(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), CoreType::INT64)) },
    { "float(a = b);",      Node::Ptr(new Cast(new AssignExpression(new Local("a"), new Local("b")), CoreType::FLOAT)) },
    { "double(a.x);",       Node::Ptr(new Cast(new ArrayUnpack(new Local("a"), new Value<int32_t>(0)), CoreType::DOUBLE)) },
    { "int32(a++);",        Node::Ptr(new Cast(new Crement(new Local("a"), Crement::Operation::Increment, true), CoreType::INT32)) },
    { "int32({a,b,c});",    Node::Ptr(new Cast(new ArrayPack({new Local("a"), new Local("b"), new Local("c")}), CoreType::INT32)) },
    { "int32((a,b,c));",    Node::Ptr(new Cast(new CommaOperator({new Local("a"), new Local("b"), new Local("c")}), CoreType::INT32)) },
    { "float(double(0));",  Node::Ptr(new Cast(new Cast(new Value<int32_t>(0), CoreType::DOUBLE), CoreType::FLOAT)) },
};

}

class TestCastNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestCastNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCastNode);

void TestCastNode::testASTNode()
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
            Node::CastNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Cast code", code) + os.str());
        }
    }
}

