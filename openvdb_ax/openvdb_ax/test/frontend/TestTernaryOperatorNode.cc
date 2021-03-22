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
    { "true ? 1 : 0;",                                   Node::Ptr(new TernaryOperator(new Value<bool>(true), new Value<int32_t>(1), new Value<int32_t>(0)))},
    { "true ? a : 1.5f;",                                Node::Ptr(new TernaryOperator(new Value<bool>(true), new Local("a"), new Value<float>(1.5f)))},
    { "false ? true : false;",                           Node::Ptr(new TernaryOperator(new Value<bool>(false), new Value<bool>(true), new Value<bool>(false)))},
    { "a == b ? 1 : 0;",                                 Node::Ptr(new TernaryOperator(
                                                            new BinaryOperator(
                                                                new Local("a"),
                                                                new Local("b"),
                                                                OperatorToken::EQUALSEQUALS),
                                                            new Value<int32_t>(1),
                                                            new Value<int32_t>(0)))},
    { "a++ ? 1 : 0;",                                   Node::Ptr(new TernaryOperator(
                                                            new Crement(new Local("a"), Crement::Operation::Increment, true),
                                                            new Value<int32_t>(1),
                                                            new Value<int32_t>(0)))},
    { "@a ? 1 : 0;",                                    Node::Ptr(new TernaryOperator(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(1), new Value<int32_t>(0)))},
    { "func() ? 1 : 0;",                                Node::Ptr(new TernaryOperator(new FunctionCall("func"), new Value<int32_t>(1), new Value<int32_t>(0)))},
    { "(true) ? 1 : 0;",                                Node::Ptr(new TernaryOperator(new Value<bool>(true), new Value<int32_t>(1), new Value<int32_t>(0)))},
    { "true ? 3 : 2 ? 1 : 0;",                          Node::Ptr(new TernaryOperator(
                                                            new Value<bool>(true),
                                                            new Value<int32_t>(3),
                                                            new TernaryOperator(new Value<int32_t>(2), new Value<int32_t>(1), new Value<int32_t>(0))))},
    { "(true ? 3 : 2) ? 1 : 0;",                        Node::Ptr(new TernaryOperator(
                                                            new TernaryOperator(new Value<bool>(true), new Value<int32_t>(3), new Value<int32_t>(2)),
                                                            new Value<int32_t>(1),
                                                            new Value<int32_t>(0)))},
    { "true ? \"foo\" : \"bar\";",                          Node::Ptr(new TernaryOperator(new Value<bool>(true), new Value<std::string>("foo"), new Value<std::string>("bar")))},
    { "true ? voidfunc1() : voidfunc2();",              Node::Ptr(new TernaryOperator(new Value<bool>(true), new FunctionCall("voidfunc1"), new FunctionCall("voidfunc2")))},
    { "true ? {1,1,1} : {0,0,0};",                      Node::Ptr(new TernaryOperator(
                                                            new Value<bool>(true),
                                                            new ArrayPack({
                                                                new Value<int32_t>(1),
                                                                new Value<int32_t>(1),
                                                                new Value<int32_t>(1)
                                                            })
                                                            ,
                                                            new ArrayPack({
                                                                new Value<int32_t>(0),
                                                                new Value<int32_t>(0),
                                                                new Value<int32_t>(0)
                                                            })
                                                            ))},
    { "true ? false ? 3 : 2 : 1;" ,                     Node::Ptr(new TernaryOperator(
                                                            new Value<bool>(true),
                                                            new TernaryOperator(
                                                                new Value<bool>(false),
                                                                new Value<int32_t>(3),
                                                                new Value<int32_t>(2)),
                                                            new Value<int32_t>(1)))},
    { "true ? false ? 3 : 2 : (true ? 4 : 5);" ,        Node::Ptr(new TernaryOperator(
                                                            new Value<bool>(true),
                                                            new TernaryOperator(
                                                                new Value<bool>(false),
                                                                new Value<int32_t>(3),
                                                                new Value<int32_t>(2)),
                                                            new TernaryOperator(
                                                                new Value<bool>(true),
                                                                new Value<int32_t>(4),
                                                                new Value<int32_t>(5))))},
    { "true ? : 0;",                                   Node::Ptr(new TernaryOperator(new Value<bool>(true), nullptr, new Value<int32_t>(0)))},
};

}

class TestTernaryOperatorNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestTernaryOperatorNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestTernaryOperatorNode);

void TestTernaryOperatorNode::testASTNode()
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
            Node::TernaryOperatorNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Ternary Operator code", code) + os.str());
        }
    }
}

