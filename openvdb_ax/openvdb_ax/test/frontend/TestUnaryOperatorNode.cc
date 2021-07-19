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
    { "-a;",            Node::Ptr(new UnaryOperator(new Local("a"), OperatorToken::MINUS)) },
    { "+a;",            Node::Ptr(new UnaryOperator(new Local("a"), OperatorToken::PLUS)) },
    { "!a;",            Node::Ptr(new UnaryOperator(new Local("a"), OperatorToken::NOT)) },
    { "~a;",            Node::Ptr(new UnaryOperator(new Local("a"), OperatorToken::BITNOT)) },
    { "~~a;",           Node::Ptr(new UnaryOperator(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), OperatorToken::BITNOT)) },
    { "!~a;",           Node::Ptr(new UnaryOperator(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), OperatorToken::NOT)) },
    { "+-a;",           Node::Ptr(new UnaryOperator(new UnaryOperator(new Local("a"), OperatorToken::MINUS), OperatorToken::PLUS)) },
    { "-+a;",           Node::Ptr(new UnaryOperator(new UnaryOperator(new Local("a"), OperatorToken::PLUS), OperatorToken::MINUS)) },
    { "!!!a;",          Node::Ptr(new UnaryOperator(
                            new UnaryOperator(
                                new UnaryOperator(new Local("a"), OperatorToken::NOT),
                                OperatorToken::NOT
                            ),
                            OperatorToken::NOT
                        ))
    },
    { "~~~a;",          Node::Ptr(new UnaryOperator(
                            new UnaryOperator(
                                new UnaryOperator(new Local("a"), OperatorToken::BITNOT),
                                OperatorToken::BITNOT
                            ),
                            OperatorToken::BITNOT
                        ))
    },
    { "-(a+b);",        Node::Ptr(new UnaryOperator(
                            new BinaryOperator(
                                new Local("a"), new Local("b"), OperatorToken::PLUS
                            ),
                            OperatorToken::MINUS
                        ))
    },
    { "!func();",       Node::Ptr(new UnaryOperator(new FunctionCall("func"), OperatorToken::NOT)) },
    { "-@a;",           Node::Ptr(new UnaryOperator(new Attribute("a", CoreType::FLOAT, true), OperatorToken::MINUS)) },
    { "!v@a;",          Node::Ptr(new UnaryOperator(new Attribute("a", CoreType::VEC3F), OperatorToken::NOT)) },
    { "~v@a;",          Node::Ptr(new UnaryOperator(new Attribute("a", CoreType::VEC3F), OperatorToken::BITNOT)) },
    { "+int(a);",       Node::Ptr(new UnaryOperator(new Cast(new Local("a"), CoreType::INT32), OperatorToken::PLUS)) },
    { "-(float(a));",   Node::Ptr(new UnaryOperator(new Cast(new Local("a"), CoreType::FLOAT), OperatorToken::MINUS)) },
    { "!a.x;",          Node::Ptr(new UnaryOperator(new ArrayUnpack(new Local("a"), new Value<int32_t>(0)), OperatorToken::NOT)) },
    { "-a[0];",         Node::Ptr(new UnaryOperator(new ArrayUnpack(new Local("a"), new Value<int32_t>(0)), OperatorToken::MINUS)) },
    { "-++a;",          Node::Ptr(new UnaryOperator(new Crement(new Local("a"), Crement::Operation::Increment, false), OperatorToken::MINUS)) },
    { "!{a,b,c};",      Node::Ptr(new UnaryOperator(
                            new ArrayPack({
                                new Local("a"),
                                new Local("b"),
                                new Local("c")
                            }),
                            OperatorToken::NOT
                        ))
    },
    { "!(a,b,c);",      Node::Ptr(new UnaryOperator(
                            new CommaOperator({
                                new Local("a"),
                                new Local("b"),
                                new Local("c")
                            }),
                            OperatorToken::NOT
                        ))
    },
    // This is a bit of a weird one - should perhaps look to making this a syntax error
    // (it will fail at compilation with an lvalue error)
    { "-a=a;",          Node::Ptr(new UnaryOperator(
                            new AssignExpression(new Local("a"), new Local("a")),
                            OperatorToken::MINUS
                        ))
    }
};

}

class TestUnaryOperatorNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestUnaryOperatorNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestUnaryOperatorNode);

void TestUnaryOperatorNode::testASTNode()
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
            Node::UnaryOperatorNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Unary Operator code", code) + os.str());
        }
    }
}

