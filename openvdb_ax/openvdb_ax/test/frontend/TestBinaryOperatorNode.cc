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
    { "a + b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "a - b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::MINUS
                                    )
                                )
    },
    { "a * b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::MULTIPLY
                                    )
                                )
    },
    { "a / b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::DIVIDE
                                    )
                                )
    },
    { "a % b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::MODULO
                                    )
                                )
    },
    { "a << b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::SHIFTLEFT
                                    )
                                )
    },
    { "a >> b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::SHIFTRIGHT
                                    )
                                )
    },
    { "a & b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::BITAND
                                    )
                                )
    },
    { "a | b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::BITOR
                                    )
                                )
    },
    { "a ^ b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::BITXOR
                                    )
                                )
    },
    { "a && b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::AND
                                    )
                                )
    },
    { "a || b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::OR
                                    )
                                )
    },
    { "a == b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::EQUALSEQUALS
                                    )
                                )
    },
    { "a != b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::NOTEQUALS
                                    )
                                )
    },
    { "a > b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::MORETHAN
                                    )
                                )
    },
    { "a < b;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::LESSTHAN
                                    )
                                )
    },
    { "a >= b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::MORETHANOREQUAL
                                    )
                                )
    },
    { "a <= b;",                Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("b"),
                                        OperatorToken::LESSTHANOREQUAL
                                    )
                                )
    },
    { "(a) + (a);",             Node::Ptr(
                                    new BinaryOperator(
                                        new Local("a"),
                                        new Local("a"),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "(a,b,c) + (d,e,f);",     Node::Ptr(
                                    new BinaryOperator(
                                        new CommaOperator({
                                            new Local("a"), new Local("b"), new Local("c")
                                        }),
                                        new CommaOperator({
                                            new Local("d"), new Local("e"), new Local("f")
                                        }),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "func1() + func2();",      Node::Ptr(
                                    new BinaryOperator(
                                        new FunctionCall("func1"),
                                        new FunctionCall("func2"),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "a + b - c;",             Node::Ptr(
                                    new BinaryOperator(
                                        new BinaryOperator(
                                            new Local("a"),
                                            new Local("b"),
                                            OperatorToken::PLUS
                                        ),
                                        new Local("c"),
                                        OperatorToken::MINUS
                                    )
                                )
    },
    { "~a + !b;",               Node::Ptr(
                                    new BinaryOperator(
                                        new UnaryOperator(new Local("a"), OperatorToken::BITNOT),
                                        new UnaryOperator(new Local("b"), OperatorToken::NOT),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "++a - --b;",             Node::Ptr(
                                    new BinaryOperator(
                                        new Crement(new Local("a"), Crement::Operation::Increment, false),
                                        new Crement(new Local("b"), Crement::Operation::Decrement, false),
                                        OperatorToken::MINUS
                                    )
                                )
    },
    { "a-- + b++;",             Node::Ptr(
                                    new BinaryOperator(
                                        new Crement(new Local("a"), Crement::Operation::Decrement, true),
                                        new Crement(new Local("b"), Crement::Operation::Increment, true),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "int(a) + float(b);",     Node::Ptr(
                                    new BinaryOperator(
                                        new Cast(new Local("a"), CoreType::INT32),
                                        new Cast(new Local("b"), CoreType::FLOAT),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "{a,b,c} + {d,e,f};",     Node::Ptr(
                                    new BinaryOperator(
                                        new ArrayPack({
                                            new Local("a"),
                                            new Local("b"),
                                            new Local("c")
                                        }),
                                        new ArrayPack({
                                            new Local("d"),
                                            new Local("e"),
                                            new Local("f")
                                        }),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "a.x + b.y;",             Node::Ptr(
                                    new BinaryOperator(
                                        new ArrayUnpack(new Local("a"), new Value<int32_t>(0)),
                                        new ArrayUnpack(new Local("b"), new Value<int32_t>(1)),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "0 + 1;",                 Node::Ptr(
                                    new BinaryOperator(
                                        new Value<int32_t>(0),
                                        new Value<int32_t>(1),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "0.0f + 1.0;",            Node::Ptr(
                                    new BinaryOperator(
                                        new Value<float>(0.0),
                                        new Value<double>(1.0),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "@a + @b;",               Node::Ptr(
                                    new BinaryOperator(
                                        new Attribute("a", CoreType::FLOAT, true),
                                        new Attribute("b", CoreType::FLOAT, true),
                                        OperatorToken::PLUS
                                    )
                                )
    },
    { "\"a\" + \"b\";",         Node::Ptr(
                                    new BinaryOperator(
                                        new Value<std::string>("a"),
                                        new Value<std::string>("b"),
                                        OperatorToken::PLUS
                                    )
                                )
    },
};

}

class TestBinaryOperatorNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestBinaryOperatorNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBinaryOperatorNode);

void TestBinaryOperatorNode::testASTNode()
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
            Node::BinaryOperatorNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Binary Operator code", code) + os.str());
        }
    }
}

