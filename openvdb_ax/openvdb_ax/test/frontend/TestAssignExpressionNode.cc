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

unittest_util::CodeTests tests =
{
    // test an attribute type passes for all expression types
    { "@a = (true);",       Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT, true), new Value<bool>(true))) },
    { "@a = (1,2,3);",      Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new CommaOperator({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3),
                                })
                            ))
    },
    { "@a = test();",       Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT, true), new FunctionCall("test"))) },
    { "@a = 1 + i@b;",      Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new BinaryOperator(new Value<int32_t>(1), new Attribute("b", CoreType::INT32), OperatorToken::PLUS)
                            ))
    },
    { "@a = -int@b;",       Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new UnaryOperator(new Attribute("b", CoreType::INT32), OperatorToken::MINUS)
                            ))
    },
    { "@a = ++float@b;",    Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Crement(new Attribute("b", CoreType::FLOAT), Crement::Operation::Increment, false)
                            ))
    },
    { "@a = bool(2);",      Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Cast(new Value<int32_t>(2), CoreType::BOOL)
                            ))
    },
    { "@a = {1, 2, 3};",    Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new ArrayPack({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3)
                                })
                            ))
    },
    { "@a = v@b.x;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new ArrayUnpack(new Attribute("b", CoreType::VEC3F), new Value<int32_t>(0))
                            ))
    },
    { "@a = \"b\";",        Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT, true), new Value<std::string>("b"))) },
    { "@a = b;",            Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT, true), new Local("b"))) },

    // test all attribute
    { "bool@a = true;",     Node::Ptr(new AssignExpression(new Attribute("a", CoreType::BOOL), new Value<bool>(true))) },
    { "int16@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::INT16), new Value<bool>(true))) },
    { "i@a = true;",        Node::Ptr(new AssignExpression(new Attribute("a", CoreType::INT32), new Value<bool>(true))) },
    { "int@a = true;",      Node::Ptr(new AssignExpression(new Attribute("a", CoreType::INT32), new Value<bool>(true))) },
    { "int32@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::INT32), new Value<bool>(true))) },
    { "int64@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::INT64), new Value<bool>(true))) },
    { "f@a = true;",        Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT), new Value<bool>(true))) },
    { "float@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::FLOAT), new Value<bool>(true))) },
    { "double@a = true;",   Node::Ptr(new AssignExpression(new Attribute("a", CoreType::DOUBLE), new Value<bool>(true))) },
    { "vec3i@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::VEC3I), new Value<bool>(true))) },
    { "v@a = true;",        Node::Ptr(new AssignExpression(new Attribute("a", CoreType::VEC3F), new Value<bool>(true))) },
    { "vec3f@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::VEC3F), new Value<bool>(true))) },
    { "vec3d@a = true;",    Node::Ptr(new AssignExpression(new Attribute("a", CoreType::VEC3D), new Value<bool>(true))) },
    { "s@a = true;",        Node::Ptr(new AssignExpression(new Attribute("a", CoreType::STRING), new Value<bool>(true))) },
    { "string@a = true;",   Node::Ptr(new AssignExpression(new Attribute("a", CoreType::STRING), new Value<bool>(true))) },

    // compound assignments (operation is stored implicitly)
    { "@a += true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::PLUS
                            ))
    },
    { "@a -= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::MINUS
                            ))
    },
    { "@a *= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::MULTIPLY
                            ))
    },
    { "@a /= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::DIVIDE
                            ))
    },
    { "@a &= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                    new Value<bool>(true),
                                    OperatorToken::BITAND
                            ))
    },
    { "@a |= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                    new Value<bool>(true),
                                    OperatorToken::BITOR
                            ))
    },
    { "@a ^= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::BITXOR
                            ))
    },
    { "@a %= true;",        Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::MODULO
                            ))
    },
    { "@a <<= true;",       Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::SHIFTLEFT
                            ))
    },
    { "@a >>= true;",       Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new Value<bool>(true),
                                OperatorToken::SHIFTRIGHT
                            ))
    },

    // test component assignment
    { "vec3i@a.x = true;",      Node::Ptr(new AssignExpression(
                                    new ArrayUnpack(
                                        new Attribute("a", CoreType::VEC3I), new Value<int32_t>(0)
                                    ),
                                    new Value<bool>(true)
                                ))
    },
    { "vec3i@a[1] = true;",     Node::Ptr(new AssignExpression(
                                    new ArrayUnpack(
                                        new Attribute("a", CoreType::VEC3I), new Value<int32_t>(1)
                                    ),
                                    new Value<bool>(true)
                                ))
    },
    { "vec3i@a.b = true;",      Node::Ptr(new AssignExpression(
                                    new ArrayUnpack(
                                        new Attribute("a", CoreType::VEC3I), new Value<int32_t>(2)
                                    ),
                                    new Value<bool>(true)
                                ))
    },
    { "vec3i@a.x += true;",     Node::Ptr(new AssignExpression(
                                    new ArrayUnpack(
                                        new Attribute("a", CoreType::VEC3I), new Value<int32_t>(0)
                                    ),
                                    new Value<bool>(true),
                                    OperatorToken::PLUS
                                ))
    },

    // test other lhs
    { "a = true;",      Node::Ptr(new AssignExpression(new Local("a"), new Value<bool>(true))) },
    { "++a = true;",    Node::Ptr(new AssignExpression(
                            new Crement(new Local("a"), Crement::Operation::Increment, false),
                            new Value<bool>(true)
                        ))
    },

    { "++@a = true;",   Node::Ptr(new AssignExpression(
                            new Crement(new Attribute("a", CoreType::FLOAT, true), Crement::Operation::Increment, false),
                            new Value<bool>(true)
                        ))
    },

    // chains
    { "@a = @b += 1;",  Node::Ptr(new AssignExpression(
                            new Attribute("a", CoreType::FLOAT, true),
                            new AssignExpression(
                                new Attribute("b", CoreType::FLOAT, true),
                                new Value<int32_t>(1),
                                OperatorToken::PLUS)
                        ))
    },
    { "@a = v@b.x = 1;",    Node::Ptr(new AssignExpression(
                                new Attribute("a", CoreType::FLOAT, true),
                                new AssignExpression(
                                    new ArrayUnpack(new Attribute("b", CoreType::VEC3F), new Value<int32_t>(0)),
                                    new Value<int32_t>(1)
                                )
                            ))
    },
    { "@a += v@b.x = x %= 1;",  Node::Ptr(new AssignExpression(
                                    new Attribute("a", CoreType::FLOAT, true),
                                    new AssignExpression(
                                        new ArrayUnpack(new Attribute("b", CoreType::VEC3F), new Value<int32_t>(0)),
                                        new AssignExpression(
                                            new Local("x"),
                                            new Value<int32_t>(1),
                                            OperatorToken::MODULO
                                        )
                                    ),
                                    OperatorToken::PLUS
                                ))
    }
};

}

class TestAssignExpressionNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestAssignExpressionNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAssignExpressionNode);

void TestAssignExpressionNode::testASTNode()
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
            Node::AssignExpressionNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Assign Expression code", code) + os.str());
        }
    }
}

