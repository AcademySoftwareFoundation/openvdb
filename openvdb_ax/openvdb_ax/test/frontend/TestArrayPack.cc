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
    { "{1, 2, {1,2,3}};",               Node::Ptr(new ArrayPack({
                                            new Value<int32_t>(1),
                                            new Value<int32_t>(2),
                                            new ArrayPack({
                                                new Value<int32_t>(1),
                                                new Value<int32_t>(2),
                                                new Value<int32_t>(3)
                                            })
                                        }))
    },
    {  "{1.f,2.0,3l};" ,                Node::Ptr(new ArrayPack({
                                            new Value<float>(1.0f),
                                            new Value<double>(2.0),
                                            new Value<int64_t>(3)
                                        }))
    },
    {  "({1.f,2.0,3l});" ,              Node::Ptr(new ArrayPack({
                                            new Value<float>(1.0f),
                                            new Value<double>(2.0),
                                            new Value<int64_t>(3)
                                        }))
    },
    { "{{a,b,c}, {d,e,f}, {g,h,i}};",   Node::Ptr(new ArrayPack({
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
                                            new ArrayPack({
                                                new Local("g"),
                                                new Local("h"),
                                                new Local("i")
                                            })
                                        }))
    },
    { "{(a),b+1,-c};",                  Node::Ptr(new ArrayPack({
                                            new Local("a"),
                                            new BinaryOperator(new Local("b"), new Value<int32_t>(1), OperatorToken::PLUS),
                                            new UnaryOperator(new Local("c"), OperatorToken::MINUS)
                                        }))
    },
    { "{@x,++z,true};",                 Node::Ptr(new ArrayPack({
                                            new Attribute("x", CoreType::FLOAT, true),
                                            new Crement(new Local("z"), Crement::Operation::Increment, false),
                                            new Value<bool>(true)
                                        }))
},
    { "{@x,z++,\"bar\"};",              Node::Ptr(new ArrayPack({
                                            new Attribute("x", CoreType::FLOAT, true),
                                            new Crement(new Local("z"), Crement::Operation::Increment, true),
                                            new Value<std::string>("bar")
                                        }))
    },
    { "{float(x),b=c,c.z};",            Node::Ptr(new ArrayPack({
                                            new Cast(new Local("x"), CoreType::FLOAT),
                                            new AssignExpression(new Local("b"), new Local("c")),
                                            new ArrayUnpack(new Local("c"), new Value<int32_t>(2))
                                        }))
    },
    { "{test(),a[0],b[1,2]};",          Node::Ptr(new ArrayPack({
                                            new FunctionCall("test"),
                                            new ArrayUnpack(new Local("a"), new Value<int32_t>(0)),
                                            new ArrayUnpack(new Local("b"), new Value<int32_t>(1), new Value<int32_t>(2))
                                        }))
    },
    { "{(1), (1,2), (1,2,(3))};",       Node::Ptr(new ArrayPack({
                                            new Value<int32_t>(1),
                                            new CommaOperator({ new Value<int32_t>(1), new Value<int32_t>(2) }),
                                            new CommaOperator({ new Value<int32_t>(1), new Value<int32_t>(2), new Value<int32_t>(3)
                                            })
                                        }))
    },
    { "{1,2,3,4,5,6,7,8,9};",           Node::Ptr(new ArrayPack({
                                            new Value<int32_t>(1), new Value<int32_t>(2), new Value<int32_t>(3),
                                            new Value<int32_t>(4), new Value<int32_t>(5), new Value<int32_t>(6),
                                            new Value<int32_t>(7), new Value<int32_t>(8), new Value<int32_t>(9)
                                        }))
    },
    { "{  1, 2, 3, 4,  \
          5, 6, 7, 8,  \
          9,10,11,12,  \
         13,14,15,16 };",               Node::Ptr(new ArrayPack({
                                            new Value<int32_t>(1),  new Value<int32_t>(2),  new Value<int32_t>(3),  new Value<int32_t>(4),
                                            new Value<int32_t>(5),  new Value<int32_t>(6),  new Value<int32_t>(7),  new Value<int32_t>(8),
                                            new Value<int32_t>(9),  new Value<int32_t>(10), new Value<int32_t>(11), new Value<int32_t>(12),
                                            new Value<int32_t>(13), new Value<int32_t>(14), new Value<int32_t>(15), new Value<int32_t>(16)
                                        }))
    },
};

}

class TestArrayPack : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestArrayPack);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestArrayPack);

void TestArrayPack::testASTNode()
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
            Node::ArrayPackNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Array Pack code", code) + os.str());
        }
    }
}

