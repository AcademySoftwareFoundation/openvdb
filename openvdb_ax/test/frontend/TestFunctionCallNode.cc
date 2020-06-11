///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>
#include <openvdb_ax/ast/PrintTree.h>
#include <openvdb_ax/Exceptions.h>
#include <openvdb_ax/test/util.h>

#include <cppunit/extensions/HelperMacros.h>

#include <string>

using namespace openvdb::ax::ast;
using namespace openvdb::ax::ast::tokens;

namespace {

static const unittest_util::CodeTests tests =
{
    { "func();",            Node::Ptr(new FunctionCall("func")) },
    { "_();",               Node::Ptr(new FunctionCall("_")) },
    { "_1();",              Node::Ptr(new FunctionCall("_1")) },
    { "a_();",              Node::Ptr(new FunctionCall("a_")) },
    { "_a();",              Node::Ptr(new FunctionCall("_a")) },
    { "A();",               Node::Ptr(new FunctionCall("A")) },
    { "D1f();",             Node::Ptr(new FunctionCall("D1f")) },
    { "f(a);",              Node::Ptr(new FunctionCall("f", new Local("a"))) },
    { "a(a,1);",            Node::Ptr(new FunctionCall("a", {
                                new Local("a"),
                                new Value<int32_t>(1)
                            }))
    },
    { "func(1);",           Node::Ptr(new FunctionCall("func",
                                new Value<int32_t>(1)
                            ))
    },
    { "func(\"string\");",  Node::Ptr(new FunctionCall("func",
                                new Value<std::string>("string")
                            ))
    },
    { "func(true);",        Node::Ptr(new FunctionCall("func",
                                new Value<bool>(true)
                            ))
    },
    { "func({a,b,c});",     Node::Ptr(new FunctionCall("func",
                                new ArrayPack({
                                    new Local("a"),
                                    new Local("b"),
                                    new Local("c")
                                })
                            ))
    },
    { "func((a,b,c));",     Node::Ptr(new FunctionCall("func",
                                new CommaOperator({
                                    new Local("a"),
                                    new Local("b"),
                                    new Local("c")
                                })
                            ))
    },
    { "func(@a);",          Node::Ptr(new FunctionCall("func",
                                new Attribute("a", CoreType::FLOAT, true)
                            ))
    },
    { "func(++a);",         Node::Ptr(new FunctionCall("func",
                                new Crement(new Local("a"), Crement::Operation::Increment, false)
                            ))
    },
    { "func(~a);",          Node::Ptr(new FunctionCall("func",
                                new UnaryOperator(new Local("a"), OperatorToken::BITNOT)
                            ))
    },
    { "func((a));",         Node::Ptr(new FunctionCall("func",
                                new Local("a")
                            ))
    },
    { "func1(func2());",    Node::Ptr(new FunctionCall("func1",
                                new FunctionCall("func2")
                            ))
    },
    { "func(a=b);",         Node::Ptr(new FunctionCall("func",
                                new AssignExpression(new Local("a"), new Local("b"))
                            ))
    },
    { "func(a==b);",        Node::Ptr(new FunctionCall("func",
                                new BinaryOperator(new Local("a"), new Local("b"), OperatorToken::EQUALSEQUALS)
                            ))
    },
    { "func(a.x);",         Node::Ptr(new FunctionCall("func",
                                new ArrayUnpack(new Local("a"), new Value<int32_t>(0))
                            ))
    },
    { "func(bool(a));",     Node::Ptr(new FunctionCall("func",
                                new Cast(new Local("a"), CoreType::BOOL)
                            ))
    },
    { "func(a,b,c,d,e,f);", Node::Ptr(new FunctionCall("func", {
                                    new Local("a"), new Local("b"), new Local("c"),
                                    new Local("d"), new Local("e"), new Local("f")
                                }
                            ))
    },
    { "func((a, b), c);", Node::Ptr(new FunctionCall("func", {
                                new CommaOperator({ new Local("a"), new Local("b") }),
                                new Local("c")
                            }))
    }
};

}

class TestFunctionCallNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestFunctionCallNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFunctionCallNode);

void TestFunctionCallNode::testASTNode()
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::Ptr tree = parse(code.c_str());
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("No AST returned", code), static_cast<bool>(tree));

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        CPPUNIT_ASSERT(result);
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid AST node", code),
            Node::FunctionCallNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Function Call code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
