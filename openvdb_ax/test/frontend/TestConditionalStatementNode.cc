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
    { "if ((a));",                                      Node::Ptr(new ConditionalStatement(new Local("a"), new Block())) },
    { "if ((a,b));",                                    Node::Ptr(new ConditionalStatement(
                                                            new CommaOperator({
                                                                new Local("a"),
                                                                new Local("b")
                                                            }),
                                                            new Block()
                                                        ))
    },
    { "if (a);",                                        Node::Ptr(new ConditionalStatement(new Local("a"), new Block())) },
    { "if(a);",                                         Node::Ptr(new ConditionalStatement(new Local("a"), new Block())) },
    { "if (@a);",                                       Node::Ptr(new ConditionalStatement(new Attribute("a", CoreType::FLOAT, true), new Block())) },
    { "if (1.0f);",                                     Node::Ptr(new ConditionalStatement(new Value<float>(1.0f), new Block())) },
    { "if (func());",                                   Node::Ptr(new ConditionalStatement(new FunctionCall("func"), new Block())) },
    { "if (a+b);",                                      Node::Ptr(new ConditionalStatement(
                                                            new BinaryOperator(
                                                                new Local("a"),
                                                                new Local("b"),
                                                                OperatorToken::PLUS
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if (-a);",                                       Node::Ptr(new ConditionalStatement(
                                                            new UnaryOperator(
                                                                new Local("a"),
                                                                OperatorToken::MINUS
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if (a = 1);",                                    Node::Ptr(new ConditionalStatement(
                                                            new AssignExpression(
                                                                new Local("a"),
                                                                new Value<int32_t>(1)
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if (a.x);",                                      Node::Ptr(new ConditionalStatement(
                                                            new ArrayUnpack(
                                                                new Local("a"),
                                                                new Value<int32_t>(0)
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if (a++);",                                      Node::Ptr(new ConditionalStatement(
                                                            new Crement(
                                                                new Local("a"),
                                                                Crement::Operation::Increment,
                                                                true
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if (float(a));",                                 Node::Ptr(new ConditionalStatement(
                                                            new Cast(
                                                                new Local("a"),
                                                                CoreType::FLOAT
                                                            ),
                                                            new Block())
                                                        )
    },
    { "if ({1.0, 2.0, 3.0});",                          Node::Ptr(new ConditionalStatement(
                                                            new ArrayPack({
                                                                new Value<double>(1.0),
                                                                new Value<double>(2.0),
                                                                new Value<double>(3.0)
                                                            }),
                                                            new Block())
                                                        )
    },
    { "if (a, b);",                                     Node::Ptr(new ConditionalStatement(
                                                                    new CommaOperator({
                                                                        new Local("a"),
                                                                        new Local("b")
                                                                    }),
                                                                    new Block())) },
    { "if (a, b, true, c = 1);",                        Node::Ptr(new ConditionalStatement(
                                                                    new CommaOperator({
                                                                        new Local("a"),
                                                                        new Local("b"),
                                                                        new Value<bool>(true),
                                                                        new AssignExpression(
                                                                            new Local("c"),
                                                                            new Value<int32_t>(1)
                                                                        ),
                                                                    }),
                                                                    new Block())) },
    { "if (a) {b;}",                                    Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(new Local("b"))))
    },
    { "if (a); else ;",                                 Node::Ptr(new ConditionalStatement(new Local("a"), new Block(), new Block())) },
    { "if (a) {} else ;",                               Node::Ptr(new ConditionalStatement(new Local("a"), new Block(), new Block())) },
    { "if (a); else {}",                                Node::Ptr(new ConditionalStatement(new Local("a"), new Block(), new Block())) },
    { "if (a); else (b);",                              Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(),
                                                            new Block(new Local("b"))))
    },
    { "if (a); else {};",                               Node::Ptr(new ConditionalStatement(new Local("a"), new Block(), new Block())) },
    { "if (a) {} else {}",                              Node::Ptr(new ConditionalStatement(new Local("a"), new Block(), new Block())) },
    { "if (a) b = 1; else {}",                          Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new AssignExpression(
                                                                    new Local("b"),
                                                                    new Value<int32_t>(1)
                                                                )
                                                            ),
                                                            new Block()))
    },

    { "if (a) {b = 1;} else {}",                        Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new AssignExpression(
                                                                    new Local("b"),
                                                                    new Value<int32_t>(1)
                                                                )
                                                            ),
                                                            new Block()))
    },
    { "if (a); else if(b) ;",                           Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block()
                                                                )
                                                            )
                                                        ))
    },
    { "if (a); else if((a,b)) ;",                       Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new CommaOperator({
                                                                        new Local("a"),
                                                                        new Local("b")
                                                                    }),
                                                                    new Block()
                                                                )
                                                            )
                                                        ))
    },
    { "if (a) if(b) ; else ;",                          Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block(),
                                                                    new Block()
                                                                )
                                                            )
                                                        ))
    },
    { "if (a) if(b) {} else {} else ;",                 Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block(),
                                                                    new Block()
                                                                )
                                                            ),
                                                            new Block()
                                                        ))
    },
    { "if (a) if(b) {if (c) ; else ;} else {} else ;",  Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block(
                                                                        new ConditionalStatement(
                                                                            new Local("c"),
                                                                            new Block(),
                                                                            new Block()
                                                                        )
                                                                    ),
                                                                    new Block()
                                                                )
                                                            ),
                                                            new Block()
                                                        ))
    },
    { "if (a) if(b) {if (c) ;} else {}",                Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block(
                                                                        new ConditionalStatement(
                                                                            new Local("c"),
                                                                            new Block()
                                                                        )
                                                                    ),
                                                                    new Block()
                                                                )
                                                            )
                                                        ))
    },
    { "if (a) {} else if(b) {if (c) ;} else {}",        Node::Ptr(new ConditionalStatement(
                                                            new Local("a"),
                                                            new Block(),
                                                            new Block(
                                                                new ConditionalStatement(
                                                                    new Local("b"),
                                                                    new Block(
                                                                        new ConditionalStatement(
                                                                            new Local("c"),
                                                                            new Block()
                                                                        )
                                                                    ),
                                                                    new Block()
                                                                )
                                                            )
                                                        ))
    },
    { "if (a) { a,a; }",                               Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new CommaOperator({ new Local("a"), new Local("a") })
                                                            )))
    },
    { "if (a) { a,a };",                               Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new ArrayPack({ new Local("a"), new Local("a") })
                                                            )))
    },
    { "if (a) { a,a }; else { a,a; }",                  Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new ArrayPack({ new Local("a"), new Local("a") })
                                                            ),
                                                            new Block(
                                                                new CommaOperator({ new Local("a"), new Local("a") })
                                                            )))
    },
    { "if (a) { a,a }; else { a,a };",                  Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new ArrayPack({ new Local("a"), new Local("a") })
                                                            ),
                                                            new Block(
                                                                new ArrayPack({ new Local("a"), new Local("a") })
                                                            )
                                                        ))
    },
    { "if (a) { { a,a; }  }",                          Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new Block(
                                                                    new CommaOperator({ new Local("a"), new Local("a") })
                                                                )
                                                            )
                                                        ))
    },
    { "if (a) { { a,a };  };",                         Node::Ptr(new ConditionalStatement(new Local("a"),
                                                            new Block(
                                                                new ArrayPack({ new Local("a"), new Local("a") })
                                                            )
                                                        ))
    }
};

}

class TestConditionalStatementNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestConditionalStatementNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestConditionalStatementNode);

void TestConditionalStatementNode::testASTNode()
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
            Node::ConditionalStatementNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Conditional Statement code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
