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
    { "for (int i = 0; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                            new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                            new Block(),
                                            new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                            new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for(int i = 0; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                            new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                            new Block(),
                                            new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                            new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (int i = 0;i < 10;++i) ;", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                        new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                        new Block(),
                                        new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                        new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (i; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new Local("i"),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (@i; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new Attribute("i", CoreType::FLOAT, true),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (!i; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new UnaryOperator(new Local("i"), OperatorToken::NOT),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (i = 0; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new AssignExpression(new Local("i"), new Value<int32_t>(0)),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (i+j; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new BinaryOperator(new Local("i"), new Local("j"), OperatorToken::PLUS),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (func(i); i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new FunctionCall("func", new Local("i")),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (1; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new Value<int32_t>(1),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (float$ext; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new ExternalVariable("ext", CoreType::FLOAT),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (i++; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new Crement(new Local("i"), Crement::Operation::Increment, true),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for ({1,2.0,3.0f}; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new ArrayPack({new Value<int32_t>(1), new Value<double>(2.0), new Value<float>(3.0f)}),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (1,2.0,3.0f; (i < 10, i > 10); (++i, --i)) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new CommaOperator({
                                        new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                        new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::MORETHAN)
                                    }),
                                    new Block(),
                                    new CommaOperator({
                                        new Value<int32_t>(1), new Value<double>(2.0), new Value<float>(3.0f)
                                    }),
                                    new CommaOperator({
                                        new Crement(new Local("i"), Crement::Operation::Increment, false),
                                        new Crement(new Local("i"), Crement::Operation::Decrement, false),
                                    })
                                ))
    },
    { "for (++i; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (x[2]; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new ArrayUnpack(new Local("x"), new Value<int32_t>(2)),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for ((x[2]); i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    new ArrayUnpack(new Local("x"), new Value<int32_t>(2)),
                                    new Crement(new Local("i"), Crement::Operation::Increment, false)))
    },
    { "for (; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                    new Block(),
                                    nullptr,
                                    new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (int i = 0; i < 10; ++i, ++j) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                                new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                                new Block(),
                                                new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                                new CommaOperator({
                                                    new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false),
                                                    new Crement(new Local("j"), Crement::Operation::Increment, /*post*/false)
                                                })))
    },
    { "for (i = 0; i < 10; ++i, ++j) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                            new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                            new Block(),
                                            new AssignExpression(new Local("i"), new Value<int32_t>(0)),
                                            new CommaOperator({
                                                new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false),
                                                new Crement(new Local("j"), Crement::Operation::Increment, /*post*/false)
                                            })))
    },
    { "for (int i = 0; i; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                        new Local("i"),
                                        new Block(),
                                        new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                        new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (int i = 0; func(i); ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                            new FunctionCall("func", new Local("i")),
                                            new Block(),
                                            new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                            new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (int i = 0; int j = func(i); ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                            new DeclareLocal(CoreType::INT, new Local("j"),new FunctionCall("func", new Local("i"))),
                                            new Block(),
                                            new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                            new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (; i < 10;) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                new Block()))
    },
    { "for (;;) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                        new Value<bool>(true),
                        new Block()))
    },
    { "for (;;) { 1,2,3 };", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                new Value<bool>(true),
                                new Block(new ArrayPack({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3)
                                }))))
    },
    { "for (;;) { 1,2,3; }", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                new Value<bool>(true),
                                new Block(new CommaOperator({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3)
                                }))))
    },
    { "for (int i = 0, j = 0, k; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                                    new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                                    new Block(),
                                                    new StatementList({new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                                                       new DeclareLocal(CoreType::INT, new Local("j"), new Value<int32_t>(0)),
                                                                       new DeclareLocal( CoreType::INT, new Local("k"))}),
                                                    new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (i = 0, j = 0; i < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                                new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                                new Block(),
                                                new CommaOperator({
                                                    new AssignExpression(new Local("i"), new Value<int32_t>(0)),
                                                    new AssignExpression(new Local("j"), new Value<int32_t>(0))
                                                }),
                                                new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "for (int i = 0; i < 10, j < 10; ++i) {}", Node::Ptr(new Loop(tokens::LoopToken::FOR,
                                                    new CommaOperator({
                                                        new BinaryOperator(new Local("i"), new Value<int32_t>(10), OperatorToken::LESSTHAN),
                                                        new BinaryOperator(new Local("j"), new Value<int32_t>(10), OperatorToken::LESSTHAN)
                                                    }),
                                                    new Block(),
                                                    new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                                    new Crement(new Local("i"), Crement::Operation::Increment, /*post*/false)))
    },
    { "while (int i = 0) {}", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                                new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                new Block()))
    },
    { "while (i = 0) {}", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                            new AssignExpression(new Local("i"), new Value<int32_t>(0)),
                            new Block()))
    },
    { "while ((a,b,c)) {}", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                                new CommaOperator({
                                    new Local("a"),
                                    new Local("b"),
                                    new Local("c")
                                }),
                                new Block()))
    },
    { "while (i < 0, j = 10) ;", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                                    new CommaOperator({
                                        new BinaryOperator(new Local("i"), new Value<int32_t>(0), OperatorToken::LESSTHAN),
                                        new AssignExpression(new Local("j"), new Value<int32_t>(10))
                                    }),
                                    new Block()))
    },
    { "while (i) { 1,2,3 };", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                                    new Local("i"),
                                        new Block(new ArrayPack({
                                            new Value<int32_t>(1),
                                            new Value<int32_t>(2),
                                            new Value<int32_t>(3)
                                        }))))
    },
    { "while (i) { 1,2,3; }", Node::Ptr(new Loop(tokens::LoopToken::WHILE,
                                    new Local("i"),
                                        new Block(new CommaOperator({
                                            new Value<int32_t>(1),
                                            new Value<int32_t>(2),
                                            new Value<int32_t>(3)
                                        }))))
    },
    { "do {} while (i < 0, j = 10)", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                        new CommaOperator({
                                            new BinaryOperator(new Local("i"), new Value<int32_t>(0), OperatorToken::LESSTHAN),
                                            new AssignExpression(new Local("j"), new Value<int32_t>(10))
                                        }),
                                        new Block()))
    },
    { "do ; while (int i = 0)", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                    new DeclareLocal(CoreType::INT, new Local("i"), new Value<int32_t>(0)),
                                    new Block()))
    },
    { "do ; while ((a,b,c))", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                       new CommaOperator({
                                           new Local("a"),
                                           new Local("b"),
                                           new Local("c")
                                       }),
                                       new Block()))
    },
    { "do ; while (a,b,c)", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                       new CommaOperator({
                                           new Local("a"),
                                           new Local("b"),
                                           new Local("c")
                                       }),
                                       new Block()))
    },
    { "do { 1,2,3 }; while (i) ", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                    new Local("i"),
                                        new Block(new ArrayPack({
                                            new Value<int32_t>(1),
                                            new Value<int32_t>(2),
                                            new Value<int32_t>(3)
                                        }))))
    },
    { "do { 1,2,3; } while (i) ", Node::Ptr(new Loop(tokens::LoopToken::DO,
                                    new Local("i"),
                                        new Block(new CommaOperator({
                                            new Value<int32_t>(1),
                                            new Value<int32_t>(2),
                                            new Value<int32_t>(3)
                                        }))))
    }
};

}

class TestLoopNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestLoopNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLoopNode);

void TestLoopNode::testASTNode()
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
            Node::LoopNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Loop code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
