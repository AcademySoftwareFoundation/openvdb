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
    { "a.x;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(0))) },
    { "a.y;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(1))) },
    { "a.z;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(2))) },
    { "a.r;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(0))) },
    { "a.g;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(1))) },
    { "a.b;",             Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(2))) },
    { "x.x;",             Node::Ptr(new ArrayUnpack(new Local("x"), new Value<int32_t>(0))) },
    { "@x.x;",            Node::Ptr(new ArrayUnpack(new Attribute("x", CoreType::FLOAT, true), new Value<int32_t>(0))) },
    { "@a.x;",            Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(0))) },
    { "@b.y;",            Node::Ptr(new ArrayUnpack(new Attribute("b", CoreType::FLOAT, true), new Value<int32_t>(1))) },
    { "@c.z;",            Node::Ptr(new ArrayUnpack(new Attribute("c", CoreType::FLOAT, true), new Value<int32_t>(2))) },
    { "@a.r;",            Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(0))) },
    { "@a.g;",            Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(1))) },
    { "@a.b;",            Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(2))) },
    { "@a[0s];",          Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int16_t>(0))) },
    { "@a[0l];",          Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int64_t>(0))) },
    { "@a[0];",           Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(0))) },
    { "@a[1];",           Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(1))) },
    { "@a[2];",           Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(2))) },
    { "@a[0.0f];",        Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<float>(0.0f))) },
    { "@a[0.0];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<double>(0.0))) },
    { "@a[\"str\"];",     Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<std::string>("str"))) },
    { "@a[true];",        Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<bool>(true))) },
    { "@a[false];",       Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<bool>(false))) },
    { "@a[a];",           Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("a"))) },
    { "@a[0,0];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(0), new Value<int32_t>(0))) },
    { "@a[1,0];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(1), new Value<int32_t>(0))) },
    { "@a[2,0];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Value<int32_t>(2), new Value<int32_t>(0))) },
    { "a[0,0];",          Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(0), new Value<int32_t>(0))) },
    { "a[1,0];",          Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(1), new Value<int32_t>(0))) },
    { "a[2,0];",          Node::Ptr(new ArrayUnpack(new Local("a"), new Value<int32_t>(2), new Value<int32_t>(0))) },
    { "@a[a,0];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("a"), new Value<int32_t>(0))) },
    { "@a[b,1];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("b"), new Value<int32_t>(1))) },
    { "@a[c,2];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("c"), new Value<int32_t>(2))) },
    { "@a[a,d];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("a"), new Local("d"))) },
    { "@a[b,e];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("b"), new Local("e"))) },
    { "@a[c,f];",         Node::Ptr(new ArrayUnpack(new Attribute("a", CoreType::FLOAT, true), new Local("c"), new Local("f"))) },
    //
    { "a[(a),1+1];",      Node::Ptr(new ArrayUnpack(new Local("a"),
                             new Local("a"),
                             new BinaryOperator(new Value<int32_t>(1), new Value<int32_t>(1), OperatorToken::PLUS)))
    },
    { "a[!0,a=b];",       Node::Ptr(new ArrayUnpack(new Local("a"),
                             new UnaryOperator(new Value<int32_t>(0), OperatorToken::NOT),
                             new AssignExpression(new Local("a"), new Local("b"))))
    },
    { "a[test(),$A];",    Node::Ptr(new ArrayUnpack(new Local("a"),
                             new FunctionCall("test"),
                             new ExternalVariable("A", CoreType::FLOAT)))
    },
    { "a[a++,++a];",      Node::Ptr(new ArrayUnpack(new Local("a"),
                             new Crement(new Local("a"), Crement::Operation::Increment, true),
                             new Crement(new Local("a"), Crement::Operation::Increment, false)))
    },
    { "a[a[0,0],0];",     Node::Ptr(new ArrayUnpack(new Local("a"),
                             new ArrayUnpack(new Local("a"), new Value<int32_t>(0), new Value<int32_t>(0)),
                             new Value<int32_t>(0)))
    },
    { "a[(1,2,3)];",    Node::Ptr(new ArrayUnpack(new Local("a"),
                            new CommaOperator({
                                new Value<int32_t>(1),
                                new Value<int32_t>(2),
                                new Value<int32_t>(3)
                            })
                        ))
    },
    { "a[(1,2,3),(4,5,6)];",    Node::Ptr(new ArrayUnpack(new Local("a"),
                                    new CommaOperator({
                                        new Value<int32_t>(1),
                                        new Value<int32_t>(2),
                                        new Value<int32_t>(3),
                                    }),
                                    new CommaOperator({
                                        new Value<int32_t>(4),
                                        new Value<int32_t>(5),
                                        new Value<int32_t>(6),
                                    })
                                ))
    },
    { "a[a[0,0],a[0]];",  Node::Ptr(new ArrayUnpack(new Local("a"),
                             new ArrayUnpack(new Local("a"), new Value<int32_t>(0), new Value<int32_t>(0)),
                             new ArrayUnpack(new Local("a"), new Value<int32_t>(0))))
    }
    // @todo  should this be a syntax error
    // { "@a[{1,2,3},{1,2,3,4}];", }
};

}

class TestArrayUnpackNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestArrayUnpackNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestArrayUnpackNode);

void TestArrayUnpackNode::testASTNode()
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
            Node::ArrayUnpackNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Array Unpack code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
