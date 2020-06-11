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
    { "bool a_;",    Node::Ptr(new DeclareLocal(CoreType::BOOL, new Local("a_"))) },
    { "short _a;",   Node::Ptr(new DeclareLocal(CoreType::SHORT, new Local("_a"))) },
    { "int _;",      Node::Ptr(new DeclareLocal(CoreType::INT, new Local("_"))) },
    { "long aa;",    Node::Ptr(new DeclareLocal(CoreType::LONG, new Local("aa"))) },
    { "float A;",    Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("A"))) },
    { "double _A;",  Node::Ptr(new DeclareLocal(CoreType::DOUBLE, new Local("_A"))) },
    { "vec2i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC2I, new Local("a1"))) },
    { "vec2f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC2F, new Local("_1"))) },
    { "vec2d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC2D, new Local("abc"))) },
    { "vec3i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC3I, new Local("a1"))) },
    { "vec3f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC3F, new Local("_1"))) },
    { "vec3d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC3D, new Local("abc"))) },
    { "vec4i a1;",   Node::Ptr(new DeclareLocal(CoreType::VEC4I, new Local("a1"))) },
    { "vec4f _1;",   Node::Ptr(new DeclareLocal(CoreType::VEC4F, new Local("_1"))) },
    { "vec4d abc;",  Node::Ptr(new DeclareLocal(CoreType::VEC4D, new Local("abc"))) },
    { "mat3f _1;",   Node::Ptr(new DeclareLocal(CoreType::MAT3F, new Local("_1"))) },
    { "mat3d abc;",  Node::Ptr(new DeclareLocal(CoreType::MAT3D, new Local("abc"))) },
    { "mat4f _1;",   Node::Ptr(new DeclareLocal(CoreType::MAT4F, new Local("_1"))) },
    { "mat4d abc;",  Node::Ptr(new DeclareLocal(CoreType::MAT4D, new Local("abc"))) },
    { "string D1f;", Node::Ptr(new DeclareLocal(CoreType::STRING, new Local("D1f"))) },
    { "float a = 1.0f;",  Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"), new Value<float>(1.0f))) },
    { "float a = 1;",     Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"), new Value<int32_t>(1))) },
    { "float a = a + 1;", Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"),
                            new BinaryOperator(new Local("a"), new Value<int32_t>(1), OperatorToken::PLUS)))
    },
    { "float a = v.x;",   Node::Ptr(new DeclareLocal(CoreType::FLOAT, new Local("a"),
                            new ArrayUnpack(new Local("v"), new Value<int32_t>(0))))
    },
    { "vec3f v = {1, 2, 3};", Node::Ptr(new DeclareLocal(CoreType::VEC3F, new Local("v"),
                                new ArrayPack({
                                    new Value<int32_t>(1),
                                    new Value<int32_t>(2),
                                    new Value<int32_t>(3),
                                })))
    },
    { "mat3f m = 1;", Node::Ptr(new DeclareLocal(CoreType::MAT3F,
                            new Local("m"),
                            new Value<int32_t>(1)))
    },
    { "string s = \"foo\";", Node::Ptr(new DeclareLocal(CoreType::STRING,
                                new Local("s"),
                                new Value<std::string>("foo")))
    },
    { "float a = b = c;", Node::Ptr(new DeclareLocal(CoreType::FLOAT,
                            new Local("a"),
                            new AssignExpression(new Local("b"), new Local("c"))))
    },
};

}

class TestDeclareLocalNode : public CppUnit::TestCase
{
    public:

    CPPUNIT_TEST_SUITE(TestDeclareLocalNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestDeclareLocalNode);

void TestDeclareLocalNode::testASTNode()
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
            Node::DeclareLocalNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Declaration code", code) + os.str());
        }
    }
}


// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
