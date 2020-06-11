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
    { "bool@_a;",   Node::Ptr(new Attribute("_a", CoreType::BOOL)) },
    { "short@a_;",  Node::Ptr(new Attribute("a_", CoreType::SHORT)) },
    { "i@a1;",      Node::Ptr(new Attribute("a1", CoreType::INT)) },
    { "int@abc;",   Node::Ptr(new Attribute("abc", CoreType::INT)) },
    { "long@a;",    Node::Ptr(new Attribute("a", CoreType::LONG)) },
    { "@a;",        Node::Ptr(new Attribute("a", CoreType::FLOAT, true)) },
    { "f@a;",       Node::Ptr(new Attribute("a", CoreType::FLOAT)) },
    { "float@a;",   Node::Ptr(new Attribute("a", CoreType::FLOAT)) },
    { "double@a;",  Node::Ptr(new Attribute("a", CoreType::DOUBLE)) },
    { "vec2i@a;",   Node::Ptr(new Attribute("a", CoreType::VEC2I)) },
    { "vec2f@a;",   Node::Ptr(new Attribute("a", CoreType::VEC2F)) },
    { "vec2d@a;",   Node::Ptr(new Attribute("a", CoreType::VEC2D)) },
    { "vec3i@a;",   Node::Ptr(new Attribute("a", CoreType::VEC3I)) },
    { "v@a;",       Node::Ptr(new Attribute("a", CoreType::VEC3F)) },
    { "vec3f@a;",   Node::Ptr(new Attribute("a", CoreType::VEC3F)) },
    { "vec3d@a;",   Node::Ptr(new Attribute("a", CoreType::VEC3D)) },
    { "vec4i@a;",   Node::Ptr(new Attribute("a", CoreType::VEC4I)) },
    { "vec4f@a;",   Node::Ptr(new Attribute("a", CoreType::VEC4F)) },
    { "vec4d@a;",   Node::Ptr(new Attribute("a", CoreType::VEC4D)) },
    { "mat3f@a;",   Node::Ptr(new Attribute("a", CoreType::MAT3F)) },
    { "mat3d@a;",   Node::Ptr(new Attribute("a", CoreType::MAT3D)) },
    { "mat4f@a;",   Node::Ptr(new Attribute("a", CoreType::MAT4F)) },
    { "mat4d@a;",   Node::Ptr(new Attribute("a", CoreType::MAT4D)) },
    { "string@a;",  Node::Ptr(new Attribute("a", CoreType::STRING)) },
    { "s@a;",       Node::Ptr(new Attribute("a", CoreType::STRING)) },
};

}

class TestAttributeNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestAttributeNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeNode);

void TestAttributeNode::testASTNode()
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
            Node::AttributeNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Attribute code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
