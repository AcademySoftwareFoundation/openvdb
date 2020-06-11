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
    { "bool(a);",           Node::Ptr(new Cast(new Local("a"), CoreType::BOOL)) },
    { "short(a);",          Node::Ptr(new Cast(new Local("a"), CoreType::SHORT)) },
    { "int(a);",            Node::Ptr(new Cast(new Local("a"), CoreType::INT)) },
    { "long(a);",           Node::Ptr(new Cast(new Local("a"), CoreType::LONG)) },
    { "float(a);",          Node::Ptr(new Cast(new Local("a"), CoreType::FLOAT)) },
    { "double(a);",         Node::Ptr(new Cast(new Local("a"), CoreType::DOUBLE)) },
    { "int((a));",          Node::Ptr(new Cast(new Local("a"), CoreType::INT)) },
    { "int(1l);",           Node::Ptr(new Cast(new Value<int64_t>(1), CoreType::INT)) },
    { "int(1);",            Node::Ptr(new Cast(new Value<int32_t>(1), CoreType::INT)) },
    { "int(0);",            Node::Ptr(new Cast(new Value<int32_t>(0), CoreType::INT)) },
    { "int(@a);",           Node::Ptr(new Cast(new Attribute("a", CoreType::FLOAT, true), CoreType::INT)) },
    { "double(true);",      Node::Ptr(new Cast(new Value<bool>(true), CoreType::DOUBLE)) },
    { "double(false);",     Node::Ptr(new Cast(new Value<bool>(false), CoreType::DOUBLE)) },
    { "int(1.0f);",         Node::Ptr(new Cast(new Value<float>(1.0f), CoreType::INT)) },
    { "long(1.0);",         Node::Ptr(new Cast(new Value<double>(1.0), CoreType::LONG)) },
    { "float(true);",       Node::Ptr(new Cast(new Value<bool>(true), CoreType::FLOAT)) },
    { "double(1s);",        Node::Ptr(new Cast(new Value<int16_t>(1), CoreType::DOUBLE)) },
    { "int(func());",       Node::Ptr(new Cast(new FunctionCall("func"), CoreType::INT)) },
    { "bool(a+b);",         Node::Ptr(new Cast(new BinaryOperator(new Local("a"), new Local("b"), OperatorToken::PLUS), CoreType::BOOL)) },
    { "short(a+b);",        Node::Ptr(new Cast(new BinaryOperator(new Local("a"), new Local("b"), OperatorToken::PLUS), CoreType::SHORT)) },
    { "int(~a);",           Node::Ptr(new Cast(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), CoreType::INT)) },
    { "long(~a);",          Node::Ptr(new Cast(new UnaryOperator(new Local("a"), OperatorToken::BITNOT), CoreType::LONG)) },
    { "float(a = b);",      Node::Ptr(new Cast(new AssignExpression(new Local("a"), new Local("b")), CoreType::FLOAT)) },
    { "double(a.x);",       Node::Ptr(new Cast(new ArrayUnpack(new Local("a"), new Value<int32_t>(0)), CoreType::DOUBLE)) },
    { "int(a++);",          Node::Ptr(new Cast(new Crement(new Local("a"), Crement::Operation::Increment, true), CoreType::INT)) },
    { "int({a,b,c});",      Node::Ptr(new Cast(new ArrayPack({new Local("a"), new Local("b"), new Local("c")}), CoreType::INT)) },
    { "int((a,b,c));",      Node::Ptr(new Cast(new CommaOperator({new Local("a"), new Local("b"), new Local("c")}), CoreType::INT)) },
    { "float(double(0));",  Node::Ptr(new Cast(new Cast(new Value<int32_t>(0), CoreType::DOUBLE), CoreType::FLOAT)) },
};

}

class TestCastNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestCastNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCastNode);

void TestCastNode::testASTNode()
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
            Node::CastNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Cast code", code) + os.str());
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
