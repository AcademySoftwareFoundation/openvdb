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

using CodeTestMap = std::map<Node::NodeType, unittest_util::CodeTests>;

static const CodeTestMap value_tests =
{
    // No limits::lowest, negative values are a unary operator

    {
        Node::NodeType::ValueBoolNode,
        {
            { "false;",  Node::Ptr(new Value<bool>(false)) },
            { "true;",   Node::Ptr(new Value<bool>(true)) },
        }
    },

    {
        Node::NodeType::ValueInt16Node,
        {
            { "1234567890s;",  Node::Ptr(new Value<int16_t>("1234567890")) }, // signed int wrap
            { "00s;",          Node::Ptr(new Value<int16_t>("0")) },
            { "0s;",           Node::Ptr(new Value<int16_t>("0")) },
            // signed int wrap
            { std::to_string(std::numeric_limits<int16_t>::max()) + "0s;",
                Node::Ptr(new Value<int16_t>(std::to_string(std::numeric_limits<int16_t>::max()) + "0"))
            },
            // signed int wrap
            { std::to_string(std::numeric_limits<uint64_t>::max()) + "s;",
                Node::Ptr(new Value<int16_t>(std::to_string(std::numeric_limits<uint64_t>::max())))
            }
        }
    },

    {
        Node::NodeType::ValueInt32Node,
        {
            { "00;",                Node::Ptr(new Value<int32_t>("0")) },
            { "1000000000000000;",  Node::Ptr(new Value<int32_t>("1000000000000000")) }, // signed int wrap
            { "0;",                 Node::Ptr(new Value<int32_t>("0")) },
            { "1234567890;",        Node::Ptr(new Value<int32_t>("1234567890")) },
            { "1;",                 Node::Ptr(new Value<int32_t>("1")) },
            // signed int wrap
            { std::to_string(std::numeric_limits<int64_t>::max()) + ";",
                Node::Ptr(new Value<int32_t>(std::to_string(std::numeric_limits<int64_t>::max())))
            },
            // signed int wrap
            { std::to_string(std::numeric_limits<uint64_t>::max()) + ";",
                Node::Ptr(new Value<int32_t>(std::to_string(std::numeric_limits<uint64_t>::max())))
            },
            // signed int wrap
            { std::to_string(std::numeric_limits<int32_t>::max()) + "0;",
                Node::Ptr(new Value<int32_t>(std::to_string(std::numeric_limits<int32_t>::max()) + "0"))
            },
            // overflow
            { std::to_string(std::numeric_limits<uint64_t>::max()) + "0;",
                Node::Ptr(new Value<int32_t>(std::to_string(std::numeric_limits<uint64_t>::max()) + "0"))
            }
        }
    },

    {
        Node::NodeType::ValueInt64Node,
        {
            { "01l;",                       Node::Ptr(new Value<int64_t>("1")) },
            { "0l;",                        Node::Ptr(new Value<int64_t>("0")) },
            { "1234567890l;",               Node::Ptr(new Value<int64_t>("1234567890l")) },
            // overflow
            { std::to_string(std::numeric_limits<int64_t>::max()) + "0l;",
                Node::Ptr(new Value<int64_t>(std::to_string(std::numeric_limits<uint64_t>::max()) + "0"))
            },
            // signed int wrap
            { std::to_string(uint64_t(std::numeric_limits<int64_t>::max()) + 1) + "l;",
                Node::Ptr(new Value<int64_t>(std::to_string(uint64_t(std::numeric_limits<int64_t>::max()) + 1ul)))
            }
        }
    },

    {
        Node::NodeType::ValueFloatNode,
        {
            { ".123456789f;",               Node::Ptr(new Value<float>(".123456789f")) },
            { "0.0f;",                      Node::Ptr(new Value<float>("0.0f")) },
            { "00.f;",                      Node::Ptr(new Value<float>("0.0f")) },
            { "0e+0f;",                     Node::Ptr(new Value<float>("0.0f")) },
            { "0e-0f;",                     Node::Ptr(new Value<float>("0.0f")) },
            { "0e0f;",                      Node::Ptr(new Value<float>("0.0f")) },
            { "1234567890.0987654321f;",    Node::Ptr(new Value<float>("1234567890.0987654321f")) },
            { "1e+6f;",                     Node::Ptr(new Value<float>("1e+6f")) },
            { "1E+6f;",                     Node::Ptr(new Value<float>("1E+6f")) },
            { "1e-6f;",                     Node::Ptr(new Value<float>("1e-6f")) },
            { "1E-6f;",                     Node::Ptr(new Value<float>("1E-6f")) },
            { "1e123456789f;",              Node::Ptr(new Value<float>("1e123456789f")) }, // overflow
            { "1e6f;",                      Node::Ptr(new Value<float>("1e6f")) },
            { "1E6f;",                      Node::Ptr(new Value<float>("1E6f")) }
        }
    },

    {
        Node::NodeType::ValueDoubleNode,
        {
            { ".123456789;",                Node::Ptr(new Value<double>(".123456789")) },
            { "0.0;",                       Node::Ptr(new Value<double>("0.0")) },
            { "0e0;",                       Node::Ptr(new Value<double>("0.0f")) },
            { "1.0;",                       Node::Ptr(new Value<double>("1.0")) },
            { "1234567890.00000000;",       Node::Ptr(new Value<double>("1234567890.0")) },
            { "1234567890.0987654321;",     Node::Ptr(new Value<double>("1234567890.0987654321")) },
            { "1234567890.10000000;",       Node::Ptr(new Value<double>("1234567890.1")) },
            { "1234567890e-0;",             Node::Ptr(new Value<double>("1234567890e-0")) },
            { "1e+6;",                      Node::Ptr(new Value<double>("1e+6")) },
            { "1e-6;",                      Node::Ptr(new Value<double>("1e-6")) },
            { "1e01;",                      Node::Ptr(new Value<double>("1e01")) },
            { "1e123456789;",               Node::Ptr(new Value<double>("1e123456789")) }, // overflow
            { "1e6;",                       Node::Ptr(new Value<double>("1e6")) },
            { "1E6;",                       Node::Ptr(new Value<double>("1E6")) },
            { std::to_string(std::numeric_limits<double>::max()) + ";",
                Node::Ptr(new Value<double>(std::to_string(std::numeric_limits<double>::max())))
            },
            { std::to_string(std::numeric_limits<double>::min()) + ";",
                Node::Ptr(new Value<double>(std::to_string(std::numeric_limits<double>::min())))
            }
        }
    },

    {
        Node::NodeType::ValueStrNode,
        {
            { "\"0.0\";",                   Node::Ptr(new Value<std::string>("0.0")) },
            { "\"0.0f\";",                  Node::Ptr(new Value<std::string>("0.0f")) },
            { "\"0\";",                     Node::Ptr(new Value<std::string>("0")) },
            { "\"1234567890.0987654321\";", Node::Ptr(new Value<std::string>("1234567890.0987654321")) },
            { "\"1234567890\";",            Node::Ptr(new Value<std::string>("1234567890")) },
            { "\"a1b2c3d4.e5f6g7.0\";",     Node::Ptr(new Value<std::string>("a1b2c3d4.e5f6g7.0")) },
            { "\"literal\";",               Node::Ptr(new Value<std::string>("literal")) },
            { "\"\";",                      Node::Ptr(new Value<std::string>("")) },
            { "\"" + std::to_string(std::numeric_limits<double>::lowest()) + "\";",
                Node::Ptr(new Value<std::string>(std::to_string(std::numeric_limits<double>::lowest())))
            },
            { "\"" + std::to_string(std::numeric_limits<double>::max()) + "\";",
                Node::Ptr(new Value<std::string>(std::to_string(std::numeric_limits<double>::max())))
            },
            { "\"" + std::to_string(std::numeric_limits<int64_t>::lowest()) + "\";",
                Node::Ptr(new Value<std::string>(std::to_string(std::numeric_limits<int64_t>::lowest())))
            },
            { "\"" + std::to_string(std::numeric_limits<int64_t>::max()) + "\";",
                Node::Ptr(new Value<std::string>(std::to_string(std::numeric_limits<int64_t>::max())))
            }
        }
    }
};

}

class TestValueNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestValueNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() {
        for (const auto& tests : value_tests) {
            TEST_SYNTAX_PASSES(tests.second);
        }
    }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestValueNode);

void TestValueNode::testASTNode()
{
    for (const auto& tests : value_tests) {
        const Node::NodeType nodeType = tests.first;
        for (const auto& test : tests.second) {
            const std::string& code = test.first;
            const Node* expected = test.second.get();
            const Tree::Ptr tree = parse(code.c_str());
            CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("No AST returned", code), static_cast<bool>(tree));

            // get the first statement
            const Node* result = tree->child(0)->child(0);
            CPPUNIT_ASSERT(result);
            CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid AST node", code),
                nodeType == result->nodetype());

            std::vector<const Node*> resultList, expectedList;
            linearize(*result, resultList);
            linearize(*expected, expectedList);

            if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
                std::ostringstream os;
                os << "\nExpected:\n";
                openvdb::ax::ast::print(*expected, true, os);
                os << "Result:\n";
                openvdb::ax::ast::print(*result, true, os);
                CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Value (literal) code", code) + os.str());
            }
        }
    }
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
