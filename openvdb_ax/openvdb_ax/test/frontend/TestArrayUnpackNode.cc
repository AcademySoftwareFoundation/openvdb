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
        const Tree::ConstPtr tree = parse(code.c_str());
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

