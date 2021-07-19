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
    { "int32 a = (1,2,3), b=1, c=(b=1);", Node::Ptr(new StatementList({
                                            new DeclareLocal(CoreType::INT32, new Local("a"),
                                            new CommaOperator({
                                                new Value<int32_t>(1),
                                                new Value<int32_t>(2),
                                                new Value<int32_t>(3),
                                            })),
                                            new DeclareLocal(CoreType::INT32, new Local("b"), new Value<int32_t>(1)),
                                            new DeclareLocal(CoreType::INT32, new Local("c"),
                                                new AssignExpression(
                                                    new Local("b"),
                                                    new Value<int32_t>(1))),
                                        }))
    },
    { "int32 a, b;",                Node::Ptr(new StatementList({
                                        new DeclareLocal(CoreType::INT32, new Local("a")),
                                        new DeclareLocal(CoreType::INT32, new Local("b"))
                                    }))
    },
    { "int32 a, b = 1;",            Node::Ptr(new StatementList({
                                        new DeclareLocal(CoreType::INT32, new Local("a")),
                                        new DeclareLocal(CoreType::INT32, new Local("b"), new Value<int32_t>(1))
                                    }))
    },
    { "int32 a, b = 1, c = 1;",     Node::Ptr(new StatementList({
                                        new DeclareLocal(CoreType::INT32, new Local("a")),
                                        new DeclareLocal(CoreType::INT32, new Local("b"), new Value<int32_t>(1)),
                                        new DeclareLocal(CoreType::INT32, new Local("c"), new Value<int32_t>(1))
                                    }))
    },
    { "int32 a, b = 1, c;",         Node::Ptr(new StatementList({
                                        new DeclareLocal(CoreType::INT32, new Local("a")),
                                        new DeclareLocal(CoreType::INT32, new Local("b"), new Value<int32_t>(1)),
                                        new DeclareLocal(CoreType::INT32, new Local("c"))
                                    }))
    },
    { "int32 a, b = 1, c, d = 1;",  Node::Ptr(new StatementList({
                                        new DeclareLocal(CoreType::INT32, new Local("a")),
                                        new DeclareLocal(CoreType::INT32, new Local("b"), new Value<int32_t>(1)),
                                        new DeclareLocal(CoreType::INT32, new Local("c")),
                                        new DeclareLocal(CoreType::INT32, new Local("d"), new Value<int32_t>(1))
                                    }))
    }
};

}

class TestStatementList : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestStatementList);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests); }
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStatementList);

void TestStatementList::testASTNode()
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
            Node::StatementListNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Statement List code", code) + os.str());
        }
    }
}

