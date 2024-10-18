// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>
#include <openvdb_ax/ast/PrintTree.h>
#include <openvdb_ax/Exceptions.h>

#include "../util.h"

#include <gtest/gtest.h>

#include <string>

using namespace openvdb::ax::ast;
using namespace openvdb::ax::ast::tokens;

namespace {

static const unittest_util::CodeTests tests =
{
    { "a_;",   Node::Ptr(new Local("a_")) },
    { "_a;",   Node::Ptr(new Local("_a")) },
    { "_;",    Node::Ptr(new Local("_")) },
    { "aa;",   Node::Ptr(new Local("aa")) },
    { "A;",    Node::Ptr(new Local("A")) },
    { "_A;",   Node::Ptr(new Local("_A")) },
    { "a1;",   Node::Ptr(new Local("a1")) },
    { "_1;",   Node::Ptr(new Local("_1")) },
    { "abc;",  Node::Ptr(new Local("abc")) },
    { "D1f;",  Node::Ptr(new Local("D1f")) },
    { "var;",  Node::Ptr(new Local("var")) }
};

}

class TestLocalNode : public ::testing::Test
{
};

TEST_F(TestLocalNode, testSyntax)
{
    TEST_SYNTAX_PASSES(tests);
}

TEST_F(TestLocalNode, testASTNode)
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(static_cast<bool>(tree)) << ERROR_MSG("No AST returned", code);

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        ASSERT_TRUE(result);
        ASSERT_TRUE(Node::LocalNode == result->nodetype()) << ERROR_MSG("Invalid AST node", code);

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            FAIL() << ERROR_MSG("Mismatching Trees for Local code", code) + os.str();
        }
    }
}


