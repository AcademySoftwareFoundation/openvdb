// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

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
    { "return;", Node::Ptr(new Keyword(KeywordToken::RETURN)) },
    { "break;", Node::Ptr(new Keyword(KeywordToken::BREAK)) },
    { "continue;", Node::Ptr(new Keyword(KeywordToken::CONTINUE)) }
};

}

class TestKeywordNode : public ::testing::Test
{
};

TEST_F(TestKeywordNode, testSyntax)
{
    TEST_SYNTAX_PASSES(tests);
}

TEST_F(TestKeywordNode, testASTNode)
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(static_cast<bool>(tree)) << ERROR_MSG("No AST returned", code);

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        ASSERT_TRUE(result);
        const Keyword* resultAsKeyword = static_cast<const Keyword*>(result);
        ASSERT_TRUE(resultAsKeyword);
        ASSERT_TRUE(Node::KeywordNode == result->nodetype()) << ERROR_MSG("Invalid AST node", code);

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            FAIL() << ERROR_MSG("Mismatching Trees for Return code", code) + os.str();
        }
    }
}

