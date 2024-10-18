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
    { "a++;",       Node::Ptr(new Crement(new Local("a"), Crement::Operation::Increment, /*post*/true)) },
    { "++a;",       Node::Ptr(new Crement(new Local("a"), Crement::Operation::Increment, /*post*/false)) },
    { "a--;",       Node::Ptr(new Crement(new Local("a"), Crement::Operation::Decrement, /*post*/true)) },
    { "--a;",       Node::Ptr(new Crement(new Local("a"), Crement::Operation::Decrement, /*post*/false)) },
    { "s@a--;",     Node::Ptr(new Crement(new Attribute("a", CoreType::STRING), Crement::Operation::Decrement, /*post*/true)) },
    { "f@a++;",     Node::Ptr(new Crement(new Attribute("a", CoreType::FLOAT), Crement::Operation::Increment, /*post*/true)) },
    { "++f@a;",     Node::Ptr(new Crement(new Attribute("a", CoreType::FLOAT), Crement::Operation::Increment, /*post*/false)) },
    { "++mat3f@a;", Node::Ptr(new Crement(new Attribute("a", CoreType::MAT3F), Crement::Operation::Increment, /*post*/false)) }
};

}

class TestCrementNode : public ::testing::Test
{
};

TEST_F(TestCrementNode, testSyntax)
{
    TEST_SYNTAX_PASSES(tests);
};

TEST_F(TestCrementNode, testASTNode)
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(static_cast<bool>(tree)) << ERROR_MSG("No AST returned", code);

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        ASSERT_TRUE(result);
        ASSERT_TRUE(Node::CrementNode == result->nodetype()) << ERROR_MSG("Invalid AST node", code);

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            FAIL() << ERROR_MSG("Mismatching Trees for Crement code", code) + os.str();
        }
    }
}

