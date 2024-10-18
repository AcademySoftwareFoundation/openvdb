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
    { "$a;",        Node::Ptr(new ExternalVariable("a", CoreType::FLOAT)) },
    { "bool$_a;",   Node::Ptr(new ExternalVariable("_a", CoreType::BOOL)) },
    { "i$a1;",      Node::Ptr(new ExternalVariable("a1", CoreType::INT32)) },
    { "int$abc;",   Node::Ptr(new ExternalVariable("abc", CoreType::INT32)) },
    { "int32$abc;", Node::Ptr(new ExternalVariable("abc", CoreType::INT32)) },
    { "int64$a;",    Node::Ptr(new ExternalVariable("a", CoreType::INT64)) },
    { "f$a;",       Node::Ptr(new ExternalVariable("a", CoreType::FLOAT)) },
    { "float$a;",   Node::Ptr(new ExternalVariable("a", CoreType::FLOAT)) },
    { "double$a;",  Node::Ptr(new ExternalVariable("a", CoreType::DOUBLE)) },
    { "vec3i$a;",   Node::Ptr(new ExternalVariable("a", CoreType::VEC3I)) },
    { "v$a;",       Node::Ptr(new ExternalVariable("a", CoreType::VEC3F)) },
    { "vec3f$a;",   Node::Ptr(new ExternalVariable("a", CoreType::VEC3F)) },
    { "vec3d$a;",   Node::Ptr(new ExternalVariable("a", CoreType::VEC3D)) },
    { "string$a;",  Node::Ptr(new ExternalVariable("a", CoreType::STRING)) },
    { "s$a;",       Node::Ptr(new ExternalVariable("a", CoreType::STRING)) },
};

}

class TestExternalVariableNode : public ::testing::Test
{
};

TEST_F(TestExternalVariableNode, testSyntax)
{
    TEST_SYNTAX_PASSES(tests);
}

TEST_F(TestExternalVariableNode, testASTNode)
{
    for (const auto& test : tests) {
        const std::string& code = test.first;
        const Node* expected = test.second.get();
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(static_cast<bool>(tree)) << ERROR_MSG("No AST returned", code);

        // get the first statement
        const Node* result = tree->child(0)->child(0);
        ASSERT_TRUE(result);
        ASSERT_TRUE(Node::ExternalVariableNode == result->nodetype()) << ERROR_MSG("Invalid AST node", code);

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            FAIL() << ERROR_MSG("Mismatching Trees for External Variable code", code) + os.str();
        }
    }
}

