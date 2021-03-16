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

class TestCrementNode : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestCrementNode);
    CPPUNIT_TEST(testSyntax);
    CPPUNIT_TEST(testASTNode);
    CPPUNIT_TEST_SUITE_END();

    void testSyntax() { TEST_SYNTAX_PASSES(tests) };
    void testASTNode();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestCrementNode);

void TestCrementNode::testASTNode()
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
            Node::CrementNode == result->nodetype());

        std::vector<const Node*> resultList, expectedList;
        linearize(*result, resultList);
        linearize(*expected, expectedList);

        if (!unittest_util::compareLinearTrees(expectedList, resultList)) {
            std::ostringstream os;
            os << "\nExpected:\n";
            openvdb::ax::ast::print(*expected, true, os);
            os << "Result:\n";
            openvdb::ax::ast::print(*result, true, os);
            CPPUNIT_FAIL(ERROR_MSG("Mismatching Trees for Crement code", code) + os.str());
        }
    }
}

