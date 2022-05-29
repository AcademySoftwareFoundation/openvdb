// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/ast/Parse.h>
#include <openvdb_ax/compiler/Logger.h>

#include <cppunit/extensions/HelperMacros.h>

class TestLogger : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestLogger);
    //CPPUNIT_TEST(testParseNewNode);
    CPPUNIT_TEST(testParseSetsTree);
    CPPUNIT_TEST(testAddError);
    CPPUNIT_TEST(testAddWarning);
    CPPUNIT_TEST(testWarningsAsErrors);
    CPPUNIT_TEST(testMaxErrors);

    CPPUNIT_TEST_SUITE_END();

    //void testParseNewNode();
    void testParseSetsTree();
    void testAddError();
    void testAddWarning();
    void testWarningsAsErrors();
    void testMaxErrors();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestLogger);

/*
extern openvdb::ax::Logger* axlog;
/// @note We don't deploy the grammar c files as part of the AX install.
///  Because the unit tests are structured to be able to build against
///  an existing version of AX we can't include the parser.cc here for
///  access to newNode. It's tested through other methods, but we
///  should restructure how this code is shared by perhaps moving it to
///  a shared header (including the definition of AXLTYPE)
void
TestLogger::testParseNewNode()
{
    openvdb::ax::Logger logger;
    axlog = &logger;// setting global Logger* used in parser
    AXLTYPE location;
    location.first_line = 100;
    location.first_column = 65;
    const auto& nodeToLineColMap = logger.mNodeToLineColMap;
    CPPUNIT_ASSERT(nodeToLineColMap.empty());

    const openvdb::ax::ast::Local* testLocal =
        newNode<openvdb::ax::ast::Local>(&location, "test");

    CPPUNIT_ASSERT_EQUAL(nodeToLineColMap.size(),static_cast<size_t>(1));
    openvdb::ax::Logger::CodeLocation lineCol = nodeToLineColMap.at(testLocal);
    CPPUNIT_ASSERT_EQUAL(lineCol.first, static_cast<size_t>(100));
    CPPUNIT_ASSERT_EQUAL(lineCol.second, static_cast<size_t>(65));
}
*/

void
TestLogger::testParseSetsTree()
{
    openvdb::ax::Logger logger;
    CPPUNIT_ASSERT(!logger.mTreePtr);
    std::string code("");
    openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(code.c_str(), logger);
    CPPUNIT_ASSERT(tree);
    CPPUNIT_ASSERT_EQUAL(tree, logger.mTreePtr);
}

void
TestLogger::testAddError()
{
    std::vector<std::string> messages;
    openvdb::ax::Logger logger([&messages](const std::string& message) {
        messages.emplace_back(message);
    });
    CPPUNIT_ASSERT(!logger.hasError());
    CPPUNIT_ASSERT_EQUAL(logger.errors(), messages.size());

    openvdb::ax::Logger::CodeLocation codeLocation(1,1);
    std::string message("test");

    logger.error(message, codeLocation);
    CPPUNIT_ASSERT(logger.hasError());
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] error: test 1:1"), 0);

    logger.error(message, codeLocation);
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(2));
    CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(2));

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasError());
    CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(0));

    openvdb::ax::ast::Local testLocal("name");
    logger.error(message, &testLocal);
    CPPUNIT_ASSERT(logger.hasError());
    CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(3));

    CPPUNIT_ASSERT(!logger.mTreePtr);
    CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] error: test"), 0);

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasError());

    // test that add error finds code location
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(" a;", logger);
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        CPPUNIT_ASSERT(local);

        logger.error(message, local);
        CPPUNIT_ASSERT(logger.hasError());
        CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(1));
        CPPUNIT_ASSERT(logger.mTreePtr);
        CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] error: test 1:2"), 0);
    }

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasError());
    // test add error finds code location even when node is deep copy
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("a;", logger);
        openvdb::ax::ast::Tree::ConstPtr treeCopy(tree->copy());
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        CPPUNIT_ASSERT(local);
        const openvdb::ax::ast::Node* localCopy = treeCopy->child(0)->child(0);
        CPPUNIT_ASSERT(localCopy);
        // add referring to copy
        logger.error(message, localCopy);
        CPPUNIT_ASSERT(logger.hasError());
        CPPUNIT_ASSERT_EQUAL(logger.errors(), static_cast<size_t>(1));
        CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(5));

        CPPUNIT_ASSERT(logger.mTreePtr);
        CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] error: test 1:1"), 0);
    }
}

void
TestLogger::testAddWarning()
{
    std::vector<std::string> messages;
    openvdb::ax::Logger logger([](const std::string&) {},
        [&messages](const std::string& message) {
            messages.emplace_back(message);
    });
    CPPUNIT_ASSERT(!logger.hasWarning());
    CPPUNIT_ASSERT_EQUAL(logger.warnings(), messages.size());

    openvdb::ax::Logger::CodeLocation codeLocation(1,1);
    std::string message("test");

    logger.warning(message, codeLocation);
    CPPUNIT_ASSERT(logger.hasWarning());
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] warning: test 1:1"), 0);

    logger.warning(message, codeLocation);
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(2));
    CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(2));

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasWarning());
    CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(0));

    openvdb::ax::ast::Local testLocal("name");
    logger.warning(message, &testLocal);
    CPPUNIT_ASSERT(logger.hasWarning());
    CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(1));
    CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(3));

    CPPUNIT_ASSERT(!logger.mTreePtr);
    CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] warning: test"), 0);

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasWarning());

    // test that add warning finds code location
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(" a;", logger);
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        CPPUNIT_ASSERT(local);

        logger.warning(message, local);
        CPPUNIT_ASSERT(logger.hasWarning());
        CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(1));
        CPPUNIT_ASSERT(logger.mTreePtr);
        CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] warning: test 1:2"), 0);
    }

    logger.clear();
    CPPUNIT_ASSERT(!logger.hasWarning());
    // test add warning finds code location even when node is deep copy
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("a;", logger);
        openvdb::ax::ast::Tree::ConstPtr treeCopy(tree->copy());
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        CPPUNIT_ASSERT(local);
        const openvdb::ax::ast::Node* localCopy = treeCopy->child(0)->child(0);
        CPPUNIT_ASSERT(localCopy);
        // add referring to copy
        logger.warning(message, localCopy);
        CPPUNIT_ASSERT(logger.hasWarning());
        CPPUNIT_ASSERT_EQUAL(logger.warnings(), static_cast<size_t>(1));
        CPPUNIT_ASSERT_EQUAL(messages.size(), static_cast<size_t>(5));

        CPPUNIT_ASSERT(logger.mTreePtr);
        CPPUNIT_ASSERT_EQUAL(strcmp(messages.back().c_str(), "[1] warning: test 1:1"), 0);
    }
}

void
TestLogger::testWarningsAsErrors()
{
    openvdb::ax::Logger logger([](const std::string&) {});
    const std::string message("test");
    const openvdb::ax::Logger::CodeLocation location(10,20);
    logger.setWarningsAsErrors(true);
    CPPUNIT_ASSERT(!logger.hasError());
    CPPUNIT_ASSERT(!logger.hasWarning());

    logger.warning(message, location);
    CPPUNIT_ASSERT(logger.hasError());
    CPPUNIT_ASSERT(!logger.hasWarning());
}

void
TestLogger::testMaxErrors()
{
    openvdb::ax::Logger logger([](const std::string&) {});
    const std::string message("test");
    const openvdb::ax::Logger::CodeLocation location(10,20);

    CPPUNIT_ASSERT(logger.error(message, location));
    CPPUNIT_ASSERT(logger.error(message, location));
    CPPUNIT_ASSERT(logger.error(message, location));
    logger.clear();
    logger.setMaxErrors(2);
    CPPUNIT_ASSERT(logger.error(message, location));
    CPPUNIT_ASSERT(!logger.error(message, location));
    CPPUNIT_ASSERT(!logger.error(message, location));
    // setMaxErrors doesn't limit the error counter
    CPPUNIT_ASSERT_EQUAL(size_t(3), logger.errors());
}

