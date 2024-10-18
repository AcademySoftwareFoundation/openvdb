// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/ast/Parse.h>
#include <openvdb_ax/compiler/Logger.h>

#include <gtest/gtest.h>

// namespace must be the same as where Logger is defined in order
// to access private fields. See also
//https://google.github.io/googletest/advanced.html#testing-private-code

namespace openvdb {
namespace OPENVDB_VERSION_NAME {
namespace ax {
class TestLogger : public ::testing::Test
{
};

/*
extern openvdb::ax::Logger* axlog;
/// @note We don't deploy the grammar c files as part of the AX install.
///  Because the unit tests are structured to be able to build against
///  an existing version of AX we can't include the parser.cc here for
///  access to newNode. It's tested through other methods, but we
///  should restructure how this code is shared by perhaps moving it to
///  a shared header (including the definition of AXLTYPE)
TEST_F(TestLogger, testParseNewNode)
{
    openvdb::ax::Logger logger;
    axlog = &logger;// setting global Logger* used in parser
    AXLTYPE location;
    location.first_line = 100;
    location.first_column = 65;
    const auto& nodeToLineColMap = logger.mNodeToLineColMap;
    ASSERT_TRUE(nodeToLineColMap.empty());

    const openvdb::ax::ast::Local* testLocal =
        newNode<openvdb::ax::ast::Local>(&location, "test");

    ASSERT_EQ(nodeToLineColMap.size(),static_cast<size_t>(1));
    openvdb::ax::Logger::CodeLocation lineCol = nodeToLineColMap.at(testLocal);
    ASSERT_EQ(lineCol.first, static_cast<size_t>(100));
    ASSERT_EQ(lineCol.second, static_cast<size_t>(65));
}
*/

TEST_F(TestLogger, testParseSetsTree)
{
    openvdb::ax::Logger logger;
    ASSERT_TRUE(!logger.mTreePtr);
    std::string code("");
    openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(code.c_str(), logger);
    ASSERT_TRUE(tree);
    ASSERT_EQ(tree, logger.mTreePtr);
}

TEST_F(TestLogger, testAddError)
{
    std::vector<std::string> messages;
    openvdb::ax::Logger logger([&messages](const std::string& message) {
        messages.emplace_back(message);
    });
    ASSERT_TRUE(!logger.hasError());
    ASSERT_EQ(logger.errors(), messages.size());

    openvdb::ax::Logger::CodeLocation codeLocation(1,1);
    std::string message("test");

    logger.error(message, codeLocation);
    ASSERT_TRUE(logger.hasError());
    ASSERT_EQ(messages.size(), static_cast<size_t>(1));
    ASSERT_EQ(logger.errors(), static_cast<size_t>(1));
    ASSERT_EQ(strcmp(messages.back().c_str(), "[1] error: test 1:1"), 0);

    logger.error(message, codeLocation);
    ASSERT_EQ(messages.size(), static_cast<size_t>(2));
    ASSERT_EQ(logger.errors(), static_cast<size_t>(2));

    logger.clear();
    ASSERT_TRUE(!logger.hasError());
    ASSERT_EQ(logger.errors(), static_cast<size_t>(0));

    openvdb::ax::ast::Local testLocal("name");
    logger.error(message, &testLocal);
    ASSERT_TRUE(logger.hasError());
    ASSERT_EQ(logger.errors(), static_cast<size_t>(1));
    ASSERT_EQ(messages.size(), static_cast<size_t>(3));

    ASSERT_TRUE(!logger.mTreePtr);
    ASSERT_EQ(strcmp(messages.back().c_str(), "[1] error: test"), 0);

    logger.clear();
    ASSERT_TRUE(!logger.hasError());

    // test that add error finds code location
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(" a;", logger);
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        ASSERT_TRUE(local);

        logger.error(message, local);
        ASSERT_TRUE(logger.hasError());
        ASSERT_EQ(logger.errors(), static_cast<size_t>(1));
        ASSERT_TRUE(logger.mTreePtr);
        ASSERT_EQ(strcmp(messages.back().c_str(), "[1] error: test 1:2"), 0);
    }

    logger.clear();
    ASSERT_TRUE(!logger.hasError());
    // test add error finds code location even when node is deep copy
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("a;", logger);
        openvdb::ax::ast::Tree::ConstPtr treeCopy(tree->copy());
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        ASSERT_TRUE(local);
        const openvdb::ax::ast::Node* localCopy = treeCopy->child(0)->child(0);
        ASSERT_TRUE(localCopy);
        // add referring to copy
        logger.error(message, localCopy);
        ASSERT_TRUE(logger.hasError());
        ASSERT_EQ(logger.errors(), static_cast<size_t>(1));
        ASSERT_EQ(messages.size(), static_cast<size_t>(5));

        ASSERT_TRUE(logger.mTreePtr);
        ASSERT_EQ(strcmp(messages.back().c_str(), "[1] error: test 1:1"), 0);
    }
}

TEST_F(TestLogger, testAddWarning)
{
    std::vector<std::string> messages;
    openvdb::ax::Logger logger([](const std::string&) {},
        [&messages](const std::string& message) {
            messages.emplace_back(message);
    });
    ASSERT_TRUE(!logger.hasWarning());
    ASSERT_EQ(logger.warnings(), messages.size());

    openvdb::ax::Logger::CodeLocation codeLocation(1,1);
    std::string message("test");

    logger.warning(message, codeLocation);
    ASSERT_TRUE(logger.hasWarning());
    ASSERT_EQ(messages.size(), static_cast<size_t>(1));
    ASSERT_EQ(logger.warnings(), static_cast<size_t>(1));
    ASSERT_EQ(strcmp(messages.back().c_str(), "[1] warning: test 1:1"), 0);

    logger.warning(message, codeLocation);
    ASSERT_EQ(messages.size(), static_cast<size_t>(2));
    ASSERT_EQ(logger.warnings(), static_cast<size_t>(2));

    logger.clear();
    ASSERT_TRUE(!logger.hasWarning());
    ASSERT_EQ(logger.warnings(), static_cast<size_t>(0));

    openvdb::ax::ast::Local testLocal("name");
    logger.warning(message, &testLocal);
    ASSERT_TRUE(logger.hasWarning());
    ASSERT_EQ(logger.warnings(), static_cast<size_t>(1));
    ASSERT_EQ(messages.size(), static_cast<size_t>(3));

    ASSERT_TRUE(!logger.mTreePtr);
    ASSERT_EQ(strcmp(messages.back().c_str(), "[1] warning: test"), 0);

    logger.clear();
    ASSERT_TRUE(!logger.hasWarning());

    // test that add warning finds code location
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(" a;", logger);
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        ASSERT_TRUE(local);

        logger.warning(message, local);
        ASSERT_TRUE(logger.hasWarning());
        ASSERT_EQ(logger.warnings(), static_cast<size_t>(1));
        ASSERT_TRUE(logger.mTreePtr);
        ASSERT_EQ(strcmp(messages.back().c_str(), "[1] warning: test 1:2"), 0);
    }

    logger.clear();
    ASSERT_TRUE(!logger.hasWarning());
    // test add warning finds code location even when node is deep copy
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("a;", logger);
        openvdb::ax::ast::Tree::ConstPtr treeCopy(tree->copy());
        const openvdb::ax::ast::Node* local = tree->child(0)->child(0);
        ASSERT_TRUE(local);
        const openvdb::ax::ast::Node* localCopy = treeCopy->child(0)->child(0);
        ASSERT_TRUE(localCopy);
        // add referring to copy
        logger.warning(message, localCopy);
        ASSERT_TRUE(logger.hasWarning());
        ASSERT_EQ(logger.warnings(), static_cast<size_t>(1));
        ASSERT_EQ(messages.size(), static_cast<size_t>(5));

        ASSERT_TRUE(logger.mTreePtr);
        ASSERT_EQ(strcmp(messages.back().c_str(), "[1] warning: test 1:1"), 0);
    }
}

TEST_F(TestLogger, testWarningsAsErrors)
{
    openvdb::ax::Logger logger([](const std::string&) {});
    const std::string message("test");
    const openvdb::ax::Logger::CodeLocation location(10,20);
    logger.setWarningsAsErrors(true);
    ASSERT_TRUE(!logger.hasError());
    ASSERT_TRUE(!logger.hasWarning());

    logger.warning(message, location);
    ASSERT_TRUE(logger.hasError());
    ASSERT_TRUE(!logger.hasWarning());
}

TEST_F(TestLogger, testMaxErrors)
{
    openvdb::ax::Logger logger([](const std::string&) {});
    const std::string message("test");
    const openvdb::ax::Logger::CodeLocation location(10,20);

    ASSERT_TRUE(logger.error(message, location));
    ASSERT_TRUE(logger.error(message, location));
    ASSERT_TRUE(logger.error(message, location));
    logger.clear();
    logger.setMaxErrors(2);
    ASSERT_TRUE(logger.error(message, location));
    ASSERT_TRUE(!logger.error(message, location));
    ASSERT_TRUE(!logger.error(message, location));
    // setMaxErrors doesn't limit the error counter
    ASSERT_EQ(size_t(3), logger.errors());
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
