// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/compiler/AttributeBindings.h>

#include <cppunit/extensions/HelperMacros.h>

class TestAttributeBindings : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestAttributeBindings);
    CPPUNIT_TEST(testSet);
    CPPUNIT_TEST(testSetFromVector);
    CPPUNIT_TEST(testSetFromInitList);
    CPPUNIT_TEST(testSetToExistingAXName);
    CPPUNIT_TEST(testSetToExistingDataName);
    CPPUNIT_TEST(testSwapNames);
    CPPUNIT_TEST_SUITE_END();

    void testSet();
    void testSetFromVector();
    void testSetFromInitList();
    void testSetToExistingAXName();
    void testSetToExistingDataName();
    void testSwapNames();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeBindings);


void
TestAttributeBindings::testSet()
{
    openvdb::ax::AttributeBindings bindings;
    CPPUNIT_ASSERT(!bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(!bindings.isBoundDataName("b"));
    bindings.set("a", "b");
    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("a"));
}

void
TestAttributeBindings::testSetFromVector()
{
    std::vector<std::pair<std::string,std::string>> vec = {{"a", "b"}};
    openvdb::ax::AttributeBindings bindings0(vec);
    CPPUNIT_ASSERT(bindings0.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings0.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings0.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings0.axNameBoundTo("b"), std::string("a"));
}

void
TestAttributeBindings::testSetFromInitList()
{
    openvdb::ax::AttributeBindings bindings0 = {{"a", "b"}};
    CPPUNIT_ASSERT(bindings0.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings0.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings0.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings0.axNameBoundTo("b"), std::string("a"));

    // initializer list ctor
    openvdb::ax::AttributeBindings bindings1({{"a","b"}});
    CPPUNIT_ASSERT(bindings1.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings1.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings1.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings1.axNameBoundTo("b"), std::string("a"));
}

void
TestAttributeBindings::testSetToExistingAXName()
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "b");
    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("a"));
    bindings.set("a", "c");
    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("c"));
    CPPUNIT_ASSERT(!bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT(!bindings.axNameBoundTo("b")); // i.e. is nullptr
    CPPUNIT_ASSERT(bindings.isBoundDataName("c"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("c"), std::string("a"));
}

void
TestAttributeBindings::testSetToExistingDataName()
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "b");
    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("a"));
    bindings.set("c", "b");
    CPPUNIT_ASSERT(bindings.isBoundAXName("c"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("c"), std::string("b"));

    CPPUNIT_ASSERT(!bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(!bindings.dataNameBoundTo("a")); // i.e. is nullptr

    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("c"));
}

void
TestAttributeBindings::testSwapNames()
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "a");
    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("a"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("a"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("a"), std::string("a"));
    bindings.set("b", "b");
    CPPUNIT_ASSERT(bindings.isBoundAXName("b"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("b"), std::string("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("b"));

    bindings.set("a", "b");

    CPPUNIT_ASSERT(bindings.isBoundAXName("a"));
    CPPUNIT_ASSERT_EQUAL(*bindings.dataNameBoundTo("a"), std::string("b"));
    CPPUNIT_ASSERT(bindings.isBoundDataName("b"));
    CPPUNIT_ASSERT_EQUAL(*bindings.axNameBoundTo("b"), std::string("a"));

    CPPUNIT_ASSERT(!bindings.isBoundDataName("a"));
    CPPUNIT_ASSERT(!bindings.axNameBoundTo("a"));
}
