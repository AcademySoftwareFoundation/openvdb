// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/compiler/AttributeBindings.h>

#include <gtest/gtest.h>

class TestAttributeBindings : public ::testing::Test
{
};

TEST_F(TestAttributeBindings, testSet)
{
    openvdb::ax::AttributeBindings bindings;
    ASSERT_TRUE(!bindings.isBoundAXName("a"));
    ASSERT_TRUE(!bindings.isBoundDataName("b"));
    bindings.set("a", "b");
    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("a"));
}

TEST_F(TestAttributeBindings, testSetFromVector)
{
    std::vector<std::pair<std::string,std::string>> vec = {{"a", "b"}};
    openvdb::ax::AttributeBindings bindings0(vec);
    ASSERT_TRUE(bindings0.isBoundAXName("a"));
    ASSERT_TRUE(bindings0.isBoundDataName("b"));
    ASSERT_EQ(*bindings0.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings0.axNameBoundTo("b"), std::string("a"));
}

TEST_F(TestAttributeBindings, testSetFromInitList)
{
    openvdb::ax::AttributeBindings bindings0 = {{"a", "b"}};
    ASSERT_TRUE(bindings0.isBoundAXName("a"));
    ASSERT_TRUE(bindings0.isBoundDataName("b"));
    ASSERT_EQ(*bindings0.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings0.axNameBoundTo("b"), std::string("a"));

    // initializer list ctor
    openvdb::ax::AttributeBindings bindings1({{"a","b"}});
    ASSERT_TRUE(bindings1.isBoundAXName("a"));
    ASSERT_TRUE(bindings1.isBoundDataName("b"));
    ASSERT_EQ(*bindings1.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings1.axNameBoundTo("b"), std::string("a"));
}

TEST_F(TestAttributeBindings, testSetToExistingAXName)
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "b");
    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("a"));
    bindings.set("a", "c");
    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("c"));
    ASSERT_TRUE(!bindings.isBoundDataName("b"));
    ASSERT_TRUE(!bindings.axNameBoundTo("b")); // i.e. is nullptr
    ASSERT_TRUE(bindings.isBoundDataName("c"));
    ASSERT_EQ(*bindings.axNameBoundTo("c"), std::string("a"));
}

TEST_F(TestAttributeBindings, testSetToExistingDataName)
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "b");
    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("a"));
    bindings.set("c", "b");
    ASSERT_TRUE(bindings.isBoundAXName("c"));
    ASSERT_EQ(*bindings.dataNameBoundTo("c"), std::string("b"));

    ASSERT_TRUE(!bindings.isBoundAXName("a"));
    ASSERT_TRUE(!bindings.dataNameBoundTo("a")); // i.e. is nullptr

    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("c"));
}

TEST_F(TestAttributeBindings, testSwapNames)
{
    openvdb::ax::AttributeBindings bindings;
    bindings.set("a", "a");
    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_TRUE(bindings.isBoundDataName("a"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("a"));
    ASSERT_EQ(*bindings.axNameBoundTo("a"), std::string("a"));
    bindings.set("b", "b");
    ASSERT_TRUE(bindings.isBoundAXName("b"));
    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.dataNameBoundTo("b"), std::string("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("b"));

    bindings.set("a", "b");

    ASSERT_TRUE(bindings.isBoundAXName("a"));
    ASSERT_EQ(*bindings.dataNameBoundTo("a"), std::string("b"));
    ASSERT_TRUE(bindings.isBoundDataName("b"));
    ASSERT_EQ(*bindings.axNameBoundTo("b"), std::string("a"));

    ASSERT_TRUE(!bindings.isBoundDataName("a"));
    ASSERT_TRUE(!bindings.axNameBoundTo("a"));
}
