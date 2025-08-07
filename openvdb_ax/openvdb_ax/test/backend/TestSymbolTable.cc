// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "util.h"

#include <openvdb_ax/codegen/SymbolTable.h>

#include <gtest/gtest.h>

template <typename T>
using LLVMType = openvdb::ax::codegen::LLVMType<T>;

class TestSymbolTable : public ::testing::Test
{
};

TEST_F(TestSymbolTable, testSingleTable)
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    llvm::Value* value1 = builder.CreateAlloca(type);
    llvm::Value* value2 = builder.CreateAlloca(type);
    ASSERT_TRUE(value1);
    ASSERT_TRUE(value2);

    openvdb::ax::codegen::SymbolTable<llvm::Value*> table;
    ASSERT_TRUE(table.map().empty());

    ASSERT_TRUE(table.insert("test", value1));
    ASSERT_TRUE(!table.insert("test", nullptr));
    ASSERT_TRUE(table.exists("test"));
    ASSERT_EQ(value1, *table.get("test"));

    table.clear();
    ASSERT_TRUE(table.map().empty());
    ASSERT_TRUE(!table.exists("test"));

    ASSERT_TRUE(table.insert("test", value1));
    ASSERT_TRUE(table.replace("test", value2));
    ASSERT_TRUE(!table.replace("other", value2));

    ASSERT_TRUE(table.exists("test"));
    ASSERT_TRUE(table.exists("other"));

    ASSERT_EQ(value2, *table.get("test"));
    ASSERT_EQ(value2, *table.get("other"));
}

TEST_F(TestSymbolTable, testTableBlocks)
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    openvdb::ax::codegen::Value value1 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value2 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value3 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value4 = openvdb::ax::codegen::Value::Alloc(builder, type);
    ASSERT_TRUE(value1);
    ASSERT_TRUE(value2);
    ASSERT_TRUE(value3);
    ASSERT_TRUE(value4);

    // test table insertion and erase

    openvdb::ax::codegen::SymbolTableBlocks tables;
    openvdb::ax::codegen::SymbolTable<openvdb::ax::codegen::Value>* table1 = tables.getOrInsert(0);
    openvdb::ax::codegen::SymbolTable<openvdb::ax::codegen::Value>* table2 = tables.getOrInsert(1);

    ASSERT_EQ(table1, tables.get(0));
    ASSERT_EQ(table2, tables.get(1));

    tables.getOrInsert(2);
    tables.getOrInsert(3);
    tables.getOrInsert(5);

    ASSERT_TRUE(tables.get(4) == nullptr);
    ASSERT_TRUE(tables.erase(5));
    ASSERT_TRUE(tables.erase(3));
    ASSERT_TRUE(tables.erase(2));

    table1->insert("top1", value1);
    table1->insert("top2", value2);

    // test find methods

    const openvdb::ax::codegen::Value* result = tables.find("top1");
    ASSERT_EQ(value1, *result);
    result = tables.find("top2");
    ASSERT_EQ(value2, *result);

    table1 = tables.getOrInsert(3);
    table2 = tables.getOrInsert(5);
    tables.getOrInsert(6);

    // test multi table find methods

    table1->insert("table_level_2", value3);
    table2->insert("table_level_4", value4);

    // test find second nested value

    result = tables.find("table_level_2", 1);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_2", 2);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_2", 3);
    ASSERT_TRUE(result);
    ASSERT_EQ(value3, *result);

    // test find fourth nested value

    result = tables.find("table_level_4", 1);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_4", 4);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_4", 5);
    ASSERT_TRUE(result);
    ASSERT_EQ(value4, *result);

    result = tables.find("table_level_4", 10000);
    ASSERT_TRUE(result);
    ASSERT_EQ(value4, *result);

    // test find fourth nested value with matching name

    table1->insert("table_level_4", value1);

    result = tables.find("table_level_4");
    ASSERT_TRUE(result);
    ASSERT_EQ(value4, *result);

    result = tables.find("table_level_4", 5);
    ASSERT_TRUE(result);
    ASSERT_EQ(value4, *result);

    result = tables.find("table_level_4", 4);
    ASSERT_TRUE(result);
    ASSERT_EQ(value1, *result);

    // test replace

    ASSERT_TRUE(tables.replace("table_level_4", value2));
    result = tables.find("table_level_4");
    ASSERT_TRUE(result);
    ASSERT_EQ(value2, *result);

    ASSERT_TRUE(!tables.replace("empty", openvdb::ax::codegen::Value::Invalid()));
}
