// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

    openvdb::ax::codegen::SymbolTable table;
    ASSERT_TRUE(table.map().empty());

    ASSERT_TRUE(table.insert("test", value1));
    ASSERT_TRUE(!table.insert("test", nullptr));
    ASSERT_TRUE(table.exists("test"));
    ASSERT_EQ(value1, table.get("test"));

    table.clear();
    ASSERT_TRUE(table.map().empty());
    ASSERT_TRUE(!table.exists("test"));

    ASSERT_TRUE(table.insert("test", value1));
    ASSERT_TRUE(table.replace("test", value2));
    ASSERT_TRUE(!table.replace("other", value2));

    ASSERT_TRUE(table.exists("test"));
    ASSERT_TRUE(table.exists("other"));

    ASSERT_EQ(value2, table.get("test"));
    ASSERT_EQ(value2, table.get("other"));
}

TEST_F(TestSymbolTable, testTableBlocks)
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    llvm::Value* value1 = builder.CreateAlloca(type);
    llvm::Value* value2 = builder.CreateAlloca(type);
    llvm::Value* value3 = builder.CreateAlloca(type);
    llvm::Value* value4 = builder.CreateAlloca(type);
    ASSERT_TRUE(value1);
    ASSERT_TRUE(value2);
    ASSERT_TRUE(value3);
    ASSERT_TRUE(value4);

    // test table insertion and erase

    openvdb::ax::codegen::SymbolTableBlocks tables;
    openvdb::ax::codegen::SymbolTable* table1 = &(tables.globals());
    openvdb::ax::codegen::SymbolTable* table2 = tables.getOrInsert(0);
    ASSERT_EQ(table1, table2);

    table2 = tables.get(0);
    ASSERT_EQ(table1, table2);

    ASSERT_THROW(tables.erase(0), std::runtime_error);

    tables.getOrInsert(1);
    tables.getOrInsert(2);
    tables.getOrInsert(4);

    ASSERT_TRUE(tables.get(3) == nullptr);
    ASSERT_TRUE(tables.erase(4));
    ASSERT_TRUE(tables.erase(2));
    ASSERT_TRUE(tables.erase(1));

    tables.globals().insert("global1", value1);
    tables.globals().insert("global2", value2);

    // test find methods

    llvm::Value* result = tables.find("global1");
    ASSERT_EQ(value1, result);
    result = tables.find("global2");
    ASSERT_EQ(value2, result);

    table1 = tables.getOrInsert(2);
    table2 = tables.getOrInsert(4);
    tables.getOrInsert(5);

    // test multi table find methods

    table1->insert("table_level_2", value3);
    table2->insert("table_level_4", value4);

    // test find second nested value

    result = tables.find("table_level_2", 0);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_2", 1);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_2", 2);
    ASSERT_EQ(value3, result);

    // test find fourth nested value

    result = tables.find("table_level_4", 0);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_4", 3);
    ASSERT_TRUE(!result);
    result = tables.find("table_level_4", 4);
    ASSERT_EQ(value4, result);

    result = tables.find("table_level_4", 10000);
    ASSERT_EQ(value4, result);

    // test find fourth nested value with matching global name

    tables.globals().insert("table_level_4", value1);

    result = tables.find("table_level_4");
    ASSERT_EQ(value4, result);

    result = tables.find("table_level_4", 4);
    ASSERT_EQ(value4, result);

    result = tables.find("table_level_4", 3);
    ASSERT_EQ(value1, result);

    // test replace

    ASSERT_TRUE(tables.replace("table_level_4", value2));
    result = tables.find("table_level_4");
    ASSERT_EQ(value2, result);

    // test global was not replaced

    result = tables.find("table_level_4", 0);
    ASSERT_EQ(value1, result);

    ASSERT_TRUE(!tables.replace("empty", nullptr));
}

