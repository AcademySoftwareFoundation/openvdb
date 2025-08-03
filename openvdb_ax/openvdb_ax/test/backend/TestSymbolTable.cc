// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "util.h"

#include <openvdb_ax/codegen/SymbolTable.h>

#include <cppunit/extensions/HelperMacros.h>

template <typename T>
using LLVMType = openvdb::ax::codegen::LLVMType<T>;

class TestSymbolTable : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestSymbolTable);
    CPPUNIT_TEST(testSingleTable);
    CPPUNIT_TEST(testTableBlocks);
    CPPUNIT_TEST_SUITE_END();

    void testSingleTable();
    void testTableBlocks();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestSymbolTable);

void
TestSymbolTable::testSingleTable()
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    llvm::Value* value1 = builder.CreateAlloca(type);
    llvm::Value* value2 = builder.CreateAlloca(type);
    CPPUNIT_ASSERT(value1);
    CPPUNIT_ASSERT(value2);

    openvdb::ax::codegen::SymbolTable<llvm::Value*> table;
    CPPUNIT_ASSERT(table.map().empty());

    CPPUNIT_ASSERT(table.insert("test", value1));
    CPPUNIT_ASSERT(!table.insert("test", nullptr));
    CPPUNIT_ASSERT(table.exists("test"));
    CPPUNIT_ASSERT_EQUAL(value1, *table.get("test"));

    table.clear();
    CPPUNIT_ASSERT(table.map().empty());
    CPPUNIT_ASSERT(!table.exists("test"));

    CPPUNIT_ASSERT(table.insert("test", value1));
    CPPUNIT_ASSERT(table.replace("test", value2));
    CPPUNIT_ASSERT(!table.replace("other", value2));

    CPPUNIT_ASSERT(table.exists("test"));
    CPPUNIT_ASSERT(table.exists("other"));

    CPPUNIT_ASSERT_EQUAL(value2, *table.get("test"));
    CPPUNIT_ASSERT_EQUAL(value2, *table.get("other"));
}

void
TestSymbolTable::testTableBlocks()
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    openvdb::ax::codegen::Value value1 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value2 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value3 = openvdb::ax::codegen::Value::Alloc(builder, type);
    openvdb::ax::codegen::Value value4 = openvdb::ax::codegen::Value::Alloc(builder, type);
    CPPUNIT_ASSERT(value1);
    CPPUNIT_ASSERT(value2);
    CPPUNIT_ASSERT(value3);
    CPPUNIT_ASSERT(value4);

    // test table insertion and erase

    openvdb::ax::codegen::SymbolTableBlocks tables;
    openvdb::ax::codegen::SymbolTable<openvdb::ax::codegen::Value>* table1 = tables.getOrInsert(0);
    openvdb::ax::codegen::SymbolTable<openvdb::ax::codegen::Value>* table2 = tables.getOrInsert(1);

    CPPUNIT_ASSERT_EQUAL(table1, tables.get(0));
    CPPUNIT_ASSERT_EQUAL(table2, tables.get(1));

    tables.getOrInsert(2);
    tables.getOrInsert(3);
    tables.getOrInsert(5);

    CPPUNIT_ASSERT(tables.get(4) == nullptr);
    CPPUNIT_ASSERT(tables.erase(5));
    CPPUNIT_ASSERT(tables.erase(3));
    CPPUNIT_ASSERT(tables.erase(2));

    table1->insert("top1", value1);
    table1->insert("top2", value2);

    // test find methods

    const openvdb::ax::codegen::Value* result = tables.find("top1");
    CPPUNIT_ASSERT_EQUAL(value1, *result);
    result = tables.find("top2");
    CPPUNIT_ASSERT_EQUAL(value2, *result);

    table1 = tables.getOrInsert(3);
    table2 = tables.getOrInsert(5);
    tables.getOrInsert(6);

    // test multi table find methods

    table1->insert("table_level_2", value3);
    table2->insert("table_level_4", value4);

    // test find second nested value

    result = tables.find("table_level_2", 1);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_2", 2);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_2", 3);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value3, *result);

    // test find fourth nested value

    result = tables.find("table_level_4", 1);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_4", 4);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_4", 5);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value4, *result);

    result = tables.find("table_level_4", 10000);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value4, *result);

    // test find fourth nested value with matching name

    table1->insert("table_level_4", value1);

    result = tables.find("table_level_4");
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value4, *result);

    result = tables.find("table_level_4", 5);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value4, *result);

    result = tables.find("table_level_4", 4);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value1, *result);

    // test replace

    CPPUNIT_ASSERT(tables.replace("table_level_4", value2));
    result = tables.find("table_level_4");
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(value2, *result);

    CPPUNIT_ASSERT(!tables.replace("empty", openvdb::ax::codegen::Value::Invalid()));
}
