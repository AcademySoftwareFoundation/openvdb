///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

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

    openvdb::ax::codegen::SymbolTable table;
    CPPUNIT_ASSERT(table.map().empty());

    CPPUNIT_ASSERT(table.insert("test", value1));
    CPPUNIT_ASSERT(!table.insert("test", nullptr));
    CPPUNIT_ASSERT(table.exists("test"));
    CPPUNIT_ASSERT_EQUAL(value1, table.get("test"));

    table.clear();
    CPPUNIT_ASSERT(table.map().empty());
    CPPUNIT_ASSERT(!table.exists("test"));

    CPPUNIT_ASSERT(table.insert("test", value1));
    CPPUNIT_ASSERT(table.replace("test", value2));
    CPPUNIT_ASSERT(!table.replace("other", value2));

    CPPUNIT_ASSERT(table.exists("test"));
    CPPUNIT_ASSERT(table.exists("other"));

    CPPUNIT_ASSERT_EQUAL(value2, table.get("test"));
    CPPUNIT_ASSERT_EQUAL(value2, table.get("other"));
}

void
TestSymbolTable::testTableBlocks()
{
    unittest_util::LLVMState state;
    llvm::IRBuilder<> builder(state.scratchBlock());

    llvm::Type* type = LLVMType<float>::get(state.context());

    llvm::Value* value1 = builder.CreateAlloca(type);
    llvm::Value* value2 = builder.CreateAlloca(type);
    llvm::Value* value3 = builder.CreateAlloca(type);
    llvm::Value* value4 = builder.CreateAlloca(type);
    CPPUNIT_ASSERT(value1);
    CPPUNIT_ASSERT(value2);
    CPPUNIT_ASSERT(value3);
    CPPUNIT_ASSERT(value4);

    // test table insertion and erase

    openvdb::ax::codegen::SymbolTableBlocks tables;
    openvdb::ax::codegen::SymbolTable* table1 = &(tables.globals());
    openvdb::ax::codegen::SymbolTable* table2 = tables.getOrInsert(0);
    CPPUNIT_ASSERT_EQUAL(table1, table2);

    table2 = &(tables.get(0));
    CPPUNIT_ASSERT_EQUAL(table1, table2);

    CPPUNIT_ASSERT_THROW(tables.erase(0), std::runtime_error);

    tables.getOrInsert(1);
    tables.getOrInsert(2);
    tables.getOrInsert(4);

    CPPUNIT_ASSERT_THROW(tables.get(3), std::runtime_error);
    CPPUNIT_ASSERT(tables.erase(4));
    CPPUNIT_ASSERT(tables.erase(2));
    CPPUNIT_ASSERT(tables.erase(1));

    tables.globals().insert("global1", value1);
    tables.globals().insert("global2", value2);

    // test find methods

    llvm::Value* result = tables.find("global1");
    CPPUNIT_ASSERT_EQUAL(value1, result);
    result = tables.find("global2");
    CPPUNIT_ASSERT_EQUAL(value2, result);

    table1 = tables.getOrInsert(2);
    table2 = tables.getOrInsert(4);
    tables.getOrInsert(5);

    // test multi table find methods

    table1->insert("table_level_2", value3);
    table2->insert("table_level_4", value4);

    // test find second nested value

    result = tables.find("table_level_2", 0);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_2", 1);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_2", 2);
    CPPUNIT_ASSERT_EQUAL(value3, result);

    // test find fourth nested value

    result = tables.find("table_level_4", 0);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_4", 3);
    CPPUNIT_ASSERT(!result);
    result = tables.find("table_level_4", 4);
    CPPUNIT_ASSERT_EQUAL(value4, result);

    result = tables.find("table_level_4", 10000);
    CPPUNIT_ASSERT_EQUAL(value4, result);

    // test find fourth nested value with matching global name

    tables.globals().insert("table_level_4", value1);

    result = tables.find("table_level_4");
    CPPUNIT_ASSERT_EQUAL(value4, result);

    result = tables.find("table_level_4", 4);
    CPPUNIT_ASSERT_EQUAL(value4, result);

    result = tables.find("table_level_4", 3);
    CPPUNIT_ASSERT_EQUAL(value1, result);

    // test replace

    CPPUNIT_ASSERT(tables.replace("table_level_4", value2));
    result = tables.find("table_level_4");
    CPPUNIT_ASSERT_EQUAL(value2, result);

    // test global was not replaced

    result = tables.find("table_level_4", 0);
    CPPUNIT_ASSERT_EQUAL(value1, result);

    CPPUNIT_ASSERT(!tables.replace("empty", nullptr));
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
