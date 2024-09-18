// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/codegen/Types.h>
#include <openvdb_ax/codegen/String.h>
#include <openvdb_ax/codegen/Functions.h>
#include <openvdb_ax/codegen/FunctionRegistry.h>

#include <openvdb/util/Assert.h>

#include <cppunit/extensions/HelperMacros.h>

#include <numeric> // iota

using String = openvdb::ax::codegen::String;

class TestStringIR : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestStringIR);
    CPPUNIT_TEST(testStringImpl);
    CPPUNIT_TEST(testStringStringIR);
    CPPUNIT_TEST(testStringAssignIR);
    CPPUNIT_TEST(testStringAddIR);
    CPPUNIT_TEST(testStringClearIR);
    CPPUNIT_TEST_SUITE_END();

    void testStringImpl();
    void testStringStringIR();
    void testStringAssignIR();
    void testStringAddIR();
    void testStringClearIR();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestStringIR);

template <size_t N>
inline std::array<char, N+1> fillCharArray(char c)
{
    std::array<char, N+1> arr;
    std::fill(std::begin(arr),std::end(arr),c);
    arr[N] = '\0';
    return arr;
}

template <size_t N>
inline std::array<char, N+1> fillCharArrayIota(char c = char(1))
{
    std::array<char, N+1> arr;
    std::iota(std::begin(arr), std::end(arr), c); // start the fill at char(1) as char(0) == '\0'
    arr[N] = '\0';
    return arr;
}

void
TestStringIR::testStringImpl()
{
    // Test the C++ implementation

    // Construction

    {
        String a;
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL(std::string(""), a.str());
    }

    {
        String a("");
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL(std::string(""), a.str());
    }

    {
        auto arr = fillCharArray<String::SSO_LENGTH-1>('0');
        String a(arr.data());
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH-1), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT(a.c_str() != arr.data());
        CPPUNIT_ASSERT_EQUAL(std::string(arr.data(), String::SSO_LENGTH-1), a.str());
    }

    {
        auto arr = fillCharArray<String::SSO_LENGTH>('0');
        String a(arr.data());
        const char* c(a);
        CPPUNIT_ASSERT(!a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT(a.c_str() != arr.data());
        CPPUNIT_ASSERT_EQUAL(std::string(arr.data(), String::SSO_LENGTH), a.str());
    }

    {
        const char* arr = "foo";
        String a(arr);
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(3), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT(a.c_str() != arr);
        CPPUNIT_ASSERT_EQUAL(std::string(arr), a.str());
    }

    {
        const std::string s(String::SSO_LENGTH-1, '0');
        String a(s);
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH-1), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT(a.c_str() != s.c_str());
        CPPUNIT_ASSERT_EQUAL(s, a.str());
    }

    {
        const std::string s(String::SSO_LENGTH, '0');
        String a(s);
        const char* c(a);
        CPPUNIT_ASSERT(!a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT(a.c_str() != s.c_str());
        CPPUNIT_ASSERT_EQUAL(s, a.str());
    }

    // Copy/Assign

    {
        String a("foo");
        String b(a);
        const char* c(b);
        CPPUNIT_ASSERT(b.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(3), b.size());
        CPPUNIT_ASSERT_EQUAL(c, b.c_str());
        CPPUNIT_ASSERT(b.c_str() != a.c_str());
        CPPUNIT_ASSERT_EQUAL('f', b.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', b.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', b.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', b.c_str()[3]);
        CPPUNIT_ASSERT_EQUAL(std::string("foo"), b.str());
    }

    {
        auto arr = fillCharArray<String::SSO_LENGTH>('a');
        String a(arr.data());
        String b(a);
        const char* c(b);
        CPPUNIT_ASSERT(!b.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), b.size());
        CPPUNIT_ASSERT_EQUAL(c, b.c_str());
        CPPUNIT_ASSERT(b.c_str() != a.c_str());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH); ++i) {
            CPPUNIT_ASSERT_EQUAL('a', b.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('\0', b.c_str()[int64_t(String::SSO_LENGTH)]);
        CPPUNIT_ASSERT_EQUAL(std::string(arr.data(), String::SSO_LENGTH), b.str());
    }

    {
        auto arr = fillCharArray<String::SSO_LENGTH>('a');
        String a(arr.data());
        String b("");
        CPPUNIT_ASSERT(b.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), b.size());
        b = a;
        const char* c(b);
        CPPUNIT_ASSERT(!b.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), b.size());
        CPPUNIT_ASSERT_EQUAL(c, b.c_str());
        CPPUNIT_ASSERT(b.c_str() != a.c_str());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH); ++i) {
            CPPUNIT_ASSERT_EQUAL('a', b.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('\0', b.c_str()[int64_t(String::SSO_LENGTH)]);
        CPPUNIT_ASSERT_EQUAL(std::string(arr.data(), String::SSO_LENGTH), b.str());
    }

    // From std::string

    {
        String a("");
        a = std::string("foo");
        const char* c(a);
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT_EQUAL(int64_t(3), a.size());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL('f', a.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', a.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', a.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[3]);
    }

    {
        const std::string s(String::SSO_LENGTH, 'a');
        String a("");
        a = s;
        const char* c(a);
        CPPUNIT_ASSERT(!a.isLocal());
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT(c != s.c_str());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), a.size());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH); ++i) {
            CPPUNIT_ASSERT_EQUAL('a', a.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[int64_t(String::SSO_LENGTH)]);
    }

    // Add

    {
        String a(""), b("");
        a = b + b;
        CPPUNIT_ASSERT(a.isLocal());
        const char* c(a);
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT(c != b.c_str());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
    }

    {
        CPPUNIT_ASSERT(String::SSO_LENGTH >= 2);
        auto arr = fillCharArray<String::SSO_LENGTH-2>('a');
        String a(""), b1(arr.data()), b2("b");
        a = b1 + b2;
        CPPUNIT_ASSERT(a.isLocal());
        const char* c(a);
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT(c != b1.c_str());
        CPPUNIT_ASSERT(c != b2.c_str());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH-1), a.size());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH-2); ++i) {
            CPPUNIT_ASSERT_EQUAL('a', a.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('b', a.c_str()[int64_t(String::SSO_LENGTH-2)]);
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[int64_t(String::SSO_LENGTH-1)]);
    }

    {
        CPPUNIT_ASSERT(String::SSO_LENGTH >= 2);
        auto arr = fillCharArray<String::SSO_LENGTH-1>('a');
        String a(""), b1(arr.data()), b2("b");
        a = b1 + b2;
        CPPUNIT_ASSERT(!a.isLocal());
        const char* c(a);
        CPPUNIT_ASSERT(c);
        CPPUNIT_ASSERT(c != b1.c_str());
        CPPUNIT_ASSERT(c != b2.c_str());
        CPPUNIT_ASSERT_EQUAL(c, a.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), a.size());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH-1); ++i) {
            CPPUNIT_ASSERT_EQUAL('a', a.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('b', a.c_str()[int64_t(String::SSO_LENGTH-1)]);
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[int64_t(String::SSO_LENGTH)]);
    }
}

void
TestStringIR::testStringStringIR()
{
    static auto setInvalidString = [](String& S) {
#if defined(__GNUC__) && !defined(__clang__)
#if OPENVDB_CHECK_GCC(8, 0)
        _Pragma("GCC diagnostic push")
        _Pragma("GCC diagnostic ignored \"-Wclass-memaccess\"")
#endif
#endif
        // zero out the data held by a String object (expected to not hold heap memory).
        // This is used to test the IR methods work as expected with the allocated, but
        // uninitialized stack mem from the compute generator
        OPENVDB_ASSERT(S.isLocal());
        std::memset(&S, 0, sizeof(String)); // uninit string, invalid class memory
#if defined(__GNUC__) && !defined(__clang__)
#if OPENVDB_CHECK_GCC(8, 0)
        _Pragma("GCC diagnostic pop")
#endif
#endif
    };

    // Test the String IR in StringFunctions.cc

    unittest_util::LLVMState state;
    llvm::Module& M = state.module();
    openvdb::ax::FunctionOptions opts;
    // This test does not test the C bindings and will fail if they are being
    // used as the function addressed wont be linked into the EE.
    // @todo  Expose the global function address binding in Compiler.cc (without
    //   exposing LLVM headers)
    CPPUNIT_ASSERT(opts.mPrioritiseIR);
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // insert all the string::string functions into a module
    const openvdb::ax::codegen::FunctionGroup* FG =
        reg->getOrInsert("string::string", opts, true);
    CPPUNIT_ASSERT(FG);
    for (auto& F : FG->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // also insert string::clear for the malloc tests
    const openvdb::ax::codegen::FunctionGroup* SC =
        reg->getOrInsert("string::clear", opts, true);
    CPPUNIT_ASSERT(SC);
    for (auto& F : SC->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // JIT gen the functions
    auto EE = state.EE();
    CPPUNIT_ASSERT(EE);

    // get the string::clear function for later
    CPPUNIT_ASSERT(!SC->list().empty());
    const int64_t addressOfClear =
        EE->getFunctionAddress(SC->list()[0]->symbol());
    CPPUNIT_ASSERT(addressOfClear);
    auto freeString =
        reinterpret_cast<std::add_pointer<void(String*)>::type>(addressOfClear);


    // Test the IR for each string function. These match the signatures
    // defined in StringFunctions.cc

    int64_t address = 0;

    // Test string::string

    CPPUNIT_ASSERT_EQUAL(size_t(2), FG->list().size()); // expects 2 signatures

    // init string to default
    address = EE->getFunctionAddress(FG->list()[0]->symbol());
    CPPUNIT_ASSERT(address);
    {
        auto F = reinterpret_cast<std::add_pointer<void(String*)>::type>(address);
        CPPUNIT_ASSERT(F);
        /// @note the IR version of string::string should handle the case where
        ///   the string memory is completely uninitialized
        String input;
        setInvalidString(input); // uninit string, invalid class memory
        F(&input); // run function
        CPPUNIT_ASSERT(input.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), input.size());
        CPPUNIT_ASSERT_EQUAL('\0', input.c_str()[0]);
    }

    // init string from char*
    address = EE->getFunctionAddress(FG->list()[1]->symbol());
    CPPUNIT_ASSERT(address);
    {
        // This test requires SSO_LENGTH > 2
        CPPUNIT_ASSERT(String::SSO_LENGTH >= 2);

        auto F = reinterpret_cast<std::add_pointer<void(String*, const char*)>::type>(address);
        CPPUNIT_ASSERT(F);

        // test to SSO storage
        {
            /// @note the IR version of string::string should handle the case where
            ///   the string memory is completely uninitialized
            String input;
            setInvalidString(input); // uninit string, invalid class memory
            auto arr = fillCharArrayIota<String::SSO_LENGTH-1>();
            const char* data = arr.data();
            CPPUNIT_ASSERT_EQUAL(std::size_t(String::SSO_LENGTH-1), std::strlen(data));

            F(&input, data); // run function

            const char* c(input);
            CPPUNIT_ASSERT(input.isLocal());
            CPPUNIT_ASSERT(c);
            CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH-1), input.size());
            CPPUNIT_ASSERT_EQUAL(c, input.c_str());
            CPPUNIT_ASSERT(input.c_str() != data);
            for (int64_t i = 0; i < int64_t(String::SSO_LENGTH-1); ++i) {
                CPPUNIT_ASSERT_EQUAL(char(i+1), input.c_str()[i]);
            }
            CPPUNIT_ASSERT_EQUAL('\0', input.c_str()[String::SSO_LENGTH-1]);
        }

        // test malloc
        // @warning  If using the IR functions this will return memory allocated by LLVM.
        //   If we've statically linked malloc (i.e. /MT or /MTd on Windows) then
        //   we can't free this ourselves and need to make sure we call string::clear to
        //   free the malloced char array
        {
            /// @note the IR version of string::string should handle the case where
            ///   the string memory is completely uninitialized
            String input;
            // Also test that string::string mallocs when size >= String::SSO_LENGTH
            setInvalidString(input); // uninit string, invalid class memory
            auto arr = fillCharArrayIota<String::SSO_LENGTH>();
            const char* data = arr.data();
            CPPUNIT_ASSERT_EQUAL(std::size_t(String::SSO_LENGTH), std::strlen(data));

            F(&input, data); // run function

            const char* c(input);
            CPPUNIT_ASSERT(!input.isLocal());
            CPPUNIT_ASSERT(c);
            CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), input.size());
            CPPUNIT_ASSERT_EQUAL(c, input.c_str());
            CPPUNIT_ASSERT(input.c_str() != data);
            for (int64_t i = 0; i < int64_t(String::SSO_LENGTH); ++i) {
                CPPUNIT_ASSERT_EQUAL(char(i+1), input.c_str()[i]);
            }
            CPPUNIT_ASSERT_EQUAL('\0', input.c_str()[String::SSO_LENGTH]);

            // free through string::clear
            freeString(&input);
        }
    }
}


void
TestStringIR::testStringAssignIR()
{
    // Test the String IR in StringFunctions.cc

    unittest_util::LLVMState state;
    llvm::Module& M = state.module();
    openvdb::ax::FunctionOptions opts;
    // This test does not test the C bindings and will fail if they are being
    // used as the function addressed wont be linked into the EE.
    // @todo  Expose the global function address binding in Compiler.cc (without
    //   exposing LLVM headers)
    CPPUNIT_ASSERT(opts.mPrioritiseIR);
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // insert all the string::op= functions into a module
    const openvdb::ax::codegen::FunctionGroup* FG =
        reg->getOrInsert("string::op=", opts, true);
    CPPUNIT_ASSERT(FG);
    for (auto& F : FG->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // also insert string::clear for the malloc tests
    const openvdb::ax::codegen::FunctionGroup* SC =
        reg->getOrInsert("string::clear", opts, true);
    CPPUNIT_ASSERT(SC);
    for (auto& F : SC->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // JIT gen the functions
    auto EE = state.EE();
    CPPUNIT_ASSERT(EE);

    // get the string::clear function for later
    CPPUNIT_ASSERT(!SC->list().empty());
    const int64_t addressOfClear =
        EE->getFunctionAddress(SC->list()[0]->symbol());
    CPPUNIT_ASSERT(addressOfClear);
    auto freeString =
        reinterpret_cast<std::add_pointer<void(String*)>::type>(addressOfClear);

    // Test the IR for each string function. These match the signatures
    // defined in StringFunctions.cc

    // Test string::op=

    CPPUNIT_ASSERT_EQUAL(size_t(1), FG->list().size()); // expects 1 signature

    const int64_t address = EE->getFunctionAddress(FG->list()[0]->symbol());
    CPPUNIT_ASSERT(address);
    auto F = reinterpret_cast<std::add_pointer<void(String*, const String*)>::type>(address);
    CPPUNIT_ASSERT(F);

    // copy from null string
    {
        String dest("foo"), src("");
        F(&dest, &src); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != src.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), dest.size());
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[0]);
    }

    // copy to null string
    {
        String dest(""), src("foo");
        F(&dest, &src); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != src.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(3), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[3]);
    }


    // copy both local
    {
        String dest("bar"), src("foo");
        F(&dest, &src); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != src.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(3), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[3]);
    }

    // copy to malloced
    // @warning  We can't run this test if we've statically linked the allocaor
    //   as we're passing memory allocated here to llvm which will free it
    //   during the assignment
    // @todo enable if we detect we're using a dynamic malloc
    /*
    {
        auto arr = fillCharArrayIota<String::SSO_LENGTH>();
        String dest(arr.data()), src("foo");
        CPPUNIT_ASSERT(!dest.isLocal());
        CPPUNIT_ASSERT(src.isLocal());

        F(&dest, &src); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != src.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(3), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[3]);
    }
    */


    // copy from malloced
    // @warning  If using the IR functions this will return memory allocated by LLVM.
    //   If we've statically linked malloc (i.e. /MT or /MTd on Windows) then
    //   we can't free this ourselves and need to make sure we call string::clear to
    //   free the malloced char array
    {
        auto arr = fillCharArrayIota<String::SSO_LENGTH>();
        String dest("foo"), src(arr.data());
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(!src.isLocal());

        F(&dest, &src); // run function
        CPPUNIT_ASSERT(!dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != src.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH), dest.size());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH-1); ++i) {
            CPPUNIT_ASSERT_EQUAL(char(i+1), dest.c_str()[i]);
        }
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[int64_t(String::SSO_LENGTH)]);

        // free through string::clear
        freeString(&dest);
    }
}


void TestStringIR::testStringAddIR()
{
    // Test the String IR in StringFunctions.cc

    unittest_util::LLVMState state;
    llvm::Module& M = state.module();
    openvdb::ax::FunctionOptions opts;
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // insert all the string::op+ functions into a module
    const openvdb::ax::codegen::FunctionGroup* FG =
        reg->getOrInsert("string::op+", opts, true);
    CPPUNIT_ASSERT(FG);
    for (auto& F : FG->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // also insert string::clear for the malloc tests
    const openvdb::ax::codegen::FunctionGroup* SC =
        reg->getOrInsert("string::clear", opts, true);
    CPPUNIT_ASSERT(SC);
    for (auto& F : SC->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // JIT gen the functions
    auto EE = state.EE();
    CPPUNIT_ASSERT(EE);

    // get the string::clear function for later
    CPPUNIT_ASSERT(!SC->list().empty());
    const int64_t addressOfClear =
        EE->getFunctionAddress(SC->list()[0]->symbol());
    CPPUNIT_ASSERT(addressOfClear);
    auto freeString =
        reinterpret_cast<std::add_pointer<void(String*)>::type>(addressOfClear);

    // Test the IR for each string function. These match the signatures
    // defined in StringFunctions.cc

    // Test string::op+
    // @note the binary ad op sets the first argument which it expects to
    //   already be initialized correctly.

    CPPUNIT_ASSERT_EQUAL(size_t(1), FG->list().size()); // expects 1 signature

    const int64_t address = EE->getFunctionAddress(FG->list()[0]->symbol());
    CPPUNIT_ASSERT(address);
    auto F = reinterpret_cast<std::add_pointer<void(String*, const String*, const String*)>::type>(address);
    CPPUNIT_ASSERT(F);

    // add from null strings
    {
        String dest("foo"), rhs(""), lhs("");
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), dest.size());
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[0]);
    }

    // add from lhs null string
    {
        String dest(""), lhs(""), rhs("foo");
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(3), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[3]);
    }

    // add from rhs null string
    {
        String dest(""), lhs("foo"), rhs("");
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(3), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[3]);
    }

    // add from both local strings
    {
        String dest(""), lhs("foo"), rhs(" bar");
        CPPUNIT_ASSERT(lhs.isLocal());
        CPPUNIT_ASSERT(rhs.isLocal());
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(7), dest.size());
        CPPUNIT_ASSERT_EQUAL('f', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('o', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL(' ', dest.c_str()[3]);
        CPPUNIT_ASSERT_EQUAL('b', dest.c_str()[4]);
        CPPUNIT_ASSERT_EQUAL('a', dest.c_str()[5]);
        CPPUNIT_ASSERT_EQUAL('r', dest.c_str()[6]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[7]);
    }

    // add from local rhs, malloced lhs
    // @warning  If using the IR functions this will return memory allocated by LLVM.
    //   If we've statically linked malloc (i.e. /MT or /MTd on Windows) then
    //   we can't free this ourselves and need to make sure we call string::clear to
    //   free the malloced char array
    {
        auto arr = fillCharArrayIota<String::SSO_LENGTH>();
        String dest(""), lhs(arr.data()), rhs(" bar");
        CPPUNIT_ASSERT(!lhs.isLocal());
        CPPUNIT_ASSERT(rhs.isLocal());
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(!dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH+4), dest.size());
        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH-1); ++i) {
            CPPUNIT_ASSERT_EQUAL(char(i+1), dest.c_str()[i]);
        }

        CPPUNIT_ASSERT_EQUAL(' ', dest.c_str()[int64_t(String::SSO_LENGTH+0)]);
        CPPUNIT_ASSERT_EQUAL('b', dest.c_str()[int64_t(String::SSO_LENGTH+1)]);
        CPPUNIT_ASSERT_EQUAL('a', dest.c_str()[int64_t(String::SSO_LENGTH+2)]);
        CPPUNIT_ASSERT_EQUAL('r', dest.c_str()[int64_t(String::SSO_LENGTH+3)]);
        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[int64_t(String::SSO_LENGTH+4)]);
        // free through string::clear
        freeString(&dest);
    }

    // add from local lhs, malloced rhs
    // @warning  If using the IR functions this will return memory allocated by LLVM.
    //   If we've statically linked malloc (i.e. /MT or /MTd on Windows) then
    //   we can't free this ourselves and need to make sure we call string::clear to
    //   free the malloced char array
    {
        auto arr = fillCharArrayIota<String::SSO_LENGTH>();
        String dest(""), lhs(" bar"), rhs(arr.data());
        CPPUNIT_ASSERT(lhs.isLocal());
        CPPUNIT_ASSERT(!rhs.isLocal());
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(!dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH+4), dest.size());

        CPPUNIT_ASSERT_EQUAL(' ', dest.c_str()[0]);
        CPPUNIT_ASSERT_EQUAL('b', dest.c_str()[1]);
        CPPUNIT_ASSERT_EQUAL('a', dest.c_str()[2]);
        CPPUNIT_ASSERT_EQUAL('r', dest.c_str()[3]);
        for (int64_t i = 4; i < int64_t(String::SSO_LENGTH+4); ++i) {
            CPPUNIT_ASSERT_EQUAL(char(i-3), dest.c_str()[i]); // i=3 as we start iterating at 4
        }

        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[int64_t(String::SSO_LENGTH+4)]);
        // free through string::clear
        freeString(&dest);
    }

    // add from malloced rhs, malloced lhs
    // @warning  If using the IR functions this will return memory allocated by LLVM.
    //   If we've statically linked malloc (i.e. /MT or /MTd on Windows) then
    //   we can't free this ourselves and need to make sure we call string::clear to
    //   free the malloced char array
    {
        auto arr1 = fillCharArrayIota<String::SSO_LENGTH>();
        auto arr2 = fillCharArrayIota<String::SSO_LENGTH>(char(1+String::SSO_LENGTH));
        String lhs(arr1.data());
        String rhs(arr2.data());
        CPPUNIT_ASSERT(!lhs.isLocal());
        CPPUNIT_ASSERT(!rhs.isLocal());
        arr1.fill('0');
        arr2.fill('0');
        String dest("");
        F(&dest, &lhs, &rhs); // run function
        CPPUNIT_ASSERT(!dest.isLocal());
        CPPUNIT_ASSERT(dest.c_str() != rhs.c_str());
        CPPUNIT_ASSERT(dest.c_str() != lhs.c_str());
        CPPUNIT_ASSERT_EQUAL(int64_t(String::SSO_LENGTH*2), dest.size());

        for (int64_t i = 0; i < int64_t(String::SSO_LENGTH*2); ++i) {
            CPPUNIT_ASSERT_EQUAL(char(i+1), dest.c_str()[i]);
        }

        CPPUNIT_ASSERT_EQUAL('\0', dest.c_str()[int64_t(String::SSO_LENGTH*2)]);
        // free through string::clear
        freeString(&dest);
    }
}


void TestStringIR::testStringClearIR()
{
    // Test the String IR in StringFunctions.cc

    unittest_util::LLVMState state;
    llvm::Module& M = state.module();
    openvdb::ax::FunctionOptions opts;
    openvdb::ax::codegen::FunctionRegistry::UniquePtr reg =
        openvdb::ax::codegen::createDefaultRegistry(&opts);

    // insert all the string::clear functions into a module
    const openvdb::ax::codegen::FunctionGroup* FG =
        reg->getOrInsert("string::clear", opts, true);
    CPPUNIT_ASSERT(FG);
    for (auto& F : FG->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // insert all the string::string functions into a module
    // for the malloc clear tests
    const openvdb::ax::codegen::FunctionGroup* MFG =
        reg->getOrInsert("string::string", opts, true);
    CPPUNIT_ASSERT(MFG);
    for (auto& F : MFG->list()) {
        llvm::Function* LF = F->create(M);
        CPPUNIT_ASSERT(LF);
    }

    // JIT gen the functions
    auto EE = state.EE();
    CPPUNIT_ASSERT(EE);

    // Test the IR for each string function. These match the signatures
    // defined in StringFunctions.cc

    // Test string::clear

    CPPUNIT_ASSERT_EQUAL(size_t(1), FG->list().size()); // expects 1 signature

    const int64_t address = EE->getFunctionAddress(FG->list()[0]->symbol());
    CPPUNIT_ASSERT(address);
    auto F = reinterpret_cast<std::add_pointer<void(String*)>::type>(address);
    CPPUNIT_ASSERT(F);

    // clear empty
    {
        String a;
        F(&a); // run function
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
    }

    // clear local
    {
        String a("foo");
        CPPUNIT_ASSERT(a.isLocal());
        F(&a); // run function
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
    }

    // clear malloced
    // @warning  We can't run this test if we've statically linked the allocaor
    //   as we're passing memory allocated here to llvm which will free it
    //   during clear
    // @todo enable if we detect we're using a dynamic malloc
    /*
    {
        auto arr = fillCharArray<String::SSO_LENGTH>('0');
        String a(arr.data());
        CPPUNIT_ASSERT(!a.isLocal());
        F(&a); // run function
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
    }
    */

    // malloc then clear
    // @warning  If using the IR functions this will attempt to free memory that
    //  we've allocated. To test clear we need to create the string through the
    //  LLVM binding then free.
    {
        // get the string::clear function which takes a char* initializer
        CPPUNIT_ASSERT(!MFG->list().empty());
        const int64_t addressOfStringCtr =
            EE->getFunctionAddress(MFG->list()[1]->symbol());
        CPPUNIT_ASSERT(addressOfStringCtr);
        auto createString =
            reinterpret_cast<std::add_pointer<void(String*, const char*)>::type>(addressOfStringCtr);

        auto arr = fillCharArray<String::SSO_LENGTH>('0');

        String a;
        createString(&a, arr.data());
        CPPUNIT_ASSERT(!a.isLocal());
        F(&a); // run function
        CPPUNIT_ASSERT(a.isLocal());
        CPPUNIT_ASSERT_EQUAL(int64_t(0), a.size());
        CPPUNIT_ASSERT_EQUAL('\0', a.c_str()[0]);
    }
}
