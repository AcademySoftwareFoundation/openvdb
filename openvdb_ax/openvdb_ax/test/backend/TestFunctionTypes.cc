// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/codegen/FunctionTypes.h>

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <sstream>

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/// @brief  Dummy derived function which implemented types
struct TestFunction : public openvdb::ax::codegen::Function
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::Function>::value,
        "Base class destructor is not virtual");
    TestFunction(const std::vector<llvm::Type*>& types,
          llvm::Type* ret,
          const std::string& symbol)
        : openvdb::ax::codegen::Function(types.size(), symbol)
        , mTypes(types), mRet(ret) {}
    ~TestFunction() override {}
    llvm::Type* types(std::vector<llvm::Type*>& types,
        llvm::LLVMContext&) const override {
        types = mTypes;
        return mRet;
    }
    const std::vector<llvm::Type*> mTypes;
    llvm::Type* mRet;
};

/// @brief  Dummy derived IR function which implemented types and
///         forwards on the generator
struct TestIRFunction : public openvdb::ax::codegen::IRFunctionBase
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::IRFunctionBase>::value,
        "Base class destructor is not virtual");
    TestIRFunction(const std::vector<llvm::Type*>& types,
          llvm::Type* ret,
          const std::string& symbol,
          const openvdb::ax::codegen::IRFunctionBase::GeneratorCb& gen)
        : openvdb::ax::codegen::IRFunctionBase(symbol, gen, types.size())
        , mTypes(types), mRet(ret) {}
    ~TestIRFunction() override {}
    llvm::Type* types(std::vector<llvm::Type*>& types,
        llvm::LLVMContext&) const override {
        types = mTypes;
        return mRet;
    }
    const std::vector<llvm::Type*> mTypes;
    llvm::Type* mRet;
};

/// @brief  static function to test c binding addresses
struct CBindings
{
    static void voidfunc() {}
    static int16_t scalarfunc(bool,int16_t,int32_t,int64_t,float,double) { return int16_t(); }
    static int32_t scalatptsfunc(bool*,int16_t*,int32_t*,int64_t*,float*,double*) { return int32_t(); }
    static int64_t arrayfunc(bool(*)[1],int16_t(*)[2],int32_t(*)[3],int64_t(*)[4],float(*)[5],double(*)[6]) { return int64_t(); }
    static void multiptrfunc(void*, void**, void***, float*, float**, float***) { }
    template <typename Type> static inline Type tmplfunc() { return Type(); }
};

/// @brief  Helper method to finalize a function (with a terminator)
///         If F is nullptr, finalizes the current function
inline llvm::Instruction*
finalizeFunction(llvm::IRBuilder<>& B, llvm::Function* F = nullptr)
{
    auto IP = B.saveIP();
    if (F) {
        if (F->empty()) {
            B.SetInsertPoint(llvm::BasicBlock::Create(B.getContext(), "", F));
        }
        else {
            B.SetInsertPoint(&(F->getEntryBlock()));
        }
    }
    llvm::Instruction* ret = B.CreateRetVoid();
    B.restoreIP(IP);
    return ret;
}

/// @brief  Defines to wrap the verification of IR
#define VERIFY_FUNCTION_IR(Function) { \
    std::string error; llvm::raw_string_ostream os(error); \
    const bool valid = !llvm::verifyFunction(*Function, &os); \
    CPPUNIT_ASSERT_MESSAGE(os.str(), valid); \
}
#define VERIFY_MODULE_IR(Module) { \
    std::string error; llvm::raw_string_ostream os(error); \
    const bool valid = !llvm::verifyModule(*Module, &os); \
    CPPUNIT_ASSERT_MESSAGE(os.str(), valid); \
}
#define VERIFY_MODULE_IR_INVALID(Module) { \
    const bool valid = llvm::verifyModule(*Module); \
    CPPUNIT_ASSERT_MESSAGE("Expected IR to be invalid!", valid); \
}
#define VERIFY_FUNCTION_IR_INVALID(Function) { \
    const bool valid = llvm::verifyFunction(*Function); \
    CPPUNIT_ASSERT_MESSAGE("Expected IR to be invalid!", valid); \
}

inline auto getNumArgFromCallInst(llvm::CallInst* CI)
{
#if LLVM_VERSION_MAJOR >= 14
    return CI->arg_size();
#else
    return CI->getNumArgOperands();
#endif
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

class TestFunctionTypes : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestFunctionTypes);
    CPPUNIT_TEST(testLLVMTypesFromSignature);
    CPPUNIT_TEST(testLLVMFunctionTypeFromSignature);
    CPPUNIT_TEST(testPrintSignature);
    // Test Function::create, Function::types and other base methods
    CPPUNIT_TEST(testFunctionCreate);
    // Test Function::call
    CPPUNIT_TEST(testFunctionCall);
    // Test Function::match
    CPPUNIT_TEST(testFunctionMatch);
    // Test derived CFunctions, mainly CFunction::create and CFunction::types
    CPPUNIT_TEST(testCFunctions);
    // Test C constant folding
    CPPUNIT_TEST(testCFunctionCF);
    // Test derived IR Function, IRFunctionBase::create and IRFunctionBase::call
    CPPUNIT_TEST(testIRFunctions);
    // Test SRET methods for both C and IR functions
    CPPUNIT_TEST(testSRETFunctions);
    CPPUNIT_TEST_SUITE_END();

    void testLLVMTypesFromSignature();
    void testLLVMFunctionTypeFromSignature();
    void testPrintSignature();
    void testFunctionCreate();
    void testFunctionCall();
    void testFunctionMatch();
    void testCFunctions();
    void testCFunctionCF();
    void testIRFunctions();
    void testSRETFunctions();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFunctionTypes);

void
TestFunctionTypes::testLLVMTypesFromSignature()
{
    using openvdb::ax::codegen::llvmTypesFromSignature;

    unittest_util::LLVMState state;
    llvm::Type* type = nullptr;
    std::vector<llvm::Type*> types;

    type = llvmTypesFromSignature<void()>(state.context());
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());

    type = llvmTypesFromSignature<void()>(state.context(), &types);
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());
    CPPUNIT_ASSERT(types.empty());

    type = llvmTypesFromSignature<float()>(state.context(), &types);
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isFloatTy());
    CPPUNIT_ASSERT(types.empty());

    type = llvmTypesFromSignature<float(double, int64_t, float(*)[3])>(state.context(), &types);

    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isFloatTy());
    CPPUNIT_ASSERT_EQUAL(size_t(3), types.size());
    CPPUNIT_ASSERT(types[0]->isDoubleTy());
    CPPUNIT_ASSERT(types[1]->isIntegerTy(64));
    CPPUNIT_ASSERT(types[2]->isPointerTy());

    type = types[2]->getPointerElementType();
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isArrayTy());
    type = type->getArrayElementType();
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isFloatTy());
}

void
TestFunctionTypes::testLLVMFunctionTypeFromSignature()
{
    using openvdb::ax::codegen::llvmFunctionTypeFromSignature;

    unittest_util::LLVMState state;
    llvm::FunctionType* ftype = nullptr;
    std::vector<llvm::Type*> types;

    ftype = llvmFunctionTypeFromSignature<void()>(state.context());
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(0u, ftype->getNumParams());


    ftype = llvmFunctionTypeFromSignature<float(double, int64_t, float(*)[3])>(state.context());

    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isFloatTy());
    CPPUNIT_ASSERT_EQUAL(3u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0)->isDoubleTy());
    CPPUNIT_ASSERT(ftype->getParamType(1)->isIntegerTy(64));
    CPPUNIT_ASSERT(ftype->getParamType(2)->isPointerTy());

    llvm::Type* type = ftype->getParamType(2)->getPointerElementType();
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isArrayTy());
    type = type->getArrayElementType();
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isFloatTy());
}

void
TestFunctionTypes::testPrintSignature()
{
    using openvdb::ax::codegen::printSignature;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    std::vector<llvm::Type*> types;
    const llvm::Type* vt = llvm::Type::getVoidTy(C);

    std::ostringstream os;

    printSignature(os, types, vt);
    CPPUNIT_ASSERT(os.str() == "void()");
    os.str("");

    types.emplace_back(llvm::Type::getInt32Ty(C));
    types.emplace_back(llvm::Type::getInt64Ty(C));
    printSignature(os, types, vt);
    CPPUNIT_ASSERT_EQUAL(std::string("void(i32; i64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "test");
    CPPUNIT_ASSERT_EQUAL(std::string("void test(i32; i64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "", {"one"}, true);
    CPPUNIT_ASSERT_EQUAL(std::string("void(int32 one; int64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "", {"one", "two"}, true);
    CPPUNIT_ASSERT_EQUAL(std::string("void(int32 one; int64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"one", "two", "three"}, true);
    CPPUNIT_ASSERT_EQUAL(std::string("void 1(int32 one; int64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"", "two"}, false);
    CPPUNIT_ASSERT_EQUAL(std::string("void 1(i32; i64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"", "two"}, false);
    CPPUNIT_ASSERT_EQUAL(std::string("void 1(i32; i64 two)"), os.str());
    os.str("");

    types.emplace_back(llvm::Type::getInt8PtrTy(C));
    types.emplace_back(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3));

    printSignature(os, types, llvm::Type::getInt64Ty(C), "test", {"", "two"}, true);
    CPPUNIT_ASSERT_EQUAL(std::string("int64 test(int32; int64 two; i8*; vec3i)"), os.str());
    os.str("");

    types.clear();

    printSignature(os, types, llvm::Type::getInt64Ty(C), "test", {"", "two"});
    CPPUNIT_ASSERT_EQUAL(std::string("i64 test()"), os.str());
    os.str("");
}

void
TestFunctionTypes::testFunctionCreate()
{
    using openvdb::ax::codegen::Function;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();
    std::vector<llvm::Type*> types;
    llvm::Type* type = nullptr;
    std::ostringstream os;

    Function::Ptr test(new TestFunction({llvm::Type::getInt32Ty(C)},
        llvm::Type::getVoidTy(C), "ax.test"));

    // test types
    type = test->types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(1), types.size());
    CPPUNIT_ASSERT(types[0]->isIntegerTy(32));
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());
    // test various getters
    CPPUNIT_ASSERT_EQUAL(std::string("ax.test"), std::string(test->symbol()));
    CPPUNIT_ASSERT_EQUAL(size_t(1), test->size());
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(0)));
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(1)));

    // test create detached
    llvm::Function* function = test->create(C);
    llvm::Function* function2 = test->create(C);
    // additional create call should create a new function
    CPPUNIT_ASSERT(function != function2);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(1), function->arg_size());

    llvm::FunctionType* ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(1u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0)->isIntegerTy(32));
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;
    delete function2;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.test"));
    function = test->create(M);
    // additional call should match
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.test"));
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(1), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(1u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0)->isIntegerTy(32));
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("void name(int32)"), os.str());

    //
    // Test empty signature

    test.reset(new TestFunction({}, llvm::Type::getInt32Ty(C), "ax.empty.test"));
    types.clear();

    // test types
    type = test->types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(0), types.size());
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isIntegerTy(32));
    // test various getters
    CPPUNIT_ASSERT_EQUAL(std::string("ax.empty.test"), std::string(test->symbol()));
    CPPUNIT_ASSERT_EQUAL(size_t(0), test->size());
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(0)));

    // test create detached
    function = test->create(C);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(0), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isIntegerTy(32));
    CPPUNIT_ASSERT_EQUAL(0u, ftype->getNumParams());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.empty.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.empty.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.empty.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("int32 name()"), os.str());

    //
    // Test scalar types

    test.reset(new TestFunction({
            llvm::Type::getInt1Ty(C),
            llvm::Type::getInt16Ty(C),
            llvm::Type::getInt32Ty(C),
            llvm::Type::getInt64Ty(C),
            llvm::Type::getFloatTy(C),
            llvm::Type::getDoubleTy(C),
        },
        llvm::Type::getInt16Ty(C), "ax.scalars.test"));
    types.clear();

    CPPUNIT_ASSERT_EQUAL(std::string("ax.scalars.test"), std::string(test->symbol()));

    type = test->types(types, state.context());
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isIntegerTy(16));
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0]->isIntegerTy(1));
    CPPUNIT_ASSERT(types[1]->isIntegerTy(16));
    CPPUNIT_ASSERT(types[2]->isIntegerTy(32));
    CPPUNIT_ASSERT(types[3]->isIntegerTy(64));
    CPPUNIT_ASSERT(types[4]->isFloatTy());
    CPPUNIT_ASSERT(types[5]->isDoubleTy());

    // test create detached
    function = test->create(C);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isIntegerTy(16));
    CPPUNIT_ASSERT_EQUAL(6u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0)->isIntegerTy(1));
    CPPUNIT_ASSERT(ftype->getParamType(1)->isIntegerTy(16));
    CPPUNIT_ASSERT(ftype->getParamType(2)->isIntegerTy(32));
    CPPUNIT_ASSERT(ftype->getParamType(3)->isIntegerTy(64));
    CPPUNIT_ASSERT(ftype->getParamType(4)->isFloatTy());
    CPPUNIT_ASSERT(ftype->getParamType(5)->isDoubleTy());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.scalars.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.scalars.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.scalars.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("int16 name(bool; int16; int32; int64; float; double)"), os.str());

    //
    // Test scalar ptrs types

    test.reset(new TestFunction({
            llvm::Type::getInt1Ty(C)->getPointerTo(),
            llvm::Type::getInt16Ty(C)->getPointerTo(),
            llvm::Type::getInt32Ty(C)->getPointerTo(),
            llvm::Type::getInt64Ty(C)->getPointerTo(),
            llvm::Type::getFloatTy(C)->getPointerTo(),
            llvm::Type::getDoubleTy(C)->getPointerTo()
        },
        llvm::Type::getInt32Ty(C), "ax.scalarptrs.test"));
    types.clear();

    CPPUNIT_ASSERT_EQUAL(std::string("ax.scalarptrs.test"), std::string(test->symbol()));

    type = test->types(types, C);
    CPPUNIT_ASSERT(type->isIntegerTy(32));
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::Type::getInt1Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[1] == llvm::Type::getInt16Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::Type::getInt32Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::Type::getInt64Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::Type::getDoubleTy(C)->getPointerTo());

    // test create detached
    function = test->create(C);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isIntegerTy(32));
    CPPUNIT_ASSERT_EQUAL(6u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == llvm::Type::getInt1Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(1) == llvm::Type::getInt16Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(2) == llvm::Type::getInt32Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(3) == llvm::Type::getInt64Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(4) == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(5) == llvm::Type::getDoubleTy(C)->getPointerTo());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.scalarptrs.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.scalarptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.scalarptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));

    //
    // Test array ptrs types

    test.reset(new TestFunction({
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo(),  // vec2i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo(),  // vec2f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo(), // vec2d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo(),  // vec3i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo(),  // vec3f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo(), // vec3d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo(),  // vec4i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo(),  // vec4f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo(), // vec4d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo(),  // ix9 (not supported by ax)
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo(),  // mat3f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo(), // mat3d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo(),  // ix16 (not supported by ax)
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo(),  // mat3f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo()  // mat3d
        },
        llvm::Type::getInt64Ty(C), "ax.arrayptrs.test"));
    types.clear();

    type = test->types(types, C);
    CPPUNIT_ASSERT(type->isIntegerTy(64));
    CPPUNIT_ASSERT_EQUAL(size_t(15), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(types[1] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(types[6] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(types[7] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(types[8] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(types[9] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(types[10] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(types[11] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(types[12] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo());
    CPPUNIT_ASSERT(types[13] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo());
    CPPUNIT_ASSERT(types[14] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo());

    // test create detached
    function = test->create(C);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(15), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isIntegerTy(64));
    CPPUNIT_ASSERT_EQUAL(15u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(1) == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(2) == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(3) == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(4) == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(5) == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(6) == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(7) == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(8) == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(9) == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(10) == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(11) == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(12) == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(13) == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(14) == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.arrayptrs.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.arrayptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.arrayptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));

    // test print - note mat/i types are not ax types
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("int64 name(vec2i; vec2f; vec2d; vec3i; vec3f; vec3d;"
        " vec4i; vec4f; vec4d; [9 x i32]*; mat3f; mat3d; [16 x i32]*; mat4f; mat4d)"),
        os.str());

    //
    // Test void ptr arguments

    test.reset(new TestFunction({
            llvm::Type::getVoidTy(C)->getPointerTo(),
            llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo(),
            llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo()->getPointerTo(),
            llvm::Type::getFloatTy(C)->getPointerTo(),
            llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo(),
            llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo()
        },
        llvm::Type::getVoidTy(C), "ax.vptrs.test"));
    types.clear();

    // Note that C++ bindings will convert void* to i8* but they should be
    // unmodified in this example where we use the derived TestFunction

    type = test->types(types, C);
    CPPUNIT_ASSERT(type->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::Type::getVoidTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[1] == llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo());

    // test create detached
    function = test->create(C);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(6u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == llvm::Type::getVoidTy(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(1) == llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(2) == llvm::Type::getVoidTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(3) == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(4) == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(5) == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    CPPUNIT_ASSERT(!M.getFunction("ax.vptrs.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.vptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.vptrs.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));

    //
    // Test creation with builder methods
    // @note  These methods may be moved to the constructor in the future

    CPPUNIT_ASSERT(test->dependencies().empty());
    CPPUNIT_ASSERT(!test->hasParamAttribute(0, llvm::Attribute::ReadOnly));
    CPPUNIT_ASSERT(!test->hasParamAttribute(-1, llvm::Attribute::ReadOnly));

    test->setDependencies({"dep"});
    CPPUNIT_ASSERT_EQUAL(size_t(1), test->dependencies().size());
    CPPUNIT_ASSERT_EQUAL(std::string("dep"), std::string(test->dependencies().front()));

    test->setDependencies({});
    CPPUNIT_ASSERT(test->dependencies().empty());

    test->setFnAttributes({llvm::Attribute::ReadOnly});
    test->setRetAttributes({llvm::Attribute::NoAlias});
    test->setParamAttributes(1, {llvm::Attribute::WriteOnly});
    test->setParamAttributes(-1, {llvm::Attribute::WriteOnly});

    CPPUNIT_ASSERT(!test->hasParamAttribute(0, llvm::Attribute::WriteOnly));
    CPPUNIT_ASSERT(!test->hasParamAttribute(2, llvm::Attribute::WriteOnly));
    CPPUNIT_ASSERT(test->hasParamAttribute(1, llvm::Attribute::WriteOnly));
    CPPUNIT_ASSERT(test->hasParamAttribute(-1, llvm::Attribute::WriteOnly));

    function = test->create(C);
    CPPUNIT_ASSERT(function);
    llvm::AttributeList list = function->getAttributes();
    CPPUNIT_ASSERT(!list.isEmpty());
    CPPUNIT_ASSERT(!list.hasParamAttrs(0));
    CPPUNIT_ASSERT(!list.hasParamAttrs(2));
    CPPUNIT_ASSERT(list.hasParamAttr(1, llvm::Attribute::WriteOnly));
#if LLVM_VERSION_MAJOR <= 13
    CPPUNIT_ASSERT(list.hasFnAttribute(llvm::Attribute::ReadOnly));
    CPPUNIT_ASSERT(list.hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias));
#else
    CPPUNIT_ASSERT(list.hasFnAttr(llvm::Attribute::ReadOnly));
    CPPUNIT_ASSERT(list.hasRetAttr(llvm::Attribute::NoAlias));
#endif
    delete function;
}

void
TestFunctionTypes::testFunctionCall()
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::LLVMType;

    //

    {
        unittest_util::LLVMState state;
        llvm::LLVMContext& C = state.context();
        llvm::Module& M = state.module();
        llvm::IRBuilder<> B(state.scratchBlock());
        llvm::Function* BaseFunction = B.GetInsertBlock()->getParent();

        Function::Ptr test(new TestFunction({llvm::Type::getInt32Ty(C)},
            llvm::Type::getVoidTy(C), "ax.test"));

        llvm::Function* function = test->create(M);
        llvm::Value* arg = B.getInt32(1);
        llvm::Value* result = test->call({arg}, B, /*cast*/false);
        CPPUNIT_ASSERT(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        CPPUNIT_ASSERT(call);
        CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
        CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
        CPPUNIT_ASSERT_EQUAL(arg, call->getArgOperand(0));
        // Test the builder is pointing to the correct location
        CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

        // add a ret void to the current function and to the created function,
        // then check the IR is valid (this will check the function arguments
        // and creation are all correct)

        finalizeFunction(B);
        finalizeFunction(B, function);
        VERIFY_FUNCTION_IR(function);
        VERIFY_MODULE_IR(&M);
    }

    {
        unittest_util::LLVMState state;
        llvm::LLVMContext& C = state.context();
        llvm::Module& M = state.module();
        llvm::IRBuilder<> B(state.scratchBlock());
        llvm::Function* BaseFunction = B.GetInsertBlock()->getParent();

        Function::Ptr test(new TestFunction({llvm::Type::getInt32Ty(C)},
            llvm::Type::getVoidTy(C), "ax.test"));

        // call first, then create

        llvm::Value* arg = B.getInt32(1);
        llvm::Value* result = test->call({arg}, B, /*cast*/false);
        llvm::Function* function = test->create(M);
        CPPUNIT_ASSERT(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        CPPUNIT_ASSERT(call);
        CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
        CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
        CPPUNIT_ASSERT_EQUAL(arg, call->getArgOperand(0));
        // Test the builder is pointing to the correct location
        CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

        // add a ret void to the current function and to the created function,
        // then check the IR is valid (this will check the function arguments
        // and creation are all correct)

        finalizeFunction(B);
        finalizeFunction(B, function);
        VERIFY_FUNCTION_IR(function);
        VERIFY_MODULE_IR(&M);
    }

    // Now test casting/argument mismatching for most argument types

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();
    llvm::IRBuilder<> B(state.scratchBlock());

    Function::Ptr test(new TestFunction({
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo(),   // vec3i
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo(),  // vec2d
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo(),  // mat3d
            llvm::Type::getInt32Ty(C), // int
            llvm::Type::getInt64Ty(C), // int64
            llvm::Type::getFloatTy(C)  // float
        },
        llvm::Type::getVoidTy(C),
        "ax.test"));

    std::vector<llvm::Type*> expected;
    test->types(expected, C);

    // default args

    llvm::Value* f32c0 = LLVMType<float>::get(C, 0.0f); // float
    llvm::Value* d64c0 = LLVMType<double>::get(C, 0.0); // double
    llvm::Value* i32c1 = B.getInt32(1); // int
    llvm::Value* i64c1 = B.getInt64(1); // int64
    llvm::Value* vec3i = openvdb::ax::codegen::arrayPack({i32c1,i32c1,i32c1}, B); // vec3i
    llvm::Value* vec2d = openvdb::ax::codegen::arrayPack({d64c0,d64c0},B); // vec2d
    llvm::Value* mat3d = openvdb::ax::codegen::arrayPack({ d64c0,d64c0,d64c0,
                                                            d64c0,d64c0,d64c0,
                                                            d64c0,d64c0,d64c0
                                                           }, B); // mat3d

    std::vector<llvm::Value*> args{vec3i, vec2d, mat3d, i32c1, i64c1, f32c0};

    llvm::Function* function = test->create(M);
    finalizeFunction(B, function);
    VERIFY_FUNCTION_IR(function);

    // also finalize the current module function, but set the inset point
    // just above it so we can continue to verify IR during this test
    llvm::Value* inst = B.CreateRetVoid();
    // This specifies that created instructions should be inserted before
    // the specified instruction.
    B.SetInsertPoint(llvm::cast<llvm::Instruction>(inst));

    // test no casting needed for valid IR

    llvm::Value* result = test->call(args, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(args[0], call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(args[1], call->getArgOperand(1));
    CPPUNIT_ASSERT_EQUAL(args[2], call->getArgOperand(2));
    CPPUNIT_ASSERT_EQUAL(args[3], call->getArgOperand(3));
    CPPUNIT_ASSERT_EQUAL(args[4], call->getArgOperand(4));
    CPPUNIT_ASSERT_EQUAL(args[5], call->getArgOperand(5));
    VERIFY_MODULE_IR(&M);

    // test no casting needed for valid IR, even with cast=true

    result = test->call(args, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(args[0], call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(args[1], call->getArgOperand(1));
    CPPUNIT_ASSERT_EQUAL(args[2], call->getArgOperand(2));
    CPPUNIT_ASSERT_EQUAL(args[3], call->getArgOperand(3));
    CPPUNIT_ASSERT_EQUAL(args[4], call->getArgOperand(4));
    CPPUNIT_ASSERT_EQUAL(args[5], call->getArgOperand(5));
    VERIFY_MODULE_IR(&M);

    //

    // Test different types of valid casting

    llvm::Value* i1c0 = LLVMType<bool>::get(C, true); // bool
    llvm::Value* vec3f = openvdb::ax::codegen::arrayPack({f32c0,f32c0,f32c0}, B); // vec3f
    llvm::Value* vec3d = openvdb::ax::codegen::arrayPack({d64c0,d64c0,d64c0}, B); // vec3d
    llvm::Value* vec2f = openvdb::ax::codegen::arrayPack({f32c0,f32c0},B); // vec2f
    llvm::Value* vec2i = openvdb::ax::codegen::arrayPack({i32c1,i32c1},B); // vecid
    llvm::Value* mat3f = openvdb::ax::codegen::arrayPack({ f32c0,f32c0,f32c0,
                                                            f32c0,f32c0,f32c0,
                                                            f32c0,f32c0,f32c0
                                                           }, B); // mat3f
    //

    std::vector<llvm::Value*> argsToCast;
    argsToCast.emplace_back(vec3f); // vec3f
    argsToCast.emplace_back(vec2f); // vec2f
    argsToCast.emplace_back(mat3f); // mat3f
    argsToCast.emplace_back(i1c0); // bool
    argsToCast.emplace_back(i1c0); // bool
    argsToCast.emplace_back(i1c0); // bool

    result = test->call(argsToCast, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(argsToCast[0] != call->getArgOperand(0));
    CPPUNIT_ASSERT(argsToCast[1] != call->getArgOperand(1));
    CPPUNIT_ASSERT(argsToCast[2] != call->getArgOperand(2));
    CPPUNIT_ASSERT(argsToCast[3] != call->getArgOperand(3));
    CPPUNIT_ASSERT(argsToCast[4] != call->getArgOperand(4));
    CPPUNIT_ASSERT(argsToCast[5] != call->getArgOperand(5));
    CPPUNIT_ASSERT(expected[0] == call->getArgOperand(0)->getType());
    CPPUNIT_ASSERT(expected[1] == call->getArgOperand(1)->getType());
    CPPUNIT_ASSERT(expected[2] == call->getArgOperand(2)->getType());
    CPPUNIT_ASSERT(expected[3] == call->getArgOperand(3)->getType());
    CPPUNIT_ASSERT(expected[4] == call->getArgOperand(4)->getType());
    CPPUNIT_ASSERT(expected[5] == call->getArgOperand(5)->getType());
    VERIFY_MODULE_IR(&M);

    //

    argsToCast.clear();
    argsToCast.emplace_back(vec3d); // vec3d
    argsToCast.emplace_back(vec2i); // vec2i
    argsToCast.emplace_back(mat3d); // mat3d - no cast required
    argsToCast.emplace_back(f32c0); // float
    argsToCast.emplace_back(f32c0); // float
    argsToCast.emplace_back(f32c0); // float - no cast required

    result = test->call(argsToCast, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(argsToCast[0] != call->getArgOperand(0));
    CPPUNIT_ASSERT(argsToCast[1] != call->getArgOperand(1));
    CPPUNIT_ASSERT_EQUAL(args[2], call->getArgOperand(2));
    CPPUNIT_ASSERT(argsToCast[3] != call->getArgOperand(3));
    CPPUNIT_ASSERT(argsToCast[4] != call->getArgOperand(4));
    CPPUNIT_ASSERT_EQUAL(args[5], call->getArgOperand(5));
    CPPUNIT_ASSERT(expected[0] == call->getArgOperand(0)->getType());
    CPPUNIT_ASSERT(expected[1] == call->getArgOperand(1)->getType());
    CPPUNIT_ASSERT(expected[3] == call->getArgOperand(3)->getType());
    CPPUNIT_ASSERT(expected[4] == call->getArgOperand(4)->getType());
    VERIFY_MODULE_IR(&M);

    //

    argsToCast.clear();
    argsToCast.emplace_back(vec3i); // vec3i - no cast required
    argsToCast.emplace_back(vec2d); // vec2d - no cast required
    argsToCast.emplace_back(mat3d); // mat3d - no cast required
    argsToCast.emplace_back(i64c1); // int64
    argsToCast.emplace_back(i64c1); // int64 - no cast required
    argsToCast.emplace_back(i64c1); // int64

    result = test->call(argsToCast, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(args[0], call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(args[1], call->getArgOperand(1));
    CPPUNIT_ASSERT_EQUAL(args[2], call->getArgOperand(2));
    CPPUNIT_ASSERT(argsToCast[3] != call->getArgOperand(3));
    CPPUNIT_ASSERT_EQUAL(args[4], call->getArgOperand(4));
    CPPUNIT_ASSERT(argsToCast[5] != call->getArgOperand(5));
    CPPUNIT_ASSERT(expected[3] == call->getArgOperand(3)->getType());
    CPPUNIT_ASSERT(expected[5] == call->getArgOperand(5)->getType());
    VERIFY_MODULE_IR(&M);

    //

    // Test that invalid IR is generated if casting cannot be performed.
    // This is just to test that call doesn't error or behave unexpectedly

    // Test called with castable arg but cast is false. Test arg is left
    // unchanged and IR is invalid due to signature size

    result = test->call({vec3f}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
    // should be the same as cast is false
    CPPUNIT_ASSERT(vec3f == call->getArgOperand(0));
    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);

    // Test called with castable arg with cast true. Test IR is invalid
    // due to signature size

    result = test->call({vec3f}, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
    // shouldn't be the same as it should have been cast
    CPPUNIT_ASSERT(vec3f != call->getArgOperand(0));
    CPPUNIT_ASSERT(expected[0] == call->getArgOperand(0)->getType());
    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);

    // Test called with non castable args, but matchign signature size.
    // Test IR is invalid due to cast being off

    result = test->call(argsToCast, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(6u, getNumArgFromCallInst(call));
    // no casting, args should match operands
    CPPUNIT_ASSERT(argsToCast[0] == call->getArgOperand(0));
    CPPUNIT_ASSERT(argsToCast[1] == call->getArgOperand(1));
    CPPUNIT_ASSERT(argsToCast[2] == call->getArgOperand(2));
    CPPUNIT_ASSERT(argsToCast[3] == call->getArgOperand(3));
    CPPUNIT_ASSERT(argsToCast[4] == call->getArgOperand(4));
    CPPUNIT_ASSERT(argsToCast[5] == call->getArgOperand(5));
    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);

    //
    // Test strings

    llvm::Type* axstr = LLVMType<openvdb::ax::codegen::String*>::get(C);  // str*
    llvm::Type* chars = LLVMType<char*>::get(C);  // char*

    // build values

    llvm::Value* chararray = B.CreateGlobalStringPtr("tmp"); // char*
    // @note  non-safer initialization of strings
    llvm::Value* strptr = B.CreateAlloca(LLVMType<openvdb::ax::codegen::String>::get(C)); // str*

    // void ax.str.test(openvdb::ax::codegen::String*, char*)
    test.reset(new TestFunction({axstr, chars},
        llvm::Type::getVoidTy(C),
        "ax.str.test"));

    std::vector<llvm::Value*> stringArgs{strptr, chararray};

    function = test->create(M);
    finalizeFunction(B, function);
    VERIFY_FUNCTION_IR(function);

    result = test->call(stringArgs, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(stringArgs[0] == call->getArgOperand(0));
    CPPUNIT_ASSERT(stringArgs[1] == call->getArgOperand(1));

    //

    result = test->call(stringArgs, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(stringArgs[0] == call->getArgOperand(0));
    CPPUNIT_ASSERT(stringArgs[1] == call->getArgOperand(1));

    // Test openvdb::ax::codegen::String -> char*

    stringArgs[0] = strptr;
    stringArgs[1] = strptr;

    result = test->call(stringArgs, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(stringArgs[0] == call->getArgOperand(0));
    CPPUNIT_ASSERT(stringArgs[1] != call->getArgOperand(1));
    CPPUNIT_ASSERT(chars == call->getArgOperand(1)->getType());

    VERIFY_MODULE_IR(&M);

    // Test char* does not catch to openvdb::ax::codegen::String

    stringArgs[0] = chararray;
    stringArgs[1] = chararray;

    result = test->call(stringArgs, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    // no valid casting
    CPPUNIT_ASSERT(stringArgs[0] == call->getArgOperand(0));
    CPPUNIT_ASSERT(stringArgs[1] == call->getArgOperand(1));

    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);

    //
    // Test ** pointers

    test.reset(new TestFunction({
            llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo(),
            llvm::Type::getDoubleTy(C)->getPointerTo()->getPointerTo()
        },
        llvm::Type::getVoidTy(C),
        "ax.ptrs.test"));

    function = test->create(M);
    finalizeFunction(B, function);
    VERIFY_FUNCTION_IR(function);

    llvm::Value* fptr = B.CreateAlloca(llvm::Type::getFloatTy(C)->getPointerTo());
    llvm::Value* dptr = B.CreateAlloca(llvm::Type::getDoubleTy(C)->getPointerTo());

    result = test->call({fptr, dptr}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(fptr == call->getArgOperand(0));
    CPPUNIT_ASSERT(dptr == call->getArgOperand(1));

    VERIFY_MODULE_IR(&M);

    //

    result = test->call({fptr, dptr}, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(fptr == call->getArgOperand(0));
    CPPUNIT_ASSERT(dptr == call->getArgOperand(1));

    VERIFY_MODULE_IR(&M);

    // switch the points, check no valid casting

    result = test->call({dptr, fptr}, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    // args unaltered as casting is invalid
    CPPUNIT_ASSERT(dptr == call->getArgOperand(0));
    CPPUNIT_ASSERT(fptr == call->getArgOperand(1));

    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);

    //
    // Test void pointers

    test.reset(new TestFunction({
            LLVMType<void*>::get(C),
        },
        llvm::Type::getVoidTy(C),
        "ax.void.test"));

    function = test->create(M);
    finalizeFunction(B, function);
    VERIFY_FUNCTION_IR(function);

    llvm::Value* vptrptr = B.CreateAlloca(LLVMType<void*>::get(C));
    llvm::Value* vptr = openvdb::ax::codegen::ir_load(B, vptrptr);

    result = test->call({vptr}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(vptr == call->getArgOperand(0));

    VERIFY_MODULE_IR(&M);

    //

    result = test->call({vptr}, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(vptr == call->getArgOperand(0));

    VERIFY_MODULE_IR(&M);

    // verify no cast from other pointers to void*

    result = test->call({fptr}, B, /*cast*/true);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(1u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT(fptr == call->getArgOperand(0));

    VERIFY_MODULE_IR_INVALID(&M);

    // Remove the bad instruction (and re-verify to double check)
    call->eraseFromParent();
    VERIFY_MODULE_IR(&M);
}

void
TestFunctionTypes::testFunctionMatch()
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::LLVMType;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    Function::SignatureMatch match;

    const std::vector<llvm::Type*> scalars {
        llvm::Type::getInt1Ty(C),   // bool
        llvm::Type::getInt16Ty(C),  // int16
        llvm::Type::getInt32Ty(C),  // int
        llvm::Type::getInt64Ty(C),  // int64
        llvm::Type::getFloatTy(C),  // float
        llvm::Type::getDoubleTy(C)  // double
    };
    const std::vector<llvm::Type*> array2 {
        llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo(),    // vec2i
        llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo(),    // vec2f
        llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo()    // vec2d
    };
    const std::vector<llvm::Type*> array3 {
        llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo(),    // vec3i
        llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo(),    // vec3f
        llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo()    // vec3d
    };
    const std::vector<llvm::Type*> array4 {
        llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo(),    // vec3i
        llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo(),    // vec3f
        llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo()    // vec3d
    };
    const std::vector<llvm::Type*> array9 {
        llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo(),    // ix9 (not supported by ax)
        llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo(),    // mat3f
        llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo()    // mat3d
    };
    const std::vector<llvm::Type*> array16 {
        llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo(),    // ix16 (not supported by ax)
        llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo(),    // mat3f
        llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo()    // mat3d
    };
    const std::vector<std::vector<llvm::Type*>> arrays {
        array2,
        array3,
        array4,
        array9,
        array16,
    };

    // test empty explicit match

    Function::Ptr test(new TestFunction({},
        llvm::Type::getVoidTy(C), "ax.test"));
    match = test->match({}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);

    //

    std::vector<llvm::Type*> types;
    types.insert(types.end(), scalars.begin(), scalars.end());
    types.insert(types.end(), array2.begin(), array2.end());
    types.insert(types.end(), array3.begin(), array3.end());
    types.insert(types.end(), array4.begin(), array4.end());
    types.insert(types.end(), array9.begin(), array9.end());
    types.insert(types.end(), array16.begin(), array16.end());
    types.insert(types.end(), LLVMType<openvdb::ax::codegen::String*>::get(C));

    // check types are unique
    CPPUNIT_ASSERT_EQUAL(std::set<llvm::Type*>(types.begin(), types.end()).size(), types.size());

    //

    test.reset(new TestFunction({
            llvm::Type::getInt1Ty(C),   // bool
            llvm::Type::getInt16Ty(C),  // int16
            llvm::Type::getInt32Ty(C),  // int32
            llvm::Type::getInt64Ty(C),  // int64
            llvm::Type::getFloatTy(C),  // float
            llvm::Type::getDoubleTy(C), // double
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo(),    // vec2i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo(),    // vec2f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo(),   // vec2d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo(),    // vec3i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo(),    // vec3f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo(),   // vec3d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo(),    // vec4i
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo(),    // vec4f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo(),   // vec4d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo(),    // ix9 (not supported by ax)
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo(),    // mat3f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo(),   // mat3d
            llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo(),   // ix16 (not supported by ax)
            llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo(),   // mat4f
            llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo(),  // mat4d
            LLVMType<openvdb::ax::codegen::String*>::get(C) // string
        },
        llvm::Type::getVoidTy(C),
        "ax.test"));

    match = test->match(types, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);

    // test size match

    llvm::Type* i32t = llvm::Type::getInt32Ty(C);
    test.reset(new TestFunction({i32t},
        llvm::Type::getVoidTy(C), "ax.test"));
    match = test->match({llvm::ArrayType::get(i32t, 1)->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Size);
    match = test->match({i32t->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Size);

    // test no match

    match = test->match({}, C);
    CPPUNIT_ASSERT_EQUAL(Function::None, match);
    match = test->match({i32t, i32t}, C);
    CPPUNIT_ASSERT_EQUAL(Function::None, match);

    // test scalar matches

   for (size_t i = 0; i < scalars.size(); ++i) {
        test.reset(new TestFunction({scalars[i]}, llvm::Type::getVoidTy(C), "ax.test"));
        std::vector<llvm::Type*> copy(scalars);
        std::swap(copy[i], copy.back());
        copy.pop_back();
        CPPUNIT_ASSERT_EQUAL(size_t(5), copy.size());
        CPPUNIT_ASSERT_EQUAL(Function::Explicit, test->match({scalars[i]}, C));
        CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[0]}, C));
        CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[1]}, C));
        CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[2]}, C));
        CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[3]}, C));
        CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[4]}, C));
   }

   //
   // Test array matches - no implicit cast as operands are not marked as readonly

   for (const std::vector<llvm::Type*>& types : arrays) {
        // test these array types
        for (size_t i = 0; i < types.size(); ++i) {
            test.reset(new TestFunction({types[i]}, llvm::Type::getVoidTy(C), "ax.test"));
            std::vector<llvm::Type*> copy(types);
            std::swap(copy[i], copy.back());
            copy.pop_back();
            CPPUNIT_ASSERT_EQUAL(size_t(2), copy.size());
            CPPUNIT_ASSERT_EQUAL(Function::Explicit, test->match({types[i]}, C));
            CPPUNIT_ASSERT_EQUAL(Function::Size, test->match({copy[0]}, C));
            CPPUNIT_ASSERT_EQUAL(Function::Size, test->match({copy[1]}, C));

            // test non matching size arrays
            for (const std::vector<llvm::Type*>& inputs : arrays) {
                if (&types == &inputs) continue;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(Function::Size, test->match({inputs[i]}, C));
                }
            }
        }
   }

   //
   // Test array matches with readonly marking

   for (const std::vector<llvm::Type*>& types : arrays) {
        // test these array types
        for (size_t i = 0; i < types.size(); ++i) {
            test.reset(new TestFunction({types[i]}, llvm::Type::getVoidTy(C), "ax.test"));
            test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
            std::vector<llvm::Type*> copy(types);
            std::swap(copy[i], copy.back());
            copy.pop_back();
            CPPUNIT_ASSERT_EQUAL(size_t(2), copy.size());
            CPPUNIT_ASSERT_EQUAL(Function::Explicit, test->match({types[i]}, C));
            CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[0]}, C));
            CPPUNIT_ASSERT_EQUAL(Function::Implicit, test->match({copy[1]}, C));

            // test non matching size arrays
            for (const std::vector<llvm::Type*>& inputs : arrays) {
                if (&types == &inputs) continue;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    CPPUNIT_ASSERT_EQUAL(Function::Size, test->match({inputs[i]}, C));
                }
            }
        }
    }

    // test strings

    test.reset(new TestFunction({LLVMType<char*>::get(C)},
        llvm::Type::getVoidTy(C), "ax.test"));
    CPPUNIT_ASSERT_EQUAL(Function::Size,
        test->match({LLVMType<openvdb::ax::codegen::String*>::get(C)}, C));
    CPPUNIT_ASSERT_EQUAL(Function::Explicit,
        test->match({LLVMType<char*>::get(C)}, C));

    test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
    CPPUNIT_ASSERT_EQUAL(Function::Implicit,
        test->match({LLVMType<openvdb::ax::codegen::String*>::get(C)}, C));

    test.reset(new TestFunction({LLVMType<openvdb::ax::codegen::String*>::get(C)},
        llvm::Type::getVoidTy(C), "ax.test"));
    CPPUNIT_ASSERT_EQUAL(Function::Size,
        test->match({LLVMType<char*>::get(C)}, C));

    // test pointers

    llvm::Type* fss = llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo();
    llvm::Type* dss = llvm::Type::getDoubleTy(C)->getPointerTo()->getPointerTo();

    test.reset(new TestFunction({fss, dss},
        llvm::Type::getVoidTy(C),
        "ax.ptrs.test"));

    CPPUNIT_ASSERT_EQUAL(Function::Explicit,
        test->match({fss, dss}, C));
    CPPUNIT_ASSERT_EQUAL(Function::Size,
        test->match({fss, fss}, C));

    // Even if pointers are marked as readonly, casting is not supported
    test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
    test->setParamAttributes(1, {llvm::Attribute::ReadOnly});

    CPPUNIT_ASSERT_EQUAL(Function::Size,
        test->match({fss, fss}, C));
}

void
TestFunctionTypes::testCFunctions()
{
    using openvdb::ax::codegen::CFunction;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Type* returnType = nullptr;
    std::vector<llvm::Type*> types;

    // test basic creation

    CFunction<void()> voidfunc("voidfunc", &CBindings::voidfunc);
    CFunction<int16_t(bool,int16_t,int32_t,int64_t,float,double)>
        scalars("scalarfunc", &CBindings::scalarfunc);
    CFunction<int32_t(bool*,int16_t*,int32_t*,int64_t*,float*,double*)>
        scalarptrs("scalatptsfunc", &CBindings::scalatptsfunc);
    CFunction<int64_t(bool(*)[1],int16_t(*)[2],int32_t(*)[3],int64_t(*)[4],float(*)[5],double(*)[6])>
        arrayptrs("arrayfunc", &CBindings::arrayfunc);
    CFunction<float()> select("tmplfunc", (float(*)())(CBindings::tmplfunc));
    CFunction<void(void*, void**, void***, float*, float**, float***)>
        mindirect("multiptrfunc", &CBindings::multiptrfunc);

    // test static void function

    CPPUNIT_ASSERT_EQUAL(size_t(0), voidfunc.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::voidfunc),
        voidfunc.address());
    CPPUNIT_ASSERT_EQUAL(std::string("voidfunc"),
        std::string(voidfunc.symbol()));

    // test scalar arguments

    CPPUNIT_ASSERT_EQUAL(size_t(6), scalars.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::scalarfunc),
        scalars.address());
    CPPUNIT_ASSERT_EQUAL(std::string("scalarfunc"),
        std::string(scalars.symbol()));

    // test scalar ptr arguments

    CPPUNIT_ASSERT_EQUAL(size_t(6), scalarptrs.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::scalatptsfunc),
        scalarptrs.address());
    CPPUNIT_ASSERT_EQUAL(std::string("scalatptsfunc"),
        std::string(scalarptrs.symbol()));

    // test array ptr arguments

    CPPUNIT_ASSERT_EQUAL(size_t(6), arrayptrs.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::arrayfunc),
        arrayptrs.address());
    CPPUNIT_ASSERT_EQUAL(std::string("arrayfunc"),
        std::string(arrayptrs.symbol()));

    // test selected template functions

    CPPUNIT_ASSERT_EQUAL(size_t(0), select.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::tmplfunc<float>),
        select.address());
    CPPUNIT_ASSERT_EQUAL(std::string("tmplfunc"),
        std::string(select.symbol()));

    // test multiple indirection layers

    CPPUNIT_ASSERT_EQUAL(size_t(6), mindirect.size());
    CPPUNIT_ASSERT_EQUAL(reinterpret_cast<uint64_t>(&CBindings::multiptrfunc),
        mindirect.address());
    CPPUNIT_ASSERT_EQUAL(std::string("multiptrfunc"),
        std::string(mindirect.symbol()));

    //
    // Test types

    // test scalar arguments

    returnType = scalars.types(types, C);
    CPPUNIT_ASSERT(returnType->isIntegerTy(16));
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0]->isIntegerTy(1));
    CPPUNIT_ASSERT(types[1]->isIntegerTy(16));
    CPPUNIT_ASSERT(types[2]->isIntegerTy(32));
    CPPUNIT_ASSERT(types[3]->isIntegerTy(64));
    CPPUNIT_ASSERT(types[4]->isFloatTy());
    CPPUNIT_ASSERT(types[5]->isDoubleTy());

    types.clear();

    // test scalar ptr arguments

    returnType = scalarptrs.types(types, C);
    CPPUNIT_ASSERT(returnType->isIntegerTy(32));
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::Type::getInt1Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[1] == llvm::Type::getInt16Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::Type::getInt32Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::Type::getInt64Ty(C)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::Type::getDoubleTy(C)->getPointerTo());

    types.clear();

    // test array ptr arguments

    returnType = arrayptrs.types(types, C);
    CPPUNIT_ASSERT(returnType->isIntegerTy(64));
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1)->getPointerTo());
    CPPUNIT_ASSERT(types[1] == llvm::ArrayType::get(llvm::Type::getInt16Ty(C), 2)->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::ArrayType::get(llvm::Type::getInt64Ty(C), 4)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 5)->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 6)->getPointerTo());

    types.clear();

    // test void ptr arguments
    // void* are inferred as int8_t* types

    returnType = mindirect.types(types, C);
    CPPUNIT_ASSERT(returnType->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
    CPPUNIT_ASSERT(types[0] == llvm::Type::getInt8PtrTy(C));
    CPPUNIT_ASSERT(types[1] == llvm::Type::getInt8PtrTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[2] == llvm::Type::getInt8PtrTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(types[3] == llvm::Type::getFloatTy(C)->getPointerTo());
    CPPUNIT_ASSERT(types[4] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo());
    CPPUNIT_ASSERT(types[5] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
}

void
TestFunctionTypes::testCFunctionCF()
{
    using openvdb::ax::codegen::CFunction;
    using openvdb::ax::codegen::LLVMType;

    static auto cftest1 = []() -> int32_t { return 10; };
    static auto cftest2 = [](float a) -> float { return a; };
    // currently unsupported for arrays
    static auto cftest3 = [](float(*a)[3]) -> float { return (*a)[0]; };
    // currently unsupported for return voids
    static auto cftest4 = [](float* a) -> void { (*a)*=5.0f; };

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::IRBuilder<> B(state.scratchBlock());

    // test with no args

    CFunction<int32_t()> test1("ax.test1", cftest1);
    // off by default
    CPPUNIT_ASSERT(!test1.hasConstantFold());
    CPPUNIT_ASSERT(test1.fold({B.getInt32(1)}, C) == nullptr);

    test1.setConstantFold(true);
    llvm::Value* result = test1.fold({B.getInt32(1)}, C);
    CPPUNIT_ASSERT(result);
    llvm::ConstantInt* constant = llvm::dyn_cast<llvm::ConstantInt>(result);
    CPPUNIT_ASSERT(constant);
    CPPUNIT_ASSERT_EQUAL(uint64_t(10), constant->getLimitedValue());

    // test with scalar arg

    CFunction<float(float)> test2("ax.test2", cftest2);
    test2.setConstantFold(true);
    result = test2.fold({LLVMType<float>::get(C, -3.2f)}, C);
    CPPUNIT_ASSERT(result);
    llvm::ConstantFP* constantfp = llvm::dyn_cast<llvm::ConstantFP>(result);
    CPPUNIT_ASSERT(constantfp);
    const llvm::APFloat& apf = constantfp->getValueAPF();
    CPPUNIT_ASSERT_EQUAL(-3.2f, apf.convertToFloat());

    // test unsupported

    CFunction<float(float(*)[3])> test3("ax.test3", cftest3);
    test3.setConstantFold(true);
    // constant arg (verify it would 100% match)
    // note that this arg is fundamentally the wrong type for this function
    // and the way in which we support vector types anyway (by ptr) - but because
    // its impossible to have a constant ptr, use this for now as this will most
    // likely be the way we support folding for arrays in the future
    llvm::Value* arg = LLVMType<float[3]>::get(C, {1,2,3});
    CPPUNIT_ASSERT(llvm::isa<llvm::Constant>(arg));
    // check fold fails
    CPPUNIT_ASSERT(test3.fold({arg}, C) == nullptr);

    //

    CFunction<void(float*)> test4("ax.test4", cftest4);
    test4.setConstantFold(true);
    // constant arg (verify it would 100% match)
    llvm::Value* nullfloat = llvm::ConstantPointerNull::get(LLVMType<float*>::get(C));
    std::vector<llvm::Type*> types;
    test4.types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(1), types.size());
    CPPUNIT_ASSERT(nullfloat->getType() == types.front());
    CPPUNIT_ASSERT(test4.fold({nullfloat}, C) == nullptr);
}

void
TestFunctionTypes::testIRFunctions()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::IRFunctionBase;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    // Small test to check the templated version of IRFunction::types.
    // All other checks work with the IRFunctionBase class
    {
        using openvdb::ax::codegen::IRFunction;

        static auto generate =
            [](const std::vector<llvm::Value*>&,
               llvm::IRBuilder<>&) -> llvm::Value*
        { return nullptr; };

        IRFunction<double(bool,
                int16_t*,
                int32_t(*)[1],
                int64_t,
                float*,
                double(*)[2])>
            mix("mix", generate);

        CPPUNIT_ASSERT_EQUAL(std::string("mix"),
            std::string(mix.symbol()));

        std::vector<llvm::Type*> types;
        llvm::Type* returnType = mix.types(types, C);
        CPPUNIT_ASSERT(returnType->isDoubleTy());
        CPPUNIT_ASSERT_EQUAL(size_t(6), types.size());
        CPPUNIT_ASSERT(types[0] == llvm::Type::getInt1Ty(C));
        CPPUNIT_ASSERT(types[1] == llvm::Type::getInt16Ty(C)->getPointerTo());
        CPPUNIT_ASSERT(types[2] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1)->getPointerTo());
        CPPUNIT_ASSERT(types[3] == llvm::Type::getInt64Ty(C));
        CPPUNIT_ASSERT(types[4] == llvm::Type::getFloatTy(C)->getPointerTo());
        CPPUNIT_ASSERT(types[5] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo());
    }

    llvm::Module& M = state.module();
    llvm::IRBuilder<> B(state.scratchBlock("TestFunction"));
    llvm::Function* BaseFunction = B.GetInsertBlock()->getParent();
    B.SetInsertPoint(finalizeFunction(B));

    // Build the following function:
    //   float test(float a, float b) {
    //     float c = a + b;
    //     return c;
    //   }

    // how to handle the terminating instruction
    int termMode = 0;
    std::string expectedName;

    auto generate =
        [&B, &M, &termMode, &expectedName]
        (const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& _B) -> llvm::Value*
    {
        // test the builder is pointing to the correct location
        CPPUNIT_ASSERT(&B != &_B);
        llvm::BasicBlock* Block = _B.GetInsertBlock();
        CPPUNIT_ASSERT(Block);
        CPPUNIT_ASSERT(Block->empty());
        llvm::Function* F = Block->getParent();
        CPPUNIT_ASSERT(F);
        CPPUNIT_ASSERT_EQUAL(expectedName, std::string(F->getName()));
        llvm::Module* _M = F->getParent();
        CPPUNIT_ASSERT_EQUAL(&M, _M);

        CPPUNIT_ASSERT_EQUAL(size_t(2), args.size());
        CPPUNIT_ASSERT(args[0] == llvm::cast<llvm::Value>(F->arg_begin()));
        CPPUNIT_ASSERT(args[1] == llvm::cast<llvm::Value>(F->arg_begin()+1));
        CPPUNIT_ASSERT(args[0]->getType()->isFloatTy());
        CPPUNIT_ASSERT(args[1]->getType()->isFloatTy());

        llvm::Value* result = _B.CreateFAdd(args[0], args[1]);
        if (termMode == 0) return _B.CreateRet(result);
        if (termMode == 1) return result;
        if (termMode == 2) return nullptr;
        CPPUNIT_ASSERT(false);
        return nullptr;
    };

    llvm::Function* function = nullptr;

    expectedName = "ax.ir.test";
    Function::Ptr test(new TestIRFunction({
            llvm::Type::getFloatTy(C),
            llvm::Type::getFloatTy(C)
        },
        llvm::Type::getFloatTy(C),
        expectedName, generate));

    // Test function prototype creation
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.test"));
    // detached
    function = test->create(C);
    llvm::Function* function2 = test->create(C);
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.test"));
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT(function != function2);
    CPPUNIT_ASSERT(!function->isVarArg());
    CPPUNIT_ASSERT_EQUAL(size_t(2), function->arg_size());

    llvm::FunctionType* ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isFloatTy());
    CPPUNIT_ASSERT_EQUAL(2u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0)->isFloatTy());
    CPPUNIT_ASSERT(ftype->getParamType(1)->isFloatTy());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());
    delete function;
    delete function2;

    // Test function creation with module and IR generation
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.ir.test"));
    CPPUNIT_ASSERT(!function->empty());
    llvm::BasicBlock* BB = &(function->getEntryBlock());
    // two instructions - the add and return
    CPPUNIT_ASSERT_EQUAL(size_t(2), BB->size());
    auto iter = BB->begin();
    llvm::BinaryOperator* binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    CPPUNIT_ASSERT(binary);
    CPPUNIT_ASSERT_EQUAL(llvm::Instruction::FAdd, binary->getOpcode());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    llvm::ReturnInst* ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    CPPUNIT_ASSERT(ret);
    llvm::Value* rvalue = ret->getReturnValue();
    CPPUNIT_ASSERT(rvalue);
    CPPUNIT_ASSERT(rvalue->getType()->isFloatTy());
    // the return is the result of the bin op
    CPPUNIT_ASSERT_EQUAL(rvalue, llvm::cast<llvm::Value>(binary));

    ++iter;
    CPPUNIT_ASSERT(BB->end() == iter);

    // additional call should match
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));
    // verify IR
    VERIFY_FUNCTION_IR(function);

    // Test call

    llvm::Value* fp1 = LLVMType<float>::get(C, 1.0f);
    llvm::Value* fp2 = LLVMType<float>::get(C, 2.0f);
    llvm::Value* result = test->call({fp1, fp2}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(fp1, call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(fp2, call->getArgOperand(1));
    // Test the builder is pointing to the correct location
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(call),
        llvm::cast<llvm::Value>(--B.GetInsertPoint()));

    // verify IR
    VERIFY_FUNCTION_IR(function);
    VERIFY_FUNCTION_IR(BaseFunction);
    VERIFY_MODULE_IR(&M);

    //
    // Test auto return - the IRFunctionBase should handle the return
    // also test that calling Function::call correctly creates the
    // function in the module

    expectedName = "ax.ir.autoret.test";
    termMode = 1;
    test.reset(new TestIRFunction({
            llvm::Type::getFloatTy(C),
            llvm::Type::getFloatTy(C)
        },
        llvm::Type::getFloatTy(C),
        expectedName, generate));

    CPPUNIT_ASSERT(!M.getFunction("ax.ir.autoret.test"));
    result = test->call({fp1, fp2}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    function = M.getFunction("ax.ir.autoret.test");
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(fp1, call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(fp2, call->getArgOperand(1));

    CPPUNIT_ASSERT(!function->empty());
    BB = &(function->getEntryBlock());
    // two instructions - the add and return
    CPPUNIT_ASSERT_EQUAL(size_t(2), BB->size());
    iter = BB->begin();
    binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    CPPUNIT_ASSERT(binary);
    CPPUNIT_ASSERT_EQUAL(llvm::Instruction::FAdd, binary->getOpcode());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    CPPUNIT_ASSERT(ret);
    rvalue = ret->getReturnValue();
    CPPUNIT_ASSERT(rvalue);
    CPPUNIT_ASSERT(rvalue->getType()->isFloatTy());
    // the return is the result of the bin op
    CPPUNIT_ASSERT_EQUAL(rvalue, llvm::cast<llvm::Value>(binary));

    ++iter;
    CPPUNIT_ASSERT(BB->end() == iter);

    // Test the builder is pointing to the correct location
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(call),
        llvm::cast<llvm::Value>(--B.GetInsertPoint()));

    // verify
    VERIFY_FUNCTION_IR(function);
    VERIFY_FUNCTION_IR(BaseFunction);
    VERIFY_MODULE_IR(&M);

    // Test invalid return

    expectedName = "ax.ir.retnull.test";
    termMode = 2;
    test.reset(new TestIRFunction({
            llvm::Type::getFloatTy(C),
            llvm::Type::getFloatTy(C)
        },
        llvm::Type::getFloatTy(C),
        expectedName, generate));

    CPPUNIT_ASSERT(!M.getFunction("ax.ir.retnull.test"));
    // will throw as the function expects a float ret, not void or null
    // NOTE: The function will still be created, but be in an invaid state
    CPPUNIT_ASSERT_THROW(test->create(M), openvdb::AXCodeGenError);
    function = M.getFunction("ax.ir.retnull.test");
    CPPUNIT_ASSERT(function);

    result = test->call({fp1, fp2}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    CPPUNIT_ASSERT_EQUAL(function, call->getCalledFunction());
    CPPUNIT_ASSERT_EQUAL(2u, getNumArgFromCallInst(call));
    CPPUNIT_ASSERT_EQUAL(fp1, call->getArgOperand(0));
    CPPUNIT_ASSERT_EQUAL(fp2, call->getArgOperand(1));

    BB = &(function->getEntryBlock());
    // two instructions - the add and return
    CPPUNIT_ASSERT_EQUAL(size_t(2), BB->size());
    iter = BB->begin();
    binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    CPPUNIT_ASSERT(binary);
    CPPUNIT_ASSERT_EQUAL(llvm::Instruction::FAdd, binary->getOpcode());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    CPPUNIT_ASSERT(ret);
    CPPUNIT_ASSERT(!ret->getReturnValue());

    ++iter;
    CPPUNIT_ASSERT(BB->end() == iter);

    // Test the builder is pointing to the correct location
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    CPPUNIT_ASSERT_EQUAL(llvm::cast<llvm::Value>(call),
        llvm::cast<llvm::Value>(--B.GetInsertPoint()));

    // verify - function is invalid as it returns void but the
    // prototype wants a float
    VERIFY_FUNCTION_IR_INVALID(function);
    VERIFY_FUNCTION_IR(BaseFunction);
    VERIFY_MODULE_IR_INVALID(&M);

    //
    // Test embedded IR

    auto embdedGen = [&B, &M]
        (const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& _B) -> llvm::Value*
    {
        // test the builder is pointing to the correct location
        // note, for embedded IR, the same builder will be used
        CPPUNIT_ASSERT_EQUAL(&B, &_B);
        llvm::BasicBlock* Block = _B.GetInsertBlock();
        CPPUNIT_ASSERT(Block);
        CPPUNIT_ASSERT(!Block->empty());
        llvm::Function* F = Block->getParent();
        CPPUNIT_ASSERT(F);
        CPPUNIT_ASSERT_EQUAL(std::string("TestFunction"), std::string(F->getName()));
        CPPUNIT_ASSERT_EQUAL(&M, F->getParent());
        CPPUNIT_ASSERT_EQUAL(size_t(2), args.size());
        CPPUNIT_ASSERT(args[0]->getType()->isFloatTy());
        CPPUNIT_ASSERT(args[1]->getType()->isFloatTy());
        // Can't just do a CreateFAdd as the IR builder won't actually even
        // write the instruction as its const and unused - so store in a new
        // alloc
        llvm::Value* alloc = _B.CreateAlloca(args[0]->getType());
        _B.CreateStore(args[0], alloc);
        return openvdb::ax::codegen::ir_load(_B, alloc);
    };

    test.reset(new TestIRFunction({
            llvm::Type::getFloatTy(C),
            llvm::Type::getFloatTy(C)
        },
        llvm::Type::getFloatTy(C),
        "ax.ir.embed.test", embdedGen));
    static_cast<IRFunctionBase&>(*test).setEmbedIR(true);

    // test create does nothing
    CPPUNIT_ASSERT(test->create(C) == nullptr);
    CPPUNIT_ASSERT(test->create(M) == nullptr);

    // test call
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.embed.test"));
    result = test->call({fp1, fp2}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.embed.test"));
    auto IP = B.GetInsertPoint();
    // check the prev instructions are as expected
    CPPUNIT_ASSERT(llvm::isa<llvm::LoadInst>(--IP));
    CPPUNIT_ASSERT(llvm::isa<llvm::StoreInst>(--IP));
    CPPUNIT_ASSERT(llvm::isa<llvm::AllocaInst>(--IP));
    // Test the builder is pointing to the correct location
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
}

void
TestFunctionTypes::testSRETFunctions()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::CFunctionSRet;
    using openvdb::ax::codegen::IRFunctionSRet;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();
    llvm::IRBuilder<> B(state.scratchBlock());
    std::vector<llvm::Type*> types;
    llvm::Type* type = nullptr;
    llvm::Value* result = nullptr;
    llvm::Function* function = nullptr;
    llvm::FunctionType* ftype = nullptr;
    Function::SignatureMatch match;
    std::ostringstream os;

    B.SetInsertPoint(finalizeFunction(B));
    llvm::Function* BaseFunction = B.GetInsertBlock()->getParent();

    // test C SRET

    static auto csret = [](float(*output)[3]) { (*output)[0] = 1.0f; };
    Function::Ptr test(new CFunctionSRet<void(float(*)[3])>
        ("ax.c.test", (void(*)(float(*)[3]))(csret)));

    // test types
    llvm::Type* vec3f = llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3);
    type = test->types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(1), types.size());
    CPPUNIT_ASSERT(types[0] == vec3f->getPointerTo(0));
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());
    // test various getters
    CPPUNIT_ASSERT_EQUAL(std::string("ax.c.test"), std::string(test->symbol()));
    CPPUNIT_ASSERT_EQUAL(size_t(1), test->size());
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(0)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("vec3f name()"), os.str());

    // test match
    match = test->match({vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::None);
    match = test->match({}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);
    match = test->Function::match({vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);

    // test create
    CPPUNIT_ASSERT(!M.getFunction("ax.c.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.c.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(1), function->arg_size());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(1u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == vec3f->getPointerTo());

    // test call - sret function do not return the CallInst as the value
    result = test->call({}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(!llvm::dyn_cast<llvm::CallInst>(result));
    CPPUNIT_ASSERT(result->getType() == vec3f->getPointerTo());
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

    VERIFY_FUNCTION_IR(function);
    VERIFY_MODULE_IR(&M);

    //
    // test sret with two arguments

    static auto csret2 = [](float(*output)[3], float(*input)[3]) { (*output)[0] = (*input)[0]; };
    test.reset(new CFunctionSRet<void(float(*)[3],float(*)[3])>
        ("ax.c2.test", (void(*)(float(*)[3],float(*)[3]))(csret2)));
    types.clear();

    // test types
    type = test->types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(2), types.size());
    CPPUNIT_ASSERT(types[0] == vec3f->getPointerTo(0));
    CPPUNIT_ASSERT(types[1] == vec3f->getPointerTo(0));
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());
    // test various getters
    CPPUNIT_ASSERT_EQUAL(std::string("ax.c2.test"), std::string(test->symbol()));
    CPPUNIT_ASSERT_EQUAL(size_t(2), test->size());
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(0)));
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(1)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("vec3f name(vec3f)"), os.str());

    // test match
    match = test->match({vec3f->getPointerTo(),vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::None);
    match = test->match({vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);
    match = test->match({}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::None);
    match = test->Function::match({vec3f->getPointerTo(),vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);

    // test create
    CPPUNIT_ASSERT(!M.getFunction("ax.c2.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT(M.getFunction("ax.c2.test"));
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));
    CPPUNIT_ASSERT(function->empty());
    CPPUNIT_ASSERT_EQUAL(size_t(2), function->arg_size());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(2u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == vec3f->getPointerTo());
    CPPUNIT_ASSERT(ftype->getParamType(1) == vec3f->getPointerTo());

    // test call - sret function do not return the CallInst as the value
    llvm::Value* f32c0 = LLVMType<float>::get(C, 0.0f); // float
    llvm::Value* vec3fv = openvdb::ax::codegen::arrayPack({f32c0,f32c0,f32c0}, B); // vec3f
    result = test->call({vec3fv}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(!llvm::dyn_cast<llvm::CallInst>(result));
    CPPUNIT_ASSERT(result->getType() == vec3f->getPointerTo());
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

    VERIFY_FUNCTION_IR(function);
    VERIFY_MODULE_IR(&M);

    //
    // test IR SRET

    // Build the following function:
    //   void test(vec3f* a) {
    //     a[0] = 1.0f;
    //   }
    // which has a front interface of:
    //   vec3f test() { vec3f a; a[0] = 1; return a;};
    //

    auto generate = [&B, &M]
        (const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& _B) -> llvm::Value*
    {
        // test the builder is pointing to the correct location
        CPPUNIT_ASSERT(&B != &_B);
        llvm::BasicBlock* Block = _B.GetInsertBlock();
        CPPUNIT_ASSERT(Block);
        CPPUNIT_ASSERT(Block->empty());
        llvm::Function* F = Block->getParent();
        CPPUNIT_ASSERT(F);
        CPPUNIT_ASSERT_EQUAL(std::string("ax.ir.test"), std::string(F->getName()));
        llvm::Module* _M = F->getParent();
        CPPUNIT_ASSERT_EQUAL(&M, _M);

        CPPUNIT_ASSERT_EQUAL(size_t(1), args.size());
        CPPUNIT_ASSERT(args[0] == llvm::cast<llvm::Value>(F->arg_begin()));
        CPPUNIT_ASSERT(args[0]->getType() ==
            llvm::ArrayType::get(llvm::Type::getFloatTy(_B.getContext()), 3)->getPointerTo());

        llvm::Value* e0 = openvdb::ax::codegen::ir_constgep2_64(_B, args[0], 0, 0);
        _B.CreateStore(LLVMType<float>::get(_B.getContext(), 1.0f), e0);
        return nullptr;
    };

    test.reset(new IRFunctionSRet<void(float(*)[3])>("ax.ir.test", generate));
    types.clear();

    // test types
    type = test->types(types, C);
    CPPUNIT_ASSERT_EQUAL(size_t(1), types.size());
    CPPUNIT_ASSERT(types[0] == vec3f->getPointerTo(0));
    CPPUNIT_ASSERT(type);
    CPPUNIT_ASSERT(type->isVoidTy());
    // test various getters
    CPPUNIT_ASSERT_EQUAL(std::string("ax.ir.test"), std::string(test->symbol()));
    CPPUNIT_ASSERT_EQUAL(size_t(1), test->size());
    CPPUNIT_ASSERT_EQUAL(std::string(""), std::string(test->argName(0)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    CPPUNIT_ASSERT_EQUAL(std::string("vec3f name()"), os.str());

    // test match
    match = test->match({vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::None);
    match = test->match({}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);
    match = test->Function::match({vec3f->getPointerTo()}, C);
    CPPUNIT_ASSERT_EQUAL(match, Function::Explicit);

    // test create
    CPPUNIT_ASSERT(!M.getFunction("ax.ir.test"));
    function = test->create(M);
    CPPUNIT_ASSERT(function);
    CPPUNIT_ASSERT_EQUAL(function, M.getFunction("ax.ir.test"));
    CPPUNIT_ASSERT(!function->empty());

    // test instructions
    llvm::BasicBlock* BB = &(function->getEntryBlock());
    CPPUNIT_ASSERT_EQUAL(size_t(3), BB->size());
    auto iter = BB->begin();
    CPPUNIT_ASSERT(llvm::isa<llvm::GetElementPtrInst>(iter++));
    CPPUNIT_ASSERT(llvm::isa<llvm::StoreInst>(iter++));
    CPPUNIT_ASSERT(llvm::isa<llvm::ReturnInst>(iter++));
    CPPUNIT_ASSERT(BB->end() == iter);

    // additional call should match
    CPPUNIT_ASSERT_EQUAL(function, test->create(M));
    CPPUNIT_ASSERT_EQUAL(size_t(1), function->arg_size());
    CPPUNIT_ASSERT(function->getAttributes().isEmpty());

    // check function type

    ftype = function->getFunctionType();
    CPPUNIT_ASSERT(ftype);
    CPPUNIT_ASSERT(ftype->getReturnType()->isVoidTy());
    CPPUNIT_ASSERT_EQUAL(1u, ftype->getNumParams());
    CPPUNIT_ASSERT(ftype->getParamType(0) == vec3f->getPointerTo());

    // test call - sret function do not return the CallInst as the value
    result = test->call({}, B, /*cast*/false);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(!llvm::dyn_cast<llvm::CallInst>(result));
    CPPUNIT_ASSERT(result->getType() == vec3f->getPointerTo());
    CPPUNIT_ASSERT_EQUAL(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

    VERIFY_FUNCTION_IR(function);
    VERIFY_MODULE_IR(&M);
}

