// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "util.h"

#include <openvdb_ax/codegen/LegacyIR.h>
#include <openvdb_ax/codegen/FunctionTypes.h>

#include <gtest/gtest.h>

#include <llvm/IR/Verifier.h>
#include <llvm/Support/raw_ostream.h>

#include <sstream>

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/// @brief  Dummy derived IR function which implemented types and
///         forwards on the generator
#if LLVM_VERSION_MAJOR <= 15
struct TestIRFunction : public openvdb::ax::codegen::IRFunctionBase
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::IRFunctionBase>::value,
        "Base class destructor is not virtual");
    TestIRFunction(const openvdb::ax::codegen::ArgInfoVector& types,
          openvdb::ax::codegen::ArgInfo ret,
          const std::string& symbol,
          const openvdb::ax::codegen::IRFunctionBase::GeneratorArgumentsCb& gen)
        : openvdb::ax::codegen::IRFunctionBase(symbol, gen, types.size())
        , mTypes(types)
        , mRet([&]() {
            ret.SetIsReturn();
            return ret;
        }())
        , mLLVMTypes([&]() {
            std::vector<llvm::Type*> sig;
            for (auto& type : types) sig.emplace_back(type.GetType());
            return sig;
        }())
        , mLLVMRet(ret.GetType()) {}
    ~TestIRFunction() override = default;

    llvm::Type* types(std::vector<llvm::Type*>& types, llvm::LLVMContext&) const override
    {
        types = mLLVMTypes;
        return mLLVMRet;
    }
    openvdb::ax::codegen::ArgInfo types(
        openvdb::ax::codegen::ArgInfoVector& types,
        llvm::LLVMContext&) const override
    {
        types = mTypes;
        return mRet;
    }
    const openvdb::ax::codegen::ArgInfoVector mTypes;
    const openvdb::ax::codegen::ArgInfo mRet;
    const std::vector<llvm::Type*> mLLVMTypes;
    llvm::Type* mLLVMRet;
};
#else
struct TestIRFunction final : public openvdb::ax::codegen::IRFunctionBase
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::IRFunctionBase>::value,
        "Base class destructor is not virtual");
    TestIRFunction(const openvdb::ax::codegen::ArgInfoVector& types,
          openvdb::ax::codegen::ArgInfo ret,
          const std::string& symbol,
          const openvdb::ax::codegen::IRFunctionBase::GeneratorArgumentsCb& gen)
        : openvdb::ax::codegen::IRFunctionBase(symbol, gen, types.size())
        , mTypes(types)
        , mRet([&]() {
            ret.SetIsReturn();
            return ret;
        }()) {}
    ~TestIRFunction() override = default;
    openvdb::ax::codegen::ArgInfo types(
        openvdb::ax::codegen::ArgInfoVector& types,
        llvm::LLVMContext&) const override
    {
        types = mTypes;
        return mRet;
    }
    const openvdb::ax::codegen::ArgInfoVector mTypes;
    const openvdb::ax::codegen::ArgInfo mRet;
};
#endif


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
    ASSERT_TRUE(valid) << os.str(); \
}
#define VERIFY_MODULE_IR(Module) { \
    std::string error; llvm::raw_string_ostream os(error); \
    const bool valid = !llvm::verifyModule(*Module, &os); \
    ASSERT_TRUE(valid) << os.str(); \
}
#define VERIFY_MODULE_IR_INVALID(Module) { \
    const bool valid = llvm::verifyModule(*Module); \
    ASSERT_TRUE(valid) << "Expected IR to be invalid!"; \
}
#define VERIFY_FUNCTION_IR_INVALID(Function) { \
    const bool valid = llvm::verifyFunction(*Function); \
    ASSERT_TRUE(valid) << "Expected IR to be invalid!"; \
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

class TestFunctionTypes : public ::testing::Test
{
};

TEST_F(TestFunctionTypes, testLLVMTypesFromSignature)
{
    using openvdb::ax::codegen::llvmTypesFromSignature;

    unittest_util::LLVMState state;
    llvm::Type* type = nullptr;
    std::vector<llvm::Type*> types;

    type = llvmTypesFromSignature<void()>(state.context());
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());

    type = llvmTypesFromSignature<void()>(state.context(), &types);
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());
    ASSERT_TRUE(types.empty());

    type = llvmTypesFromSignature<float()>(state.context(), &types);
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isFloatTy());
    ASSERT_TRUE(types.empty());

    type = llvmTypesFromSignature<float(double, int64_t, float(*)[3])>(state.context(), &types);

    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isFloatTy());
    ASSERT_EQ(size_t(3), types.size());
    ASSERT_TRUE(types[0]->isDoubleTy());
    ASSERT_TRUE(types[1]->isIntegerTy(64));
    ASSERT_TRUE(types[2]->isPointerTy());

#if LLVM_VERSION_MAJOR <= 15
    type = types[2]->getPointerElementType();
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isArrayTy());
    type = type->getArrayElementType();
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isFloatTy());
#endif
}

TEST_F(TestFunctionTypes, testArgInfoTypesFromSignature)
{
    using openvdb::ax::codegen::llvmArgTypesFromSignature;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;

    unittest_util::LLVMState state;
    ArgInfoVector types;

    ArgInfo type = llvmArgTypesFromSignature<void()>(state.context());
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(type.IsReturn());
    ASSERT_TRUE(!type.IsPtr());

    type = llvmArgTypesFromSignature<void()>(state.context(), &types);
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(type.IsReturn());
    ASSERT_TRUE(!type.IsPtr());
    ASSERT_TRUE(types.empty());

    type = llvmArgTypesFromSignature<float()>(state.context(), &types);
    ASSERT_TRUE(type.IsNative());
    ASSERT_TRUE(type.IsReturn());
    ASSERT_TRUE(!type.IsPtr());
    ASSERT_TRUE(type.GetType()->isFloatTy());
    ASSERT_TRUE(types.empty());

    type = llvmArgTypesFromSignature<float(double, int64_t, float(*)[3])>(state.context(), &types);

    ASSERT_TRUE(type.IsNative());
    ASSERT_TRUE(type.IsReturn());
    ASSERT_TRUE(!type.IsPtr());
    ASSERT_TRUE(type.GetType()->isFloatTy());
    ASSERT_EQ(size_t(3), types.size());
    ASSERT_TRUE(types[0].GetType()->isDoubleTy());
    ASSERT_TRUE(types[0].IsNative());
    ASSERT_TRUE(!types[0].IsPtr());
    ASSERT_TRUE(types[1].GetType()->isIntegerTy(64));
    ASSERT_TRUE(types[1].IsNative());
    ASSERT_TRUE(!types[1].IsPtr());
    ASSERT_TRUE(types[2].GetType()->isPointerTy());
    ASSERT_TRUE(types[2].IsNative());
    ASSERT_TRUE(types[2].IsPtr());
    ASSERT_TRUE(types[2].NumPtrs() == 1);
    ASSERT_TRUE(types[2].GetUnderlyingType()->isArrayTy());
    ASSERT_TRUE(types[2].GetUnderlyingType()->getArrayElementType()->isFloatTy());
}

TEST_F(TestFunctionTypes, testLLVMFunctionTypeFromSignature)
{
    using openvdb::ax::codegen::llvmFunctionTypeFromSignature;

    unittest_util::LLVMState state;
    llvm::FunctionType* ftype = nullptr;

    ftype = llvmFunctionTypeFromSignature<void()>(state.context());
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(0u, ftype->getNumParams());

    ftype = llvmFunctionTypeFromSignature<float(double, int64_t, float(*)[3])>(state.context());

    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isFloatTy());
    ASSERT_EQ(3u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)->isDoubleTy());
    ASSERT_TRUE(ftype->getParamType(1)->isIntegerTy(64));
    ASSERT_TRUE(ftype->getParamType(2)->isPointerTy());

#if LLVM_VERSION_MAJOR <= 15
    llvm::Type* type = ftype->getParamType(2)->getPointerElementType();
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isArrayTy());
    type = type->getArrayElementType();
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isFloatTy());
#endif
}

TEST_F(TestFunctionTypes, testPrintSignature)
{
    using openvdb::ax::codegen::printSignature;
    using openvdb::ax::codegen::ArgInfoVector;
    using openvdb::ax::codegen::ArgInfo;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> types;
    const llvm::Type* vt = llvm::Type::getVoidTy(C);
    const llvm::Type* int64t = llvm::Type::getInt64Ty(C);
#else
    ArgInfoVector types;
    const ArgInfo vt(llvm::Type::getVoidTy(C));
    const ArgInfo int64t(llvm::Type::getInt64Ty(C));
#endif

    std::ostringstream os;

    printSignature(os, types, vt);
    ASSERT_TRUE(os.str() == "void()");
    os.str("");

    types.emplace_back(llvm::Type::getInt32Ty(C));
    types.emplace_back(llvm::Type::getInt64Ty(C));
    printSignature(os, types, vt);
    ASSERT_EQ(std::string("void(i32; i64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "test");
    ASSERT_EQ(std::string("void test(i32; i64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "", {"one"}, true);
    ASSERT_EQ(std::string("void(int32 one; int64)"), os.str());
    os.str("");

    printSignature(os, types, vt, "", {"one", "two"}, true);
    ASSERT_EQ(std::string("void(int32 one; int64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"one", "two", "three"}, true);
    ASSERT_EQ(std::string("void 1(int32 one; int64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"", "two"}, false);
    ASSERT_EQ(std::string("void 1(i32; i64 two)"), os.str());
    os.str("");

    printSignature(os, types, vt, "1", {"", "two"}, false);
    ASSERT_EQ(std::string("void 1(i32; i64 two)"), os.str());
    os.str("");

#if LLVM_VERSION_MAJOR <= 15
    types.emplace_back(llvm::Type::getInt8PtrTy(C));
#else
    types.emplace_back(llvm::Type::getInt8Ty(C), 1);
#endif
    types.emplace_back(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3));

    printSignature(os, types, int64t, "test", {"", "two"}, true);
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_EQ(std::string("int64 test(int32; int64 two; i8*; vec3i)"), os.str());
#else
    ASSERT_EQ(std::string("int64 test(int32; int64 two; ptr; vec3i)"), os.str());
#endif
    os.str("");

    types.clear();

    printSignature(os, types, int64t, "test", {"", "two"});
    ASSERT_EQ(std::string("i64 test()"), os.str());
    os.str("");
}

// Test Function::create, Function::types and other base methods
TEST_F(TestFunctionTypes, testFunctionCreate)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();

#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> types;
#else
    ArgInfoVector types;
#endif

    std::ostringstream os;

    Function::Ptr test(new unittest_util::TestFunction({ ArgInfo(llvm::Type::getInt32Ty(C)) },
        ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));

    // test types
    auto type = test->types(types, C);
    ASSERT_EQ(size_t(1), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(types[0]->isIntegerTy(32));
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());
#else
    ASSERT_TRUE(types[0].GetType()->isIntegerTy(32));
    ASSERT_TRUE(types[0].IsNative());
    ASSERT_TRUE(!types[0].IsReturn());
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(type.IsReturn());
#endif

    // test various getters
    ASSERT_EQ(std::string("ax.test"), std::string(test->symbol()));
    ASSERT_EQ(size_t(1), test->size());
    ASSERT_EQ(std::string(""), std::string(test->argName(0)));
    ASSERT_EQ(std::string(""), std::string(test->argName(1)));

    // test create detached
    llvm::Function* function = test->create(C);
    llvm::Function* function2 = test->create(C);
    // additional create call should create a new function
    ASSERT_TRUE(function != function2);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(1), function->arg_size());

    llvm::FunctionType* ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(1u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)->isIntegerTy(32));
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;
    delete function2;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.test"));
    function = test->create(M);
    // additional call should match
    ASSERT_EQ(function, test->create(M));
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.test"));
    ASSERT_EQ(function, M.getFunction("ax.test"));
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(1), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(1u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)->isIntegerTy(32));
    ASSERT_TRUE(function->getAttributes().isEmpty());

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("void name(int32)"), os.str());

    //
    // Test empty signature

    test.reset(new unittest_util::TestFunction({}, ArgInfo(llvm::Type::getInt32Ty(C)), "ax.empty.test"));
    types.clear();

    // test types
    type = test->types(types, C);
    ASSERT_EQ(size_t(0), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isIntegerTy(32));
#else
    ASSERT_TRUE(type.GetType()->isIntegerTy(32));
#endif
    // test various getters
    ASSERT_EQ(std::string("ax.empty.test"), std::string(test->symbol()));
    ASSERT_EQ(size_t(0), test->size());
    ASSERT_EQ(std::string(""), std::string(test->argName(0)));

    // test create detached
    function = test->create(C);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(0), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isIntegerTy(32));
    ASSERT_EQ(0u, ftype->getNumParams());
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.empty.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.empty.test"));
    ASSERT_EQ(function, M.getFunction("ax.empty.test"));
    ASSERT_EQ(function, test->create(M));

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("int32 name()"), os.str());

    //
    // Test scalar types

    test.reset(new unittest_util::TestFunction({
            ArgInfo(llvm::Type::getInt1Ty(C)),
            ArgInfo(llvm::Type::getInt16Ty(C)),
            ArgInfo(llvm::Type::getInt32Ty(C)),
            ArgInfo(llvm::Type::getInt64Ty(C)),
            ArgInfo(llvm::Type::getFloatTy(C)),
            ArgInfo(llvm::Type::getDoubleTy(C)),
        },
        ArgInfo(llvm::Type::getInt16Ty(C)), "ax.scalars.test"));
    types.clear();

    ASSERT_EQ(std::string("ax.scalars.test"), std::string(test->symbol()));

    type = test->types(types, state.context());
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isIntegerTy(16));
    ASSERT_TRUE(types[0]->isIntegerTy(1));
    ASSERT_TRUE(types[1]->isIntegerTy(16));
    ASSERT_TRUE(types[2]->isIntegerTy(32));
    ASSERT_TRUE(types[3]->isIntegerTy(64));
    ASSERT_TRUE(types[4]->isFloatTy());
    ASSERT_TRUE(types[5]->isDoubleTy());
#else
    ASSERT_TRUE(type.GetType()->isIntegerTy(16));
    ASSERT_TRUE(types[0].GetType()->isIntegerTy(1));
    ASSERT_TRUE(types[1].GetType()->isIntegerTy(16));
    ASSERT_TRUE(types[2].GetType()->isIntegerTy(32));
    ASSERT_TRUE(types[3].GetType()->isIntegerTy(64));
    ASSERT_TRUE(types[4].GetType()->isFloatTy());
    ASSERT_TRUE(types[5].GetType()->isDoubleTy());
#endif

    // test create detached
    function = test->create(C);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isIntegerTy(16));
    ASSERT_EQ(6u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)->isIntegerTy(1));
    ASSERT_TRUE(ftype->getParamType(1)->isIntegerTy(16));
    ASSERT_TRUE(ftype->getParamType(2)->isIntegerTy(32));
    ASSERT_TRUE(ftype->getParamType(3)->isIntegerTy(64));
    ASSERT_TRUE(ftype->getParamType(4)->isFloatTy());
    ASSERT_TRUE(ftype->getParamType(5)->isDoubleTy());
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.scalars.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.scalars.test"));
    ASSERT_EQ(function, M.getFunction("ax.scalars.test"));
    ASSERT_EQ(function, test->create(M));

    // test print
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("int16 name(bool; int16; int32; int64; float; double)"), os.str());

    types.clear();

    //
    // Test scalar ptrs types

    test.reset(new unittest_util::TestFunction({
            ArgInfo(llvm::Type::getInt1Ty(C), 1),
            ArgInfo(llvm::Type::getInt16Ty(C), 1),
            ArgInfo(llvm::Type::getInt32Ty(C), 1),
            ArgInfo(llvm::Type::getInt64Ty(C), 1),
            ArgInfo(llvm::Type::getFloatTy(C), 1),
            ArgInfo(llvm::Type::getDoubleTy(C), 1)
        },
        ArgInfo(llvm::Type::getInt32Ty(C)), "ax.scalarptrs.test"));

    ASSERT_EQ(std::string("ax.scalarptrs.test"), std::string(test->symbol()));

    type = test->types(types, C);
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(type->isIntegerTy(32));
    ASSERT_TRUE(types[0] == llvm::PointerType::get(llvm::Type::getInt1Ty(C), 0));
    ASSERT_TRUE(types[1] == llvm::PointerType::get(llvm::Type::getInt16Ty(C), 0));
    ASSERT_TRUE(types[2] == llvm::PointerType::get(llvm::Type::getInt32Ty(C), 0));
    ASSERT_TRUE(types[3] == llvm::PointerType::get(llvm::Type::getInt64Ty(C), 0));
    ASSERT_TRUE(types[4] == llvm::PointerType::get(llvm::Type::getFloatTy(C), 0));
    ASSERT_TRUE(types[5] == llvm::PointerType::get(llvm::Type::getDoubleTy(C), 0));
#else
    ASSERT_TRUE(type.GetType()->isIntegerTy(32));
    ASSERT_TRUE(types[0].GetUnderlyingType()->isIntegerTy(1));
    ASSERT_TRUE(types[1].GetUnderlyingType()->isIntegerTy(16));
    ASSERT_TRUE(types[2].GetUnderlyingType()->isIntegerTy(32));
    ASSERT_TRUE(types[3].GetUnderlyingType()->isIntegerTy(64));
    ASSERT_TRUE(types[4].GetUnderlyingType()->isFloatTy());
    ASSERT_TRUE(types[5].GetUnderlyingType()->isDoubleTy());
    ASSERT_TRUE(types[0].NumPtrs() == 1);
    ASSERT_TRUE(types[1].NumPtrs() == 1);
    ASSERT_TRUE(types[2].NumPtrs() == 1);
    ASSERT_TRUE(types[3].NumPtrs() == 1);
    ASSERT_TRUE(types[4].NumPtrs() == 1);
    ASSERT_TRUE(types[5].NumPtrs() == 1);
#endif

    // test create detached
    function = test->create(C);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isIntegerTy(32));
    ASSERT_EQ(6u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0) == llvm::PointerType::get(llvm::Type::getInt1Ty(C), 0));
    ASSERT_TRUE(ftype->getParamType(1) == llvm::PointerType::get(llvm::Type::getInt16Ty(C), 0));
    ASSERT_TRUE(ftype->getParamType(2) == llvm::PointerType::get(llvm::Type::getInt32Ty(C), 0));
    ASSERT_TRUE(ftype->getParamType(3) == llvm::PointerType::get(llvm::Type::getInt64Ty(C), 0));
    ASSERT_TRUE(ftype->getParamType(4) == llvm::PointerType::get(llvm::Type::getFloatTy(C), 0));
    ASSERT_TRUE(ftype->getParamType(5) == llvm::PointerType::get(llvm::Type::getDoubleTy(C), 0));
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.scalarptrs.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.scalarptrs.test"));
    ASSERT_EQ(function, M.getFunction("ax.scalarptrs.test"));
    ASSERT_EQ(function, test->create(M));

    //
    // Test array ptrs types

    test.reset(new unittest_util::TestFunction({
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 1),  // vec2i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2), 1),  // vec2f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 1), // vec2d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1),  // vec3i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3), 1),  // vec3f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3), 1), // vec3d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4), 1),  // vec4i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4), 1),  // vec4f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4), 1), // vec4d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9), 1),  // ix9 (not supported by ax)
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9), 1),  // mat3f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9), 1), // mat3d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16), 1),  // ix16 (not supported by ax)
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16), 1),  // mat3f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16), 1)  // mat3d
        },
        ArgInfo(llvm::Type::getInt64Ty(C)), "ax.arrayptrs.test"));
    types.clear();

    type = test->types(types, C);
    ASSERT_EQ(size_t(15), types.size());

#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(type->isIntegerTy(64));
    ASSERT_TRUE(types[0] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo());
    ASSERT_TRUE(types[1] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2)->getPointerTo());
    ASSERT_TRUE(types[2] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2)->getPointerTo());
    ASSERT_TRUE(types[3] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo());
    ASSERT_TRUE(types[4] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3)->getPointerTo());
    ASSERT_TRUE(types[5] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3)->getPointerTo());
    ASSERT_TRUE(types[6] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4)->getPointerTo());
    ASSERT_TRUE(types[7] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4)->getPointerTo());
    ASSERT_TRUE(types[8] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4)->getPointerTo());
    ASSERT_TRUE(types[9] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9)->getPointerTo());
    ASSERT_TRUE(types[10] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9)->getPointerTo());
    ASSERT_TRUE(types[11] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9)->getPointerTo());
    ASSERT_TRUE(types[12] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16)->getPointerTo());
    ASSERT_TRUE(types[13] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16)->getPointerTo());
    ASSERT_TRUE(types[14] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16)->getPointerTo());
#else
    ASSERT_TRUE(types[0].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2));
    ASSERT_TRUE(types[1].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2));
    ASSERT_TRUE(types[2].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2));
    ASSERT_TRUE(types[3].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3));
    ASSERT_TRUE(types[4].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3));
    ASSERT_TRUE(types[5].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3));
    ASSERT_TRUE(types[6].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4));
    ASSERT_TRUE(types[7].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4));
    ASSERT_TRUE(types[8].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4));
    ASSERT_TRUE(types[9].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9));
    ASSERT_TRUE(types[10].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9));
    ASSERT_TRUE(types[11].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9));
    ASSERT_TRUE(types[12].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16));
    ASSERT_TRUE(types[13].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16));
    ASSERT_TRUE(types[14].GetUnderlyingType() == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16));
    ASSERT_TRUE(types[0].NumPtrs() == 1);
    ASSERT_TRUE(types[1].NumPtrs() == 1);
    ASSERT_TRUE(types[2].NumPtrs() == 1);
    ASSERT_TRUE(types[3].NumPtrs() == 1);
    ASSERT_TRUE(types[4].NumPtrs() == 1);
    ASSERT_TRUE(types[5].NumPtrs() == 1);
    ASSERT_TRUE(types[6].NumPtrs() == 1);
    ASSERT_TRUE(types[7].NumPtrs() == 1);
    ASSERT_TRUE(types[8].NumPtrs() == 1);
    ASSERT_TRUE(types[9].NumPtrs() == 1);
    ASSERT_TRUE(types[10].NumPtrs() == 1);
    ASSERT_TRUE(types[11].NumPtrs() == 1);
    ASSERT_TRUE(types[12].NumPtrs() == 1);
    ASSERT_TRUE(types[13].NumPtrs() == 1);
    ASSERT_TRUE(types[14].NumPtrs() == 1);
#endif

    // test create detached
    function = test->create(C);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(15), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isIntegerTy(64));
    ASSERT_EQ(15u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 0));
    ASSERT_TRUE(ftype->getParamType(1)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2), 0));
    ASSERT_TRUE(ftype->getParamType(2)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 0));
    ASSERT_TRUE(ftype->getParamType(3)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 0));
    ASSERT_TRUE(ftype->getParamType(4)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3), 0));
    ASSERT_TRUE(ftype->getParamType(5)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3), 0));
    ASSERT_TRUE(ftype->getParamType(6)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4), 0));
    ASSERT_TRUE(ftype->getParamType(7)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4), 0));
    ASSERT_TRUE(ftype->getParamType(8)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4), 0));
    ASSERT_TRUE(ftype->getParamType(9)  == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9), 0));
    ASSERT_TRUE(ftype->getParamType(10) == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9), 0));
    ASSERT_TRUE(ftype->getParamType(11) == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9), 0));
    ASSERT_TRUE(ftype->getParamType(12) == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16), 0));
    ASSERT_TRUE(ftype->getParamType(13) == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16), 0));
    ASSERT_TRUE(ftype->getParamType(14) == llvm::PointerType::get(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16), 0));
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.arrayptrs.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.arrayptrs.test"));
    ASSERT_EQ(function, M.getFunction("ax.arrayptrs.test"));
    ASSERT_EQ(function, test->create(M));

    // test print - note mat/i types are not ax types
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_EQ(std::string("int64 name(vec2i; vec2f; vec2d; vec3i; vec3f; vec3d;"
        " vec4i; vec4f; vec4d; [9 x i32]*; mat3f; mat3d; [16 x i32]*; mat4f; mat4d)"),
        os.str());
#else
    ASSERT_EQ(std::string("int64 name(vec2i; vec2f; vec2d; vec3i; vec3f; vec3d;"
        " vec4i; vec4f; vec4d; ptr; mat3f; mat3d; ptr; mat4f; mat4d)"),
        os.str());
#endif

    //
    // Test void ptr arguments

    test.reset(new unittest_util::TestFunction({
            ArgInfo(llvm::Type::getInt8Ty(C), 1),
            ArgInfo(llvm::Type::getInt8Ty(C), 2),
            ArgInfo(llvm::Type::getInt8Ty(C), 3),
            ArgInfo(llvm::Type::getFloatTy(C), 1),
            ArgInfo(llvm::Type::getFloatTy(C), 2),
            ArgInfo(llvm::Type::getFloatTy(C), 3)
        },
        ArgInfo(llvm::Type::getVoidTy(C)), "ax.vptrs.test"));
    types.clear();

    // Note that C++ bindings will convert void* to i8* but they should be
    // unmodified in this example where we use the derived TestFunction

    type = test->types(types, C);
    ASSERT_EQ(size_t(6), types.size());

#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(type->isVoidTy());
    ASSERT_TRUE(types[0] == llvm::Type::getInt8Ty(C)->getPointerTo());
    ASSERT_TRUE(types[1] == llvm::Type::getInt8Ty(C)->getPointerTo()->getPointerTo());
    ASSERT_TRUE(types[2] == llvm::Type::getInt8Ty(C)->getPointerTo()->getPointerTo()->getPointerTo());
    ASSERT_TRUE(types[3] == llvm::Type::getFloatTy(C)->getPointerTo());
    ASSERT_TRUE(types[4] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo());
    ASSERT_TRUE(types[5] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
#else
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(types[0].GetUnderlyingType() == llvm::Type::getInt8Ty(C));
    ASSERT_TRUE(types[1].GetUnderlyingType() == llvm::Type::getInt8Ty(C));
    ASSERT_TRUE(types[2].GetUnderlyingType() == llvm::Type::getInt8Ty(C));
    ASSERT_TRUE(types[3].GetUnderlyingType() == llvm::Type::getFloatTy(C));
    ASSERT_TRUE(types[4].GetUnderlyingType() == llvm::Type::getFloatTy(C));
    ASSERT_TRUE(types[5].GetUnderlyingType() == llvm::Type::getFloatTy(C));
    ASSERT_TRUE(types[0].NumPtrs() == 1);
    ASSERT_TRUE(types[1].NumPtrs() == 2);
    ASSERT_TRUE(types[2].NumPtrs() == 3);
    ASSERT_TRUE(types[3].NumPtrs() == 1);
    ASSERT_TRUE(types[4].NumPtrs() == 2);
    ASSERT_TRUE(types[5].NumPtrs() == 3);
#endif

    // test create detached
    function = test->create(C);
    ASSERT_TRUE(function);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(6), function->arg_size());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(6u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0) == llvm::PointerType::get(llvm::Type::getInt8Ty(C), 0));
    ASSERT_TRUE(ftype->getParamType(1) == llvm::PointerType::get(llvm::PointerType::get(llvm::Type::getInt8Ty(C), 0), 0));
    ASSERT_TRUE(ftype->getParamType(2) == llvm::PointerType::get(llvm::PointerType::get(llvm::PointerType::get(llvm::Type::getInt8Ty(C), 0), 0), 0));
    ASSERT_TRUE(ftype->getParamType(3) == llvm::PointerType::get(llvm::Type::getFloatTy(C), 0));
    ASSERT_TRUE(ftype->getParamType(4) == llvm::PointerType::get(llvm::PointerType::get(llvm::Type::getFloatTy(C), 0), 0));
    ASSERT_TRUE(ftype->getParamType(5) == llvm::PointerType::get(llvm::PointerType::get(llvm::PointerType::get(llvm::Type::getFloatTy(C), 0), 0), 0));
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;

    // test create with a module (same as above, but check inserted into M)
    ASSERT_TRUE(!M.getFunction("ax.vptrs.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.vptrs.test"));
    ASSERT_EQ(function, M.getFunction("ax.vptrs.test"));
    ASSERT_EQ(function, test->create(M));

    //
    // Test creation with builder methods
    // @note  These methods may be moved to the constructor in the future

    ASSERT_TRUE(test->dependencies().empty());
    ASSERT_TRUE(!test->hasParamAttribute(0, llvm::Attribute::ReadOnly));
    ASSERT_TRUE(!test->hasParamAttribute(-1, llvm::Attribute::ReadOnly));

    test->setDependencies({"dep"});
    ASSERT_EQ(size_t(1), test->dependencies().size());
    ASSERT_EQ(std::string("dep"), std::string(test->dependencies().front()));

    test->setDependencies({});
    ASSERT_TRUE(test->dependencies().empty());

    test->setFnAttributes({llvm::Attribute::ReadOnly});
    test->setRetAttributes({llvm::Attribute::NoAlias});
    test->setParamAttributes(1, {llvm::Attribute::WriteOnly});
    test->setParamAttributes(-1, {llvm::Attribute::WriteOnly});

    ASSERT_TRUE(!test->hasParamAttribute(0, llvm::Attribute::WriteOnly));
    ASSERT_TRUE(!test->hasParamAttribute(2, llvm::Attribute::WriteOnly));
    ASSERT_TRUE(test->hasParamAttribute(1, llvm::Attribute::WriteOnly));
    ASSERT_TRUE(test->hasParamAttribute(-1, llvm::Attribute::WriteOnly));

    function = test->create(C);
    ASSERT_TRUE(function);
    llvm::AttributeList list = function->getAttributes();
    ASSERT_TRUE(!list.isEmpty());
    ASSERT_TRUE(!list.hasParamAttrs(0));
    ASSERT_TRUE(!list.hasParamAttrs(2));
    ASSERT_TRUE(list.hasParamAttr(1, llvm::Attribute::WriteOnly));
#if LLVM_VERSION_MAJOR <= 13
    ASSERT_TRUE(list.hasFnAttribute(llvm::Attribute::ReadOnly));
    ASSERT_TRUE(list.hasAttribute(llvm::AttributeList::ReturnIndex, llvm::Attribute::NoAlias));
#else
    ASSERT_TRUE(list.hasFnAttr(llvm::Attribute::ReadOnly));
    ASSERT_TRUE(list.hasRetAttr(llvm::Attribute::NoAlias));
#endif
    delete function;
}

// Test Function::call
TEST_F(TestFunctionTypes, testFunctionCall)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;
    using openvdb::ax::codegen::NativeArguments;
    using openvdb::ax::codegen::Arguments;
    using openvdb::ax::codegen::Value;

    //

    {
        unittest_util::LLVMState state;
        llvm::LLVMContext& C = state.context();
        llvm::Module& M = state.module();
        llvm::IRBuilder<> B(state.scratchBlock());
        llvm::Function* BaseFunction = B.GetInsertBlock()->getParent();

        Function::Ptr test(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt32Ty(C))},
            ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));

        llvm::Function* function = test->create(M);
        llvm::Value* arg = B.getInt32(1);
        llvm::Value* result = test->call({arg}, B);
        ASSERT_TRUE(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        ASSERT_EQ(arg, call->getArgOperand(0));
        // Test the builder is pointing to the correct location
        ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

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

        Function::Ptr test(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt32Ty(C))},
            ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));

        // call first, then create

        llvm::Value* arg = B.getInt32(1);
        llvm::Value* result = test->call({arg}, B);
        llvm::Function* function = test->create(M);
        ASSERT_TRUE(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        ASSERT_EQ(arg, call->getArgOperand(0));
        // Test the builder is pointing to the correct location
        ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

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

    Function::Ptr test(new unittest_util::TestFunction({
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1),   // vec3i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 1),  // vec2d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9), 1),  // mat3d
            ArgInfo(llvm::Type::getInt32Ty(C)), // int
            ArgInfo(llvm::Type::getInt64Ty(C)), // int64
            ArgInfo(llvm::Type::getFloatTy(C))  // float
        },
        ArgInfo(llvm::Type::getVoidTy(C)),
        "ax.test"));

    llvm::Function* function = test->create(M);
    finalizeFunction(B, function);
    VERIFY_FUNCTION_IR(function);
    // also finalize the current module function, but set the inset point
    // just above it so we can continue to verify IR during this test
    llvm::Value* inst = B.CreateRetVoid();
    // This specifies that created instructions should be inserted before
    // the specified instruction.
    B.SetInsertPoint(llvm::cast<llvm::Instruction>(inst));

    // default args

    Value f32c0 = Value::Create<float>(C, 0);
    Value d64c0 = Value::Create<double>(C, 0);
    Value i32c1 = Value::Create<int32_t>(C, 0);
    Value i64c1 = Value::Create<int64_t>(C, 0);
    Value vec3i = Value::ScalarsToArray(B, {i32c1,i32c1,i32c1}); // vec3i
    Value vec2d = Value::ScalarsToArray(B, {d64c0,d64c0}); // vec2d
    Value mat3d = Value::ScalarsToArray(B, {
        d64c0,d64c0,d64c0,
        d64c0,d64c0,d64c0,
        d64c0,d64c0,d64c0
    }); // mat3d

    {
        // Different ways to provide the args
        const NativeArguments nativeargs({vec3i, vec2d, mat3d, i32c1, i64c1, f32c0});
        const Arguments args(nativeargs);
        const std::vector<llvm::Value*> llvmargs = args.AsLLVMValues();

        // test no casting needed for valid IR

#if LLVM_VERSION_MAJOR <= 15
        llvm::Value* llvmresult = test->call(llvmargs, B, /*cast*/false);
#else
        llvm::Value* llvmresult = test->call(llvmargs, B);
#endif
        ASSERT_TRUE(llvmresult);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(llvmresult);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(llvmargs[0], call->getArgOperand(0));
        ASSERT_EQ(llvmargs[1], call->getArgOperand(1));
        ASSERT_EQ(llvmargs[2], call->getArgOperand(2));
        ASSERT_EQ(llvmargs[3], call->getArgOperand(3));
        ASSERT_EQ(llvmargs[4], call->getArgOperand(4));
        ASSERT_EQ(llvmargs[5], call->getArgOperand(5));
        VERIFY_MODULE_IR(&M);

        // test no casting needed for valid IR, even with cast=true

#if LLVM_VERSION_MAJOR <= 15
        llvmresult = test->call(llvmargs, B, /*cast*/true);
        ASSERT_TRUE(llvmresult);
        call = llvm::dyn_cast<llvm::CallInst>(llvmresult);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(llvmargs[0], call->getArgOperand(0));
        ASSERT_EQ(llvmargs[1], call->getArgOperand(1));
        ASSERT_EQ(llvmargs[2], call->getArgOperand(2));
        ASSERT_EQ(llvmargs[3], call->getArgOperand(3));
        ASSERT_EQ(llvmargs[4], call->getArgOperand(4));
        ASSERT_EQ(llvmargs[5], call->getArgOperand(5));
        VERIFY_MODULE_IR(&M);
#endif

        // Test using Argument containers

        Value result = test->call(nativeargs, B);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result.GetValue());
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(nativeargs[0].GetValue(), call->getArgOperand(0));
        ASSERT_EQ(nativeargs[1].GetValue(), call->getArgOperand(1));
        ASSERT_EQ(nativeargs[2].GetValue(), call->getArgOperand(2));
        ASSERT_EQ(nativeargs[3].GetValue(), call->getArgOperand(3));
        ASSERT_EQ(nativeargs[4].GetValue(), call->getArgOperand(4));
        ASSERT_EQ(nativeargs[5].GetValue(), call->getArgOperand(5));
        VERIFY_MODULE_IR(&M);


        result = test->call(args, B);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result.GetValue());
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(args[0], call->getArgOperand(0));
        ASSERT_EQ(args[1], call->getArgOperand(1));
        ASSERT_EQ(args[2], call->getArgOperand(2));
        ASSERT_EQ(args[3], call->getArgOperand(3));
        ASSERT_EQ(args[4], call->getArgOperand(4));
        ASSERT_EQ(args[5], call->getArgOperand(5));
        VERIFY_MODULE_IR(&M);
    }

#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> expected;
    test->types(expected, C);
#endif

    ArgInfoVector arginfo;
    test->types(arginfo, C);

    //

    // Test different types of valid casting

    Value i1c0  = Value::Create<bool>(C, true);
    Value vec3f = Value::ScalarsToArray(B, {f32c0,f32c0,f32c0}); // vec3f
    Value vec3d = Value::ScalarsToArray(B, {d64c0,d64c0,d64c0}); // vec3d
    Value vec2f = Value::ScalarsToArray(B, {f32c0,f32c0}); // vec2f
    Value vec2i = Value::ScalarsToArray(B, {i32c1,i32c1}); // vecid
    Value mat3f = Value::ScalarsToArray(B, {
        f32c0,f32c0,f32c0,
        f32c0,f32c0,f32c0,
        f32c0,f32c0,f32c0
    }); // mat3f
    //

    {
        const NativeArguments nativeargs({vec3f, vec2f, mat3f, i1c0, i1c0, i1c0});
        llvm::CallInst* call;

#if LLVM_VERSION_MAJOR <= 15
        const Arguments args(nativeargs);
        const std::vector<llvm::Value*> llvmargs = args.AsLLVMValues();
        llvm::Value* llvmresult = test->call(llvmargs, B, /*cast*/true);
        ASSERT_TRUE(llvmresult);
        call = llvm::dyn_cast<llvm::CallInst>(llvmresult);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_TRUE(llvmargs[0] != call->getArgOperand(0));
        ASSERT_TRUE(llvmargs[1] != call->getArgOperand(1));
        ASSERT_TRUE(llvmargs[2] != call->getArgOperand(2));
        ASSERT_TRUE(llvmargs[3] != call->getArgOperand(3));
        ASSERT_TRUE(llvmargs[4] != call->getArgOperand(4));
        ASSERT_TRUE(llvmargs[5] != call->getArgOperand(5));
        ASSERT_TRUE(expected[0] == call->getArgOperand(0)->getType());
        ASSERT_TRUE(expected[1] == call->getArgOperand(1)->getType());
        ASSERT_TRUE(expected[2] == call->getArgOperand(2)->getType());
        ASSERT_TRUE(expected[3] == call->getArgOperand(3)->getType());
        ASSERT_TRUE(expected[4] == call->getArgOperand(4)->getType());
        ASSERT_TRUE(expected[5] == call->getArgOperand(5)->getType());
        ASSERT_TRUE(arginfo[0].GetType() == call->getArgOperand(0)->getType());
        ASSERT_TRUE(arginfo[1].GetType() == call->getArgOperand(1)->getType());
        ASSERT_TRUE(arginfo[2].GetType() == call->getArgOperand(2)->getType());
        ASSERT_TRUE(arginfo[3].GetType() == call->getArgOperand(3)->getType());
        ASSERT_TRUE(arginfo[4].GetType() == call->getArgOperand(4)->getType());
        ASSERT_TRUE(arginfo[5].GetType() == call->getArgOperand(5)->getType());
        VERIFY_MODULE_IR(&M);
#endif

        Value result = test->call(nativeargs, B);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result.GetValue());
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_TRUE(nativeargs[0].GetValue() != call->getArgOperand(0));
        ASSERT_TRUE(nativeargs[1].GetValue() != call->getArgOperand(1));
        ASSERT_TRUE(nativeargs[2].GetValue() != call->getArgOperand(2));
        ASSERT_TRUE(nativeargs[3].GetValue() != call->getArgOperand(3));
        ASSERT_TRUE(nativeargs[4].GetValue() != call->getArgOperand(4));
        ASSERT_TRUE(nativeargs[5].GetValue() != call->getArgOperand(5));
        ASSERT_TRUE(arginfo[0].GetType() == call->getArgOperand(0)->getType());
        ASSERT_TRUE(arginfo[1].GetType() == call->getArgOperand(1)->getType());
        ASSERT_TRUE(arginfo[2].GetType() == call->getArgOperand(2)->getType());
        ASSERT_TRUE(arginfo[3].GetType() == call->getArgOperand(3)->getType());
        ASSERT_TRUE(arginfo[4].GetType() == call->getArgOperand(4)->getType());
        ASSERT_TRUE(arginfo[5].GetType() == call->getArgOperand(5)->getType());
        VERIFY_MODULE_IR(&M);
    }

    //

    {
        const NativeArguments nativeargs({
            vec3d,
            vec2i,
            mat3d, // mat3d - no cast required
            f32c0,
            f32c0,
            f32c0  // float - no cast required
        });
        llvm::CallInst* call;

#if LLVM_VERSION_MAJOR <= 15
        const Arguments args(nativeargs);
        const std::vector<llvm::Value*> llvmargs = args.AsLLVMValues();
        llvm::Value* llvmresult = test->call(llvmargs, B, /*cast*/true);
        ASSERT_TRUE(llvmresult);
        call = llvm::dyn_cast<llvm::CallInst>(llvmresult);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_TRUE(llvmargs[0] != call->getArgOperand(0));
        ASSERT_TRUE(llvmargs[1] != call->getArgOperand(1));
        ASSERT_EQ(llvmargs[2], call->getArgOperand(2)); // no cast
        ASSERT_TRUE(llvmargs[3] != call->getArgOperand(3));
        ASSERT_TRUE(llvmargs[4] != call->getArgOperand(4));
        ASSERT_EQ(llvmargs[5], call->getArgOperand(5)); // no cast
        ASSERT_TRUE(expected[0] == call->getArgOperand(0)->getType());
        ASSERT_TRUE(expected[1] == call->getArgOperand(1)->getType());
        ASSERT_TRUE(expected[3] == call->getArgOperand(3)->getType());
        ASSERT_TRUE(expected[4] == call->getArgOperand(4)->getType());
        ASSERT_TRUE(arginfo[0].GetType() == call->getArgOperand(0)->getType());
        ASSERT_TRUE(arginfo[1].GetType() == call->getArgOperand(1)->getType());
        ASSERT_TRUE(arginfo[3].GetType() == call->getArgOperand(3)->getType());
        ASSERT_TRUE(arginfo[4].GetType() == call->getArgOperand(4)->getType());
        VERIFY_MODULE_IR(&M);
#endif

        Value result = test->call(nativeargs, B);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result.GetValue());
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_TRUE(nativeargs[0].GetValue() != call->getArgOperand(0));
        ASSERT_TRUE(nativeargs[1].GetValue() != call->getArgOperand(1));
        ASSERT_EQ(nativeargs[2].GetValue(), call->getArgOperand(2));
        ASSERT_TRUE(nativeargs[3].GetValue() != call->getArgOperand(3));
        ASSERT_TRUE(nativeargs[4].GetValue() != call->getArgOperand(4));
        ASSERT_EQ(nativeargs[5].GetValue(), call->getArgOperand(5));
        ASSERT_TRUE(arginfo[0].GetType() == call->getArgOperand(0)->getType());
        ASSERT_TRUE(arginfo[1].GetType() == call->getArgOperand(1)->getType());
        ASSERT_TRUE(arginfo[3].GetType() == call->getArgOperand(3)->getType());
        ASSERT_TRUE(arginfo[4].GetType() == call->getArgOperand(4)->getType());
        VERIFY_MODULE_IR(&M);
    }

    {
        const NativeArguments nativeargs({
            vec3i, // no cast required
            vec2d, // no cast required
            mat3d, // no cast required
            i64c1, //
            i64c1, // no cast required
            i64c1  //
        });
        llvm::CallInst* call;

#if LLVM_VERSION_MAJOR <= 15
        const Arguments args(nativeargs);
        const std::vector<llvm::Value*> llvmargs = args.AsLLVMValues();
        llvm::Value* llvmresult = test->call(llvmargs, B, /*cast*/true);
        ASSERT_TRUE(llvmresult);
        call = llvm::dyn_cast<llvm::CallInst>(llvmresult);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(args[0], call->getArgOperand(0));
        ASSERT_EQ(args[1], call->getArgOperand(1));
        ASSERT_EQ(args[2], call->getArgOperand(2));
        ASSERT_TRUE(args[3] != call->getArgOperand(3));
        ASSERT_EQ(args[4], call->getArgOperand(4));
        ASSERT_TRUE(args[5] != call->getArgOperand(5));
        ASSERT_TRUE(expected[3] == call->getArgOperand(3)->getType());
        ASSERT_TRUE(expected[5] == call->getArgOperand(5)->getType());
        VERIFY_MODULE_IR(&M);
#endif

        Value result = test->call(nativeargs, B);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result.GetValue());
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        ASSERT_EQ(nativeargs[0].GetValue(), call->getArgOperand(0));
        ASSERT_EQ(nativeargs[1].GetValue(), call->getArgOperand(1));
        ASSERT_EQ(nativeargs[2].GetValue(), call->getArgOperand(2));
        ASSERT_TRUE(nativeargs[3].GetValue() != call->getArgOperand(3));
        ASSERT_EQ(nativeargs[4].GetValue(), call->getArgOperand(4));
        ASSERT_TRUE(nativeargs[5].GetValue() != call->getArgOperand(5));
        ASSERT_TRUE(arginfo[3].GetType() == call->getArgOperand(3)->getType());
        ASSERT_TRUE(arginfo[5].GetType() == call->getArgOperand(5)->getType());
        VERIFY_MODULE_IR(&M);
    }

    // Test that invalid IR is generated if casting cannot be performed.
    // This is just to test that call doesn't error or behave unexpectedly

    // Test called with castable arg but cast is false. Test arg is left
    // unchanged and IR is invalid due to signature size

    {
#ifdef OPENVDB_AX_NO_LLVM_ASSERTS // only test in release otherwise LLVM asserts

        Arguments args;
        args.AddArg(vec3f);

#if LLVM_VERSION_MAJOR <= 15
        llvm::Value* result = test->call(args.AsLLVMValues(), B, /*cast*/false);
#else
        llvm::Value* result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        auto call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        // should be the same as cast is false
        ASSERT_TRUE(vec3f.GetValue() == call->getArgOperand(0));
        VERIFY_MODULE_IR_INVALID(&M);

        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);

        // Test called with castable arg with cast true. Test IR is invalid
        // due to signature size

#if LLVM_VERSION_MAJOR <= 15
        result = test->call(args.AsLLVMValues(), B, /*cast*/true);
#else
        result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
#if LLVM_VERSION_MAJOR <= 15
        // shouldn't be the same as it should have been cast
        ASSERT_TRUE(vec3f.GetValue() != call->getArgOperand(0));
        ASSERT_TRUE(expected[0] == call->getArgOperand(0)->getType());
#endif
        VERIFY_MODULE_IR_INVALID(&M);

        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);

        // Test called with non castable args, but matching signature size.
        // Test IR is invalid due to cast being off

        const NativeArguments nativeargs({vec3i,vec2d,mat3d,i64c1,i64c1,i64c1});
        args = Arguments(nativeargs);
#if LLVM_VERSION_MAJOR <= 15
        result = test->call(args.AsLLVMValues(), B, /*cast*/false);
#else
        result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(6u, getNumArgFromCallInst(call));
        // no casting, args should match operands
        ASSERT_TRUE(args[0] == call->getArgOperand(0));
        ASSERT_TRUE(args[1] == call->getArgOperand(1));
        ASSERT_TRUE(args[2] == call->getArgOperand(2));
        ASSERT_TRUE(args[3] == call->getArgOperand(3));
        ASSERT_TRUE(args[4] == call->getArgOperand(4));
        ASSERT_TRUE(args[5] == call->getArgOperand(5));
        VERIFY_MODULE_IR_INVALID(&M);

        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);
#endif
    }

    //
    // Test strings
    {
        llvm::Type* axstr = LLVMType<openvdb::ax::codegen::String>::get(C);
        llvm::Type* chars = LLVMType<char>::get(C);  // char

        // build values

        llvm::Value* chararray = B.CreateGlobalStringPtr("tmp"); // char*
        // @note  non-safer initialization of strings
        llvm::Value* strptr = B.CreateAlloca(LLVMType<openvdb::ax::codegen::String>::get(C)); // str*

        // void ax.str.test(openvdb::ax::codegen::String*, char*)
        test.reset(new unittest_util::TestFunction({
                ArgInfo(axstr, 1),
                ArgInfo(chars, 1)
            },
            ArgInfo(llvm::Type::getVoidTy(C)),
            "ax.str.test"));

        Arguments args;
        args.AddArg(strptr, ArgInfo(axstr, 1));
        args.AddArg(chararray, ArgInfo(chars, 1));

        function = test->create(M);
        finalizeFunction(B, function);
        VERIFY_FUNCTION_IR(function);

#if LLVM_VERSION_MAJOR <= 15
        llvm::Value* result = test->call(args.AsLLVMValues(), B, /*cast*/false);
#else
        llvm::Value* result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        ASSERT_TRUE(args[0] == call->getArgOperand(0));
        ASSERT_TRUE(args[1] == call->getArgOperand(1));

        //

#if LLVM_VERSION_MAJOR <= 15
        result = test->call(args.AsLLVMValues(), B, /*cast*/true);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        ASSERT_TRUE(args[0] == call->getArgOperand(0));
        ASSERT_TRUE(args[1] == call->getArgOperand(1));
#endif

        // Test openvdb::ax::codegen::String -> char*
        NativeArguments strargs;
        strargs.AddArg(Value(strptr, axstr));
        strargs.AddArg(Value(strptr, axstr));

#if LLVM_VERSION_MAJOR <= 15
        result = test->call(Arguments(strargs).AsLLVMValues(), B, /*cast*/true);
#else
        result = test->call(strargs, B).GetValue();
#endif
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        ASSERT_TRUE(args[0] == call->getArgOperand(0));
        ASSERT_TRUE(args[1] != call->getArgOperand(1));
        VERIFY_MODULE_IR(&M);

        // Test char* does not cast to openvdb::ax::codegen::String

#ifdef OPENVDB_AX_NO_LLVM_ASSERTS // only test in release otherwise LLVM asserts
        Arguments charargs;
        charargs.AddArg(chararray, ArgInfo(chars, 1));
        charargs.AddArg(chararray, ArgInfo(chars, 1));

#if LLVM_VERSION_MAJOR <= 15
        result = test->call(charargs.AsLLVMValues(), B, /*cast*/true);
#else
        result = test->call(charargs, B).GetValue();
#endif
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        // no valid casting
        ASSERT_TRUE(charargs[0] == call->getArgOperand(0));
        ASSERT_TRUE(charargs[1] == call->getArgOperand(1));
        // The IR is actually valid from LLVM 16 as ptrs are opaque :|
#if LLVM_VERSION_MAJOR <= 15
        VERIFY_MODULE_IR_INVALID(&M);
#endif
        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);
#endif
    }

    // Test ** pointers

    llvm::Value* fptr = B.CreateAlloca(llvm::PointerType::get(llvm::Type::getFloatTy(C), 0));
    llvm::Value* dptr = B.CreateAlloca(llvm::PointerType::get(llvm::Type::getDoubleTy(C), 0));

    {
        test.reset(new unittest_util::TestFunction({
                ArgInfo(llvm::Type::getFloatTy(C), 2),
                ArgInfo(llvm::Type::getDoubleTy(C), 2)
            },
            ArgInfo(llvm::Type::getVoidTy(C)),
            "ax.ptrs.test"));

        function = test->create(M);
        finalizeFunction(B, function);
        VERIFY_FUNCTION_IR(function);

        Arguments args;
        args.AddArg(fptr, ArgInfo(llvm::Type::getFloatTy(C), 1));
        args.AddArg(dptr, ArgInfo(llvm::Type::getDoubleTy(C), 1));

#if LLVM_VERSION_MAJOR <= 15
        llvm::Value* result = test->call(Arguments(args).AsLLVMValues(), B, /*cast*/false);
#else
        llvm::Value* result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        ASSERT_TRUE(fptr == call->getArgOperand(0));
        ASSERT_TRUE(dptr == call->getArgOperand(1));
        VERIFY_MODULE_IR(&M);

        //

#if LLVM_VERSION_MAJOR <= 15 // non native ** args, no casting
        result = test->call({fptr, dptr}, B, /*cast*/true);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        ASSERT_TRUE(fptr == call->getArgOperand(0));
        ASSERT_TRUE(dptr == call->getArgOperand(1));
        VERIFY_MODULE_IR(&M);
#ifdef OPENVDB_AX_NO_LLVM_ASSERTS // only test in release otherwise LLVM asserts
        // switch the points, check no valid casting
        result = test->call({dptr, fptr}, B, /*cast*/true);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(2u, getNumArgFromCallInst(call));
        // args unaltered as casting is invalid
        ASSERT_TRUE(dptr == call->getArgOperand(0));
        ASSERT_TRUE(fptr == call->getArgOperand(1));

        VERIFY_MODULE_IR_INVALID(&M);
        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);
#endif
#endif
    }

    {
        // Test void pointers

        test.reset(new unittest_util::TestFunction({
                ArgInfo(LLVMType<int8_t>::get(C), 1),
            },
            ArgInfo(llvm::Type::getVoidTy(C)),
            "ax.void.test"));

        function = test->create(M);
        finalizeFunction(B, function);
        VERIFY_FUNCTION_IR(function);

        llvm::Value* vptrptr = B.CreateAlloca(LLVMType<void*>::get(C));
        llvm::Value* vptr = B.CreateLoad(LLVMType<void*>::get(C), vptrptr);

        Arguments args;
        args.AddArg(vptr, ArgInfo(LLVMType<int8_t>::get(C), 1));

#if LLVM_VERSION_MAJOR <= 15
        llvm::Value* result = test->call(Arguments(args).AsLLVMValues(), B, /*cast*/false);
#else
        llvm::Value* result = test->call(args, B).GetValue();
#endif
        ASSERT_TRUE(result);
        llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        ASSERT_TRUE(vptr == call->getArgOperand(0));
        VERIFY_MODULE_IR(&M);

        //

#if LLVM_VERSION_MAJOR <= 15 // non native ** args, no casting
        result = test->call({vptr}, B, /*cast*/true);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        ASSERT_TRUE(vptr == call->getArgOperand(0));
        VERIFY_MODULE_IR(&M);

#ifdef OPENVDB_AX_NO_LLVM_ASSERTS // only test in release otherwise LLVM asserts
        // verify no cast from other pointers to void*
        result = test->call({fptr}, B, /*cast*/true);
        ASSERT_TRUE(result);
        call = llvm::dyn_cast<llvm::CallInst>(result);
        ASSERT_TRUE(call);
        ASSERT_EQ(function, call->getCalledFunction());
        ASSERT_EQ(1u, getNumArgFromCallInst(call));
        ASSERT_TRUE(fptr == call->getArgOperand(0));
        VERIFY_MODULE_IR_INVALID(&M);
        // Remove the bad instruction (and re-verify to double check)
        call->eraseFromParent();
        VERIFY_MODULE_IR(&M);
#endif
#endif
    }
}

// Test Function::match
TEST_F(TestFunctionTypes, testFunctionMatch)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    Function::SignatureMatch match;

    const ArgInfoVector scalars {
        ArgInfo(llvm::Type::getInt1Ty(C)),   // bool
        ArgInfo(llvm::Type::getInt16Ty(C)),  // int16
        ArgInfo(llvm::Type::getInt32Ty(C)),  // int
        ArgInfo(llvm::Type::getInt64Ty(C)),  // int64
        ArgInfo(llvm::Type::getFloatTy(C)),  // float
        ArgInfo(llvm::Type::getDoubleTy(C))  // double
    };
    const ArgInfoVector array2 {
        ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 1),    // vec2i
        ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2), 1),    // vec2f
        ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 1)    // vec2d
    };
    const ArgInfoVector array3 {
        ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1),    // vec3i
        ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3), 1),    // vec3f
        ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3), 1)    // vec3d
    };
    const ArgInfoVector array4 {
        ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4), 1),    // vec3i
        ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4), 1),    // vec3f
        ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4), 1)    // vec3d
    };
    const ArgInfoVector array9 {
        ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9), 1),    // ix9 (not supported by ax)
        ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9), 1),    // mat3f
        ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9), 1)    // mat3d
    };
    const ArgInfoVector array16 {
        ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16), 1),    // ix16 (not supported by ax)
        ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16), 1),    // mat3f
        ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16), 1)    // mat3d
    };
    const std::vector<ArgInfoVector> arrays {
        array2,
        array3,
        array4,
        array9,
        array16,
    };

    // test empty explicit match

    Function::Ptr test(new unittest_util::TestFunction({},
        ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
#if LLVM_VERSION_MAJOR <= 15
    match = test->match(std::vector<llvm::Type*>{}, C);
    ASSERT_EQ(match, Function::Explicit);
#endif
    match = test->match(ArgInfoVector{}, C);
    ASSERT_EQ(match, Function::Explicit);

    //

    ArgInfoVector types;
    for (auto& info : scalars) types.emplace_back(info);
    for (auto& info : array2)  types.emplace_back(info);
    for (auto& info : array3)  types.emplace_back(info);
    for (auto& info : array4)  types.emplace_back(info);
    for (auto& info : array9)  types.emplace_back(info);
    for (auto& info : array16) types.emplace_back(info);
    types.emplace_back(ArgInfo(LLVMType<openvdb::ax::codegen::String>::get(C), 1));

#if LLVM_VERSION_MAJOR <= 15
    // check types are unique
    {
        auto llvmtypes = types.AsLLVMTypes();
        ASSERT_EQ(std::set<llvm::Type*>(llvmtypes.begin(), llvmtypes.end()).size(), llvmtypes.size());
    }
#endif

    //

    test.reset(new unittest_util::TestFunction({
            ArgInfo(llvm::Type::getInt1Ty(C)),   // bool
            ArgInfo(llvm::Type::getInt16Ty(C)),  // int16
            ArgInfo(llvm::Type::getInt32Ty(C)),  // int32
            ArgInfo(llvm::Type::getInt64Ty(C)),  // int64
            ArgInfo(llvm::Type::getFloatTy(C)),  // float
            ArgInfo(llvm::Type::getDoubleTy(C)), // double
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 1),    // vec2i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 2), 1),    // vec2f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 1),   // vec2d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1),    // vec3i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 3), 1),    // vec3f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 3), 1),   // vec3d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 4), 1),    // vec4i
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 4), 1),    // vec4f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 4), 1),   // vec4d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 9), 1),    // ix9 (not supported by ax)
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 9), 1),    // mat3f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 9), 1),   // mat3d
            ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 16), 1),   // ix16 (not supported by ax)
            ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 16), 1),   // mat4f
            ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 16), 1),  // mat4d
            ArgInfo(LLVMType<openvdb::ax::codegen::String>::get(C), 1) // string
        },
        ArgInfo(llvm::Type::getVoidTy(C)),
        "ax.test"));

    match = test->match(types, C);
    ASSERT_EQ(match, Function::Explicit);

    // test size match

    llvm::Type* i32t = llvm::Type::getInt32Ty(C);
    test.reset(new unittest_util::TestFunction({ArgInfo(i32t)},
        ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
#if LLVM_VERSION_MAJOR <= 15
    match = test->match({llvm::ArrayType::get(i32t, 1)->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Size);
    match = test->match({i32t->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Size);
#endif

    match = test->match({ArgInfo(llvm::ArrayType::get(i32t, 1))}, C);
    ASSERT_EQ(match, Function::Size);
    match = test->match({ArgInfo(llvm::ArrayType::get(i32t, 1), 1)}, C);
    ASSERT_EQ(match, Function::Size);
    match = test->match({ArgInfo(i32t, 1)}, C);
    ASSERT_EQ(match, Function::Size);

    // test no match

#if LLVM_VERSION_MAJOR <= 15
    match = test->match(std::vector<llvm::Type*>{}, C);
    ASSERT_EQ(Function::None, match);
    match = test->match({i32t, i32t}, C);
    ASSERT_EQ(Function::None, match);
#endif
    match = test->match(ArgInfoVector{}, C);
    ASSERT_EQ(Function::None, match);
    match = test->match(ArgInfoVector{ArgInfo(i32t), ArgInfo(i32t)}, C);
    ASSERT_EQ(Function::None, match);

    // test scalar matches

   for (size_t i = 0; i < scalars.size(); ++i) {
        test.reset(new unittest_util::TestFunction({scalars[i]}, ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
        ArgInfoVector copy(scalars);
        copy[i] = copy.back();
        copy.pop_back();
        ASSERT_EQ(size_t(5), copy.size());
        ASSERT_EQ(Function::Explicit, test->match({scalars[i]}, C));
        ASSERT_EQ(Function::Implicit, test->match({copy[0]}, C));
        ASSERT_EQ(Function::Implicit, test->match({copy[1]}, C));
        ASSERT_EQ(Function::Implicit, test->match({copy[2]}, C));
        ASSERT_EQ(Function::Implicit, test->match({copy[3]}, C));
        ASSERT_EQ(Function::Implicit, test->match({copy[4]}, C));
   }

   //
   // Test array matches - no implicit cast as operands are not marked as readonly

   for (const auto& types : arrays) {
        // test these array types
        for (size_t i = 0; i < types.size(); ++i) {
            test.reset(new unittest_util::TestFunction({types[i]}, ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
            ArgInfoVector copy(types);
            copy[i] = copy.back();
            copy.pop_back();
            ASSERT_EQ(size_t(2), copy.size());
            ASSERT_EQ(Function::Explicit, test->match({types[i]}, C));
            ASSERT_EQ(Function::Size, test->match({copy[0]}, C));
            ASSERT_EQ(Function::Size, test->match({copy[1]}, C));

            // test non matching size arrays
            for (const ArgInfoVector& inputs : arrays) {
                if (&types == &inputs) continue;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    ASSERT_EQ(Function::Size, test->match({inputs[i]}, C));
                }
            }
        }
   }

   //
   // Test array matches with readonly marking

   for (const ArgInfoVector& types : arrays) {
        // test these array types
        for (size_t i = 0; i < types.size(); ++i) {
            test.reset(new unittest_util::TestFunction({types[i]}, ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
            test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
            ArgInfoVector copy(types);
            copy[i] = copy.back();
            copy.pop_back();
            ASSERT_EQ(size_t(2), copy.size());
            ASSERT_EQ(Function::Explicit, test->match({types[i]}, C));
            ASSERT_EQ(Function::Implicit, test->match({copy[0]}, C));
            ASSERT_EQ(Function::Implicit, test->match({copy[1]}, C));

            // test non matching size arrays
            for (const ArgInfoVector& inputs : arrays) {
                if (&types == &inputs) continue;
                for (size_t i = 0; i < inputs.size(); ++i) {
                    ASSERT_EQ(Function::Size, test->match({inputs[i]}, C));
                }
            }
        }
    }

    // test strings
    {
        test.reset(new unittest_util::TestFunction({ArgInfo(LLVMType<char>::get(C), 1)},
            ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
        ASSERT_EQ(Function::Size,
            test->match({ArgInfo(LLVMType<openvdb::ax::codegen::String>::get(C), 1)}, C));
        ASSERT_EQ(Function::Explicit,
            test->match({ArgInfo(LLVMType<char>::get(C), 1)}, C));

        test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
        ASSERT_EQ(Function::Implicit,
            test->match({ArgInfo(LLVMType<openvdb::ax::codegen::String>::get(C), 1)}, C));

        test.reset(new unittest_util::TestFunction({ArgInfo(LLVMType<openvdb::ax::codegen::String>::get(C), 1)},
            ArgInfo(llvm::Type::getVoidTy(C)), "ax.test"));
        ASSERT_EQ(Function::Size,
            test->match({ArgInfo(LLVMType<char>::get(C), 1)}, C));
    }

    // test pointers
    {
        ArgInfo fss(llvm::Type::getFloatTy(C), 2);
        ArgInfo dss(llvm::Type::getDoubleTy(C), 2);

        test.reset(new unittest_util::TestFunction({fss, dss},
            ArgInfo(llvm::Type::getVoidTy(C)),
            "ax.ptrs.test"));

        ASSERT_EQ(Function::Explicit,
            test->match({fss, dss}, C));
        ASSERT_EQ(Function::Size,
            test->match({fss, fss}, C));

        // Even if pointers are marked as readonly, casting is not supported
        test->setParamAttributes(0, {llvm::Attribute::ReadOnly});
        test->setParamAttributes(1, {llvm::Attribute::ReadOnly});

        ASSERT_EQ(Function::Size,
            test->match({fss, fss}, C));
    }
}

// Test derived CFunctions, mainly CFunction::create and CFunction::types
TEST_F(TestFunctionTypes, testCFunctions)
{
    using openvdb::ax::codegen::CFunction;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> types;
#else
    ArgInfoVector types;
#endif

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

    ASSERT_EQ(size_t(0), voidfunc.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::voidfunc),
        voidfunc.address());
    ASSERT_EQ(std::string("voidfunc"),
        std::string(voidfunc.symbol()));

    // test scalar arguments

    ASSERT_EQ(size_t(6), scalars.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::scalarfunc),
        scalars.address());
    ASSERT_EQ(std::string("scalarfunc"),
        std::string(scalars.symbol()));

    // test scalar ptr arguments

    ASSERT_EQ(size_t(6), scalarptrs.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::scalatptsfunc),
        scalarptrs.address());
    ASSERT_EQ(std::string("scalatptsfunc"),
        std::string(scalarptrs.symbol()));

    // test array ptr arguments

    ASSERT_EQ(size_t(6), arrayptrs.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::arrayfunc),
        arrayptrs.address());
    ASSERT_EQ(std::string("arrayfunc"),
        std::string(arrayptrs.symbol()));

    // test selected template functions

    ASSERT_EQ(size_t(0), select.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::tmplfunc<float>),
        select.address());
    ASSERT_EQ(std::string("tmplfunc"),
        std::string(select.symbol()));

    // test multiple indirection layers

    ASSERT_EQ(size_t(6), mindirect.size());
    ASSERT_EQ(reinterpret_cast<uint64_t>(&CBindings::multiptrfunc),
        mindirect.address());
    ASSERT_EQ(std::string("multiptrfunc"),
        std::string(mindirect.symbol()));

    //
    // Test types

    // test scalar arguments

    auto returnType = scalars.types(types, C);
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(returnType->isIntegerTy(16));
    ASSERT_TRUE(types[0]->isIntegerTy(1));
    ASSERT_TRUE(types[1]->isIntegerTy(16));
    ASSERT_TRUE(types[2]->isIntegerTy(32));
    ASSERT_TRUE(types[3]->isIntegerTy(64));
    ASSERT_TRUE(types[4]->isFloatTy());
    ASSERT_TRUE(types[5]->isDoubleTy());
#else
    ASSERT_TRUE(returnType.GetType()->isIntegerTy(16));
    ASSERT_TRUE(types[0].GetType()->isIntegerTy(1));
    ASSERT_TRUE(types[1].GetType()->isIntegerTy(16));
    ASSERT_TRUE(types[2].GetType()->isIntegerTy(32));
    ASSERT_TRUE(types[3].GetType()->isIntegerTy(64));
    ASSERT_TRUE(types[4].GetType()->isFloatTy());
    ASSERT_TRUE(types[5].GetType()->isDoubleTy());
#endif
    types.clear();

    // test scalar ptr arguments

    returnType = scalarptrs.types(types, C);
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(returnType->isIntegerTy(32));
    ASSERT_TRUE(types[0] == llvm::Type::getInt1Ty(C)->getPointerTo());
    ASSERT_TRUE(types[1] == llvm::Type::getInt16Ty(C)->getPointerTo());
    ASSERT_TRUE(types[2] == llvm::Type::getInt32Ty(C)->getPointerTo());
    ASSERT_TRUE(types[3] == llvm::Type::getInt64Ty(C)->getPointerTo());
    ASSERT_TRUE(types[4] == llvm::Type::getFloatTy(C)->getPointerTo());
    ASSERT_TRUE(types[5] == llvm::Type::getDoubleTy(C)->getPointerTo());
#else
    ASSERT_TRUE(returnType.GetType()->isIntegerTy(32));
    ASSERT_TRUE(types[0] == ArgInfo(llvm::Type::getInt1Ty(C), 1));
    ASSERT_TRUE(types[1] == ArgInfo(llvm::Type::getInt16Ty(C), 1));
    ASSERT_TRUE(types[2] == ArgInfo(llvm::Type::getInt32Ty(C), 1));
    ASSERT_TRUE(types[3] == ArgInfo(llvm::Type::getInt64Ty(C), 1));
    ASSERT_TRUE(types[4] == ArgInfo(llvm::Type::getFloatTy(C), 1));
    ASSERT_TRUE(types[5] == ArgInfo(llvm::Type::getDoubleTy(C), 1));
#endif
    types.clear();

    // test array ptr arguments

    returnType = arrayptrs.types(types, C);
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(returnType->isIntegerTy(64));
    ASSERT_TRUE(types[0] == llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1)->getPointerTo());
    ASSERT_TRUE(types[1] == llvm::ArrayType::get(llvm::Type::getInt16Ty(C), 2)->getPointerTo());
    ASSERT_TRUE(types[2] == llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3)->getPointerTo());
    ASSERT_TRUE(types[3] == llvm::ArrayType::get(llvm::Type::getInt64Ty(C), 4)->getPointerTo());
    ASSERT_TRUE(types[4] == llvm::ArrayType::get(llvm::Type::getFloatTy(C), 5)->getPointerTo());
    ASSERT_TRUE(types[5] == llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 6)->getPointerTo());
#else
    ASSERT_TRUE(returnType.GetType()->isIntegerTy(64));
    ASSERT_TRUE(types[0] == ArgInfo(llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1), 1));
    ASSERT_TRUE(types[1] == ArgInfo(llvm::ArrayType::get(llvm::Type::getInt16Ty(C), 2), 1));
    ASSERT_TRUE(types[2] == ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1));
    ASSERT_TRUE(types[3] == ArgInfo(llvm::ArrayType::get(llvm::Type::getInt64Ty(C), 4), 1));
    ASSERT_TRUE(types[4] == ArgInfo(llvm::ArrayType::get(llvm::Type::getFloatTy(C), 5), 1));
    ASSERT_TRUE(types[5] == ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 6), 1));
#endif
    types.clear();

    // test void ptr arguments
    // void* are inferred as int8_t* types

    returnType = mindirect.types(types, C);
    ASSERT_EQ(size_t(6), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(returnType->isVoidTy());
    ASSERT_TRUE(types[0] == llvm::Type::getInt8PtrTy(C));
    ASSERT_TRUE(types[1] == llvm::Type::getInt8PtrTy(C)->getPointerTo());
    ASSERT_TRUE(types[2] == llvm::Type::getInt8PtrTy(C)->getPointerTo()->getPointerTo());
    ASSERT_TRUE(types[3] == llvm::Type::getFloatTy(C)->getPointerTo());
    ASSERT_TRUE(types[4] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo());
    ASSERT_TRUE(types[5] == llvm::Type::getFloatTy(C)->getPointerTo()->getPointerTo()->getPointerTo());
#else
    ASSERT_TRUE(returnType.IsVoid());
    ASSERT_TRUE(types[0] == ArgInfo(llvm::Type::getInt8Ty(C), 1));
    ASSERT_TRUE(types[1] == ArgInfo(llvm::Type::getInt8Ty(C), 2));
    ASSERT_TRUE(types[2] == ArgInfo(llvm::Type::getInt8Ty(C), 3));
    ASSERT_TRUE(types[3] == ArgInfo(llvm::Type::getFloatTy(C), 1));
    ASSERT_TRUE(types[4] == ArgInfo(llvm::Type::getFloatTy(C), 2));
    ASSERT_TRUE(types[5] == ArgInfo(llvm::Type::getFloatTy(C), 3));
#endif
}

// Test C constant folding
TEST_F(TestFunctionTypes, testCFunctionCF)
{
    using openvdb::ax::codegen::CFunction;
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::ArgInfoVector;

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
    ASSERT_TRUE(!test1.hasConstantFold());
    ASSERT_TRUE(test1.fold({B.getInt32(1)}, C) == nullptr);

    test1.setConstantFold(true);
    llvm::Value* result = test1.fold({B.getInt32(1)}, C);
    ASSERT_TRUE(result);
    llvm::ConstantInt* constant = llvm::dyn_cast<llvm::ConstantInt>(result);
    ASSERT_TRUE(constant);
    ASSERT_EQ(uint64_t(10), constant->getLimitedValue());

    // test with scalar arg

    CFunction<float(float)> test2("ax.test2", cftest2);
    test2.setConstantFold(true);
    result = test2.fold({LLVMType<float>::get(C, -3.2f)}, C);
    ASSERT_TRUE(result);
    llvm::ConstantFP* constantfp = llvm::dyn_cast<llvm::ConstantFP>(result);
    ASSERT_TRUE(constantfp);
    const llvm::APFloat& apf = constantfp->getValueAPF();
    ASSERT_EQ(-3.2f, apf.convertToFloat());

    // test unsupported

    CFunction<float(float(*)[3])> test3("ax.test3", cftest3);
    test3.setConstantFold(true);
    // constant arg (verify it would 100% match)
    // note that this arg is fundamentally the wrong type for this function
    // and the way in which we support vector types anyway (by ptr) - but because
    // its impossible to have a constant ptr, use this for now as this will most
    // likely be the way we support folding for arrays in the future
    llvm::Value* arg = LLVMType<float[3]>::get(C, {1,2,3});
    ASSERT_TRUE(llvm::isa<llvm::Constant>(arg));
    // check fold fails
    ASSERT_TRUE(test3.fold({arg}, C) == nullptr);

    //

    CFunction<void(float*)> test4("ax.test4", cftest4);
    test4.setConstantFold(true);

    // constant arg (verify it would 100% match)
    llvm::Value* nullfloat = llvm::ConstantPointerNull::get(LLVMType<float*>::get(C));

#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> types;
#else
    ArgInfoVector types;
#endif
    test4.types(types, C);

    ASSERT_EQ(size_t(1), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(nullfloat->getType() == types.front());
#else
    ASSERT_TRUE(nullfloat->getType() == types.front().GetType());
#endif
    ASSERT_TRUE(test4.fold({nullfloat}, C) == nullptr);
}

// Test derived IR Function, IRFunctionBase::create and IRFunctionBase::call
TEST_F(TestFunctionTypes, testIRFunctions)
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::IRFunctionBase;
    using openvdb::ax::codegen::Arguments;
    using openvdb::ax::codegen::ArgInfoVector;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::Value;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    // Small test to check the templated version of IRFunction::types.
    // All other checks work with the IRFunctionBase class
    {
        using openvdb::ax::codegen::IRFunction;

        static auto generate =
            [](const Arguments&,
               llvm::IRBuilder<>&) -> Value
        { return Value::Invalid(); };

        IRFunction<double(bool,
                int16_t*,
                int32_t(*)[1],
                int64_t,
                float*,
                double(*)[2])>
            mix("mix", generate);

        ASSERT_EQ(std::string("mix"),
            std::string(mix.symbol()));

        ArgInfoVector types;
        ArgInfo returnType = mix.types(types, C);
        ASSERT_TRUE(returnType.GetType()->isDoubleTy());
        ASSERT_EQ(size_t(6), types.size());
        ASSERT_TRUE(types[0] == ArgInfo(llvm::Type::getInt1Ty(C)));
        ASSERT_TRUE(types[1] == ArgInfo(llvm::Type::getInt16Ty(C), 1));
        ASSERT_TRUE(types[2] == ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1), 1));
        ASSERT_TRUE(types[3] == ArgInfo(llvm::Type::getInt64Ty(C)));
        ASSERT_TRUE(types[4] == ArgInfo(llvm::Type::getFloatTy(C), 1));
        ASSERT_TRUE(types[5] == ArgInfo(llvm::ArrayType::get(llvm::Type::getDoubleTy(C), 2), 1));
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
        (const Arguments& args,
         llvm::IRBuilder<>& FunctionBB) -> Value
    {
        // NOTE: GTest's ASSERT_* macros create a 'return;' statement
        // which errors here. Exceptions are being used instead to
        // exit early.

        // test the builder is pointing to the correct location
        if (&B == &FunctionBB)
        {
            throw std::runtime_error("Builder should be different");
        }
        llvm::BasicBlock* Block = FunctionBB.GetInsertBlock();
        if (!Block || !Block->empty())
        {
            throw std::runtime_error("Basic block should exist and be empty");
        }
        llvm::Function* F = Block->getParent();
        if (!F)
        {
            throw std::runtime_error("Function should exist");
        }
        EXPECT_EQ(expectedName, std::string(F->getName()));
        llvm::Module* _M = F->getParent();
        EXPECT_EQ(&M, _M);

        if (args.size() != 2)
        {
            throw std::runtime_error("Function should take two arguments");
        }
        EXPECT_TRUE(args[0] == llvm::cast<llvm::Value>(F->arg_begin()));
        EXPECT_TRUE(args[1] == llvm::cast<llvm::Value>(F->arg_begin()+1));
        EXPECT_TRUE(args[0]->getType()->isFloatTy());
        EXPECT_TRUE(args[1]->getType()->isFloatTy());

        llvm::Value* result = FunctionBB.CreateFAdd(args[0], args[1]);
        Value val(result, result->getType());

        if (termMode == 0) return Value::Return(FunctionBB, &val);
        if (termMode == 1) return val;
        if (termMode == 2) return Value::Invalid();
        throw std::runtime_error("No valid test mode");
        return Value::Invalid();
    };

    llvm::Function* function = nullptr;

    expectedName = "ax.ir.test";
    Function::Ptr test(new TestIRFunction({
            ArgInfo(llvm::Type::getFloatTy(C)),
            ArgInfo(llvm::Type::getFloatTy(C))
        },
        ArgInfo(llvm::Type::getFloatTy(C)),
        expectedName, generate));

    // Test function prototype creation
    ASSERT_TRUE(!M.getFunction("ax.ir.test"));
    // detached
    function = test->create(C);
    llvm::Function* function2 = test->create(C);
    ASSERT_TRUE(!M.getFunction("ax.ir.test"));
    ASSERT_TRUE(function);
    ASSERT_TRUE(function->empty());
    ASSERT_TRUE(function != function2);
    ASSERT_TRUE(!function->isVarArg());
    ASSERT_EQ(size_t(2), function->arg_size());

    llvm::FunctionType* ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isFloatTy());
    ASSERT_EQ(2u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0)->isFloatTy());
    ASSERT_TRUE(ftype->getParamType(1)->isFloatTy());
    ASSERT_TRUE(function->getAttributes().isEmpty());
    delete function;
    delete function2;

    // Test function creation with module and IR generation
    ASSERT_TRUE(!M.getFunction("ax.ir.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_EQ(function, M.getFunction("ax.ir.test"));
    ASSERT_TRUE(!function->empty());
    llvm::BasicBlock* BB = &(function->getEntryBlock());
    // two instructions - the add and return
    ASSERT_EQ(size_t(2), BB->size());
    auto iter = BB->begin();
    llvm::BinaryOperator* binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    ASSERT_TRUE(binary);
    ASSERT_EQ(llvm::Instruction::FAdd, binary->getOpcode());
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    llvm::ReturnInst* ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    ASSERT_TRUE(ret);
    llvm::Value* rvalue = ret->getReturnValue();
    ASSERT_TRUE(rvalue);
    ASSERT_TRUE(rvalue->getType()->isFloatTy());
    // the return is the result of the bin op
    ASSERT_EQ(rvalue, llvm::cast<llvm::Value>(binary));

    ++iter;
    ASSERT_TRUE(BB->end() == iter);

    // additional call should match
    ASSERT_EQ(function, test->create(M));
    // verify IR
    VERIFY_FUNCTION_IR(function);

    // Test call

    llvm::Value* fp1 = LLVMType<float>::get(C, 1.0f);
    llvm::Value* fp2 = LLVMType<float>::get(C, 2.0f);
    llvm::Value* result = test->call({fp1, fp2}, B);
    ASSERT_TRUE(result);
    llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>(result);
    ASSERT_TRUE(call);
    ASSERT_EQ(function, call->getCalledFunction());
    ASSERT_EQ(2u, getNumArgFromCallInst(call));
    ASSERT_EQ(fp1, call->getArgOperand(0));
    ASSERT_EQ(fp2, call->getArgOperand(1));
    // Test the builder is pointing to the correct location
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    ASSERT_EQ(llvm::cast<llvm::Value>(call),
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
            ArgInfo(llvm::Type::getFloatTy(C)),
            ArgInfo(llvm::Type::getFloatTy(C))
        },
        ArgInfo(llvm::Type::getFloatTy(C)),
        expectedName, generate));

    ASSERT_TRUE(!M.getFunction("ax.ir.autoret.test"));
    result = test->call({fp1, fp2}, B);
    ASSERT_TRUE(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    ASSERT_TRUE(call);
    function = M.getFunction("ax.ir.autoret.test");
    ASSERT_EQ(function, call->getCalledFunction());
    ASSERT_EQ(2u, getNumArgFromCallInst(call));
    ASSERT_EQ(fp1, call->getArgOperand(0));
    ASSERT_EQ(fp2, call->getArgOperand(1));

    ASSERT_TRUE(!function->empty());
    BB = &(function->getEntryBlock());
    // two instructions - the add and return
    ASSERT_EQ(size_t(2), BB->size());
    iter = BB->begin();
    binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    ASSERT_TRUE(binary);
    ASSERT_EQ(llvm::Instruction::FAdd, binary->getOpcode());
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    ASSERT_TRUE(ret);
    rvalue = ret->getReturnValue();
    ASSERT_TRUE(rvalue);
    ASSERT_TRUE(rvalue->getType()->isFloatTy());
    // the return is the result of the bin op
    ASSERT_EQ(rvalue, llvm::cast<llvm::Value>(binary));

    ++iter;
    ASSERT_TRUE(BB->end() == iter);

    // Test the builder is pointing to the correct location
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    ASSERT_EQ(llvm::cast<llvm::Value>(call),
        llvm::cast<llvm::Value>(--B.GetInsertPoint()));

    // verify
    VERIFY_FUNCTION_IR(function);
    VERIFY_FUNCTION_IR(BaseFunction);
    VERIFY_MODULE_IR(&M);

    // Test invalid return

    expectedName = "ax.ir.retnull.test";
    termMode = 2;
    test.reset(new TestIRFunction({
            ArgInfo(llvm::Type::getFloatTy(C)),
            ArgInfo(llvm::Type::getFloatTy(C))
        },
        ArgInfo(llvm::Type::getFloatTy(C)),
        expectedName, generate));

    ASSERT_TRUE(!M.getFunction("ax.ir.retnull.test"));
    // will throw as the function expects a float ret, not void or null
    // NOTE: The function will still be created, but be in an invaid state
    ASSERT_THROW(test->create(M), openvdb::AXCodeGenError);
    function = M.getFunction("ax.ir.retnull.test");
    ASSERT_TRUE(function);

    result = test->call({fp1, fp2}, B);
    ASSERT_TRUE(result);
    call = llvm::dyn_cast<llvm::CallInst>(result);
    ASSERT_TRUE(call);
    ASSERT_EQ(function, call->getCalledFunction());
    ASSERT_EQ(2u, getNumArgFromCallInst(call));
    ASSERT_EQ(fp1, call->getArgOperand(0));
    ASSERT_EQ(fp2, call->getArgOperand(1));

    BB = &(function->getEntryBlock());
    // two instructions - the add and return
    ASSERT_EQ(size_t(2), BB->size());
    iter = BB->begin();
    binary = llvm::dyn_cast<llvm::BinaryOperator>(iter);
    ASSERT_TRUE(binary);
    ASSERT_EQ(llvm::Instruction::FAdd, binary->getOpcode());
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()),
        binary->getOperand(0));
    ASSERT_EQ(llvm::cast<llvm::Value>(function->arg_begin()+1),
        binary->getOperand(1));

    ++iter;
    ret = llvm::dyn_cast<llvm::ReturnInst>(iter);
    ASSERT_TRUE(ret);
    ASSERT_TRUE(!ret->getReturnValue());

    ++iter;
    ASSERT_TRUE(BB->end() == iter);

    // Test the builder is pointing to the correct location
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
    ASSERT_EQ(llvm::cast<llvm::Value>(call),
        llvm::cast<llvm::Value>(--B.GetInsertPoint()));

    // verify - function is invalid as it returns void but the
    // prototype wants a float
    VERIFY_FUNCTION_IR_INVALID(function);
    VERIFY_FUNCTION_IR(BaseFunction);
    VERIFY_MODULE_IR_INVALID(&M);

    //
    // Test embedded IR

    auto embdedGen = [&B, &M]
        (const Arguments& args,
         llvm::IRBuilder<>& FunctionBB) -> Value
    {
        // NOTE: GTest's ASSERT_* macros create a 'return;' statement
        // which errors here. Exceptions are being used instead to
        // exit early.

        // test the builder is pointing to the correct location
        // note, for embedded IR, the same builder will be used
        if (&B != &FunctionBB)
        {
            throw std::runtime_error("Builders should be the same");
        }
        llvm::BasicBlock* Block = FunctionBB.GetInsertBlock();
        if (!Block || Block->empty())
        {
            throw std::runtime_error("Basic block should exist and be populated");
        }
        llvm::Function* F = Block->getParent();
        if (!F)
        {
            throw std::runtime_error("Function should exist");
        }
        EXPECT_EQ(std::string("TestFunction"), std::string(F->getName()));
        EXPECT_EQ(&M, F->getParent());
        if (args.size() != 2)
        {
            throw std::runtime_error("Function should take two arguments");
        }
        EXPECT_TRUE(args[0]->getType()->isFloatTy());
        EXPECT_TRUE(args[1]->getType()->isFloatTy());
        // Can't just do a CreateFAdd as the IR builder won't actually even
        // write the instruction as its const and unused - so store in a new
        // alloc
        llvm::Value* alloc = FunctionBB.CreateAlloca(args[0]->getType());
        FunctionBB.CreateStore(args[0], alloc);
        llvm::Value* result = FunctionBB.CreateLoad(args[0]->getType(), alloc);
        return Value(result, args[0]->getType());
    };

    test.reset(new TestIRFunction({
            ArgInfo(llvm::Type::getFloatTy(C)),
            ArgInfo(llvm::Type::getFloatTy(C))
        },
        ArgInfo(llvm::Type::getFloatTy(C)),
        "ax.ir.embed.test", embdedGen));
    static_cast<IRFunctionBase&>(*test).setEmbedIR(true);

    // test create does nothing
    ASSERT_TRUE(test->create(C) == nullptr);
    ASSERT_TRUE(test->create(M) == nullptr);

    // test call
    ASSERT_TRUE(!M.getFunction("ax.ir.embed.test"));
    result = test->call({fp1, fp2}, B);
    ASSERT_TRUE(result);
    ASSERT_TRUE(!M.getFunction("ax.ir.embed.test"));
    auto IP = B.GetInsertPoint();
    // check the prev instructions are as expected
    ASSERT_TRUE(llvm::isa<llvm::LoadInst>(--IP));
    ASSERT_TRUE(llvm::isa<llvm::StoreInst>(--IP));
    ASSERT_TRUE(llvm::isa<llvm::AllocaInst>(--IP));
    // Test the builder is pointing to the correct location
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());
}

// Test SRET methods for both C and IR functions
TEST_F(TestFunctionTypes, testSRETFunctions)
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::CFunctionSRet;
    using openvdb::ax::codegen::IRFunctionSRet;
    using openvdb::ax::codegen::ArgInfo;
    using openvdb::ax::codegen::ArgInfoVector;
    using openvdb::ax::codegen::Arguments;
    using openvdb::ax::codegen::Value;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();
    llvm::IRBuilder<> B(state.scratchBlock());

#if LLVM_VERSION_MAJOR <= 15
    std::vector<llvm::Type*> types;
#else
    ArgInfoVector types;
#endif
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

    auto type = test->types(types, C);
    ASSERT_EQ(size_t(1), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(types[0] == vec3f->getPointerTo(0));
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());
#else
    ASSERT_TRUE(types[0] == ArgInfo(vec3f, 1, true));
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(type.IsReturn());
#endif
    // test various getters
    ASSERT_EQ(std::string("ax.c.test"), std::string(test->symbol()));
    ASSERT_EQ(size_t(1), test->size());
    ASSERT_EQ(std::string(""), std::string(test->argName(0)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("vec3f name()"), os.str());

    // test match
#if LLVM_VERSION_MAJOR <= 15
    match = test->match({vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match(std::vector<llvm::Type*>{}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->Function::match({vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Explicit);
#else
    match = test->match({ArgInfo(vec3f, 1)}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match(ArgInfoVector{}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->Function::match({ArgInfo(vec3f, 1, true)}, C);
    ASSERT_EQ(match, Function::Explicit);
#endif

    // test create
    ASSERT_TRUE(!M.getFunction("ax.c.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.c.test"));
    ASSERT_EQ(function, test->create(M));
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(1), function->arg_size());
    ASSERT_TRUE(function->getAttributes().isEmpty());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(1u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0) == llvm::PointerType::get(vec3f, 0));

    // test call - sret function do not return the CallInst as the value
    result = test->call(Arguments{}, B).GetValue();
    ASSERT_TRUE(result);
    ASSERT_TRUE(!llvm::dyn_cast<llvm::CallInst>(result));
    ASSERT_TRUE(result->getType() == llvm::PointerType::get(vec3f, 0));
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

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
    ASSERT_EQ(size_t(2), types.size());
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_TRUE(types[0] == vec3f->getPointerTo(0));
    ASSERT_TRUE(types[1] == vec3f->getPointerTo(0));
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());
#else
    ASSERT_TRUE(types[0] == ArgInfo(vec3f, 1, true));
    ASSERT_TRUE(types[0].IsReturn());
    ASSERT_TRUE(types[1] == ArgInfo(vec3f, 1));
    ASSERT_TRUE(type.IsVoid());
    ASSERT_TRUE(type.IsReturn());
#endif
    // test various getters
    ASSERT_EQ(std::string("ax.c2.test"), std::string(test->symbol()));
    ASSERT_EQ(size_t(2), test->size());
    ASSERT_EQ(std::string(""), std::string(test->argName(0)));
    ASSERT_EQ(std::string(""), std::string(test->argName(1)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("vec3f name(vec3f)"), os.str());

    // test match
#if LLVM_VERSION_MAJOR <= 15
    match = test->match({vec3f->getPointerTo(),vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match({vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->match(std::vector<llvm::Type*>{}, C);
    ASSERT_EQ(match, Function::None);
    match = test->Function::match({vec3f->getPointerTo(),vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Explicit);
#else
    match = test->match({ArgInfo(vec3f, 1), ArgInfo(vec3f, 1)}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match({ArgInfo(vec3f, 1)}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->match(ArgInfoVector{}, C);
    ASSERT_EQ(match, Function::None);
    match = test->Function::match({ArgInfo(vec3f, 1, true), ArgInfo(vec3f, 1)}, C);
    ASSERT_EQ(match, Function::Explicit);
#endif

    // test create
    ASSERT_TRUE(!M.getFunction("ax.c2.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_TRUE(M.getFunction("ax.c2.test"));
    ASSERT_EQ(function, test->create(M));
    ASSERT_TRUE(function->empty());
    ASSERT_EQ(size_t(2), function->arg_size());
    ASSERT_TRUE(function->getAttributes().isEmpty());

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(2u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0) == llvm::PointerType::get(vec3f, 0));
    ASSERT_TRUE(ftype->getParamType(1) == llvm::PointerType::get(vec3f, 0));

    // test call - sret function do not return the CallInst as the value
    Value f32c0 = Value::Create<float>(C, 0.0f); // float
    llvm::Value* vec3fv = Value::ScalarsToArray(B, {f32c0,f32c0,f32c0}).GetValue(); // vec3f
    result = test->call({vec3fv}, B);
    ASSERT_TRUE(result);
    ASSERT_TRUE(!llvm::dyn_cast<llvm::CallInst>(result));
    ASSERT_TRUE(result->getType() == llvm::PointerType::get(vec3f, 0));
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

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
        (const Arguments& args,
         llvm::IRBuilder<>& FunctionBB) -> Value
    {
        // NOTE: GTest's ASSERT_* macros create a 'return;' statement
        // which errors here. Exceptions are being used instead to
        // exit early.

        // test the builder is pointing to the correct location
        if (&B == &FunctionBB)
        {
            throw std::runtime_error("Builders should be different");
        }
        llvm::BasicBlock* Block = FunctionBB.GetInsertBlock();
        if (!Block || !Block->empty()) {
            throw std::runtime_error("Basic block should exist and be empty");
        }
        llvm::Function* F = Block->getParent();
        if (!F)
        {
            throw std::runtime_error("Function should exist");
        }
        EXPECT_EQ(std::string("ax.ir.test"), std::string(F->getName()));
        llvm::Module* _M = F->getParent();
        EXPECT_EQ(&M, _M);

        if (args.size() != 1){
            throw std::runtime_error("Function should take one argument");
        }

        llvm::Type* arrayT = llvm::ArrayType::get(llvm::Type::getFloatTy(FunctionBB.getContext()), 3);
        EXPECT_TRUE(args[0] == llvm::cast<llvm::Value>(F->arg_begin()));
 #if LLVM_VERSION_MAJOR > 15
        EXPECT_TRUE(args.GetArgInfo(0) == ArgInfo(arrayT, 1, true));
 #endif

        llvm::Value* e0 = FunctionBB.CreateConstGEP2_64(arrayT, args[0], 0, 0);
        FunctionBB.CreateStore(LLVMType<float>::get(FunctionBB.getContext(), 1.0f), e0);
        return Value::Invalid();
    };

    test.reset(new IRFunctionSRet<void(float(*)[3])>("ax.ir.test", generate));
    types.clear();

    // test types
    type = test->types(types, C);
#if LLVM_VERSION_MAJOR <= 15
    ASSERT_EQ(size_t(1), types.size());
    ASSERT_TRUE(types[0] == vec3f->getPointerTo(0));
    ASSERT_TRUE(type);
    ASSERT_TRUE(type->isVoidTy());
#else
    ASSERT_EQ(size_t(1), types.size());
    ASSERT_TRUE(types[0] == ArgInfo(vec3f, 1, true));
    ASSERT_TRUE(type.IsVoid());
#endif
    // test various getters
    ASSERT_EQ(std::string("ax.ir.test"), std::string(test->symbol()));
    ASSERT_EQ(size_t(1), test->size());
    ASSERT_EQ(std::string(""), std::string(test->argName(0)));

    // test printing
    os.str("");
    test->print(C, os, "name", /*axtypes=*/true);
    ASSERT_EQ(std::string("vec3f name()"), os.str());

    // test match
#if LLVM_VERSION_MAJOR <= 15
    match = test->match({vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match(std::vector<llvm::Type*>{}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->Function::match({vec3f->getPointerTo()}, C);
    ASSERT_EQ(match, Function::Explicit);
#else
    match = test->match({ArgInfo(vec3f, 1)}, C);
    ASSERT_EQ(match, Function::None);
    match = test->match(ArgInfoVector{}, C);
    ASSERT_EQ(match, Function::Explicit);
    match = test->Function::match({ArgInfo(vec3f, 1, true)}, C);
    ASSERT_EQ(match, Function::Explicit);
#endif

    // test create
    ASSERT_TRUE(!M.getFunction("ax.ir.test"));
    function = test->create(M);
    ASSERT_TRUE(function);
    ASSERT_EQ(function, M.getFunction("ax.ir.test"));
    ASSERT_TRUE(!function->empty());

    // test instructions
    llvm::BasicBlock* BB = &(function->getEntryBlock());
    ASSERT_EQ(size_t(3), BB->size());
    auto iter = BB->begin();
    ASSERT_TRUE(llvm::isa<llvm::GetElementPtrInst>(iter++));
    ASSERT_TRUE(llvm::isa<llvm::StoreInst>(iter++));
    ASSERT_TRUE(llvm::isa<llvm::ReturnInst>(iter++));
    ASSERT_TRUE(BB->end() == iter);

    // additional call should match
    ASSERT_EQ(function, test->create(M));
    ASSERT_EQ(size_t(1), function->arg_size());
    ASSERT_TRUE(function->getAttributes().isEmpty());

    // check function type

    ftype = function->getFunctionType();
    ASSERT_TRUE(ftype);
    ASSERT_TRUE(ftype->getReturnType()->isVoidTy());
    ASSERT_EQ(1u, ftype->getNumParams());
    ASSERT_TRUE(ftype->getParamType(0) == llvm::PointerType::get(vec3f, 0));

    // test call - sret function do not return the CallInst as the value
    result = test->call(Arguments{}, B).GetValue();
    ASSERT_TRUE(result);
    ASSERT_TRUE(!llvm::dyn_cast<llvm::CallInst>(result));
    ASSERT_TRUE(result->getType() == llvm::PointerType::get(vec3f, 0));
    ASSERT_EQ(&(BaseFunction->getEntryBlock()), B.GetInsertBlock());

    VERIFY_FUNCTION_IR(function);
    VERIFY_MODULE_IR(&M);
}
