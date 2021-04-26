// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/codegen/FunctionTypes.h>

#include <cppunit/extensions/HelperMacros.h>

#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// Framework methods for the subsequent unit tests

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

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestscalar(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    llvm::Type* voidty = llvm::Type::getVoidTy(C);
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new TestFunction({llvm::Type::getDoubleTy(C)}, voidty, "ax.testd")),
            Function::Ptr(new TestFunction({llvm::Type::getFloatTy(C)}, voidty, "ax.testf")),
            Function::Ptr(new TestFunction({llvm::Type::getInt64Ty(C)}, voidty, "ax.testi64")),
            Function::Ptr(new TestFunction({llvm::Type::getInt32Ty(C)}, voidty, "ax.testi32")),
            Function::Ptr(new TestFunction({llvm::Type::getInt16Ty(C)}, voidty, "ax.testi16")),
            Function::Ptr(new TestFunction({llvm::Type::getInt1Ty(C)}, voidty, "ax.testi1"))
         }));

    return group;
}

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestsize(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    llvm::Type* voidty = llvm::Type::getVoidTy(C);
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new TestFunction({}, voidty, "ax.empty")),
            Function::Ptr(new TestFunction({llvm::Type::getDoubleTy(C)}, voidty, "ax.d")),
            Function::Ptr(new TestFunction({
                llvm::Type::getDoubleTy(C),
                llvm::Type::getDoubleTy(C)
            }, voidty, "ax.dd")),
         }));

    return group;
}

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestmulti(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    llvm::Type* voidty = llvm::Type::getVoidTy(C);
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new TestFunction({}, voidty, "ax.empty")),
            Function::Ptr(new TestFunction({llvm::Type::getInt32Ty(C)}, voidty, "ax.i32")),
            Function::Ptr(new TestFunction({
                llvm::Type::getDoubleTy(C),
                llvm::Type::getDoubleTy(C)
            }, voidty, "ax.dd")),
            Function::Ptr(new TestFunction({
                llvm::Type::getInt32Ty(C),
                llvm::Type::getDoubleTy(C)
            }, voidty, "ax.i32d")),
            Function::Ptr(new TestFunction({
                llvm::Type::getDoubleTy(C)->getPointerTo(),
                llvm::Type::getInt32Ty(C),
                llvm::Type::getDoubleTy(C)
            }, voidty, "ax.d*i32d")),
            Function::Ptr(new TestFunction({
                llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1)->getPointerTo(),
            }, voidty, "ax.i32x1")),
            Function::Ptr(new TestFunction({
                llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo(),
            }, voidty, "ax.i32x2")),
         }));

    return group;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

class TestFunctionGroup : public CppUnit::TestCase
{
public:

    // Test FunctionGroup signature matching and execution errors
    CPPUNIT_TEST_SUITE(TestFunctionGroup);
    CPPUNIT_TEST(testFunctionGroup);
    CPPUNIT_TEST(testMatch);
    CPPUNIT_TEST(testExecute);
    CPPUNIT_TEST_SUITE_END();

    void testFunctionGroup();
    void testMatch();
    void testExecute();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestFunctionGroup);

void
TestFunctionGroup::testFunctionGroup()
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    llvm::Type* voidty = llvm::Type::getVoidTy(C);

    Function::Ptr decl1(new TestFunction({}, voidty, "ax.test1"));
    Function::Ptr decl2(new TestFunction({}, voidty, "ax.test2"));
    Function::Ptr decl3(new TestFunction({}, voidty, "ax.test3"));

    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            decl1, decl2, decl3
         }));

    CPPUNIT_ASSERT_EQUAL(std::string("test"), std::string(group->name()));
    CPPUNIT_ASSERT_EQUAL(std::string("The documentation"), std::string(group->doc()));
    CPPUNIT_ASSERT_EQUAL(size_t(3), group->list().size());
    CPPUNIT_ASSERT_EQUAL(decl1, group->list()[0]);
    CPPUNIT_ASSERT_EQUAL(decl2, group->list()[1]);
    CPPUNIT_ASSERT_EQUAL(decl3, group->list()[2]);
}

void
TestFunctionGroup::testMatch()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    std::vector<llvm::Type*> types;
    Function::SignatureMatch match;
    const Function* result;

    //

    FunctionGroup::Ptr group = axtestscalar(C);
    const std::vector<Function::Ptr>* list = &group->list();

    // test explicit matching

    types.resize(1);
    types[0] = llvm::Type::getInt1Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[5].get()), result);

    //

    types[0] = llvm::Type::getInt16Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[4].get()), result);

    //

    types[0] = llvm::Type::getInt32Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[3].get()), result);

    //

    types[0] = llvm::Type::getInt64Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    //

    types[0] = llvm::Type::getFloatTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[1].get()), result);

    //

    types[0] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result);

    // test unsigned integers automatic type creation - these are not supported in the
    // language however can be constructed from the API. The function framework does
    // not differentiate between signed and unsigned integers

    types[0] = LLVMType<uint64_t>::get(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    // test implicit matching - types should match to the first available castable signature
    // which is always the void(double) "tsfd" function for all provided scalars

    types[0] = llvm::Type::getInt8Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result);

    types.clear();

    // test invalid matching - Size matching returns the first function which matched
    // the size

    result = group->match(types, C, &match);
    CPPUNIT_ASSERT_EQUAL(Function::SignatureMatch::None, match);
    CPPUNIT_ASSERT(!result);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == match);
    CPPUNIT_ASSERT(!result);

    //

    types.emplace_back(llvm::Type::getInt1Ty(C)->getPointerTo());
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Size == match);
    CPPUNIT_ASSERT(!result);

    //

    types[0] = llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Size == match);
    CPPUNIT_ASSERT(!result);

    //

    types[0] = llvm::Type::getInt1Ty(C);
    types.emplace_back(llvm::Type::getInt1Ty(C));
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == match);
    CPPUNIT_ASSERT(!result);

    //
    // Test varying argument size function
    // test explicit matching

    group = axtestsize(C);
    list = &group->list();

    types.resize(2);
    types[0] = llvm::Type::getDoubleTy(C);
    types[1] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    //

    types.resize(1);
    types[0] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[1].get()), result);

    //

    types.clear();
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result);

    // Test implicit matching

    types.resize(2);

    types[0] = llvm::Type::getFloatTy(C);
    types[1] = llvm::Type::getInt32Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    // Test non matching

    types.resize(3);

    types[0] = llvm::Type::getDoubleTy(C);
    types[1] = llvm::Type::getDoubleTy(C);
    types[2] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == match);
    CPPUNIT_ASSERT(!result);

    //
    // Test multi function

    group = axtestmulti(C);
    list = &group->list();

    // test explicit/implicit matching

    types.clear();

    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result);

    //

    types.resize(2);
    types[0] = llvm::Type::getDoubleTy(C);
    types[1] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    //

    types[0] = llvm::Type::getInt32Ty(C);
    types[1] = llvm::Type::getDoubleTy(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[3].get()), result);

    //

    types[0] = llvm::Type::getInt32Ty(C);
    types[1] = llvm::Type::getInt32Ty(C);
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result);

    //

    types.resize(1);

    types[0] = llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1)->getPointerTo();
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[5].get()), result);

    //

    types[0] = llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2)->getPointerTo();
    result = group->match(types, C, &match);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == match);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[6].get()), result);
}

void
TestFunctionGroup::testExecute()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::IRBuilder<> B(state.scratchBlock());
    llvm::Value* result = nullptr;
    llvm::CallInst* call = nullptr;
    llvm::Function* target = nullptr;
    std::vector<llvm::Value*> args;

    // test execution
    // test invalid arguments throws

    FunctionGroup::Ptr group(new FunctionGroup("empty", "", {}));
    CPPUNIT_ASSERT(!group->execute(/*args*/{}, B, result));

    group = axtestscalar(C);
    const std::vector<Function::Ptr>* list = &group->list();

    CPPUNIT_ASSERT(!group->execute({}, B, result));
    CPPUNIT_ASSERT(!group->execute({
            B.getTrue(),
            B.getTrue()
        }, B, result));

    args.resize(1);

    // test llvm function calls - execute and get the called function.
    // check this is already inserted into the module and is expected
    // llvm::Function using create on the expected function signature

    args[0] = B.getTrue();
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[5]->create(state.module()), target);

    //

    args[0] = B.getInt16(1);
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[4]->create(state.module()), target);

    //

    args[0] = B.getInt32(1);
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[3]->create(state.module()), target);

    //

    args[0] = B.getInt64(1);
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[2]->create(state.module()), target);

    //

    args[0] = llvm::ConstantFP::get(llvm::Type::getFloatTy(C), 1.0f);
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[1]->create(state.module()), target);

    //

    args[0] = llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0);
    result = group->execute(args, B);

    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[0]->create(state.module()), target);

    //
    // Test multi function

    group = axtestmulti(C);

    list = &group->list();
    args.clear();

    result = group->execute(args, B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[0]->create(state.module()), target);

    //

    args.resize(1);
    args[0] = B.CreateAlloca(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1));

    result = group->execute(args, B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[5]->create(state.module()), target);

    //

    args[0] = B.CreateAlloca(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2));

    result = group->execute(args, B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[6]->create(state.module()), target);

    //

    args.resize(2);
    args[0] = llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0);
    args[1] = llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0);

    result = group->execute(args, B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(result));
    call = llvm::cast<llvm::CallInst>(result);
    CPPUNIT_ASSERT(call);
    target = call->getCalledFunction();
    CPPUNIT_ASSERT(target);
    CPPUNIT_ASSERT_EQUAL((*list)[2]->create(state.module()), target);
}

