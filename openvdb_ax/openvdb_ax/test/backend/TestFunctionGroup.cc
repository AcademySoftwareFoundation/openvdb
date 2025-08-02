// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "util.h"

#include <openvdb_ax/codegen/FunctionTypes.h>

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/Config/llvm-config.h>

#include <memory>
#include <string>

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

// Framework methods for the subsequent unit tests

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestscalar(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    using openvdb::ax::codegen::ArgInfo;
    ArgInfo voidty(llvm::Type::getVoidTy(C));
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getDoubleTy(C))}, voidty, "ax.testd")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getFloatTy(C))}, voidty, "ax.testf")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt64Ty(C))}, voidty, "ax.testi64")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt32Ty(C))}, voidty, "ax.testi32")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt16Ty(C))}, voidty, "ax.testi16")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt1Ty(C))}, voidty, "ax.testi1"))
         }));

    return group;
}

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestsize(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    using openvdb::ax::codegen::ArgInfo;
    ArgInfo voidty(llvm::Type::getVoidTy(C));
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new unittest_util::TestFunction({}, voidty, "ax.empty")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getDoubleTy(C))}, voidty, "ax.d")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::Type::getDoubleTy(C)),
                ArgInfo(llvm::Type::getDoubleTy(C))
            }, voidty, "ax.dd")),
         }));

    return group;
}

inline openvdb::ax::codegen::FunctionGroup::Ptr
axtestmulti(llvm::LLVMContext& C)
{
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    using openvdb::ax::codegen::ArgInfo;
    ArgInfo voidty(llvm::Type::getVoidTy(C));
    using openvdb::ax::codegen::ArgInfo;
    FunctionGroup::Ptr group(new FunctionGroup("test",
        "The documentation", {
            Function::Ptr(new unittest_util::TestFunction({}, voidty, "ax.empty")),
            Function::Ptr(new unittest_util::TestFunction({ArgInfo(llvm::Type::getInt32Ty(C))}, voidty, "ax.i32")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::Type::getDoubleTy(C)),
                ArgInfo(llvm::Type::getDoubleTy(C))
            }, voidty, "ax.dd")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::Type::getInt32Ty(C)),
                ArgInfo(llvm::Type::getDoubleTy(C))
            }, voidty, "ax.i32d")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::Type::getDoubleTy(C), 1),
                ArgInfo(llvm::Type::getInt32Ty(C)),
                ArgInfo(llvm::Type::getDoubleTy(C))
            }, voidty, "ax.d*i32d")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1), 1)
            }, voidty, "ax.i32x1")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 1)
            }, voidty, "ax.i32x2")),
            Function::Ptr(new unittest_util::TestFunction({
                ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3), 1)
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
    using openvdb::ax::codegen::ArgInfo;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();

    ArgInfo voidty(llvm::Type::getVoidTy(C));

    Function::Ptr decl1(new unittest_util::TestFunction({}, voidty, "ax.test1"));
    Function::Ptr decl2(new unittest_util::TestFunction({}, voidty, "ax.test2"));
    Function::Ptr decl3(new unittest_util::TestFunction({}, voidty, "ax.test3"));

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

#if LLVM_VERSION_MAJOR <= 15
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

#else

void
TestFunctionGroup::testMatch()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    using openvdb::ax::codegen::ArgInfo;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    ArgInfo int1t(llvm::Type::getInt1Ty(C));
    ArgInfo int8t(llvm::Type::getInt8Ty(C));
    ArgInfo int16t(llvm::Type::getInt16Ty(C));
    ArgInfo int32t(llvm::Type::getInt32Ty(C));
    ArgInfo int64t(llvm::Type::getInt64Ty(C));
    ArgInfo floatt(llvm::Type::getFloatTy(C));
    ArgInfo doublet(llvm::Type::getDoubleTy(C));

    //

    FunctionGroup::Ptr group = axtestscalar(C);
    const std::vector<Function::Ptr>* list = &group->list();

    // test explicit matching

    auto result = group->match({int1t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[5].get()), result.first);

    //

    result = group->match({int16t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[4].get()), result.first);

    //

    result = group->match({int32t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[3].get()), result.first);

    //

    result = group->match({int64t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    //

    result = group->match({floatt}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[1].get()), result.first);

    //

    result = group->match({doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result.first);

    // test unsigned integers automatic type creation - these are not supported in the
    // language however can be constructed from the API. The function framework does
    // not differentiate between signed and unsigned integers

    result = group->match({ArgInfo(LLVMType<uint64_t>::get(C))}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    // test implicit matching - types should match to the first available castable signature
    // which is always the void(double) "tsfd" function for all provided scalars

    result = group->match({int8t}, C);
#if LLVM_VERSION_MAJOR <= 15
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == result.second);
#else
    CPPUNIT_ASSERT(Function::SignatureMatch::Ambiguous == result.second);
#endif
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result.first);

    // test invalid matching - Size matching returns the first function which matched
    // the size

    result = group->match({}, C);
    CPPUNIT_ASSERT_EQUAL(Function::SignatureMatch::None, result.second);
    CPPUNIT_ASSERT(!result.first);
    result = group->match({}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == result.second);
    CPPUNIT_ASSERT(!result.first);

    //

    result = group->match({ArgInfo(int1t.GetUnderlyingType(), 1)}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Size == result.second);
    CPPUNIT_ASSERT(!result.first);

    //

    result = group->match({ArgInfo(llvm::ArrayType::get(llvm::Type::getInt1Ty(C), 1))}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Size == result.second);
    CPPUNIT_ASSERT(!result.first);

    //

    result = group->match({int1t, int1t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == result.second);
    CPPUNIT_ASSERT(!result.first);

    //
    // Test varying argument size function
    // test explicit matching

    group = axtestsize(C);
    list = &group->list();

    result = group->match({doublet, doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    //

    result = group->match({doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[1].get()), result.first);

    //

    result = group->match({}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result.first);

    // Test implicit matching

    result = group->match({floatt, int32t}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    // Test non matching

    result = group->match({doublet, doublet, doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::None == result.second);
    CPPUNIT_ASSERT(!result.first);

    //
    // Test multi function

    group = axtestmulti(C);
    list = &group->list();

    // test explicit/implicit matching

    result = group->match({}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[0].get()), result.first);

    //

    result = group->match({doublet, doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    //

    result = group->match({int32t, doublet}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[3].get()), result.first);

    //

    result = group->match({int32t, int32t}, C);
#if LLVM_VERSION_MAJOR <= 15
    CPPUNIT_ASSERT(Function::SignatureMatch::Implicit == result.second);
#else
    CPPUNIT_ASSERT(Function::SignatureMatch::Ambiguous == result.second);
#endif
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[2].get()), result.first);

    //

    result = group->match({ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 1), 1)}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[5].get()), result.first);

    //

    result = group->match({ArgInfo(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2), 1)}, C);
    CPPUNIT_ASSERT(Function::SignatureMatch::Explicit == result.second);
    CPPUNIT_ASSERT(result.first);
    CPPUNIT_ASSERT_EQUAL(const_cast<const Function*>((*list)[6].get()), result.first);
}
#endif

void
TestFunctionGroup::testExecute()
{
    using openvdb::ax::codegen::LLVMType;
    using openvdb::ax::codegen::Function;
    using openvdb::ax::codegen::FunctionGroup;
    using openvdb::ax::codegen::NativeArguments;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::IRBuilder<> B(state.scratchBlock());

#if LLVM_VERSION_MAJOR <= 15
    auto get_args = [](const std::vector<llvm::Value*>& values) { return values; };
    auto get_called_fn = [](llvm::Value* value)
    {
        CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(value));
        auto call = llvm::cast<llvm::CallInst>(value);
        CPPUNIT_ASSERT(call);
        auto target = call->getCalledFunction();
        CPPUNIT_ASSERT(target);
        return target;
    };
#else
    auto get_args = [](const std::vector<llvm::Value*>& values)
    {
        NativeArguments args;
        for (auto& v : values) {
            llvm::Type* utype = v->getType();
            if (utype->isPointerTy()) {
                auto alloc = llvm::cast<llvm::AllocaInst>(v);
                CPPUNIT_ASSERT(alloc);
                utype = alloc->getAllocatedType();
            }
            args.AddArg(openvdb::ax::codegen::Value(v, utype));
        }
        return args;
    };
    auto get_called_fn = [](openvdb::ax::codegen::Value value)
    {
        CPPUNIT_ASSERT(llvm::isa<llvm::CallInst>(value.GetValue()));
        auto call = llvm::cast<llvm::CallInst>(value.GetValue());
        CPPUNIT_ASSERT(call);
        auto target = call->getCalledFunction();
        CPPUNIT_ASSERT(target);
        return target;
    };
#endif

    // test execution
    // test invalid arguments throws

    FunctionGroup::Ptr group(new FunctionGroup("empty", "", {}));
#if LLVM_VERSION_MAJOR <= 15
    llvm::Value* output = nullptr;

    CPPUNIT_ASSERT(!group->execute(/*args*/{}, B, output));

    group = axtestscalar(C);
    const std::vector<Function::Ptr>* list = &group->list();

    CPPUNIT_ASSERT(!group->execute({}, B, output));
    CPPUNIT_ASSERT(!group->execute({
            B.getTrue(),
            B.getTrue()
        }, B, output));
#else
    // @note  Calling execute with invalid Arguments{} is UB. Only defined
    //   with NativeArguments{}
    CPPUNIT_ASSERT(!group->execute(NativeArguments{}, B));

    group = axtestscalar(C);
    const std::vector<Function::Ptr>* list = &group->list();

    CPPUNIT_ASSERT(!group->execute(NativeArguments{}, B));
    CPPUNIT_ASSERT(!group->execute(get_args({
            B.getTrue(),
            B.getTrue()
        }), B));
#endif

    // test llvm function calls - execute and get the called function.
    // check this is already inserted into the module and is expected
    // llvm::Function using create on the expected function signature

    auto result = group->execute(get_args({B.getTrue()}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[5]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({B.getInt16(1)}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[4]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({B.getInt32(1)}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[3]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({B.getInt64(1)}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[2]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({llvm::ConstantFP::get(llvm::Type::getFloatTy(C), 1.0f)}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[1]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0)}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[0]->create(state.module()), get_called_fn(result));

    //
    // Test multi function

    group = axtestmulti(C);
    list = &group->list();

    result = group->execute(get_args({}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[0]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({B.CreateAlloca(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 2))}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[6]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({B.CreateAlloca(llvm::ArrayType::get(llvm::Type::getInt32Ty(C), 3))}), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[7]->create(state.module()), get_called_fn(result));

    //

    result = group->execute(get_args({
        llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0),
        llvm::ConstantFP::get(llvm::Type::getDoubleTy(C), 1.0)
    }), B);
    CPPUNIT_ASSERT(result);
    CPPUNIT_ASSERT_EQUAL((*list)[2]->create(state.module()), get_called_fn(result));
}

