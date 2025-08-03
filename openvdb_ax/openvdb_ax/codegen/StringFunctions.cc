// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/StringFunctions.cc
///
/// @authors Nick Avramoussis
///
/// @brief  A set of internal AX/IR functions for strings
///

#include "Functions.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"
#include "String.h"

#include <openvdb/util/Assert.h>

#include "openvdb_ax/compiler/CompilerOptions.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

// String

inline FunctionGroup::UniquePtr axstrlen(const FunctionOptions& op)
{
    // @todo llvm::emitStrLen(args[1], B, M->getDataLayout()); from llvm/BuildLibCalls.h
    //  The emitStrLen requires the TargetLibraryInfo class, although this is
    //  only used to verify that the platform actually has strlen. The main
    //  benefit of calling this method is for function and parameter attribute
    //  tagging. TargetLibraryInfo is fairly expensive to construct so should
    //  be passed by the compiler if we need it
    return FunctionBuilder("strlen")
        .addSignature<std::size_t(const char*)>(std::strlen, "strlen")
        .setArgumentNames({"ptr"})
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(FunctionBuilder::C)
        .setDocumentation("strlen")
        .get();
}

inline FunctionGroup::UniquePtr axstringalloc(const FunctionOptions& op)
{
    auto generate =
        [](const NativeArguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        OPENVDB_ASSERT(args[0].IsString());
        OPENVDB_ASSERT(args[1].IsInteger());
        llvm::LLVMContext& C = B.getContext();
        llvm::Function* base = B.GetInsertBlock()->getParent();
        llvm::StructType* strType = LLVMType<codegen::String>::get(C);

        Value str = args[0];
        Value size = args[1];
        llvm::Value* cptr = B.CreateStructGEP(strType, str.GetValue(), 0); // char**
        llvm::Value* sso = B.CreateStructGEP(strType, str.GetValue(), 1); // char[]*
        llvm::Value* sso_load = B.CreateConstGEP2_64(strType->getTypeAtIndex(1), sso, 0, 0); // char*

        llvm::Value* cptr_load = B.CreateLoad(strType->getTypeAtIndex(0u), cptr); // char*
        llvm::Value* neq = B.CreateICmpNE(cptr_load, sso_load);

        llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "then", base);
        llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "post", base);
        B.CreateCondBr(neq, then, post);
        B.SetInsertPoint(then);
        {
#if LLVM_VERSION_MAJOR < 18
            llvm::Instruction* inst = llvm::CallInst::CreateFree(cptr_load, B.GetInsertBlock());
            OPENVDB_ASSERT(inst);
            B.Insert(inst);
#else
            B.CreateFree(cptr_load);
#endif
            B.CreateBr(post);
        }

        B.SetInsertPoint(post);

        Value gt = size.GreaterThan(B, Value::Create<int64_t>(C, codegen::String::SSO_LENGTH-1));

        then = llvm::BasicBlock::Create(C, "then", base);
        llvm::BasicBlock* el = llvm::BasicBlock::Create(C, "else", base);
        post = llvm::BasicBlock::Create(C, "post", base);
        B.CreateCondBr(gt.GetValue(), then, el);
        B.SetInsertPoint(then);
        {
#if LLVM_VERSION_MAJOR < 18
            llvm::Instruction* inst = llvm::CallInst::CreateMalloc(B.GetInsertBlock(), // location
                    B.getInt64Ty(), // int ptr type
                    B.getInt8Ty(),  // return type
                    B.CreateAdd(size.GetValue(), B.getInt64(1)), // size
                    nullptr,
                    nullptr);
            OPENVDB_ASSERT(inst);
            B.Insert(inst);
#else
            llvm::Instruction* inst = B.CreateMalloc(
                    B.getInt64Ty(), // int ptr type
                    B.getInt8Ty(),  // return type
                    B.CreateAdd(size.GetValue(), B.getInt64(1)), // size
                    nullptr,
                    nullptr);
            OPENVDB_ASSERT(inst);
#endif
            B.CreateStore(inst, cptr);
            B.CreateBr(post);
        }

        B.SetInsertPoint(el);
        {
            B.CreateStore(sso_load, cptr);
            B.CreateBr(post);
        }

        B.SetInsertPoint(post);
        // re-load cptr
        cptr_load =  B.CreateLoad(strType->getTypeAtIndex(0u), cptr); // char*
        llvm::Value* clast = B.CreateGEP(LLVMType<char>::get(C), cptr_load, size.GetValue());
        B.CreateStore(B.getInt8(int8_t('\0')), clast); // this->ptr[size] = '\0';
        llvm::Value* len = B.CreateStructGEP(strType, str.GetValue(), 2);
        B.CreateStore(size.GetValue(), len);
        return Value::Invalid();
    };

    static auto stralloc = [](codegen::String* str, const int64_t s) {
        str->alloc(s);
    };

    return FunctionBuilder("string::alloc")
        .addSignature<void(codegen::String*, const int64_t)>(generate, stralloc)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axstring(const FunctionOptions& op)
{
    auto generate =
        [op](const NativeArguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() >= 1);

        llvm::LLVMContext& C = B.getContext();
        llvm::StructType* strType = LLVMType<codegen::String>::get(C);
        llvm::Type* ctype = LLVMType<char>::get(C);
        Value str = args[0];

        llvm::Value* carr;
        if (args.size() == 1) carr = B.CreateGlobalStringPtr("");
        else                  carr = args[1].GetValue();
        OPENVDB_ASSERT(carr);

        Arguments strlenargs;
        strlenargs.AddArg(carr, ArgInfo(ctype, 1));
        Value slen = axstrlen(op)->execute(strlenargs, B);

        llvm::Value* cptr = B.CreateStructGEP(strType, str.GetValue(), 0); // char**
        llvm::Value* sso = B.CreateStructGEP(strType, str.GetValue(), 1); // char[]*
        llvm::Value* sso_load = B.CreateConstGEP2_64(strType->getTypeAtIndex(1), sso, 0, 0); // char*
        llvm::Value* len = B.CreateStructGEP(strType, str.GetValue(), 2);
        B.CreateStore(sso_load, cptr); // this->ptr = this->SSO;
        B.CreateStore(B.getInt64(0), len); // this->len = 0;

        axstringalloc(op)->execute(NativeArguments{str, slen}, B);

        llvm::Value* cptr_load = B.CreateLoad(strType->getTypeAtIndex(0u), cptr); // char*
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(cptr_load, /*dest-align*/llvm::MaybeAlign(0),
            carr, /*src-align*/llvm::MaybeAlign(0), slen.GetValue());
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(cptr_load, /*dest-align*/0, carr, /*src-align*/0, slen.GetValue());
#else
        B.CreateMemCpy(cptr_load, carr, slen.GetValue(), /*align*/0);
#endif
        return Value::Invalid();
    };

    static auto strinitc = [](codegen::String* str, const char* c) {
        const int64_t s = std::strlen(c);
        str->ptr = str->SSO;
        str->len = 0;
        str->alloc(s);
        std::memcpy(str->ptr, c, s);
    };

    static auto strinit = [](codegen::String* str) {
        strinitc(str, "");
    };

    return FunctionBuilder("string::string")
        .addSignature<void(codegen::String*), true>(generate, strinit)
        .addSignature<void(codegen::String*, const char*), true>(generate, strinitc)
        // dummy signature for initing an alloced string - use insertStaticAlloca instead
        //.addSignature<void(codegen::String*)>(generate)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .addDependency("strlen")
        .addDependency("string::alloc")
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axstringassign(const FunctionOptions& op)
{
    auto generate =
        [op](const NativeArguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        llvm::StructType* strType = LLVMType<codegen::String>::get(B.getContext());
        Value str0 = args[0];
        Value str1 = args[1];

        llvm::Value* cptr0 = B.CreateStructGEP(strType, str0.GetValue(), 0);
        llvm::Value* cptr1 = B.CreateStructGEP(strType, str1.GetValue(), 0);
        llvm::Value* len = B.CreateLoad(strType->getTypeAtIndex(2), B.CreateStructGEP(strType, str1.GetValue(), 2));

        Arguments strallocargs;
        strallocargs.AddArg(str0);
        strallocargs.AddArg(len, ArgInfo(strType->getTypeAtIndex(2)));
        axstringalloc(op)->execute(strallocargs, B);

        llvm::Value* cptr0_load = B.CreateLoad(strType->getTypeAtIndex(0u), cptr0);
        llvm::Value* cptr1_load = B.CreateLoad(strType->getTypeAtIndex(0u), cptr1);
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(cptr0_load, /*dest-align*/llvm::MaybeAlign(0),
            cptr1_load, /*src-align*/llvm::MaybeAlign(0), len);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(cptr0_load, /*dest-align*/0, cptr1_load, /*src-align*/0, len);
#else
        B.CreateMemCpy(cptr0_load, cptr1_load, len, /*align*/0);
#endif
        return Value::Invalid();
    };

    static auto strassign = [](codegen::String* a, const codegen::String* b) {
        *a = *b;
    };

    return FunctionBuilder("string::op=")
        .addSignature<void(codegen::String*, const codegen::String*)>(generate, strassign)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .addDependency("string::alloc")
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axstringadd(const FunctionOptions& op)
{
    auto generate =
        [op](const Arguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        llvm::StructType* strType = LLVMType<codegen::String>::get(B.getContext());
        llvm::Value* result = args[0];
        // don't need to init string as it will have been created with
        // insertStaticAlloca which makes sure that cptr=SSO and len=0
        // axstring(op)->execute(Arguments{result}, B);

        llvm::Value* str0 = args[1];
        llvm::Value* str1 = args[2];
        llvm::Value* len0 = B.CreateLoad(strType->getTypeAtIndex(2), B.CreateStructGEP(strType, str0, 2));
        llvm::Value* len1 = B.CreateLoad(strType->getTypeAtIndex(2), B.CreateStructGEP(strType, str1, 2));

        llvm::Value* total = B.CreateAdd(len0, len1);

        Arguments strallocargs;
        strallocargs.AddArg(result, ArgInfo(strType, 1));
        strallocargs.AddArg(total, ArgInfo(strType->getTypeAtIndex(2)));
        axstringalloc(op)->execute(strallocargs, B);

        llvm::Value* dst = B.CreateLoad(strType->getTypeAtIndex(0u), B.CreateStructGEP(strType, result, 0)); //char*
        llvm::Value* src0 = B.CreateLoad(strType->getTypeAtIndex(0u), B.CreateStructGEP(strType, str0, 0)); //char*
        llvm::Value* src1 = B.CreateLoad(strType->getTypeAtIndex(0u), B.CreateStructGEP(strType, str1, 0)); //char*

        // cpy first
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(dst, /*dest-align*/llvm::MaybeAlign(0),
            src0, /*src-align*/llvm::MaybeAlign(0), len0);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(dst, /*dest-align*/0, src0, /*src-align*/0, len0);
#else
        B.CreateMemCpy(dst, src0, len0, /*align*/0);
#endif

        // cpy second
        dst = B.CreateGEP(LLVMType<char>::get(B.getContext()), dst, len0);
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(dst, /*dest-align*/llvm::MaybeAlign(0),
            src1, /*src-align*/llvm::MaybeAlign(0), len1);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(dst, /*dest-align*/0, src1, /*src-align*/0, len1);
#else
        B.CreateMemCpy(dst, src1, len1, /*align*/0);
#endif
        return Value::Invalid();
    };

    static auto stradd = [](codegen::String* a, const codegen::String* b, const codegen::String* c) {
        *a = *b + *c;
    };

    return FunctionBuilder("string::op+")
        .addSignature<void(codegen::String*, const codegen::String*, const codegen::String*), true>(generate, stradd)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .addDependency("string::string")
        .addDependency("string::alloc")
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axstringclear(const FunctionOptions& op)
{
    auto generate =
        [op](const Arguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        Arguments strallocargs;
        strallocargs.AddArg(args[0], args.GetArgInfo(0));
        strallocargs.AddArg(Value::Create<int64_t>(B.getContext(), 0));
        axstringalloc(op)->execute(strallocargs, B);
        return Value::Invalid();
    };

    static auto strclear = [](codegen::String* a) { a->clear(); };

    return FunctionBuilder("string::clear")
        .addSignature<void(codegen::String*)>(generate, strclear)
        .setConstantFold(op.mConstantFoldCBindings)
        .setPreferredImpl(op.mPrioritiseIR ? FunctionBuilder::IR : FunctionBuilder::C)
        .addDependency("string::alloc")
        .setDocumentation("")
        .get();
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


void insertStringFunctions(FunctionRegistry& registry,
    const FunctionOptions* options)
{
    const bool create = options && !options->mLazyFunctions;
    auto add = [&](const std::string& name,
        const FunctionRegistry::ConstructorT creator,
        const bool internal = false)
    {
        if (create) registry.insertAndCreate(name, creator, *options, internal);
        else        registry.insert(name, creator, internal);
    };

    add("strlen", axstrlen, true);
    add("string::alloc", axstringalloc, true);
    add("string::string", axstring, true);
    add("string::op=", axstringassign, true);
    add("string::op+", axstringadd, true);
    add("string::clear", axstringclear, true);
}


} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

