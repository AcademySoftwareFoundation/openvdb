// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
        [](const std::vector<llvm::Value*>& args,
           llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() == 2);
        llvm::LLVMContext& C = B.getContext();
        llvm::Function* base = B.GetInsertBlock()->getParent();
        llvm::Type* strType = LLVMType<codegen::String>::get(C);

        llvm::Value* str = args[0];
        llvm::Value* size = args[1];
        llvm::Value* cptr = B.CreateStructGEP(strType, str, 0); // char**
        llvm::Value* sso = B.CreateStructGEP(strType, str, 1); // char[]*
        llvm::Value* sso_load = ir_constgep2_64(B, sso, 0 ,0); // char*

        llvm::Value* cptr_load = ir_load(B, cptr); // char*
        llvm::Value* neq = B.CreateICmpNE(cptr_load, sso_load);

        llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "then", base);
        llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "post", base);
        B.CreateCondBr(neq, then, post);
        B.SetInsertPoint(then);
        {
            llvm::BasicBlock* BB = B.GetInsertBlock();
            llvm::Instruction* inst = llvm::CallInst::CreateFree(cptr_load, BB);
            OPENVDB_ASSERT(inst);
            B.Insert(inst);
            B.CreateBr(post);
        }

        B.SetInsertPoint(post);

        llvm::Value* gt = B.CreateICmpSGT(size, B.getInt64(codegen::String::SSO_LENGTH-1));

        then = llvm::BasicBlock::Create(C, "then", base);
        llvm::BasicBlock* el = llvm::BasicBlock::Create(C, "else", base);
        post = llvm::BasicBlock::Create(C, "post", base);
        B.CreateCondBr(gt, then, el);
        B.SetInsertPoint(then);
        {
            llvm::BasicBlock* BB = B.GetInsertBlock();
            llvm::Instruction* inst =
                llvm::CallInst::CreateMalloc(BB, // location
                    B.getInt64Ty(), // int ptr type
                    B.getInt8Ty(),  // return type
                    B.CreateAdd(size, B.getInt64(1)), // size
                    nullptr,
                    nullptr);
            OPENVDB_ASSERT(inst);
            B.Insert(inst);
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
        cptr_load = ir_load(B, cptr); // char*
        llvm::Value* clast = ir_gep(B, cptr_load, size);
        B.CreateStore(B.getInt8(int8_t('\0')), clast); // this->ptr[size] = '\0';
        llvm::Value* len = B.CreateStructGEP(strType, str, 2);
        B.CreateStore(size, len);
        return nullptr;
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
        [op](const std::vector<llvm::Value*>& args,
           llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() >= 1);

        llvm::LLVMContext& C = B.getContext();
        llvm::Type* strType = LLVMType<codegen::String>::get(C);
        llvm::Value* str = args[0];

        llvm::Value* carr;
        if (args.size() == 1) carr = B.CreateGlobalStringPtr("");
        else                  carr = args[1];
        OPENVDB_ASSERT(carr);
        llvm::Value* slen = axstrlen(op)->execute({carr}, B);

        llvm::Value* cptr = B.CreateStructGEP(strType, str, 0); // char**
        llvm::Value* sso = B.CreateStructGEP(strType, str, 1); // char[]*
        llvm::Value* sso_load = ir_constgep2_64(B, sso, 0 ,0); // char*
        llvm::Value* len = B.CreateStructGEP(strType, str, 2);
        B.CreateStore(sso_load, cptr); // this->ptr = this->SSO;
        B.CreateStore(B.getInt64(0), len); // this->len = 0;

        axstringalloc(op)->execute({str, slen}, B);

        llvm::Value* cptr_load = ir_load(B, cptr);
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(cptr_load, /*dest-align*/llvm::MaybeAlign(0),
            carr, /*src-align*/llvm::MaybeAlign(0), slen);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(cptr_load, /*dest-align*/0, carr, /*src-align*/0, slen);
#else
        B.CreateMemCpy(cptr_load, carr, slen, /*align*/0);
#endif
        return nullptr;
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
        [op](const std::vector<llvm::Value*>& args,
           llvm::IRBuilder<>& B) -> llvm::Value*
    {
        OPENVDB_ASSERT(args.size() == 2);
        llvm::Type* strType = LLVMType<codegen::String>::get(B.getContext());
        llvm::Value* str0 = args[0];
        llvm::Value* str1 = args[1];

        llvm::Value* cptr0 = B.CreateStructGEP(strType, str0, 0);
        llvm::Value* cptr1 = B.CreateStructGEP(strType, str1, 0);
        llvm::Value* len = ir_load(B, B.CreateStructGEP(strType, str1, 2));

        axstringalloc(op)->execute({str0, len}, B);

        llvm::Value* cptr0_load = ir_load(B, cptr0);
        llvm::Value* cptr1_load = ir_load(B, cptr1);
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(cptr0_load, /*dest-align*/llvm::MaybeAlign(0),
            cptr1_load, /*src-align*/llvm::MaybeAlign(0), len);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(cptr0_load, /*dest-align*/0, cptr1_load, /*src-align*/0, len);
#else
        B.CreateMemCpy(cptr0_load, cptr1_load, len, /*align*/0);
#endif
        return nullptr;
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
        [op](const std::vector<llvm::Value*>& args,
           llvm::IRBuilder<>& B) -> llvm::Value*
    {
        llvm::Type* strType = LLVMType<codegen::String>::get(B.getContext());
        llvm::Value* result = args[0];
        // don't need to init string as it will have been created with
        // insertStaticAlloca which makes sure that cptr=SSO and len=0
        // axstring(op)->execute({result}, B);

        llvm::Value* str0 = args[1];
        llvm::Value* str1 = args[2];
        llvm::Value* len0 = ir_load(B, B.CreateStructGEP(strType, str0, 2));
        llvm::Value* len1 = ir_load(B, B.CreateStructGEP(strType, str1, 2));

        llvm::Value* total = B.CreateAdd(len0, len1);
        axstringalloc(op)->execute({result, total}, B);

        llvm::Value* dst = ir_load(B, B.CreateStructGEP(strType, result, 0)); //char*
        llvm::Value* src0 = ir_load(B, B.CreateStructGEP(strType, str0, 0)); //char*
        llvm::Value* src1 = ir_load(B, B.CreateStructGEP(strType, str1, 0)); //char*

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
        dst = ir_gep(B, dst, len0);
#if LLVM_VERSION_MAJOR >= 10
        B.CreateMemCpy(dst, /*dest-align*/llvm::MaybeAlign(0),
            src1, /*src-align*/llvm::MaybeAlign(0), len1);
#elif LLVM_VERSION_MAJOR > 6
        B.CreateMemCpy(dst, /*dest-align*/0, src1, /*src-align*/0, len1);
#else
        B.CreateMemCpy(dst, src1, len1, /*align*/0);
#endif
        return nullptr;
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
        [op](const std::vector<llvm::Value*>& args,
           llvm::IRBuilder<>& B) -> llvm::Value*
    {
        axstringalloc(op)->execute({args[0], B.getInt64(0)}, B);
        return nullptr;
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

