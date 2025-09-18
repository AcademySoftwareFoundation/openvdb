// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED

#include <openvdb_ax/codegen/Types.h>
#include <openvdb_ax/codegen/FunctionTypes.h>

#include <openvdb/util/Assert.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#if LLVM_VERSION_MAJOR <= 15
#include <llvm/Support/Host.h>
#elif LLVM_VERSION_MAJOR < 17
// for DynamicLibrarySearchGenerator
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#endif

#include <memory>
#include <string>

namespace unittest_util
{

/// @brief  Dummy derived function which implemented types

#if LLVM_VERSION_MAJOR <= 15
struct TestFunction : public openvdb::ax::codegen::Function
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::Function>::value,
        "Base class destructor is not virtual");
    TestFunction(const openvdb::ax::codegen::ArgInfoVector& types,
          openvdb::ax::codegen::ArgInfo ret,
          const std::string& symbol)
        : openvdb::ax::codegen::Function(types.size(), symbol)
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
    ~TestFunction() override = default;

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
struct TestFunction final : public openvdb::ax::codegen::Function
{
    static_assert(std::has_virtual_destructor
        <openvdb::ax::codegen::Function>::value,
        "Base class destructor is not virtual");
    TestFunction(const openvdb::ax::codegen::ArgInfoVector& types,
          openvdb::ax::codegen::ArgInfo ret,
          const std::string& symbol)
        : openvdb::ax::codegen::Function(types.size(), symbol)
        , mTypes(types)
        , mRet([&]() {
            ret.SetIsReturn();
            return ret;
        }()) {}
    ~TestFunction() override = default;
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

#if LLVM_VERSION_MAJOR <= 15
struct LLVMState
{
    LLVMState(const std::string& name = "__test_module")
        : mCtx(new llvm::LLVMContext), mModule(new llvm::Module(name, *mCtx)) {
#if LLVM_VERSION_MAJOR == 15
    // This will not work from LLVM 16
    // https://llvm.org/docs/OpaquePointers.html
    mCtx->setOpaquePointers(false);
#endif
        }

    llvm::LLVMContext& context() {  OPENVDB_ASSERT(mModule); return *mCtx; }
    llvm::Module& module() {  OPENVDB_ASSERT(mModule); return *mModule; }

    inline llvm::BasicBlock*
    scratchBlock(const std::string& functionName = "TestFunction",
                 const std::string& blockName = "TestEntry")
    {
        llvm::FunctionType* type =
            llvm::FunctionType::get(openvdb::ax::codegen::LLVMType<void>::get(this->context()),
                /**var-args*/false);
        llvm::Function* dummyFunction =
            llvm::Function::Create(type, llvm::Function::ExternalLinkage, functionName, &this->module());
        return llvm::BasicBlock::Create(this->context(), blockName, dummyFunction);
    }

    inline void CreateEE()
    {
        OPENVDB_ASSERT(mModule);
        mModule->setTargetTriple(llvm::sys::getDefaultTargetTriple());
        llvm::StringMap<bool> HostFeatures;
        llvm::sys::getHostCPUFeatures(HostFeatures);
        std::vector<llvm::StringRef> features;
        for (auto& feature : HostFeatures) {
            if (feature.second) features.emplace_back(feature.first());
        }

        std::unique_ptr<llvm::ExecutionEngine>
            EE(llvm::EngineBuilder(std::move(mModule))
                .setEngineKind(llvm::EngineKind::JIT)
                .setOptLevel(llvm::CodeGenOpt::Level::Default)
                .setMCPU(llvm::sys::getHostCPUName())
                .setMAttrs(features)
                .create());

        mModule.reset();
        OPENVDB_ASSERT(EE.get());
        mEE = std::move(EE);
    }

    uint64_t GetGlobalAddress(const std::string& name)
    {
        OPENVDB_ASSERT(!mModule);
        OPENVDB_ASSERT(mEE);
        return mEE->getFunctionAddress(name);
    }


private:
    std::unique_ptr<llvm::LLVMContext> mCtx;
    std::unique_ptr<llvm::Module> mModule;
    std::unique_ptr<llvm::ExecutionEngine> mEE;
};

#else

struct LLVMState
{
    LLVMState(const std::string& name = "__test_module")
        : mModule(), mEE(nullptr)
    {
        llvm::orc::ThreadSafeContext ctx(std::make_unique<llvm::LLVMContext>());
        auto module = std::make_unique<llvm::Module>(name, *(ctx.getContext()));
        mModule = llvm::orc::ThreadSafeModule(std::move(module), std::move(ctx));
    }

    llvm::LLVMContext& context() { OPENVDB_ASSERT(mModule); return *(mModule.getContext().getContext()); }
    llvm::Module& module() { OPENVDB_ASSERT(mModule); return *mModule.getModuleUnlocked(); }

    inline llvm::BasicBlock*
    scratchBlock(const std::string& functionName = "TestFunction",
                 const std::string& blockName = "TestEntry")
    {
        llvm::FunctionType* type =
            llvm::FunctionType::get(openvdb::ax::codegen::LLVMType<void>::get(this->context()),
                /**var-args*/false);
        llvm::Function* dummyFunction =
            llvm::Function::Create(type, llvm::Function::ExternalLinkage, functionName, &this->module());
        return llvm::BasicBlock::Create(this->context(), blockName, dummyFunction);
    }

    inline void CreateEE()
    {
        OPENVDB_ASSERT(mModule);
        auto EE = llvm::orc::LLJITBuilder().create();
        if (!EE) llvm::report_fatal_error(EE.takeError());

#if LLVM_VERSION_MAJOR < 17
        // LLVM 17 and 18 introduced several improvements to the ORC JIT
        // infrastructure, including:
        //
        //   - Better default symbol resolution: LLVM 17+ improved how it
        //     automatically resolves symbols from the host process.
        //   - Enhanced support for DynamicLibrarySearchGenerator: This makes it
        //     easier to import symbols from the host's dynamic libraries (like
        //     libc).
        //   - More robust LLJITBuilder defaults: Later versions configure the JIT
        //     environment more intelligently, reducing the need for manual symbol
        //     injection.
        //
        // Before this, we need to add a host generator ourselves for things like
        // malloc, free, memcpy
        {
            auto Generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                (*EE)->getDataLayout().getGlobalPrefix());
            if (Generator) {
                (*EE)->getMainJITDylib().addGenerator(std::move(*Generator));
            }
        }
#endif

        if (auto Err = (*EE)->addIRModule(std::move(mModule))) {
            llvm::report_fatal_error(std::move(Err));
        }
        mModule = llvm::orc::ThreadSafeModule(); // empty
        mEE = std::move(*EE);
        //mEE->getMainJITDylib().dump(llvm::errs());
    }

    uint64_t GetGlobalAddress(const std::string& name)
    {
        OPENVDB_ASSERT(!mModule);
        OPENVDB_ASSERT(mEE);
        auto EntrySym = mEE->lookup(name);
        if (!bool(EntrySym)) return 0;
        return EntrySym->getValue();
    }

private:
    llvm::orc::ThreadSafeModule mModule;
    std::unique_ptr<llvm::orc::LLJIT> mEE;
};

#endif


}

#endif // OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED

