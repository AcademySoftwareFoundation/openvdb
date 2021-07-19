// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED

#include <openvdb_ax/codegen/Types.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Support/Host.h>

#include <memory>
#include <string>

namespace unittest_util
{

struct LLVMState
{
    LLVMState(const std::string& name = "__test_module")
        : mCtx(new llvm::LLVMContext), mModule(new llvm::Module(name, *mCtx)) {}

    llvm::LLVMContext& context() { return *mCtx; }
    llvm::Module& module() { return *mModule; }

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

    inline std::unique_ptr<llvm::ExecutionEngine> EE()
    {
        mModule->setTargetTriple(llvm::sys::getDefaultTargetTriple());
        llvm::StringMap<bool> HostFeatures;
        llvm::sys::getHostCPUFeatures(HostFeatures);
        std::vector<llvm::StringRef> features;
        for (auto& feature : HostFeatures) {
            if (feature.second) features.emplace_back(feature.first());
        }

        auto M = std::move(mModule);
        mModule.reset(new llvm::Module(M->getName(), *mCtx));
        std::unique_ptr<llvm::ExecutionEngine>
            EE(llvm::EngineBuilder(std::move(M))
                .setEngineKind(llvm::EngineKind::JIT)
                .setOptLevel(llvm::CodeGenOpt::Level::Default)
                .setMCPU(llvm::sys::getHostCPUName())
                .setMAttrs(features)
                .create());

        assert(EE.get());
        return EE;
    }

private:
    std::unique_ptr<llvm::LLVMContext> mCtx;
    std::unique_ptr<llvm::Module> mModule;
};

}

#endif // OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED

