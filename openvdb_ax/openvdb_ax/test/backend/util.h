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

#include <memory>
#include <string>

namespace unittest_util
{

/// @brief  Dummy derived function which implemented types

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

struct LLVMState
{
    LLVMState(const std::string& name = "__test_module")
        : mCtx(std::make_unique<llvm::LLVMContext>())
        , mModule(std::make_unique<llvm::Module>(name, *mCtx))
        , mEE(nullptr) {}

    llvm::LLVMContext& context() { OPENVDB_ASSERT(mCtx); return *mCtx; }
    llvm::Module& module() { OPENVDB_ASSERT(mModule); return *mModule; }

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
        OPENVDB_ASSERT(mCtx);
        OPENVDB_ASSERT(mModule);
        llvm::orc::ThreadSafeModule module(std::move(mModule), std::move(mCtx));

        auto EE = llvm::orc::LLJITBuilder().create();
        if (!EE) llvm::report_fatal_error(EE.takeError());

        if (auto Err = (*EE)->addIRModule(std::move(module))) {
            llvm::report_fatal_error(std::move(Err));
        }
        module = llvm::orc::ThreadSafeModule(); // empty
        mEE = std::move(*EE);
        //mEE->getMainJITDylib().dump(llvm::errs());
    }

    uint64_t GetGlobalAddress(const std::string& name)
    {
        OPENVDB_ASSERT(!mCtx);
        OPENVDB_ASSERT(!mModule);
        OPENVDB_ASSERT(mEE);
        auto EntrySym = mEE->lookup(name);
        if (!bool(EntrySym)) return 0;
        return EntrySym->getValue();
    }

private:
    std::unique_ptr<llvm::LLVMContext> mCtx;
    std::unique_ptr<llvm::Module> mModule;
    std::unique_ptr<llvm::orc::LLJIT> mEE;
};


}

#endif // OPENVDB_AX_UNITTEST_BACKEND_UTIL_HAS_BEEN_INCLUDED

