// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/PointComputeGenerator.cc

#include "PointComputeGenerator.h"

#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include "../Exceptions.h"
#include "../ast/Scanners.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Pass.h>
#include <llvm/Support/MathExtras.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {


const std::array<std::string, PointKernel::N_ARGS>&
PointKernel::argumentKeys()
{
    static const std::array<std::string, PointKernel::N_ARGS> arguments = {{
        "custom_data",
        "attribute_set",
        "point_index",
        "attribute_handles",
        "group_handles",
        "leaf_data"
    }};

    return arguments;
}

std::string PointKernel::getDefaultName() { return "ax.compute.point"; }

std::string PointRangeKernel::getDefaultName() { return "ax.compute.pointrange"; }


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

PointComputeGenerator::PointComputeGenerator(llvm::Module& module,
                                             const FunctionOptions& options,
                                             FunctionRegistry& functionRegistry,
                                             Logger& logger)
    : ComputeGenerator(module, options, functionRegistry, logger) {}

AttributeRegistry::Ptr PointComputeGenerator::generate(const ast::Tree& tree)
{
    llvm::FunctionType* type =
        llvmFunctionTypeFromSignature<PointKernel::Signature>(mContext);

    mFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        PointKernel::getDefaultName(),
        &mModule);

    // Set up arguments for initial entry

    llvm::Function::arg_iterator argIter = mFunction->arg_begin();
    const auto arguments = PointKernel::argumentKeys();
    auto keyIter = arguments.cbegin();

    for (; argIter != mFunction->arg_end(); ++argIter, ++keyIter) {
        argIter->setName(*keyIter);
    }

    type = llvmFunctionTypeFromSignature<PointRangeKernel::Signature>(mContext);

    llvm::Function* rangeFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        PointRangeKernel::getDefaultName(),
        &mModule);

    // Set up arguments for initial entry for the range function

    std::vector<llvm::Value*> kPointRangeArguments;
    argIter = rangeFunction->arg_begin();
    for (; argIter != rangeFunction->arg_end(); ++argIter) {
        kPointRangeArguments.emplace_back(llvm::cast<llvm::Value>(argIter));
    }

    {
        // Generate the range function which calls mFunction point_count times

        // For the pointRangeKernelSignature function, create a for loop which calls
        // kPoint for every point index 0 to mPointCount. The argument types for
        // pointRangeKernelSignature and kPoint are the same, but the 'point_index' argument for
        // kPoint is the point index rather than the point range

        auto iter = std::find(arguments.begin(), arguments.end(), "point_index");
        assert(iter != arguments.end());
        const size_t argumentIndex = std::distance(arguments.begin(), iter);

        llvm::BasicBlock* preLoop = llvm::BasicBlock::Create(mContext,
            "entry_" + PointRangeKernel::getDefaultName(), rangeFunction);
        mBuilder.SetInsertPoint(preLoop);

        llvm::Value* pointCountValue = kPointRangeArguments[argumentIndex];
        llvm::Value* indexMinusOne = mBuilder.CreateSub(pointCountValue, mBuilder.getInt64(1));

        llvm::BasicBlock* loop =
            llvm::BasicBlock::Create(mContext, "loop_compute_point", rangeFunction);
        mBuilder.CreateBr(loop);
        mBuilder.SetInsertPoint(loop);

        llvm::PHINode* incr = mBuilder.CreatePHI(mBuilder.getInt64Ty(), 2, "i");
        incr->addIncoming(/*start*/mBuilder.getInt64(0), preLoop);

        // Call kPoint with incr which will be updated per branch

        // Map the function arguments. For the 'point_index' argument, we don't pull in the provided
        // args, but instead use the value of incr. incr will correspond to the index of the
        // point being accessed within the pointRangeKernelSignature loop.

        std::vector<llvm::Value*> args(kPointRangeArguments);
        args[argumentIndex] = incr;
        mBuilder.CreateCall(mFunction, args);

        llvm::Value* next = mBuilder.CreateAdd(incr, mBuilder.getInt64(1), "nextval");
        llvm::Value* endCondition = mBuilder.CreateICmpULT(incr, indexMinusOne, "endcond");
        llvm::BasicBlock* loopEnd = mBuilder.GetInsertBlock();

        llvm::BasicBlock* postLoop =
            llvm::BasicBlock::Create(mContext, "post_loop_compute_point", rangeFunction);
        mBuilder.CreateCondBr(endCondition, loop, postLoop);
        mBuilder.SetInsertPoint(postLoop);
        incr->addIncoming(next, loopEnd);

        mBuilder.CreateRetVoid();
        mBuilder.ClearInsertionPoint();
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mContext,
        "entry_" + PointKernel::getDefaultName(), mFunction);
    mBuilder.SetInsertPoint(entry);

    // build the attribute registry

    AttributeRegistry::Ptr registry = AttributeRegistry::create(tree);

    // Visit all attributes and allocate them in local IR memory - assumes attributes
    // have been verified by the ax compiler
    // @note  Call all attribute allocs at the start of this block so that llvm folds
    // them into the function prologue (as a static allocation)

    SymbolTable* localTable = this->mSymbolTables.getOrInsert(1);

    // run allocations and update the symbol table

    for (const AttributeRegistry::AccessData& access : registry->data()) {
        llvm::Value* value = insertStaticAlloca(mBuilder, llvmTypeFromToken(access.type(), mContext));
        assert(llvm::cast<llvm::AllocaInst>(value)->isStaticAlloca());
        localTable->insert(access.tokenname(), value);
    }

    // insert getters for read variables

    for (const AttributeRegistry::AccessData& access : registry->data()) {
        if (!access.reads()) continue;
        const std::string token = access.tokenname();
        this->getAttributeValue(token, localTable->get(token));
    }

    // full code generation
    // errors can stop traversal, but dont always, so check the log

    if (!this->traverse(&tree) || mLog.hasError()) return nullptr;

    // insert set code and deallocations

    std::vector<const AttributeRegistry::AccessData*> write;
    for (const AttributeRegistry::AccessData& access : registry->data()) {
        if (access.writes()) write.emplace_back(&access);
    }

    for (auto block = mFunction->begin(); block != mFunction->end(); ++block) {

        // Only inset set calls if theres a valid return instruction in this block

        llvm::Instruction* inst = block->getTerminator();
        if (!inst || !llvm::isa<llvm::ReturnInst>(inst)) continue;
        mBuilder.SetInsertPoint(inst);

        // Insert set attribute instructions before termination

        for (const AttributeRegistry::AccessData* access : write) {

            const std::string token = access->tokenname();
            llvm::Value* value = localTable->get(token);

            // Expected to be used more than one (i.e. should never be zero)
            assert(value->hasNUsesOrMore(1));

            // Check to see if this value is still being used - it may have
            // been cleaned up due to returns. If there's only one use, it's
            // the original get of this attribute.
            if (value->hasOneUse()) {
                // @todo  The original get can also be optimized out in this case
                // this->globals().remove(variable.first);
                // mModule.getGlobalVariable(variable.first)->eraseFromParent();
                continue;
            }

            llvm::Type* type = value->getType()->getPointerElementType();
            llvm::Type* strType = LLVMType<codegen::String>::get(mContext);
            const bool usingString = type == strType;

            llvm::Value* handlePtr = this->attributeHandleFromToken(token);
            const FunctionGroup* const function = this->getFunction("setattribute", true);

            // load the result (if its a scalar)
            if (type->isIntegerTy() || type->isFloatingPointTy()) {
                value = mBuilder.CreateLoad(value);
            }

            llvm::Value* pointidx = extractArgument(mFunction, "point_index");
            assert(pointidx);

            // construct function arguments
            std::vector<llvm::Value*> args {
                handlePtr, // handle
                pointidx, // point index
                value // set value
            };

            if (usingString) {
                llvm::Value* leafdata = extractArgument(mFunction, "leaf_data");
                assert(leafdata);
                args.emplace_back(leafdata);
            }

            function->execute(args, mBuilder);
        }
    }

    this->createFreeSymbolStrings(mBuilder);

    return registry;
}

bool PointComputeGenerator::visit(const ast::Attribute* node)
{
    const std::string globalName = node->tokenname();
    SymbolTable* localTable = this->mSymbolTables.getOrInsert(1);
    llvm::Value* value = localTable->get(globalName);
    assert(value);
    mValues.push(value);
    return true;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

void PointComputeGenerator::getAttributeValue(const std::string& globalName, llvm::Value* location)
{
    std::string name, type;
    ast::Attribute::nametypeFromToken(globalName, &name, &type);

    llvm::Value* handlePtr = this->attributeHandleFromToken(globalName);
    llvm::Value* pointidx = extractArgument(mFunction, "point_index");
    llvm::Value* leafdata = extractArgument(mFunction, "leaf_data");
    assert(leafdata);
    assert(pointidx);

    std::vector<llvm::Value*> args;

    const bool usingString = type == "string";

    if (usingString) {
        args.reserve(4);
    }
    else {
        args.reserve(3);
    }

    args.emplace_back(handlePtr);
    args.emplace_back(pointidx);
    args.emplace_back(location);

    if (usingString) args.emplace_back(leafdata);

    const FunctionGroup* const function = this->getFunction("getattribute", true);
    function->execute(args, mBuilder);
}

llvm::Value* PointComputeGenerator::attributeHandleFromToken(const std::string& token)
{
    // Visiting an attribute - get the attribute handle out of a vector of void pointers

    // insert the attribute into the map of global variables and get a unique global representing
    // the location which will hold the attribute handle offset.

    llvm::Value* index = llvm::cast<llvm::GlobalVariable>
        (mModule.getOrInsertGlobal(token, LLVMType<int64_t>::get(mContext)));
    this->globals().insert(token, index);

    // index into the void* array of handles and load the value.
    // The result is a loaded void* value

    index = mBuilder.CreateLoad(index);

    llvm::Value* handles = extractArgument(mFunction, "attribute_handles");
    assert(handles);
    llvm::Value* handlePtr = mBuilder.CreateGEP(handles, index);

    // return loaded void** = void*
    return mBuilder.CreateLoad(handlePtr);
}

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

