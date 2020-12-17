// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/VolumeComputeGenerator.cc

#include "VolumeComputeGenerator.h"
#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include "../Exceptions.h"
#include "../ast/Scanners.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

const std::array<std::string, VolumeKernel::N_ARGS>&
VolumeKernel::argumentKeys()
{
    static const std::array<std::string, VolumeKernel::N_ARGS> arguments = {{
        "custom_data",
        "coord_is",
        "coord_ws",
        "accessors",
        "transforms",
        "write_index",
        "write_acccessor"
    }};

    return arguments;
}

std::string VolumeKernel::getDefaultName() { return "ax.compute.voxel"; }


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

VolumeComputeGenerator::VolumeComputeGenerator(llvm::Module& module,
                                               const FunctionOptions& options,
                                               FunctionRegistry& functionRegistry,
                                               Logger& logger)
    : ComputeGenerator(module, options, functionRegistry, logger) {}

AttributeRegistry::Ptr VolumeComputeGenerator::generate(const ast::Tree& tree)
{
    llvm::FunctionType* type =
        llvmFunctionTypeFromSignature<VolumeKernel::Signature>(mContext);

    mFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        VolumeKernel::getDefaultName(),
        &mModule);

    // Set up arguments for initial entry

    llvm::Function::arg_iterator argIter = mFunction->arg_begin();
    const auto arguments = VolumeKernel::argumentKeys();
    auto keyIter = arguments.cbegin();

    for (; argIter != mFunction->arg_end(); ++argIter, ++keyIter) {
        argIter->setName(*keyIter);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mContext,
        "entry_" + VolumeKernel::getDefaultName(), mFunction);
    mBuilder.SetInsertPoint(entry);

    // build the attribute registry

    AttributeRegistry::Ptr registry = AttributeRegistry::create(tree);

    // Visit all attributes and allocate them in local IR memory - assumes attributes
    // have been verified by the ax compiler
    // @note  Call all attribute allocs at the start of this block so that llvm folds
    // them into the function prologue (as a static allocation)

    SymbolTable* localTable = this->mSymbolTables.getOrInsert(1);

    // run allocations and update the symbol table

    for (const AttributeRegistry::AccessData& data : registry->data()) {
        llvm::Value* value = mBuilder.CreateAlloca(llvmTypeFromToken(data.type(), mContext));
        assert(llvm::cast<llvm::AllocaInst>(value)->isStaticAlloca());
        localTable->insert(data.tokenname(), value);
    }

    // insert getters for read variables

    for (const AttributeRegistry::AccessData& data : registry->data()) {
        if (!data.reads()) continue;
        const std::string token = data.tokenname();
        this->getAccessorValue(token, localTable->get(token));
    }

    // full code generation
    // errors can stop traversal, but dont always, so check the log

    if (!this->traverse(&tree) || mLog.hasError()) return nullptr;

    // insert set code

    std::vector<const AttributeRegistry::AccessData*> write;
    for (const AttributeRegistry::AccessData& access : registry->data()) {
        if (access.writes()) write.emplace_back(&access);
    }
    if (write.empty()) return registry;

    // Cache the basic blocks which have been created as we will create
    // new branches below
    std::vector<llvm::BasicBlock*> blocks;
    for (auto block = mFunction->begin(); block != mFunction->end(); ++block) {
        blocks.emplace_back(&*block);
    }

    // insert set voxel calls

    for (auto& block : blocks) {

        // Only inset set calls if theres a valid return instruction in this block

        llvm::Instruction* inst = block->getTerminator();
        if (!inst || !llvm::isa<llvm::ReturnInst>(inst)) continue;
        // remove the old return statement (we'll point to the final return jump)
        inst->eraseFromParent();

        // Set builder to the end of this block

        mBuilder.SetInsertPoint(&(*block));

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

            llvm::Value* coordis = extractArgument(mFunction, "coord_is");
            llvm::Value* accessIndex = extractArgument(mFunction, "write_index");
            llvm::Value* accessor = extractArgument(mFunction, "write_acccessor");
            assert(coordis);
            assert(accessor);
            assert(accessIndex);

            llvm::Value* registeredIndex = llvm::cast<llvm::GlobalVariable>
                (mModule.getOrInsertGlobal(token, LLVMType<int64_t>::get(mContext)));
            registeredIndex = mBuilder.CreateLoad(registeredIndex);

            llvm::Value* result = mBuilder.CreateICmpEQ(accessIndex, registeredIndex);
            result = boolComparison(result, mBuilder);

            llvm::BasicBlock* thenBlock =
                llvm::BasicBlock::Create(mContext, "post_assign " + token, mFunction);
            llvm::BasicBlock* continueBlock =
                llvm::BasicBlock::Create(mContext, "post_continue", mFunction);

            mBuilder.CreateCondBr(result, thenBlock, continueBlock);
            mBuilder.SetInsertPoint(thenBlock);

            llvm::Type* type = value->getType()->getPointerElementType();

            // load the result (if its a scalar)
            if (type->isIntegerTy() || type->isFloatingPointTy()) {
                value = mBuilder.CreateLoad(value);
            }

            const FunctionGroup* const function = this->getFunction("setvoxel", true);
            function->execute({accessor, coordis, value}, mBuilder);

            mBuilder.CreateBr(continueBlock);
            mBuilder.SetInsertPoint(continueBlock);
        }

        mBuilder.CreateRetVoid();
    }

    return registry;
}

bool VolumeComputeGenerator::visit(const ast::Attribute* node)
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


void VolumeComputeGenerator::getAccessorValue(const std::string& globalName, llvm::Value* location)
{
    std::string name, type;
    ast::Attribute::nametypeFromToken(globalName, &name, &type);

    llvm::Value* registeredIndex = llvm::cast<llvm::GlobalVariable>
        (mModule.getOrInsertGlobal(globalName, LLVMType<int64_t>::get(mContext)));
    this->globals().insert(globalName, registeredIndex);
    registeredIndex = mBuilder.CreateLoad(registeredIndex);

    // index into the void* array of handles and load the value.
    // The result is a loaded void* value

    llvm::Value* accessorPtr = extractArgument(mFunction, "accessors");
    llvm::Value* transformPtr = extractArgument(mFunction, "transforms");
    llvm::Value* coordws = extractArgument(mFunction, "coord_ws");
    assert(accessorPtr);
    assert(transformPtr);
    assert(coordws);

    accessorPtr = mBuilder.CreateGEP(accessorPtr, registeredIndex);
    transformPtr = mBuilder.CreateGEP(transformPtr, registeredIndex);

    llvm::Value* accessor = mBuilder.CreateLoad(accessorPtr);
    llvm::Value* transform = mBuilder.CreateLoad(transformPtr);

    const FunctionGroup* const function = this->getFunction("getvoxel", true);
    function->execute({accessor, transform, coordws, location}, mBuilder);
}

llvm::Value* VolumeComputeGenerator::accessorHandleFromToken(const std::string& globalName)
{
    // Visiting an "attribute" - get the volume accessor out of a vector of void pointers
    // mAttributeHandles is a void pointer to a vector of void pointers (void**)

    llvm::Value* registeredIndex = llvm::cast<llvm::GlobalVariable>
        (mModule.getOrInsertGlobal(globalName, LLVMType<int64_t>::get(mContext)));
    this->globals().insert(globalName, registeredIndex);

    registeredIndex = mBuilder.CreateLoad(registeredIndex);

    // index into the void* array of handles and load the value.
    // The result is a loaded void* value

    llvm::Value* accessorPtr = extractArgument(mFunction, "accessors");
    assert(accessorPtr);
    accessorPtr = mBuilder.CreateGEP(accessorPtr, registeredIndex);

    // return loaded void** = void*
    return mBuilder.CreateLoad(accessorPtr);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

