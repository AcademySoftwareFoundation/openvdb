// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/VolumeComputeGenerator.cc

#include "VolumeComputeGenerator.h"
#include "VolumeKernelFunctions.h"
#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include <openvdb/util/Assert.h>

#include "../Exceptions.h"
#include "../ast/Scanners.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {
namespace codegen_internal {

inline void VolumeComputeGenerator::computek2(llvm::Function* compute, const AttributeRegistry&)
{
    auto generate =
        [&](const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B) -> llvm::Value*
    {
        using VBuffPtrT = VolumeKernelBuffer::FunctionTraitsT::Arg<2>::Type; // leaf value buffer argument ptr type
        using ABuffPtrT = VolumeKernelBuffer::FunctionTraitsT::Arg<3>::Type; // active buffer type (word masks) argument ptr type

        llvm::LLVMContext& C = B.getContext();

        [[maybe_unused]] llvm::Type* vbuffT = LLVMType<std::remove_pointer_t<VBuffPtrT>>::get(C);
        llvm::Type* abuffT = LLVMType<std::remove_pointer_t<ABuffPtrT>>::get(C);
        llvm::Type* incT   = LLVMType<int64_t>::get(C); // type of the loop counter
        OPENVDB_ASSERT(vbuffT == B.getVoidTy()); // sanity check
        OPENVDB_ASSERT(abuffT == B.getInt64Ty()); // sanity check
        vbuffT = B.getInt8Ty(); // points to bytes

        OPENVDB_ASSERT(args.size() == 9);
        llvm::Value* vbuff = args[2]; //extractArgument(rangeFunction, "value_buffer");
        llvm::Value* abuff = args[3]; //extractArgument(rangeFunction, "active_buffer");
        llvm::Value* buffSize = args[4]; //extractArgument(rangeFunction, "buffer_size");
        llvm::Value* mode = args[5]; //extractArgument(rangeFunction, "mode");
        OPENVDB_ASSERT(buffSize);
        OPENVDB_ASSERT(vbuff);
        OPENVDB_ASSERT(abuff);
        OPENVDB_ASSERT(mode);

        OPENVDB_ASSERT(AssertOpaquePtrs(vbuff, vbuffT));
        OPENVDB_ASSERT(AssertOpaquePtrs(abuff, abuffT));

        llvm::Function* base = B.GetInsertBlock()->getParent();

        llvm::BasicBlock* conditionBlock = llvm::BasicBlock::Create(C, "k2.condition", base);
        llvm::BasicBlock* bodyBlock = llvm::BasicBlock::Create(C, "k2.body", base);
        llvm::BasicBlock* iterBlock = llvm::BasicBlock::Create(C, "k2.iter", base);

        // init var - loops from 0 -> buffSize
        llvm::Value* incr = insertStaticAlloca(B, incT);
        B.CreateStore(B.getInt64(0), incr);
        B.CreateBr(conditionBlock);

        // increment
        B.SetInsertPoint(iterBlock);
        llvm::Value* new_incr = B.CreateAdd(B.CreateLoad(incT, incr), B.getInt64(1));
        B.CreateStore(new_incr, incr);
        B.CreateBr(conditionBlock);

        // generate loop body
        B.SetInsertPoint(bodyBlock);
        llvm::Value* lincr = B.CreateLoad(incT, incr);

        // Extract mask bit from array of words
        // NodeMask::isOn() = (0 != (mWords[n >> 6] & (Word(1) << (n & 63))));
        llvm::Value* mask = binaryOperator(B.getInt64(1),
            binaryOperator(lincr, B.getInt64(63), ast::tokens::BITAND, B),
                ast::tokens::SHIFTLEFT, B);
        llvm::Value* word_idx = binaryOperator(lincr, B.getInt64(6), ast::tokens::SHIFTRIGHT, B);
        llvm::Value* word = B.CreateGEP(abuffT, abuff, word_idx);
        word = B.CreateLoad(abuffT, word);
        word = binaryOperator(word, mask, ast::tokens::BITAND, B);
        llvm::Value* ison = B.CreateICmpNE(word, B.getInt64(0));

        // Check if we should run the kernel depending on the mode.
        //   mode == 0, inactive values
        //   mode == 1, active values
        //   mode == 2, all values
        llvm::Value* matches_mode = B.CreateICmpEQ(B.CreateZExt(ison, mode->getType()), mode);
        llvm::Value* mode_is_all = B.CreateICmpEQ(mode, B.getInt64(2));
        llvm::Value* process = binaryOperator(matches_mode, mode_is_all, ast::tokens::OR, B);
        llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "k2.invoke_k1", base);

        B.CreateCondBr(process, then, iterBlock);
        B.SetInsertPoint(then);
        {
            // invoke the volume kernel for this value
            const std::array<llvm::Value*, 8> input {
                args[0],            // ax::CustomData
                args[1],            // index space coordinate
                vbuff,              // value buffer
                ison,               // active/inactive
                B.CreateLoad(incT, incr), // offset in the value buffer
                args[6],            // read accessors
                args[7],            // transforms
                args[8]             // write index
            };
            B.CreateCall(compute, input);
            B.CreateBr(iterBlock);
        }

        B.SetInsertPoint(conditionBlock);
        llvm::Value* endCondition = B.CreateICmpULT(B.CreateLoad(incT, incr), buffSize);

        llvm::BasicBlock* postBlock = llvm::BasicBlock::Create(C, "k2.end", base);
        B.CreateCondBr(endCondition, bodyBlock, postBlock);
        B.SetInsertPoint(postBlock);
        return B.CreateRetVoid();
    };

    // Use the function builder to generate the correct prototype and body for K2
    auto k2 = FunctionBuilder(VolumeKernelBuffer::getDefaultName())
        .addSignature<VolumeKernelBuffer::Signature>(generate, VolumeKernelBuffer::getDefaultName())
        .setConstantFold(false)
        .setEmbedIR(false)
        .setExternalKernel()
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .get();

    k2->list()[0]->create(mContext, &mModule);
}

inline void VolumeComputeGenerator::computek3(llvm::Function* compute, const AttributeRegistry& reg)
{
    const SymbolTable<Value>& localTable = *(this->mSymbolTables.get(1));

    auto generate =
        [&, this](const Arguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 6);
        llvm::Value* isc = args[1]; // index space coord
        llvm::Value* wi = args[4]; // write index
        llvm::Value* wa = args[5]; // write_accessor

        llvm::Function* base = B.GetInsertBlock()->getParent();
        llvm::LLVMContext& C = B.getContext();

        for (const AttributeRegistry::AccessData& access : reg.data()) {
            if (!access.writes()) continue;

            const std::string token = access.tokenname();
            const Value& v = *localTable.get(token);

            llvm::Value* registeredIndex = this->mModule.getGlobalVariable(token);
            OPENVDB_ASSERT(registeredIndex);
            registeredIndex = B.CreateLoad(B.getInt64Ty(), registeredIndex);
            llvm::Value* result = B.CreateICmpEQ(wi, registeredIndex);

            llvm::BasicBlock* thenBlock = llvm::BasicBlock::Create(C, "k3.invoke_k1_" + token, base);
            llvm::BasicBlock* continueBlock = llvm::BasicBlock::Create(C, "k3.next", base);

            B.CreateCondBr(result, thenBlock, continueBlock);
            B.SetInsertPoint(thenBlock);

            llvm::Value* location = insertStaticAlloca(B, v.GetUnderlyingType());
            llvm::Value* ison = insertStaticAlloca(B, B.getInt1Ty());

            Arguments probeargs;
            probeargs.AddArg(wa,       ArgInfo(LLVMType<int8_t>::get(C), 1));
            probeargs.AddArg(isc,      ArgInfo(LLVMType<openvdb::math::Vec3<int32_t>>::get(C), 1));
            probeargs.AddArg(ison,     ArgInfo(B.getInt1Ty(), 1));
            probeargs.AddArg(location, ArgInfo(v.GetUnderlyingType(), 1));

            const FunctionGroup* const F = this->getFunction("probevalue", true);
            OPENVDB_ASSERT(F);
            F->execute(probeargs, B);
            ison = B.CreateLoad(B.getInt1Ty(), ison);

            llvm::Value* vptr = B.CreatePointerCast(location, LLVMType<void*>::get(C));

            const std::array<llvm::Value*, 8> input {
                args[0],        // ax::CustomData
                args[1],        // index space coordinate
                vptr,           // value buffer (in this case, a pointer to a single value)
                ison,           // active/inactive
                B.getInt64(0),  // offset in the value buffer, always zero
                args[2],        // read accessors
                args[3],        // transforms
                wi              // write index
            };
            B.CreateCall(compute, input);

            // set the voxel - load the result (if its a scalar)
            if (v.IsScalar()) {
                location = B.CreateLoad(v.GetUnderlyingType(), location);
            }

            Arguments setvargs;
            setvargs.AddArg(wa,  ArgInfo(LLVMType<int8_t>::get(C), 1));
            setvargs.AddArg(isc, ArgInfo(LLVMType<openvdb::math::Vec3<int32_t>>::get(C), 1));
            setvargs.AddArg(B.getInt32(-1), ArgInfo(B.getInt32Ty())); /*level=unknown*/
            setvargs.AddArg(ison,           ArgInfo(B.getInt1Ty()));
            setvargs.AddArg(location,       ArgInfo(v.GetUnderlyingType(), v.IsScalar() ? 0 : 1));

            const FunctionGroup* const function = this->getFunction("setvoxel", true);
            OPENVDB_ASSERT(function);
            function->execute(setvargs, B);

            B.CreateBr(continueBlock);
            B.SetInsertPoint(continueBlock);
        }

        Value ret = Value::Return(B);
        // insert string frees for k3 which can allocate them
        this->createFreeSymbolStrings(B);
        return ret;
    };

    auto k3 = FunctionBuilder(VolumeKernelNode::getDefaultName())
        .addSignature<VolumeKernelNode::Signature>(generate, VolumeKernelNode::getDefaultName())
        .setConstantFold(false)
        .setEmbedIR(false)
        .setExternalKernel()
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .get();

    k3->list()[0]->create(mContext, &mModule);
}

VolumeComputeGenerator::VolumeComputeGenerator(llvm::Module& module,
                                               const FunctionOptions& options,
                                               FunctionRegistry& functionRegistry,
                                               Logger& logger)
    : ComputeGenerator(module, options, functionRegistry, logger)
    , mInternalPtrs() {}

AttributeRegistry::Ptr VolumeComputeGenerator::generate(const ast::Tree& tree)
{
    llvm::FunctionType* type =
        llvmFunctionTypeFromSignature<VolumeKernelValue::Signature>(mContext);

    mFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        VolumeKernelValue::getDefaultName(),
        &mModule);

    // Set up arguments for initial entry

    llvm::Function::arg_iterator argIter = mFunction->arg_begin();
    const auto arguments = VolumeKernelValue::argumentKeys();
    auto keyIter = arguments.cbegin();

    for (; argIter != mFunction->arg_end(); ++argIter, ++keyIter) {
        argIter->setName(*keyIter);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mContext, "k1.entry", mFunction);
    mBuilder.SetInsertPoint(entry);

    // build the attribute registry

    AttributeRegistry::Ptr registry = AttributeRegistry::create(tree);

    // Visit all attributes and allocate them in local IR memory - assumes attributes
    // have been verified by the ax compiler
    // @note  Call all attribute allocs at the start of this block so that llvm folds
    // them into the function prologue (as a static allocation)

    SymbolTable<Value>* localTable = this->mSymbolTables.getOrInsert(1);

    // run allocations and update the symbol table

    for (const AttributeRegistry::AccessData& data : registry->data()) {
        llvm::Type* type = llvmTypeFromToken(data.type(), mContext);
        {
            llvm::Value* vptr = mBuilder.CreateAlloca(llvm::PointerType::get(type, 0));
            mInternalPtrs.insert(data.tokenname() + "_vptr", vptr);
            OPENVDB_ASSERT(llvm::cast<llvm::AllocaInst>(vptr)->isStaticAlloca());
        }

        // @warning This method will insert the alloc before the above alloc.
        //  This is fine, but is worth noting
        Value value = Value::Alloc(mBuilder, type);
        OPENVDB_ASSERT(llvm::cast<llvm::AllocaInst>(value.GetValue())->isStaticAlloca());

        // @note  this technically doesn't need to live in the local table
        //  (only the pointer to this value (_vptr) needs to) but it's
        //  re-accessed by the subsequent loop. could remove this.
        localTable->insert(data.tokenname(), value);
    }

    // insert getters for read variables

    for (const AttributeRegistry::AccessData& data : registry->data()) {
        if (!data.reads()) continue;
        const std::string token = data.tokenname();
        this->getAccessorValue(token, *(localTable->get(token)));
    }

    // full code generation
    // errors can stop traversal, but dont always, so check the log

    const size_t err = mLog.errors();
    if (!this->traverse(&tree) || (mLog.errors() > err)) return nullptr;

    // insert free calls for any strings

    this->createFreeSymbolStrings(mBuilder);

    this->computek2(mFunction, *registry);
    this->computek3(mFunction, *registry);

    return registry;
}

bool VolumeComputeGenerator::visit(const ast::Attribute* node)
{
    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    const std::string globalName = node->tokenname();
    auto ptrToAlloc = mInternalPtrs.get(globalName + "_vptr"); // llvm::Value**
    OPENVDB_ASSERT(ptrToAlloc);
    llvm::Value* alloc = mBuilder.CreateLoad(llvm::PointerType::get(type, 0), *ptrToAlloc);
    OPENVDB_ASSERT(alloc);
    mValues.push(Value(alloc, type));
    return true;
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


void VolumeComputeGenerator::getAccessorValue(const std::string& globalName, Value location)
{
    std::string name, type;
    ast::Attribute::nametypeFromToken(globalName, &name, &type);

    llvm::Value* registeredIndex = llvm::cast<llvm::GlobalVariable>
        (mModule.getOrInsertGlobal(globalName, LLVMType<int64_t>::get(mContext)));
    this->globals().insert(globalName, registeredIndex);
    registeredIndex = mBuilder.CreateLoad(LLVMType<int64_t>::get(mContext), registeredIndex);

    // first see if pre cached node exists.

    llvm::Value* accessIndex = extractArgument(mFunction, "write_index");
    llvm::Value* result = mBuilder.CreateICmpEQ(accessIndex, registeredIndex);
    result = boolComparison(result, mBuilder);

    llvm::BasicBlock* then = llvm::BasicBlock::Create(mContext, "then", mFunction);
    llvm::BasicBlock* els  = llvm::BasicBlock::Create(mContext, "else", mFunction);
    llvm::BasicBlock* post = llvm::BasicBlock::Create(mContext, "post", mFunction);
    mBuilder.CreateCondBr(result, then, els);

    mBuilder.SetInsertPoint(then);
    {
        llvm::Value* valueptr = extractArgument(mFunction, "value");
        llvm::Value* offset = extractArgument(mFunction, "offset");
        OPENVDB_ASSERT(valueptr);
        OPENVDB_ASSERT(offset);
        OPENVDB_ASSERT(AssertOpaquePtrs(valueptr, LLVMType<int8_t>::get(mContext)));

        OPENVDB_ASSERT(location.IsPtr());
        llvm::Type* type = location.GetValue()->getType(); // ValueType*
        valueptr = mBuilder.CreatePointerCast(valueptr, type);

        OPENVDB_ASSERT(AssertOpaquePtrs(valueptr, location.GetUnderlyingType()));
        llvm::Value* value = mBuilder.CreateGEP(location.GetUnderlyingType(), valueptr, offset);
        mBuilder.CreateStore(value, *mInternalPtrs.get(globalName + "_vptr"));
        mBuilder.CreateBr(post);
    }

    mBuilder.SetInsertPoint(els);
    {
        // If no node, index into the void* array of handles and load the value
        // through an accessor

        llvm::Value* accessorPtr = extractArgument(mFunction, "accessors");
        llvm::Value* transformPtr = extractArgument(mFunction, "transforms");
        llvm::Value* origin = extractArgument(mFunction, "origin");
        llvm::Value* offset = extractArgument(mFunction, "offset");
        OPENVDB_ASSERT(accessorPtr);
        OPENVDB_ASSERT(transformPtr);
        OPENVDB_ASSERT(origin);
        OPENVDB_ASSERT(offset);

        llvm::Type* voidstar = LLVMType<void*>::get(mContext);

        OPENVDB_ASSERT(AssertOpaquePtrs(accessorPtr, voidstar));
        OPENVDB_ASSERT(AssertOpaquePtrs(transformPtr, voidstar));

        accessorPtr = mBuilder.CreateGEP(voidstar, accessorPtr, registeredIndex);
        llvm::Value* targetTransform = mBuilder.CreateGEP(voidstar, transformPtr, registeredIndex);
        llvm::Value* sourceTransform = mBuilder.CreateGEP(voidstar, transformPtr, accessIndex);

        llvm::Value* accessor = mBuilder.CreateLoad(voidstar, accessorPtr);
        targetTransform = mBuilder.CreateLoad(voidstar, targetTransform);
        sourceTransform = mBuilder.CreateLoad(voidstar, sourceTransform);

        Arguments getvargs;
        getvargs.AddArg(accessor,            ArgInfo(LLVMType<int8_t>::get(mContext), 1));
        getvargs.AddArg(sourceTransform,     ArgInfo(LLVMType<int8_t>::get(mContext), 1));
        getvargs.AddArg(targetTransform,     ArgInfo(LLVMType<int8_t>::get(mContext), 1));
        getvargs.AddArg(origin,              ArgInfo(LLVMType<openvdb::math::Vec3<int32_t>>::get(mContext), 1));
        getvargs.AddArg(offset,              ArgInfo(mBuilder.getInt64Ty()));
        getvargs.AddArg(location.GetValue(), ArgInfo(location));

        const FunctionGroup* const F = this->getFunction("getvoxel", true);
        OPENVDB_ASSERT(F);
        F->execute(getvargs, mBuilder);

        mBuilder.CreateStore(location.GetValue(), *mInternalPtrs.get(globalName + "_vptr"));
        mBuilder.CreateBr(post);
    }

    mBuilder.SetInsertPoint(post);
}

llvm::Value* VolumeComputeGenerator::accessorHandleFromToken(const std::string& globalName)
{
    // Visiting an "attribute" - get the volume accessor out of a vector of void pointers
    // mAttributeHandles is a void pointer to a vector of void pointers (void**)

    llvm::Value* registeredIndex = llvm::cast<llvm::GlobalVariable>
        (mModule.getOrInsertGlobal(globalName, LLVMType<int64_t>::get(mContext)));
    this->globals().insert(globalName, registeredIndex);

    registeredIndex = mBuilder.CreateLoad(LLVMType<int64_t>::get(mContext), registeredIndex);

    // index into the void* array of handles and load the value.
    // The result is a loaded void* value

    llvm::Type* voidstar = LLVMType<void*>::get(mContext);
    llvm::Value* accessorPtr = extractArgument(mFunction, "accessors");
    OPENVDB_ASSERT(accessorPtr);
    OPENVDB_ASSERT(AssertOpaquePtrs(accessorPtr, voidstar));

    accessorPtr = mBuilder.CreateGEP(voidstar, accessorPtr, registeredIndex);
    // return loaded void** = void*
    return mBuilder.CreateLoad(voidstar, accessorPtr);
}

} // namespace codegen_internal
} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

