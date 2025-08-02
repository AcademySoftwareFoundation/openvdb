// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/PointComputeGenerator.cc

#include "PointComputeGenerator.h"

#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"
#include "Codecs.h"

#include "openvdb_ax/Exceptions.h"
#include "openvdb_ax/ast/Scanners.h"

#include <openvdb/util/Assert.h>

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

const std::array<const char*, PointKernelValue::N_ARGS>&
PointKernelValue::argumentKeys()
{
    static const std::array<const char*, PointKernelValue::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value_buffer",
        "isactive",
        "point_index",
        "transforms",
        "values",
        "flags",
        "attribute_set",
        "group_handles",
        "leaf_data"
    }};

    return arguments;
}

const char* PointKernelValue::getDefaultName() { return "ax.compute.point.PKV"; }

//

const std::array<const char*, PointKernelBufferRange::N_ARGS>&
PointKernelBufferRange::argumentKeys()
{
    static const std::array<const char*, PointKernelBufferRange::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value_buffer",
        "active_buffer",
        "buffer_size",
        "mode",
        "transforms",
        "buffers",
        "flags",
        "attribute_set",
        "group_handles",
        "leaf_data"
    }};

    return arguments;
}

const char* PointKernelBufferRange::getDefaultName() { return "ax.compute.point.PKBR"; }

//

const std::array<const char*, PointKernelBuffer::N_ARGS>&
PointKernelBuffer::argumentKeys()
{
    static const std::array<const char*, PointKernelBuffer::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value_buffer",
        "isactive",
        "point_index",
        "transforms",
        "buffers",
        "flags",
        "attribute_set",
        "group_handles",
        "leaf_data"
    }};

    return arguments;
}

const char* PointKernelBuffer::getDefaultName() { return "ax.compute.point.PKB"; }

//

const std::array<const char*, PointKernelAttributeArray::N_ARGS>&
PointKernelAttributeArray::argumentKeys()
{
    static const std::array<const char*, PointKernelAttributeArray::N_ARGS> arguments = {{
        "custom_data",
        "origin",
        "value_buffer",
        "isactive",
        "point_index",
        "transforms",
        "attribute_arrays",
        "flags",
        "attribute_set",
        "group_handles",
        "leaf_data"
    }};

    return arguments;
}

const char* PointKernelAttributeArray::getDefaultName() { return "ax.compute.point.PKAA"; }

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

inline void PointComputeGenerator::computePKBR(const AttributeRegistry&)
{
    llvm::Function* compute = mModule.getFunction(PointKernelBuffer::getDefaultName());

    auto generate =
        [&](const Arguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        using VBuffPtrT = PointKernelBufferRange::FunctionTraitsT::Arg<2>::Type; // leaf value buffer argument ptr type
        using ABuffPtrT = PointKernelBufferRange::FunctionTraitsT::Arg<3>::Type; // active buffer type (word masks) argument ptr type

        llvm::LLVMContext& C = B.getContext();

        llvm::Type* vbuffT = LLVMType<std::remove_pointer_t<VBuffPtrT>>::get(C);
        llvm::Type* abuffT = LLVMType<std::remove_pointer_t<ABuffPtrT>>::get(C);
        llvm::Type* incT   = LLVMType<int64_t>::get(C); // type of the loop counter
        OPENVDB_ASSERT(vbuffT == B.getInt32Ty()); // sanity check
        OPENVDB_ASSERT(abuffT == B.getInt64Ty()); // sanity check

        OPENVDB_ASSERT(args.size() == 12);
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
        llvm::BasicBlock* iterBlock = llvm::BasicBlock::Create(C, "k2.buffiter", base);

        // init var - loops from 0 -> buffSize
        llvm::Value* incr = insertStaticAlloca(B, incT);
        B.CreateStore(B.getInt64(0), incr);
        B.CreateBr(conditionBlock);

        // increment
        B.SetInsertPoint(iterBlock);
        {
            llvm::Value* new_incr = B.CreateLoad(incT, incr);
            new_incr = B.CreateAdd(new_incr, B.getInt64(1));
            B.CreateStore(new_incr, incr);
            B.CreateBr(conditionBlock);
        }

        // generate loop body
        B.SetInsertPoint(bodyBlock);
        {
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
            llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "k2.do_points", base);

            B.CreateCondBr(process, then, iterBlock);
            B.SetInsertPoint(then);
            {
                // branches for getting the end point index
                llvm::BasicBlock* pthen = llvm::BasicBlock::Create(C, "k2.get_0_end", base);
                llvm::BasicBlock* pelse = llvm::BasicBlock::Create(C, "k2.get_p_end", base);

                // loop branches
                llvm::BasicBlock* pcondition = llvm::BasicBlock::Create(C, "k2.pcond", base);
                llvm::BasicBlock* pbody = llvm::BasicBlock::Create(C, "k2.pbody", base);
                llvm::BasicBlock* piter = llvm::BasicBlock::Create(C, "k2.piter", base);

                // loops from pindex->pindexend (point grids have 32bit buffers)
                Value pindex = Value::Alloc(B, vbuffT);
                llvm::Value* pindexend = B.CreateGEP(vbuffT, vbuff, lincr);
                pindexend = B.CreateLoad(vbuffT, pindexend);

                llvm::Value* firstvoxel = binaryOperator(lincr, B.getInt64(0), ast::tokens::EQUALSEQUALS, B);
                B.CreateCondBr(firstvoxel, pthen, pelse);
                B.SetInsertPoint(pthen);
                {
                    pindex.Assign(B, Value::Create<int32_t>(C, 0));
                    B.CreateBr(pcondition);
                }

                B.SetInsertPoint(pelse);
                {
                    llvm::Value* prevv = binaryOperator(lincr, B.getInt64(1), ast::tokens::MINUS, B);
                    llvm::Value* pindexcount = B.CreateGEP(vbuffT, vbuff, prevv);
                    B.CreateStore(B.CreateLoad(vbuffT, pindexcount), pindex.GetValue());
                    B.CreateBr(pcondition);
                }

                B.SetInsertPoint(pcondition);
                {
                    llvm::Value* end = B.CreateICmpULT(pindex.Load(B).GetValue(), pindexend);
                    B.CreateCondBr(end, pbody, iterBlock);
                }

                B.SetInsertPoint(piter);
                {
                    pindex.Assign(B, pindex.Add(B, Value::Create<int32_t>(C, 1)));
                    B.CreateBr(pcondition);
                }

                B.SetInsertPoint(pbody);
                {
                    // invoke the point kernel for this value
                    const std::array<llvm::Value*, 11> input {
                        args[0],  // ax::CustomData
                        args[1],  // index space coordinate
                        vbuff,    // value buffer
                        ison,     // active/inactive
                        pindex.CastToPrecision(B, B.getInt64Ty()).LoadIfPtr(B).GetValue(), // offset in the point array
                        args[6],  // transforms
                        args[7],  // buffers
                        args[8],  // flags
                        args[9],  // attr set
                        args[10], // groups
                        args[11]  // leafdata
                    };

                    B.CreateCall(compute, input);
                    B.CreateBr(piter);
                }
            }
        }

        B.SetInsertPoint(conditionBlock);
        llvm::Value* endCondition = B.CreateICmpULT(B.CreateLoad(incT, incr), buffSize);

        llvm::BasicBlock* postBlock = llvm::BasicBlock::Create(C, "k2.end", base);
        B.CreateCondBr(endCondition, bodyBlock, postBlock);
        B.SetInsertPoint(postBlock);
        return Value::Invalid();
    };

    // Use the function builder to generate the correct prototype and body for K2
    auto k = FunctionBuilder(PointKernelBufferRange::getDefaultName())
        .addSignature<PointKernelBufferRange::Signature>(generate, PointKernelBufferRange::getDefaultName())
        .setConstantFold(false)
        .setEmbedIR(false)
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(0, llvm::Attribute::NoCapture)
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .addParameterAttribute(1, llvm::Attribute::NoCapture)
        .addParameterAttribute(1, llvm::Attribute::NoAlias)
        .addParameterAttribute(2, llvm::Attribute::NoCapture)
        .addParameterAttribute(2, llvm::Attribute::NoAlias)
        .addParameterAttribute(3, llvm::Attribute::NoCapture)
        .addParameterAttribute(3, llvm::Attribute::NoAlias)
        .addParameterAttribute(6, llvm::Attribute::NoCapture)
        .addParameterAttribute(6, llvm::Attribute::NoAlias)
        .addParameterAttribute(7, llvm::Attribute::NoCapture)
        .addParameterAttribute(7, llvm::Attribute::NoAlias)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        .get();

    k->list()[0]->create(mContext, &mModule);
}

/// @brief  Decode the value held by buffer[pid] based on the flag type.
/// @details Certain types either cannot be encoded or can only be encoded with
///   certain encoders, so we use the type information to filter out what
///   encoders we need to loop over in IR.
/// @return  Either returns buffer[pid] if no decoding was necessary or a
///   pointer to the newly allocated var of the decoded type which will have
///   been allocated in the function prologue
inline llvm::Value*
decode(llvm::Value* buffer,
    llvm::Value* pid,
    llvm::Value* flag,
    llvm::Value* store,
    ast::tokens::CoreType decodedType,
    llvm::IRBuilder<>& B)
{
    llvm::LLVMContext& C = B.getContext();
    llvm::Type* type = llvmTypeFromToken(decodedType, C);

    // see if this type might be encoded. If not, just return the original value
    const auto* codecs = getTypeSupportedCodecs(decodedType);
    if (!codecs) {
        // Value can't be encoded so the buffer is guaranteed to be the decoded type
        buffer = B.CreatePointerCast(buffer, llvm::PointerType::get(type, 0));
        return B.CreateGEP(type, buffer, pid);
    }
    OPENVDB_ASSERT(!codecs->empty());
    OPENVDB_ASSERT(store);

    llvm::Function* self = B.GetInsertBlock()->getParent();
    llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "k1.get_buffer.decode", self);

    for (const auto& codecNamePair : *codecs) {
        const std::string& name = codecNamePair.first;
        const Codec* codec = codecNamePair.second;

        llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "k1.get_buffer." + name, self);
        llvm::BasicBlock* els  = llvm::BasicBlock::Create(C, "", self);

        llvm::Value* usescodec =
            B.CreateAnd(flag, LLVMType<uint64_t>::get(C, codec->flag()));
        usescodec = boolComparison(usescodec, B);
        B.CreateCondBr(usescodec, then, els);

        B.SetInsertPoint(then);
        {
            // If this is the codec in use, get the appropriate function, cast
            // the input value and decode the value.
            const FunctionGroup* const F = codec->decoder();
            llvm::Type* encodedType = codec->decodedToEncoded(decodedType, C);
            OPENVDB_ASSERT(encodedType);
            llvm::Type* encodedTypePtr = llvm::PointerType::get(encodedType, 0);

            // guranteed to be castable
            llvm::Value* typedBuffer = B.CreatePointerCast(buffer, encodedTypePtr);
            llvm::Value* encoded = B.CreateGEP(encodedType, typedBuffer, pid);

            Arguments args;
            args.AddArg(store, ArgInfo(type, 1));
            args.AddArg(encoded, ArgInfo(encodedType, 1));
            OPENVDB_ASSERT(F->match(args.GetArgInfo(), C).first);
            F->execute(args, B);
            B.CreateBr(post);
        }

        B.SetInsertPoint(els);
    }

    // if we're here (the final else), the value is not encoded
    // @todo  We could instead register all vaid nullcodecs which would give
    //   guarantees should a codec not exist
    llvm::Value* typedBuffer = B.CreatePointerCast(buffer, llvm::PointerType::get(type, 0));
    llvm::Value* value = B.CreateGEP(type, typedBuffer, pid);
    B.CreateStore(B.CreateLoad(type, value), store);
    B.CreateBr(post);

    B.SetInsertPoint(post);
    return store;
}

/// @brief  Encode the value held by "in" based on the flag type, to the
///   location pointed to by buffer[pid].
/// @details Certain types either cannot be encoded or can only be encoded with
///   certain codec, so we use the type information to filter out what
///   codec we need to loop over in IR.
/// @return  Either returns immediately if no encoding was necessary or stores
///   the encoded version of "in" at the location pointed to by buffer[pid]
inline void
encode(llvm::Value* in,
    llvm::Value* buffer,
    llvm::Value* pid,
    llvm::Value* flag,
    ast::tokens::CoreType decodedType,
    llvm::IRBuilder<>& B)
{
    llvm::LLVMContext& C = B.getContext();
    llvm::Type* type = llvmTypeFromToken(decodedType, C);

    // see if this type might be encoded. If not, just store the original value
    const auto* codecs = getTypeSupportedCodecs(decodedType);
    if (!codecs) {
        // Value can't be encoded so the buffer is guaranteed to be the decoded type
        buffer = B.CreatePointerCast(buffer, llvm::PointerType::get(type, 0));
        llvm::Value* store = B.CreateGEP(type, buffer, pid);
        B.CreateStore(B.CreateLoad(type, in), store);
        return;
    }
    OPENVDB_ASSERT(!codecs->empty());

    llvm::Function* self = B.GetInsertBlock()->getParent();
    llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "k1.set_buffer.encode", self);

    for (const auto& codecNamePair : *codecs) {
        const std::string& name = codecNamePair.first;
        const Codec* codec = codecNamePair.second;

        llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "k1.set_buffer." + name, self);
        llvm::BasicBlock* els  = llvm::BasicBlock::Create(C, "", self);

        llvm::Value* usescodec =
            B.CreateAnd(flag, LLVMType<uint64_t>::get(C, codec->flag()));
        usescodec = boolComparison(usescodec, B);
        B.CreateCondBr(usescodec, then, els);

        B.SetInsertPoint(then);
        {
            const FunctionGroup* const F = codec->encoder();
            llvm::Type* encodedType = codec->decodedToEncoded(decodedType, C);
            OPENVDB_ASSERT(encodedType);
            llvm::Type* encodedTypePtr = llvm::PointerType::get(encodedType, 0);
            llvm::Value* typedBuffer = B.CreatePointerCast(buffer, encodedTypePtr);
            llvm::Value* loc = B.CreateGEP(encodedType, typedBuffer, pid);

            Arguments args;
            args.AddArg(loc, ArgInfo(encodedType, 1));
            args.AddArg(in, ArgInfo(type, 1));
            OPENVDB_ASSERT(F->match(args.GetArgInfo(), C).first);
            F->execute(args, B);

            B.CreateBr(post);
        }

        B.SetInsertPoint(els);
    }

    // if we're here (the final else), the value is not encodable
    // @todo  We could instead register all vaid nullcodecs which would give
    //   guarantees should a codec not exist
    llvm::Value* typedBuffer = B.CreatePointerCast(buffer, llvm::PointerType::get(type, 0));
    llvm::Value* loc = B.CreateGEP(type, typedBuffer, pid);
    B.CreateStore(B.CreateLoad(type, in), loc);
    B.CreateBr(post);

    B.SetInsertPoint(post);
}

inline void PointComputeGenerator::computePKB(const AttributeRegistry& registry)
{
    llvm::Function* compute = mModule.getFunction(PointKernelValue::getDefaultName());

    auto generate =
        [&](const Arguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 11);
        auto& C = B.getContext();
        llvm::Function* self = B.GetInsertBlock()->getParent();
        llvm::Value* pindex = extractArgument(self, "point_index");
        llvm::Value* flags = extractArgument(self, "flags");
        llvm::Value* buffers = extractArgument(self, "buffers");
        OPENVDB_ASSERT(buffers);
        OPENVDB_ASSERT(pindex);
        OPENVDB_ASSERT(flags);

        llvm::Type* voidstar = LLVMType<void*>::get(C);
        llvm::Type* buffT = voidstar;
        llvm::Type* flagT = LLVMType<uint64_t>::get(C);
        OPENVDB_ASSERT(AssertOpaquePtrs(buffers, buffT));
        OPENVDB_ASSERT(AssertOpaquePtrs(flags, flagT));

        // create array of void*. each pointer will encode an address to a stored typed value
        llvm::Type* locType = llvm::ArrayType::get(voidstar, registry.data().size()); // [SIZE x i8*]
        llvm::Value* loc = insertStaticAlloca(B, locType);

        OPENVDB_ASSERT(AssertOpaquePtrs(loc, locType));

        size_t i = 0;
        for (const AttributeRegistry::AccessData& data : registry.data())
        {
            const std::string token = data.tokenname();
            llvm::Type* type = llvmTypeFromToken(data.type(), C);
            llvm::Type* typestarstar = llvm::PointerType::get(llvm::PointerType::get(type, 0), 0);

            llvm::Value* decodedPtrs = B.CreateConstInBoundsGEP2_64(locType, loc, 0, i++); // void**, location to hold the typed ptr
            decodedPtrs = B.CreatePointerCast(decodedPtrs, typestarstar); // ValueType**

            llvm::Value* index = mModule.getGlobalVariable(token);
            OPENVDB_ASSERT(index);
            index = B.CreateLoad(B.getInt64Ty(), index);
            llvm::Value* buffer = B.CreateGEP(buffT, buffers, index);
            buffer = B.CreateLoad(buffT, buffer); // void** = void*

            llvm::Value* flag = B.CreateLoad(flagT, B.CreateGEP(flagT, flags, index));

            // @todo  write handles shouldn't need to do this check
            llvm::Value* isuniform = B.CreateAnd(flag, LLVMType<uint64_t>::get(C, (uint64_t(1) << 63)));
            isuniform = boolComparison(isuniform, B);

            // If the value type has supported codecs we have to allocate the
            // expected decoded type that will be stored. Otherwise, decode()
            // will simply extract the value ptr directly from the buffer.
            llvm::Value* decodedStore = nullptr;
            const auto* codecs = getTypeSupportedCodecs(data.type());
            if (codecs) decodedStore = insertStaticAlloca(B, llvmTypeFromToken(data.type(), C)); // allocated to prologue

            llvm::BasicBlock* then = llvm::BasicBlock::Create(C, "k1.get_buffer.uniform", self);
            llvm::BasicBlock* els  = llvm::BasicBlock::Create(C, "k1.get_buffer.nuniform", self);
            llvm::BasicBlock* cont = llvm::BasicBlock::Create(C, "k1.get_buffer.continue", self);

            B.CreateCondBr(isuniform, then, els);

            B.SetInsertPoint(then);
            {
                llvm::Value* ptr = decode(buffer, B.getInt64(0), flag, decodedStore, data.type(), B);
                B.CreateStore(ptr, decodedPtrs);
                B.CreateBr(cont);
            }

            B.SetInsertPoint(els);
            {
                llvm::Value* ptr = decode(buffer, pindex, flag, decodedStore, data.type(), B);
                B.CreateStore(ptr, decodedPtrs);
                B.CreateBr(cont);
            }

            B.SetInsertPoint(cont);
        }

        llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "k1.run_compute", self);
        B.CreateBr(post);
        B.SetInsertPoint(post);

        // invoke the point kernel for this value
        std::array<llvm::Value*, 11> input;
        std::copy_n(args.AsLLVMValues().begin(), 11, input.begin());
        input[6] = B.CreateConstInBoundsGEP2_64(locType, loc, 0, 0); // void**, replace the buffers with the extracted values
        B.CreateCall(compute, input);

        // insert writes to any attributes that were potentially compressed
        i = 0;
        for (const AttributeRegistry::AccessData& data : registry.data())
        {
            if (!data.writes()) {
                i++;
                continue;
            }

            const std::string token = data.tokenname();
            llvm::Type* type = llvmTypeFromToken(data.type(), C);

            llvm::Value* store = B.CreateConstInBoundsGEP2_64(locType, loc, 0, i++); // void**, location to hold the typed ptr
            OPENVDB_ASSERT(AssertOpaquePtrs(store, voidstar));
            store = B.CreateLoad(voidstar, store); // void*
            store = B.CreatePointerCast(store, llvm::PointerType::get(type, 0)); // ValueType*

            llvm::Value* index = mModule.getGlobalVariable(token);
            OPENVDB_ASSERT(index);
            index = B.CreateLoad(B.getInt64Ty(), index);
            llvm::Value* flag = B.CreateLoad(flagT, B.CreateGEP(flagT, flags, index));

            llvm::Value* buffer = B.CreateGEP(buffT, buffers, index);
            buffer = B.CreateLoad(buffT, buffer); // void** = void*

            // The buffer should not be uniform if we're writing to it, so no
            // need to branch for this case (this should be guaranteed by the
            // PointExectuable)
            encode(store, buffer, pindex, flag, data.type(), B);
        }

        return Value::Invalid();
    };

    const auto& keys = PointKernelBuffer::argumentKeys();

    // Use the function builder to generate the correct prototype and body for K2
    auto k = FunctionBuilder(PointKernelBuffer::getDefaultName())
        .addSignature<PointKernelBuffer::Signature>(generate, PointKernelBuffer::getDefaultName())
        .setConstantFold(false)
        .setEmbedIR(false)
        .setArgumentNames(std::vector<const char*>(keys.begin(), keys.end()))
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(0, llvm::Attribute::NoCapture)
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .addParameterAttribute(1, llvm::Attribute::NoCapture)
        .addParameterAttribute(1, llvm::Attribute::NoAlias)
        .addParameterAttribute(2, llvm::Attribute::NoCapture)
        .addParameterAttribute(2, llvm::Attribute::NoAlias)
        .addParameterAttribute(5, llvm::Attribute::NoCapture)
        .addParameterAttribute(5, llvm::Attribute::NoAlias)
        .addParameterAttribute(6, llvm::Attribute::NoCapture)
        .addParameterAttribute(6, llvm::Attribute::NoAlias)
        .addParameterAttribute(7, llvm::Attribute::NoCapture)
        .addParameterAttribute(7, llvm::Attribute::NoAlias)
        .addParameterAttribute(8, llvm::Attribute::NoCapture)
        .addParameterAttribute(8, llvm::Attribute::NoAlias)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        .get();

    k->list()[0]->create(mContext, &mModule);
}

inline void PointComputeGenerator::computePKAA(const AttributeRegistry& registry)
{
    llvm::Function* compute = mModule.getFunction(PointKernelValue::getDefaultName());

    /// @brief PKAA function for getting a point value from an attribute array
    auto getAttributeValue = [this](const std::string& token,
        llvm::Value* pindex,
        llvm::Value* store,
        ast::tokens::CoreType storeType,
        llvm::IRBuilder<>& B)
    {
        llvm::Function* self = B.GetInsertBlock()->getParent();
        llvm::Module* M = self->getParent();
        llvm::LLVMContext& C = B.getContext();
        llvm::Type* stype = llvmTypeFromToken(storeType, C);
        llvm::Type* voidstar = LLVMType<void*>::get(C);

        // insert the attribute into the map of global variables and get a unique global representing
        // the location which will hold the attribute handle offset.
        llvm::Value* index = M->getGlobalVariable(token);
        OPENVDB_ASSERT(index);
        index = B.CreateLoad(B.getInt64Ty(), index);

        llvm::Value* arrays = extractArgument(self, "attribute_arrays");
        OPENVDB_ASSERT(arrays);
        OPENVDB_ASSERT(AssertOpaquePtrs(arrays, voidstar));

        llvm::Value* array = B.CreateGEP(voidstar, arrays, index);
        array = B.CreateLoad(voidstar, array); // void** = void*

        // invoke C binding
        Arguments args;
        args.AddArg(array,  ArgInfo(LLVMType<int8_t>::get(C), 1));
        args.AddArg(pindex, ArgInfo(B.getInt64Ty()));
        args.AddArg(store , ArgInfo(stype, 1));
        if (storeType == ast::tokens::CoreType::STRING) {
            args.AddArg(extractArgument(self, "leaf_data"), ArgInfo(LLVMType<int8_t>::get(C), 1));
        }
        const FunctionGroup* const F = this->getFunction("getattribute", true);
        F->execute(args, B);
    };

    /// @brief PKAA function for setting a point value on an attribute array
    auto setAttributeValue = [this](const std::string& token,
        llvm::Value* pindex,
        llvm::Value* load,
        ast::tokens::CoreType loadType,
        llvm::IRBuilder<>& B)
    {
        llvm::Function* self = B.GetInsertBlock()->getParent();
        llvm::Module* M = self->getParent();
        llvm::LLVMContext& C = B.getContext();
        llvm::Type* voidstar = LLVMType<void*>::get(C);
        llvm::Type* ltype = llvmTypeFromToken(loadType, C);

        // insert the attribute into the map of global variables and get a unique global representing
        // the location which will hold the attribute handle offset.
        llvm::Value* index = M->getGlobalVariable(token);
        OPENVDB_ASSERT(index);
        index = B.CreateLoad(B.getInt64Ty(), index);

        llvm::Value* arrays = extractArgument(self, "attribute_arrays");
        OPENVDB_ASSERT(arrays);
        OPENVDB_ASSERT(AssertOpaquePtrs(arrays, voidstar));

        llvm::Value* array = B.CreateGEP(voidstar, arrays, index);
        array = B.CreateLoad(voidstar, array); // void** = void*

        // load the result (if its a scalar)
        const bool loadScalar = (ltype->isIntegerTy() || ltype->isFloatingPointTy()) ? true : false;
        if (loadScalar) {
            load = B.CreateLoad(ltype, load);
        }

        // invoke C binding
        Arguments args;
        args.AddArg(array,  ArgInfo(LLVMType<int8_t>::get(C), 1)); // handle
        args.AddArg(pindex, ArgInfo(B.getInt64Ty())); // point index
        args.AddArg(load,   ArgInfo(ltype, loadScalar ? 0 : 1)); // set value
        if (loadType == ast::tokens::CoreType::STRING) {
            args.AddArg(extractArgument(self, "leaf_data"), ArgInfo(LLVMType<int8_t>::get(C), 1));
        }
        const FunctionGroup* const function = this->getFunction("setattribute", true);
        function->execute(args, B);
    };

    //

    auto generate =
        [&](const Arguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 11);
        auto& C = B.getContext();
        llvm::Function* self = B.GetInsertBlock()->getParent();
        llvm::Value* pindex = extractArgument(self, "point_index");

        SymbolTable<llvm::Value*> table;

        // create array of void*. each element will hold the attribute values
        llvm::Type* locType = llvm::ArrayType::get(LLVMType<void*>::get(C), registry.data().size()); // [SIZE x i8*]
        llvm::Value* loc = insertStaticAlloca(B, locType);

        // run allocations
        size_t i = 0;
        for (const AttributeRegistry::AccessData& access : registry.data()) {
            llvm::Value* value = insertStaticAlloca(B, llvmTypeFromToken(access.type(), C));
            OPENVDB_ASSERT(llvm::cast<llvm::AllocaInst>(value)->isStaticAlloca());
            table.insert(access.tokenname(), value);

            // store the allocated ptr in the array of void*
            llvm::Value* store = B.CreateConstInBoundsGEP2_64(locType, loc, 0, i); // void**, location to hold the typed ptr
            value = B.CreatePointerCast(value, LLVMType<void*>::get(C));
            B.CreateStore(value, store);

            ++i;
        }

        // get attributes
        for (const AttributeRegistry::AccessData& data : registry.data()) {
            const std::string token = data.tokenname();
            llvm::Value* store = *(table.get(token));
            getAttributeValue(token, pindex, store, data.type(), B);
        }

        // invoke the point kernel for this value
        std::array<llvm::Value*, 11> input;
        std::copy_n(args.AsLLVMValues().begin(), 11, input.begin());
        input[6] = B.CreateConstInBoundsGEP2_64(locType, loc, 0, 0); // void**, replace the buffers with the extracted values
        B.CreateCall(compute, input);

        // insert set code and deallocations
        for (const AttributeRegistry::AccessData& data : registry.data()) {
            if (!data.writes()) continue;

            const std::string token = data.tokenname();
            llvm::Value* value = *(table.get(token));
            // // Expected to be used more than one (i.e. should never be zero)
            // OPENVDB_ASSERT(value->hasNUsesOrMore(1));
            // // Check to see if this value is still being used - it may have
            // // been cleaned up due to returns. If there's only one use, it's
            // // the original get of this attribute.
            // if (value->hasOneUse()) {
            //     // @todo  The original get can also be optimized out in this case
            //     // this->globals().remove(variable.first);
            //     // mModule.getGlobalVariable(variable.first)->eraseFromParent();
            //     continue;
            // }
            setAttributeValue(token, pindex, value, data.type(), B);
        }

        Value last = Value::Return(B);

        // insert free calls for any strings
        this->createFreeSymbolStrings(B);

        return last;
    };

    const auto& keys = PointKernelAttributeArray::argumentKeys();

    // Use the function builder to generate the correct prototype and body for K2
    auto k = FunctionBuilder(PointKernelAttributeArray::getDefaultName())
        .addSignature<PointKernelAttributeArray::Signature>(generate, PointKernelAttributeArray::getDefaultName())
        .setConstantFold(false)
        .setEmbedIR(false)
        .setArgumentNames(std::vector<const char*>(keys.begin(), keys.end()))
        .addParameterAttribute(0, llvm::Attribute::ReadOnly)
        .addParameterAttribute(0, llvm::Attribute::NoCapture)
        .addParameterAttribute(0, llvm::Attribute::NoAlias)
        .addParameterAttribute(1, llvm::Attribute::ReadOnly)
        .addParameterAttribute(1, llvm::Attribute::NoCapture)
        .addParameterAttribute(1, llvm::Attribute::NoAlias)
        .addParameterAttribute(2, llvm::Attribute::NoCapture)
        .addParameterAttribute(2, llvm::Attribute::NoAlias)
        .addParameterAttribute(5, llvm::Attribute::NoCapture)
        .addParameterAttribute(5, llvm::Attribute::NoAlias)
        .addParameterAttribute(6, llvm::Attribute::NoCapture)
        .addParameterAttribute(6, llvm::Attribute::NoAlias)
        .addParameterAttribute(7, llvm::Attribute::NoCapture)
        .addParameterAttribute(7, llvm::Attribute::NoAlias)
        .addParameterAttribute(8, llvm::Attribute::NoCapture)
        .addParameterAttribute(8, llvm::Attribute::NoAlias)
        .addFunctionAttribute(llvm::Attribute::NoRecurse)
        .get();

    k->list()[0]->create(mContext, &mModule);
}

PointComputeGenerator::PointComputeGenerator(llvm::Module& module,
                                             const FunctionOptions& options,
                                             FunctionRegistry& functionRegistry,
                                             Logger& logger)
    : ComputeGenerator(module, options, functionRegistry, logger) {}


AttributeRegistry::Ptr PointComputeGenerator::generate(const ast::Tree& tree)
{
    llvm::FunctionType* type =
        llvmFunctionTypeFromSignature<PointKernelValue::Signature>(mContext);

    mFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        PointKernelValue::getDefaultName(),
        &mModule);

    // @note  Might be worth always inlining the value kernel into the buffer kernel
    //mFunction->addFnAttr(llvm::Attribute::AlwaysInline);

    // Set up arguments for initial entry

    llvm::Function::arg_iterator argIter = mFunction->arg_begin();
    const auto arguments = PointKernelValue::argumentKeys();
    auto keyIter = arguments.cbegin();

    for (; argIter != mFunction->arg_end(); ++argIter, ++keyIter) {
        argIter->setName(*keyIter);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mContext, "k1.entry", mFunction);
    mBuilder.SetInsertPoint(entry);

    // build the attribute registry

    AttributeRegistry::Ptr registry = AttributeRegistry::create(tree);

    // intialise the global indices - do this here so it's only done once

    for (const AttributeRegistry::AccessData& access : registry->data()) {
        const std::string token = access.tokenname();
        llvm::Value* index = llvm::cast<llvm::GlobalVariable>
            (mModule.getOrInsertGlobal(token, LLVMType<int64_t>::get(mContext)));
        this->globals().insert(token, index);
    }

    // full code generation
    // errors can stop traversal, but dont always, so check the log

    const size_t err = mLog.errors();
    if (!this->traverse(&tree) || (mLog.errors() > err)) return nullptr;

    // insert free calls for any strings

    this->createFreeSymbolStrings(mBuilder);

    // compute extra kernels (order here is important)

    this->computePKB(*registry);
    this->computePKAA(*registry);
    // must come after PKB
    this->computePKBR(*registry);

    return registry;
}

bool PointComputeGenerator::visit(const ast::Attribute* node)
{
    llvm::Value* index = mModule.getGlobalVariable(node->tokenname());
    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    llvm::Type* voidstar = LLVMType<void*>::get(mContext);

    OPENVDB_ASSERT(index);
    // index into the void* array of handles and load the value.
    index = mBuilder.CreateLoad(mBuilder.getInt64Ty(), index);
    llvm::Value* value = extractArgument(mFunction, "values"); // void**
    OPENVDB_ASSERT(AssertOpaquePtrs(value, voidstar));

    value = mBuilder.CreateGEP(voidstar, value, index); // void**
    value = mBuilder.CreateLoad(voidstar, value); // void*
    value = mBuilder.CreatePointerCast(value, llvm::PointerType::get(type, 0)); // void* = ValueType*

    mValues.push(Value(value, type));
    return true;
}

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

