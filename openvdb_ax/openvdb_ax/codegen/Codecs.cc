// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <openvdb/points/AttributeArray.h> // for native codec types
#include <openvdb/util/Assert.h>

#include "Codecs.h"

#include "openvdb_ax/codegen/Functions.h"
#include "openvdb_ax/codegen/FunctionTypes.h"
#include "openvdb_ax/codegen/Types.h"
#include "openvdb_ax/codegen/Utils.h"
#include "openvdb_ax/util/x86.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace codegen {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// Codec functions

/// @note  Expected signature for decoders void(Type* out, Type* in);
/// @note  Expected signature for encoders void(Type* out, Type* in);

using namespace codegen;

inline FunctionGroup::UniquePtr axtrncdecode()
{
    static auto generate =
        [](const NativeArguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        Value out = args[0];
        Value in = args[1];

        if (in.IsScalar())
        {
            OPENVDB_ASSERT(in.IsInteger() || in.GetUnderlyingType()->isHalfTy());
            in = in.Load(B);
            Value result = in.IsInteger() ?
                in.CastToPrecision(B, B.getInt32Ty()) :
                in.CastToPrecision(B, B.getFloatTy());
            out.Assign(B, result);
        }
        else {
            std::vector<Value> outelem, inelem;
            out.ArrayToScalars(B, outelem, /*load*/false);
            in.ArrayToScalars(B, inelem, /*load*/true);
            OPENVDB_ASSERT(outelem.size() == inelem.size());

            OPENVDB_ASSERT(inelem.front().IsInteger() ||
                inelem.front().GetUnderlyingType()->isHalfTy());

            llvm::Type* precision = inelem.front().IsInteger() ? B.getInt32Ty() : B.getFloatTy();
            for (auto& elem : inelem) elem = elem.CastToPrecision(B, precision);

            for (size_t i = 0; i < inelem.size(); ++i) {
                outelem[i].Assign(B, inelem[i]);
            }
        }
        return Value::Invalid();
    };

    return FunctionBuilder("__trncdecode")
        .addSignature<void(float*, openvdb::math::half*)>(generate)
        .addSignature<void(int32_t*, int16_t*)>(generate)
        .addSignature<void(openvdb::math::Vec2<int32_t>*,openvdb::math::Vec2<int16_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec2<float>*,openvdb::math::Vec2<openvdb::math::half>*)>(generate)
        .addSignature<void(openvdb::math::Vec3<int32_t>*,openvdb::math::Vec3<int16_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec3<float>*,openvdb::math::Vec3<openvdb::math::half>*)>(generate)
        .addSignature<void(openvdb::math::Vec4<int32_t>*,openvdb::math::Vec4<int16_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec4<float>*,openvdb::math::Vec4<openvdb::math::half>*)>(generate)
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axtrncencode()
{
    static auto generate =
        [](const NativeArguments& args,
           llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        Value out = args[0];
        Value in = args[1];

        if (in.IsScalar())
        {
            OPENVDB_ASSERT(in.IsInteger() || in.GetUnderlyingType()->isFloatTy());
            in = in.Load(B);
            Value result = in.IsInteger() ?
                in.CastToPrecision(B, B.getInt16Ty()) :
                in.CastToPrecision(B, B.getHalfTy());
            out.Assign(B, result);
        }
        else {
            std::vector<Value> outelem, inelem;
            out.ArrayToScalars(B, outelem, /*load*/false);
            in.ArrayToScalars(B, inelem, /*load*/true);

            OPENVDB_ASSERT(outelem.size() == inelem.size());

            OPENVDB_ASSERT(inelem.front().IsInteger() ||
                inelem.front().GetUnderlyingType()->isFloatTy());

            llvm::Type* precision = inelem.front().IsInteger() ? B.getInt16Ty() : B.getHalfTy();
            for (auto& elem : inelem) elem = elem.CastToPrecision(B, precision);

            for (size_t i = 0; i < inelem.size(); ++i) {
                outelem[i].Assign(B, inelem[i]);
            }
        }
        return Value::Invalid();
    };

    return FunctionBuilder("__trncencode")
        .addSignature<void(openvdb::math::half*, float*)>(generate)
        .addSignature<void(int16_t*, int32_t*)>(generate)
        .addSignature<void(openvdb::math::Vec2<int16_t>*, openvdb::math::Vec2<int32_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec2<openvdb::math::half>*, openvdb::math::Vec2<float>*)>(generate)
        .addSignature<void(openvdb::math::Vec3<int16_t>*, openvdb::math::Vec3<int32_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec3<openvdb::math::half>*, openvdb::math::Vec3<float>*)>(generate)
        .addSignature<void(openvdb::math::Vec4<int16_t>*, openvdb::math::Vec4<int32_t>*)>(generate)
        .addSignature<void(openvdb::math::Vec4<openvdb::math::half>*, openvdb::math::Vec4<float>*)>(generate)
        .setDocumentation("")
        .get();
}

inline FunctionGroup::UniquePtr axfxptdecode(const bool OneByte, const bool IsPositionRange)
{
    auto generate =
        [IsPositionRange](
            const NativeArguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        Value out = args[0];
        Value in = args[1];
        llvm::Type* intype = in.GetUnderlyingScalarType();
        OPENVDB_ASSERT(intype->isIntegerTy(8) || intype->isIntegerTy(16));

        Value offset(LLVMType<float>::get(B.getContext(), 0.5f));

        if (in.IsInteger())
        {
            in = in.Load(B);


            llvm::Value* s = B.CreateUIToFP(in.GetValue(), B.getFloatTy());
            llvm::Value* d = intype->isIntegerTy(8) ?
                LLVMType<float>::get(B.getContext(), float(std::numeric_limits<uint8_t>::max())) :
                LLVMType<float>::get(B.getContext(), float(std::numeric_limits<uint16_t>::max()));
            llvm::Value* result = B.CreateFDiv(s, d);
            if (IsPositionRange) result = B.CreateFSub(result, offset.GetValue());
            B.CreateStore(result, out.GetValue());
        }
        else {
            std::vector<Value> outelem, inelem;
            out.ArrayToScalars(B, outelem, /*load*/false);
            in.ArrayToScalars(B, inelem, /*load*/true);
            OPENVDB_ASSERT(inelem.size() >= 3);
            OPENVDB_ASSERT(outelem.size() == inelem.size());

            llvm::Value* d = intype->isIntegerTy(8) ?
                LLVMType<float>::get(B.getContext(), float(std::numeric_limits<uint8_t>::max())) :
                LLVMType<float>::get(B.getContext(), float(std::numeric_limits<uint16_t>::max()));

            for (size_t i = 0; i < inelem.size(); ++i) {
                llvm::Value* result = B.CreateUIToFP(inelem[i].GetValue(), B.getFloatTy());
                result = B.CreateFDiv(result, d);
                if (IsPositionRange) result = B.CreateFSub(result, offset.GetValue());
                B.CreateStore(result, outelem[i].GetValue());
            }
        }
        return Value::Invalid();
    };

    if (OneByte) {
        return FunctionBuilder(IsPositionRange ? "__prfxpt8decode" : "__ufxpt8decode")
            .addSignature<void(float*, uint8_t*)>(generate)
            .addSignature<void(openvdb::math::Vec3<float>*,openvdb::math::Vec3<uint8_t>*)>(generate)
            .get();
    }
    else {
        return FunctionBuilder(IsPositionRange ? "__prfxpt16decode" : "__ufxpt16decode")
            .addSignature<void(float*, uint16_t*)>(generate)
            .addSignature<void(openvdb::math::Vec3<float>*,openvdb::math::Vec3<uint16_t>*)>(generate)
            .get();
    }
}

inline FunctionGroup::UniquePtr axfxptencode(const bool OneByte, const bool IsPositionRange)
{
    auto generate =
        [IsPositionRange](
            const NativeArguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);
        Value out = args[0];
        Value in = args[1];

        llvm::LLVMContext& C = B.getContext();
        llvm::Function* base = B.GetInsertBlock()->getParent();
        in = in.Load(B);

        Value offset(LLVMType<float>::get(B.getContext(), 0.5f));
        if (IsPositionRange) in = in.Add(B, offset);

        const bool ftx8 = args[0].GetUnderlyingType()->isIntegerTy(8);

        llvm::BasicBlock* lt0 = llvm::BasicBlock::Create(C, "lt0", base);
        llvm::BasicBlock* els = llvm::BasicBlock::Create(C, "else", base);
        llvm::BasicBlock* fin = llvm::BasicBlock::Create(C, "finish", base);
        llvm::Value* r1 = binaryOperator(LLVMType<float>::get(C, 0.0f), in.GetValue(), ast::tokens::MORETHAN, B);
        B.CreateCondBr(r1, lt0, els);

        B.SetInsertPoint(lt0);
        {
            llvm::Value* d = ftx8 ?
                LLVMType<uint8_t>::get(C, std::numeric_limits<uint8_t>::min()) :
                LLVMType<uint16_t>::get(C, std::numeric_limits<uint16_t>::min());
            B.CreateStore(d, out.GetValue());
            B.CreateBr(fin);
        }

        B.SetInsertPoint(els);
        {
            llvm::BasicBlock* lte1 = llvm::BasicBlock::Create(C, "lte1", base);
            llvm::BasicBlock* post = llvm::BasicBlock::Create(C, "post", base);
            r1 = binaryOperator(LLVMType<float>::get(C, 1.0f), in.GetValue(), ast::tokens::LESSTHANOREQUAL, B);
            B.CreateCondBr(r1, lte1, post);
            B.SetInsertPoint(lte1);
            {
                llvm::Value* d = ftx8 ?
                    LLVMType<uint8_t>::get(C, std::numeric_limits<uint8_t>::max()) :
                    LLVMType<uint16_t>::get(C, std::numeric_limits<uint16_t>::max());
                B.CreateStore(d, out.GetValue());
                B.CreateBr(fin);
            }

            B.SetInsertPoint(post);
            {
                llvm::Value* d = ftx8 ?
                    LLVMType<float>::get(C, float(std::numeric_limits<uint8_t>::max())) :
                    LLVMType<float>::get(C, float(std::numeric_limits<uint16_t>::max()));
                d = binaryOperator(in.GetValue(), d, ast::tokens::MULTIPLY, B);
                d = B.CreateFPToUI(d, args[0].GetUnderlyingType());
                B.CreateStore(d, out.GetValue());
                B.CreateBr(fin);
            }
        }

        B.SetInsertPoint(fin);
        return Value::Invalid();
    };

    auto generate_vec =
        [OneByte, IsPositionRange](
            const NativeArguments& args,
            llvm::IRBuilder<>& B) -> Value
    {
        OPENVDB_ASSERT(args.size() == 2);

        std::vector<Value> out, in;
        args[0].ArrayToScalars(B, out, /*load*/false);
        args[1].ArrayToScalars(B, in,  /*load*/false);
        OPENVDB_ASSERT(in.size() >= 3);
        OPENVDB_ASSERT(out.size() == in.size());

        auto F = axfxptencode(OneByte, IsPositionRange);
        for (size_t i = 0; i < in.size(); ++i)
        {
            Arguments fxptencodeargs;
            fxptencodeargs.AddArg(out[i]);
            fxptencodeargs.AddArg(in[i]);
            F->execute(fxptencodeargs, B);
        }
        return Value::Invalid();
    };

    if (OneByte) {
        return FunctionBuilder(IsPositionRange ? "__prfxpt8encode" : "__ufxpt8encode")
            .addSignature<void(uint8_t*, float*)>(generate)
            .addSignature<void(openvdb::math::Vec3<uint8_t>*,openvdb::math::Vec3<float>*)>(generate_vec)
            .get();
    }
    else {
        return FunctionBuilder(IsPositionRange ? "__prfxpt16encode" : "__ufxpt16encode")
            .addSignature<void(uint16_t*, float*)>(generate)
            .addSignature<void(openvdb::math::Vec3<uint16_t>*,openvdb::math::Vec3<float>*)>(generate_vec)
            .get();
    }
}

/// @note  For some reason templating axfxptdecode/axfxptencode with a bool
///   doesn't compile i.e. template <IsPositionRange>
inline FunctionGroup::UniquePtr axufxpt8decode()   { return axfxptdecode(true, false); }
inline FunctionGroup::UniquePtr axufxpt8encode()   { return axfxptencode(true, false); }
inline FunctionGroup::UniquePtr axprfxpt8decode()  { return axfxptdecode(true, true); }
inline FunctionGroup::UniquePtr axprfxpt8encode()  { return axfxptencode(true, true); }

inline FunctionGroup::UniquePtr axufxpt16decode()  { return axfxptdecode(false, false); }
inline FunctionGroup::UniquePtr axufxpt16encode()  { return axfxptencode(false, false); }
inline FunctionGroup::UniquePtr axprfxpt16decode() { return axfxptdecode(false, true); }
inline FunctionGroup::UniquePtr axprfxpt16encode() { return axfxptencode(false, true); }

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

const CodecTypeMap& getCodecTypeMap()
{
    // Initialise the static codec registry of supported types.
    // This can easily be exposed to users so they can write their own codecs,
    // but it would turn into either another static mutex regitry or another
    // object that would have to be passed from the codegen to the executables.
    // When we have AX pipelines we should expose this

    // @note  This should really be another static registry which mirrors
    //   whatever is registered in the AttributeArray registry. This is easy to
    //   do but required changes to the exposed AttributeArray API. To be done
    //   in a separate change set.

    static std::array<Codec::UniquePtr, 5> codecs {
        std::make_unique<Codec>(axtrncencode(), axtrncdecode(), 1<<0),
        std::make_unique<Codec>(axufxpt8encode(), axufxpt8decode(), 1<<1),
        std::make_unique<Codec>(axufxpt16encode(), axufxpt16decode(), 1<<2),
        std::make_unique<Codec>(axprfxpt8encode(), axprfxpt8decode(), 1<<3),
        std::make_unique<Codec>(axprfxpt16encode(), axprfxpt16decode(), 1<<4),
    };

    // If on X86, see if the hardware supports f16c. For other platforms we
    // currently assume hardware support for half/float conversion. This only
    // applies to the truncate codec.
    // @todo  Add software support. Will be simpler with AX function support.
    static bool HasF16C =
        ax::x86::CheckX86Feature("f16c") != ax::x86::CpuFlagStatus::Unsupported;

    static auto GetFloatCodecs = []() -> CodecNameMap {
        if (HasF16C) {
            return {
                { points::TruncateCodec::name(),                             codecs[0].get() },
                { points::FixedPointCodec<true, points::UnitRange>::name(),  codecs[1].get() },
                { points::FixedPointCodec<false, points::UnitRange>::name(), codecs[2].get() }
            };
        }
        else {
            return {
                { points::FixedPointCodec<true, points::UnitRange>::name(),  codecs[1].get() },
                { points::FixedPointCodec<false, points::UnitRange>::name(), codecs[2].get() }
            };
        }
    };

    static auto GetVectorCodecs = []() -> CodecNameMap {
        if (HasF16C) {
            return {
                { points::TruncateCodec::name(),                                 codecs[0].get() },
                { points::FixedPointCodec<true, points::UnitRange>::name(),      codecs[1].get() },
                { points::FixedPointCodec<false, points::UnitRange>::name(),     codecs[2].get() },
                { points::FixedPointCodec<true, points::PositionRange>::name(),  codecs[3].get() },
                { points::FixedPointCodec<false, points::PositionRange>::name(), codecs[4].get() }
            };
        }
        else {
            return {
                { points::FixedPointCodec<true, points::UnitRange>::name(),      codecs[1].get() },
                { points::FixedPointCodec<false, points::UnitRange>::name(),     codecs[2].get() },
                { points::FixedPointCodec<true, points::PositionRange>::name(),  codecs[3].get() },
                { points::FixedPointCodec<false, points::PositionRange>::name(), codecs[4].get() }
            };
        }
    };

    static CodecTypeMap map {
        {
            ast::tokens::FLOAT, GetFloatCodecs()
        },
        {
            ast::tokens::VEC3F, GetVectorCodecs()
        },
    };

    return map;
}

llvm::Type* Codec::findReturnTypeFromArg(const codegen::FunctionGroup* const group, llvm::Type* arg) const
{
    OPENVDB_ASSERT(!arg->isPointerTy());
    const auto& functions = group->list();
    codegen::ArgInfoVector types;
    for (const auto& F : functions) {
        types.clear();
        F->types(types, arg->getContext());
        OPENVDB_ASSERT(types.size() == 2);
        if (types[1].GetUnderlyingType() != arg) continue;
        return types[0].GetUnderlyingType();
    }
    // no supported conversion
    return nullptr;
}

const Codec* getCodec(const ast::tokens::CoreType type, const std::string& name)
{
    const CodecTypeMap& map = getCodecTypeMap();

    auto typeiter = map.find(type);
    if (typeiter != map.cend()) {
        auto iter = typeiter->second.find(name);
        if (iter != typeiter->second.cend()) {
            return iter->second;
        }
    }
    return nullptr;
}

const CodecNameMap* getTypeSupportedCodecs(const ast::tokens::CoreType type)
{
    const CodecTypeMap& map = getCodecTypeMap();

    auto typeiter = map.find(type);
    if (typeiter != map.cend()) {
        return &(typeiter->second);
    }
    return nullptr;
}


} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

