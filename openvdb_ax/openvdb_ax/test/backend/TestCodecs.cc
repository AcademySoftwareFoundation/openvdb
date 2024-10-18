// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "util.h"

#include <openvdb_ax/codegen/Types.h>
#include <openvdb_ax/codegen/Codecs.h>
#include <openvdb_ax/util/x86.h>

#include <openvdb/points/AttributeArray.h> // for native codec types

#include <gtest/gtest.h>

#include <limits>

using namespace openvdb;
using namespace openvdb::ax;
using namespace openvdb::ax::codegen;
using namespace openvdb::points;
using HalfTy = openvdb::math::half;

struct UFxpt8   { static const bool OneByte = true;  using type = FixedPointCodec<OneByte, UnitRange>; };
struct UFxpt16  { static const bool OneByte = false; using type = FixedPointCodec<OneByte, UnitRange>; };
struct PRFxpt8  { static const bool OneByte = true;  using type = FixedPointCodec<OneByte, PositionRange>; };
struct PRFxpt16 { static const bool OneByte = false; using type = FixedPointCodec<OneByte, PositionRange>; };

class TestCodecs : public ::testing::Test
{
};

inline const Codec*
getCodecByCodecName(const std::string& name)
{
    std::vector<const Codec*> codecs;
    const auto& map = getCodecTypeMap();
    for (const auto& typemap : map) {
        for (const auto& nameToCodec : typemap.second) {
            if (nameToCodec.first == name) {
                return nameToCodec.second;
            }
        }
    }
    return nullptr;
}

TEST_F(TestCodecs, testRegisteredCodecs)
{
    // For each codec, verify the way its been setup. Really this should be
    // enforced as part of the API but the majority of the setup code is internal.

    llvm::LLVMContext C;
#if LLVM_VERSION_MAJOR >= 15
    // This will not work from LLVM 16. We'll need to fix this
    // https://llvm.org/docs/OpaquePointers.html
    C.setOpaquePointers(false);
#endif

    // Get all unique registered codecs
    std::set<const Codec*> codecs;
    const auto& map = getCodecTypeMap();

    for (const auto& typemap : map) {
        for (const auto& nameToCodec : typemap.second) {
            codecs.insert(nameToCodec.second);
        }
    }

    size_t count = 5;
#if defined(__i386__) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(_M_X64)
    if (x86::CheckX86Feature("f16c") == x86::CpuFlagStatus::Unsupported) count = 4;
#endif

    // currently only 5 codecs are registered by default
    ASSERT_EQ(codecs.size(), count);

    //  for each codec, check:
    //    make sure the codecs flags are unique
    //    make sure the encoder/decoder functions have unique signatures

    std::set<uint32_t> flags;
    for (const Codec* codec : codecs) {
        ASSERT_TRUE(!flags.count(codec->flag()));
        flags.insert(codec->flag());
    }

    //

    for (const Codec* codec : codecs) {
        const codegen::FunctionGroup* encoder = codec->encoder();
        const codegen::FunctionGroup* decoder = codec->decoder();
        ASSERT_TRUE(encoder);
        ASSERT_TRUE(decoder);
        ASSERT_TRUE(!encoder->list().empty());
        ASSERT_TRUE(!decoder->list().empty());

        std::set<std::vector<llvm::Type*>> decoderSignatures, encoderSignatures;

        for (const auto& F : decoder->list()) {
            // check the function takes 2 arguments (in/out)
            // @note  This could change in the future e.g. a value is returned
            ASSERT_EQ(F->size(), size_t(2)); // input/output
            std::vector<llvm::Type*> types;
            llvm::Type* ret = F->types(types, C);
            // currently expect codecs to ret void
            ASSERT_EQ(ret, codegen::LLVMType<void>::get(C));
            // signature should be unqiue
            ASSERT_TRUE(!decoderSignatures.count(types));
            decoderSignatures.insert(types);
        }

        for (const auto& F : encoder->list()) {
            // check the function takes 2 arguments (in/out)
            // @note  This could change in the future e.g. a value is returned
            ASSERT_EQ(F->size(), size_t(2)); // input/output
            std::vector<llvm::Type*> types;
            llvm::Type* ret = F->types(types, C);
            // currently expect codecs to ret void
            ASSERT_EQ(ret, codegen::LLVMType<void>::get(C));
            // signature should be unqiue
            ASSERT_TRUE(!encoderSignatures.count(types));
            encoderSignatures.insert(types);
        }

        ASSERT_TRUE(!encoderSignatures.empty());
        ASSERT_TRUE(!decoderSignatures.empty());
        ASSERT_EQ(decoderSignatures.size(), encoderSignatures.size());

        // check signatures have unique input/output types
        // @note  This is necessary so that the IR knows what type to expect for a given input

        std::vector<std::vector<llvm::Type*>> copy(decoderSignatures.size());
        std::copy(decoderSignatures.begin(), decoderSignatures.end(), copy.begin());

        for (size_t i = 0; i < copy.size(); ++i) {
            const auto types = copy.back();
            const llvm::Type* first = types[0];
            const llvm::Type* second = types[1];
            copy.pop_back();
            for (const auto& remaining : copy) {
                ASSERT_TRUE(first != remaining[0]);
                ASSERT_TRUE(second != remaining[1]);
            }
        }

        // check signatures have unique input/output types
        // @note  This is necessary so that the IR knows what type to expect for a given input

        copy.resize(decoderSignatures.size());
        std::copy(encoderSignatures.begin(), encoderSignatures.end(), copy.begin());

        for (size_t i = 0; i < copy.size(); ++i) {
            const auto types = copy.back();
            const llvm::Type* first = types[0];
            const llvm::Type* second = types[1];
            copy.pop_back();
            for (const auto& remaining : copy) {
                ASSERT_TRUE(first != remaining[0]);
                ASSERT_TRUE(second != remaining[1]);
            }
        }

        // Check that every decoder has a matching encoder signature

        for (const auto& types : decoderSignatures) {
            std::vector<llvm::Type*> rev = types;
            std::reverse(rev.begin(), rev.end());
            ASSERT_TRUE(encoderSignatures.find(rev) != encoderSignatures.end());
        }
    }
}

TEST_F(TestCodecs, testTruncateCodec)
{
#if defined(__i386__) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(_M_X64)
    if (x86::CheckX86Feature("f16c") == x86::CpuFlagStatus::Unsupported) return;
#endif

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();

    const Codec* const codec =
        getCodecByCodecName(TruncateCodec::name());
    ASSERT_TRUE(codec);

    llvm::Type* floatty = codegen::LLVMType<float>::get(C);
    llvm::Type* vfloatty = codegen::LLVMType<math::Vec3<float>>::get(C);
    llvm::Type* halfty = codegen::LLVMType<HalfTy>::get(C);
    llvm::Type* vhalfty = codegen::LLVMType<math::Vec3<HalfTy>>::get(C);

    ASSERT_EQ(halfty,  codec->decodedToEncoded(ast::tokens::CoreType::FLOAT, C));
    ASSERT_EQ(vhalfty, codec->decodedToEncoded(ast::tokens::CoreType::VEC3F, C));
    ASSERT_EQ(floatty,  codec->encodedToDecoded(halfty));
    ASSERT_EQ(vfloatty, codec->encodedToDecoded(vhalfty));

    // JIT the codec and test the IR

    const codegen::FunctionGroup* encoder = codec->encoder();
    const codegen::FunctionGroup* decoder = codec->decoder();
    ASSERT_TRUE(encoder);
    ASSERT_TRUE(decoder);
    ASSERT_TRUE(!encoder->list().empty());
    ASSERT_TRUE(!decoder->list().empty());

    for (auto& F : encoder->list()) ASSERT_TRUE(F->create(M));
    for (auto& F : decoder->list()) ASSERT_TRUE(F->create(M));

    auto EE = state.EE();
    ASSERT_TRUE(EE);

    // test truncate encoders

    using  FloatToHalf = void(HalfTy*, float*);
    using VFloatToHalf = void(math::Vec3<HalfTy>*, math::Vec3<float>*);

    const std::vector<float> floatInputs {
        1.0f, 0.0f, -1.0f,
        0.5f, 0.13454f, -0.98781f,
        1.0431e-6f, 1.0431e+6f, std::numeric_limits<float>::max(),
        313.33f, std::numeric_limits<float>::min(), std::numeric_limits<float>::lowest()
    };

    const std::vector<math::Vec3<float>> vfloatInputs {
        math::Vec3<float>(floatInputs.data() + 0),
        math::Vec3<float>(floatInputs.data() + 3),
        math::Vec3<float>(floatInputs.data() + 6),
        math::Vec3<float>(floatInputs.data() + 9)
    };

    {
        const int64_t address = EE->getFunctionAddress(encoder->list()[0]->symbol());
        ASSERT_TRUE(address);
        const auto truncEncodeFloatToHalf = reinterpret_cast<std::add_pointer<FloatToHalf>::type>(address);

        HalfTy result1, result2;

        for (float input : floatInputs)
        {
            const float tmp = input;
            truncEncodeFloatToHalf(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            TruncateCodec::encode<HalfTy, float>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    {
        const int64_t address = EE->getFunctionAddress(encoder->list()[5]->symbol());
        ASSERT_TRUE(address);
        const auto truncEncodeVecFloatToHalf = reinterpret_cast<std::add_pointer<VFloatToHalf>::type>(address);

        math::Vec3<HalfTy> result1, result2;

        for (math::Vec3<float> input : vfloatInputs)
        {
            const math::Vec3<float> tmp(input);
            truncEncodeVecFloatToHalf(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            TruncateCodec::encode<math::Vec3<HalfTy>, math::Vec3<float>>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    // test truncate decoders

    using  HalfToFloat = void(float*, HalfTy*);
    using VHalfToFloat = void(math::Vec3<float>*, math::Vec3<HalfTy>*);

    const std::vector<HalfTy> halfInputs {
        1.0f, 0.0f, -1.0f,
        0.5f, 0.13454f, -0.98781f,
        1.0431e-6f, 1.0431e+6f, std::numeric_limits<HalfTy>::max(),
        313.33f, std::numeric_limits<HalfTy>::min(), std::numeric_limits<HalfTy>::lowest()
    };

    const std::vector<math::Vec3<HalfTy>> vhalfInputs {
        math::Vec3<HalfTy>(halfInputs.data() + 0),
        math::Vec3<HalfTy>(halfInputs.data() + 3),
        math::Vec3<HalfTy>(halfInputs.data() + 6),
        math::Vec3<HalfTy>(halfInputs.data() + 9)
    };

    {
        const int64_t address = EE->getFunctionAddress(decoder->list()[0]->symbol());
        ASSERT_TRUE(address);
        const auto truncDecodeHalfToFloat = reinterpret_cast<std::add_pointer<HalfToFloat>::type>(address);

        float result1, result2;

        for (HalfTy input : halfInputs)
        {
            const HalfTy tmp = input;
            truncDecodeHalfToFloat(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            TruncateCodec::encode<float, HalfTy>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    {
        const int64_t address = EE->getFunctionAddress(decoder->list()[5]->symbol());
        ASSERT_TRUE(address);
        const auto truncDecodeVecHalfToFloat = reinterpret_cast<std::add_pointer<VHalfToFloat>::type>(address);

        math::Vec3<float> result1, result2;

        for (math::Vec3<HalfTy> input : vhalfInputs)
        {
            const math::Vec3<HalfTy> tmp(input);
            truncDecodeVecHalfToFloat(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            TruncateCodec::encode<math::Vec3<float>, math::Vec3<HalfTy>>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }
}

template <typename FxptCodecT>
void testFxptCodec()
{
    static const bool OneByte = FxptCodecT::OneByte;
    using IntT = typename std::conditional<OneByte, uint8_t, uint16_t>::type;
    using FixedPointCodecType = typename FxptCodecT::type;

    unittest_util::LLVMState state;
    llvm::LLVMContext& C = state.context();
    llvm::Module& M = state.module();

    const Codec* const codec = getCodecByCodecName(FixedPointCodecType::name());
    ASSERT_TRUE(codec);

    llvm::Type* uintty = OneByte ? codegen::LLVMType<uint8_t>::get(C) : codegen::LLVMType<uint16_t>::get(C);
    llvm::Type* vuintty = OneByte ? codegen::LLVMType<math::Vec3<uint8_t>>::get(C) : codegen::LLVMType<math::Vec3<uint16_t>>::get(C);
    llvm::Type* floatty = codegen::LLVMType<float>::get(C);
    llvm::Type* vfloatty = codegen::LLVMType<math::Vec3<float>>::get(C);

    ASSERT_TRUE(nullptr == codec->decodedToEncoded(ast::tokens::CoreType::INT32, C));
    ASSERT_TRUE(nullptr == codec->decodedToEncoded(ast::tokens::CoreType::VEC2F, C));
    ASSERT_TRUE(nullptr == codec->decodedToEncoded(ast::tokens::CoreType::STRING, C));
    ASSERT_EQ(uintty,  codec->decodedToEncoded(ast::tokens::CoreType::FLOAT, C));
    ASSERT_EQ(vuintty, codec->decodedToEncoded(ast::tokens::CoreType::VEC3F, C));
    ASSERT_EQ(floatty,  codec->encodedToDecoded(uintty));
    ASSERT_EQ(vfloatty, codec->encodedToDecoded(vuintty));

    // JIT the codec and test the IR

    const codegen::FunctionGroup* encoder = codec->encoder();
    const codegen::FunctionGroup* decoder = codec->decoder();
    ASSERT_TRUE(encoder);
    ASSERT_TRUE(decoder);
    ASSERT_TRUE(!encoder->list().empty());
    ASSERT_TRUE(!decoder->list().empty());
    ASSERT_EQ(encoder->list().size(), size_t(2));
    ASSERT_EQ(decoder->list().size(), size_t(2));

    for (auto& F : encoder->list()) ASSERT_TRUE(F->create(M));
    for (auto& F : decoder->list()) ASSERT_TRUE(F->create(M));

    auto EE = state.EE();
    ASSERT_TRUE(EE);

    // test truncate encoders

    using  FloatToFxpt = void(*)(IntT*, float*);
    using VFloatToFxpt = void(*)(math::Vec3<IntT>*, math::Vec3<float>*);

    const std::vector<float> floatInputs {
        1.0f, 0.0f, -1.0f,
        0.5f, 0.20024414435034715f, -0.98781f,
        1e-3f, 0.2f, 0.6f,
        0.8f, 1.5f, -1.5f,
        100.0f, std::numeric_limits<float>::lowest(), -100.0f
    };

    const std::vector<math::Vec3<float>> vfloatInputs {
        math::Vec3<float>(floatInputs.data() + 0),
        math::Vec3<float>(floatInputs.data() + 3),
        math::Vec3<float>(floatInputs.data() + 6),
        math::Vec3<float>(floatInputs.data() + 9),
        math::Vec3<float>(floatInputs.data() + 12)
    };

    {
        const int64_t address = EE->getFunctionAddress(encoder->list()[0]->symbol());
        ASSERT_TRUE(address);
        const auto fxptEncodeFloat = reinterpret_cast<FloatToFxpt>(address);

        IntT result1, result2;

        for (float input : floatInputs)
        {
            const float tmp = input;
            fxptEncodeFloat(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            FixedPointCodecType::template encode<IntT, float>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    {
        const int64_t address = EE->getFunctionAddress(encoder->list()[1]->symbol());
        ASSERT_TRUE(address);
        const auto fxptEncodeVFloat = reinterpret_cast<VFloatToFxpt>(address);

        math::Vec3<IntT> result1, result2;

        for (math::Vec3<float> input : vfloatInputs)
        {
            const math::Vec3<float> tmp(input);
            fxptEncodeVFloat(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            FixedPointCodecType::template encode<math::Vec3<IntT>, math::Vec3<float>>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    // test truncate decoders

    using  FxptToFloat = void(*)(float*, IntT*);
    using VFxptToFloat = void(*)(math::Vec3<float>*, math::Vec3<IntT>*);

    std::vector<IntT> uintInputs = {
        0, 1, 2,
        100, 200, 212,
        10, 215, 94,
        54, std::numeric_limits<IntT>::max(), 199
    };

    if (OneByte) uintInputs.insert(uintInputs.end(), { IntT(13132), IntT(31334), IntT(60000) });
    else         uintInputs.insert(uintInputs.end(), { 111, 222, 18 });

    const std::vector<math::Vec3<IntT>> vuintInputs {
        math::Vec3<IntT>(uintInputs.data() + 0),
        math::Vec3<IntT>(uintInputs.data() + 3),
        math::Vec3<IntT>(uintInputs.data() + 6),
        math::Vec3<IntT>(uintInputs.data() + 9),
        math::Vec3<IntT>(uintInputs.data() + 12)
    };

    {
        const int64_t address = EE->getFunctionAddress(decoder->list()[0]->symbol());
        ASSERT_TRUE(address);
        const auto fxptDecodeUint8 = reinterpret_cast<FxptToFloat>(address);

        float result1, result2;

        for (IntT input : uintInputs)
        {
            const IntT tmp = input;
            fxptDecodeUint8(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            FixedPointCodecType::template decode<IntT, float>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }

    {
        const int64_t address = EE->getFunctionAddress(decoder->list()[1]->symbol());
        ASSERT_TRUE(address);
        const auto fxptDecodeVuint8 = reinterpret_cast<VFxptToFloat>(address);

        math::Vec3<float> result1, result2;

        for (math::Vec3<IntT> input : vuintInputs)
        {
            const math::Vec3<IntT> tmp(input);
            fxptDecodeVuint8(&result1, &input);
            ASSERT_EQ(input, tmp); // doesn't change
            FixedPointCodecType::template decode<math::Vec3<IntT>, math::Vec3<float>>(input, result2);
            ASSERT_EQ(result2, result1);
        }
    }
}

TEST_F(TestCodecs, testFxptCodecUFxpt8)
{
    testFxptCodec<UFxpt8>();
}

TEST_F(TestCodecs, testFxptCodecUFxpt16)
{
    testFxptCodec<UFxpt16>();
}

TEST_F(TestCodecs, testFxptCodecPRFxpt8)
{
    testFxptCodec<PRFxpt8>();
}

TEST_F(TestCodecs, testFxptCodecPRFxpt16)
{
    testFxptCodec<PRFxpt16>();
}
