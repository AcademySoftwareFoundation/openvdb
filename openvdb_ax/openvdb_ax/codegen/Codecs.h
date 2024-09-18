// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#ifndef OPENVDB_AX_CODEGEN_CODECS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_CODECS_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include "openvdb_ax/ast/Tokens.h"
#include "openvdb_ax/codegen/FunctionTypes.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace ax {
namespace codegen {

class Codec;

using CodecNameMap = std::map<const std::string, const Codec*>;
using CodecTypeMap = std::map<const ast::tokens::CoreType, CodecNameMap>;
using Codecs = std::vector<const Codec*>;

/// @brief Get the global codec map
OPENVDB_AX_API const CodecTypeMap& getCodecTypeMap();

/// @brief Get a specific codec. Returns a nullptr if no codec exists.
/// @param type  The type the codec encodes
/// @param name  The name of the codec
OPENVDB_AX_API const Codec* getCodec(const ast::tokens::CoreType type, const std::string& name);

/// @brief Get a specific set of codecs which encode a given type. Returns a
///   nullptr if no codec exists.
/// @param type  The type the codecs encode
OPENVDB_AX_API const CodecNameMap* getTypeSupportedCodecs(const ast::tokens::CoreType type);

class OPENVDB_AX_API Codec
{
public:
    using UniquePtr = std::unique_ptr<Codec>;

    Codec(codegen::FunctionGroup::UniquePtr encoder,
        codegen::FunctionGroup::UniquePtr decoder,
        uint32_t flag)
    : mEncoder(std::move(encoder))
    , mDecoder(std::move(decoder))
    , mFlag(flag) {
#ifndef NDEBUG
        OPENVDB_ASSERT(!mEncoder->list().empty());
        OPENVDB_ASSERT(!mDecoder->list().empty());
        OPENVDB_ASSERT(mEncoder->list().size() == mDecoder->list().size());
        for (const auto& F : mEncoder->list()) {
            OPENVDB_ASSERT(F->size() == 1 || F->size() == 2);
        }
#endif
    }

    /// @brief  Given a core type supported by the AX frontend, return a llvm
    ///   compatible type which represents how the core type is encoded in
    ///   memory.
    /// @return  A llvm type representing the encoded C type. Can be a nullptr
    ///   if this codec does not support the provided core type.
    llvm::Type* decodedToEncoded(const ast::tokens::CoreType& in, llvm::LLVMContext& C) const
    {
        // the input "decoded" type - unlike the encoded type, the decoded type
        // has to be available as an AX "CoreType" which is why this function
        // takes a CoreType in
        llvm::Type* type = codegen::llvmTypeFromToken(in, C);
        // For each encoder function in this codec, find the one which
        // one takes the provided "in" decoded type and return the type
        // of that function return signature
        llvm::Type* ret = findReturnTypeFromArg(this->encoder(), type->getPointerTo());
        return ret ? ret->getPointerElementType() : nullptr;
    }

    /// @brief  Given a llvm type, return a compatible llvm type which
    ///   represents how the provided type should be exposed to the AX frontend.
    /// @note  The return type is guaranteed to either be a supported CoreType
    ///   (such that ax::codegen::tokenFromLLVMType(in) returns a valid value)
    ///   or a nullptr.
    /// @return  A llvm type representing the decoded C type. Can be a nullptr
    ///   if this codec does not support the provided core type.
    llvm::Type* encodedToDecoded(llvm::Type* in) const
    {
        // For each decoder function in this codec, find the one which
        // one takes the provided "in" encoded type and return the type
        // of that function return signature
        if (!in->isPointerTy()) in = in->getPointerTo();
        llvm::Type* ret = findReturnTypeFromArg(this->decoder(), in);
        return ret ? ret->getPointerElementType() : nullptr;
    }

    const codegen::FunctionGroup* encoder() const { return mEncoder.get(); }
    const codegen::FunctionGroup* decoder() const { return mDecoder.get(); }
    inline uint32_t flag() const { return mFlag; }

private:
    llvm::Type* findReturnTypeFromArg(const codegen::FunctionGroup* const, llvm::Type*) const;

    const codegen::FunctionGroup::UniquePtr mEncoder;
    const codegen::FunctionGroup::UniquePtr mDecoder;
    const uint32_t mFlag;
};


} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_CODECS_HAS_BEEN_INCLUDED

