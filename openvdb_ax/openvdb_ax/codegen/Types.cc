// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/Types.cc
///
/// @authors Nick Avramoussis
///

#include "Types.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief Returns an llvm IntegerType given a requested size and context
/// @param size  The number of bits of the integer type
/// @param C     The LLVMContext to request the Type from.
///
llvm::IntegerType*
llvmIntType(const uint32_t size, llvm::LLVMContext& C)
{
    switch (size) {
        case 1 :  return llvm::cast<llvm::IntegerType>(LLVMType<bool>::get(C));
        case 8 :  return llvm::cast<llvm::IntegerType>(LLVMType<int8_t>::get(C));
        case 16 : return llvm::cast<llvm::IntegerType>(LLVMType<int16_t>::get(C));
        case 32 : return llvm::cast<llvm::IntegerType>(LLVMType<int32_t>::get(C));
        case 64 : return llvm::cast<llvm::IntegerType>(LLVMType<int64_t>::get(C));
        default : return llvm::Type::getIntNTy(C, size);
    }
}


/// @brief Returns an llvm floating point Type given a requested size and context
/// @param size  The size of the float to request, i.e. float - 32, double - 64 etc.
/// @param C     The LLVMContext to request the Type from.
///
llvm::Type*
llvmFloatType(const uint32_t size, llvm::LLVMContext& C)
{
    switch (size) {
        case 32 : return LLVMType<float>::get(C);
        case 64 : return LLVMType<double>::get(C);
        default : OPENVDB_THROW(AXCodeGenError,
            "Invalid float size requested from LLVM Context");
    }
}

/// @brief  Returns an llvm type representing a type defined by a string.
/// @note   For string types, this function returns the element type, not the
///         object type! The llvm type representing a char block of memory
///         is LLVMType<char*>::get(C);
/// @param type  The name of the type to request.
/// @param C     The LLVMContext to request the Type from.
///
llvm::Type*
llvmTypeFromToken(const ast::tokens::CoreType& type,
                  llvm::LLVMContext& C)
{
    switch (type) {
        case ast::tokens::BOOL    : return LLVMType<bool>::get(C);
        case ast::tokens::INT16   : return LLVMType<int16_t>::get(C);
        case ast::tokens::INT32   : return LLVMType<int32_t>::get(C);
        case ast::tokens::INT64   : return LLVMType<int64_t>::get(C);
        case ast::tokens::FLOAT   : return LLVMType<float>::get(C);
        case ast::tokens::DOUBLE  : return LLVMType<double>::get(C);
        case ast::tokens::VEC2I   : return LLVMType<int32_t[2]>::get(C);
        case ast::tokens::VEC2F   : return LLVMType<float[2]>::get(C);
        case ast::tokens::VEC2D   : return LLVMType<double[2]>::get(C);
        case ast::tokens::VEC3I   : return LLVMType<int32_t[3]>::get(C);
        case ast::tokens::VEC3F   : return LLVMType<float[3]>::get(C);
        case ast::tokens::VEC3D   : return LLVMType<double[3]>::get(C);
        case ast::tokens::VEC4I   : return LLVMType<int32_t[4]>::get(C);
        case ast::tokens::VEC4F   : return LLVMType<float[4]>::get(C);
        case ast::tokens::VEC4D   : return LLVMType<double[4]>::get(C);
        case ast::tokens::MAT3F   : return LLVMType<float[9]>::get(C);
        case ast::tokens::MAT3D   : return LLVMType<double[9]>::get(C);
        case ast::tokens::MAT4F   : return LLVMType<float[16]>::get(C);
        case ast::tokens::MAT4D   : return LLVMType<double[16]>::get(C);
        case ast::tokens::STRING  : return LLVMType<codegen::String>::get(C);
        case ast::tokens::UNKNOWN :
        default      :
            OPENVDB_THROW(AXCodeGenError,
                "Token type not recognised in request for LLVM type");
    }
}

ast::tokens::CoreType
tokenFromLLVMType(const llvm::Type* type)
{
    if (type->isPointerTy()) {
        type = type->getPointerElementType();
    }
    if (type->isIntegerTy(1))   return ast::tokens::BOOL;
    if (type->isIntegerTy(16))  return ast::tokens::INT16;
    if (type->isIntegerTy(32))  return ast::tokens::INT32;
    if (type->isIntegerTy(64))  return ast::tokens::INT64;
    if (type->isFloatTy())      return ast::tokens::FLOAT;
    if (type->isDoubleTy())     return ast::tokens::DOUBLE;
    if (type->isArrayTy()) {
        const ast::tokens::CoreType elementType =
            tokenFromLLVMType(type->getArrayElementType());
        const size_t size = type->getArrayNumElements();
        if (size == 2) {
            if (elementType == ast::tokens::INT32)   return ast::tokens::VEC2I;
            if (elementType == ast::tokens::FLOAT)   return ast::tokens::VEC2F;
            if (elementType == ast::tokens::DOUBLE)  return ast::tokens::VEC2D;
        }
        else if (size == 3) {
            if (elementType == ast::tokens::INT32)   return ast::tokens::VEC3I;
            if (elementType == ast::tokens::FLOAT)   return ast::tokens::VEC3F;
            if (elementType == ast::tokens::DOUBLE)  return ast::tokens::VEC3D;
        }
        else if (size == 4) {
            if (elementType == ast::tokens::INT32)   return ast::tokens::VEC4I;
            if (elementType == ast::tokens::FLOAT)   return ast::tokens::VEC4F;
            if (elementType == ast::tokens::DOUBLE)  return ast::tokens::VEC4D;
        }
        else if (size == 9) {
            if (elementType == ast::tokens::FLOAT)   return ast::tokens::MAT3F;
            if (elementType == ast::tokens::DOUBLE)  return ast::tokens::MAT3D;
        }
        else if (size == 16) {
            if (elementType == ast::tokens::FLOAT)   return ast::tokens::MAT4F;
            if (elementType == ast::tokens::DOUBLE)  return ast::tokens::MAT4D;
        }
    }
    if (type == LLVMType<codegen::String>::get(type->getContext())) {
        return ast::tokens::STRING;
    }
    return ast::tokens::UNKNOWN;
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

