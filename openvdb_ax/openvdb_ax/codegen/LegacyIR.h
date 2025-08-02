// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/LegacyIR.h
///
/// @authors Nick Avramoussis
///
/// @brief  Legacy IR utilities for LLVM_VERSION_MAJOR <= 15
///

#ifndef OPENVDB_AX_CODEGEN_LEGACY_IR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_LEGACY_IR_HAS_BEEN_INCLUDED

#include <llvm/Config/llvm-config.h>

#if LLVM_VERSION_MAJOR <= 15

#include "Types.h"
#include "Utils.h"
#include <openvdb/version.h>
#include <openvdb/util/Assert.h>
#include <llvm/IR/IRBuilder.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

OPENVDB_DEPRECATED
inline auto ir_load(llvm::IRBuilder<>& B, llvm::Value* ptr, const char* Name = "")
{
    OPENVDB_ASSERT(ptr);
    OPENVDB_ASSERT(ptr->getType()->isPointerTy());
    return B.CreateLoad(ptr->getType()->getPointerElementType(), ptr, Name);
}

OPENVDB_DEPRECATED
inline auto ir_gep(llvm::IRBuilder<>& B,
    llvm::Value* ptr, llvm::ArrayRef<llvm::Value*> IdxList, const char* Name = "")
{
    OPENVDB_ASSERT(ptr);
    OPENVDB_ASSERT(ptr->getType()->getScalarType());
    OPENVDB_ASSERT(ptr->getType()->getScalarType()->isPointerTy());
    return B.CreateGEP(ptr->getType()->getScalarType()->getPointerElementType(),
                 ptr, IdxList, Name);
#
}

OPENVDB_DEPRECATED
inline auto ir_constgep2_64(llvm::IRBuilder<>& B,
    llvm::Value* ptr, uint64_t Idx0, uint64_t Idx1, const char* Name = "")
{
    OPENVDB_ASSERT(ptr);
    OPENVDB_ASSERT(ptr->getType()->getScalarType());
    OPENVDB_ASSERT(ptr->getType()->getScalarType()->isPointerTy());
    return B.CreateConstGEP2_64(
        ptr->getType()->getScalarType()->getPointerElementType(), ptr, Idx0,
        Idx1, Name);
}

OPENVDB_DEPRECATED
inline auto ir_constinboundsgep2_64(llvm::IRBuilder<>& B,
    llvm::Value* ptr, uint64_t Idx0, uint64_t Idx1, const char* Name = "")
{
    OPENVDB_ASSERT(ptr);
    OPENVDB_ASSERT(ptr->getType()->getScalarType());
    OPENVDB_ASSERT(ptr->getType()->getScalarType()->isPointerTy());
    return B.CreateConstInBoundsGEP2_64(
        ptr->getType()->getScalarType()->getPointerElementType(), ptr, Idx0,
        Idx1, Name);
}

/// @brief  Return the base llvm value which is being pointed to through
///         any number of layered pointers.
/// @note   This function does not check for cyclical pointer dependencies
///
/// @param  type  A llvm pointer type to traverse
///
OPENVDB_DEPRECATED
inline llvm::Type*
getBaseContainedType(llvm::Type* const type)
{
    llvm::Type* elementType = type;
    while (elementType->isPointerTy()) {
        elementType = elementType->getContainedType(0);
    }
    return elementType;
}

/// @brief  Casts an array to another array of equal size but of a different element
///         type. Both source and target array element types must be scalar types.
///         The source array llvm Value should be a pointer to the array to cast.
///
/// @param ptrToArray         A llvm value which is a pointer to a llvm array
/// @param targetElementType  The target llvm scalar type to convert each element
///                           of the input array
/// @param builder            The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline llvm::Value*
arrayCast(llvm::Value* ptrToArray,
          llvm::Type* targetElementType,
          llvm::IRBuilder<>& builder)
{
    OPENVDB_ASSERT(targetElementType && (targetElementType->isIntegerTy() ||
        targetElementType->isFloatingPointTy()) &&
        "Target element type is not a scalar type");
    OPENVDB_ASSERT(ptrToArray && ptrToArray->getType()->isPointerTy() &&
        "Input to arrayCast is not a pointer type.");

    llvm::Type* arrayType = ptrToArray->getType()->getContainedType(0);
    OPENVDB_ASSERT(arrayType && llvm::isa<llvm::ArrayType>(arrayType));

    // getArrayElementType() calls getContainedType(0)
    llvm::Type* sourceElementType = arrayType->getArrayElementType();
    OPENVDB_ASSERT(sourceElementType && (sourceElementType->isIntegerTy() ||
        sourceElementType->isFloatingPointTy()) &&
        "Source element type is not a scalar type");

    if (sourceElementType == targetElementType) return ptrToArray;

    CastFunction llvmCastFunction = llvmArithmeticConversion(sourceElementType, targetElementType);

    const size_t elementSize = arrayType->getArrayNumElements();
    llvm::Value* targetArray =
        insertStaticAlloca(builder,
            llvm::ArrayType::get(targetElementType, elementSize));

    for (size_t i = 0; i < elementSize; ++i) {
        llvm::Value* target = builder.CreateConstGEP2_64(targetArray->getType()->getScalarType()->getPointerElementType(), targetArray, 0, i);
        llvm::Value* source = builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, i);
        source = builder.CreateLoad(source->getType()->getPointerElementType(), source);
        source = llvmCastFunction(builder, source, targetElementType);
        builder.CreateStore(source, target);
    }

    return targetArray;
}

/// @brief  Unpack a particular element of an array and return a pointer to that element
///         The provided llvm Value is expected to be a pointer to an array
///
/// @param ptrToArray  A llvm value which is a pointer to a llvm array
/// @param index       The index at which to access the array
/// @param builder     The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline llvm::Value*
arrayIndexUnpack(llvm::Value* ptrToArray,
                 const int16_t index,
                 llvm::IRBuilder<>& builder)
{
    return builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, index);
}

/// @brief  Unpack an array type into llvm Values which represent all its elements
///         The provided llvm Value is expected to be a pointer to an array
///         If loadElements is true, values will store loaded llvm values instead
///         of pointers to the array elements
///
/// @param ptrToArray    A llvm value which is a pointer to a llvm array
/// @param values        A vector of llvm values where to store the array elements
/// @param builder       The current llvm IRBuilder
/// @param loadElements  Whether or not to load each array element into a register
///
OPENVDB_DEPRECATED
inline void
arrayUnpack(llvm::Value* ptrToArray,
            std::vector<llvm::Value*>& values,
            llvm::IRBuilder<>& builder,
            const bool loadElements = false)
{
    const size_t elements =
        ptrToArray->getType()->getContainedType(0)->getArrayNumElements();

    values.reserve(elements);
    for (size_t i = 0; i < elements; ++i) {
        llvm::Value* value = builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, i);
        if (loadElements) value = builder.CreateLoad(value->getType()->getPointerElementType(), value);
        values.push_back(value);
    }
}

/// @brief  Unpack the first three elements of an array.
///         The provided llvm Value is expected to be a pointer to an array
/// @note   The elements are note loaded
///
/// @param ptrToArray    A llvm value which is a pointer to a llvm array
/// @param value1        The first array value
/// @param value2        The second array value
/// @param value3        The third array value
/// @param builder       The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline void
array3Unpack(llvm::Value* ptrToArray,
             llvm::Value*& value1,
             llvm::Value*& value2,
             llvm::Value*& value3,
             llvm::IRBuilder<>& builder)
{
    OPENVDB_ASSERT(ptrToArray && ptrToArray->getType()->isPointerTy() &&
        "Input to array3Unpack is not a pointer type.");

    value1 = builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, 0);
    value2 = builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, 1);
    value3 = builder.CreateConstGEP2_64(ptrToArray->getType()->getScalarType()->getPointerElementType(), ptrToArray, 0, 2);
}

/// @brief  Pack three values into a new array and return a pointer to the
///         newly allocated array. If the values are of a mismatching type,
///         the highets order type is uses, as defined by typePrecedence. All
///         llvm values are expected to a be a loaded scalar type
///
/// @param value1   The first array value
/// @param value2   The second array value
/// @param value3   The third array value
/// @param builder  The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline llvm::Value*
array3Pack(llvm::Value* value1,
           llvm::Value* value2,
           llvm::Value* value3,
           llvm::IRBuilder<>& builder)
{
    llvm::Type* type = typePrecedence(value1->getType(), value2->getType());
    type = typePrecedence(type, value3->getType());

    value1 = arithmeticConversion(value1, type, builder);
    value2 = arithmeticConversion(value2, type, builder);
    value3 = arithmeticConversion(value3, type, builder);

    llvm::Type* vectorType = llvm::ArrayType::get(type, 3);
    llvm::Value* vector = insertStaticAlloca(builder, vectorType);

    llvm::Value* e1 = builder.CreateConstGEP2_64(vector->getType()->getScalarType()->getPointerElementType(), vector, 0, 0);
    llvm::Value* e2 = builder.CreateConstGEP2_64(vector->getType()->getScalarType()->getPointerElementType(), vector, 0, 1);
    llvm::Value* e3 = builder.CreateConstGEP2_64(vector->getType()->getScalarType()->getPointerElementType(), vector, 0, 2);

    builder.CreateStore(value1, e1);
    builder.CreateStore(value2, e2);
    builder.CreateStore(value3, e3);

    return vector;
}

/// @brief  Pack a loaded llvm scalar value into a new array of a specified
///         size and return a pointer to the newly allocated array. Each element
///         of the new array will have the value of the given scalar
///
/// @param value    The uniform scalar llvm value to pack into the array
/// @param builder  The current llvm IRBuilder
/// @param size     The size of the newly allocated array
///
OPENVDB_DEPRECATED
inline llvm::Value*
arrayPack(llvm::Value* value,
          llvm::IRBuilder<>& builder,
          const size_t size = 3)
{
    OPENVDB_ASSERT(value && (value->getType()->isIntegerTy() ||
        value->getType()->isFloatingPointTy()) &&
        "value type is not a scalar type");

    llvm::Type* type = value->getType();
    llvm::Value* array =
        insertStaticAlloca(builder,
            llvm::ArrayType::get(type, size));

    for (size_t i = 0; i < size; ++i) {
        llvm::Value* element = builder.CreateConstGEP2_64(array->getType()->getScalarType()->getPointerElementType(), array, 0, i);
        builder.CreateStore(value, element);
    }

    return array;
}

/// @brief  Pack a vector of loaded llvm scalar values into a new array of
///         equal size and return a pointer to the newly allocated array.
///
/// @param values   A vector of loaded llvm scalar values to pack
/// @param builder  The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline llvm::Value*
arrayPack(const std::vector<llvm::Value*>& values,
          llvm::IRBuilder<>& builder)
{
    llvm::Type* type = values.front()->getType();
    llvm::Value* array = insertStaticAlloca(builder,
        llvm::ArrayType::get(type, values.size()));

    size_t idx = 0;
    for (llvm::Value* const& value : values) {
        llvm::Value* element = builder.CreateConstGEP2_64(array->getType()->getScalarType()->getPointerElementType(), array, 0, idx++);
        builder.CreateStore(value, element);
    }

    return array;
}

/// @brief  Pack a vector of loaded llvm scalar values into a new array of
///         equal size and return a pointer to the newly allocated array.
///         arrayPackCast first checks all the contained types in values
///         and casts all types to the highest order type present. All llvm
///         values in values are expected to be loaded scalar types
///
/// @param values   A vector of loaded llvm scalar values to pack
/// @param builder  The current llvm IRBuilder
///
OPENVDB_DEPRECATED
inline llvm::Value*
arrayPackCast(std::vector<llvm::Value*>& values,
              llvm::IRBuilder<>& builder)
{
    // get the highest order type present

    llvm::Type* type = LLVMType<bool>::get(builder.getContext());
    for (llvm::Value* const& value : values) {
        type = typePrecedence(type, value->getType());
    }

    // convert all to this type

    for (llvm::Value*& value : values) {
        value = arithmeticConversion(value, type, builder);
    }

    return arrayPack(values, builder);
}

OPENVDB_DEPRECATED
inline llvm::Value*
scalarToMatrix(llvm::Value* scalar,
     llvm::IRBuilder<>& builder,
     const size_t dim = 3)
{
    OPENVDB_ASSERT(scalar && (scalar->getType()->isIntegerTy() ||
        scalar->getType()->isFloatingPointTy()) &&
        "value type is not a scalar type");

    llvm::Type* type = scalar->getType();
    llvm::Value* array =
        insertStaticAlloca(builder,
            llvm::ArrayType::get(type, dim*dim));

    llvm::Value* zero = llvmConstant(0, type);
    for (size_t i = 0; i < dim*dim; ++i) {
        llvm::Value* m = ((i % (dim+1) == 0) ? scalar : zero);
        llvm::Value* element = builder.CreateConstGEP2_64(array->getType()->getScalarType()->getPointerElementType(), array, 0, i);
        builder.CreateStore(m, element);
    }

    return array;
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // LLVM_VERSION_MAJOR <= 15

#endif // OPENVDB_AX_CODEGEN_LEGACY_IR_HAS_BEEN_INCLUDED

