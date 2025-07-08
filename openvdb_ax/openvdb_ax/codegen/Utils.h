// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/Utils.h
///
/// @authors Nick Avramoussis
///
/// @brief  Utility code generation methods for performing various llvm
///   operations
///

#ifndef OPENVDB_AX_CODEGEN_UTILS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_UTILS_HAS_BEEN_INCLUDED

#include "Types.h"

#include "../ast/Tokens.h"
#include "../Exceptions.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_ostream.h> // llvm::errs()

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @note Function definitions for some types returned from automatic token to
/// llvm IR operations. See llvmArithmeticConversion and llvmBianryConversion

using CastFunction = std::function<llvm::Value*
    (llvm::IRBuilder<>&, llvm::Value*, llvm::Type*)>;

using BinaryFunction = std::function<llvm::Value*
    (llvm::IRBuilder<>&, llvm::Value*, llvm::Value*)>;

inline bool AssertOpaquePtrs([[maybe_unused]]llvm::Value* opaque, [[maybe_unused]]llvm::Type* type)
{
#if LLVM_VERSION_MAJOR <= 15
    if (opaque->getType()->getPointerElementType() != type) {
#ifdef OPENVDB_ENABLE_ASSERTS
        llvm::errs() << "Opaque ptr check failed:\n";
        llvm::errs() << "Input Value: "; opaque->print(llvm::errs()); llvm::errs() << "\n";
        llvm::errs() << "Input Ptr Type: "; opaque->getType()->getPointerElementType()->print(llvm::errs()); llvm::errs() << "\n";
        llvm::errs() << "Type: "; type->print(llvm::errs()); llvm::errs() << "\n";
#endif
        return false;
    }
#endif
    return true;
}

/// @brief  Populate a vector of llvm Types from a vector of llvm values
///
/// @param  values  A vector of llvm values to retrieve types from
/// @param  types   A vector of llvm types to populate
///
inline void
valuesToTypes(const std::vector<llvm::Value*>& values,
              std::vector<llvm::Type*>& types)
{
    types.reserve(values.size());
    for (const auto& v : values) {
        types.emplace_back(v->getType());
    }
}

/// @brief  Prints an llvm type to a std string
///
/// @param  type  The llvm type to convert
/// @param  str   The string to store the type info to
///
inline void
llvmTypeToString(const llvm::Type* const type, std::string& str)
{
    llvm::raw_string_ostream os(str);
    type->print(os);
    os.flush();
}

/// @brief  Return an llvm value representing a pointer to the provided ptr builtin
///         ValueT.
/// @note   This is probably not a suitable solution for anything other than POD
///         types and should be used with caution.
///
/// @param  ptr      A pointer to a type of ValueT whose address will be computed and
///                  returned
/// @param  builder  The current llvm IRBuilder
///
template <typename ValueT>
inline llvm::Value*
llvmPointerFromAddress(const ValueT* const& ptr,
                       llvm::IRBuilder<>& builder)
{
    llvm::Value* address =
        llvm::ConstantInt::get(llvm::Type::getIntNTy(builder.getContext(), sizeof(uintptr_t)*8),
                               reinterpret_cast<uintptr_t>(ptr));
    return builder.CreateIntToPtr(address, LLVMType<ValueT*>::get(builder.getContext()));
}

/// @brief  Insert a stack allocation at the beginning of the current function
///         of the provided type and size. The IRBuilder's insertion point must
///         be set to a BasicBlock with a valid Function parent.
/// @note   If a size is provided, the size must not depend on any other
///         instructions. If it does, invalid LLVM IR will bb generated.
///
/// @param  B  The IRBuilder
/// @param  type  The type to allocate
/// @param  size  Optional count of allocations. If nullptr, runs a single allocation
inline llvm::Value*
insertStaticAlloca(llvm::IRBuilder<>& B,
                   llvm::Type* type,
                   llvm::Value* size = nullptr)
{
    llvm::StructType* strtype = LLVMType<codegen::String>::get(B.getContext());
    // Create the allocation at the start of the function block
    llvm::Function* parent = B.GetInsertBlock()->getParent();
    OPENVDB_ASSERT(parent && !parent->empty());
    auto IP = B.saveIP();
    llvm::BasicBlock& block = parent->front();
    if (block.empty()) B.SetInsertPoint(&block);
    else B.SetInsertPoint(&(block.front()));
    llvm::Value* result = B.CreateAlloca(type, size);

    /// @note  Strings need to be initialised correctly when they are
    ///   created. We alloc them at the start of the function but
    ///   strings in branches may not ever be set to anything. If
    ///   we don't init these correctly, the clearup frees will
    ///   try and free uninitialised memory
    if (type == strtype) {
        llvm::Value* cptr = B.CreateStructGEP(strtype, result, 0); // char**
        llvm::Value* sso = B.CreateStructGEP(strtype, result, 1); // char[]*
        llvm::Value* sso_load = B.CreateConstGEP2_64(strtype->getTypeAtIndex(1), sso, 0, 0); // char[]
        llvm::Value* len = B.CreateStructGEP(strtype, result, 2);
        B.CreateStore(sso_load, cptr); // this->ptr = this->SSO;
        B.CreateStore(B.getInt64(0), len);
    }
    B.restoreIP(IP);
    return result;
}

inline llvm::Argument*
extractArgument(llvm::Function* F, const size_t idx)
{
    if (!F) return nullptr;
    if (idx >= F->arg_size()) return nullptr;
    return llvm::cast<llvm::Argument>(F->arg_begin() + idx);
}

inline llvm::Argument*
extractArgument(llvm::Function* F, const std::string& name)
{
    if (!F) return nullptr;
    for (auto iter = F->arg_begin(); iter != F->arg_end(); ++iter) {
        llvm::Argument* arg = llvm::cast<llvm::Argument>(iter);
        if (arg->getName() == name) return arg;
    }
    return nullptr;
}

/// @brief  Returns the highest order type from two LLVM Scalar types
///
/// @param  typeA  The first scalar llvm type
/// @param  typeB  The second scalar llvm type
///
inline llvm::Type*
typePrecedence(llvm::Type* const typeA,
               llvm::Type* const typeB)
{
    OPENVDB_ASSERT(typeA && (typeA->isIntegerTy() || typeA->isFloatingPointTy()) &&
        "First Type in typePrecedence is not a scalar type");
    OPENVDB_ASSERT(typeB && (typeB->isIntegerTy() || typeB->isFloatingPointTy()) &&
        "Second Type in typePrecedence is not a scalar type");

    // handle implicit arithmetic conversion
    // (http://osr507doc.sco.com/en/tools/clang_conv_implicit.html)

    if (typeA->isDoubleTy()) return typeA;
    if (typeB->isDoubleTy()) return typeB;

    if (typeA->isFloatTy()) return typeA;
    if (typeB->isFloatTy()) return typeB;

    if (typeA->isIntegerTy(64)) return typeA;
    if (typeB->isIntegerTy(64)) return typeB;

    if (typeA->isIntegerTy(32)) return typeA;
    if (typeB->isIntegerTy(32)) return typeB;

    if (typeA->isIntegerTy(16)) return typeA;
    if (typeB->isIntegerTy(16)) return typeB;

    if (typeA->isIntegerTy(8)) return typeA;
    if (typeB->isIntegerTy(8)) return typeB;

    if (typeA->isIntegerTy(1)) return typeA;
    if (typeB->isIntegerTy(1)) return typeB;

    OPENVDB_ASSERT(false && "invalid LLVM type precedence");
    return nullptr;
}

/// @brief  Returns a CastFunction which represents the corresponding instruction
///         to convert a source llvm Type to a target llvm Type. If the conversion
///         is unsupported, throws an error.
/// @warning  This assumes any integer types are signed.
/// @param  sourceType  The source type to cast
/// @param  targetType  The target type to cast to
/// @param  twine       An optional string description of the cast function. This can
///                     be used for for more verbose llvm information on IR compilation
///                     failure
/// @todo   Deprecate this method, move functionality into codegen::Value
inline CastFunction
llvmArithmeticConversion(const llvm::Type* const sourceType,
                         const llvm::Type* const targetType,
                         const std::string& twine = "")
{

#define BIND_ARITHMETIC_CAST_OP(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine)

#if LLVM_VERSION_MAJOR >= 19
// For Trunc, assume argument is never NUW/NSW (last flags)
#define BIND_ARITHMETIC_CAST_OP_TRUNC(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine, \
        /*IsNUW*/false, \
        /*IsNSW*/false)
#else
#define BIND_ARITHMETIC_CAST_OP_TRUNC(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine)
#endif

#if LLVM_VERSION_MAJOR >= 19
// For UItoFP, assume argument is never negative (last flag)
#define BIND_ARITHMETIC_CAST_OP_UIFP(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine, \
        false) // IsNonNeg
#else
#define BIND_ARITHMETIC_CAST_OP_UIFP(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine)
#endif

#if LLVM_VERSION_MAJOR >= 18
// For Zext, assume argument is never negative (last flag)
#define BIND_ARITHMETIC_CAST_OP_ZEXT(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine, \
        false) // IsNonNeg
#else
#define BIND_ARITHMETIC_CAST_OP_ZEXT(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine)
#endif

#if LLVM_VERSION_MAJOR >= 20
#define BIND_ARITHMETIC_CAST_OP_FP(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine, \
        nullptr) // MDNode *FPMathTag
#else
#define BIND_ARITHMETIC_CAST_OP_FP(Function, Twine) \
    std::bind(&Function, \
        std::placeholders::_1, \
        std::placeholders::_2, \
        std::placeholders::_3, \
        Twine)
#endif

    if (targetType->isDoubleTy()) {
        if (sourceType->isFloatTy())           return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPExt, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPExt, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_UIFP(llvm::IRBuilder<>::CreateUIToFP, twine);
    }
    else if (targetType->isFloatTy()) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPTrunc, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPExt, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_UIFP(llvm::IRBuilder<>::CreateUIToFP, twine);
    }
    else if (targetType->isHalfTy()) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPTrunc, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP_FP(llvm::IRBuilder<>::CreateFPTrunc, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSIToFP, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_UIFP(llvm::IRBuilder<>::CreateUIToFP, twine);
    }
    else if (targetType->isIntegerTy(64)) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_ZEXT(llvm::IRBuilder<>::CreateZExt, twine);
    }
    else if (targetType->isIntegerTy(32)) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_ZEXT(llvm::IRBuilder<>::CreateZExt, twine);
    }
    else if (targetType->isIntegerTy(16)) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateSExt, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_ZEXT(llvm::IRBuilder<>::CreateZExt, twine);
    }
    else if (targetType->isIntegerTy(8)) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToSI, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(1))   return BIND_ARITHMETIC_CAST_OP_ZEXT(llvm::IRBuilder<>::CreateZExt, twine);
    }
    else if (targetType->isIntegerTy(1)) {
        if (sourceType->isDoubleTy())          return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToUI, twine);
        else if (sourceType->isFloatTy())      return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToUI, twine);
        else if (sourceType->isHalfTy())       return BIND_ARITHMETIC_CAST_OP(llvm::IRBuilder<>::CreateFPToUI, twine);
        else if (sourceType->isIntegerTy(64))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(32))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(16))  return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
        else if (sourceType->isIntegerTy(8))   return BIND_ARITHMETIC_CAST_OP_TRUNC(llvm::IRBuilder<>::CreateTrunc, twine);
    }

#undef BIND_ARITHMETIC_CAST_OP
    OPENVDB_ASSERT(false && "invalid LLVM type conversion");
    return CastFunction();
}

/// @brief  Returns a BinaryFunction representing the corresponding instruction to
///         perform on two scalar values, relative to a provided operator token. Note that
///         not all operations are supported on floating point types! If the token is not
///         supported, or the llvm type is not a scalar type, throws an error.
/// @note   Various default arguments are bound to provide a simple function call
///         signature. For floating point operations, this includes a null pointer to
///         the optional metadata node. For integer operations, this includes disabling
///         all overflow/rounding optimisations
///
/// @param  type   The type defining the precision of the binary operation
/// @param  token  The token used to create the relative binary operation
/// @param  twine  An optional string description of the binary function. This can
///                be used for for more verbose llvm information on IR compilation
///                failure
inline BinaryFunction
llvmBinaryConversion(const llvm::Type* const type,
                     const ast::tokens::OperatorToken& token,
                     const std::string& twine = "")
{

#define BIND_BINARY_OP(Function) \
    [twine](llvm::IRBuilder<>& B, llvm::Value* L, llvm::Value* R) \
        -> llvm::Value* { return B.Function(L, R, twine); }

    // NOTE: Binary % and / ops always take sign into account (CreateSDiv vs CreateUDiv, CreateSRem vs CreateURem).
    // See http://stackoverflow.com/questions/5346160/llvm-irbuildercreateudiv-createsdiv-createexactudiv
    // a%b in AX is implemented as a floored modulo op and is handled explicitly in binaryExpression

    if (type->isFloatingPointTy()) {
        OPENVDB_ASSERT(!(ast::tokens::operatorType(token) == ast::tokens::LOGICAL ||
            ast::tokens::operatorType(token) == ast::tokens::BITWISE)
                && "unable to perform logical or bitwise operation on floating point values");

        if (token == ast::tokens::PLUS)                 return BIND_BINARY_OP(CreateFAdd);
        else if (token == ast::tokens::MINUS)           return BIND_BINARY_OP(CreateFSub);
        else if (token == ast::tokens::MULTIPLY)        return BIND_BINARY_OP(CreateFMul);
        else if (token == ast::tokens::DIVIDE)          return BIND_BINARY_OP(CreateFDiv);
        else if (token == ast::tokens::MODULO)          return BIND_BINARY_OP(CreateFRem); // Note this is NOT a%b in AX.
        else if (token == ast::tokens::EQUALSEQUALS)    return BIND_BINARY_OP(CreateFCmpOEQ);
        else if (token == ast::tokens::NOTEQUALS)       return BIND_BINARY_OP(CreateFCmpONE);
        else if (token == ast::tokens::MORETHAN)        return BIND_BINARY_OP(CreateFCmpOGT);
        else if (token == ast::tokens::LESSTHAN)        return BIND_BINARY_OP(CreateFCmpOLT);
        else if (token == ast::tokens::MORETHANOREQUAL) return BIND_BINARY_OP(CreateFCmpOGE);
        else if (token == ast::tokens::LESSTHANOREQUAL) return BIND_BINARY_OP(CreateFCmpOLE);
        OPENVDB_ASSERT(false && "unrecognised binary operator");
    }
    else if (type->isIntegerTy()) {
        if (token == ast::tokens::PLUS)                  return BIND_BINARY_OP(CreateAdd); // No Unsigned/Signed Wrap
        else if (token == ast::tokens::MINUS)            return BIND_BINARY_OP(CreateSub); // No Unsigned/Signed Wrap
        else if (token == ast::tokens::MULTIPLY)         return BIND_BINARY_OP(CreateMul); // No Unsigned/Signed Wrap
        else if (token == ast::tokens::DIVIDE)           return BIND_BINARY_OP(CreateSDiv); // IsExact = false - when true, poison value if the reuslt is rounded
        else if (token == ast::tokens::MODULO)           return BIND_BINARY_OP(CreateSRem); // Note this is NOT a%b in AX.
        else if (token == ast::tokens::EQUALSEQUALS)     return BIND_BINARY_OP(CreateICmpEQ);
        else if (token == ast::tokens::NOTEQUALS)        return BIND_BINARY_OP(CreateICmpNE);
        else if (token == ast::tokens::MORETHAN)         return BIND_BINARY_OP(CreateICmpSGT);
        else if (token == ast::tokens::LESSTHAN)         return BIND_BINARY_OP(CreateICmpSLT);
        else if (token == ast::tokens::MORETHANOREQUAL)  return BIND_BINARY_OP(CreateICmpSGE);
        else if (token == ast::tokens::LESSTHANOREQUAL)  return BIND_BINARY_OP(CreateICmpSLE);
        else if (token == ast::tokens::AND)              return BIND_BINARY_OP(CreateAnd); // same as BITAND (bool/circuting logic handled a layer up)
        else if (token == ast::tokens::OR)               return BIND_BINARY_OP(CreateOr);
        else if (token == ast::tokens::SHIFTLEFT)        return BIND_BINARY_OP(CreateShl); // No Unsigned/Signed Wrap
        else if (token == ast::tokens::SHIFTRIGHT)       return BIND_BINARY_OP(CreateAShr); // IsExact = false - poison value if any of the bits shifted out are non-zero.
        else if (token == ast::tokens::BITAND)           return BIND_BINARY_OP(CreateAnd); // note same as AND
        else if (token == ast::tokens::BITOR)            return BIND_BINARY_OP(CreateOr);
        else if (token == ast::tokens::BITXOR)           return BIND_BINARY_OP(CreateXor);
        OPENVDB_ASSERT(false && "unrecognised binary operator");
    }

#undef BIND_BINARY_OP
    OPENVDB_ASSERT(false && "invalid LLVM type for binary operation");
    return BinaryFunction();
}

/// @brief  Returns true if the llvm Type 'from' can be safely cast to the llvm
///         Type 'to'.
inline bool isValidCast(llvm::Type* from, llvm::Type* to)
{
    OPENVDB_ASSERT(from && "llvm Type 'from' is null in isValidCast");
    OPENVDB_ASSERT(to && "llvm Type 'to' is null in isValidCast");

    if ((from->isIntegerTy() || from->isFloatingPointTy()) &&
        (to->isIntegerTy() || to->isFloatingPointTy())) {
        return true;
    }
    if (from->isArrayTy() && to->isArrayTy()) {
        llvm::ArrayType* af = llvm::cast<llvm::ArrayType>(from);
        llvm::ArrayType* at = llvm::cast<llvm::ArrayType>(to);
        if (af->getArrayNumElements() == at->getArrayNumElements()) {
            return isValidCast(af->getArrayElementType(),
                at->getArrayElementType());
        }
    }
    return false;
}

/// @brief  Casts a scalar llvm Value to a target scalar llvm Type. Returns
///         the cast scalar value of type targetType.
/// @warning  This assumes any integer types are signed.
/// @param value       A llvm scalar value to convert
/// @param targetType  The target llvm scalar type to convert to
/// @param builder     The current llvm IRBuilder
inline llvm::Value*
arithmeticConversion(llvm::Value* value,
                     llvm::Type* targetType,
                     llvm::IRBuilder<>& builder)
{
    OPENVDB_ASSERT(value && (value->getType()->isIntegerTy() || value->getType()->isFloatingPointTy()) &&
        "First Value in arithmeticConversion is not a scalar type");
    OPENVDB_ASSERT(targetType && (targetType->isIntegerTy() || targetType->isFloatingPointTy()) &&
        "Target Type in arithmeticConversion is not a scalar type");

    const llvm::Type* const valueType = value->getType();
    if (valueType == targetType) return value;

    CastFunction llvmCastFunction = llvmArithmeticConversion(valueType, targetType);
    return llvmCastFunction(builder, value, targetType);
}

/// @brief Converts a vector of loaded llvm scalar values of the same type to a
///   target scalar type. Each value is converted individually and the loaded
///   result stored in the same location within values.
/// @warning  This assumes any integer types are signed.
/// @param values             A vector of llvm scalar values to convert
/// @param targetElementType  The target llvm scalar type to convert each value
///                           of the input vector
/// @param builder            The current llvm IRBuilder
inline void
arithmeticConversion(std::vector<llvm::Value*>& values,
                     llvm::Type* targetElementType,
                     llvm::IRBuilder<>& builder)
{
    OPENVDB_ASSERT(targetElementType && (targetElementType->isIntegerTy() ||
        targetElementType->isFloatingPointTy()) &&
        "Target element type is not a scalar type");

    llvm::Type* sourceElementType = values.front()->getType();
    OPENVDB_ASSERT(sourceElementType && (sourceElementType->isIntegerTy() ||
        sourceElementType->isFloatingPointTy()) &&
        "Source element type is not a scalar type");

    if (sourceElementType == targetElementType) return;

    CastFunction llvmCastFunction = llvmArithmeticConversion(sourceElementType, targetElementType);

    for (llvm::Value*& value : values) {
        value = llvmCastFunction(builder, value, targetElementType);
    }
}

/// @brief  Converts a vector of loaded llvm scalar values to the highest precision
///         type stored amongst them. Any values which are not scalar types are ignored
/// @warning  This assumes any integer types are signed.
/// @param values   A vector of llvm scalar values to convert
/// @param builder  The current llvm IRBuilder
inline void
arithmeticConversion(std::vector<llvm::Value*>& values,
                     llvm::IRBuilder<>& builder)
{
    llvm::Type* typeCast = LLVMType<bool>::get(builder.getContext());
    for (llvm::Value*& value : values) {
        llvm::Type* type = value->getType();
        if (type->isIntegerTy() || type->isFloatingPointTy()) {
            typeCast = typePrecedence(typeCast, type);
        }
    }

    arithmeticConversion(values, typeCast, builder);
}

/// @brief  Chooses the highest order llvm Type as defined by typePrecedence
///         from either of the two incoming values and casts the other value to
///         the choosen type if it is not already. The types of valueA and valueB
///         are guaranteed to match. Both values must be scalar LLVM types
/// @warning  This assumes any integer types are signed.
/// @param valueA   The first llvm value
/// @param valueB   The second llvm value
/// @param builder  The current llvm IRBuilder
inline void
arithmeticConversion(llvm::Value*& valueA,
                     llvm::Value*& valueB,
                     llvm::IRBuilder<>& builder)
{
    llvm::Type* type = typePrecedence(valueA->getType(), valueB->getType());
    valueA = arithmeticConversion(valueA, type, builder);
    valueB = arithmeticConversion(valueB, type, builder);
}

/// @brief  Performs a C style boolean comparison from a given scalar LLVM value
///
/// @param value    The scalar llvm value to convert to a boolean
/// @param builder  The current llvm IRBuilder
///
inline llvm::Value*
boolComparison(llvm::Value* value,
               llvm::IRBuilder<>& builder)
{
    llvm::Type* type = value->getType();

    if (type->isFloatingPointTy())  return builder.CreateFCmpONE(value, llvm::ConstantFP::get(type, 0.0));
    else if (type->isIntegerTy(1))  return builder.CreateICmpNE(value, llvm::ConstantInt::get(type, 0));
    else if (type->isIntegerTy())   return builder.CreateICmpNE(value, llvm::ConstantInt::getSigned(type, 0));
    OPENVDB_ASSERT(false && "Invalid type for bool conversion");
    return nullptr;
}

/// @ brief  Performs a binary operation on two loaded llvm scalar values of the same type.
///          The type of operation performed is defined by the token (see the list of supported
///          tokens in ast/Tokens.h. Returns a loaded llvm scalar result
///
/// @param lhs       The left hand side value of the binary operation
/// @param rhs       The right hand side value of the binary operation
/// @param token     The token representing the binary operation to perform
/// @param builder   The current llvm IRBuilder
inline llvm::Value*
binaryOperator(llvm::Value* lhs, llvm::Value* rhs,
               const ast::tokens::OperatorToken& token,
               llvm::IRBuilder<>& builder)
{
    llvm::Type* lhsType = lhs->getType();
    OPENVDB_ASSERT(lhsType == rhs->getType() ||
        (token == ast::tokens::SHIFTLEFT ||
         token == ast::tokens::SHIFTRIGHT));

    const ast::tokens::OperatorType opType = ast::tokens::operatorType(token);

    if (opType == ast::tokens::LOGICAL) {
        lhs = boolComparison(lhs, builder);
        rhs = boolComparison(rhs, builder);
        lhsType = lhs->getType(); // now bool type
    }

    const BinaryFunction llvmBinaryFunction = llvmBinaryConversion(lhsType, token);
    return llvmBinaryFunction(builder, lhs, rhs);
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_UTILS_HAS_BEEN_INCLUDED

