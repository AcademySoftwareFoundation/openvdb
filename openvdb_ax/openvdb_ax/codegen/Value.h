// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/Value.h
///
/// @authors Nick Avramoussis
///
/// @brief  Intermediate representation of supported AX values
///

#ifndef OPENVDB_AX_CODEGEN_VALUE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_VALUE_HAS_BEEN_INCLUDED

#include "Types.h"
#include "Utils.h"

#include "../ast/AST.h"
#include "../ast/Tokens.h"
#include "../compiler/Logger.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  Intermediate representation wrapper for supported value types in AX
///   as immutable instances.
///@details This class allows you to generate LLVM IR for common operations
///   supported by the AX grammar whilst abstracting away a lot of the
///   complexity of the underlying LLVM state. This class is not meant as a
///   definitive representation of all possible LLVM IR instructions that can
///   be emitted, more as a strict helper translation layer for some supported
///   AX instructions to LLVM IR. More generally, this is intended to be used
///   for all arithmetic operations and less so for program control flow
///   (branches, loops, etc). Importantly, this class abstracts away the
///   concept of ptr/loaded instructions. That is, users of this API usually do
///   not need to worry about explicitly loading or querying the state of LLVM
///   IR allocations when calling methods. Instances of Value types also ensure
///   that the required underlying type information is retained, necessary from
///   LLVM 15 onwards (due to the introduction of LLVM's opaque ptr changes).
///
///   The subset of possible types this class supports are
///    - void types (for function returns)
///    - AX scalar values (bool, ints, floats)
///    - AX array values (element types of ints or floats)
///    - AX string values
///
///   Note that a Value can have a variety of arithmetic type/precision states.
///   The API of this class does not guaranteed compatibility between all
///   states. Some failure cases may report a log message and return
///   Value::Invalid() or generate invalid IR/result in undefined behaviour if
///   the inputs are not correct. Refer to individual API methods for more
///   details.
class OPENVDB_AX_API Value
{
public:
    /// @brief  Create a Value with a provided underlying type
    /// @warning  This constructor assumes that the underlying type is correct.
    ///   This cannot be asserted from LLVM 15 onwards. This should be used
    ///   sparingly and currently only exists to support some holes in the
    ///   compute generator. This constructor should eventually be removed as
    ///   these get closed, in favour of the static create/alloc methods.
    Value(llvm::Value* val, llvm::Type* utype)
        : mVal(val)
        , mUType(utype) {
            OPENVDB_ASSERT((!mVal && !mUType) || bool(*this));
        }
    /// @brief Initialize from a constant value
    explicit Value(llvm::Constant* costant)
        : Value(static_cast<llvm::Value*>(costant), costant->getType()) {}

    Value(const Value&) = default;
    Value(Value&&) = default;
    Value& operator=(const Value&) = default;
    Value& operator=(Value&&) = default;

    /// @brief  Return an invalid Value. This is used to represent various fail
    ///   cases. Note that the operator bool(Value) will return false in for
    ///   Value::Invalid
    static Value Invalid() { return Value(nullptr, nullptr); }

    /// @brief  Create an arithmetic literal
    template <typename ValueType>
    static Value Create(llvm::LLVMContext& C, const ValueType& value)
    {
        return Value(LLVMType<ValueType>::get(C, value),
            LLVMType<ValueType>::get(C));
    }

    /// @brief  Create a value that represents a return value from a function.
    ///   Really only intended to be used by the function framework. If ret is
    ///   not provided, a void return is created.
    static Value Return(llvm::IRBuilder<>& B, Value* ret = nullptr)
    {
        return ret ? Value(B.CreateRet(ret->GetValue()), B.getVoidTy()) :
            Value(B.CreateRetVoid(), B.getVoidTy());
    }

    /// @brief  Return true if the underlying type held by utype is supported
    ///   via the interface of this Value class
    static bool Supports(llvm::Type* utype)
    {
        llvm::Type* strtype = LLVMType<codegen::String>::get(utype->getContext());
        return utype->isVoidTy() ||
            utype->isIntegerTy() ||
            utype->isFloatingPointTy() ||
            utype == strtype ||
            (utype->isArrayTy() &&
                (utype->getArrayElementType()->isIntegerTy() ||
                 utype->getArrayElementType()->isFloatingPointTy()));
    }

    /// @brief  Emit IR inserting an allocation at the front of the BasicBlock
    ///   pointed to by the provided IRBuilder. The type is expected to be a
    ///   supported utype.
    static Value Alloc(llvm::IRBuilder<>& B, llvm::Type* type, llvm::Value* size = nullptr)
    {
        OPENVDB_ASSERT(Value::Supports(type));
        // Create the allocation at the start of the function block
        llvm::Function* parent = B.GetInsertBlock()->getParent();
        OPENVDB_ASSERT(parent && !parent->empty());
        auto IP = B.saveIP();
        llvm::BasicBlock& block = parent->front();
        if (block.empty()) B.SetInsertPoint(&block);
        else               B.SetInsertPoint(&(block.front()));
        llvm::Value* result = B.CreateAlloca(type, size);

        /// @note  Strings need to be initialised correctly when they are
        ///   created. We alloc them at the start of the function but
        ///   strings in branches may not ever be set to anything. If
        ///   we don't init these correctly, the clearup frees will
        ///   try and free uninitialised memory
        llvm::StructType* strtype = LLVMType<codegen::String>::get(B.getContext());
        if (type == strtype) {
            llvm::Value* cptr = B.CreateStructGEP(strtype, result, 0); // char**
            llvm::Value* sso = B.CreateStructGEP(strtype, result, 1); // char[]*
            OPENVDB_ASSERT(AssertOpaquePtrs(sso, strtype->getTypeAtIndex(1)));
            llvm::Value* sso_load = B.CreateConstInBoundsGEP2_64(strtype->getTypeAtIndex(1), sso, 0, 0); // char[]
            llvm::Value* len = B.CreateStructGEP(strtype, result, 2);
            B.CreateStore(sso_load, cptr); // this->ptr = this->SSO;
            B.CreateStore(B.getInt64(0), len);
        }
        B.restoreIP(IP);
        return Value(result, type);
    }

    /// @brief  Emit IR to create an array from a set of scalar values. Will
    ///   generate invalid IR if the values are not all scalar or are of
    ///   different precision. values cannot be empty.
    static Value ScalarsToArray(llvm::IRBuilder<>& B, const std::vector<Value>& values)
    {
        OPENVDB_ASSERT(!values.empty());

        llvm::Type* type = values.front().GetUnderlyingType();
        llvm::Type* arrayType = llvm::ArrayType::get(type, values.size());
        Value array = Value::Alloc(B, arrayType);

        size_t idx = 0;
        for (const Value& value : values)
        {
            OPENVDB_ASSERT(value);
            OPENVDB_ASSERT(value.IsScalar());
            OPENVDB_ASSERT(value.GetUnderlyingType() == type);
            Value element = array.GetArrayElement(B, idx++);
            B.CreateStore(value.GetValue(), element.GetValue());
        }

        return array;
    }

    /// @brief  Create a new zero scalar Value using the underlying scalar
    ///   precision of this Value. Does not generate IR, however will return
    ///   an invalid constant if this Value is a string
    Value Zero() const
    {
        return Value(llvmConstant<uint64_t>(0, this->GetUnderlyingScalarType()));
    }

    /// @brief  Create a new one scalar Value using the underlying scalar
    ///   precision of this Value. Does not generate IR, however will return
    ///   an invalid constant if this Value is a string
    Value One() const
    {
        return Value(llvmConstant<uint64_t>(1, this->GetUnderlyingScalarType()));
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Check if this Value contains an active underlying llvm
    ///   Value/Type. When asserts are enabled. This method strictly checks all
    ///   possible valid combination types of a Value.
    operator bool() const;

    /// @brief  See bool operator
    bool operator!() const { return !bool(*this); }

    /// @brief  Return true if this value represents a void type. This is
    ///   typically only possible for void function returns
    /// @note   Void types are only ever explicitly void, never pointers to void
    bool IsVoid() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isVoidTy();
    }

    /// @brief  Return true if the underlying type is a bool type
    /// @note  A bool's underlying state can be either as a pointer or
    ///   loaded. This method returns true in both instances if the underlying
    ///   type is a bool.
    bool IsBool() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isIntegerTy(1);
    }

    /// @brief  Return true if the underlying type is a scalar type (bool, int
    ///   or float).
    /// @note  A scalar's underlying state can be either as a pointer or
    ///   loaded. This method returns true in both instances if the underlying
    ///   type is a scalar.
    bool IsScalar() const
    {
        OPENVDB_ASSERT(*this);
        return (mUType->isIntegerTy() || mUType->isFloatingPointTy());
    }

    /// @brief  Return true if the underlying type is an integer type
    /// @note  A integer's underlying state can be either as a pointer or
    ///   loaded. This method returns true in both instances if the underlying
    ///   type is a integer.
    bool IsInteger() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isIntegerTy();
    }

    /// @brief  Return true if the underlying type is an floating point type
    ///   (float or double).
    /// @note  A float's underlying state can be either as a pointer or
    ///   loaded. This method returns true in both instances if the underlying
    ///   type is a float/double.
    bool IsFloat() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isFloatingPointTy();
    }

    /// @brief  Return true if the underlying type is an array type
    /// @note   An array type's state is only ever a pointer to an array
    ///   allocation
    bool IsArray() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isArrayTy();
    }

    /// @brief  Return true if the underlying type is an vector 2/3/4 type
    /// @note   An vector type's state is only ever a pointer to a vector
    ///   allocation
    bool IsVector() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isArrayTy() && !this->IsMatrix();
    }

    /// @brief  Return true if the underlying type is an matrix 3/4 type
    /// @note   An matrix type's state is only ever a pointer to a matrix
    ///   allocation
    bool IsMatrix() const
    {
        OPENVDB_ASSERT(*this);
        return mUType->isArrayTy() &&
            // @todo  This is dumb, add type-metadata for this
            (this->GetArrayNumElements() == 9 || this->GetArrayNumElements() == 16);
    }

    /// @brief  Return true if the underlying type is a string type
    /// @note   An string type's state is only ever a pointer to a string
    ///   allocation
    bool IsString() const
    {
        OPENVDB_ASSERT(*this);
        return mUType == LLVMType<codegen::String>::get(mUType->getContext());
    }

    /// @brief  Return true if this Value is a pointer type
    bool IsPtr() const
    {
        OPENVDB_ASSERT(*this);
        return mVal->getType()->isPointerTy();
    }

    /// @brief  Return true if this Value is a constant
    bool IsConstant() const
    {
        return bool(llvm::dyn_cast<llvm::Constant>(this->GetValue()));
    }

    /// @brief  Return the number of elements in this array type
    size_t GetArrayNumElements() const
    {
        OPENVDB_ASSERT(this->IsArray());
        return mUType->getArrayNumElements();
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    // The following method generate IR and return new Value instances

    /// @brief  Emit IR to check whether this value is NaN. Only works on Float
    ///   types and will generate invalid IR if this Value is not a Float
    ///   instance.
    Value IsNan(llvm::IRBuilder<>& B) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsFloat());
        Value self = this->LoadIfPtr(B);
        llvm::Value* result = B.CreateFCmpUNO(self.GetValue(), self.GetValue());
        return Value(result, result->getType());
    }

    /// @brief  Emit IR to load the current value. Not typically required to
    ///   call directly. Will generate invalid IR if this Value is not a Ptr.
    /// @warning  Should only ever be called for ptrs to scalar types
    Value Load(llvm::IRBuilder<>& B) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsPtr());
        OPENVDB_ASSERT(this->IsScalar()); // Should only be loading scalars
        return Value(B.CreateLoad(mUType, mVal), mUType);
    }

    /// @brief  Emit IR to load the current value if it is a ptr. Not typically
    ///   required to call directly.
    /// @warning  Should only ever be called for ptrs to scalar types
    Value LoadIfPtr(llvm::IRBuilder<>& B) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsScalar()); // Should only be loading scalars
        return this->IsPtr() ? this->Load(B) : *this;
    }

    /// @brief  Emit IR to return a scalar at the provided index from this
    ///   array value. Will generate invalid IR if this Value is not an array
    ///   or if idx is not an integer.
    Value GetArrayElement(llvm::IRBuilder<>& B, Value idx) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsArray());
        OPENVDB_ASSERT(idx.IsInteger());
        idx = idx.LoadIfPtr(B);
        if (this->IsPtr()) {
            Value zero = idx.Zero(); // same int precision as idx
            return Value(B.CreateInBoundsGEP(mUType, mVal, {zero.GetValue(), idx.GetValue()}), mUType->getArrayElementType());
        }
        else {
            OPENVDB_ASSERT(this->IsConstant());
            return Value(B.CreateInBoundsGEP(mUType, mVal, {idx.GetValue()}), mUType->getArrayElementType());
        }
    }

    /// @brief  Emit IR to return a scalar at the provided index from this
    ///   array value. Will generate invalid IR if this Value is not an array.
    Value GetArrayElement(llvm::IRBuilder<>& B, uint64_t idx) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsArray());
        OPENVDB_ASSERT(idx < this->GetArrayNumElements());
        if (this->IsPtr()) {
            return Value(B.CreateConstInBoundsGEP2_64(mUType, mVal, uint64_t(0), idx), mUType->getArrayElementType());
        }
        else {
            OPENVDB_ASSERT(this->IsConstant());
            return Value(B.CreateConstInBoundsGEP1_64(mUType, mVal, idx), mUType->getArrayElementType());
        }
    }

    /// @brief  Emit IR to extract scalar values from the elements in this
    ///   array and populate the provided vector with them. The scalars are
    ///   additionally loaded if load is true. Will generate invalid IR if this
    ///   is not an array
    void ArrayToScalars(llvm::IRBuilder<>& B,
        std::vector<Value>& elements,
        const bool load = false) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsArray());

        const size_t size = this->GetArrayNumElements();
        elements.reserve(size);

        for (size_t i = 0; i < size; ++i)
        {
            Value elem = this->GetArrayElement(B, i);
            if (load) elem = elem.Load(B);
            OPENVDB_ASSERT(elem);
            elements.emplace_back(elem);
        }

        OPENVDB_ASSERT(!elements.empty());
    }

    /// @brief  Emit IR to broadcast this scalar to a new array. Will generated
    ///   invalid IR if this is not a scalar or if size is zero.
    /// @warning  This fills the array with the current scalar value. It does
    ///   NOT do scalar->matrix promotion.
    Value ScalarToArray(llvm::IRBuilder<>& B, size_t size) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT_MESSAGE(this->IsScalar(), "value type is not a scalar type");
        Value scalar = this->LoadIfPtr(B);

        llvm::Type* type = llvm::ArrayType::get(scalar.GetUnderlyingType(), size);
        Value array = Value::Alloc(B, type);
        for (size_t i = 0; i < size; ++i) {
            Value element = array.GetArrayElement(B, i);
            B.CreateStore(scalar.GetValue(), element.GetValue());
        }
        return array;
    }

    /// @brief  Emit IR to create a new 3x3 matrix from this scalar value,
    ///   adhering to scalar->matrix promotion rules. Will generate invalid IR
    ///   if this is not a scalar value.
    Value ScalarToIdentMatrix3(llvm::IRBuilder<>& B) const
    {
        return this->ScalarToIdentMatrixN<3>(B);
    }

    /// @brief  Emit IR to create a new 4x4 matrix from this scalar value,
    ///   adhering to scalar->matrix promotion rules. Will generate invalid IR
    ///   if this is not a scalar value.
    Value ScalarToIdentMatrix4(llvm::IRBuilder<>& B) const
    {
        return this->ScalarToIdentMatrixN<4>(B);
    }

    /// @brief  Emit IR to perform standard boolean comparison on this scalar
    ///   i.e. bool(scalar) or bool(scalar == 0). Returns a Value of type bool.
    ///   Will return Value::Invalid() if this is not a bool/int/float.
    Value ScalarBoolComparison(llvm::IRBuilder<>& B) const;

    /// @brief  Emit IR to select a value based on this boolean scalar value.
    ///   Will generate invalid IR if this is not a boolean value, or if
    ///   trueval and falseval have different types.
    Value Select(llvm::IRBuilder<>& B, const Value& trueval, const Value& falseval) const;

    /// @brief  Emit IR to cast this scalar or array to a new value of the
    ///   provided scalar precision. Returns a new value of the same class type
    ///   (scalar or array) but with a new precision. Will generate invalid IR
    ///   if the precision is not an integer or floating point precision type.
    ///   If this is not an array or scalar, no IR is emitted and
    ///   Value::Invalid() is returned. Additionally, if the provided precision
    ///   is the same as this scalar/array's underlying precision, no IR is
    ///   emitted and no new value is created.
    Value CastToPrecision(llvm::IRBuilder<>& B, llvm::Type* precision) const;

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    // Unary IR emission

    /// @brief  Emit IR to create a unary not instruction on this scalar or
    ///   integer array (i.e. !value). If the value is not a int/float/integer
    ///   array, no IR is emitted, a warning is logged (if a logger is provided)
    ///   and Value::Invalid is returned.
    Value Not(llvm::IRBuilder<>& B, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to create a unary not instruction on this integer or
    ///   integer array (i.e. ~value). If the value is not a int/integer array,
    ///   no IR is emitted, a warning is logged (if a logger is provided) and
    ///   Value::Invalid is returned.
    Value BitNot(llvm::IRBuilder<>& B, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to create a unary not instruction on this scalar or
    ///   array (i.e. -value). If the value is not a int/float/array, no IR is
    ///   is emitted, a warning is logged (if a logger is provided) and
    ///   Value::Invalid is returned.
    Value Negate(llvm::IRBuilder<>& B, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to increment this scalar (i.e. value + 1). Will
    ///   return Value::Invalid() and report a message to a logger (if provided)
    ///   if this is not an integer (non-bool) or float scalar.
    Value Increment(llvm::IRBuilder<>& B, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to increment this scalar (i.e. value - 1). Will
    ///   return Value::Invalid() and report a message to a logger (if provided)
    ///   if this is not an integer (non-bool) or float scalar.
    Value Decrement(llvm::IRBuilder<>& B, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Emit IR to perform a binary operation on this LHS value and a
    ///   provided RHS value. If the operation is not a valid binary operation,
    ///   Value::Invalid() is returned. Defer to the explicit binary methods on
    ///   this class for more details.
    /// @note For all methods, performing type promotion/casting where necessary
    ///   or return Value::Invalid() on incompatible inputs and report a
    ///   message to the log, if provided.
    Value Binary(llvm::IRBuilder<>& B,
        Value rhs,
        const ast::tokens::OperatorToken& op,
        Logger* log = nullptr,
        const ax::ast::Node* node = nullptr) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(rhs);
        switch (op)
        {
            case ast::tokens::OperatorToken::PLUS : return this->Add(B, rhs, log, node);
            case ast::tokens::OperatorToken::MINUS : return this->Subtract(B, rhs, log, node);
            case ast::tokens::OperatorToken::MULTIPLY : return this->Multiply(B, rhs, log, node);
            case ast::tokens::OperatorToken::DIVIDE : return this->Divide(B, rhs, log, node);
            case ast::tokens::OperatorToken::MODULO : return this->Modulo(B, rhs, log, node);
            case ast::tokens::OperatorToken::EQUALSEQUALS : return this->Equals(B, rhs, log, node);
            case ast::tokens::OperatorToken::NOTEQUALS : return this->NotEquals(B, rhs, log, node);
            case ast::tokens::OperatorToken::MORETHAN : return this->GreaterThan(B, rhs, log, node);
            case ast::tokens::OperatorToken::MORETHANOREQUAL : return this->GreaterThanEquals(B, rhs, log, node);
            case ast::tokens::OperatorToken::LESSTHAN : return this->LessThan(B, rhs, log, node);
            case ast::tokens::OperatorToken::LESSTHANOREQUAL : return this->LessThanEquals(B, rhs, log, node);
            case ast::tokens::OperatorToken::SHIFTLEFT : return this->ShiftLeft(B, rhs, log, node);
            case ast::tokens::OperatorToken::SHIFTRIGHT : return this->ShiftRight(B, rhs, log, node);
            case ast::tokens::OperatorToken::BITAND : return this->BitAnd(B, rhs, log, node);
            case ast::tokens::OperatorToken::BITOR : return this->BitOr(B, rhs, log, node);
            case ast::tokens::OperatorToken::BITXOR : return this->BitXor(B, rhs, log, node);
            default: return Value::Invalid();
        }
    }

    // Binary arithmetic IR emission.

    /// @brief  Emit IR to perform a && operation on two scalars. Assumes both
    ///   inputs are scalars (this checking is currently done in the
    ///   ComputeGenerator) and will cause undefined behaviour if they are not.
    /// @warning  This does not perform short circuiting. See:
    ///    ComputeGenerator::visit(const ast::BinaryOperator*)
    Value And(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(rhs);
        OPENVDB_ASSERT(this->IsScalar());
        OPENVDB_ASSERT(rhs.IsScalar());
        Value lhs = *this;
        lhs = lhs.ScalarBoolComparison(B);
        rhs = rhs.ScalarBoolComparison(B);
        return this->TrivialBinary(B, rhs, ast::tokens::AND, log, node);
    }

    /// @brief  Emit IR to perform a || operation on two scalars. Assumes both
    ///   inputs are scalars (this checking is currently done in the
    ///   ComputeGenerator) and will cause undefined behaviour if they are not.
    /// @warning  This does not perform short circuiting. See:
    ///    ComputeGenerator::visit(const ast::BinaryOperator*)
    Value Or(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(rhs);
        OPENVDB_ASSERT(this->IsScalar());
        OPENVDB_ASSERT(rhs.IsScalar());
        Value lhs = *this;
        lhs = lhs.ScalarBoolComparison(B);
        rhs = rhs.ScalarBoolComparison(B);
        return this->TrivialBinary(B, rhs, ast::tokens::OR, log, node);
    }

    /// @brief  Emit IR to perform a + operation on two values
    Value Add(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a - operation on two values
    Value Subtract(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a * operation on two values
    Value Multiply(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a / operation on two values If the denominator is constant and
    ///   zero, returns Value::Invalid()
    Value Divide(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a FLOORED % operation on two values
    Value Modulo(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a TRUNCATED % operation on two values
    Value TruncatedModulo(llvm::IRBuilder<>& B, Value rhs) const;

    // Binary Relational IR emission

    /// @brief  Emit IR to perform a == operation on two values
    Value Equals(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a != operation on two values
    Value NotEquals(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    /// @brief  Emit IR to perform a > operation on two values
    Value GreaterThan(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Relational(B, rhs, ast::tokens::MORETHAN, log, node);
    }

    /// @brief  Emit IR to perform a >= operation on two values
    Value GreaterThanEquals(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Relational(B, rhs, ast::tokens::MORETHANOREQUAL, log, node);
    }

    /// @brief  Emit IR to perform a < operation on two values
    Value LessThan(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Relational(B, rhs, ast::tokens::LESSTHAN, log, node);
    }

    /// @brief  Emit IR to perform a <= operation on two values
    Value LessThanEquals(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Relational(B, rhs, ast::tokens::LESSTHANOREQUAL, log, node);
    }

    // Binary Bitwise IR emission

    /// @brief  Emit IR to perform a << operation. Both values must be integers
    Value ShiftLeft(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Bitwise(B, rhs, ast::tokens::SHIFTLEFT, log, node);
    }

    /// @brief  Emit IR to perform a >> operation. Both values must be integers
    Value ShiftRight(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Bitwise(B, rhs, ast::tokens::SHIFTRIGHT, log, node);
    }

    /// @brief  Emit IR to perform a & operation. Both values must be integers
    Value BitAnd(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Bitwise(B, rhs, ast::tokens::BITAND, log, node);
    }

    /// @brief  Emit IR to perform a | operation. Both values must be integers
    Value BitOr(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Bitwise(B, rhs, ast::tokens::BITOR, log, node);
    }

    /// @brief  Emit IR to perform a ^ operation. Both values must be integers
    Value BitXor(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const
    {
        return this->Bitwise(B, rhs, ast::tokens::BITXOR, log, node);
    }

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Emit IR to assign the provided rhs to this value.
    Value Assign(llvm::IRBuilder<>& B, Value rhs, Logger* log = nullptr, const ax::ast::Node* node = nullptr) const;

    ///////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    /// @brief  Dump this value/type to llvm::errs
    void Print() const;

    /// @brief  Access the underlying llvm Value
    llvm::Value* GetValue() const { return mVal; }

    /// @brief  Access the underlying llvm Type
    llvm::Type* GetUnderlyingType() const { return mUType; }

    /// @brief  Access the underlying scalar type. This method assumes the
    ///   current value is a scalar or array
    llvm::Type* GetUnderlyingScalarType() const
    {
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT(this->IsScalar() || this->IsArray());
        return mUType->isArrayTy() ? mUType->getArrayElementType() : mUType;
    }

private:
    template <size_t Dim>
    Value ScalarToIdentMatrixN(llvm::IRBuilder<>& B) const
    {
        static_assert(Dim == 3 || Dim == 4);
        OPENVDB_ASSERT(*this);
        OPENVDB_ASSERT_MESSAGE(this->IsScalar(), "value type is not a scalar type");

        Value scalar = this->LoadIfPtr(B);
        llvm::Type* type = llvm::ArrayType::get(scalar.GetUnderlyingType(), Dim*Dim);
        Value array = Value::Alloc(B, type);
        Value zero = scalar.Zero();

        for (size_t i = 0; i < Dim*Dim; ++i) {
            const Value& m = ((i % (Dim+1) == 0) ? scalar : zero);
            llvm::Value* element = array.GetArrayElement(B, i).GetValue();
            B.CreateStore(m.GetValue(), element);
        }

        return array;
    }

    Value TrivialBinary(llvm::IRBuilder<>& B,
        Value rhs,
        const ast::tokens::OperatorToken& op,
        Logger* log,
        const ax::ast::Node* node) const;

    Value Bitwise(llvm::IRBuilder<>& B,
        Value rhs,
        const ast::tokens::OperatorToken& op,
        Logger* log,
        const ax::ast::Node* node) const;

    Value Relational(llvm::IRBuilder<>& B,
        Value rhs,
        const ast::tokens::OperatorToken& op,
        Logger* log,
        const ax::ast::Node* node) const;


    static Value Reduce(llvm::IRBuilder<>& B,
        const std::vector<Value>& bools,
        const ast::tokens::OperatorToken& op)
    {
        OPENVDB_ASSERT(!bools.empty());
        OPENVDB_ASSERT(op == ast::tokens::AND || op == ast::tokens::OR);
        Value result = bools.front();
        OPENVDB_ASSERT(result.IsBool());
        for (size_t i = 1; i < bools.size(); ++i) {
            result = result.TrivialBinary(B, bools[i], op, nullptr, nullptr);
        }
        return result;
    }

    Value ReduceBoolArray(llvm::IRBuilder<>& B, const ast::tokens::OperatorToken& op) const
    {
        OPENVDB_ASSERT(this->IsArray());
        OPENVDB_ASSERT(this->GetUnderlyingScalarType() == LLVMType<bool>::get(B.getContext()));
        std::vector<Value> elements;
        this->ArrayToScalars(B, elements);
        return Value::Reduce(B, elements, op);
    }

    static bool WImplicitScalarToMatrix(Logger* log, const ax::ast::Node* node)
    {
        if (!node) return true;
        if (auto* child = node->child(0)) {
            if (child->isType<ast::ArrayPack>()) {
                if (log) log->error("unable to deduce implicit {...} type for binary op as value "
                    "may be a matrix or array. assign to a local mat variable", child);
                return false;
            }
        }
        if (log && !log->warning("implicit cast to matrix from scalar. resulting "
            "cast will be equal to scalar <op> identity.", node->child(1))) return false;
        return true;
    }

    static Value WUnsupportedOp(const ast::tokens::CoreType& type,
        const ast::tokens::OperatorToken& op,
        Logger* log,
        const ax::ast::Node* node)
    {
        const std::string typestr = ast::tokens::typeStringFromToken(type);
        return Value::WUnsupportedOp(typestr, op, log, node);
    }

    static Value WUnsupportedOp(const std::string& typestr,
        const ast::tokens::OperatorToken& op,
        Logger* log,
        const ax::ast::Node* node)
    {
        if (log) {
            const std::string opstr = ast::tokens::operatorNameFromToken(op);
            log->error("unsupported " + typestr + " operation \"" + opstr + "\"", node);
        }
        return Value::Invalid();
    }

private:
    llvm::Value* mVal {nullptr};
    llvm::Type* mUType {nullptr};
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_VALUE_HAS_BEEN_INCLUDED

