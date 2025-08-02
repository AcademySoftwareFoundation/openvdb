// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/Utils.h
///
/// @authors Nick Avramoussis
///
/// @brief  Utility code generation methods for performing various llvm
///   operations
///

#include "Types.h"
#include "FunctionRegistry.h"
#include "Value.h"

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  Check if this Value contains an active underlying llvm
///   Value/Type. When asserts are enabled. This method strictly checks all
///   possible valid combination types of a Value.
Value::operator bool() const
{
    OPENVDB_ASSERT_MESSAGE((!mVal && !mUType) || (mVal && mUType),
        "Expected both a value and a type to exist, or both to be null");
    if (!mVal) return false;

#if LLVM_VERSION_MAJOR <= 15
    llvm::Type* base = mVal->getType();
    while (base->isPointerTy()) base = base->getContainedType(0);
    OPENVDB_ASSERT_MESSAGE(base == mUType,
        "Base type of Value does not match stored underlying type");
#endif

    [[maybe_unused]] llvm::Type* strtype = LLVMType<codegen::String>::get(mUType->getContext());
    // Underlying type should never be a ptr type
    OPENVDB_ASSERT_MESSAGE(!mUType->isPointerTy(),
        "Underlying type must not be a pointer type");
    // Value type should either be the underlying type or a pointer type
    OPENVDB_ASSERT_MESSAGE(mVal->getType() == mUType ||
        (mVal->getType()->isPointerTy()),
        "Value type must match the underlying type or be a opaque ptr");
    // Underlying type should be void, int, float, array or builtin string type
    OPENVDB_ASSERT_MESSAGE(
        mUType->isVoidTy() ||
        mUType->isIntegerTy() ||
        mUType->isFloatingPointTy() ||
        mUType->isArrayTy() ||
        mUType == strtype,
        "Unsupported underlying type for Value");
    // Array value types expected to be ints or floats
    OPENVDB_ASSERT_MESSAGE((mUType->isArrayTy() &&
        (mUType->getArrayElementType()->isIntegerTy() ||
         mUType->getArrayElementType()->isFloatingPointTy())) ||
        !mUType->isArrayTy(),
        "Unsupported element type for array");
    // Array types should be > 1
    OPENVDB_ASSERT_MESSAGE((mUType->isArrayTy() && mUType->getArrayNumElements() > 1) ||
        !mUType->isArrayTy(), "Underlying array type has a <= 1 size");
    // Value type should be a pointer if the underlying type is an array
    OPENVDB_ASSERT_MESSAGE(!mUType->isArrayTy() ||
        (mVal->getType()->isPointerTy() && mUType->isArrayTy()) ||
        (!mVal->getType()->isPointerTy() && this->IsConstant()),
        "Value types should be a pointer if the underlying type is an array or be a constant.");
    // Value type should be a pointer if the underlying type is an array
    OPENVDB_ASSERT_MESSAGE((mVal->getType()->isPointerTy() && mUType == strtype) ||
        mUType != strtype,
        "Value types should be a pointer if the underlying type is a string.");
    return true;
}

Value Value::ScalarBoolComparison(llvm::IRBuilder<>& B) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT_MESSAGE(this->IsScalar(), "value type is not a scalar type");

    Value scalar = this->LoadIfPtr(B);
    llvm::Value* result;
    llvm::Type* type = scalar.GetUnderlyingType();

    if (type->isFloatingPointTy())  result = B.CreateFCmpONE(scalar.GetValue(), llvm::ConstantFP::get(type, 0.0));
    else if (type->isIntegerTy(1))  result = B.CreateICmpNE(scalar.GetValue(), llvm::ConstantInt::get(type, 0));
    else if (type->isIntegerTy())   result = B.CreateICmpNE(scalar.GetValue(), llvm::ConstantInt::getSigned(type, 0));
    else {
        OPENVDB_ASSERT_MESSAGE(false, "Invalid type for bool conversion");
        return Value::Invalid();
    }
    OPENVDB_ASSERT(result);
    OPENVDB_ASSERT(result->getType()->isIntegerTy(1));
    return Value(result, result->getType());
}

Value Value::Select(llvm::IRBuilder<>& B, const Value& trueval, const Value& falseval) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(trueval);
    OPENVDB_ASSERT(falseval);
    OPENVDB_ASSERT(this->IsBool());
    OPENVDB_ASSERT(trueval.GetUnderlyingType() == falseval.GetUnderlyingType());
    Value self = this->LoadIfPtr(B);
    llvm::Value* result = B.CreateSelect(self.GetValue(), trueval.GetValue(), falseval.GetValue());
    return Value(result, trueval.GetUnderlyingType());
}

Value Value::CastToPrecision(llvm::IRBuilder<>& B, llvm::Type* precision) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT_MESSAGE(precision && (precision->isIntegerTy() ||
        precision->isFloatingPointTy()),
        "Target element type is not a scalar type");
    OPENVDB_ASSERT(this->IsScalar() || this->IsArray());

    llvm::Type* scalarTy = this->GetUnderlyingScalarType();
    if (scalarTy == precision) return Value(*this);

    if (this->IsScalar())
    {
        Value in = this->LoadIfPtr(B);
        // if target is bool, perform standard boolean conversion (*not* truncation).
        if (precision->isIntegerTy(1)) return in.ScalarBoolComparison(B);
        const CastFunction llvmCastFunction = llvmArithmeticConversion(scalarTy, precision);
        return Value(llvmCastFunction(B, in.GetValue(), precision), precision);
    }
    else if (this->IsArray())
    {
        const size_t size = this->GetArrayNumElements();
        llvm::Type* type = llvm::ArrayType::get(precision, size);
        Value array = Value::Alloc(B, type);

        for (size_t i = 0; i < size; ++i) {
            Value target = array.GetArrayElement(B, i);
            Value source = this->GetArrayElement(B, i).Load(B);
            OPENVDB_ASSERT(target);
            OPENVDB_ASSERT(source);
            OPENVDB_ASSERT(source.IsScalar());
            source = source.CastToPrecision(B, precision);
            OPENVDB_ASSERT(source);
            B.CreateStore(source.GetValue(), target.GetValue());
        }

        return array;
    }

    return Value::Invalid();
}

Value Value::Not(llvm::IRBuilder<>& B, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsScalar())
    {
        Value scalar = this->LoadIfPtr(B);
        llvm::Type* type = scalar.GetUnderlyingType();
        llvm::Value* result = nullptr;
        if (scalar.IsInteger()) {
            llvm::Value* rhs = scalar.IsBool() ? llvm::ConstantInt::get(type, 0) : llvm::ConstantInt::getSigned(type, 0);
            result = B.CreateICmpEQ(scalar.GetValue(), rhs);
        }
        else if (scalar.IsFloat()) {
            result = B.CreateFCmpOEQ(scalar.GetValue(), llvm::ConstantFP::get(type, 0));
        }
        else if (log) {
            log->error("unrecognised scalar type", node);
        }

        if (!result) return Value::Invalid();
        return Value(result, result->getType());
    }
    //@todo support NOT for fp arrays?
    else if (this->IsArray())
    {
        if (this->GetUnderlyingScalarType()->isIntegerTy()) {
            std::vector<Value> elements;
            this->ArrayToScalars(B, elements, /*load=*/true);
            for (Value& element : elements) {
                element = element.Not(B);
                OPENVDB_ASSERT(element);
            }
            return Value::ScalarsToArray(B, elements);
        }
        else if (this->GetUnderlyingScalarType()->isFloatingPointTy()) {
            //@todo support?
            if (log) log->error("unable to perform operation \"!\" on floating point arrays/vectors", node);
            return Value::Invalid();
        }
    }

    if (log) log->error("value is not a scalar or vector", node);
    return Value::Invalid();
}

Value Value::BitNot(llvm::IRBuilder<>& B, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsScalar())
    {
        Value scalar = this->LoadIfPtr(B);
        // if bool, cast to int32 for unary minus and bitnot
        if (scalar.IsBool()) {
            scalar = scalar.CastToPrecision(B, LLVMType<int32_t>::get(B.getContext()));
        }
        if (scalar.IsInteger()) {
            llvm::Value* result = B.CreateNot(scalar.GetValue());
            return Value(result, result->getType());
        }
        else if (scalar.IsFloat() && log) {
            log->error("unable to perform operation \"~\" on floating point values", node);
        }
        else if (log) {
            log->error("unrecognised scalar type", node);
        }
    }
    else if (this->IsArray())
    {
        if (this->GetUnderlyingScalarType()->isIntegerTy()) {
            std::vector<Value> elements;
            this->ArrayToScalars(B, elements, /*load=*/true);
            for (Value& element : elements) {
                element = element.BitNot(B);
                OPENVDB_ASSERT(element);
            }
            return Value::ScalarsToArray(B, elements);
        }
        else if (this->GetUnderlyingScalarType()->isFloatingPointTy()) {
            //@todo support?
            if (log) log->error("unable to perform operation \"~\" on floating point arrays/vectors", node);
            return Value::Invalid();
        }
    }

    else if (log)
    {
        log->error("value is not a scalar or vector", node);
    }

    return Value::Invalid();
}

Value Value::Negate(llvm::IRBuilder<>& B, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsScalar())
    {
        Value scalar = this->LoadIfPtr(B);
        // if bool, cast to int32 for unary minus and bitnot
        if (scalar.IsBool()) {
            scalar = scalar.CastToPrecision(B, LLVMType<int32_t>::get(B.getContext()));
        }
        if (scalar.IsInteger()) {
            llvm::Value* result = B.CreateNeg(scalar.GetValue());
            return Value(result, result->getType());
        }
        else if (scalar.IsFloat())
        {
            llvm::Value* result = B.CreateFNeg(scalar.GetValue());
            return Value(result, result->getType());
        }
        else if (log) {
            log->error("unrecognised scalar type", node);
        }
    }
    else if (this->IsArray())
    {
        std::vector<Value> elements;
        this->ArrayToScalars(B, elements, /*load=*/true);
        for (Value& element : elements) element = element.Negate(B);
        return Value::ScalarsToArray(B, elements);
    }

    if (log) log->error("value is not a scalar or vector for unary '-'", node);
    return Value::Invalid();
}

Value Value::Increment(llvm::IRBuilder<>& B, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsBool() || (!this->IsInteger() && !this->IsFloat())) {
        if (log) log->error("variable is an unsupported type for "
                "increment. Must be a non-boolean scalar", node);
        return Value::Invalid();
    }
    return this->TrivialBinary(B, this->One(), ast::tokens::PLUS, log, node);
}

Value Value::Decrement(llvm::IRBuilder<>& B, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsBool() || (!this->IsInteger() && !this->IsFloat())) {
        if (log) log->error("variable is an unsupported type for "
                "decrement. Must be a non-boolean scalar", node);
        return Value::Invalid();
    }
    return this->TrivialBinary(B, this->One(), ast::tokens::MINUS, log, node);
}

Value Value::Add(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    const Value& lhs = *this;

    if (lhs.IsString() && rhs.IsString())
    {
        auto reg = internal::GetMappedFunctionRegistry(&B.getContext());
        OPENVDB_ASSERT(reg.first);
        const FunctionGroup* axstringplus = reg.first->getOrInsert("string::op+", reg.second, /*internal*/true);
        return axstringplus->execute(NativeArguments{lhs, rhs}, B);
    }
    return this->TrivialBinary(B, rhs, ast::tokens::PLUS, log, node);
}

Value Value::Subtract(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    if (this->IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::MINUS, log, node);
    }
    return this->TrivialBinary(B, rhs, ast::tokens::MINUS, log, node);
}

Value Value::Multiply(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    Value lhs = *this;

    if (lhs.IsScalar() && rhs.IsMatrix()) {
        const size_t size = rhs.GetArrayNumElements();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix4(B); }
    }
    if (lhs.IsMatrix() && rhs.IsScalar()) {
        const size_t size = lhs.GetArrayNumElements();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix4(B); }
    }
    if ((lhs.IsMatrix() && rhs.IsArray()) ||
        (lhs.IsArray() &&  rhs.IsMatrix())) // matrix <op> matrix/vector or matrix/vector <op> matrix
    {
        const size_t lsize = lhs.GetArrayNumElements();
        const size_t rsize = rhs.GetArrayNumElements();
        auto reg = internal::GetMappedFunctionRegistry(&B.getContext());
        OPENVDB_ASSERT(reg.first);

        // The function framework will handle the appropriate casting
        // @todo  We should probably just do it here instead
        if ((lsize == 9 && rsize == 9) ||
            (lsize == 16 && rsize == 16))
        {
            // matrix matrix multiplication all handled through mmmult
            return reg.first->getOrInsert("mmmult", reg.second, /*internal*/true)->execute(NativeArguments{lhs, rhs}, B);
        }
        else if ((lsize ==  9 && rsize == 3) ||
                 (lsize == 16 && rsize == 3) ||
                 (lsize == 16 && rsize == 4)) {
            // matrix vector multiplication all handled through pretransform
            return reg.first->getOrInsert("pretransform", reg.second, /*internal*/false)->execute(NativeArguments{lhs, rhs}, B);
        }
        else if ((lsize == 3 && rsize ==  9) ||
                 (lsize == 3 && rsize == 16) ||
                 (lsize == 4 && rsize == 16)) {
            // vector matrix multiplication all handled through transform
            return reg.first->getOrInsert("transform", reg.second, /*internal*/false)->execute(NativeArguments{lhs, rhs}, B);
        }

        if (log) log->error("unsupported * operator on vector/matrix sizes", node);
        return Value::Invalid();
    }

    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::MULTIPLY, log, node);
    }
    // all other binary types
    return lhs.TrivialBinary(B, rhs, ast::tokens::MULTIPLY, log, node);
}

Value Value::Divide(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    const Value& lhs = *this;

    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::DIVIDE, log, node);
    }
    if (lhs.IsMatrix() || rhs.IsMatrix()) {
        return Value::WUnsupportedOp("matrix", ast::tokens::DIVIDE, log, node);
    }
    if (llvm::Constant* c = llvm::dyn_cast<llvm::Constant>(rhs.GetValue())) {
        if (c->isZeroValue()) {
            if (log && !log->warning("division by zero is undefined", node)) return Value::Invalid();
        }
    }
    return lhs.TrivialBinary(B, rhs, ast::tokens::DIVIDE, log, node);
}

Value Value::Modulo(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    Value lhs = *this;

    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::MODULO, log, node);
    }
    if (lhs.IsMatrix() || rhs.IsMatrix())
    {
        if (log) log->error("unsupported matrix operation \"%\"", node);
        return Value::Invalid();
    }
    if (llvm::Constant* c = llvm::dyn_cast<llvm::Constant>(rhs.GetValue())) {
        if (c->isZeroValue()) {
            if (log && !log->warning("modulo by zero is undefined", node)) return Value::Invalid();
        }
    }

    if (lhs.IsScalar() && rhs.IsScalar()) // scalar <op> scalar
    {
        llvm::Type* precision = typePrecedence(lhs.GetUnderlyingType(), rhs.GetUnderlyingType());
        // if bool, the lowest precision and subsequent result should be int32
        // for arithmetic, bitwise and certain other ops
        if (precision->isIntegerTy(1)) precision = LLVMType<int32_t>::get(B.getContext());
        lhs = lhs.CastToPrecision(B, precision).LoadIfPtr(B);
        rhs = rhs.CastToPrecision(B, precision).LoadIfPtr(B);

        // handle floored modulo
        auto reg = internal::GetMappedFunctionRegistry(&B.getContext());
        OPENVDB_ASSERT(reg.first);
        const FunctionGroup* mod = reg.first->getOrInsert("floormod", reg.second, /*internal*/true);
        OPENVDB_ASSERT(mod);
        return mod->execute(NativeArguments{lhs,rhs}, B);
    }
    if (lhs.IsScalar() && rhs.IsVector()) // scalar <op> vec
    {
        const size_t size = rhs.GetArrayNumElements();
        Value lhs = Value::Invalid();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix4(B); }
        else                 lhs = lhs.ScalarToArray(B, size);
        return lhs.Modulo(B, rhs, log, node); // recurse
    }
    if (lhs.IsVector() && rhs.IsScalar()) // vec <op> scalar
    {
        const size_t size = lhs.GetArrayNumElements();
        Value rhs = Value::Invalid();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix4(B); }
        else                 rhs = rhs.ScalarToArray(B, size);
        return lhs.Modulo(B, rhs, log, node);
    }
    if (lhs.IsVector() && rhs.IsVector()) // vec <op> vec
    {
        if (lhs.GetArrayNumElements() != rhs.GetArrayNumElements())
        {
            if (log) log->error("unsupported binary operator on vector/matrix "
                "arguments of mismatching sizes", node);
            return Value::Invalid();
        }

        const size_t size = lhs.GetArrayNumElements();

        std::vector<Value> elements;
        elements.reserve(size);
        for (size_t i = 0; i < size; ++i)
        {
            Value left  = lhs.GetArrayElement(B, i);
            Value right = rhs.GetArrayElement(B, i);
            elements.emplace_back(left.Modulo(B, right, log, node));
        }

        llvm::Type* arrayTy = llvm::ArrayType::get(elements.front().GetUnderlyingScalarType(), size);
        Value result = Value::Alloc(B, arrayTy);
        for (size_t i = 0; i < size; ++i) {
            B.CreateStore(elements[i].GetValue(), result.GetArrayElement(B, i).GetValue());
        }
        return result;
    }

    if (log) log->error("unsupported implicit cast in binary op", node);
    return Value::Invalid();
}

Value Value::TruncatedModulo(llvm::IRBuilder<>& B, Value rhs) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    OPENVDB_ASSERT(rhs.IsScalar());
    OPENVDB_ASSERT(this->IsScalar());

    llvm::Value* result = binaryOperator(this->LoadIfPtr(B).GetValue(),
        rhs.LoadIfPtr(B).GetValue(), ast::tokens::MODULO, B);
    return Value(result, result->getType());
}

Value Value::Equals(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    const Value& lhs = *this;

    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::EQUALSEQUALS, log, node);
    }

    Value result = lhs.TrivialBinary(B, rhs, ast::tokens::EQUALSEQUALS, log, node);
    OPENVDB_ASSERT(result);
    if (result.IsBool()) return result;
    OPENVDB_ASSERT(result.IsArray()); // array of bool comparisons
    return result.ReduceBoolArray(B, ast::tokens::AND);
}

/// @brief  Emit IR to perform a != operation on two values
Value Value::NotEquals(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    const Value& lhs = *this;

    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, ast::tokens::NOTEQUALS, log, node);
    }
    Value result = lhs.TrivialBinary(B, rhs, ast::tokens::NOTEQUALS, log, node);
    OPENVDB_ASSERT(result);
    if (result.IsBool()) return result;
    OPENVDB_ASSERT(result.IsArray()); // array of bool comparisons
    return result.ReduceBoolArray(B, ast::tokens::OR);
}

Value Value::Assign(llvm::IRBuilder<>& B, Value rhs, Logger* log, const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    OPENVDB_ASSERT(this->IsPtr());
    Value lhs = *this;

    if (!lhs.IsPtr()) {
        // AX grammar should ensure this is never possible
        if (log) log->error("Cannot assign to a literal value", node);
        return Value::Invalid();
    }

    if (lhs.IsString() && rhs.IsString())
    {
        auto reg = internal::GetMappedFunctionRegistry(&B.getContext());
        OPENVDB_ASSERT(reg.first);
        const FunctionGroup* axstringassign = reg.first->getOrInsert("string::op=", reg.second, /*internal*/true);
        axstringassign->execute(NativeArguments{lhs, rhs}, B);
        return lhs;
    }

    OPENVDB_ASSERT(!(this->IsString() && rhs.IsString()));

    size_t lsize = lhs.IsArray() ? lhs.GetArrayNumElements() : 1;
    size_t rsize = rhs.IsArray() ? rhs.GetArrayNumElements() : 1;

    // Handle scalar->matrix promotion if necessary
    // @todo promote all values (i.e. scalar to vectors) to make below branching
    // easier. Need to verifier IR is able to optimise to the same logic

    if (lhs.IsMatrix() && rhs.IsScalar())
    {
        rhs = rhs.CastToPrecision(B, lhs.GetUnderlyingScalarType());
        if (lhs.GetArrayNumElements() == 9)       rhs = rhs.ScalarToIdentMatrix3(B);
        else if (lhs.GetArrayNumElements() == 16) rhs = rhs.ScalarToIdentMatrix4(B);
        else {
            OPENVDB_ASSERT(false);
        }
        rsize = lsize;
    }

    if (lsize != rsize) {
        if (lsize > 1 && rsize > 1) {
            if (log) log->error("unable to assign vector/array "
                "attributes with mismatching sizes", node);
            return Value::Invalid();
        }
        else if (lsize == 1) {
            OPENVDB_ASSERT(rsize > 1);
            if (log) log->error("cannot assign a scalar value "
                "from a vector or matrix. Consider using the [] operator to "
                "get a particular element", node);
            return Value::Invalid();
        }
    }

    // All remaining operators are either componentwise or invalid implicit casts

    const bool componentwise =
        (rhs.IsScalar() || rhs.IsArray()) &&
        (lhs.IsScalar() || lhs.IsArray());

    if (componentwise)
    {
        OPENVDB_ASSERT(rsize == lsize || (rsize == 1 || lsize == 1));
        const size_t resultsize = std::max(lsize, rsize);

        // compute the componentwise precision
        llvm::Type* targetType = lhs.GetUnderlyingScalarType();
        if (targetType != rhs.GetUnderlyingScalarType()) {
            llvm::Type* highest = typePrecedence(targetType, rhs.GetUnderlyingScalarType());
            if (highest != targetType) {
                if (log && !log->warning("implicit conversion in assignment (possible truncation)", node)) return Value::Invalid();
            }
        }

        // if target is bool, perform standard boolean conversion (*not* truncation).
        // i.e. if rhs is anything but zero, lhs is true
        // @todo zeroval should be at rhstype
        if (lhs.IsBool())
        {
            Value zero(LLVMType<int32_t>::get(B.getContext(), 0), LLVMType<int32_t>::get(B.getContext()));
            Value newRhs = zero.NotEquals(B, rhs, log, node);
            OPENVDB_ASSERT(newRhs);
            if (!newRhs) return Value::Invalid();
            rhs = newRhs;
        }

        for (size_t i = 0; i < resultsize; ++i) {
            Value lelement = lsize == 1 ? lhs : lhs.GetArrayElement(B, i);
            Value relement = rsize == 1 ? rhs : rhs.GetArrayElement(B, i).Load(B);
            relement = relement.CastToPrecision(B, targetType); // doesn't load if precision matches
            relement = relement.LoadIfPtr(B);
            B.CreateStore(relement.GetValue(), lelement.GetValue());
        }
    }
    else {
        if (log) log->error("unsupported implicit cast in assignment", node);
        return Value::Invalid();
    }
    OPENVDB_ASSERT(lhs);
    return lhs;
}

void Value::Print() const
{
    if (!mVal) llvm::errs() << "No value";
    else {
        mVal->print(llvm::errs()); llvm::errs() << '\n';
        mVal->getType()->print(llvm::errs());
    }
    llvm::errs() << '\n';
    if (!mUType) llvm::errs() << "No UType";
    else         mUType->print(llvm::errs());
    llvm::errs() << '\n';
}

Value Value::TrivialBinary(llvm::IRBuilder<>& B,
    Value rhs,
    const ast::tokens::OperatorToken& op,
    Logger* log,
    const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    Value lhs = *this;

    OPENVDB_ASSERT(!(op == ast::tokens::MULTIPLY && lhs.IsScalar() && rhs.IsMatrix()));
    OPENVDB_ASSERT(!(op == ast::tokens::MULTIPLY && lhs.IsMatrix() && rhs.IsScalar()));
    OPENVDB_ASSERT(!(op == ast::tokens::MULTIPLY && lhs.IsVector() && rhs.IsMatrix()));
    OPENVDB_ASSERT(!(op == ast::tokens::MULTIPLY && lhs.IsMatrix() && rhs.IsVector()));
    OPENVDB_ASSERT(!(op == ast::tokens::MULTIPLY && lhs.IsMatrix() && rhs.IsMatrix()));

    if (lhs.IsScalar() && rhs.IsScalar()) // scalar <op> scalar
    {
        llvm::Type* precision = typePrecedence(lhs.GetUnderlyingType(), rhs.GetUnderlyingType());
        // if bool, the lowest precision and subsequent result should be int32
        // for arithmetic, bitwise and certain other ops
        if (precision->isIntegerTy(1)) {
            const ast::tokens::OperatorType opType = ast::tokens::operatorType(op);
            if (opType == ast::tokens::ARITHMETIC ||
                opType == ast::tokens::BITWISE ||
                op == ast::tokens::MORETHAN ||
                op == ast::tokens::LESSTHAN ||
                op == ast::tokens::MORETHANOREQUAL ||
                op == ast::tokens::LESSTHANOREQUAL) {
                precision = LLVMType<int32_t>::get(B.getContext());
            }
        }
        lhs = lhs.CastToPrecision(B, precision).LoadIfPtr(B);
        rhs = rhs.CastToPrecision(B, precision).LoadIfPtr(B);
        const BinaryFunction llvmBinaryFunction = llvmBinaryConversion(precision, op);
        llvm::Value* result = llvmBinaryFunction(B, lhs.GetValue(), rhs.GetValue());
        return Value(result, result->getType());
    }
    if (lhs.IsScalar() && rhs.IsArray()) // scalar <op> vec
    {
        const size_t size = rhs.GetArrayNumElements();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); lhs = lhs.ScalarToIdentMatrix4(B); }
        else                 lhs = lhs.ScalarToArray(B, size);
        return lhs.TrivialBinary(B, rhs, op, log, node);
    }
    if (lhs.IsArray() && rhs.IsScalar()) // vec <op> scalar
    {
        const size_t size = lhs.GetArrayNumElements();
        if (size == 9)       { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix3(B); }
        else if (size == 16) { if (!WImplicitScalarToMatrix(log, node)) return Value::Invalid(); rhs = rhs.ScalarToIdentMatrix4(B); }
        else                 rhs = rhs.ScalarToArray(B, size);
        return lhs.TrivialBinary(B, rhs, op, log, node);
    }
    if (lhs.IsArray() && rhs.IsArray()) // vec <op> vec
    {
        if (lhs.GetArrayNumElements() != rhs.GetArrayNumElements())
        {
            if (log) log->error("unsupported binary operator on vector/matrix "
                "arguments of mismatching sizes", node);
            return Value::Invalid();
        }

        const size_t size = lhs.GetArrayNumElements();

        std::vector<Value> elements;
        elements.reserve(size);
        for (size_t i = 0; i < size; ++i)
        {
            Value left  = lhs.GetArrayElement(B, i);
            Value right = rhs.GetArrayElement(B, i);
            elements.emplace_back(left.TrivialBinary(B, right, op, log, node));
        }

        llvm::Type* arrayTy = llvm::ArrayType::get(elements.front().GetUnderlyingScalarType(), size);
        Value result = Value::Alloc(B, arrayTy);
        for (size_t i = 0; i < size; ++i) {
            OPENVDB_ASSERT(elements[i]);
            OPENVDB_ASSERT(!elements[i].IsPtr());
            OPENVDB_ASSERT(elements[i].IsScalar());
            B.CreateStore(elements[i].GetValue(), result.GetArrayElement(B, i).GetValue());
        }
        return result;
    }

    if (log) log->error("unsupported implicit cast in binary op", node);
    return Value::Invalid();
}

Value Value::Bitwise(llvm::IRBuilder<>& B,
    Value rhs,
    const ast::tokens::OperatorToken& op,
    Logger* log,
    const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    OPENVDB_ASSERT(ast::tokens::operatorType(op) == ast::tokens::BITWISE);
    Value lhs = *this;

    if (lhs.IsString() || rhs.IsString())
    {
        if (log) log->error("unsupported string operation \"" +
            ast::tokens::operatorNameFromToken(op) + "\"", node);
        return Value::Invalid();
    }
    if (lhs.IsArray() || rhs.IsArray())
    {
        if (log) log->error("unsupported vector/matrix operation \"" +
            ast::tokens::operatorNameFromToken(op) + "\"", node);
        return Value::Invalid();
    }
    if (lhs.IsFloat() || rhs.IsFloat())
    {
        if (log) log->error("unsupported operation for floating points \"" +
            ast::tokens::operatorNameFromToken(op) + "\"", node);
        return Value::Invalid();
    }
    return lhs.TrivialBinary(B, rhs, op, log, node);
}

Value Value::Relational(llvm::IRBuilder<>& B,
    Value rhs,
    const ast::tokens::OperatorToken& op,
    Logger* log,
    const ax::ast::Node* node) const
{
    OPENVDB_ASSERT(*this);
    OPENVDB_ASSERT(rhs);
    OPENVDB_ASSERT(ast::tokens::operatorType(op) == ast::tokens::RELATIONAL);
    // These are handled uniquely
    OPENVDB_ASSERT(op != ast::tokens::EQUALS);
    OPENVDB_ASSERT(op != ast::tokens::NOTEQUALS);

    const Value& lhs = *this;
    if (lhs.IsString() || rhs.IsString()) {
        return Value::WUnsupportedOp(ast::tokens::STRING, op, log, node);
    }
    if (lhs.IsArray() || rhs.IsArray()) {
        return Value::WUnsupportedOp("vector/matrix", op, log, node);
    }
    return lhs.TrivialBinary(B, rhs, op, log, node);
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
