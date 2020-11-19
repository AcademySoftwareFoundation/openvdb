// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/Types.h
///
/// @authors Nick Avramoussis
///
/// @brief  Consolidated llvm types for most supported types
///

#ifndef OPENVDB_AX_CODEGEN_TYPES_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_TYPES_HAS_BEEN_INCLUDED

#include "../ast/Tokens.h"
#include "../Exceptions.h"
#include "../compiler/CustomData.h" // for AXString

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <openvdb/math/Mat3.h>
#include <openvdb/math/Mat4.h>
#include <openvdb/math/Vec3.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>

#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

template <size_t Bits> struct int_t;
template <> struct int_t<8>  { using type = int8_t;  };
template <> struct int_t<16> { using type = int16_t; };
template <> struct int_t<32> { using type = int32_t; };
template <> struct int_t<64> { using type = int64_t; };

/// @brief LLVM type mapping from pod types
/// @note  LLVM Types do not store information about the value sign, only meta
///        information about the primitive type (i.e. float, int, pointer) and
///        the precision width. LLVMType<uint64_t>::get(C) will provide the same
///        type as LLVMType<int64_t>::get(C), however sign is taken into account
///        during construction of LLVM constants.
/// @note  LLVMType classes are importantly used to provided automatic external
///        function mapping. Note that references are not supported, pointers
///        should be used instead.
/// @note  Provide your own custom class mapping by specializing the below.
template <typename T>
struct LLVMType
{
    static_assert(!std::is_reference<T>::value,
        "Reference types/arguments are not supported for automatic "
        "LLVM Type conversion. Use pointers instead.");
    static_assert(!std::is_class<T>::value,
        "Object types/arguments are not supported for automatic "
        "LLVM Type conversion.");

    /// @brief  Return an LLVM type which represents T
    /// @param C  The LLVMContext to request the Type from.
    static inline llvm::Type*
    get(llvm::LLVMContext& C)
    {
        // @note bools always treated as i1 values as the constants
        // true and false from the IRBuilder are i1
        if (std::is_same<T, bool>::value) {
            return llvm::Type::getInt1Ty(C);
        }

#if LLVM_VERSION_MAJOR > 6
        return llvm::Type::getScalarTy<T>(C);
#else
        int bits = sizeof(T) * CHAR_BIT;
        if (std::is_integral<T>::value) {
            return llvm::Type::getIntNTy(C, bits);
        }
        else if (std::is_floating_point<T>::value) {
            switch (bits) {
                case 32: return llvm::Type::getFloatTy(C);
                case 64: return llvm::Type::getDoubleTy(C);
            }
        }
        OPENVDB_THROW(AXCodeGenError, "LLVMType called with an unsupported type \"" +
            std::string(typeNameAsString<T>()) + "\".");
#endif
    }

    /// @brief  Return an LLVM constant Value which represents T value
    /// @param C  The LLVMContext
    /// @param V  The value to convert to an LLVM constant
    /// @return  If successful, returns a pointer to an LLVM constant which
    ///          holds the value T.
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T V)
    {
        llvm::Type* type = LLVMType<T>::get(C);
        llvm::Constant* constant = nullptr;

        if (std::is_floating_point<T>::value) {
            assert(llvm::ConstantFP::isValueValidForType(type,
                llvm::APFloat(static_cast<typename std::conditional
                    <std::is_floating_point<T>::value, T, double>::type>(V))));
            constant = llvm::ConstantFP::get(type, static_cast<double>(V));
        }
        else if (std::is_integral<T>::value) {
            const constexpr bool isSigned = std::is_signed<T>::value;
            assert((isSigned && llvm::ConstantInt::isValueValidForType(type, static_cast<int64_t>(V))) ||
                   (!isSigned && llvm::ConstantInt::isValueValidForType(type, static_cast<uint64_t>(V))));
            constant = llvm::ConstantInt::get(type, static_cast<uint64_t>(V), isSigned);
        }

        assert(constant);
        return constant;
    }

    /// @brief  Return an LLVM constant which holds an uintptr_t, representing
    ///         the current address of the given value.
    /// @param C  The LLVMContext
    /// @param V  The address of a given type to convert to an LLVM constant
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T* const V)
    {
        return LLVMType<uintptr_t>::get(C,
            reinterpret_cast<uintptr_t>(V));
    }
};

template <typename T, size_t S>
struct LLVMType<T[S]>
{
    static_assert(S != 0,
        "Zero size array types are not supported for automatic LLVM "
        "Type conversion");

    static inline llvm::Type*
    get(llvm::LLVMContext& C) {
        return llvm::ArrayType::get(LLVMType<T>::get(C), S);
    }
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T(&array)[S]) {
        return llvm::ConstantDataArray::get(C, array);
    }
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T(*array)[S])
    {
        return LLVMType<uintptr_t>::get(C,
            reinterpret_cast<uintptr_t>(array));
    }
};

template <typename T>
struct LLVMType<T*>
{
    static inline llvm::PointerType*
    get(llvm::LLVMContext& C) {
        return LLVMType<T>::get(C)->getPointerTo(0);
    }
};

template <>
struct LLVMType<char> : public LLVMType<uint8_t>
{
    static_assert(std::is_same<uint8_t, unsigned char>::value,
        "This library requires std::uint8_t to be implemented as unsigned char.");
};

template <>
struct LLVMType<AXString>
{
    static inline llvm::StructType*
    get(llvm::LLVMContext& C) {
        const std::vector<llvm::Type*> types {
            LLVMType<char*>::get(C),  // array
            LLVMType<AXString::SizeType>::get(C) // size
        };
        return llvm::StructType::get(C, types);
    }
    static inline llvm::Value*
    get(llvm::LLVMContext& C, llvm::Constant* string, llvm::Constant* size) {
        return llvm::ConstantStruct::get(LLVMType<AXString>::get(C), {string, size});
    }
    /// @note Creating strings from a literal requires a GEP instruction to
    ///   store the string ptr on the struct.
    /// @note Usually you should be using s = builder.CreateGlobalStringPtr()
    ///   followed by LLVMType<AXString>::get(C, s) rather than allocating
    ///   a non global string
    static inline llvm::Value*
    get(llvm::LLVMContext& C, const std::string& string, llvm::IRBuilder<>& builder) {
        llvm::Constant* constant =
            llvm::ConstantDataArray::getString(C, string, /*terminator*/true);
        llvm::Constant* size = llvm::cast<llvm::Constant>
            (LLVMType<AXString::SizeType>::get(C, static_cast<AXString::SizeType>(string.size())));
        llvm::Value* zero = LLVMType<int32_t>::get(C, 0);
        llvm::Value* args[] = { zero, zero };
        constant = llvm::cast<llvm::Constant>
            (builder.CreateInBoundsGEP(constant->getType(), constant, args));
        return LLVMType<AXString>::get(C, constant, size);
    }
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const AXString* const string)
    {
        return LLVMType<uintptr_t>::get(C,
            reinterpret_cast<uintptr_t>(string));
    }
};

template <>
struct LLVMType<void>
{
    static inline llvm::Type*
    get(llvm::LLVMContext& C) {
        return llvm::Type::getVoidTy(C);
    }
};

/// @note void* implemented as signed int_t* to match clang IR generation
template <> struct LLVMType<void*> : public LLVMType<int_t<sizeof(void*)>::type*> {};

template <typename T> struct LLVMType<const T> : public LLVMType<T> {};
template <typename T> struct LLVMType<const T*> : public LLVMType<T*> {};

/// @brief  Alias mapping between two types, a frontend type T1 and a backend
///         type T2. This class is the intended interface for binding objects
///         which implement supported backend AX/IR types to this given backend
///         type. More specifically, it's current and expected usage is limited
///         to objects which hold a single member of a supported backend type
///         and implements a StandardLayoutType as defined by the standard.
///         Fundamentally, T1->T2 mapping should be supported by
///         reinterpret_cast<> as defined by the type aliasing rules.
/// @note   The static asserts provide preliminary checks but are by no means
///         a guarantee that a provided mapping is correct. Ensure the above
///         requirements are met when instantiating an alias.
template <typename T1, typename T2>
struct AliasTypeMap
{
    using LLVMTypeT = LLVMType<T2>;

    static_assert(sizeof(T1) == sizeof(T2),
        "T1 differs in size to T2 during alias mapping. Types should have "
        "the same memory layout.");
    static_assert(std::is_standard_layout<T1>::value,
        "T1 in instantiation of an AliasTypeMap does not have a standard layout. "
        "This will most likely cause undefined behaviour when attempting to map "
        "T1->T2.");

    static inline llvm::Type*
    get(llvm::LLVMContext& C) {
        return LLVMTypeT::get(C);
    }
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T1& value) {
        return LLVMTypeT::get(C, reinterpret_cast<const T2&>(value));
    }
    static inline llvm::Constant*
    get(llvm::LLVMContext& C, const T1* const value) {
        return LLVMTypeT::get(C, reinterpret_cast<const T2* const>(value));
    }
};

/// @brief  Supported aliasing for VDB math types, allowing use in external
///         function signatures.
template <typename T> struct LLVMType<math::Vec2<T>> : public AliasTypeMap<math::Vec2<T>, T[2]> {};
template <typename T> struct LLVMType<math::Vec3<T>> : public AliasTypeMap<math::Vec3<T>, T[3]> {};
template <typename T> struct LLVMType<math::Vec4<T>> : public AliasTypeMap<math::Vec4<T>, T[4]> {};
template <typename T> struct LLVMType<math::Mat3<T>> : public AliasTypeMap<math::Mat3<T>, T[9]> {};
template <typename T> struct LLVMType<math::Mat4<T>> : public AliasTypeMap<math::Mat4<T>, T[16]> {};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/// @brief  Templated function traits which provides compile-time index access to
///         the types of the function signature
///
template<typename SignatureT>
struct FunctionTraits;

template<typename R, typename... Args>
struct FunctionTraits<R(&)(Args...)> : public FunctionTraits<R(Args...)> {};

template<typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)> : public FunctionTraits<R(Args...)> {};

template<typename ReturnT, typename ...Args>
struct FunctionTraits<ReturnT(Args...)>
{
    using ReturnType = ReturnT;
    using SignatureType = ReturnType(Args...);
    static const size_t N_ARGS = sizeof...(Args);

    template <size_t I>
    struct Arg
    {
    public:
        static_assert(I < N_ARGS,
            "Invalid index specified for function argument access");
        using Type = typename std::tuple_element<I, std::tuple<Args...>>::type;
        static_assert(!std::is_reference<Type>::value,
            "Reference types/arguments are not supported for automatic "
            "LLVM Type conversion. Use pointers instead.");
    };
};

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

/// @brief Returns an llvm Constant holding a scalar value
/// @param t  The scalar constant
/// @param type  The LLVM type. Can differ from the type of t, in which
///   case the value will be cast to the llvm type
///
template <typename T>
inline llvm::Constant*
llvmConstant(const T t, llvm::Type* type)
{
    static_assert(std::is_floating_point<T>::value || std::is_integral<T>::value,
        "T type for llvmConstant must be a floating point or integral type.");

    if (type->isIntegerTy()) {
        return llvm::ConstantInt::get(type, static_cast<uint64_t>(t), /*signed*/true);
    }
    else {
        assert(type->isFloatingPointTy());
        return llvm::ConstantFP::get(type, static_cast<double>(t));
    }
}

/// @brief Returns an llvm IntegerType given a requested size and context
/// @param size  The number of bits of the integer type
/// @param C     The LLVMContext to request the Type from.
///
llvm::IntegerType* llvmIntType(const uint32_t size, llvm::LLVMContext& C);

/// @brief Returns an llvm floating point Type given a requested size and context
/// @param size  The size of the float to request, i.e. float - 32, double - 64 etc.
/// @param C     The LLVMContext to request the Type from.
///
llvm::Type* llvmFloatType(const uint32_t size, llvm::LLVMContext& C);

/// @brief  Returns an llvm type representing a type defined by a string.
/// @note   For string types, this function returns the element type, not the
///         object type! The llvm type representing a char block of memory
///         is LLVMType<char*>::get(C);
/// @param type  The AX token type
/// @param C     The LLVMContext to request the Type from.
///
llvm::Type* llvmTypeFromToken(const ast::tokens::CoreType& type, llvm::LLVMContext& C);

/// @brief  Return a corresponding AX token which represents the given LLVM Type.
/// @note   If the type does not exist in AX, ast::tokens::UNKNOWN is returned.
///         Must not be a nullptr.
/// @param type  a valid LLVM Type
///
ast::tokens::CoreType tokenFromLLVMType(const llvm::Type* type);

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_TYPES_HAS_BEEN_INCLUDED

