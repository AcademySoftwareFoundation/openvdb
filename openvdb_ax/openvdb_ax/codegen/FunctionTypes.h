// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/FunctionTypes.h
///
/// @authors Nick Avramoussis
///
/// @brief  Contains frameworks for creating custom AX functions which can
///   be registered within the FunctionRegistry and used during code
///   generation. The intended and safest way to build a function is to
///   use the FunctionBuilder struct with its addSignature methods. Note
///   that the derived Function classes provided can also be subclassed
///   for more granular control, however may be subject to more substantial
///   API changes.
///
/// @details  There are a variety of different ways to build a function
///   which are tailored towards different function types. The two currently
///   supported function implementations are C Bindings and IR generation.
///   Additionally, depending on the return type of the function, you may
///   need to declare your function an SRET (structural return) function.
///
///   C Bindings:
///     As the name suggests, the CFunction class infrastructure provides
///     the quickest and easiest way to bind to methods in your host
///     application. The most important thing to consider when choosing
///     this approach is performance. LLVM will have no knowledge of the
///     function body during optimization passes. Depending on the
///     implementation of your method and the user's usage from AX, C
///     bindings may be subject to limited optimizations in comparison to
///     IR functions. For example, a static function which is called from
///     within a loop cannot be unrolled. See the CFunction templated
///     class.
///
///   IR Functions:
///     IR Functions expect implementations to generate the body of the
///     function directly into IR during code generation. This ensures
///     optimal performance during optimization passes however can be
///     trickier to design. Note that, in the future, AX functions will
///     be internally supported to provide a better solution for
///     IR generated functions. See the IRFunction templated class.
///
///   SRET Functions:
///     Both C Bindings and IR Functions can be marked as SRET methods.
///     SRET methods, in AX, are any function which returns a value which
///     is not a scalar (e.g. vectors, matrices). This follows the same
///     optimization logic as clang which will rebuild function signatures
///     with their return type as the first argument if the return type is
///     greater than a given size. You should never attempt to return
///     alloca's directly from functions (unless malloced).
///
///   Some other things to consider:
///     - Ensure C Binding dependencies have been correctly mapped.
///     - Avoid calling B.CreateAlloca inside of IR functions - instead
///       rely on the utility method insertStaticAlloca() where possible.
///     - Ensure both floating point and integer argument signatures are
///       provided if you wish to avoid floats truncating.
///     - Array arguments (vectors/matrices) are always passed by pointer.
///       Scalar arguments are always passed by copy.
///     - Ensure array arguments which will not be modified are marked as
///       readonly. Currently, only array arguments can be passed by
///       "reference".
///     - Ensure function bodies, return types and parameters and marked
///       with desirable llvm attributes.
///

#ifndef OPENVDB_AX_CODEGEN_FUNCTION_TYPES_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_FUNCTION_TYPES_HAS_BEEN_INCLUDED

#include "Types.h"
#include "Value.h"
#include "Utils.h" // isValidCast
#include "ConstantFolding.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <stack>
#include <type_traits>
#include <map>
#include <vector>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

struct Function; // fwd

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief  Typedef a stack allocated array with malloc grow support for
///   anything which is relatively small and bases its container size on the
///   number of function arguments.
/// @note   LLVM computes N as 3 (currently) for Value types, but we explicitly
///   set this as this is a well-motivates choice for average/median amount of
///   function arguments in builtin AX functions.
template <typename T>
using SmallArgumentVector = llvm::SmallVector<T, 3>;

/// @brief  Object to array conversion methods to allow functions to return
///         vector types. These containers provided an interface for automatic
///         conversion of C++ objects to LLVM types as array types.

template <typename T, size_t _SIZE = 1>
struct ArgType {
    using Type = T;
    static const size_t SIZE = _SIZE;
    using ArrayType = Type[SIZE];
    ArrayType mmArgs;
};

template <typename T, size_t S>
struct LLVMType<ArgType<T,S>> : public AliasTypeMap<ArgType<T,S>, T[S]> {};

using V2D = ArgType<double, 2>;
using V2F = ArgType<float, 2>;
using V2I = ArgType<int32_t, 2>;
using V3D = ArgType<double, 3>;
using V3F = ArgType<float, 3>;
using V3I = ArgType<int32_t, 3>;
using V4D = ArgType<double, 4>;
using V4F = ArgType<float, 4>;
using V4I = ArgType<int32_t, 4>;
using M3D = ArgType<double, 9>;
using M3F = ArgType<float, 9>;
using M4D = ArgType<double, 16>;
using M4F = ArgType<float, 16>;

////////////////////////////////////////////////////////////////////////////////

/// @brief  Type to symbol conversions - these characters are used to build each
///         functions unique signature. They differ from standard AX or LLVM
///         syntax to be as short as possible i.e. vec4d, [4 x double] = d4

template <typename T> struct TypeToSymbol { static inline std::string s() { return "?"; } };
template <> struct TypeToSymbol<void> { static inline std::string s() { return "v"; } };
template <> struct TypeToSymbol<char> { static inline std::string s() { return "c"; } };
template <> struct TypeToSymbol<uint8_t>  { static inline std::string s() { return "u8"; } };
template <> struct TypeToSymbol<uint16_t> { static inline std::string s() { return "us"; } };
template <> struct TypeToSymbol<uint32_t> { static inline std::string s() { return "ui"; } };
template <> struct TypeToSymbol<uint64_t> { static inline std::string s() { return "ul"; } };
template <> struct TypeToSymbol<int8_t>  { static inline std::string s() { return "8"; } };
template <> struct TypeToSymbol<int16_t> { static inline std::string s() { return "s"; } };
template <> struct TypeToSymbol<int32_t> { static inline std::string s() { return "i"; } };
template <> struct TypeToSymbol<int64_t> { static inline std::string s() { return "l"; } };
template <> struct TypeToSymbol<float> { static inline std::string s() { return "f"; } };
template <> struct TypeToSymbol<double> { static inline std::string s() { return "d"; } };
template <> struct TypeToSymbol<codegen::String> { static inline std::string s() { return "a"; } };

template <typename T>
struct TypeToSymbol<T*> {
    static inline std::string s() { return TypeToSymbol<T>::s() + "*"; }
};

template <typename T, size_t S>
struct TypeToSymbol<T[S]> {
    static inline std::string s() { return TypeToSymbol<T>::s() + std::to_string(S); }
};

template <typename T, size_t S> struct TypeToSymbol<ArgType<T,S>> : public TypeToSymbol<T[S]> {};
template <typename T> struct TypeToSymbol<math::Vec2<T>> : public TypeToSymbol<T[2]> {};
template <typename T> struct TypeToSymbol<math::Vec3<T>> : public TypeToSymbol<T[3]> {};
template <typename T> struct TypeToSymbol<math::Vec4<T>> : public TypeToSymbol<T[4]> {};
template <typename T> struct TypeToSymbol<math::Mat3<T>> : public TypeToSymbol<T[9]> {};
template <typename T> struct TypeToSymbol<math::Mat4<T>> : public TypeToSymbol<T[16]> {};
template <typename T> struct TypeToSymbol<const T> : public TypeToSymbol<T> {};
template <typename T> struct TypeToSymbol<const T*> : public TypeToSymbol<T*> {};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief  Templated argument iterator which implements various small functions
///         per argument type, resolved at compile time.
///
template <typename SignatureT, size_t I = FunctionTraits<SignatureT>::N_ARGS>
struct ArgumentIterator
{
    using ArgT = typename FunctionTraits<SignatureT>::template Arg<I-1>;
    using ArgumentValueType = typename ArgT::Type;

    /// @brief  Whether this signature contains types that are representable
    ///   in AX's Value type.
    static const bool IsNativeSignature =
        LLVMType<ArgumentValueType>::CXXUTypeIsNativeType &&
            ArgumentIterator<SignatureT, I-1>::IsNativeSignature;

    template <typename OpT>
    static void apply(const OpT& op, const bool forwards) {
        if (forwards) {
            ArgumentIterator<SignatureT, I-1>::apply(op, forwards);
            op(ArgumentValueType());
        }
        else {
            op(ArgumentValueType());
            ArgumentIterator<SignatureT, I-1>::apply(op, forwards);
        }
    }
};

template <typename SignatureT>
struct ArgumentIterator<SignatureT, 0>
{
    static const bool IsNativeSignature = true;
    template <typename OpT>
    static void apply(const OpT&, const bool) {}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief  Metadata associated with a function argument or return value.
struct ArgInfo
{
    explicit ArgInfo(const Value& val)
        : ArgInfo(val.GetUnderlyingType(), val.IsPtr() ? 1 : 0) {}
    explicit ArgInfo(llvm::Type* utype) : ArgInfo(utype, 0) {}
    ArgInfo(llvm::Type* utype, uint8_t ptrs, bool ret = false)
        : mUType(utype), mPtrs(ptrs), mReturn(ret) {
        OPENVDB_ASSERT(mUType);
        OPENVDB_ASSERT(!mUType->isPointerTy());
        // void*'s need to be provided as int8_t's
        OPENVDB_ASSERT(!(mUType->isVoidTy() && mPtrs > 0));
    }
    ArgInfo(ArgInfo&&) = default;
    ArgInfo(const ArgInfo&) = default;
    ArgInfo& operator=(ArgInfo&&) = default;
    ArgInfo& operator=(const ArgInfo&) = default;
    bool operator==(const ArgInfo& other) const
    {
        return
            mUType == other.mUType &&
            mPtrs == other.mPtrs &&
            mReturn == other.mReturn;
    }
    bool operator!=(const ArgInfo& other) const { return !this->operator==(other); }
    bool IsMatchingType(const ArgInfo& other) const
    {
        return
            mUType == other.mUType &&
            mPtrs == other.mPtrs;
    }
    bool IsPtr() const { return mPtrs > 0; }
    uint8_t NumPtrs() const { return mPtrs; }
    bool IsNative() const { return Value::Supports(mUType) && mPtrs <= 1; }
    bool IsVoid() const { return mUType->isVoidTy(); }
    bool IsReturn() const { return mReturn; }
    llvm::Type* GetUnderlyingType() const { return mUType; }
    llvm::Type* GetType() const
    {
        llvm::Type* type = mUType;
        for (uint8_t i = 0; i < mPtrs; ++i) {
            type = llvm::PointerType::get(type, 0);
        }
        return type;
    }
    void SetIsReturn() { mReturn = true; }
private:
    llvm::Type* mUType; // the underlying argument type
    uint8_t mPtrs; // num ptrs to the type
    // true if this is the return argument. For Sret functions, both the
    // void return and first argument are marked as true.
    bool mReturn;
};

/// @brief  Container of ArgInfos. This class makes up part of the Function
///   API for querying signature information.
struct ArgInfoVector
{
public:
    using ContainerT = SmallArgumentVector<ArgInfo>;

    ArgInfoVector() = default;
    ArgInfoVector(const std::initializer_list<ArgInfo>& info)
        : mInfoVec(info) {}
    ArgInfoVector(ArgInfoVector&&) = default;
    ArgInfoVector(const ArgInfoVector&) = default;
    ArgInfoVector& operator=(ArgInfoVector&&) = default;
    ArgInfoVector& operator=(const ArgInfoVector&) = default;

    bool operator==(const ArgInfoVector& other) const { return mInfoVec == other.mInfoVec; }
    bool operator!=(const ArgInfoVector& other) const { return !this->operator==(other); }

    auto  begin() { return mInfoVec.begin(); }
    auto  end() { return mInfoVec.end(); }
    auto  begin() const { return mInfoVec.begin(); }
    auto  end() const { return mInfoVec.end(); }
    auto  rbegin() { return mInfoVec.rbegin(); }
    auto  rend() { return mInfoVec.rend(); }
    auto  rbegin() const { return mInfoVec.rbegin(); }
    auto  rend() const { return mInfoVec.rend(); }
    auto& front() { return mInfoVec.front(); }
    auto& front() const { return mInfoVec.front(); }
    auto& back() { return mInfoVec.back(); }
    auto& back() const { return mInfoVec.back(); }
    auto  pop_back() { return mInfoVec.pop_back(); }
    auto  clear() { return mInfoVec.clear(); }
    auto  size() const { return mInfoVec.size(); }
    auto  empty() const { return mInfoVec.empty(); }
    auto  erase(ContainerT::const_iterator iter) { return mInfoVec.erase(iter); }

    void reserve(size_t i) { mInfoVec.reserve(i); }
    template <typename ...Args>
    void emplace_back(Args&& ...args) { mInfoVec.emplace_back(std::move(args)...); }
    ArgInfo& operator[](size_t pos)
    {
        OPENVDB_ASSERT(pos < mInfoVec.size());
        return mInfoVec[pos];
    }
    const ArgInfo& operator[](size_t pos) const
    {
        OPENVDB_ASSERT(pos < mInfoVec.size());
        return mInfoVec[pos];
    }

    SmallArgumentVector<llvm::Type*> AsLLVMTypes() const
    {
        SmallArgumentVector<llvm::Type*> types;
        types.reserve(mInfoVec.size());
        for (auto& info : mInfoVec) {
            types.emplace_back(info.GetType());
        }
        return types;
    }

private:
    ContainerT mInfoVec;
};

/// @brief  Wrapper struct to represent "native" function arguments; that is,
///   the set of Value type that the AX grammar supports. NativeArguments
///   have two benefits; they support casting and implicit function matching
///   through FunctionGroups and can be used directly in IR generators (to
///   leverage the AX Value API). Functions can still be generated and called
///   with "non-native" arguments, but in these cases FunctionGroup::execute
///   must result in an explicit signature match
struct NativeArguments
{
    NativeArguments() = default;
    NativeArguments(const std::initializer_list<Value>& args)
        : mArgs(args) {}
    explicit NativeArguments(const std::vector<Value>& args)
        : mArgs(args.begin(), args.end()) {}
    NativeArguments(NativeArguments&&) = default;
    NativeArguments(const NativeArguments&) = default;
    NativeArguments& operator=(NativeArguments&&) = default;
    NativeArguments& operator=(const NativeArguments&) = default;
    size_t size() const { return mArgs.size(); }
    Value& operator[](size_t pos)
    {
        OPENVDB_ASSERT(pos < mArgs.size());
        return mArgs[pos];
    }
    const Value& operator[](size_t pos) const
    {
        OPENVDB_ASSERT(pos < mArgs.size());
        return mArgs[pos];
    }
    void AddArg(const Value& val) { mArgs.emplace_back(val); }
    /// @brief  Cast these arguments to match the given function's signature
    OPENVDB_AX_API NativeArguments Cast(const Function& F, llvm::IRBuilder<>& B) const;
private:
    SmallArgumentVector<Value> mArgs;
};

/// @brief  Arbitrary, potentially "non-native" arguments. This wrapper struct
///   can be used when generating function which cannot be called from AX
///   itself (e.g. VDB accessor functions or Volume/Point kernels etc). They
///   do not support implicit function signature matching or casting.
struct Arguments
{
    Arguments() = default;

    /// @brief  Create a set of arguments from native arguments. The function
    ///   framework typically works on generic arguments types.
    explicit Arguments(const NativeArguments& args) {
        mArgs.reserve(args.size());
        mTypes.reserve(args.size());
        for (size_t i = 0; i < args.size(); ++i) {
            this->AddArg(args[i]);
        }
    }

    Arguments(Arguments&&) = default;
    Arguments(const Arguments&) = default;
    Arguments& operator=(Arguments&&) = default;
    Arguments& operator=(const Arguments&) = default;

    size_t size() const { return mArgs.size(); }

    bool AreNativeValues() const
    {
        for (const auto& types : mTypes) {
            if (!types.IsNative()) return false;
        }
        return true;
    }

    Value AsNativeValue(const size_t i) const
    {
        OPENVDB_ASSERT(mTypes[i].IsNative());
        return Value(mArgs[i], mTypes[i].GetUnderlyingType());
    }

    NativeArguments AsNativeValues() const
    {
        OPENVDB_ASSERT(this->AreNativeValues());
        NativeArguments args;
        for (size_t i = 0; i < mArgs.size(); ++i) {
            args.AddArg(this->AsNativeValue(i));
        }
        return args;
    }

    const llvm::ArrayRef<llvm::Value*> AsLLVMValues() const { return mArgs; }
    const ArgInfo& GetArgInfo(size_t pos) const { return mTypes[pos]; }
    const ArgInfoVector& GetArgInfo() const { return mTypes; }

    llvm::Value* operator[](size_t pos) const
    {
        OPENVDB_ASSERT(pos < mTypes.size());
        return mArgs[pos];
    }

    void AddArg(llvm::Value* val, const ArgInfo& type)
    {
#if LLVM_VERSION_MAJOR <= 15
        [[maybe_unused]] llvm::Type* base = val->getType();
        while (base->isPointerTy()) base = base->getContainedType(0);
        OPENVDB_ASSERT_MESSAGE((base == type.GetUnderlyingType()),
            "Base type of val does not match stored underlying type");
#endif
        mArgs.emplace_back(val);
        mTypes.emplace_back(type);
    }

    void AddArg(const Value& val)
    {
        mArgs.emplace_back(val.GetValue());
        mTypes.emplace_back(val);
    }

    void PrependArg(const Value& val)
    {
        this->AddArg(val);
        std::rotate(mArgs.rbegin(), mArgs.rbegin() + 1, mArgs.rend());
        std::rotate(mTypes.rbegin(), mTypes.rbegin() + 1, mTypes.rend());
    }

private:
    SmallArgumentVector<llvm::Value*> mArgs;
    ArgInfoVector mTypes;
};

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/// @brief  Populate a vector of llvm types from a function signature
///    declaration.
/// @warning  From LLVM 16 onwards, pointer argument type cannot be
///    introspected
///
/// @param C     The llvm context
/// @param types A vector of types to populate
///
template <typename SignatureT>
inline llvm::Type*
llvmTypesFromSignature(llvm::LLVMContext& C,
                std::vector<llvm::Type*>* types = nullptr)
{
    using Traits = FunctionTraits<SignatureT>;
    using ArgumentIteratorT =
        ArgumentIterator<SignatureT, Traits::N_ARGS>;

    if (types) {
        types->reserve(Traits::N_ARGS);
        auto callback = [&types, &C](auto type) {
            using Type = decltype(type);
            types->emplace_back(LLVMType<Type>::get(C));
        };
        ArgumentIteratorT::apply(callback, /*forwards*/true);
    }
    using Type = typename Traits::ReturnType;
    return LLVMType<Type>::get(C);
}

/// @brief  Populate a vector of ArgInfos from a function signature
///    declaration.
///
/// @param C     The llvm context
/// @param types A ArgInfoVector to populate
///
template <typename SignatureT>
inline ArgInfo
llvmArgTypesFromSignature(llvm::LLVMContext& C,
    ArgInfoVector* types = nullptr)
{
    using Traits = FunctionTraits<SignatureT>;
    using ArgumentIteratorT =
        ArgumentIterator<SignatureT, Traits::N_ARGS>;

    if (types)
    {
        types->reserve(Traits::N_ARGS);
        auto callback = [&types, &C](auto type)
        {
            using UnderlyingType = std::remove_cv_t<typename RemoveAllPtrTypes<decltype(type)>::Type>;
            static constexpr auto NPtrs = CountNPtrs<decltype(type)>::value;
            static constexpr bool IsVoid = std::is_same_v<UnderlyingType, void>;
            // args can't be void without pts
            static_assert(!IsVoid || NPtrs > 0);
            // void* arguments alias to int8_t
            using Type = std::conditional_t<IsVoid, int8_t, UnderlyingType>;
            types->emplace_back(LLVMType<Type>::get(C), NPtrs);
        };
        ArgumentIteratorT::apply(callback, /*forwards*/true);
    }

    using UnderlyingType = std::remove_cv_t<typename RemoveAllPtrTypes<typename Traits::ReturnType>::Type>;
    static constexpr auto NPtrs = CountNPtrs<typename Traits::ReturnType>::value;

    if constexpr (std::is_same_v<UnderlyingType, void> && NPtrs > 0)
    {
        // if underlying type is void, alias to int8_t if its a void*
        ArgInfo ret{LLVMType<int8_t>::get(C), NPtrs};
        ret.SetIsReturn();
        return ret;
    }
    else {
        ArgInfo ret{LLVMType<UnderlyingType>::get(C), NPtrs};
        ret.SetIsReturn();
        return ret;
    }
}


/// @brief  Generate an LLVM FunctionType from a function signature
///
/// @param C  The llvm context
///
template <typename SignatureT>
inline llvm::FunctionType*
llvmFunctionTypeFromSignature(llvm::LLVMContext& C)
{
    ArgInfoVector types;
    ArgInfo returnType =
        llvmArgTypesFromSignature<SignatureT>(C, &types);
    return llvm::FunctionType::get(returnType.GetType(),
            types.AsLLVMTypes(), /*isVarArg=*/false);
}

/// @brief  Print a function signature to the provided ostream.
///
/// @param  os    The stream to print to
/// @param  types The function argument types
/// @param  returnType  The return type of the function. Must not be a nullptr
/// @param  name  The name of the function. If not provided, the return type
///               neighbours the first parenthesis
/// @param  names Names of the function parameters. If a name is nullptr, it
///               skipped
/// @param  axTypes Whether to try and convert the llvm::Types provided to
///                 AX types. If false, the llvm types are used.
OPENVDB_AX_API void
printSignature(std::ostream& os,
               const ArgInfoVector& types,
               const ArgInfo& returnType,
               const char* name = nullptr,
               const llvm::ArrayRef<const char*>& names = {},
               const bool axTypes = false);


#if LLVM_VERSION_MAJOR <= 15
OPENVDB_DEPRECATED_MESSAGE("Switch to AX's internal ArgInfo types for LLVM 16 onwards")
OPENVDB_AX_API void
printSignature(std::ostream& os,
               const std::vector<llvm::Type*>& types,
               const llvm::Type* returnType,
               const char* name = nullptr,
               const std::vector<const char*>& names = {},
               const bool axTypes = false);
#endif

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Forward declare builder for private access to Function types
struct FunctionBuilder;

/// @brief  The base/abstract representation of an AX function. Derived classes
///         must implement the Function::types call to describe their signature.
struct OPENVDB_AX_API Function
{
    using Ptr = std::shared_ptr<Function>;

    Function(const size_t size, const std::string& symbol)
        : mSize(size)
        , mSymbol(symbol)
        , mAttributes(nullptr)
        , mNames()
        , mDeps() {
            // symbol must be a valid string
            OPENVDB_ASSERT(!symbol.empty());
        }

    virtual ~Function() = default;

    /// @brief  Populate a vector of ArgInfos which describe this function
    ///         signature. This method is used by Function::create,
    ///         Function::print and Function::match.
    /// @note   The variant that takes a vector of ArgInfos is optional with
    ///         LLVM 15 for compatibility but must be implemented with newer
    ///         versions.
#if LLVM_VERSION_MAJOR <= 15
    virtual ArgInfo types(ArgInfoVector&, llvm::LLVMContext&) const
    {
        OPENVDB_THROW(AXCodeGenError,
            std::string("New AX API for function arguments has been called but has not "
            "been implemented by function: ") + this->symbol());
    }
    OPENVDB_DEPRECATED_MESSAGE("Switch to AX's internal ArgInfo types for LLVM 16 onwards")
    virtual llvm::Type* types(std::vector<llvm::Type*>&, llvm::LLVMContext&) const = 0;
#else
    virtual ArgInfo types(ArgInfoVector&, llvm::LLVMContext&) const = 0;
#endif

    /// @brief   Converts and creates this AX function into a llvm Function.
    /// @details This method uses the result from Function::types() to construct
    ///          a llvm::FunctionType and a subsequent a llvm::Function. Any
    ///          parameter, return or function attributes are also added to the
    ///          function. If a module is provided, the module if first checked
    ///          to see if the function already exists. If it does, it is
    ///          immediately returned. If the function doesn't exist in the
    ///          module, its prototype is created and also inserted into the end
    ///          of the modules function list. If no module is provided, the
    ///          function is left detached and must be added to a valid Module
    ///          to be callable.
    /// @warning If a module is not provided, the caller takes ownership of the
    ///          returned function and is responsible for deallocating it.
    /// @note    The body of the function is left to derived classes to
    ///          implement. As you need a Module to generate the prototype/body,
    ///          this function serves two purposes. The first is to return the
    ///          detached function signature if only a context is provided.
    ///          The second is to ensure the function prototype and body (if
    ///          required) is inserted into the module prior to returning.
    /// @note    It is possible to end up with function symbol collisions if you
    ///          do not have unique function symbols in your module
    ///
    /// @param C  The LLVM Context
    /// @param M  The Module to write the function to
    virtual llvm::Function*
    create(llvm::LLVMContext& C, llvm::Module* M = nullptr) const;

    /// @brief  Convenience method which always uses the provided module to find
    ///         the function or insert it if necessary.
    /// @param M  The llvm::Module to use
    llvm::Function* create(llvm::Module& M) const {
        return this->create(M.getContext(), &M);
    }

    /// @brief  Convenience method for calling M.getFunction(symbol). Returns a
    ///         nullptr if the function has not yet been created or if it is
    ///         embedded IR.
    /// @param M  The llvm::Module to use
    llvm::Function* get(const llvm::Module& M) const;

    /// @brief  Uses the IRBuilder to create a call to this function with the
    ///         given arguments, creating the function and inserting it into the
    ///         IRBuilder's Module if necessary (through Function::create).
    ///         Returns the result of the function call which can be a nullptr
    ///         if the function is a non-sret void call.
    /// @note   The IRBuilder must have a valid llvm Module/Function/Block
    ///         attached
    /// @note   If the number of provided arguments do not match the size of the
    ///         current function, invalid IR will be generated.
    /// @note   If the provided argument types do not match the current function
    ///         and cast is false, invalid IR will be generated. Additionally,
    ///         invalid IR will be generated if cast is true but no valid cast
    ///         exists for a given argument.
    /// @note   When casting arguments, the readonly flags of the function are
    ///         not checked (unlike Function::match). Casting an argument will
    ///         cause a new copy of the argument to be created and passed to the
    ///         function. These new values do not propagate back any changes to
    ///         the original argument. Separate functions for all writable
    ///         argument types must be created.
    ///
    /// @param args    The llvm Value arguments to call this function with
    /// @param B       The llvm IRBuilder
    /// @param cast    Whether to allow implicit casting of arguments
#if LLVM_VERSION_MAJOR <= 15
    OPENVDB_DEPRECATED_MESSAGE("Function::call which takes llvm::Value's and "
        "supports casting is incompatible with LLVM 16+ and will be removed.")
    virtual llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast) const;
#endif
    /// From LLVM 16 onwards, this version of call does not support argument
    /// casting. This must be performed using the NativeArguments struct
    virtual llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) const;

    virtual Value call(const Arguments& args, llvm::IRBuilder<>& B) const;

    Value call(const NativeArguments& args, llvm::IRBuilder<>& B) const
    {
        return this->call(Arguments(args.Cast(*this, B)), B);
    }

    /// @brief  The result type from calls to Function::match
    /// @note   Function::match cannot return Ambiguous - this is only returned
    ///   by the FunctionGroup API.
    enum SignatureMatch
    {
        None = 0,  // Mismatching argument sizes
        Size,      // Correct number of arguments but incompatible types
        Implicit,  // Correct number of arguments and castable types
        Ambiguous, // Correct number of arguments and castable types but multiple available signatures
        Explicit   // Correct number of arguments and types match exactly
    };

    /// @brief  The base implementation for determining how a vector of llvm
    ///         arguments translates to this functions signature. Returns an
    ///         enum which represents the available mapping.
    /// @details This method calls types() to figure out the function signature,
    ///          then compares each argument type to the type in the input
    ///          vector. If the types match exactly, an Explicit match is found.
    ///          If the sizes of the inputs and signature differ, no match is
    ///          found and None is returned. If however, the sizes match and
    ///          there exists a valid implicit cast from the input type to the
    ///          signature type for every input, an Implicit match is returned.
    ///          Finally, if the sizes match but there is no implicit cast
    ///          mapping, Size is returned.
    ///            i8 -> i32        : Implicit
    ///            i32 -> i32       : Explicit
    ///            str -> i32       : Size
    ///            (i32,i32) -> i32 : None
    ///          Never returns Ambiguous (this state is used by FunctionGroup)
    /// @note  Due to the way CFunctionSRet is implemented, the LLVM Context
    ///        must be provided in case we have a zero arg function signature
    ///        with a SRET.
    /// @param inputs  The input types
    /// @param C       The LLVM Context
    virtual SignatureMatch match(const ArgInfoVector& inputs, llvm::LLVMContext& C) const;
#if LLVM_VERSION_MAJOR <= 15
    OPENVDB_DEPRECATED_MESSAGE("Switch to AX's internal ArgInfo types for LLVM 16 onwards")
    virtual SignatureMatch match(const std::vector<llvm::Type*>& inputs, llvm::LLVMContext& C) const;
#endif

    /// @brief  The number of arguments that this function has
    inline size_t size() const { return mSize; }

    /// @brief  The function symbol name.
    /// @details  This will be used as its identifier in IR and must be unique.
    inline const char* symbol() const { return mSymbol.c_str(); }

    /// @brief  Returns the descriptive name of the given argument index
    /// @details  If the index is greater than the number of arguments, an empty
    ///           string is returned.
    ///
    /// @param idx  The index of the argument
    inline const char* argName(const size_t idx) const
    {
        return idx < mNames.size() ? mNames[idx] : "";
    }

    /// @brief  Print this function's signature to the provided ostream.
    /// @details  This is intended to return a descriptive front end user string
    ///           rather than the function's IR representation. This function is
    ///           virtual so that derived classes can customize how they present
    ///           frontend information.
    /// @sa  printSignature
    ///
    /// @param C     The llvm context
    /// @param os    The ostream to print to
    /// @param name  The name to insert into the description.
    /// @param axTypes  Whether to print llvm IR or AX Types.
    virtual void print(llvm::LLVMContext& C,
            std::ostream& os,
            const char* name = nullptr,
            const bool axTypes = true) const;

    const SmallArgumentVector<const char*>& dependencies() const { return mDeps; }

    /// Deprecated builder methods, no longer public

    OPENVDB_DEPRECATED_MESSAGE("This method incorrectly returns the attributes "
        "of the function set by the FunctionBuilder, not by the codegen. To "
        "inspect function attributes, retrieve the created function from the "
        "llvm::Module.")
    inline bool hasParamAttribute(const size_t i,
            const llvm::Attribute::AttrKind& kind) const
    {
        if (!mAttributes) return false;
        const auto iter = mAttributes->mParamAttrs.find(i);
        if (iter == mAttributes->mParamAttrs.end()) return false;
        const auto& vec = iter->second;
        return std::find(vec.begin(), vec.end(), kind) != vec.end();
    }

    OPENVDB_DEPRECATED_MESSAGE("Use the FunctionBuilder to construct Functions")
    inline void setArgumentNames(std::vector<const char*> names)
    {
        mNames.assign(names.begin(), names.end());
    }

    OPENVDB_DEPRECATED_MESSAGE("Use the FunctionBuilder to construct Functions")
    inline void setDependencies(std::vector<const char*> deps)
    {
        mDeps.assign(deps.begin(), deps.end());
    }

    OPENVDB_DEPRECATED_MESSAGE("Use the FunctionBuilder to construct Functions")
    inline void setFnAttributes(const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mFnAttrs.assign(in.begin(), in.end());
    }

    OPENVDB_DEPRECATED_MESSAGE("Use the FunctionBuilder to construct Functions")
    inline void setRetAttributes(const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mRetAttrs.assign(in.begin(), in.end());
    }

    OPENVDB_DEPRECATED_MESSAGE("Use the FunctionBuilder to construct Functions")
    inline void setParamAttributes(const size_t i,
            const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mParamAttrs[i].assign(in.begin(), in.end());
    }

protected:

#if LLVM_VERSION_MAJOR <= 15
    /// @brief  Cast the provided arguments to the given type as supported by
    ///         implicit casting of function types. If the types already match
    ///         OR if a cast cannot be performed, nothing is done to the argument.
    /// @todo   This should really be generalized out for Function::call and
    ///         Function::match to both use. However, due to SRET functions,
    ///         this logic must be performed somewhere in the Function class
    ///         hierarchy and not in FunctionGroup
    static void cast(std::vector<llvm::Value*>& args,
                const std::vector<llvm::Type*>& types,
                llvm::IRBuilder<>& B);
#endif

private:
    friend FunctionBuilder;

    enum class Type {
        kNormal,
        kBuiltin,
        kKernel
    };

    struct Attributes
    {
        SmallArgumentVector<llvm::Attribute::AttrKind> mFnAttrs, mRetAttrs;
        std::map<size_t, SmallArgumentVector<llvm::Attribute::AttrKind>> mParamAttrs;
        bool mReadOnly {false};
        Type mType {Type::kNormal};
    };

    inline Attributes& attrs()
    {
        if (!mAttributes) mAttributes.reset(new Attributes());
        return *mAttributes;
    }

    /// @brief  Temporary method until we move to ArgInfo introspection
    bool IsParamReadOnly(const size_t idx) const
    {
        if (!mAttributes) return false; // can't be certain if no attrs set
        if (mAttributes->mReadOnly) return true;
        // @todo  REMOVE - switch to using writable info on the ArgInfo types
        const auto iter = mAttributes->mParamAttrs.find(idx);
        if (iter == mAttributes->mParamAttrs.end()) return false;
        const auto& vec = iter->second;
        return std::find(vec.begin(), vec.end(),
            llvm::Attribute::AttrKind::ReadOnly) != vec.end();
    }

    llvm::AttributeList flattenAttrs(llvm::Function* F) const;

private:
    const size_t mSize;
    const std::string mSymbol;
    std::unique_ptr<Attributes> mAttributes;
    SmallArgumentVector<const char*> mNames;
    SmallArgumentVector<const char*> mDeps;
};

/// @brief  Templated interface class for SRET functions. This struct provides
///         the interface for functions that wish to return arrays (vectors or
///         matrices) by internally remapping the first argument for the user.
///         As far as LLVM and any bindings are concerned, the function
///         signature remains unchanged - however the first argument becomes
///         "invisible" to the user and is instead allocated by LLVM before the
///         function is executed. Importantly, the argument has no impact on
///         the user facing AX signature and doesn't affect declaration selection.
/// @note   This class is not intended to be instantiated directly, but instead
///         used by derived implementation which hold a valid implementations
///         of member functions required to create a llvm::Function (such as
///         Function::types and Function::call). This exists as an interface to
///         avoid virtual inheritance.
///
template <typename SignatureT, typename DerivedFunction>
struct SRetFunction : public DerivedFunction
{
    using Ptr = std::shared_ptr<SRetFunction<SignatureT, DerivedFunction>>;
    using Traits = FunctionTraits<SignatureT>;

    // check there actually are arguments
    static_assert(Traits::N_ARGS > 0,
        "SRET Function object has been setup with the first argument as the return "
        "value, however the provided signature is empty.");

    // check no return value exists
    static_assert(std::is_same<typename Traits::ReturnType, void>::value,
        "SRET Function object has been setup with the first argument as the return "
        "value and a non void return type.");

private:

    using FirstArgument = typename Traits::template Arg<0>::Type;
    static_assert(std::is_pointer<FirstArgument>::value,
        "SRET Function object has been setup with the first argument as the return "
        "value, but this argument it is not a pointer type.");
    static_assert(!std::is_const_v<FirstArgument>,
        "SRET Function object has been setup with the first argument as the return "
        "value, but this argument is const.");
    using SRetType = typename std::remove_pointer<FirstArgument>::type;

public:

    /// @brief Overide the ArgInfo type method. This does NOT change the arg
    ///   order, it simply marks the first argument as a return argument. Note
    ///   that the void ret type is also left as a return type.
    ArgInfo types(ArgInfoVector& args, llvm::LLVMContext& C) const override
    {
        ArgInfo ret = DerivedFunction::types(args, C);
        OPENVDB_ASSERT(!args.empty());
        OPENVDB_ASSERT(ret.IsVoid());
        OPENVDB_ASSERT(!args[0].IsVoid());
        OPENVDB_ASSERT(args[0].IsPtr());
        args[0].SetIsReturn();
        return ret;
    }

    /// @brief  Override of match which inserts the SRET type such that the base
    ///         class methods ignore it.
    Function::SignatureMatch match(
        const ArgInfoVector& args,
        llvm::LLVMContext& C) const override
    {
        ArgInfoVector inputs(args);
        // Create a dummy sret ptr type for derived match impls
        llvm::Type* stype = LLVMType<SRetType>::get(C);
        // llvm::Constant* zero = llvmConstant(0, llvm::Type::getInt64Ty(C));
        // llvm::Constant* dummy = llvm::ConstantExpr::getPointerCast(zero, stype->getPointerTo());
        inputs.emplace_back(stype, 1);
        inputs.back().SetIsReturn();
        std::rotate(inputs.rbegin(), inputs.rbegin() + 1, inputs.rend());
        return DerivedFunction::match(inputs, C);
    }

    /// @brief  Override of call which allocates the required SRET llvm::Value
    ///         for this function.
    /// @note   Unlike other function where the returned llvm::Value* is a
    ///         llvm::CallInst (which also represents the return value),
    ///         SRET functions return the allocated 1st argument i.e. not a
    ///         llvm::CallInst
    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) const override
    {
        // append return value and right rotate
        std::vector<llvm::Value*> inputs(args);
        llvm::Type* sret = LLVMType<SRetType>::get(B.getContext());
        inputs.emplace_back(insertStaticAlloca(B, sret));
        std::rotate(inputs.rbegin(), inputs.rbegin() + 1, inputs.rend());
        DerivedFunction::call(inputs, B);
        return inputs.front();
    }

    Value call(const Arguments& args, llvm::IRBuilder<>& B) const override
    {
        // append return value and right rotate
        Arguments inputs(args);
        Value sret = Value::Alloc(B,  LLVMType<SRetType>::get(B.getContext()));
        inputs.PrependArg(sret);
        DerivedFunction::call(inputs, B);
        return sret;
    }

    /// @brief  Override of print to avoid printing out the SRET type
    void print(llvm::LLVMContext& C,
           std::ostream& os,
           const char* name = nullptr,
           const bool axTypes = true) const override
    {
        ArgInfoVector current;
        ArgInfo ret = this->types(current, C);
        // left rotate
        std::rotate(current.begin(), current.begin() + 1, current.end());
        ret = current.back();
        current.pop_back();

        SmallArgumentVector<const char*> names;
        names.reserve(this->size());
        for (size_t i = 0; i < this->size()-1; ++i) {
            names.emplace_back(this->argName(i));
        }
        printSignature(os, current, ret, name, names, axTypes);
    }

#if LLVM_VERSION_MAJOR <= 15
    /// @note  Bring in deprecated type methods
    using Function::types;

    /// @note  This is deprecated! Omitting the warning as it invokes the
    ///   parent function which is also deprecated. Use NativeArguments to
    ///   perform argument casting
    Function::SignatureMatch match(const std::vector<llvm::Type*>& args,
            llvm::LLVMContext& C) const override
    {
        // append return type and right rotate
        std::vector<llvm::Type*> inputs(args);
        inputs.emplace_back(LLVMType<SRetType*>::get(C));
        std::rotate(inputs.rbegin(), inputs.rbegin() + 1, inputs.rend());
        return DerivedFunction::match(inputs, C);
    }

    /// @note  This is deprecated! Omitting the warning as it invokes the
    ///   parent function which is also deprecated. Use NativeArguments to
    ///   perform argument casting
    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast) const override
    {
        // append return value and right rotate
        std::vector<llvm::Value*> inputs(args);
        llvm::Type* sret = LLVMType<SRetType>::get(B.getContext());
        inputs.emplace_back(insertStaticAlloca(B, sret));
        std::rotate(inputs.rbegin(), inputs.rbegin() + 1, inputs.rend());
        DerivedFunction::call(inputs, B, cast);
        return inputs.front();
    }
#endif

protected:
    /// @brief  Forward all arguments to the derived class
    template <typename ...Args>
    SRetFunction(Args&&... ts) : DerivedFunction(ts...) {}
};

/// @brief  The base class for all C bindings.
struct CFunctionBase : public Function
{
    using Ptr = std::shared_ptr<CFunctionBase>;

    ~CFunctionBase() override = default;

    /// @brief  Returns the global address of this function.
    /// @note   This is only required for C bindings.
    virtual uint64_t address() const = 0;

    inline void setConstantFold(bool on) { mConstantFold = on; }
    inline bool hasConstantFold() const { return mConstantFold; }

#if LLVM_VERSION_MAJOR <= 15
    OPENVDB_DEPRECATED
    inline virtual llvm::Value* fold(
        const std::vector<llvm::Value*>&,
        llvm::LLVMContext&) const
    {
        return nullptr;
    }
#else
    inline virtual llvm::Value* fold(
        const llvm::ArrayRef<llvm::Value*>&,
        llvm::LLVMContext&) const
    {
        return nullptr;
    }
#endif

protected:
    CFunctionBase(const size_t size,
        const std::string& symbol)
        : Function(size, symbol)
        , mConstantFold(false) {}

private:
    bool mConstantFold;
};

/// @brief  Represents a concrete C function binding.
///
/// @note This struct is templated on the signature to allow for evaluation of
///       the arguments to llvm types from any llvm context.
///
template <typename SignatureT>
struct CFunction : public CFunctionBase
{
    using CFunctionT = CFunction<SignatureT>;
    using Ptr = std::shared_ptr<CFunctionT>;
    using Traits = FunctionTraits<SignatureT>;

    // Assert that the return argument is not a pointer (relaxed for void* for mallocs).
    // Note that this is relaxed for IR functions where it's allowed if the function is
    // forcefully inlined.
    static_assert(std::is_same<typename Traits::ReturnType, void*>::value ||
        !std::is_pointer<typename Traits::ReturnType>::value,
        "CFunction object has been setup with a pointer return argument. C bindings "
        "cannot return memory locations to LLVM - Consider using a CFunctionSRet.");

    CFunction(const std::string& symbol, SignatureT* function)
        : CFunctionBase(Traits::N_ARGS, symbol)
        , mFunction(function) {}

    ~CFunction() override = default;

    inline ArgInfo types(ArgInfoVector& types, llvm::LLVMContext& C) const override
    {
        return llvmArgTypesFromSignature<SignatureT>(C, &types);
    }

    inline uint64_t address() const override final
    {
        return reinterpret_cast<uint64_t>(mFunction);
    }

    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) const override
    {
        llvm::Value* result = this->fold(args, B.getContext());
        if (result) return result;
        return Function::call(args, B);
    }

    Value call(const Arguments& args, llvm::IRBuilder<>& B) const override
    {
        llvm::Constant* result = this->fold(args.AsLLVMValues(), B.getContext());
        if (result) return Value(result);
        return Function::call(args, B);
    }

#if LLVM_VERSION_MAJOR <= 15
    llvm::Constant* fold(const std::vector<llvm::Value*>& args, llvm::LLVMContext& C) const override final
#else
    llvm::Constant* fold(const llvm::ArrayRef<llvm::Value*>& args, llvm::LLVMContext& C) const override final
#endif
    {
        if (!this->hasConstantFold()) return nullptr;
        SmallArgumentVector<llvm::Constant*> constants;
        for (auto& value : args) {
            if (!llvm::isa<llvm::Constant>(value)) return nullptr;
            constants.emplace_back(llvm::cast<llvm::Constant>(value));
        }
        // no guarantee that fold() will be able to cast all arguments
        return ConstantFolder<SignatureT>::fold(constants, *mFunction, C);
    }

#if LLVM_VERSION_MAJOR <= 15
    inline llvm::Type* types(std::vector<llvm::Type*>& types, llvm::LLVMContext& C) const override
    {
        return llvmTypesFromSignature<SignatureT>(C, &types);
    }

    /// @note  This is deprecated! Omitting the warning as it invokes the
    ///   parent function which is also deprecated. Use NativeArguments to
    ///   perform argument casting
    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast) const override
    {
        llvm::Value* result = this->fold(args, B.getContext());
        if (result) return result;
        return Function::call(args, B, cast);
    }
#endif

private:
    SignatureT* mFunction;
};

/// @brief  The base/abstract definition for an IR function.
struct OPENVDB_AX_API IRFunctionBase : public Function
{
    using Ptr = std::shared_ptr<IRFunctionBase>;

    /// @brief  The IR callback function which will write the LLVM IR for this
    ///         function's body.
    /// @details  The first argument is the vector of functional arguments. i.e.
    ///           a representation of the value that the callback has been invoked
    ///           with.
    ///           The last argument is the IR builder which should be used to
    ///           generate the function body IR.
    /// @note     You can return a ret void instruction, an actual value or
    ///           Value::Invalid() which will cause the function framework to
    ///           insert a ret void if necessary.
    using GeneratorNativeCb = std::function<Value
        (const NativeArguments&, llvm::IRBuilder<>&)>;

    using GeneratorArgumentsCb = std::function<Value
        (const Arguments&, llvm::IRBuilder<>&)>;

    /// @brief  Legacy callback, will eventually be deprecated in favour of
    ///   using the GeneratorArgumentsCb
    using GeneratorCb = std::function<llvm::Value*
        (const std::vector<llvm::Value*>&,
         llvm::IRBuilder<>&)>;

    /// @brief  Enable or disable the embedding of IR. Embedded IR is currently
    ///         required for function which use parent function parameters.
    inline void setEmbedIR(bool on) { mEmbedIR = on; }
    inline bool hasEmbedIR() const { return mEmbedIR; }

    /// @brief  Override for the creation of an IR function. This ensures that
    ///         the body and prototype of the function are generated if a Module
    ///         is provided.
    /// @note   A nullptr is returned if mEmbedIR is true and no action is
    ///         performed.
    /// @note   Throws if this function has been initialized with a nullptr
    ///         generator callback. In this case, the function prototype will
    ///         be created, but not the function body.
    /// @note   Throws if the return type of the generator callback does not
    ///         match the function prototype. In this case, both the prototype
    ///         and the function body will be created and inserted, but the IR
    ///         will be invalid.
    llvm::Function*
    create(llvm::LLVMContext& C, llvm::Module* M) const override;

    /// @brief  Override for call, which is only necessary if mEmbedIR is true,
    ///         as the IR generation for embedded functions is delayed until
    ///         the function is called. If mEmbedIR is false, this simply calls
    ///         Function::call
    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B) const override;

    Value call(const Arguments& args, llvm::IRBuilder<>&) const override;

#if LLVM_VERSION_MAJOR <= 15
    // @note  This is deprecated! Omitting the warning as it invokes the
    //   parent function which is also deprecated
    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast) const override;
#endif

protected:

    // @todo This should ideally live in FunctionGroup::execute, but the return
    //       type is allowed to differ for sret C bindings.
    inline void
    verifyResultType(const llvm::Type* result, const llvm::Type* expected) const
    {
        if (result == expected) return;
        std::string source, target;
        if (result) llvmTypeToString(result, source);
        llvmTypeToString(expected, target);
        OPENVDB_THROW(AXCodeGenError, "Function \"" + std::string(this->symbol()) +
            "\" has been invoked with a mismatching return type. Expected: \"" +
            target + "\", got \"" + source + "\".");
    }
    IRFunctionBase(const std::string& symbol,
        const GeneratorCb& gen,
        const size_t size)
        : Function(size, symbol)
        , mGen([this, gen](const Arguments& args, llvm::IRBuilder<>& B) {
            llvm::Value* result = gen(args.AsLLVMValues(), B);
            if (!result) return Value::Invalid();
            // For older callbacks, we have to figure out the type from the
            // function signature if the return type is a ptr (from LLVM 16
            // onwards)
            if (result->getType()->isPointerTy())
            {
#if LLVM_VERSION_MAJOR <= 15
                return Value(result, result->getType()->getPointerElementType());
#else
                ArgInfoVector unused;
                ArgInfo r = this->types(unused, result->getContext());
                return Value(result, r.GetUnderlyingType());
#endif
            }
            else {
                // otherwise, can we introspec the type directly
                return Value(result, result->getType());
            }
        })
        , mEmbedIR(false) {}
    IRFunctionBase(const std::string& symbol,
        const GeneratorNativeCb& gen,
        const size_t size)
        : Function(size, symbol)
        , mGen([gen](const Arguments& args, llvm::IRBuilder<>& B) {
            OPENVDB_ASSERT(args.AreNativeValues());
            return gen(args.AsNativeValues(), B);
        })
        , mEmbedIR(false) {}
    IRFunctionBase(const std::string& symbol,
        const GeneratorArgumentsCb& gen,
        const size_t size)
        : Function(size, symbol)
        , mGen(gen)
        , mEmbedIR(false) {}
    ~IRFunctionBase() override = default;

    const GeneratorArgumentsCb mGen;
    bool mEmbedIR;
};

/// @brief  Represents a concrete IR function.
template <typename SignatureT>
struct IRFunction : public IRFunctionBase
{
    using Traits = FunctionTraits<SignatureT>;
    using Ptr = std::shared_ptr<IRFunction>;

    IRFunction(const std::string& symbol, const GeneratorCb& gen)
        : IRFunctionBase(symbol, gen, Traits::N_ARGS) {}
    IRFunction(const std::string& symbol, const GeneratorNativeCb& gen)
        : IRFunctionBase(symbol, gen, Traits::N_ARGS) {}
    IRFunction(const std::string& symbol, const GeneratorArgumentsCb& gen)
        : IRFunctionBase(symbol, gen, Traits::N_ARGS) {}

    inline ArgInfo types(ArgInfoVector& types, llvm::LLVMContext& C) const override
    {
        return llvmArgTypesFromSignature<SignatureT>(C, &types);
    }

#if LLVM_VERSION_MAJOR <= 15
    inline llvm::Type*
    types(std::vector<llvm::Type*>& types, llvm::LLVMContext& C) const override
    {
        return llvmTypesFromSignature<SignatureT>(C, &types);
    }
#endif
};

/// @brief  Represents a concrete C function binding with the first argument as
///         its return type.
template <typename SignatureT>
struct CFunctionSRet : public SRetFunction<SignatureT, CFunction<SignatureT>>
{
    using BaseT = SRetFunction<SignatureT, CFunction<SignatureT>>;
    CFunctionSRet(const std::string& symbol, const SignatureT function)
        : BaseT(symbol, function) {}
    ~CFunctionSRet() override = default;
};

/// @brief  Represents a concrete IR function with the first argument as
///         its return type.
template <typename SignatureT>
struct IRFunctionSRet : public SRetFunction<SignatureT, IRFunction<SignatureT>>
{
    using BaseT = SRetFunction<SignatureT, IRFunction<SignatureT>>;
    IRFunctionSRet(const std::string& symbol,
        const IRFunctionBase::GeneratorCb& gen)
        : BaseT(symbol, gen) {}
    IRFunctionSRet(const std::string& symbol,
        const IRFunctionBase::GeneratorNativeCb& gen)
        : BaseT(symbol, gen) {}
    IRFunctionSRet(const std::string& symbol,
        const IRFunctionBase::GeneratorArgumentsCb& gen)
        : BaseT(symbol, gen) {}

    ~IRFunctionSRet() override = default;
};

/// @brief  A group of functions which all have the same name but different
///   signatures. For example:
///      float abs(float)
///      double abs(double)
///   As well as serving as a way of grouping common functions, this class
///   provides an API for selecting the best possible function signature,
///   should a match exist, against a provided set of argument types.
struct OPENVDB_AX_API FunctionGroup
{
    using Ptr = std::shared_ptr<FunctionGroup>;
    using UniquePtr = std::unique_ptr<FunctionGroup>;
    using FunctionList = std::vector<Function::Ptr>;

    FunctionGroup(const char* name,
            const char* doc,
            const FunctionList& list)
        : mName(name)
        , mDoc(doc)
        , mFunctionList(list) {}
    ~FunctionGroup() = default;

    /// @brief  Verify the function signatures in this group.
    bool HasUniqueTypeSignatures(llvm::LLVMContext& C) const;

    /// @brief  Given a vector of args, automatically returns the best
    ///         possible function declaration from the stored function list. The
    ///         'best' declaration is determined by the provided types
    ///         compatibility to each functions signature.
    /// @note   Returns a nullptr if no compatible match was found or if the
    ///         function list is empty. A compatible match is defined as an
    ///         Explicit, Implicit or Ambiguous match (where the latter returns
    ///         the first matched implicit function where other implicit
    ///         matches exist).
    ///
    /// @note   If multiple implicit matches are found, the first match is
    ///         returned and 'type' is set to Ambiguous (if provided).
    /// @warning All funcions in this group must implement the types(ArgInfo)
    ///         virtual function, which is optional in LLVM 15 but required from
    ///         LLVM 16.
    ///
    /// @param args  Args representing the function argument types
    /// @param C     The llvm context
    std::pair<const Function*, Function::SignatureMatch>
    match(const ArgInfoVector& args, llvm::LLVMContext& C) const;

    /// @brief  Given a set of Arguments, find an EXPLICIT signature match,
    ///         generate and execute the function body. If no explicit match
    ///         exists, Value::Invalid() is returned.
    /// @note   To ensure something is matched/executed, consider calling
    ///         match() and Function::call instead. This method should only be
    ///         used by internal methods that can assert an explicit match
    ///         exists.
    ///
    /// @param args     Function arguments
    /// @param B        The current llvm IRBuilder
    Value execute(const Arguments& args, llvm::IRBuilder<>& B) const;

    /// @brief  Given a set of NativeArguments, find the best possible function
    ///         signature, generate and execute the function body. Returns the
    ///         return value of the function or Value::Invalid() if no Explicit
    ///         or Implicit match is found.
    /// @note   This function will throw if no valid return is provided by the
    ///         matched declaration implementation.
    ///
    /// @param args     Natively supported function arguments
    /// @param B        The current llvm IRBuilder
    Value execute(const NativeArguments& args, llvm::IRBuilder<>& B) const;

    /// @brief  Accessor to the underlying function signature list
    inline const FunctionList& list() const { return mFunctionList; }
    const char* name() const { return mName; }
    const char* doc() const { return mDoc; }

#if LLVM_VERSION_MAJOR <= 15
    /// @warning  Does not support detecting Ambiguous functions (returns Implicit
    ///   in these cases).
    OPENVDB_DEPRECATED_MESSAGE("Switch to AX's internal ArgInfo types for LLVM 16 onwards")
    const Function*
    match(const std::vector<llvm::Type*>& types,
          llvm::LLVMContext& C,
          Function::SignatureMatch* type = nullptr) const;

    OPENVDB_DEPRECATED_MESSAGE("FunctionGroup::execute which takes llvm::Value's and "
        "supports argument matching/casting is incompatible with LLVM 16+ and will be "
        "removed.")
    llvm::Value*
    execute(const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B) const;

    /// @brief  Given a vector of llvm values, find the best possible function
    ///         signature, generate and execute the function body. Returns the
    ///         Function that was selected and executed or a nullptr if no
    ///         valid match was found. Sets the result variable to the return
    ///         value of the function (nullptr if void). If no match is found,
    ///         the result variable if left unset.
    /// @note   This function will throw if no valid return is provided by the
    ///         matched declaration implementation.
    ///
    /// @param args     A vector of values representing the function arguments
    /// @param B        The current llvm IRBuilder
    /// @param result   The result to set. nullptr on void return.
    /// @return The matched function. nullptr if no match was found
    OPENVDB_DEPRECATED_MESSAGE("FunctionGroup::execute which takes llvm::Value's and "
        "supports argument matching/casting is incompatible with LLVM 16+ and will be "
        "removed.")
    const Function*
    execute(const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B,
            llvm::Value*& result) const;
#endif

private:
    const char* mName;
    const char* mDoc;
    const FunctionList mFunctionList;
};

/// @brief  The FunctionBuilder class provides a builder pattern framework to
///         allow easy and valid construction of AX functions. There are a
///         number of complex tasks which may need to be performed during
///         construction of C or IR function which are delegated to this
///         builder, whilst ensuring that the constructed functions are
///         guaranteed to be valid.
/// @details  Use the FunctionBuilder::addSignature methods to append function
///           signatures. Finalize the group of functions with
///           FunctionBuilder::get.
struct FunctionBuilder
{
    enum DeclPreferrence {
        C, IR, Any
    };

    struct Settings
    {
        using Ptr = std::shared_ptr<Settings>;

        inline bool isDefault() const
        {
            if (mNames) return false;
            if (mType != Function::Type::kNormal) return false;
            if (!mDeps.empty()) return false;
            if (mConstantFold || mEmbedIR || mReadOnly) return false;
            if (!mFnAttrs.empty()) return false;
            if (!mRetAttrs.empty()) return false;
            if (!mParamAttrs.empty()) return false;
            return true;
        }

        std::unique_ptr<SmallArgumentVector<const char*>> mNames = nullptr;
        SmallArgumentVector<const char*> mDeps = {};
        bool mConstantFold = false;
        bool mEmbedIR = false;
        bool mReadOnly = false;
        Function::Type mType = Function::Type::kNormal;
        SmallArgumentVector<llvm::Attribute::AttrKind> mFnAttrs = {};
        SmallArgumentVector<llvm::Attribute::AttrKind> mRetAttrs = {};
        std::map<size_t, SmallArgumentVector<llvm::Attribute::AttrKind>> mParamAttrs = {};
    };

    FunctionBuilder(const char* name)
        : mName(name)
        , mCurrentSettings(std::make_shared<Settings>()) {}

    // C-Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const Signature* ptr, const char* symbol = nullptr)
    {
        using CFType = typename std::conditional
            <!SRet, CFunction<Signature>, CFunctionSRet<Signature>>::type;
        const std::string s = symbol ? symbol : this->genSymbol<Signature>();
        this->addSignatureImpl<CFType>(s, ptr);
        return *this;
    }

    // Non native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorCb& cb,
            const char* symbol = nullptr)
    {
        using IRFType = typename std::conditional
            <!SRet, IRFunction<Signature>, IRFunctionSRet<Signature>>::type;
        OPENVDB_ASSERT(!(SRet && mCurrentSettings->mReadOnly));
        const std::string s = symbol ? symbol : this->genSymbol<Signature>();
        this->addSignatureImpl<IRFType>(s, cb);
        return *this;
    }

    // Non native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorCb& cb, const Signature* ptr, const char* symbol = nullptr)
    {
        this->addSignature<Signature, SRet>(cb, symbol);
        this->addSignature<Signature, SRet>(ptr, symbol);
        return *this;
    }

    // Native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorNativeCb& cb, const char* symbol = nullptr)
    {
        static_assert(ArgumentIterator<Signature>::IsNativeSignature);
        using IRFType = typename std::conditional
            <!SRet, IRFunction<Signature>, IRFunctionSRet<Signature>>::type;
        OPENVDB_ASSERT(!(SRet && mCurrentSettings->mReadOnly));
        const std::string s = symbol ? symbol : this->genSymbol<Signature>();
        this->addSignatureImpl<IRFType>(s, cb);
        return *this;
    }

    // Native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorNativeCb& cb, const Signature* ptr, const char* symbol = nullptr)
    {
        static_assert(ArgumentIterator<Signature>::IsNativeSignature);
        this->addSignature<Signature, SRet>(cb, symbol);
        this->addSignature<Signature, SRet>(ptr, symbol);
        return *this;
    }

    // Native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorArgumentsCb& cb,
            const char* symbol = nullptr)
    {
        using IRFType = typename std::conditional
            <!SRet, IRFunction<Signature>, IRFunctionSRet<Signature>>::type;
        OPENVDB_ASSERT(!(SRet && mCurrentSettings->mReadOnly));
        const std::string s = symbol ? symbol : this->genSymbol<Signature>();
        this->addSignatureImpl<IRFType>(s, cb);
        return *this;
    }

    // Native Binding
    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorArgumentsCb& cb, const Signature* ptr, const char* symbol = nullptr)
    {
        this->addSignature<Signature, SRet>(cb, symbol);
        this->addSignature<Signature, SRet>(ptr, symbol);
        return *this;
    }

    inline FunctionBuilder& addDependency(const char* name)
    {
        mCurrentSettings->mDeps.emplace_back(name); return *this;
    }

    inline FunctionBuilder& setEmbedIR(bool on) { mCurrentSettings->mEmbedIR = on; return *this; }
    inline FunctionBuilder& setConstantFold(bool on) { mCurrentSettings->mConstantFold = on; return *this; }
    inline FunctionBuilder& setArgumentNames(const std::vector<const char*>& names)
    {
        mCurrentSettings->mNames = std::make_unique<SmallArgumentVector<const char*>>();
        mCurrentSettings->mNames->assign(names.begin(), names.end());
        return *this;
    }

    /// @brief Parameter and Function Attributes. When designing a C binding,
    ///   llvm will be unable to assign parameter markings to the return
    ///   type, function body or parameter attributes due to there not
    ///   being any visibility on the function itself during codegen.
    ///   The best way to ensure performant C bindings is to ensure
    ///   that the function is marked with the required llvm parameters.
    /// @note  Some of the most common are shown in FunctionBuilder::setBuiltin,
    ///   but also consider FunctionBuilder::setReadOnly
    inline FunctionBuilder&
    addParameterAttribute(const size_t idx, const llvm::Attribute::AttrKind attr)
    {
        mCurrentSettings->mParamAttrs[idx].emplace_back(attr);
        return *this;
    }

    inline FunctionBuilder&
    addReturnAttribute(const llvm::Attribute::AttrKind attr)
    {
        mCurrentSettings->mRetAttrs.emplace_back(attr);
        return *this;
    }

    inline FunctionBuilder&
    addFunctionAttribute(const llvm::Attribute::AttrKind attr)
    {
        mCurrentSettings->mFnAttrs.emplace_back(attr);
        return *this;
    }

    /// @brief  Mark functions currently sharing settings with as "readonly".
    ///   This enables the strictest possible memory effects for this function
    ///   in llvm and implies that the function does not write to any memory
    ///   (i.e. CreateStore) and only reads memory directly from its function
    ///   arguments (i.e. nothing external to the function). Functions marked
    ///   as "readonly" but do not adhere to the above will cause UB.
    /// @note  Obviously invalid for SRET functions!
    inline FunctionBuilder&
    setReadOnly(const bool on)
    {
        mCurrentSettings->mReadOnly = on;
        return *this;
    }

    /// @brief  Mark functions currently sharing settings as builtin AX methods.
    ///   At compile time, this causes the IR function body, arguments and
    ///   return value to be marked with a set of default attributes that
    ///   apply to all builtin methods, as they are expected to only be called
    ///   by AX programs. Currently this results in:
    ///
    ///    Function Attributes:
    ///        llvm::Attribute::NoFree
    ///        llvm::Attribute::WillReturn
    ///        llvm::Attribute::NoRecurse
    ///        llvm::Attribute::NoUnwind
    ///        llvm::Attribute::AlwaysInline
    ///    Return Attributes: -
    ///    Parameter Attributes (Ptrs):
    ///        llvm::Attribute::NonNull
    ///        llvm::Attribute::NoUndef
    ///        llvm::Attribute::NoFree
    ///        llvm::Attribute::NoCapture / (llvm::CaptureInfo::none() from LLVM 21)
    ///
    /// @warning Attributes in this method may be extended - as such, you
    ///   should mark external function with individual attributes instead of
    ///   calling this.
    inline FunctionBuilder&
    setBuiltin()
    {
        // note that we have to defer the attribute setting to compile time as
        // various attributes need to know the function types. Would be nice if
        // we could do it all in the builder.
        mCurrentSettings->mType = Function::Type::kBuiltin;
        return *this;
    }

    /// @brief  Mark this function as an external kernel entry point. Should
    ///   only be used by the ComputeGenerators. Results in a number of function
    ///   attributes/parameters being set.
    inline FunctionBuilder&
    setExternalKernel()
    {
        mCurrentSettings->mType = Function::Type::kKernel;
        return *this;
    }

    inline FunctionBuilder& setDocumentation(const char* doc)
    {
        mDoc = doc;
        return *this;
    }

    inline FunctionBuilder& setPreferredImpl(DeclPreferrence pref)
    {
        mDeclPref = pref;
        return *this;
    }

    inline FunctionGroup::UniquePtr get() const
    {
        for (auto& decl : mCFunctions)
        {
            const auto& s = mSettings.at(decl.get());
            if (s->mNames) decl->mNames = *s->mNames;
            decl->mDeps = s->mDeps;
            decl->setConstantFold(s->mConstantFold);
            if (!s->mFnAttrs.empty())  decl->attrs().mFnAttrs = s->mFnAttrs;
            if (!s->mRetAttrs.empty()) decl->attrs().mRetAttrs = s->mRetAttrs;
            if (!s->mParamAttrs.empty()) {
                for (auto& idxAttrs : s->mParamAttrs) {
                    if (idxAttrs.first > decl->size()) continue;
                    decl->attrs().mParamAttrs[idxAttrs.first] = idxAttrs.second;
                }
            }
            if (s->mReadOnly) decl->attrs().mReadOnly = true;
            if (s->mType != Function::Type::kNormal) {
                decl->attrs().mType = s->mType;
            }
        }

        for (auto& decl : mIRFunctions)
        {
            const auto& s = mSettings.at(decl.get());
            if (s->mNames) decl->mNames = *s->mNames;
            decl->mDeps = s->mDeps;
            decl->setEmbedIR(s->mEmbedIR);
            if (!s->mFnAttrs.empty())  decl->attrs().mFnAttrs = s->mFnAttrs;
            if (!s->mRetAttrs.empty()) decl->attrs().mRetAttrs = s->mRetAttrs;
            if (!s->mParamAttrs.empty()) {
                for (auto& idxAttrs : s->mParamAttrs) {
                    if (idxAttrs.first > decl->size()) continue;
                    decl->attrs().mParamAttrs[idxAttrs.first] = idxAttrs.second;
                }
            }
            if (s->mReadOnly) decl->attrs().mReadOnly = true;
            if (s->mType != Function::Type::kNormal) {
                decl->attrs().mType = s->mType;
            }
        }

        FunctionGroup::FunctionList functions;

        if (mDeclPref == DeclPreferrence::IR) {
            functions.insert(functions.end(), mIRFunctions.begin(), mIRFunctions.end());
        }
        if (mDeclPref == DeclPreferrence::C) {
            functions.insert(functions.end(), mCFunctions.begin(), mCFunctions.end());
        }
        if (functions.empty()) {
            functions.insert(functions.end(), mIRFunctions.begin(), mIRFunctions.end());
            functions.insert(functions.end(), mCFunctions.begin(), mCFunctions.end());
        }

        return std::make_unique<FunctionGroup>(mName, mDoc, std::move(functions));
    }

private:
    template <typename FunctionT, typename...Args>
    inline FunctionBuilder& addSignatureImpl(Args&& ...args)
    {
        Settings::Ptr settings = mCurrentSettings;
        if (!mCurrentSettings->isDefault()) {
            settings = std::make_shared<Settings>();
        }
        auto ptr = std::make_shared<FunctionT>(std::move(args)...);
        if constexpr (std::is_base_of_v<IRFunctionBase, FunctionT>) {
            mIRFunctions.emplace_back(ptr);
        }
        else {
            static_assert(std::is_base_of_v<CFunctionBase, FunctionT>);
            mCFunctions.emplace_back(ptr);
        }
        mSettings[ptr.get()] = settings;
        mCurrentSettings = settings;
        return *this;
    }

    template <typename Signature>
    std::string genSymbol() const
    {
        using Traits = FunctionTraits<Signature>;

        std::string args;
        auto callback = [&args](auto type) {
            using Type = decltype(type);
            args += TypeToSymbol<Type>::s();
        };

        ArgumentIterator<Signature>::apply(callback, /*forwards*/true);
        /// @note  important to prefix all symbols with "ax." so that
        ///        they will never conflict with internal llvm symbol
        ///        names (such as standard library methods e.g, cos, cosh

        // assemble the symbol
        return "ax." + std::string(this->mName) + "." +
            TypeToSymbol<typename Traits::ReturnType>::s() + args;
    }

private:
    const char* mName = "";
    const char* mDoc = "";
    DeclPreferrence mDeclPref = IR;
    std::vector<CFunctionBase::Ptr> mCFunctions = {};
    std::vector<IRFunctionBase::Ptr> mIRFunctions = {};
    std::map<const Function*, Settings::Ptr> mSettings = {};
    Settings::Ptr mCurrentSettings = nullptr;
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_FUNCTION_TYPES_HAS_BEEN_INCLUDED

