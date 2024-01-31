// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
#include "Utils.h" // isValidCast
#include "ConstantFolding.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

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

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief  Object to array conversion methods to allow functions to return
///         vector types. These containers provided an interface for automatic
///         conversion of C++ objects to LLVM types as array types.

template <typename T, size_t _SIZE = 1>
struct ArgType {
    using Type = T;
    static const size_t SIZE = _SIZE;
    using ArrayType = Type[SIZE];
    ArrayType mData;
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
    template <typename OpT>
    static void apply(const OpT&, const bool) {}
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// @brief  Populate a vector of llvm types from a function signature declaration.
///
/// @param C  The llvm context
/// @param types   A vector of types to populate
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
    return LLVMType<typename Traits::ReturnType>::get(C);
}

/// @brief  Generate an LLVM FunctionType from a function signature
///
/// @param C  The llvm context
///
template <typename SignatureT>
inline llvm::FunctionType*
llvmFunctionTypeFromSignature(llvm::LLVMContext& C)
{
    std::vector<llvm::Type*> types;
    llvm::Type* returnType =
        llvmTypesFromSignature<SignatureT>(C, &types);
    return llvm::FunctionType::get(/*Result=*/returnType,
            /*Params=*/llvm::ArrayRef<llvm::Type*>(types),
            /*isVarArg=*/false);
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
               const std::vector<llvm::Type*>& types,
               const llvm::Type* returnType,
               const char* name = nullptr,
               const std::vector<const char*>& names = {},
               const bool axTypes = false);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

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

    /// @brief  Populate a vector of llvm::Types which describe this function
    ///         signature. This method is used by Function::create,
    ///         Function::print and Function::match.
    virtual llvm::Type* types(std::vector<llvm::Type*>&, llvm::LLVMContext&) const = 0;

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
    virtual llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast = false) const;

    /// @brief  The result type from calls to Function::match
    enum SignatureMatch { None = 0, Size, Implicit, Explicit };

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
    /// @note  Due to the way CFunctionSRet is implemented, the LLVM Context
    ///        must be provided in case we have a zero arg function signature
    ///        with a SRET.
    /// @param inputs  The input types
    /// @param C       The LLVM Context
    virtual SignatureMatch match(const std::vector<llvm::Type*>& inputs, llvm::LLVMContext& C) const;

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
    inline const char* argName(const size_t idx) const {
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

    /// Builder methods

    inline bool hasParamAttribute(const size_t i,
            const llvm::Attribute::AttrKind& kind) const
    {
        if (!mAttributes) return false;
        const auto iter = mAttributes->mParamAttrs.find(i);
        if (iter == mAttributes->mParamAttrs.end()) return false;
        const auto& vec = iter->second;
        return std::find(vec.begin(), vec.end(), kind) != vec.end();
    }

    inline void setArgumentNames(std::vector<const char*> names) { mNames = names; }

    const std::vector<const char*>& dependencies() const { return mDeps; }
    inline void setDependencies(std::vector<const char*> deps) { mDeps = deps; }

    inline void setFnAttributes(const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mFnAttrs = in;
    }
    inline void setRetAttributes(const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mRetAttrs = in;
    }
    inline void setParamAttributes(const size_t i,
            const std::vector<llvm::Attribute::AttrKind>& in)
    {
        this->attrs().mParamAttrs[i] = in;
    }

protected:

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

private:

    struct Attributes {
        std::vector<llvm::Attribute::AttrKind> mFnAttrs, mRetAttrs;
        std::map<size_t, std::vector<llvm::Attribute::AttrKind>> mParamAttrs;
    };

    inline Attributes& attrs() {
        if (!mAttributes) mAttributes.reset(new Attributes());
        return *mAttributes;
    }

    llvm::AttributeList flattenAttrs(llvm::LLVMContext& C) const;

    const size_t mSize;
    const std::string mSymbol;
    std::unique_ptr<Attributes> mAttributes;
    std::vector<const char*> mNames;
    std::vector<const char*> mDeps;
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
    using SRetType = typename std::remove_pointer<FirstArgument>::type;

public:

    /// @brief  Override of match which inserts the SRET type such that the base
    ///         class methods ignore it.
    Function::SignatureMatch match(const std::vector<llvm::Type*>& args,
            llvm::LLVMContext& C) const override
    {
        // append return type and right rotate
        std::vector<llvm::Type*> inputs(args);
        inputs.emplace_back(LLVMType<SRetType*>::get(C));
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

    /// @brief  Override of print to avoid printing out the SRET type
    void print(llvm::LLVMContext& C,
           std::ostream& os,
           const char* name = nullptr,
           const bool axTypes = true) const override
    {
        std::vector<llvm::Type*> current;
        llvm::Type* ret = this->types(current, C);
        // left rotate
        std::rotate(current.begin(), current.begin() + 1, current.end());
        ret = current.back();
        current.pop_back();

        std::vector<const char*> names;
        names.reserve(this->size());
        for (size_t i = 0; i < this->size()-1; ++i) {
            names.emplace_back(this->argName(i));
        }
        printSignature(os, current, ret, name, names, axTypes);
    }

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

    inline virtual llvm::Value* fold(const std::vector<llvm::Value*>&,
            llvm::LLVMContext&) const {
        return nullptr;
    }

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

    inline llvm::Type* types(std::vector<llvm::Type*>& types, llvm::LLVMContext& C) const override
    {
        return llvmTypesFromSignature<SignatureT>(C, &types);
    }

    inline uint64_t address() const override final {
        return reinterpret_cast<uint64_t>(mFunction);
    }

    llvm::Value*
    call(const std::vector<llvm::Value*>& args,
         llvm::IRBuilder<>& B,
         const bool cast) const override
    {
        llvm::Value* result = this->fold(args, B.getContext());
        if (result) return result;
        return Function::call(args, B, cast);
    }

    llvm::Value* fold(const std::vector<llvm::Value*>& args, llvm::LLVMContext& C) const override final
    {
        auto allconst =
            [](const std::vector<llvm::Value*>& vals) -> bool {
            for (auto& value : vals) {
                if (!llvm::isa<llvm::Constant>(value)) return false;
            }
            return true;
        };

        if (!this->hasConstantFold()) return nullptr;
        if (!allconst(args))  return nullptr;
        std::vector<llvm::Constant*> constants;
        constants.reserve(args.size());
        for (auto& value : args) {
            constants.emplace_back(llvm::cast<llvm::Constant>(value));
        }

        // no guarantee that fold() will be able to cast all arguments
        return ConstantFolder<SignatureT>::fold(constants, *mFunction, C);
    }

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
    /// @note     You can return a nullptr from this method which will represent
    ///           a ret void, a ret void instruction, or an actual value
    using GeneratorCb = std::function<llvm::Value*
        (const std::vector<llvm::Value*>&, llvm::IRBuilder<>&)>;

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
         llvm::IRBuilder<>& B,
         const bool cast) const override;

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
        , mGen(gen)
        , mEmbedIR(false) {}
    ~IRFunctionBase() override = default;

    const GeneratorCb mGen;
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

    inline llvm::Type*
    types(std::vector<llvm::Type*>& types, llvm::LLVMContext& C) const override
    {
        return llvmTypesFromSignature<SignatureT>(C, &types);
    }
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
    ~IRFunctionSRet() override = default;
};

/// @brief  todo
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

    /// @brief  Given a vector of llvm types, automatically returns the best
    ///         possible function declaration from the stored function list. The
    ///         'best' declaration is determined by the provided types
    ///         compatibility to each functions signature.
    /// @note   If multiple implicit matches are found, the first match is
    ///         returned.
    /// @note   Returns a nullptr if no compatible match was found or if the
    ///         function list is empty. A compatible match is defined as an
    ///         Explicit or Implicit match.
    ///
    /// @param types  A vector of types representing the function argument types
    /// @param C      The llvm context
    /// @param type   If provided, type is set to the type of match that occurred
    const Function*
    match(const std::vector<llvm::Type*>& types,
          llvm::LLVMContext& C,
          Function::SignatureMatch* type = nullptr) const;

    /// @brief  Given a vector of llvm values, find the best possible function
    ///         signature, generate and execute the function body. Returns the
    ///         return value of the function (nullptr if void). The behaviour
    ///         is undefined if a valid match does not exist. For such cases,
    ///         call the second version of FunctionGroup::execute.
    /// @note   This function will throw if no valid return is provided by the
    ///         matched declaration implementation.
    ///
    /// @param args     A vector of values representing the function arguments
    /// @param B        The current llvm IRBuilder
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
    const Function*
    execute(const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B,
            llvm::Value*& result) const;

    /// @brief  Accessor to the underlying function signature list
    inline const FunctionList& list() const { return mFunctionList; }
    const char* name() const { return mName; }
    const char* doc() const { return mDoc; }

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

        inline bool isDefault() const {
            if (mNames) return false;
            if (!mDeps.empty()) return false;
            if (mConstantFold || mEmbedIR) return false;
            if (!mFnAttrs.empty()) return false;
            if (!mRetAttrs.empty()) return false;
            if (!mParamAttrs.empty()) return false;
            return true;
        }

        std::shared_ptr<std::vector<const char*>> mNames = nullptr;
        std::vector<const char*> mDeps = {};
        bool mConstantFold = false;
        bool mEmbedIR = false;
        std::vector<llvm::Attribute::AttrKind> mFnAttrs = {};
        std::vector<llvm::Attribute::AttrKind> mRetAttrs = {};
        std::map<size_t, std::vector<llvm::Attribute::AttrKind>> mParamAttrs = {};
    };

    FunctionBuilder(const char* name)
        : mName(name)
        , mCurrentSettings(new Settings()) {}


    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorCb& cb,
            const char* symbol = nullptr)
    {
        using IRFType = typename std::conditional
            <!SRet, IRFunction<Signature>, IRFunctionSRet<Signature>>::type;
        using IRPtr = typename IRFType::Ptr;

        Settings::Ptr settings = mCurrentSettings;
        if (!mCurrentSettings->isDefault()) {
            settings.reset(new Settings());
        }

        std::string s;
        if (symbol) s = std::string(symbol);
        else s = this->genSymbol<Signature>();

        auto ir = IRPtr(new IRFType(s, cb));
        mIRFunctions.emplace_back(ir);
        mSettings[ir.get()] = settings;
        mCurrentSettings = settings;
        return *this;
    }

    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const Signature* ptr,
            const char* symbol = nullptr)
    {
        using CFType = typename std::conditional
            <!SRet, CFunction<Signature>, CFunctionSRet<Signature>>::type;
        using CPtr = typename CFType::Ptr;

        Settings::Ptr settings = mCurrentSettings;
        if (!mCurrentSettings->isDefault()) {
            settings.reset(new Settings());
        }

        std::string s;
        if (symbol) s = std::string(symbol);
        else s = this->genSymbol<Signature>();

        auto c = CPtr(new CFType(s, ptr));
        mCFunctions.emplace_back(c);
        mSettings[c.get()] = settings;
        mCurrentSettings = settings;
        return *this;
    }

    template <typename Signature, bool SRet = false>
    inline FunctionBuilder&
    addSignature(const IRFunctionBase::GeneratorCb& cb, const Signature* ptr, const char* symbol = nullptr)
    {
        this->addSignature<Signature, SRet>(cb, symbol);
        this->addSignature<Signature, SRet>(ptr, symbol);
        return *this;
    }

    inline FunctionBuilder& addDependency(const char* name) {
        mCurrentSettings->mDeps.emplace_back(name); return *this;
    }

    inline FunctionBuilder& setEmbedIR(bool on) { mCurrentSettings->mEmbedIR = on; return *this; }
    inline FunctionBuilder& setConstantFold(bool on) { mCurrentSettings->mConstantFold = on; return *this; }
    inline FunctionBuilder& setArgumentNames(const std::vector<const char*>& names) {
        mCurrentSettings->mNames.reset(new std::vector<const char*>(names));
        return *this;
    }

    /// @details  Parameter and Function Attributes. When designing a C binding,
    ///           llvm will be unable to assign parameter markings to the return
    ///           type, function body or parameter attributes due to there not
    ///           being any visibility on the function itself during codegen.
    ///           The best way to ensure performant C bindings is to ensure
    ///           that the function is marked with the required llvm parameters.
    ///           Some of the heavy hitters (which can have the most impact)
    ///           are below:
    ///
    ///           Functions:
    ///             - norecurse
    ///                 This function attribute indicates that the function does
    ///                 not call itself either directly or indirectly down any
    ///                 possible call path.
    ///
    ///             - willreturn
    ///                 This function attribute indicates that a call of this
    ///                 function will either exhibit undefined behavior or comes
    ///                 back and continues execution at a point in the existing
    ///                 call stack that includes the current invocation.
    ///
    ///             - nounwind
    ///                 This function attribute indicates that the function never
    ///                 raises an exception.
    ///
    ///             - readnone
    ///                 On a function, this attribute indicates that the function
    ///                 computes its result (or decides to unwind an exception) based
    ///                 strictly on its arguments, without dereferencing any pointer
    ///                 arguments or otherwise accessing any mutable state (e.g. memory,
    ///                 control registers, etc) visible to caller functions.
    ///
    ///             - readonly
    ///                 On a function, this attribute indicates that the function
    ///                 does not write through any pointer arguments (including byval
    ///                 arguments) or otherwise modify any state (e.g. memory, control
    ///                 registers, etc) visible to caller functions.
    ///                 control registers, etc) visible to caller functions.
    ///
    ///             - writeonly
    ///                 On a function, this attribute indicates that the function may
    ///                 write to but does not read from memory.
    ///
    ///           Parameters:
    ///             - noalias
    ///                 This indicates that objects accessed via pointer values based
    ///                 on the argument or return value are not also accessed, during
    ///                 the execution of the function, via pointer values not based on
    ///                 the argument or return value.
    ///
    ///             - nonnull
    ///                 This indicates that the parameter or return pointer is not null.
    ///
    ///             - readonly
    ///                 Indicates that the function does not write through this pointer
    ///                 argument, even though it may write to the memory that the pointer
    ///                 points to.
    ///
    ///             - writeonly
    ///                 Indicates that the function may write to but does not read through
    ///                 this pointer argument (even though it may read from the memory
    ///                 that the pointer points to).
    ///
    inline FunctionBuilder&
    addParameterAttribute(const size_t idx, const llvm::Attribute::AttrKind attr) {
        mCurrentSettings->mParamAttrs[idx].emplace_back(attr);
        return *this;
    }

    inline FunctionBuilder&
    addReturnAttribute(const llvm::Attribute::AttrKind attr)  {
        mCurrentSettings->mRetAttrs.emplace_back(attr);
        return *this;
    }

    inline FunctionBuilder&
    addFunctionAttribute(const llvm::Attribute::AttrKind attr)  {
        mCurrentSettings->mFnAttrs.emplace_back(attr);
        return *this;
    }

    inline FunctionBuilder& setDocumentation(const char* doc) { mDoc = doc; return *this; }
    inline FunctionBuilder& setPreferredImpl(DeclPreferrence pref) { mDeclPref = pref; return *this; }

    inline FunctionGroup::UniquePtr get() const
    {
        for (auto& decl : mCFunctions) {
            const auto& s = mSettings.at(decl.get());
            decl->setDependencies(s->mDeps);
            decl->setConstantFold(s->mConstantFold);
            if (!s->mFnAttrs.empty())  decl->setFnAttributes(s->mFnAttrs);
            if (!s->mRetAttrs.empty()) decl->setRetAttributes(s->mRetAttrs);
            if (!s->mParamAttrs.empty()) {
                for (auto& idxAttrs : s->mParamAttrs) {
                    if (idxAttrs.first > decl->size()) continue;
                    decl->setParamAttributes(idxAttrs.first, idxAttrs.second);
                }
            }
            if (s->mNames) decl->setArgumentNames(*s->mNames);
        }

        for (auto& decl : mIRFunctions) {
            const auto& s = mSettings.at(decl.get());
            decl->setDependencies(s->mDeps);
            decl->setEmbedIR(s->mEmbedIR);
            if (!s->mFnAttrs.empty())  decl->setFnAttributes(s->mFnAttrs);
            if (!s->mRetAttrs.empty()) decl->setRetAttributes(s->mRetAttrs);
            if (!s->mParamAttrs.empty()) {
                for (auto& idxAttrs : s->mParamAttrs) {
                    if (idxAttrs.first > decl->size()) continue;
                    decl->setParamAttributes(idxAttrs.first, idxAttrs.second);
                }
            }
            if (s->mNames) decl->setArgumentNames(*s->mNames);
        }

        std::vector<Function::Ptr> functions;

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

        FunctionGroup::UniquePtr group(new FunctionGroup(mName, mDoc, functions));
        return group;
    }

private:

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

