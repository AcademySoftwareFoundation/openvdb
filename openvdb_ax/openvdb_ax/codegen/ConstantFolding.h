// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/ConstantFolding.h
///
/// @authors Nick Avramoussis
///
/// @brief  Constant folding for C++ bindings.
///

#ifndef OPENVDB_AX_CODEGEN_CONSTANT_FOLDING_HAS_BEEN_INCLUDED
#define OPENVDB_AX_CODEGEN_CONSTANT_FOLDING_HAS_BEEN_INCLUDED

#include "Types.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <llvm/IR/Constants.h>

#include <type_traits>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  Constant folding support structure
///
template <typename SignatureT,
    size_t I = FunctionTraits<SignatureT>::N_ARGS>
struct ConstantFolder
{
    using ArgT = typename FunctionTraits<SignatureT>::template Arg<I-1>;
    using ArgumentValueType = typename ArgT::Type;

    // @brief    Attempts evaluate a given function with a provided set of constant llvm
    //           values. If successful, the function is invoked and the result is stored
    //           and returned in an llvm::Value.
    // @details  Currently only scalar constant folding is supported due to the way
    //           vectors and matrices are alloced. Functions which return void are also
    //           not supported for constant folding.
    // @param args  The vector of llvm constants that comprise the function arguments.
    //              Note that the size of this vector is expected to match the size of
    //              the required function arguments and the templated parameter I
    // @param function  The function to invoke if all arguments have a valid mapping.
    // @param C   The llvm Context
    // @param ts  The list of evaluated C types from the provided llvm constants. This
    //            is expected to be empty (not provided) on the first call to fold and
    //            is used on subsequent recursive calls.
    template <typename ...Tys>
    static llvm::Value*
    fold(const std::vector<llvm::Constant*>& args,
         const SignatureT& function,
         llvm::LLVMContext& C,
         Tys&&... ts)
    {
        OPENVDB_ASSERT(I-1 < args.size());
        llvm::Constant* constant = args[I-1];
        const llvm::Type* type = constant->getType();
        if (type->isIntegerTy()) {
            OPENVDB_ASSERT(llvm::isa<llvm::ConstantInt>(constant));
            llvm::ConstantInt* cint =
                llvm::cast<llvm::ConstantInt>(constant);
            const uint64_t val = cint->getLimitedValue();
            return call<uint64_t, ArgumentValueType>(args, function, C, val, ts...);
        }
        else if (type->isFloatTy() || type->isDoubleTy()) {
            OPENVDB_ASSERT(llvm::isa<llvm::ConstantFP>(constant));
            llvm::ConstantFP* cfp =
                llvm::cast<llvm::ConstantFP>(constant);
            const llvm::APFloat& apf = cfp->getValueAPF();
            if (type->isFloatTy()) {
                const float val = apf.convertToFloat();
                return call<float, ArgumentValueType>(args, function, C, val, ts...);
            }
            if (type->isDoubleTy()) {
                const double val = apf.convertToDouble();
                return call<double, ArgumentValueType>(args, function, C, val, ts...);
            }
        }
        else if (type->isArrayTy()) {
            // @todo currently all arrays are alloced anyway which
            // needs to be handled or changed
            return nullptr;
        }
        // fallback
        return nullptr;
    }
private:
    // @brief  Specialization for supported implicit casting matching AX's supported
    //         scalar casting. Continues to traverse the constant argument list.
    template <typename In, typename Out, typename ...Tys>
    static typename std::enable_if<std::is_convertible<In, Out>::value, llvm::Value*>::type
    call(const std::vector<llvm::Constant*>& args,
         const SignatureT& function,
         llvm::LLVMContext& C,
         const In& arg,
         Tys&&... ts)
    {
        using Next = ConstantFolder<SignatureT, I-1>;
        return Next::fold(args, function, C, Out(arg), ts...);
    }

    // @brief  Specialization for unsupported implicit casting. Bails out with a
    //         nullptr return.
    template <typename In, typename Out, typename ...Tys>
    static typename std::enable_if<!std::is_convertible<In, Out>::value, llvm::Value*>::type
    call(const std::vector<llvm::Constant*>&,
         const SignatureT&,
         llvm::LLVMContext&,
         const In&, Tys&&...)
    {
        return nullptr;
    }
};

template <typename SignatureT>
struct ConstantFolder<SignatureT, 0>
{
    // @brief  The final call to fold when all arguments have been evaluated (or no
    //         arguments exist).
    template <typename ...Tys>
    static llvm::Value*
    fold(const std::vector<llvm::Constant*>& args,
         const SignatureT& function,
         llvm::LLVMContext& C,
         Tys&&... ts)
    {
        using ReturnT = typename FunctionTraits<SignatureT>::ReturnType;
        return call<ReturnT>(args, function, C, ts...);
    }

private:

    // @brief  Specialization for the invoking of the provided function if the return
    //         type is not void or a pointer
    template <typename ReturnT, typename ...Tys>
    static typename std::enable_if<!std::is_pointer<ReturnT>::value &&
        !std::is_same<ReturnT, void>::value, llvm::Value*>::type
    call(const std::vector<llvm::Constant*>&,
         const SignatureT& function,
         llvm::LLVMContext& C,
         Tys&&... ts)
    {
        const ReturnT result = function(ts...);
        return LLVMType<ReturnT>::get(C, result);
    }

    // @brief  Specialization if the return type is void or a pointer. No folding is
    //         supported.
    template <typename ReturnT, typename ...Tys>
    static typename std::enable_if<std::is_pointer<ReturnT>::value ||
        std::is_same<ReturnT, void>::value, llvm::Value*>::type
    call(const std::vector<llvm::Constant*>&,
         const SignatureT&,
         llvm::LLVMContext&,
         Tys&&...)
    {
        return nullptr;
    }
};

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_CODEGEN_CONSTANT_FOLDING_HAS_BEEN_INCLUDED

