// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/FunctionTypes.cc

#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include "../Exceptions.h"

#include <openvdb/util/Name.h>
#include <openvdb/util/Assert.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/ModRef.h> // MemoryEffects

#include <vector>
#include <set>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

namespace {

inline void
printType(const ArgInfo type, llvm::raw_os_ostream& stream, const bool axTypes)
{
    const ast::tokens::CoreType token =
        axTypes ? tokenFromLLVMType(type.GetUnderlyingType()) : ast::tokens::UNKNOWN;
    if (token == ast::tokens::UNKNOWN) type.GetType()->print(stream);
    else {
        stream << ast::tokens::typeStringFromToken(token);
        // don't print ptrs if axtypes is true
        if (axTypes) return;
        for (uint8_t i = 0; i < type.NumPtrs(); ++i) stream << '*';
    }
}

inline void
printTypes(llvm::raw_os_ostream& stream,
           const ArgInfoVector& types,
           const llvm::ArrayRef<const char*>& names = {},
           const std::string sep = "; ",
           const bool axTypes = false)
{
    if (types.empty()) return;
    auto typeIter = types.begin();
    auto nameIter = names.begin();

    for (; typeIter != types.end() - 1; ++typeIter) {
        printType(*typeIter, stream, axTypes);
        if (!names.empty() && nameIter != names.end()) {
            if (*nameIter && (*nameIter)[0] != '\0') {
                stream << ' ' << *nameIter;
            }
            ++nameIter;
        }
        stream << sep;
    }

    printType(*typeIter, stream, axTypes);
    if (!names.empty() && nameIter != names.end()) {
        if (*nameIter && (*nameIter)[0] != '\0') {
            stream << ' ' << *nameIter;
        }
    }
}

}

void
printSignature(std::ostream& os,
               const ArgInfoVector& signature,
               const ArgInfo& returnType,
               const char* name,
               const llvm::ArrayRef<const char*>& names,
               const bool axTypes)
{
    llvm::raw_os_ostream stream(os);
    printType(returnType, stream, axTypes);
    if (name && name[0] != '\0') stream << ' ' << name;
    stream << '(';
    printTypes(stream, signature, names, "; ", axTypes);
    stream << ')';
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

NativeArguments NativeArguments::Cast(const Function& F, llvm::IRBuilder<>& B) const
{
    NativeArguments newargs(*this);
    if (newargs.size() == 0) return newargs;

    // Cast arguments
    ArgInfoVector arginfo;
    [[maybe_unused]] ArgInfo ret = F.types(arginfo, B.getContext());
    OPENVDB_ASSERT(!arginfo.empty());

    if (arginfo.front().IsReturn())
    {
        OPENVDB_ASSERT(ret.IsVoid());
        // first argument is the return type, ignore it
        arginfo.erase(arginfo.begin());
    }

    OPENVDB_ASSERT(newargs.size() == arginfo.size());
    for (size_t i = 0; i < newargs.size(); ++i)
    {
        if (i >= arginfo.size()) return newargs;
        Value& value = newargs[i];
        OPENVDB_ASSERT(!(value.IsPtr() && value.IsScalar()));
        const ArgInfo& targetType = arginfo[i];

        if (value.IsScalar()) {
            if (targetType.GetUnderlyingType()->isIntegerTy(1)) {
                // assume boolean target value
                value = value.ScalarBoolComparison(B);
            }
            else {
                value = value.CastToPrecision(B, targetType.GetUnderlyingType());
            }
        }
        else if (value.IsArray()) {
            value = value.CastToPrecision(B, targetType.GetUnderlyingType()->getArrayElementType());
        }
        else if (value.IsString() && targetType.IsPtr() &&
            targetType.GetUnderlyingType() == LLVMType<char>::get(B.getContext()))
        {
            llvm::StructType* strType = LLVMType<codegen::String>::get(B.getContext());
            llvm::Value* cstr = B.CreateStructGEP(strType, value.GetValue(), 0); // char**
            cstr = B.CreateLoad(strType->getTypeAtIndex(0u), cstr); // char*
            value = Value(cstr, LLVMType<char>::get(B.getContext()));
        }
    }

    return newargs;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

llvm::Function*
Function::create(llvm::LLVMContext& C, llvm::Module* M) const
{
    if (M)  {
        if (llvm::Function* function = M->getFunction(this->symbol())) {
            return function;
        }
    }

    ArgInfoVector parms;
    parms.reserve(this->size());
    ArgInfo ret = this->types(parms, C);
    llvm::FunctionType* type =
        llvm::FunctionType::get(ret.GetType(), parms.AsLLVMTypes(),
            false); // varargs

    llvm::Function* function =
        llvm::Function::Create(type,
            llvm::Function::ExternalLinkage,
            this->symbol(),
            M);

    if (!mNames.empty()) {
        // If some argument names have been specified, name the llvm values.
        // This provides a more reliable way for function to index into values
        // rather than relying on their position in the argument vector
        // @note  This does not guarantee that all arguments will have valid
        //   names
        for (llvm::Argument& arg : function->args()) {
            const char* name = this->argName(arg.getArgNo());
            if (name) arg.setName(name);
        }
    }

    function->setAttributes(this->flattenAttrs(function));
    return function;
}

llvm::Function* Function::get(const llvm::Module& M) const
{
    return M.getFunction(this->symbol());
}

llvm::Value*
Function::call(const std::vector<llvm::Value*>& args,
     llvm::IRBuilder<>& B) const
{
    llvm::BasicBlock* block = B.GetInsertBlock();
    OPENVDB_ASSERT(block);
    llvm::Function* currentFunction = block->getParent();
    OPENVDB_ASSERT(currentFunction);
    llvm::Module* M = currentFunction->getParent();
    OPENVDB_ASSERT(M);
    llvm::Function* function = this->create(B.getContext(), M);
    return B.CreateCall(function, args);
}

Value
Function::call(const Arguments& args,
     llvm::IRBuilder<>& B) const
{
    llvm::Value* result = Function::call(args.AsLLVMValues(), B);
    if (!result) return Value::Invalid();
    ArgInfoVector unused;
    auto ret = this->types(unused, B.getContext());
    return Value(result, ret.GetUnderlyingType());
}

Function::SignatureMatch
Function::match(const ArgInfoVector& args, llvm::LLVMContext& C) const
{
    if (args.size() != this->size()) return None;
    if (args.empty() && this->size() == 0) return Explicit;
    OPENVDB_ASSERT(!args.empty());
    ArgInfoVector types;
    this->types(types, C);
    if (args == types) return Explicit;
    // If types are not native, we can't implicitly cast
    for (const auto& arg : args) {
        if (!arg.IsNative()) return Size;
    }
    // See if we have an implicit cast option
    llvm::Type* strtype = LLVMType<codegen::String>::get(C);
    llvm::Type* ctype = LLVMType<char>::get(C);
    for (size_t i = 0; i < types.size(); ++i)
    {
        const ArgInfo& from = args[i];
        const ArgInfo& to = types[i];
        // if exactly matching, continue. not we only check the type metadata,
        // not anything else
        if (from.IsMatchingType(to)) continue;
        // if arg is a ptr and is not marked as readonly, fail - memory will be modified
        // @todo  save this typeinfo on the ArgInfo struct
        if (to.IsPtr() && !this->IsParamReadOnly(i)) return Size;
        // allow for string->char*. Note that this is only allowed from inputs->signature
        if (from.GetUnderlyingType() == strtype && to.IsPtr() &&
            (to.NumPtrs() == 1) && to.GetUnderlyingType() == ctype) continue;

        if (from.NumPtrs() != to.NumPtrs()) return Size;
        if (!isValidCast(from.GetUnderlyingType(), to.GetUnderlyingType())) return Size;
    }

    return Implicit;
}

void
Function::print(llvm::LLVMContext& C,
    std::ostream& os,
    const char* name,
    const bool axTypes) const
{
    ArgInfoVector current;
    ArgInfo ret = this->types(current, C);

    std::vector<const char*> names;
    names.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        names.emplace_back(this->argName(i));
    }
    printSignature(os, current, ret, name, names, axTypes);
}

llvm::AttributeList
Function::flattenAttrs(llvm::Function* F) const
{
    if (!mAttributes) return llvm::AttributeList();

    OPENVDB_ASSERT(F);
    llvm::LLVMContext& C = F->getContext();
    llvm::AttrBuilder AB(C);

    // @todo  allow for marking a function with readOnly instead, a weaker
    //   form of argMemOnly(Ref)
    //AB.addMemoryAttr(llvm::MemoryEffects::readOnly());

    for (auto& attr : mAttributes->mFnAttrs) AB.addAttribute(attr);
    if (mAttributes->mReadOnly) {
        AB.addMemoryAttr(llvm::MemoryEffects::argMemOnly(llvm::ModRefInfo::Ref));
    }
    // If you change this list, updated the FunctionBuilder::setBuiltin docs
    if (mAttributes->mType == Function::Type::kBuiltin) {
        AB.addAttribute(llvm::Attribute::NoFree);
        AB.addAttribute(llvm::Attribute::WillReturn);
        AB.addAttribute(llvm::Attribute::NoRecurse);
        AB.addAttribute(llvm::Attribute::NoUnwind);
        AB.addAttribute(llvm::Attribute::AlwaysInline);
    }
    else if (mAttributes->mType == Function::Type::kKernel) {
        AB.addAttribute(llvm::Attribute::NoRecurse);
        AB.addAttribute(llvm::Attribute::WillReturn);
    }
    const llvm::AttributeSet FnAttrs = llvm::AttributeSet::get(C, AB);

    AB.clear();

    for (auto& attr : mAttributes->mRetAttrs) AB.addAttribute(attr);
    const llvm::AttributeSet RetAttrs = llvm::AttributeSet::get(C, AB);

    SmallArgumentVector<llvm::AttributeSet> ParamAttrs;

    for (uint32_t i = 0; i < uint32_t(this->size()); ++i)
    {
        AB.clear();
        auto iter = mAttributes->mParamAttrs.find(i);
        if (iter != mAttributes->mParamAttrs.end()) {
            for (auto& attr : iter->second) AB.addAttribute(attr);
        }

        if (F->getArg(i)->getType()->isPointerTy())
        {
            if (mAttributes->mReadOnly) {
                // @note Technically not necessary to also mark params of the function
                //   as argMemOnly(Ref), but some optimizations passes may not infer it.
                //   Make the IR more robust/provide the clearest signal to llvm about
                //   mem accesses.
                AB.addAttribute(llvm::Attribute::ReadOnly);
            }
            // If you change this list, updated the FunctionBuilder::setBuiltin docs
            if (mAttributes->mType == Function::Type::kBuiltin) {
                // @todo mark attributes as dereferencable i.e:
                //     AB.addDereferenceableAttr()
                // @todo mark sret param[0] args as NoAlias - can't assume for
                //   other arguments e.g. dot(a,a);
                AB.addAttribute(llvm::Attribute::NonNull);
                AB.addAttribute(llvm::Attribute::NoUndef);
                AB.addAttribute(llvm::Attribute::NoFree);
#if LLVM_VERSION_MAJOR < 21
                AB.addAttribute(llvm::Attribute::NoCapture);
#else
                AB.addCapturesAttr(llvm::CaptureInfo::none());
#endif
            }
            else if (mAttributes->mType == Function::Type::kKernel) {
                AB.addAttribute(llvm::Attribute::NoFree);
                AB.addAttribute(llvm::Attribute::NoAlias);
                AB.addAttribute(llvm::Attribute::NoUndef);
#if LLVM_VERSION_MAJOR < 21
                AB.addAttribute(llvm::Attribute::NoCapture);
#else
                AB.addCapturesAttr(llvm::CaptureInfo::none());
#endif
            }
        }

        ParamAttrs.emplace_back(llvm::AttributeSet::get(C, AB));
    }

    return llvm::AttributeList::get(C, FnAttrs, RetAttrs, ParamAttrs);
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


llvm::Function*
IRFunctionBase::create(llvm::LLVMContext& C, llvm::Module* M) const
{
    if (this->hasEmbedIR()) return nullptr;

    llvm::Function* F = this->Function::create(C, M);
    OPENVDB_ASSERT(F);
    OPENVDB_ASSERT(F->arg_size() == this->size());

    // return if the function has already been generated or if no
    // module has been provided (just the function prototype requested)
    if (!F->empty() || !M) return F;

    // generate the body
    llvm::BasicBlock* BB =
        llvm::BasicBlock::Create(C,
            "entry_" + std::string(this->symbol()), F);

    Arguments args;
    ArgInfoVector types;
    this->types(types, C);
    OPENVDB_ASSERT(F->arg_size() == types.size());
    size_t i = 0;
    for (auto arg = F->arg_begin(), arg_end = F->arg_end();
         arg != arg_end; ++arg, ++i) {
        llvm::Value* val = llvm::cast<llvm::Value>(arg);
        args.AddArg(val, types[i]);
    }

    // create a new builder per function (its lightweight)
    // @todo could pass in the builder similar to Function::call
    llvm::IRBuilder<> B(BB);
    Value lastInstruction = mGen(args, B);

    // Allow the user to return a nullptr, an actual value or a return
    // instruction from the generator callback. This facilitates the same
    // generator being used for inline IR

    // if nullptr, insert a ret void inst, otherwise if it's not a return
    // instruction, either return the value if its supported or insert a
    // ret void
    if (!lastInstruction) {
        // @note  if the ret type is not expected to be void, this will
        //        cause verifyResultType to throw
        lastInstruction = Value::Return(B);
    }
    else if (!llvm::isa<llvm::ReturnInst>(lastInstruction.GetValue())) {
        OPENVDB_ASSERT(lastInstruction);
        if (lastInstruction.GetUnderlyingType()->isVoidTy()) {
            lastInstruction = Value::Return(B);
        }
        else {
            lastInstruction = Value::Return(B, &lastInstruction);
        }
    }
    OPENVDB_ASSERT(lastInstruction);
    OPENVDB_ASSERT(llvm::isa<llvm::ReturnInst>(lastInstruction.GetValue()));

    // pull out the ret type - is null if void
    llvm::Value* rvalue =
        llvm::cast<llvm::ReturnInst>
            (lastInstruction.GetValue())->getReturnValue();
    llvm::Type* type = rvalue ? rvalue->getType() :
        llvm::Type::getVoidTy(C);
    this->verifyResultType(type, F->getReturnType());
    return F;
}

llvm::Value* IRFunctionBase::call(const std::vector<llvm::Value*>& args,
     llvm::IRBuilder<>& B) const
{
    if (!this->hasEmbedIR()) {
        return this->Function::call(args, B);
    }

    Arguments newargs;

    ArgInfoVector types;
    this->types(types, B.getContext());
    for (size_t i = 0; i < args.size(); ++i) {
        newargs.AddArg(args[i], types[i]);
    }

    Value result = mGen(newargs, B);

    // only verify if result is not nullptr to
    // allow for embedded instructions
    if (result)
    {
        ArgInfoVector unused;
        this->verifyResultType(result.GetValue()->getType(),
            this->types(unused, B.getContext()).GetType());
    }
    return result.GetValue();
}

Value IRFunctionBase::call(const Arguments& args,
    llvm::IRBuilder<>& B) const
{
    if (!this->hasEmbedIR()) {
        return this->Function::call(args, B);
    }

    Value result = mGen(args, B);

    // only verify if result is not nullptr to
    // allow for embedded instructions
    if (result)
    {
        ArgInfoVector unused;
        this->verifyResultType(result.GetValue()->getType(),
            this->types(unused, B.getContext()).GetType());
    }
    return result;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

bool
FunctionGroup::HasUniqueFunctionSymbols() const
{
    std::set<std::string> symset;
    for (const auto& function : mFunctionList) {
        if (!symset.insert(function->symbol()).second) return false;
    }
    return true;
}

bool
FunctionGroup::HasUniqueTypeSignatures(llvm::LLVMContext& C) const
{
    std::vector<ArgInfoVector> typeset;
    for (const auto& function : mFunctionList)
    {
        ArgInfoVector types;
        function->types(types, C);
        if (std::find(typeset.begin(), typeset.end(), types) != typeset.end()) return false;
        typeset.emplace_back(types);
    }
    return true;
}

std::pair<const Function*, Function::SignatureMatch>
FunctionGroup::match(const ArgInfoVector& types, llvm::LLVMContext& C) const
{
    OPENVDB_ASSERT(this->HasUniqueTypeSignatures(C));
    std::pair<Function*, Function::SignatureMatch> result {
        nullptr, Function::SignatureMatch::None
    };

    for (const auto& function : mFunctionList) {

        const Function::SignatureMatch matchtype = function->match(types, C);
        OPENVDB_ASSERT(matchtype != Function::SignatureMatch::Ambiguous);
        result.second = std::max(matchtype, result.second);

        if (matchtype == Function::SignatureMatch::None)      continue;
        else if (matchtype == Function::SignatureMatch::Size) continue;
        else if (matchtype == Function::SignatureMatch::Explicit) {
            result.first = function.get();
            return result;
        }
        else if (matchtype == Function::SignatureMatch::Implicit) {
            if (result.first) {
                // we previously matched an implicit function
                result.second = Function::SignatureMatch::Ambiguous;
            }
            else {
                result.first = function.get();
            }
        }
    }

    return result;
}

Value FunctionGroup::execute(const Arguments& args, llvm::IRBuilder<>& B) const
{
    Value result = Value::Invalid();
    const auto match = this->match(args.GetArgInfo(), B.getContext());
    OPENVDB_ASSERT(match.second == Function::SignatureMatch::Explicit);
    if (match.second != Function::SignatureMatch::Explicit) return result;
    result = match.first->call(args, B);
    OPENVDB_ASSERT(result);
    return result;
}

Value FunctionGroup::execute(const NativeArguments& args, llvm::IRBuilder<>& B) const
{
    const auto match = this->match(Arguments(args).GetArgInfo(), B.getContext());
    if (!match.first) return Value::Invalid();
    Value result = match.first->call(args, B);
    OPENVDB_ASSERT(result);
    return result;
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

