// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

namespace {

inline void
printType(const llvm::Type* type, llvm::raw_os_ostream& stream, const bool axTypes)
{
    const ast::tokens::CoreType token =
        axTypes ? tokenFromLLVMType(type) : ast::tokens::UNKNOWN;
    if (token == ast::tokens::UNKNOWN) type->print(stream);
    else stream << ast::tokens::typeStringFromToken(token);
}

inline void
printTypes(llvm::raw_os_ostream& stream,
           const std::vector<llvm::Type*>& types,
           const std::vector<const char*>& names = {},
           const std::string sep = "; ",
           const bool axTypes = false)
{
    if (types.empty()) return;
    auto typeIter = types.cbegin();
    std::vector<const char*>::const_iterator nameIter;
    if (!names.empty()) nameIter = names.cbegin();

    for (; typeIter != types.cend() - 1; ++typeIter) {
        printType(*typeIter, stream, axTypes);
        if (!names.empty() && nameIter != names.cend()) {
            if (*nameIter && (*nameIter)[0] != '\0') {
                stream << ' ' << *nameIter;
            }
            ++nameIter;
        }
        stream << sep;
    }

    printType(*typeIter, stream, axTypes);
    if (!names.empty() && nameIter != names.cend()) {
        if (*nameIter && (*nameIter)[0] != '\0') {
            stream << ' ' << *nameIter;
        }
    }
}

}

void
printSignature(std::ostream& os,
               const std::vector<llvm::Type*>& signature,
               const llvm::Type* returnType,
               const char* name,
               const std::vector<const char*>& names,
               const bool axTypes)
{
    llvm::raw_os_ostream stream(os);

    printType(returnType, stream, axTypes);
    if (name && name[0] != '\0') {
        stream << " " << name;
    }
    stream << '(';
    printTypes(stream, signature, names, "; ", axTypes);
    stream << ')';
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

    std::vector<llvm::Type*> parms;
    parms.reserve(this->size());
    llvm::Type* ret = this->types(parms, C);

    llvm::FunctionType* type =
        llvm::FunctionType::get(ret, parms,
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

    function->setAttributes(this->flattenAttrs(C));
    return function;
}

llvm::Function* Function::get(const llvm::Module& M) const
{
    return M.getFunction(this->symbol());
}

llvm::Value*
Function::call(const std::vector<llvm::Value*>& args,
     llvm::IRBuilder<>& B,
     const bool cast) const
{
    llvm::BasicBlock* block = B.GetInsertBlock();
    OPENVDB_ASSERT(block);
    llvm::Function* currentFunction = block->getParent();
    OPENVDB_ASSERT(currentFunction);
    llvm::Module* M = currentFunction->getParent();
    OPENVDB_ASSERT(M);
    llvm::Function* function = this->create(B.getContext(), M);
    std::vector<llvm::Value*> inputs(args);
    if (cast) {
        std::vector<llvm::Type*> types;
        this->types(types, B.getContext());
        this->cast(inputs, types, B);
    }
    return B.CreateCall(function, inputs);
}

Function::SignatureMatch
Function::match(const std::vector<llvm::Type*>& inputs, llvm::LLVMContext& C) const
{
    // these checks mean we can design the match function signature to not
    // require the llvm context and instead pull it out of the type vector
    // which is guaranteed to not be empty
    if (inputs.size() != this->size()) return None;
    if (inputs.empty() && this->size() == 0) return Explicit;

    OPENVDB_ASSERT(!inputs.empty());
    //llvm::LLVMContext& C = inputs.front()->getContext();

    std::vector<llvm::Type*> signature;
    this->types(signature, C);
    if (inputs == signature) return Explicit;

    llvm::Type* strType = LLVMType<codegen::String>::get(C);

    // try implicit - signature should not be empty here
    for (size_t i = 0; i < signature.size(); ++i) {
        llvm::Type* from = inputs[i];
        llvm::Type* to = signature[i];
        // if exactly matching, continue
        if (from == to) continue;

        // if arg is a ptr and is not marked as readonly, fail - memory will be modified
        if (to->isPointerTy() && !this->hasParamAttribute(i,
                llvm::Attribute::AttrKind::ReadOnly)) return Size;

        // compare contained types if both are pointers
        if (from->isPointerTy() && to->isPointerTy()) {
            from = from->getContainedType(0);
            to = to->getContainedType(0);
        }

        // allow for string->char*. Note that this is only allowed from inputs->signature
        if (from == strType && to == LLVMType<char>::get(C)) continue;
        if (!isValidCast(from, to)) return Size;
    }

    return Implicit;
}

void
Function::print(llvm::LLVMContext& C,
    std::ostream& os,
    const char* name,
    const bool axTypes) const
{
    std::vector<llvm::Type*> current;
    llvm::Type* ret = this->types(current, C);

    std::vector<const char*> names;
    names.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        names.emplace_back(this->argName(i));
    }
    printSignature(os, current, ret, name, names, axTypes);
}

void
Function::cast(std::vector<llvm::Value*>& args,
            const std::vector<llvm::Type*>& types,
            llvm::IRBuilder<>& B)
{
    llvm::LLVMContext& C = B.getContext();
    for (size_t i = 0; i < args.size(); ++i) {
        if (i >= types.size()) break;
        llvm::Value*& value = args[i];
        llvm::Type* type = value->getType();
        if (type->isIntegerTy() || type->isFloatingPointTy()) {
            if (types[i]->isIntegerTy(1)) {
                // assume boolean target value
                value = boolComparison(value, B);
            }
            else {
                value = arithmeticConversion(value, types[i], B);
            }
        }
        else if (type->getContainedType(0)->isArrayTy()) {
            llvm::Type* arrayType = getBaseContainedType(types[i]);
            value = arrayCast(value, arrayType->getArrayElementType(), B);
        }
        else {
            if (types[i] == LLVMType<char*>::get(C)) {
                llvm::Type* strType = LLVMType<codegen::String>::get(C);
                if (type->getContainedType(0) == strType) {
                    value = B.CreateStructGEP(strType, value, 0); // char**
                    value = ir_load(B, value); // char*
                }
            }
        }
    }
}

llvm::AttributeList
Function::flattenAttrs(llvm::LLVMContext& C) const
{
    if (!mAttributes) return llvm::AttributeList();

    auto buildSetFromKinds = [&C](llvm::AttrBuilder& ab,
        const std::vector<llvm::Attribute::AttrKind>& kinds)
            -> llvm::AttributeSet {
        for (auto& attr : kinds) {
            ab.addAttribute(attr);
        }
        const llvm::AttributeSet set = llvm::AttributeSet::get(C, ab);
        ab.clear();
        return set;
    };

#if LLVM_VERSION_MAJOR <= 13
    llvm::AttrBuilder ab;
#else
    llvm::AttrBuilder ab(C);
#endif

    const llvm::AttributeSet fn = buildSetFromKinds(ab, mAttributes->mFnAttrs);
    const llvm::AttributeSet ret = buildSetFromKinds(ab, mAttributes->mRetAttrs);

    std::vector<llvm::AttributeSet> parms(this->size());

    for (auto& idxAttr : mAttributes->mParamAttrs) {
        const size_t idx = idxAttr.first;
        if (idx >= this->size()) continue;
        parms[idx] = buildSetFromKinds(ab, idxAttr.second);
    }

    return llvm::AttributeList::get(C, fn, ret, parms);
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


llvm::Function*
IRFunctionBase::create(llvm::LLVMContext& C, llvm::Module* M) const
{
    if (this->hasEmbedIR()) return nullptr;

    llvm::Function* F = this->Function::create(C, M);
    OPENVDB_ASSERT(F);
    // return if the function has already been generated or if no
    // module has been provided (just the function prototype requested)
    if (!F->empty() || !M) return F;

    // generate the body
    llvm::BasicBlock* BB =
        llvm::BasicBlock::Create(C,
            "entry_" + std::string(this->symbol()), F);

    std::vector<llvm::Value*> fnargs;
    fnargs.reserve(this->size());
    for (auto arg = F->arg_begin(), arg_end = F->arg_end();
         arg != arg_end; ++arg) {
        fnargs.emplace_back(llvm::cast<llvm::Value>(arg));
    }

    // create a new builder per function (its lightweight)
    // @todo could pass in the builder similar to Function::call
    llvm::IRBuilder<> B(BB);
    llvm::Value* lastInstruction = mGen(fnargs, B);

    // Allow the user to return a nullptr, an actual value or a return
    // instruction from the generator callback. This facilitates the same
    // generator being used for inline IR

    // if nullptr, insert a ret void inst, otherwise if it's not a return
    // instruction, either return the value if its supported or insert a
    // ret void
    if (!lastInstruction) {
        // @note  if the ret type is not expected to be void, this will
        //        cause verifyResultType to throw
        lastInstruction = B.CreateRetVoid();
    }
    else if (!llvm::isa<llvm::ReturnInst>(lastInstruction)) {
        OPENVDB_ASSERT(lastInstruction);
        if (lastInstruction->getType()->isVoidTy()) {
            lastInstruction = B.CreateRetVoid();
        }
        else {
            lastInstruction = B.CreateRet(lastInstruction);
        }
    }
    OPENVDB_ASSERT(lastInstruction);
    OPENVDB_ASSERT(llvm::isa<llvm::ReturnInst>(lastInstruction));

    // pull out the ret type - is null if void
    llvm::Value* rvalue =
        llvm::cast<llvm::ReturnInst>
            (lastInstruction)->getReturnValue();
    llvm::Type* type = rvalue ? rvalue->getType() :
        llvm::Type::getVoidTy(C);

    this->verifyResultType(type, F->getReturnType());
    return F;
}

llvm::Value* IRFunctionBase::call(const std::vector<llvm::Value*>& args,
     llvm::IRBuilder<>& B,
     const bool cast) const
{
    if (!this->hasEmbedIR()) {
        return this->Function::call(args, B, cast);
    }

    std::vector<llvm::Value*> inputs(args);
    if (cast) {
        std::vector<llvm::Type*> types;
        this->types(types, B.getContext());
        this->cast(inputs, types, B);
    }

    llvm::Value* result = mGen(inputs, B);
    if (result) {
        // only verify if result is not nullptr to
        // allow for embedded instructions
        std::vector<llvm::Type*> unused;
        this->verifyResultType(result->getType(),
            this->types(unused, B.getContext()));
    }
    return result;
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


const Function*
FunctionGroup::match(const std::vector<llvm::Type*>& types,
      llvm::LLVMContext& C,
      Function::SignatureMatch* type) const
{
    const Function* targetFunction = nullptr;
    if (type) *type = Function::SignatureMatch::None;

    for (const auto& function : mFunctionList) {

        const Function::SignatureMatch matchtype = function->match(types, C);
        if (type) *type = std::max(matchtype, *type);

        if (matchtype == Function::SignatureMatch::None)      continue;
        else if (matchtype == Function::SignatureMatch::Size) continue;
        else if (matchtype == Function::SignatureMatch::Explicit) {
            return function.get();
        }
        else if (matchtype == Function::SignatureMatch::Implicit) {
            if (!targetFunction) targetFunction = function.get();
        }
    }

    return targetFunction;
}

llvm::Value*
FunctionGroup::execute(const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B) const
{
    std::vector<llvm::Type*> inputTypes;
    valuesToTypes(args, inputTypes);

    Function::SignatureMatch match;
    const Function* target = this->match(inputTypes, B.getContext(), &match);
    OPENVDB_ASSERT(target);
    llvm::Value* result =
        target->call(args, B, /*cast=*/match == Function::SignatureMatch::Implicit);

#ifndef NDEBUG
    std::vector<llvm::Type*> unused;
    llvm::Type* ret = target->types(unused, B.getContext());
    OPENVDB_ASSERT(result || ret->isVoidTy());
#endif
    return result;
}

const Function*
FunctionGroup::execute(const std::vector<llvm::Value*>& args,
            llvm::IRBuilder<>& B,
            llvm::Value*& result) const
{
    std::vector<llvm::Type*> inputTypes;
    valuesToTypes(args, inputTypes);

    Function::SignatureMatch match;
    const Function* target = this->match(inputTypes, B.getContext(), &match);
    if (!target) return nullptr;

    result = target->call(args, B, /*cast=*/match == Function::SignatureMatch::Implicit);

#ifndef NDEBUG
    std::vector<llvm::Type*> unused;
    llvm::Type* ret = target->types(unused, B.getContext());
    OPENVDB_ASSERT(result || ret->isVoidTy());
#endif

    return target;
}

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

