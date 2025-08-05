// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/// @file codegen/ComputeGenerator.cc

#include "ComputeGenerator.h"
#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "Types.h"
#include "Utils.h"

#include "../ast/AST.h"
#include "../ast/Tokens.h"
#include "../compiler/CustomData.h"
#include "../Exceptions.h"

#include <openvdb/util/Assert.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Pass.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/BuildLibCalls.h>

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
           const ArgInfoVector& types,
           const std::vector<const char*>& names = {},
           const std::string sep = "; ",
           const bool axTypes = true)
{
    if (types.empty()) return;
    auto typeIter = types.begin();
    std::vector<const char*>::const_iterator nameIter;
    if (!names.empty()) nameIter = names.cbegin();

    for (; typeIter != types.end() - 1; ++typeIter) {
        printType(typeIter->GetUnderlyingType(), stream, axTypes);
        if (!names.empty() && nameIter != names.cend()) {
            if (*nameIter && (*nameIter)[0] != '\0') {
                stream << ' ' << *nameIter;
            }
            ++nameIter;
        }
        stream << sep;
    }

    printType(typeIter->GetUnderlyingType(), stream, axTypes);
    if (!names.empty() && nameIter != names.cend()) {
        if (*nameIter && (*nameIter)[0] != '\0') {
            stream << ' ' << *nameIter;
        }
    }
}

}

const std::array<std::string, ComputeKernel::N_ARGS>&
ComputeKernel::getArgumentKeys()
{
    static const std::array<std::string, ComputeKernel::N_ARGS> arguments = {
        { "custom_data" }
    };

    return arguments;
}

std::string ComputeKernel::getDefaultName() { return "ax.compute"; }


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

ComputeGenerator::ComputeGenerator(llvm::Module& module,
                                   const FunctionOptions& options,
                                   FunctionRegistry& functionRegistry,
                                   Logger& logger)
    : mModule(module)
    , mContext(module.getContext())
    , mBuilder(module.getContext())
    , mValues()
    , mBreakContinueStack()
    , mScopeIndex(1) // 0 represents globals
    , mSymbolTables()
    , mFunction(nullptr)
    , mOptions(options)
    , mLog(logger)
    , mFunctionRegistry(functionRegistry)
{
    codegen::internal::InsertMappedFunctionRegistry(&mContext, &mFunctionRegistry, mOptions);
}

ComputeGenerator::~ComputeGenerator()
{
    codegen::internal::RemoveMappedFunctionRegistry(&mContext);
}

bool ComputeGenerator::generate(const ast::Tree& tree)
{
    llvm::FunctionType* type =
        llvmFunctionTypeFromSignature<ComputeKernel::Signature>(mContext);

    mFunction = llvm::Function::Create(type,
        llvm::Function::ExternalLinkage,
        ComputeKernel::getDefaultName(),
        &mModule);

    // Set up arguments for initial entry

    llvm::Function::arg_iterator argIter = mFunction->arg_begin();
    const auto arguments = ComputeKernel::getArgumentKeys();
    auto keyIter = arguments.cbegin();

    for (; argIter != mFunction->arg_end(); ++argIter, ++keyIter) {
        argIter->setName(*keyIter);
    }

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(mContext,
        "entry_" + ComputeKernel::getDefaultName(), mFunction);
    mBuilder.SetInsertPoint(entry);

    // if traverse is false, log should have error, but can error
    // without stopping traversal, so check both
    const size_t err = mLog.errors();
    if (!this->traverse(&tree) || (mLog.errors() > err)) return false;

    // free strings at terminating blocks

    this->createFreeSymbolStrings(mBuilder);

    return true;
}

bool ComputeGenerator::visit(const ast::Block* block)
{
    mScopeIndex++;

    // traverse the contents of the block
    const size_t children = block->children();

    for (size_t i = 0; i < children; ++i) {
        if (!this->traverse(block->child(i)) && mLog.atErrorLimit()) {
            return false;
        }
        // reset the value stack for each statement
        mValues = std::stack<Value>();
    }

    mSymbolTables.erase(mScopeIndex);
    mScopeIndex--;

    OPENVDB_ASSERT(mScopeIndex > 0);
    return true;
}

bool ComputeGenerator::visit(const ast::CommaOperator* comma)
{
    // traverse the contents of the comma expression
    const size_t children = comma->children();
    Value value = Value::Invalid();
    bool hasErrored = false;
    for (size_t i = 0; i < children; ++i) {
        if (this->traverse(comma->child(i))) {
            value = mValues.top(); mValues.pop();
        }
        else {
            if (mLog.atErrorLimit()) return false;
            hasErrored = true;
        }
    }
    // only keep the last value
    if (!value || hasErrored) return false;
    mValues.push(value);
    return true;
}

bool ComputeGenerator::visit(const ast::ConditionalStatement* cond)
{
    llvm::BasicBlock* postIfBlock = llvm::BasicBlock::Create(mContext, "block", mFunction);
    llvm::BasicBlock* thenBlock = llvm::BasicBlock::Create(mContext, "then", mFunction);
    const bool hasElse = cond->hasFalse();
    llvm::BasicBlock* elseBlock = hasElse ? llvm::BasicBlock::Create(mContext, "else", mFunction) : postIfBlock;

    // generate conditional
    if (this->traverse(cond->condition())) {
        Value condition = mValues.top(); mValues.pop();
        // check the type of the condition branch is bool-convertable
        if (condition.IsScalar()) {
            condition = condition.ScalarBoolComparison(mBuilder);
            mBuilder.CreateCondBr(condition.GetValue(), thenBlock, elseBlock);
        } else {
            if (!mLog.error("cannot convert non-scalar type to bool in condition", cond->condition())) return false;
        }
    } else if (mLog.atErrorLimit()) return false;

    // generate if-then branch
    mBuilder.SetInsertPoint(thenBlock);
    if (!this->traverse(cond->trueBranch()) && mLog.atErrorLimit()) return false;
    mBuilder.CreateBr(postIfBlock);

    if (hasElse) {
        // generate else-then branch
        mBuilder.SetInsertPoint(elseBlock);
        if (!this->traverse(cond->falseBranch()) && mLog.atErrorLimit()) return false;
        mBuilder.CreateBr(postIfBlock);
    }

    // reset to continue block
    mBuilder.SetInsertPoint(postIfBlock);

    // reset the value stack
    mValues = std::stack<Value>();

    return true;
}

bool ComputeGenerator::visit(const ast::TernaryOperator* tern)
{
    llvm::BasicBlock* trueBlock = llvm::BasicBlock::Create(mContext, "ternary_true", mFunction);
    llvm::BasicBlock* falseBlock = llvm::BasicBlock::Create(mContext, "ternary_false", mFunction);
    llvm::BasicBlock* returnBlock = llvm::BasicBlock::Create(mContext, "ternary_return", mFunction);

    Value trueValue = Value::Invalid();
    Value falseValue = Value::Invalid();

    // generate conditional
    bool conditionSuccess = this->traverse(tern->condition());
    if (conditionSuccess) {
        // get the condition
        trueValue = mValues.top(); mValues.pop();
        OPENVDB_ASSERT(trueValue);
        // check the type of the condition branch is bool-convertable
        if (trueValue.IsScalar()) {
            Value comp = trueValue.ScalarBoolComparison(mBuilder);
            mBuilder.CreateCondBr(comp.GetValue(), trueBlock, falseBlock);
        }
        else {
            if (!mLog.error("cannot convert non-scalar type to bool in condition", tern->condition())) return false;
            conditionSuccess = false;
        }
    }
    else if (mLog.atErrorLimit()) return false;

    // generate true branch, if it exists otherwise take condition as true value

    mBuilder.SetInsertPoint(trueBlock);
    bool trueSuccess = conditionSuccess;
    if (tern->hasTrue()) {
        trueSuccess = this->traverse(tern->trueBranch());
        if (trueSuccess) {
            trueValue = mValues.top(); mValues.pop();// get true value from true expression
        }
        else if (mLog.atErrorLimit()) return false;
    }
    llvm::BranchInst* trueBranch = mBuilder.CreateBr(returnBlock); // trueblock -> return block

    // generate false branch

    mBuilder.SetInsertPoint(falseBlock);
    bool falseSuccess = this->traverse(tern->falseBranch());
    // even if the condition isnt successful but the others are, we continue to code gen to find type errors in branches
    if (!(trueSuccess && falseSuccess)) return false;

    llvm::BranchInst* falseBranch = mBuilder.CreateBr(returnBlock); // falseblock -> return block

    falseValue = mValues.top(); mValues.pop();
    OPENVDB_ASSERT(trueValue);
    OPENVDB_ASSERT(falseValue);

    if (trueValue.IsVoid() && falseValue.IsVoid()) {
        // Thsi is possible in statements like "true ? print(1) : print(2);"
        // jsut push void value to stop use of return from this expression
        mBuilder.SetInsertPoint(returnBlock);
        mValues.push(falseValue);
        return conditionSuccess && trueSuccess && falseSuccess;
    }

    if (trueValue.GetUnderlyingType() != falseValue.GetUnderlyingType())
    {
        if ((trueValue.IsScalar() || trueValue.IsArray()) &&
            (falseValue.IsScalar() || falseValue.IsArray()))
        {
            // Scalar precision of the return type
            llvm::Type* precision = typePrecedence(trueValue.GetUnderlyingScalarType(), falseValue.GetUnderlyingScalarType());

            if (trueValue.IsScalar())
            {
                mBuilder.SetInsertPoint(trueBranch);
                trueValue = trueValue.CastToPrecision(mBuilder, precision);
            }
            if (falseValue.IsScalar())
            {
                mBuilder.SetInsertPoint(falseBranch);
                falseValue = falseValue.CastToPrecision(mBuilder, precision);
            }

            // If Scalar -> Array, promote the scalar side
            if (trueValue.IsScalar() && falseValue.IsArray())
            {
                mBuilder.SetInsertPoint(trueBranch);
                const size_t arraySize = falseValue.GetArrayNumElements();
                if (arraySize == 9)       trueValue = trueValue.ScalarToIdentMatrix3(mBuilder);
                else if (arraySize == 16) trueValue = trueValue.ScalarToIdentMatrix4(mBuilder);
                else                      trueValue = trueValue.ScalarToArray(mBuilder, arraySize);
            }
            else if (falseValue.IsScalar() && trueValue.IsArray())
            {
                mBuilder.SetInsertPoint(falseBranch);
                const size_t arraySize = trueValue.GetArrayNumElements();
                if (arraySize == 9)       falseValue = falseValue.ScalarToIdentMatrix3(mBuilder);
                else if (arraySize == 16) falseValue = falseValue.ScalarToIdentMatrix4(mBuilder);
                else                      falseValue = falseValue.ScalarToArray(mBuilder, arraySize);
            }

            // If Array -> Array, cast
            if (falseValue.IsArray() && trueValue.IsArray())
            {
                mBuilder.SetInsertPoint(trueBranch);
                trueValue = trueValue.CastToPrecision(mBuilder, precision);
                mBuilder.SetInsertPoint(falseBranch);
                falseValue = falseValue.CastToPrecision(mBuilder, precision);
            }
        }

        // Otherwise unsupported ternary
        if (trueValue.GetUnderlyingType() != falseValue.GetUnderlyingType())
        {
            mLog.error("unsupported implicit cast in ternary operation",
                       tern->hasTrue() ? tern->trueBranch() : tern->falseBranch());
            return false;
        }
    }

    // Load scalars for ternary results
    if (trueValue.IsScalar())
    {
        mBuilder.SetInsertPoint(trueBranch);
        trueValue = trueValue.LoadIfPtr(mBuilder);
    }
    if (falseValue.IsScalar())
    {
        mBuilder.SetInsertPoint(falseBranch);
        falseValue = falseValue.LoadIfPtr(mBuilder);
    }

    // reset to continue block
    mBuilder.SetInsertPoint(returnBlock);
    llvm::PHINode* ternary = mBuilder.CreatePHI(trueValue.GetValue()->getType(), 2, "ternary");

    // if nesting branches the blocks for true and false branches may have been updated
    // so get these again rather than reusing trueBlock/falseBlock
    ternary->addIncoming(trueValue.GetValue(), trueBranch->getParent());
    ternary->addIncoming(falseValue.GetValue(), falseBranch->getParent());

    mValues.push(Value(ternary, trueValue.GetUnderlyingType()));
    return conditionSuccess && trueSuccess && falseSuccess;
}

bool ComputeGenerator::visit(const ast::Loop* loop)
{
    OPENVDB_ASSERT(mScopeIndex > 0);
    mScopeIndex++;

    llvm::BasicBlock* postLoopBlock = llvm::BasicBlock::Create(mContext, "block", mFunction);
    llvm::BasicBlock* conditionBlock = llvm::BasicBlock::Create(mContext, "loop_condition", mFunction);
    llvm::BasicBlock* bodyBlock = llvm::BasicBlock::Create(mContext, "loop_body", mFunction);

    llvm::BasicBlock* postBodyBlock = conditionBlock;

    const ast::tokens::LoopToken loopType = loop->loopType();
    OPENVDB_ASSERT((loopType == ast::tokens::LoopToken::FOR ||
            loopType == ast::tokens::LoopToken::WHILE ||
            loopType == ast::tokens::LoopToken::DO) &&
            "Unsupported loop type");

    if (loopType == ast::tokens::LoopToken::FOR) {
        // init -> condition -> body -> iter -> condition ... continue

        // generate initial statement
        if (loop->hasInit()) {
            if (!this->traverse(loop->initial()) && mLog.atErrorLimit()) return false;
            // reset the value stack
            mValues = std::stack<Value>();
        }
        mBuilder.CreateBr(conditionBlock);

        // generate iteration
        if (loop->hasIter()) {
            llvm::BasicBlock* iterBlock = llvm::BasicBlock::Create(mContext, "loop_iteration", mFunction);
            postBodyBlock = iterBlock;

            mBuilder.SetInsertPoint(iterBlock);
            if (!this->traverse(loop->iteration()) && mLog.atErrorLimit()) return false;
            mBuilder.CreateBr(conditionBlock);
        }
    }
    else if (loopType == ast::tokens::LoopToken::DO)  {
        //  body -> condition -> body -> condition ... continue
        mBuilder.CreateBr(bodyBlock);
    }
    else if (loopType == ast::tokens::LoopToken::WHILE) {
        //  condition -> body -> condition ... continue
        mBuilder.CreateBr(conditionBlock);
    }

    // store the destinations for break and continue
    mBreakContinueStack.push({postLoopBlock, postBodyBlock});

    // generate loop body
    mBuilder.SetInsertPoint(bodyBlock);
    if (!this->traverse(loop->body()) && mLog.atErrorLimit()) return false;
    mBuilder.CreateBr(postBodyBlock);

    // generate condition
    mBuilder.SetInsertPoint(conditionBlock);
    if (this->traverse(loop->condition()))
    {
        Value condition = mValues.top(); mValues.pop();
        // check the type of the condition branch is bool-convertable
        if (condition.IsScalar()) {
            condition = condition.ScalarBoolComparison(mBuilder);
            mBuilder.CreateCondBr(condition.GetValue(), bodyBlock, postLoopBlock);
        }
        else {
            if (!mLog.error("cannot convert non-scalar type to bool in condition", loop->condition())) return false;
        }
        // reset the value stack
        mValues = std::stack<Value>();
    }
    else if (mLog.atErrorLimit()) return false;

    // reset to post loop block
    mBuilder.SetInsertPoint(postLoopBlock);

    // discard break and continue
    mBreakContinueStack.pop();

    // remove the symbol table created in this scope
    mSymbolTables.erase(mScopeIndex);
    mScopeIndex--;
    OPENVDB_ASSERT(mScopeIndex > 0);

    // reset the value stack
    mValues = std::stack<Value>();

    return true;
}

bool ComputeGenerator::visit(const ast::Keyword* node)
{
    const ast::tokens::KeywordToken keyw = node->keyword();
    OPENVDB_ASSERT((keyw == ast::tokens::KeywordToken::RETURN ||
            keyw == ast::tokens::KeywordToken::BREAK  ||
            keyw == ast::tokens::KeywordToken::CONTINUE) &&
            "Unsupported keyword");

    if (keyw == ast::tokens::KeywordToken::RETURN) {
        mBuilder.CreateRetVoid();
    }
    else if (keyw == ast::tokens::KeywordToken::BREAK ||
             keyw == ast::tokens::KeywordToken::CONTINUE) {
        // find the parent loop, if it exists
        const ast::Node* child = node;
        const ast::Node* parentLoop = node->parent();
        while (parentLoop) {
            if (parentLoop->nodetype() == ast::Node::NodeType::LoopNode) {
                break;
            }
            child = parentLoop;
            parentLoop = child->parent();
        }
        if (!parentLoop) {
            if (!mLog.error("keyword \"" + ast::tokens::keywordNameFromToken(keyw)
                    + "\" used outside of loop.", node)) return false;
        }
        else {
            const std::pair<llvm::BasicBlock*, llvm::BasicBlock*>
                breakContinue = mBreakContinueStack.top();

            if (keyw == ast::tokens::KeywordToken::BREAK) {
                OPENVDB_ASSERT(breakContinue.first);
                mBuilder.CreateBr(breakContinue.first);
            }
            else if (keyw == ast::tokens::KeywordToken::CONTINUE) {
                OPENVDB_ASSERT(breakContinue.second);
                mBuilder.CreateBr(breakContinue.second);
            }
        }
    }

    llvm::BasicBlock* nullBlock = llvm::BasicBlock::Create(mContext, "null", mFunction);
    // insert all remaining instructions in scope into a null block
    // this will incorporate all instructions that follow until new insert point is set
    mBuilder.SetInsertPoint(nullBlock);
    return true;
}

bool ComputeGenerator::visit(const ast::BinaryOperator* node)
{
    const ast::tokens::OperatorToken op = node->operation();
    // if AND or OR, need to handle short-circuiting
    if (op == ast::tokens::OperatorToken::AND ||
        op == ast::tokens::OperatorToken::OR)
    {
        llvm::BranchInst* lhsBranch = nullptr;
        llvm::BasicBlock* rhsBlock = llvm::BasicBlock::Create(mContext, "binary_rhs", mFunction);
        llvm::BasicBlock* returnBlock = llvm::BasicBlock::Create(mContext, "binary_return", mFunction);

        Value lhs = Value::Invalid();
        bool lhsSuccess = this->traverse(node->lhs());
        if (lhsSuccess)
        {
            lhs = mValues.top(); mValues.pop();
            if (lhs.IsScalar()) {
                lhs = lhs.ScalarBoolComparison(mBuilder);
                if (op == ast::tokens::OperatorToken::AND) {
                    lhsBranch = mBuilder.CreateCondBr(lhs.GetValue(), rhsBlock, returnBlock);
                }
                else {
                    lhsBranch = mBuilder.CreateCondBr(lhs.GetValue(), returnBlock, rhsBlock);
                }
            }
            else {
                mLog.error("cannot convert non-scalar lhs to bool", node->lhs());
                lhsSuccess = false;
            }
        }

        if (mLog.atErrorLimit()) return false;

        mBuilder.SetInsertPoint(rhsBlock);
        bool rhsSuccess = this->traverse(node->rhs());
        if (rhsSuccess) {
            Value rhs = mValues.top(); mValues.pop();
            if (rhs.IsScalar())
            {
                rhs = rhs.ScalarBoolComparison(mBuilder);
                llvm::BranchInst* rhsBranch = mBuilder.CreateBr(returnBlock);

                mBuilder.SetInsertPoint(returnBlock);
                if (lhsBranch) {// i.e. lhs was successful
                    OPENVDB_ASSERT(rhs);
                    OPENVDB_ASSERT(lhs);
                    llvm::PHINode* result = mBuilder.CreatePHI(LLVMType<bool>::get(mContext), 2, "binary_op");
                    result->addIncoming(lhs.GetValue(), lhsBranch->getParent());
                    result->addIncoming(rhs.GetValue(), rhsBranch->getParent());
                    mValues.push(Value(result, LLVMType<bool>::get(mContext)));
                }
            }
            else {
                mLog.error("cannot convert non-scalar rhs to bool", node->rhs());
                rhsSuccess = false;
            }
        }
        return lhsSuccess && rhsSuccess;
    }
    else
    {
        Value lhs = Value::Invalid();
        Value rhs = Value::Invalid();
        if (this->traverse(node->lhs())) {
            lhs = mValues.top(); mValues.pop();
        }
        else if (mLog.atErrorLimit()) return false;

        if (this->traverse(node->rhs())) {
            rhs = mValues.top(); mValues.pop();
        }
        else if (mLog.atErrorLimit()) return false;

        if (!(lhs && rhs)) return false;

        Value result = lhs.Binary(mBuilder, rhs, op, &mLog, node);
        if (!result) return false;
        mValues.push(result);
    }

    return true;
}

bool ComputeGenerator::visit(const ast::UnaryOperator* node)
{
    // If the unary operation is a +, keep the value ptr on the stack and
    // continue (avoid any new allocations or unecessary loads)

    const ast::tokens::OperatorToken op = node->operation();
    if (op == ast::tokens::PLUS) return true;

    if (op != ast::tokens::MINUS &&
        op != ast::tokens::BITNOT &&
        op != ast::tokens::NOT) {
        mLog.error("unrecognised unary operator \"" +
                ast::tokens::operatorNameFromToken(op) + "\"", node);
        return false;
    }

    // unary operator uses default traversal so value should be on the stack
    Value top = mValues.top();
    if (op == ast::tokens::NOT)         top = top.Not(mBuilder, &mLog, node);
    else if (op == ast::tokens::MINUS)  top = top.Negate(mBuilder, &mLog, node);
    else if (op == ast::tokens::BITNOT) top = top.BitNot(mBuilder, &mLog, node);
    if (!top) return false;
    mValues.pop();
    mValues.push(top);
    return true;
}

bool ComputeGenerator::visit(const ast::AssignExpression* node)
{
    // default traversal, should have rhs and lhs on stack
    // leave LHS on stack
    Value rhs = mValues.top(); mValues.pop();
    Value& lhs = mValues.top();

    if (node->isCompound()) {
        rhs = lhs.Binary(mBuilder, rhs, node->operation(), &mLog, node);
        if (!rhs) return false;
    }

    lhs = lhs.Assign(mBuilder, rhs, &mLog, node);
    if (!lhs) return false;
    return true;
}

bool ComputeGenerator::visit(const ast::Crement* node)
{
    Value value = mValues.top(); mValues.pop();
    if (!value.IsPtr()) {
        mLog.error("Cannot assign an increment or decrement to a literal value", node);
        return false;
    }

    Value crement = Value::Invalid();
    OPENVDB_ASSERT((node->increment() || node->decrement()) && "unrecognised crement operation");
    if (node->increment())      crement = value.Increment(mBuilder, &mLog, node);
    else if (node->decrement()) crement = value.Decrement(mBuilder, &mLog, node);
    if (!crement) return false;

    // Load the original value for post ops before the new store
    Value original = Value::Invalid();
    if (node->post()) {
        original = value.LoadIfPtr(mBuilder);
    }

    OPENVDB_ASSERT(!crement.IsPtr());
    OPENVDB_ASSERT(value.IsPtr());
    mBuilder.CreateStore(crement.GetValue(), value.GetValue());

    // decide what to put on the expression stack
    if (node->pre()) {
        mValues.push(value);
    }
    else {
        OPENVDB_ASSERT(node->post());
        OPENVDB_ASSERT(original);
        mValues.push(original);
    }
    return true;
}

bool ComputeGenerator::visit(const ast::FunctionCall* node)
{
    const FunctionGroup* const function =
        mFunctionRegistry.getOrInsert(node->name(), mOptions, false);
    if (!function) {
        mLog.error("unable to locate function \"" + node->name() + "\"", node);
        return false;
    }

    const size_t args = node->children();
    OPENVDB_ASSERT(mValues.size() >= args);

    // initialize arguments. scalars are always passed by value, arrays
    // and strings always by pointer

    std::vector<Value> vals;
    vals.resize(args, Value::Invalid());

    // Explicitly load scalar ptrs
    for (auto r = vals.rbegin(); r != vals.rend(); ++r) {
        Value arg = mValues.top(); mValues.pop();
        *r = (arg.IsPtr() && arg.IsScalar()) ? arg.Load(mBuilder) : arg;
    }

    NativeArguments nativeargs(vals);
    Arguments genericargs(nativeargs);
    auto arginfo = genericargs.GetArgInfo();

    const auto result = function->match(arginfo, mContext);

    if (!result.first) {
        OPENVDB_ASSERT(!function->list().empty()
               && "FunctionGroup has no function declarations");

        std::ostringstream os;
        if (result.second == Function::SignatureMatch::None) {
            os << "wrong number of arguments. \"" << node->name() << "\""
               << " was called with: (";
            llvm::raw_os_ostream stream(os);
            printTypes(stream, arginfo);
            stream << ")";
        }
        else {
            // match == Function::SignatureMatch::Size
            os << "no matching function for ";
            printSignature(os, arginfo,
                ArgInfo(LLVMType<void>::get(mContext), 0),
                node->name().c_str(), {}, true);
        }

        os << " \ncandidates are: ";
        for (const auto& sig : function->list()) {
            os << std::endl;
            sig->print(mContext, os, node->name().c_str());
        }
        mLog.error(os.str(), node);
        mValues.push(Value::Invalid()); // push invalid
        return false;
    }

    if (result.second == Function::SignatureMatch::Implicit ||
        result.second == Function::SignatureMatch::Ambiguous)
    {
        /// @todo  print a better warning on Function::SignatureMatch::Ambiguous
        if (!mLog.warning("implicit conversion in function call", node)) return false;
        // Get the new argument info from the function
        nativeargs = nativeargs.Cast(*result.first, mBuilder);
        // reset input to new casted arguments
        genericargs = Arguments(nativeargs);
    }

    Value ret = result.first->call(genericargs, mBuilder);
    OPENVDB_ASSERT_MESSAGE(ret, "Function has been invoked with no valid Value return");
    mValues.push(ret);
    return true;
}

bool ComputeGenerator::visit(const ast::Cast* node)
{
    Value value = mValues.top();
    if (!value.IsScalar()) {
        mLog.error("unable to cast non scalar values", node);
        return false;
    }

    // If the value to cast is already the correct type, return
    llvm::Type* targetType = llvmTypeFromToken(node->type(), mContext);
    if (value.GetUnderlyingType() == targetType) return true;

    mValues.pop(); // now pop value

    if (value.IsBool()) {
        // if target is bool, perform standard boolean conversion (*not* truncation).
        value = value.ScalarBoolComparison(mBuilder);
    }
    else {
        value = value.CastToPrecision(mBuilder, targetType);
    }
    mValues.push(value);
    return true;
}

bool ComputeGenerator::visit(const ast::DeclareLocal* node)
{
    // create storage for the local value.
    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    Value value = Value::Invalid();

    // @note  For strings, we call the string::string function rather than
    //  rely on the behaviour of insertStaticAlloca. The key difference here is
    //  that the string::string method performs the complete list of functions
    //  that are comprised by the ax::codegen::String constructor. In other
    //  words, it ensures all observable behaviour matches between the IR for
    //  strings and the C++ string implementation. Importantly,
    //  insertStaticAlloca does not initialise the first character of the SSO
    //  array to '\0' and does not call alloc (which, although does not change
    //  the string state compared to insertStaticAlloca, may change the order
    //  of assignments and other observable behaviour). Ideally,
    //  insertStaticAlloca should call string::string.
    if (node->type() == ast::tokens::STRING) {
        const FunctionGroup* axstring = this->getFunction("string::string", /*internal*/true);
        value = axstring->execute(Arguments{}, mBuilder);
    }
    else {
        value = Value::Alloc(mBuilder, type);
    }

    OPENVDB_ASSERT(value);
    OPENVDB_ASSERT(mScopeIndex > 0);
    SymbolTable<Value>* current = mSymbolTables.getOrInsert(mScopeIndex);

    const std::string& name = node->local()->name();
    if (!current->insert(name, value)) {
        mLog.error("local variable \"" + name +
                "\" has already been declared", node);
        return false;
    }

    if (mSymbolTables.find(name, mScopeIndex - 1)) {
        if (!mLog.warning("declaration of variable \"" + name
                + "\" shadows a previous declaration", node)) return false;
    }


    // do this to ensure all AST nodes are visited
    // shouldn't ever fail
    if (this->traverse(node->local())) {
        value = mValues.top(); mValues.pop();
    }
    else if (mLog.atErrorLimit()) return false;

    if (node->hasInit()) {
        if (this->traverse(node->init())) {
            Value init = mValues.top(); mValues.pop();
            value = value.Assign(mBuilder, init, &mLog, node);
            if (!value) return false;
            // note that loop conditions allow uses of initialized declarations
            // and so require the value
            mValues.push(value);
        }
        else if (mLog.atErrorLimit()) return false;
    }
    return true;
}

bool ComputeGenerator::visit(const ast::Local* node)
{
    // Reverse iterate through the current blocks and use the first declaration found
    // The current block number looks something like as follows
    //
    // ENTRY: Block 1
    //
    // if(...) // Block 3
    // {
    //     if(...) {} // Block 5
    // }
    // else {} // Block 2
    //
    // Note that in block 5, block 2 variables will be queried. However block variables
    // are constructed from the top down, so although the block number is used for
    // reverse iterating, block 2 will not contain any information
    //

    const Value* value = mSymbolTables.find(node->name());
    if (value) {
        mValues.push(*value);
    }
    else {
        mLog.error("variable \"" + node->name() + "\" hasn't been declared", node);
        return false;
    }
    return true;
}

bool ComputeGenerator::visit(const ast::ArrayUnpack* node)
{
    Value value = mValues.top(); mValues.pop();
    Value component0 = mValues.top(); mValues.pop();
    Value component1 = Value::Invalid();

    if (node->isMatrixIndex()) {
        component1 = mValues.top(); mValues.pop();
        // if double indexing, the two component values will be
        // pushed onto the stack with the first index last. i.e.
        //      top: expression
        //           2nd index (if matrix access)
        //   bottom: 1st index
        // so swap the components
        std::swap(component0, component1);
    }

    if (!value.IsArray()) {
        mLog.error("variable is not a valid type for component access", node);
        return false;
    }

    const size_t size = value.GetArrayNumElements();
    if (component1 && size <= 4) {
        mLog.error("attribute or local variable is not a compatible matrix type "
            "for [,] indexing", node);
        return false;
    }

    if (!component0.IsInteger() || (component1 && !component1.IsInteger()))
    {
        std::ostringstream os;
        llvm::raw_os_ostream stream(os);
        component0.GetUnderlyingType()->print(stream);
        if (component1) {
            stream << ", ";
            component1.GetUnderlyingType()->print(stream);
        }
        stream.flush();
        {
            mLog.error("unable to index into array with a non integer value. Types are ["
                    + os.str() + "]", node);
            return false;
        }
    }

    if (component1) {
        // component0 = row, component1 = column. Index into the matrix array
        // which is layed out in row major = (component0*dim + component1)
        OPENVDB_ASSERT(size == 9 || size == 16);
        const int32_t dim = size == 9 ? 3 : 4;
        Value offset = Value::Create<int32_t>(mContext, static_cast<int32_t>(dim));
        component0 = component0.Multiply(mBuilder, offset);
        component0 = component0.Add(mBuilder, component1);
    }

    value = value.GetArrayElement(mBuilder, component0);
    mValues.push(value);
    return true;
}

bool ComputeGenerator::visit(const ast::ArrayPack* node)
{
    const size_t num = node->children();
    // if there is only one element on the stack, leave it as a pointer to a scalar
    // or another array
    if (num == 1) return true;

    std::vector<Value> values;
    values.reserve(num);
    for (size_t i = 0; i < num; ++i)
    {
        Value value = mValues.top(); mValues.pop();
        if (value.IsArray()) {
            mLog.error("cannot build nested arrays", node->child(num-(i+1)));
            return false;
        }
        if (value.IsString()) {
            mLog.error("cannot build arrays of strings", node->child(num-(i+1)));
            return false;
        }
        values.emplace_back(value);
    }

    // reserve the values
    // @todo this should probably be handled by the AST
    std::reverse(values.begin(), values.end());

    // get the highest order type present
    llvm::Type* type = LLVMType<bool>::get(mContext);
    for (const Value& value : values) {
        type = typePrecedence(type, value.GetUnderlyingType());
    }

    // convert all to this type
    for (Value& value : values) {
        value = value.CastToPrecision(mBuilder, type);
    }

    mValues.push(Value::ScalarsToArray(mBuilder, values));
    return true;
}

bool ComputeGenerator::visit(const ast::Value<bool>* node)
{
    mValues.push(Value::Create<bool>(mContext, node->value()));
    return true;
}

bool ComputeGenerator::visit(const ast::Value<int16_t>* node)
{
    return visit<int16_t>(node);
}

bool ComputeGenerator::visit(const ast::Value<int32_t>* node)
{
    return visit<int32_t>(node);
}

bool ComputeGenerator::visit(const ast::Value<int64_t>* node)
{
    return visit<int64_t>(node);
}

bool ComputeGenerator::visit(const ast::Value<float>* node)
{
    return visit<float>(node);
}

bool ComputeGenerator::visit(const ast::Value<double>* node)
{
    return visit<double>(node);
}

bool ComputeGenerator::visit(const ast::Value<std::string>* node)
{
    OPENVDB_ASSERT(node->value().size() < static_cast<size_t>(std::numeric_limits<size_t>::max()));
    const FunctionGroup* axstring = this->getFunction("string::string", /*internal*/true);
    llvm::Value* loc = mBuilder.CreateGlobalStringPtr(node->value()); // char*
    Arguments args;
    args.AddArg(loc, ArgInfo(LLVMType<char>::get(mContext), 1));
    Value str = axstring->execute(args, mBuilder); // axstring*
    mValues.push(str);
    return true;
}

const FunctionGroup* ComputeGenerator::getFunction(const std::string &identifier,
                                                const bool allowInternal)
{
    const FunctionGroup* F =
        mFunctionRegistry.getOrInsert(identifier, mOptions, allowInternal);
    OPENVDB_ASSERT(F);
    return F;
}

template <typename ValueType>
typename std::enable_if<std::is_integral<ValueType>::value, bool>::type
ComputeGenerator::visit(const ast::Value<ValueType>* node)
{
    using ContainerT = typename ast::Value<ValueType>::ContainerType;

    static const ContainerT max =
        static_cast<ContainerT>(std::numeric_limits<ValueType>::max());
    if (node->asContainerType() > max) {
        if (!mLog.warning("signed integer overflow in integer literal "
            + std::to_string(node->asContainerType()), node)) return false;
    }

    mValues.push(Value::Create<ValueType>(mContext, node->value()));
    return true;
}

template <typename ValueType>
typename std::enable_if<std::is_floating_point<ValueType>::value, bool>::type
ComputeGenerator::visit(const ast::Value<ValueType>* node)
{
    OPENVDB_ASSERT(std::isinf(node->value()) || node->value() >= 0.0);
    mValues.push(Value::Create<ValueType>(mContext, node->value()));
    return true;
}

bool ComputeGenerator::visit(const ast::ExternalVariable* node)
{
    const std::string globalName = node->tokenname();
    auto ptrToAddress = this->globals().get(globalName);

    llvm::Value* newGlobalAddr;
    if (!ptrToAddress) {
        newGlobalAddr = llvm::cast<llvm::GlobalVariable>
            (mModule.getOrInsertGlobal(globalName, LLVMType<uintptr_t>::get(mContext)));
        this->globals().insert(globalName, newGlobalAddr);
    }
    else {
        newGlobalAddr = *ptrToAddress;
    }

    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    llvm::Value* address = mBuilder.CreateLoad(LLVMType<uintptr_t>::get(mContext), newGlobalAddr);
    llvm::Value* value = mBuilder.CreateIntToPtr(address, llvm::PointerType::get(type, 0));
    mValues.push(Value(value, type));
    return true;
}

bool ComputeGenerator::visit(const ast::Tree*)
{
    // In case we haven't returned already (i.e. we are NOT in a null block)
    // we insert a ret void. If we are, this will just get cleaned up anyway below.
    mBuilder.CreateRetVoid();
    mBuilder.SetInsertPoint(&mFunction->back());
    return true;
}

bool ComputeGenerator::visit(const ast::Attribute*)
{
    OPENVDB_ASSERT(false && "Base ComputeGenerator attempted to generate code for an Attribute. "
        "PointComputeGenerator or VolumeComputeGenerator should be used for "
        "attribute accesses.");
    return false;
}


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////


void ComputeGenerator::createFreeSymbolStrings(llvm::IRBuilder<>& B)
{
    llvm::Type* strtype = LLVMType<codegen::String>::get(mContext);

    // Loop through the initial function allocations and create string clears
    // to any strings that were created. Allocs should only be made at the
    // start of the function, so we only have to scan the function entry block.
    //
    // @note technically, AX guarantees that the first set of instructions are
    // allocs, so we could stop on the first instr that isn't an alloc. This
    // would be hard to test though should this change in the future.

    llvm::Function* F = B.GetInsertBlock()->getParent();
    llvm::BasicBlock& entry = F->getEntryBlock();

    std::vector<llvm::Value*> ptrs;

    // collect string allocas
    for (auto& inst : entry) {
        if (!llvm::isa<llvm::AllocaInst>(inst)) continue;
        llvm::AllocaInst* alloc = llvm::cast<llvm::AllocaInst>(&inst);
        if (alloc->getAllocatedType() != strtype) continue;
        ptrs.emplace_back(alloc);
    }

    if (ptrs.empty()) return;

    // clear the strings to make sure malloc has been freed
    const FunctionGroup* axstringclear =
        this->getFunction("string::clear", /*internal*/true);

    const auto IP = B.saveIP();

    for (llvm::BasicBlock& BB : *F) {
        llvm::Instruction* TI = BB.getTerminator();
        OPENVDB_ASSERT(TI);
        if (llvm::isa<llvm::ReturnInst>(TI)) {
            B.SetInsertPoint(TI);
            for (auto ptr : ptrs) {
                Arguments args;
                args.AddArg(ptr, ArgInfo(strtype, 1));
                axstringclear->execute(args, B);
            }
        }
    }

    B.restoreIP(IP);
}


} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

