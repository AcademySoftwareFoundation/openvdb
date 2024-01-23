// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

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
           const std::vector<llvm::Type*>& types,
           const std::vector<const char*>& names = {},
           const std::string sep = "; ",
           const bool axTypes = true)
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
    , mScopeIndex(1)
    , mSymbolTables()
    , mFunction(nullptr)
    , mOptions(options)
    , mLog(logger)
    , mFunctionRegistry(functionRegistry) {}

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
        mValues = std::stack<llvm::Value*>();
    }

    mSymbolTables.erase(mScopeIndex);
    mScopeIndex--;
    return true;
}

bool ComputeGenerator::visit(const ast::CommaOperator* comma)
{
    // traverse the contents of the comma expression
    const size_t children = comma->children();
    llvm::Value* value = nullptr;
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
        llvm::Value* condition = mValues.top(); mValues.pop();

        if (condition->getType()->isPointerTy()) {
            condition = ir_load(mBuilder, condition);
        }
        llvm::Type* conditionType = condition->getType();
        // check the type of the condition branch is bool-convertable
        if (conditionType->isFloatingPointTy() || conditionType->isIntegerTy()) {
            condition = boolComparison(condition, mBuilder);
            mBuilder.CreateCondBr(condition, thenBlock, elseBlock);
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
    mValues = std::stack<llvm::Value*>();

    return true;
}

bool ComputeGenerator::visit(const ast::TernaryOperator* tern)
{
    llvm::BasicBlock* trueBlock = llvm::BasicBlock::Create(mContext, "ternary_true", mFunction);
    llvm::BasicBlock* falseBlock = llvm::BasicBlock::Create(mContext, "ternary_false", mFunction);
    llvm::BasicBlock* returnBlock = llvm::BasicBlock::Create(mContext, "ternary_return", mFunction);

    llvm::Value* trueValue = nullptr;
    llvm::Type* trueType = nullptr;
    bool truePtr = false;
    // generate conditional
    bool conditionSuccess = this->traverse(tern->condition());
    if (conditionSuccess) {
        // get the condition
        trueValue = mValues.top(); mValues.pop();
        OPENVDB_ASSERT(trueValue);

        trueType = trueValue->getType();
        truePtr = trueType->isPointerTy();

        llvm::Type* conditionType = truePtr ? trueType->getPointerElementType() : trueType;
        llvm::Value* boolCondition = nullptr;
        // check the type of the condition branch is bool-convertable
        if (conditionType->isFloatingPointTy() || conditionType->isIntegerTy()) {
            boolCondition = truePtr ?
                boolComparison(ir_load(mBuilder, trueValue), mBuilder) : boolComparison(trueValue, mBuilder);
            mBuilder.CreateCondBr(boolCondition, trueBlock, falseBlock);
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
            // update true type details
            trueType = trueValue->getType();
        }
        else if (mLog.atErrorLimit()) return false;
    }

    llvm::BranchInst* trueBranch = mBuilder.CreateBr(returnBlock);

    // generate false branch

    mBuilder.SetInsertPoint(falseBlock);
    bool falseSuccess = this->traverse(tern->falseBranch());
    // even if the condition isnt successful but the others are, we continue to code gen to find type errors in branches
    if (!(trueSuccess && falseSuccess)) return false;

    llvm::BranchInst* falseBranch = mBuilder.CreateBr(returnBlock);

    llvm::Value* falseValue = mValues.top(); mValues.pop();
    llvm::Type* falseType = falseValue->getType();
    OPENVDB_ASSERT(trueType);
    // if both variables of same type do no casting or loading
    if (trueType != falseType) {
        // get the (contained) types of the expressions
        truePtr = trueType->isPointerTy();
        if (truePtr) trueType = trueType->getPointerElementType();

        const bool falsePtr = falseType->isPointerTy();
        if (falsePtr) falseType = falseType->getPointerElementType();

        // if same contained type but one needs loading
        // can only have one pointer, one not, for scalars right now, i.e. no loaded arrays or strings
        if (trueType == falseType) {
            OPENVDB_ASSERT(!(truePtr && falsePtr));
            if (truePtr) {
                mBuilder.SetInsertPoint(trueBranch);
                trueValue = ir_load(mBuilder, trueValue);
            }
            else {
                mBuilder.SetInsertPoint(falseBranch);
                falseValue = ir_load(mBuilder, falseValue);
            }
        }
        else { // needs casting

            // get type for return
            llvm::Type* returnType = nullptr;

            const bool trueScalar = (trueType->isIntegerTy() || trueType->isFloatingPointTy());
            if (trueScalar &&
                 (falseType->isIntegerTy() || falseType->isFloatingPointTy())) {
                OPENVDB_ASSERT(trueType != falseType);
                // SCALAR_SCALAR
                returnType = typePrecedence(trueType, falseType);
                // always load scalars here, even if they are the correct type
                mBuilder.SetInsertPoint(trueBranch);
                if (truePtr) trueValue = ir_load(mBuilder, trueValue);
                trueValue = arithmeticConversion(trueValue, returnType, mBuilder);
                mBuilder.SetInsertPoint(falseBranch);
                if (falsePtr) falseValue = ir_load(mBuilder, falseValue);
                falseValue = arithmeticConversion(falseValue, returnType, mBuilder);
            }
            else if (trueType->isArrayTy() && falseType->isArrayTy()
                 && (trueType->getArrayNumElements() == falseType->getArrayNumElements())) {
                // ARRAY_ARRAY
                trueType = trueType->getArrayElementType();
                falseType = falseType->getArrayElementType();
                returnType = typePrecedence(trueType, falseType);

                if (trueType != returnType) {
                    mBuilder.SetInsertPoint(trueBranch);
                    trueValue = arrayCast(trueValue, returnType, mBuilder);
                }
                else if (falseType != returnType) {
                    mBuilder.SetInsertPoint(falseBranch);
                    falseValue = arrayCast(falseValue, returnType, mBuilder);
                }
            }
            else if (trueScalar && falseType->isArrayTy()) {
                // SCALAR_ARRAY
                returnType = typePrecedence(trueType, falseType->getArrayElementType());
                mBuilder.SetInsertPoint(trueBranch);
                if (truePtr) trueValue = ir_load(mBuilder, trueValue);
                trueValue = arithmeticConversion(trueValue, returnType, mBuilder);
                const size_t arraySize = falseType->getArrayNumElements();
                if (arraySize == 9 || arraySize == 16) {
                    trueValue = scalarToMatrix(trueValue, mBuilder, arraySize == 9 ? 3 : 4);
                }
                else {
                    trueValue = arrayPack(trueValue, mBuilder, arraySize);
                }
                if (falseType->getArrayElementType() != returnType) {
                    mBuilder.SetInsertPoint(falseBranch);
                    falseValue = arrayCast(falseValue, returnType, mBuilder);
                }
            }
            else if (trueType->isArrayTy() &&
                     (falseType->isIntegerTy() || falseType->isFloatingPointTy())) {
                // ARRAY_SCALAR
                returnType = typePrecedence(trueType->getArrayElementType(), falseType);
                if (trueType->getArrayElementType() != returnType) {
                    mBuilder.SetInsertPoint(trueBranch);
                    trueValue = arrayCast(trueValue, returnType, mBuilder);
                }
                mBuilder.SetInsertPoint(falseBranch);
                if (falsePtr) falseValue = ir_load(mBuilder, falseValue);
                falseValue = arithmeticConversion(falseValue, returnType, mBuilder);
                const size_t arraySize = trueType->getArrayNumElements();
                if (arraySize == 9 || arraySize == 16) {
                    falseValue = scalarToMatrix(falseValue, mBuilder, arraySize == 9 ? 3 : 4);
                }
                else {
                    falseValue = arrayPack(falseValue, mBuilder, arraySize);
                }
            }
            else {
                mLog.error("unsupported implicit cast in ternary operation",
                           tern->hasTrue() ? tern->trueBranch() : tern->falseBranch());
                return false;
            }
        }
    }
    else if (trueType->isVoidTy() && falseType->isVoidTy()) {
        // void type ternary acts like if-else statement
        // push void value to stop use of return from this expression
        mBuilder.SetInsertPoint(returnBlock);
        mValues.push(falseValue);
        return conditionSuccess && trueSuccess && falseSuccess;
    }

    // reset to continue block
    mBuilder.SetInsertPoint(returnBlock);
    llvm::PHINode* ternary = mBuilder.CreatePHI(trueValue->getType(), 2, "ternary");

    // if nesting branches the blocks for true and false branches may have been updated
    // so get these again rather than reusing trueBlock/falseBlock
    ternary->addIncoming(trueValue, trueBranch->getParent());
    ternary->addIncoming(falseValue, falseBranch->getParent());

    mValues.push(ternary);
    return conditionSuccess && trueSuccess && falseSuccess;
}

bool ComputeGenerator::visit(const ast::Loop* loop)
{
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
            mValues = std::stack<llvm::Value*>();
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
    if (this->traverse(loop->condition())) {
        llvm::Value* condition = mValues.top(); mValues.pop();
        if (condition->getType()->isPointerTy()) {
            condition = ir_load(mBuilder, condition);
        }
        llvm::Type* conditionType = condition->getType();
        // check the type of the condition branch is bool-convertable
        if (conditionType->isFloatingPointTy() || conditionType->isIntegerTy()) {
            condition = boolComparison(condition, mBuilder);
            mBuilder.CreateCondBr(condition, bodyBlock, postLoopBlock);
        }
        else {
            if (!mLog.error("cannot convert non-scalar type to bool in condition", loop->condition())) return false;
        }
        // reset the value stack
        mValues = std::stack<llvm::Value*>();
    }
    else if (mLog.atErrorLimit()) return false;

    // reset to post loop block
    mBuilder.SetInsertPoint(postLoopBlock);

    // discard break and continue
    mBreakContinueStack.pop();

    // remove the symbol table created in this scope
    mSymbolTables.erase(mScopeIndex);
    mScopeIndex--;

    // reset the value stack
    mValues = std::stack<llvm::Value*>();

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
    openvdb::ax::ast::tokens::OperatorToken opToken = node->operation();
    // if AND or OR, need to handle short-circuiting
    if (opToken == openvdb::ax::ast::tokens::OperatorToken::AND
        || opToken == openvdb::ax::ast::tokens::OperatorToken::OR) {
        llvm::BranchInst* lhsBranch = nullptr;
        llvm::BasicBlock* rhsBlock = llvm::BasicBlock::Create(mContext, "binary_rhs", mFunction);
        llvm::BasicBlock* returnBlock = llvm::BasicBlock::Create(mContext, "binary_return", mFunction);
        llvm::Value* lhs = nullptr;
        bool lhsSuccess = this->traverse(node->lhs());
        if (lhsSuccess) {
            lhs = mValues.top(); mValues.pop();
            llvm::Type* lhsType = lhs->getType();
            if (lhsType->isPointerTy()) {
                lhs = ir_load(mBuilder, lhs);
                lhsType = lhsType->getPointerElementType();
            }

            if (lhsType->isFloatingPointTy() || lhsType->isIntegerTy()) {
                lhs = boolComparison(lhs, mBuilder);

                if (opToken == openvdb::ax::ast::tokens::OperatorToken::AND) {
                    lhsBranch = mBuilder.CreateCondBr(lhs, rhsBlock, returnBlock);
                }
                else {
                    lhsBranch = mBuilder.CreateCondBr(lhs, returnBlock, rhsBlock);
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
            llvm::Value* rhs = mValues.top(); mValues.pop();
            llvm::Type* rhsType = rhs->getType();
            if (rhsType->isPointerTy()) {
                rhs = ir_load(mBuilder, rhs);
                rhsType = rhsType->getPointerElementType();
            }

            if (rhsType->isFloatingPointTy() || rhsType->isIntegerTy()) {
                rhs = boolComparison(rhs, mBuilder);
                llvm::BranchInst* rhsBranch = mBuilder.CreateBr(returnBlock);

                mBuilder.SetInsertPoint(returnBlock);
                if (lhsBranch) {// i.e. lhs was successful
                    OPENVDB_ASSERT(rhs && lhs);
                    llvm::PHINode* result = mBuilder.CreatePHI(LLVMType<bool>::get(mContext), 2, "binary_op");
                    result->addIncoming(lhs, lhsBranch->getParent());
                    result->addIncoming(rhs, rhsBranch->getParent());
                    mValues.push(result);
                }
            }
            else {
                mLog.error("cannot convert non-scalar rhs to bool", node->rhs());
                rhsSuccess = false;
            }
        }
        return lhsSuccess && rhsSuccess;
    }
    else {
        llvm::Value* lhs = nullptr;
        if (this->traverse(node->lhs())) {
            lhs = mValues.top(); mValues.pop();
        }
        else if (mLog.atErrorLimit()) return false;
        llvm::Value* rhs = nullptr;
        if (this->traverse(node->rhs())) {
            rhs = mValues.top(); mValues.pop();
        }
        else if (mLog.atErrorLimit()) return false;
        llvm::Value* result = nullptr;
        if (!(lhs && rhs) || !this->binaryExpression(result, lhs, rhs, node->operation(), node)) return false;

        if (result) {
            mValues.push(result);
        }
    }
    return true;
}

bool ComputeGenerator::visit(const ast::UnaryOperator* node)
{
    // If the unary operation is a +, keep the value ptr on the stack and
    // continue (avoid any new allocations or unecessary loads)

    const ast::tokens::OperatorToken token = node->operation();
    if (token == ast::tokens::PLUS) return true;

    if (token != ast::tokens::MINUS &&
        token != ast::tokens::BITNOT &&
        token != ast::tokens::NOT) {
        mLog.error("unrecognised unary operator \"" +
                ast::tokens::operatorNameFromToken(token) + "\"", node);
        return false;
    }
    // unary operator uses default traversal so value should be on the stack
    llvm::Value* value = mValues.top();
    llvm::Type* type = value->getType();
    if (type->isPointerTy()) {
        type = type->getPointerElementType();
        if (type->isIntegerTy() || type->isFloatingPointTy()) {
            value = ir_load(mBuilder, value);
        }
    }

    llvm::Value* result = nullptr;
    if (type->isIntegerTy()) {
        if (token == ast::tokens::NOT) {
            if (type->isIntegerTy(1))  result = mBuilder.CreateICmpEQ(value, llvm::ConstantInt::get(type, 0));
            else                       result = mBuilder.CreateICmpEQ(value, llvm::ConstantInt::getSigned(type, 0));
        }
        else {
            // if bool, cast to int32 for unary minus and bitnot
            if (type->isIntegerTy(1)) {
                type = LLVMType<int32_t>::get(mContext);
                value = arithmeticConversion(value, type, mBuilder);
            }
            if (token == ast::tokens::MINUS)        result = mBuilder.CreateNeg(value);
            else if (token == ast::tokens::BITNOT)  result = mBuilder.CreateNot(value);
        }
    }
    else if (type->isFloatingPointTy()) {
        if (token == ast::tokens::MINUS)         result = mBuilder.CreateFNeg(value);
        else if (token == ast::tokens::NOT)      result = mBuilder.CreateFCmpOEQ(value, llvm::ConstantFP::get(type, 0));
        else if (token == ast::tokens::BITNOT) {
            mLog.error("unable to perform operation \""
                    + ast::tokens::operatorNameFromToken(token) + "\" on floating point values", node);
            return false;
        }
    }
    else if (type->isArrayTy()) {
        type = type->getArrayElementType();
        std::vector<llvm::Value*> elements;
        arrayUnpack(value, elements, mBuilder, /*load*/true);
        OPENVDB_ASSERT(elements.size() > 0);

        if (type->isIntegerTy()) {
            if (token == ast::tokens::MINUS) {
                for (llvm::Value*& element : elements) {
                    element = mBuilder.CreateNeg(element);
                }
            }
            else if (token == ast::tokens::NOT) {
                for (llvm::Value*& element : elements) {
                    element = mBuilder.CreateICmpEQ(element,
                        llvm::ConstantInt::getSigned(type, 0));
                }
            }
            else if (token == ast::tokens::BITNOT) {
                for (llvm::Value*& element : elements) {
                    element = mBuilder.CreateNot(element);
                }
            }
        }
        else if (type->isFloatingPointTy()) {
            if (token == ast::tokens::MINUS) {
                for (llvm::Value*& element : elements) {
                    element = mBuilder.CreateFNeg(element);
                }
            }
            else {
                //@todo support NOT?
                mLog.error("unable to perform operation \""
                        + ast::tokens::operatorNameFromToken(token) + "\" on arrays/vectors", node);
                return false;
            }
        }
        else {
            mLog.error("unrecognised array element type", node);
            return false;
        }

        result = arrayPack(elements, mBuilder);
    }
    else {
        mLog.error("value is not a scalar or vector", node);
        return false;
    }
    OPENVDB_ASSERT(result);
    mValues.pop();
    mValues.push(result);
    return true;
}

bool ComputeGenerator::visit(const ast::AssignExpression* assign)
{
    // default traversal, should have rhs and lhs on stack
    // leave LHS on stack
    llvm::Value* rhs = mValues.top(); mValues.pop();
    llvm::Value* lhs = mValues.top();

    llvm::Type* rhsType = rhs->getType();
    if (assign->isCompound()) {
        llvm::Value* rhsValue = nullptr;
        if (!this->binaryExpression(rhsValue, lhs, rhs, assign->operation(), assign)) return false;
        OPENVDB_ASSERT(rhsValue);
        rhs = rhsValue;
        rhsType = rhs->getType();
    }
    // rhs must be loaded for assignExpression() if it's a scalar
    if (rhsType->isPointerTy()) {
        rhsType = rhsType->getPointerElementType();
        if (rhsType->isIntegerTy() || rhsType->isFloatingPointTy()) {
            rhs = ir_load(mBuilder, rhs);
        }
    }

    if (!this->assignExpression(lhs, rhs, assign)) return false;
    return true;
}

bool ComputeGenerator::visit(const ast::Crement* node)
{
    llvm::Value* value = mValues.top();
    if (!value->getType()->isPointerTy()) {
        mLog.error("unable to assign to an rvalue", node);
        return false;
    }
    llvm::Value* rvalue = ir_load(mBuilder, value);
    llvm::Type* type = rvalue->getType();

    if (type->isIntegerTy(1) || (!type->isIntegerTy() && !type->isFloatingPointTy())) {
        mLog.error("variable is an unsupported type for "
                "crement. Must be a non-boolean scalar", node);
        return false;
    }
    else {
        llvm::Value* crement = nullptr;
        OPENVDB_ASSERT((node->increment() || node->decrement()) && "unrecognised crement operation");
        if (node->increment())      crement = LLVMType<int32_t>::get(mContext, 1);
        else if (node->decrement()) crement = LLVMType<int32_t>::get(mContext, -1);

        crement = arithmeticConversion(crement, type, mBuilder);
        if (type->isIntegerTy())       crement = mBuilder.CreateAdd(rvalue, crement);
        if (type->isFloatingPointTy()) crement = mBuilder.CreateFAdd(rvalue, crement);

        mBuilder.CreateStore(crement, value);

        // decide what to put on the expression stack
    }
    mValues.pop();

    if (node->post()) mValues.push(rvalue);
    else              mValues.push(value);
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
    else {
        const size_t args = node->children();
        OPENVDB_ASSERT(mValues.size() >= args);

        // initialize arguments. scalars are always passed by value, arrays
        // and strings always by pointer

        std::vector<llvm::Value*> arguments;
        arguments.resize(args);

        for (auto r = arguments.rbegin(); r != arguments.rend(); ++r) {
            llvm::Value* arg = mValues.top(); mValues.pop();
            llvm::Type* type = arg->getType();
            if (type->isPointerTy()) {
                type = type->getPointerElementType();
                if (type->isIntegerTy() || type->isFloatingPointTy()) {
                    // pass by value
                    arg = ir_load(mBuilder, arg);
                }
            }
            else {
                // arrays should never be loaded
                OPENVDB_ASSERT(!type->isArrayTy() && type != LLVMType<codegen::String>::get(mContext));
                if (type->isIntegerTy() || type->isFloatingPointTy()) {
                    /*pass by value*/
                }
            }
            *r = arg;
        }

        std::vector<llvm::Type*> inputTypes;
        valuesToTypes(arguments, inputTypes);

        Function::SignatureMatch match;
        const Function* target = function->match(inputTypes, mContext, &match);

        if (!target) {
            OPENVDB_ASSERT(!function->list().empty()
                   && "FunctionGroup has no function declarations");

            std::ostringstream os;
            if (match == Function::SignatureMatch::None) {
                os << "wrong number of arguments. \"" << node->name() << "\""
                   << " was called with: (";
                llvm::raw_os_ostream stream(os);
                printTypes(stream, inputTypes);
                stream << ")";
            }
            else {
                // match == Function::SignatureMatch::Size
                os << "no matching function for ";
                printSignature(os, inputTypes,
                    LLVMType<void>::get(mContext),
                    node->name().c_str(), {}, true);
            }

            os << " \ncandidates are: ";
            for (const auto& sig : function->list()) {
                os << std::endl;
                sig->print(mContext, os, node->name().c_str());
            }
            mLog.error(os.str(), node);
            return false;
        }
        else {
            llvm::Value* result = nullptr;
            if (match == Function::SignatureMatch::Implicit) {
                if (!mLog.warning("implicit conversion in function call", node)) return false;
                result = target->call(arguments, mBuilder, /*cast=*/true);
            }
            else {
                // match == Function::SignatureMatch::Explicit
                result = target->call(arguments, mBuilder, /*cast=*/false);
            }

            OPENVDB_ASSERT(result && "Function has been invoked with no valid llvm Value return");
            mValues.push(result);
        }
    }
    return true;
}

bool ComputeGenerator::visit(const ast::Cast* node)
{
    llvm::Value* value = mValues.top(); mValues.pop();

    llvm::Type* type =
        value->getType()->isPointerTy() ?
        value->getType()->getPointerElementType() :
        value->getType();

    if (!type->isIntegerTy() && !type->isFloatingPointTy()) {
        mLog.error("unable to cast non scalar values", node);
        return false;
    }
    else {
        // If the value to cast is already the correct type, return
        llvm::Type* targetType = llvmTypeFromToken(node->type(), mContext);
        if (type == targetType) return true;


        if (value->getType()->isPointerTy()) {
            value = ir_load(mBuilder, value);
        }

        if (targetType->isIntegerTy(1)) {
            // if target is bool, perform standard boolean conversion (*not* truncation).
            value = boolComparison(value, mBuilder);
            OPENVDB_ASSERT(value->getType()->isIntegerTy(1));
        }
        else {
            value = arithmeticConversion(value, targetType, mBuilder);
        }
        mValues.push(value);
    }

    return true;
}

bool ComputeGenerator::visit(const ast::DeclareLocal* node)
{
    // create storage for the local value.
    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    llvm::Value* value;

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
        value = axstring->execute({}, mBuilder);
    }
    else {
        value = insertStaticAlloca(mBuilder, type);
    }

    OPENVDB_ASSERT(value);
    SymbolTable* current = mSymbolTables.getOrInsert(mScopeIndex);

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
            llvm::Value* init = mValues.top(); mValues.pop();
            llvm::Type* initType = init->getType();

            if (initType->isPointerTy()) {
                initType = initType->getPointerElementType();
                if (initType->isIntegerTy() || initType->isFloatingPointTy()) {
                    init = ir_load(mBuilder, init);
                }
            }
            if (!this->assignExpression(value, init, node)) return false;

            // note that loop conditions allow uses of initialized declarations
            // and so require the value
            if (value) mValues.push(value);
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

    llvm::Value* value = mSymbolTables.find(node->name());
    if (value) {
        mValues.push(value);
    }
    else {
        mLog.error("variable \"" + node->name() + "\" hasn't been declared", node);
        return false;
    }
    return true;
}

bool ComputeGenerator::visit(const ast::ArrayUnpack* node)
{
    llvm::Value* value = mValues.top(); mValues.pop();
    llvm::Value* component0 = mValues.top(); mValues.pop();
    llvm::Value* component1 = nullptr;

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

    llvm::Type* type = value->getType();
    if (!type->isPointerTy() ||
        !type->getPointerElementType()->isArrayTy()) {
        mLog.error("variable is not a valid type for component access", node);
        return false;
    }

    // type now guaranteed to be an array type
    type = type->getPointerElementType();
    const size_t size = type->getArrayNumElements();
    if (component1 && size <= 4) {
        {
            mLog.error("attribute or local variable is not a compatible matrix type "
                "for [,] indexing", node);
            return false;
        }
    }

    if (component0->getType()->isPointerTy()) {
        component0 = ir_load(mBuilder, component0);
    }
    if (component1 && component1->getType()->isPointerTy()) {
        component1 = ir_load(mBuilder, component1);
    }

    if (!component0->getType()->isIntegerTy() ||
        (component1 && !component1->getType()->isIntegerTy())) {
        std::ostringstream os;
        llvm::raw_os_ostream stream(os);
        component0->getType()->print(stream);
        if (component1) {
            stream << ", ";
            component1->getType()->print(stream);
        }
        stream.flush();
        {
            mLog.error("unable to index into array with a non integer value. Types are ["
                    + os.str() + "]", node);
            return false;
        }
    }

    llvm::Value* zero = LLVMType<int32_t>::get(mContext, 0);
    if (!component1) {
        value = ir_gep(mBuilder, value, {zero, component0});
    }
    else {
        // component0 = row, component1 = column. Index into the matrix array
        // which is layed out in row major = (component0*dim + component1)
        OPENVDB_ASSERT(size == 9 || size == 16);
        const int32_t dim = size == 9 ? 3 : 4;
        llvm::Value* offset =
            LLVMType<int32_t>::get(mContext, static_cast<int32_t>(dim));
        component0 = binaryOperator(component0, offset, ast::tokens::MULTIPLY, mBuilder);
        component0 = binaryOperator(component0, component1, ast::tokens::PLUS, mBuilder);
        value = ir_gep(mBuilder, value, {zero, component0});
    }

    mValues.push(value);
    return true;
}

bool ComputeGenerator::visit(const ast::ArrayPack* node)
{
    const size_t num = node->children();

    // if there is only one element on the stack, leave it as a pointer to a scalar
    // or another array
    if (num == 1) return true;

    llvm::Type* strtype = LLVMType<codegen::String>::get(mContext);

    std::vector<llvm::Value*> values;
    values.reserve(num);
    for (size_t i = 0; i < num; ++i) {
        llvm::Value* value = mValues.top(); mValues.pop();
        if (value->getType()->isPointerTy()) {
            value = ir_load(mBuilder, value);
        }
        if (value->getType()->isArrayTy()) {
            mLog.error("cannot build nested arrays", node->child(num-(i+1)));
            return false;
        }
        if (value->getType() == strtype) {
            mLog.error("cannot build arrays of strings", node->child(num-(i+1)));
            return false;
        }

        values.emplace_back(value);
    }

    // reserve the values
    // @todo this should probably be handled by the AST
    std::reverse(values.begin(), values.end());

    llvm::Value* array = arrayPackCast(values, mBuilder);
    mValues.push(array);
    return true;
}

bool ComputeGenerator::visit(const ast::Value<bool>* node)
{
    llvm::Constant* value = LLVMType<bool>::get(mContext, node->value());
    mValues.push(value);
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
    llvm::Value* result = axstring->execute({loc}, mBuilder);
    mValues.push(result);
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

    llvm::Constant* value = LLVMType<ValueType>::get(mContext, node->value());
    mValues.push(value);
    return true;
}

template <typename ValueType>
typename std::enable_if<std::is_floating_point<ValueType>::value, bool>::type
ComputeGenerator::visit(const ast::Value<ValueType>* node)
{
    OPENVDB_ASSERT(std::isinf(node->value()) || node->value() >= 0.0);
    llvm::Constant* value = LLVMType<ValueType>::get(mContext, node->value());
    mValues.push(value);
    return true;
}

bool ComputeGenerator::visit(const ast::ExternalVariable* node)
{
    const std::string globalName = node->tokenname();
    llvm::Value* ptrToAddress = this->globals().get(globalName);

    if (!ptrToAddress) {
        ptrToAddress = llvm::cast<llvm::GlobalVariable>
            (mModule.getOrInsertGlobal(globalName, LLVMType<uintptr_t>::get(mContext)));
        this->globals().insert(globalName, ptrToAddress);
    }

    llvm::Type* type = llvmTypeFromToken(node->type(), mContext);
    llvm::Value* address = ir_load(mBuilder, ptrToAddress);
    llvm::Value* value = mBuilder.CreateIntToPtr(address, type->getPointerTo(0));
    if (type->isIntegerTy() || type->isFloatingPointTy()) {
        value = ir_load(mBuilder, value);
    }
    mValues.push(value);
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

bool ComputeGenerator::assignExpression(llvm::Value* lhs, llvm::Value*& rhs, const ast::Node* node)
{
    llvm::Type* strtype = LLVMType<codegen::String>::get(mContext);

    llvm::Type* ltype = lhs->getType();
    llvm::Type* rtype = rhs->getType();

    if (!ltype->isPointerTy()) {
        mLog.error("unable to assign to an rvalue", node);
        return false;
    }

    ltype = ltype->getPointerElementType();
    if (rtype->isPointerTy()) rtype = rtype->getPointerElementType();

    size_t lsize = ltype->isArrayTy() ? ltype->getArrayNumElements() : 1;
    size_t rsize = rtype->isArrayTy() ? rtype->getArrayNumElements() : 1;

    // Handle scalar->matrix promotion if necessary
    // @todo promote all values (i.e. scalar to vectors) to make below branching
    // easier. Need to verifier IR is able to optimise to the same logic

    if (lsize == 9 || lsize == 16) {
        if (rtype->isIntegerTy() || rtype->isFloatingPointTy()) {
            if (rhs->getType()->isPointerTy()) {
                rhs = ir_load(mBuilder, rhs);
            }
            rhs = arithmeticConversion(rhs, ltype->getArrayElementType(), mBuilder);
            rhs = scalarToMatrix(rhs, mBuilder, lsize == 9 ? 3 : 4);
            rtype = rhs->getType()->getPointerElementType();
            rsize = lsize;
        }
    }

    if (lsize != rsize) {
        if (lsize > 1 && rsize > 1) {
            mLog.error("unable to assign vector/array "
                "attributes with mismatching sizes", node);
            return false;
        }
        else if (lsize == 1) {
            OPENVDB_ASSERT(rsize > 1);
            mLog.error("cannot assign a scalar value "
                "from a vector or matrix. Consider using the [] operator to "
                "get a particular element", node);
            return false;
        }
    }

    // All remaining operators are either componentwise, string or invalid implicit casts

    const bool string =
        (ltype == strtype && rtype == strtype);

    const bool componentwise = !string &&
        (rtype->isFloatingPointTy() || rtype->isIntegerTy() || rtype->isArrayTy()) &&
        (ltype->isFloatingPointTy() || ltype->isIntegerTy() || ltype->isArrayTy());

    if (componentwise) {
        OPENVDB_ASSERT(rsize == lsize || (rsize == 1 || lsize == 1));
        const size_t resultsize = std::max(lsize, rsize);

        if (ltype != rtype) {
            llvm::Type* letype = ltype->isArrayTy() ? ltype->getArrayElementType() : ltype;
            llvm::Type* retype = rtype->isArrayTy() ? rtype->getArrayElementType() : rtype;
            if (letype != retype) {
                llvm::Type* highest = typePrecedence(letype, retype);
                if (highest != letype) {
                    if (!mLog.warning("implicit conversion in assignment (possible truncation)", node)) return false;
                }
            }
        }

        // compute the componentwise precision

        llvm::Type* opprec = ltype->isArrayTy() ? ltype->getArrayElementType() : ltype;
        // if target is bool, perform standard boolean conversion (*not* truncation).
        // i.e. if rhs is anything but zero, lhs is true
        // @todo zeroval should be at rhstype
        if (opprec->isIntegerTy(1)) {
            llvm::Value* newRhs = nullptr;
            if (!this->binaryExpression(newRhs, LLVMType<int32_t>::get(mContext, 0), rhs, ast::tokens::NOTEQUALS, node)) return false;
            if (!newRhs) return true;
            rhs = newRhs;
            OPENVDB_ASSERT(newRhs->getType()->isIntegerTy(1));
        }

        for (size_t i = 0; i < resultsize; ++i) {
            llvm::Value* lelement = lsize == 1 ? lhs : ir_constgep2_64(mBuilder, lhs, 0, i);
            llvm::Value* relement = rsize == 1 ? rhs : ir_load(mBuilder, ir_constgep2_64(mBuilder, rhs, 0, i));
            relement = arithmeticConversion(relement, opprec, mBuilder);
            mBuilder.CreateStore(relement, lelement);
        }
    }
    else if (string) {
        const FunctionGroup* axstringassign = this->getFunction("string::op=", /*internal*/true);
        axstringassign->execute({lhs, rhs}, mBuilder);
    }
    else {
        mLog.error("unsupported implicit cast in assignment", node);
        return false;
    }
    return true;
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
                axstringclear->execute({ptr}, B);
            }
        }
    }

    B.restoreIP(IP);
}

bool ComputeGenerator::binaryExpression(llvm::Value*& result, llvm::Value* lhs, llvm::Value* rhs,
    const ast::tokens::OperatorToken op, const ast::Node* node)
{
    llvm::Type* strtype = LLVMType<codegen::String>::get(mContext);

    llvm::Type* ltype = lhs->getType();
    llvm::Type* rtype = rhs->getType();

    if (ltype->isPointerTy()) ltype = ltype->getPointerElementType();
    if (rtype->isPointerTy()) rtype = rtype->getPointerElementType();

    size_t lsize = ltype->isArrayTy() ? ltype->getArrayNumElements() : 1;
    size_t rsize = rtype->isArrayTy() ? rtype->getArrayNumElements() : 1;

    // Handle scalar->matrix promotion if necessary
    // @todo promote all values (i.e. scalar to vectors) to make below branching
    // easier. Need to verifier IR is able to optimise to the same logic

    if (lsize == 9 || lsize == 16) {
        if (rtype->isIntegerTy() || rtype->isFloatingPointTy()) {
            if (rhs->getType()->isPointerTy()) {
                rhs = ir_load(mBuilder, rhs);
            }
            rhs = arithmeticConversion(rhs, ltype->getArrayElementType(), mBuilder);
            rhs = scalarToMatrix(rhs, mBuilder, lsize == 9 ? 3 : 4);
            rtype = rhs->getType()->getPointerElementType();
            rsize = lsize;
            if (auto* child = node->child(0)) {
                if (child->isType<ast::ArrayPack>()) {
                    mLog.error("unable to deduce implicit {...} type for binary op as value "
                        "may be a matrix or array. assign to a local mat variable", child);
                    return false;
                }
            }
            if (!mLog.warning("implicit cast to matrix from scalar. resulting "
                "cast will be equal to scalar * identity.", node->child(1))) return false;
        }
    }

    if (rsize == 9 || rsize == 16) {
        if (ltype->isIntegerTy() || ltype->isFloatingPointTy()) {
            if (lhs->getType()->isPointerTy()) {
                lhs = ir_load(mBuilder, lhs);
            }
            lhs = arithmeticConversion(lhs, rtype->getArrayElementType(), mBuilder);
            lhs = scalarToMatrix(lhs, mBuilder, rsize == 9 ? 3 : 4);
            ltype = lhs->getType()->getPointerElementType();
            lsize = rsize;
            if (auto* child = node->child(1)) {
                if (child->isType<ast::ArrayPack>()) {
                    mLog.error("unable to deduce implicit {...} type for binary op as value "
                        "may be a matrix or array. assign to a local mat variable", child);
                    return false;
                }
            }
            if (!mLog.warning("implicit cast to matrix from scalar. resulting "
                "cast will be equal to scalar * identity.", node->child(0))) return false;
        }
    }

    //

    const ast::tokens::OperatorType opType = ast::tokens::operatorType(op);
    result = nullptr;
    // Handle custom matrix operators

    if (lsize >= 9 || rsize >= 9)
    {
        if (op == ast::tokens::MULTIPLY) {
            if ((lsize == 9 && rsize == 9) ||
                (lsize == 16 && rsize == 16)) {
                // matrix matrix multiplication all handled through mmmult
                result = this->getFunction("mmmult", /*internal*/true)->execute({lhs, rhs}, mBuilder);
            }
            else if ((lsize ==  9 && rsize == 3) ||
                     (lsize == 16 && rsize == 3) ||
                     (lsize == 16 && rsize == 4)) {
                // matrix vector multiplication all handled through pretransform
                result = this->getFunction("pretransform")->execute({lhs, rhs}, mBuilder);
            }
            else if ((lsize == 3 && rsize ==  9) ||
                     (lsize == 3 && rsize == 16) ||
                     (lsize == 4 && rsize == 16)) {
                // vector matrix multiplication all handled through transform
                result = this->getFunction("transform")->execute({lhs, rhs}, mBuilder);
            }
            else {
                mLog.error("unsupported * operator on "
                    "vector/matrix sizes", node);
                return false;
            }
        }
        else if (op == ast::tokens::MORETHAN ||
                 op == ast::tokens::LESSTHAN ||
                 op == ast::tokens::MORETHANOREQUAL ||
                 op == ast::tokens::LESSTHANOREQUAL ||
                 op == ast::tokens::DIVIDE || // no / support for mats
                 op == ast::tokens::MODULO || // no % support for mats
                 opType == ast::tokens::LOGICAL ||
                 opType == ast::tokens::BITWISE) {
            mLog.error("call to unsupported operator \""
                + ast::tokens::operatorNameFromToken(op) +
                "\" with a vector/matrix argument", node);
            return false;
        }
    }

    if (!result) {
        // Handle matrix/vector ops of mismatching sizes
        if (lsize > 1 || rsize > 1) {
            if (lsize != rsize && (lsize > 1 && rsize > 1)) {
                mLog.error("unsupported binary operator on vector/matrix "
                    "arguments of mismatching sizes", node);
                return false;
            }
            if (op == ast::tokens::MORETHAN ||
                op == ast::tokens::LESSTHAN ||
                op == ast::tokens::MORETHANOREQUAL ||
                op == ast::tokens::LESSTHANOREQUAL ||
                opType == ast::tokens::LOGICAL ||
                opType == ast::tokens::BITWISE) {
                mLog.error("call to unsupported operator \""
                    + ast::tokens::operatorNameFromToken(op) +
                    "\" with a vector/matrix argument", node);
                return false;
            }
        }

        // Handle invalid floating point ops
        if (rtype->isFloatingPointTy() || ltype->isFloatingPointTy()) {
            if (opType == ast::tokens::BITWISE) {
                mLog.error("call to unsupported operator \""
                    + ast::tokens::operatorNameFromToken(op) +
                    "\" with a floating point argument", node);
                return false;
            }
        }
    }

    // All remaining operators are either componentwise, string or invalid implicit casts

    const bool componentwise = !result &&
        (rtype->isFloatingPointTy() || rtype->isIntegerTy() || rtype->isArrayTy()) &&
        (ltype->isFloatingPointTy() || ltype->isIntegerTy() || ltype->isArrayTy());

    if (componentwise)
    {
        OPENVDB_ASSERT(ltype->isArrayTy() || ltype->isFloatingPointTy() || ltype->isIntegerTy());
        OPENVDB_ASSERT(rtype->isArrayTy() || rtype->isFloatingPointTy() || rtype->isIntegerTy());
        OPENVDB_ASSERT(rsize == lsize || (rsize == 1 || lsize == 1));

        if (op == ast::tokens::DIVIDE || op == ast::tokens::MODULO) {
            if (llvm::Constant* c = llvm::dyn_cast<llvm::Constant>(rhs)) {
                if (c->isZeroValue()) {
                    if (op == ast::tokens::DIVIDE) {
                        if (!mLog.warning("division by zero is undefined", node)) return false;
                    }
                    else {
                        if (!mLog.warning("modulo by zero is undefined", node)) return false;
                    }
                }
            }
        }

        // compute the componentwise precision

        llvm::Type* opprec = ltype->isArrayTy() ? ltype->getArrayElementType() : ltype;
        opprec = rtype->isArrayTy() ?
            typePrecedence(opprec, rtype->getArrayElementType()) :
            typePrecedence(opprec, rtype);

        // if bool, the lowest precision and subsequent result should be int32
        // for arithmetic, bitwise and certain other ops
        // @note - no bool containers, so if the type is a container, it can't
        // contain booleans
        if (opprec->isIntegerTy(1)) {
            if (opType == ast::tokens::ARITHMETIC ||
                opType == ast::tokens::BITWISE ||
                op == ast::tokens::MORETHAN ||
                op == ast::tokens::LESSTHAN ||
                op == ast::tokens::MORETHANOREQUAL ||
                op == ast::tokens::LESSTHANOREQUAL) {
                opprec = LLVMType<int32_t>::get(mContext);
            }
        }

        // load scalars once

        if (!ltype->isArrayTy()) {
            if (lhs->getType()->isPointerTy()) {
                lhs = ir_load(mBuilder, lhs);
            }
        }
        if (!rtype->isArrayTy()) {
            if (rhs->getType()->isPointerTy()) {
                rhs = ir_load(mBuilder, rhs);
            }
        }

        const size_t resultsize = std::max(lsize, rsize);
        std::vector<llvm::Value*> elements;
        elements.reserve(resultsize);

        // handle floored modulo
        const Function* target = nullptr;
        auto runop = [&target, op, this](llvm::Value* a, llvm::Value* b) {
            if (target) return target->call({a,b}, this->mBuilder, /*cast=*/false);
            else        return binaryOperator(a, b, op, this->mBuilder);
        };

        if (op == ast::tokens::MODULO) {
            const FunctionGroup* mod = this->getFunction("floormod");
            OPENVDB_ASSERT(mod);
            target = mod->match({opprec,opprec}, mContext);
            OPENVDB_ASSERT(target);
        }

        // perform op
        for (size_t i = 0; i < resultsize; ++i) {
            llvm::Value* lelement = lsize == 1 ? lhs : ir_load(mBuilder, ir_constgep2_64(mBuilder, lhs, 0, i));
            llvm::Value* relement = rsize == 1 ? rhs : ir_load(mBuilder, ir_constgep2_64(mBuilder, rhs, 0, i));
            lelement = arithmeticConversion(lelement, opprec, mBuilder);
            relement = arithmeticConversion(relement, opprec, mBuilder);
            elements.emplace_back(runop(lelement, relement));
        }

        // handle vec/mat results
        if (resultsize > 1) {
            if (op == ast::tokens::EQUALSEQUALS || op == ast::tokens::NOTEQUALS) {
                const ast::tokens::OperatorToken reductionOp =
                    op == ast::tokens::EQUALSEQUALS ? ast::tokens::AND : ast::tokens::OR;
                result = elements.front();
                OPENVDB_ASSERT(result->getType() == LLVMType<bool>::get(mContext));
                for (size_t i = 1; i < resultsize; ++i) {
                    result = binaryOperator(result, elements[i], reductionOp, mBuilder);
                }
            }
            else {
                // Create the allocation at the start of the function block
                result = insertStaticAlloca(mBuilder,
                    llvm::ArrayType::get(opprec, resultsize));
                for (size_t i = 0; i < resultsize; ++i) {
                    mBuilder.CreateStore(elements[i], ir_constgep2_64(mBuilder, result, 0, i));
                }
            }
        }
        else {
            result = elements.front();
        }
    }

    const bool string = !result &&
        (ltype == strtype && rtype == strtype);

    if (string)
    {
        if (op != ast::tokens::PLUS) {
            mLog.error("unsupported string operation \""
                + ast::tokens::operatorNameFromToken(op) + "\"", node);
            return false;
        }

        const FunctionGroup* axstringplus = this->getFunction("string::op+", /*internal*/true);
        result = axstringplus->execute({lhs, rhs}, mBuilder);
    }

    if (!result) {
        mLog.error("unsupported implicit cast in binary op", node);
        return false;
    }

    return true;
}

} // namespace codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

