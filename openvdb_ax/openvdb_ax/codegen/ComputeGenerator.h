// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/ComputeGenerator.h
///
/// @authors Nick Avramoussis, Matt Warner, Francisco Gochez, Richard Jones
///
/// @brief  The core visitor framework for code generation
///

#ifndef OPENVDB_AX_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

#include "FunctionRegistry.h"
#include "FunctionTypes.h"
#include "SymbolTable.h"

#include "../ast/AST.h"
#include "../ast/Visitor.h"
#include "../compiler/CompilerOptions.h"
#include "../compiler/Logger.h"

#include <openvdb/version.h>

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <stack>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

/// @brief  The function definition and signature which is built by the
///         ComputeGenerator.
///
///         The argument structure is as follows:
///
///           1) - A void pointer to the CustomData
///
struct ComputeKernel
{
    /// The name of the generated function
    static const std::string Name;

    /// The signature of the generated function
    using Signature = void(const void* const);
    using FunctionTraitsT = codegen::FunctionTraits<Signature>;
    static const size_t N_ARGS = FunctionTraitsT::N_ARGS;

    /// The argument key names available during code generation
    static const std::array<std::string, N_ARGS>& getArgumentKeys();
    static std::string getDefaultName();
};


///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

namespace codegen_internal {

/// @brief Visitor object which will generate llvm IR for a syntax tree. This
///   provides the majority of the code generation functionality except for
///   attribute access. This design allows for custom geometry to define their
///   IR implementations for these accesses by deriving and extending this
///   generator with ast::Attribute handling (see PointComputeGenerator.h and
///   VolumeComputeGenerator.h for examples).
/// @note The visit/traverse methods work slightly differently to the normal
///   Visitor to allow proper handling of errors and visitation history. Nodes
///   that inherit from ast::Expression can return false from visit() (and so
///   traverse()), but this will not necessarily stop traversal altogether.
///   Instead, any ast::Statements that are not also ast::Expressions i.e.
///   Block, ConditionalStatement, Loop, DeclareLocal, etc override their visit
///   and traverse methods to handle custom traversal order, and the catching
///   of failed child Expression visit/traverse calls. This allows errors in
///   independent Statements to not halt traversal for future Statements and so
///   allow capturing of multiple errors in an ast::Tree in a single call to
///   ComputeGenerator::generate().
struct OPENVDB_AX_API ComputeGenerator : public ast::Visitor<ComputeGenerator>
{
    ComputeGenerator(llvm::Module& module,
                     const FunctionOptions& options,
                     FunctionRegistry& functionRegistry,
                     Logger& logger);

    virtual ~ComputeGenerator() = default;

    bool generate(const ast::Tree&);

    inline SymbolTable& globals() { return mSymbolTables.globals(); }
    inline const SymbolTable& globals() const { return mSymbolTables.globals(); }

    // Visitor pattern

    using ast::Visitor<ComputeGenerator>::traverse;
    using ast::Visitor<ComputeGenerator>::visit;

    /// @brief  Code generation always runs post order
    inline bool postOrderNodes() const { return true; }

    /// @brief  Custom traversal of scoped blocks
    /// @note   This overrides the default traversal to incorporate
    ///         the scoping of variables declared in this block
    bool traverse(const ast::Block* block)
    {
        if (!block) return true;
        if (!this->visit(block)) return false;
        return true;
    }

    /// @brief  Custom traversal of comma expression
    /// @note   This overrides the default traversal to handle errors
    ///         without stopping generation of entire list
    /// @todo   Replace with a binary operator that simply returns the second value
    bool traverse(const ast::CommaOperator* comma)
    {
        if (!comma) return true;
        if (!this->visit(comma)) return false;
        return true;
    }


    /// @brief  Custom traversal of conditional statements
    /// @note   This overrides the default traversal to handle
    ///         branching between different code paths
    bool traverse(const ast::ConditionalStatement* cond)
    {
        if (!cond) return true;
        if (!this->visit(cond)) return false;
        return true;
    }

    /// @brief  Custom traversal of binary operators
    /// @note   This overrides the default traversal to handle
    ///         short-circuiting in logical AND and OR
    bool traverse(const ast::BinaryOperator* bin)
    {
        if (!bin) return true;
        if (!this->visit(bin)) return false;
        return true;
    }

    /// @brief  Custom traversal of ternary operators
    /// @note   This overrides the default traversal to handle
    ///         branching between different code paths
    bool traverse(const ast::TernaryOperator* tern)
    {
        if (!tern) return true;
        if (!this->visit(tern)) return false;
        return true;
    }

    /// @brief  Custom traversal of loops
    /// @note   This overrides the default traversal to handle
    ///         branching between different code paths and the
    ///         scoping of variables in for-loop initialisation
    bool traverse(const ast::Loop* loop)
    {
        if (!loop) return true;
        if (!this->visit(loop)) return false;
        return true;
    }

    /// @brief  Custom traversal of declarations
    /// @note   This overrides the default traversal to
    ///         handle traversal of the local and
    ///         assignment of initialiser, if it exists
    bool traverse(const ast::DeclareLocal* decl)
    {
        if (!decl) return true;
        if (!this->visit(decl)) return false;
        return true;
    }

    ///@{
    /// @brief  Visitor methods for all AST nodes which implement IR generation
    virtual bool visit(const ast::CommaOperator*);
    virtual bool visit(const ast::AssignExpression*);
    virtual bool visit(const ast::Crement*);
    virtual bool visit(const ast::FunctionCall*);
    virtual bool visit(const ast::Attribute*);
    virtual bool visit(const ast::Tree*);
    virtual bool visit(const ast::Block*);
    virtual bool visit(const ast::ConditionalStatement*);
    virtual bool visit(const ast::Loop*);
    virtual bool visit(const ast::Keyword*);
    virtual bool visit(const ast::UnaryOperator*);
    virtual bool visit(const ast::BinaryOperator*);
    virtual bool visit(const ast::TernaryOperator*);
    virtual bool visit(const ast::Cast*);
    virtual bool visit(const ast::DeclareLocal*);
    virtual bool visit(const ast::Local*);
    virtual bool visit(const ast::ExternalVariable*);
    virtual bool visit(const ast::ArrayUnpack*);
    virtual bool visit(const ast::ArrayPack*);
    virtual bool visit(const ast::Value<bool>*);
    virtual bool visit(const ast::Value<int16_t>*);
    virtual bool visit(const ast::Value<int32_t>*);
    virtual bool visit(const ast::Value<int64_t>*);
    virtual bool visit(const ast::Value<float>*);
    virtual bool visit(const ast::Value<double>*);
    virtual bool visit(const ast::Value<std::string>*);

    template <typename ValueType>
    typename std::enable_if<std::is_integral<ValueType>::value, bool>::type
    visit(const ast::Value<ValueType>* node);
    template <typename ValueType>

    typename std::enable_if<std::is_floating_point<ValueType>::value, bool>::type
    visit(const ast::Value<ValueType>* node);
    ///@}

protected:
    const FunctionGroup* getFunction(const std::string& identifier,
            const bool allowInternal = false);

    bool binaryExpression(llvm::Value*& result, llvm::Value* lhs, llvm::Value* rhs,
        const ast::tokens::OperatorToken op, const ast::Node* node);
    bool assignExpression(llvm::Value* lhs, llvm::Value*& rhs, const ast::Node* node);

    /// @brief Clear any strings which were allocated in a given function.
    ///   This method accepts an IRBuilder which is expected to be attached to
    ///   a valid block/function. For each block in the function with a return
    ///   instruction, this function calls the appropriate memory methods to
    ///   deallocate any strings (which are alloced in the function prologue).
    void createFreeSymbolStrings(llvm::IRBuilder<>&);

    llvm::Module& mModule;
    llvm::LLVMContext& mContext;
    llvm::IRBuilder<> mBuilder;

    // The stack of accessed values
    std::stack<llvm::Value*> mValues;

    // The stack of blocks for keyword branching
    std::stack<std::pair<llvm::BasicBlock*, llvm::BasicBlock*>> mBreakContinueStack;

    // The current scope number used to track scoped declarations
    size_t mScopeIndex;

    // The map of scope number to local variable names to values
    SymbolTableBlocks mSymbolTables;

    // The function used as the base code block
    llvm::Function* mFunction;

    const FunctionOptions mOptions;

    Logger& mLog;

private:
    FunctionRegistry& mFunctionRegistry;
};

} // codegen_internal

} // namespace codegen
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

