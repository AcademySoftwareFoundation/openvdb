///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2020 DNEG
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DNEG nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////

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

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Visitor.h>
#include <openvdb_ax/compiler/CompilerOptions.h>

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


struct ComputeGenerator : public ast::Visitor<ComputeGenerator>
{
    ComputeGenerator(llvm::Module& module,
                     const FunctionOptions& options,
                     FunctionRegistry& functionRegistry,
                     std::vector<std::string>* const warnings = nullptr);

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

    /// @brief  Custom traversal of conditional statements
    /// @note   This overrides the default traversal to handle
    ///         branching between different code paths
    bool traverse(const ast::ConditionalStatement* cond)
    {
        if (!cond) return true;
        if (!this->visit(cond)) return false;
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

protected:

    FunctionGroup::Ptr getFunction(const std::string& identifier,
            const bool allowInternal = false);

    llvm::Value* binaryExpression(llvm::Value* lhs, llvm::Value* rhs,
        const ast::tokens::OperatorToken op);
    void assignExpression(llvm::Value* lhs, llvm::Value*& rhs);

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

    // Warnings that are generated during code generation
    std::vector<std::string>* const mWarnings;

    // The function used as the base code block
    llvm::Function* mFunction;

    const FunctionOptions mOptions;

private:
    FunctionRegistry& mFunctionRegistry;
};

}
}
}
}

#endif // OPENVDB_AX_COMPUTE_GENERATOR_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
