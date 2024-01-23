// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/AST.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Provides the definition for every abstract and concrete derived
///   class which represent a particular abstract syntax tree (AST) node
///   type.
///
///   AST nodes represents a particular branch of a complete AST. Concrete
///   nodes can be thought of as leaf node types which hold semantic
///   information of a partial or complete statement or expression. A
///   string of AX can be fully represented by building the correct
///   AST structure. The AX grammar defined in axparser.y represents the
///   valid mapping of a tokenized string to AST nodes.
///
///   AST node classes can either represent a "leaf-level" semantic
///   component of a given AX AST, or an abstract base type. The latter are
///   used by the parser and leaf-level AST nodes for storage of compatible
///   child nodes, and provide grouping of various nodes which share common
///   semantics. The main two types of abstract AST nodes are statements
///   and expressions.
///

#ifndef OPENVDB_AX_AST_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_HAS_BEEN_INCLUDED

#include "Tokens.h"

#include <openvdb/version.h>
#include <openvdb/util/Assert.h>

#include <memory>
#include <utility>
#include <vector>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

/// @brief  Forward declaration of the base Abstract Syntax Tree type.
/// @note   Not to be confused with ast::Node types, which are the base abstract
///         type for all AST nodes. Tree nodes are the highest possible concrete
///         node type (in terms of hierarchy) which represent a full AX file.
///         They are always returned from the parser.
struct Tree;

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// @details  A reference list of all abstract and concrete AST nodes in
///           hierarchical order (non-linear)
///  Abstract nodes:
///  - Node
///  - Statement
///  - Expression
///  - Variable
///  - ValueBase
///
/// Concrete nodes:
///  - Tree
///  - StatementList
///  - Block
///  - Loop
///  - Keyword
///  - ConditionalStatement
///  - CommaOperator
///  - BinaryOperator
///  - TernaryOperator
///  - AssignExpression
///  - Crement
///  - UnaryOperator
///  - Cast
///  - FunctionCall
///  - ArrayUnpack
///  - ArrayPack
///  - Attribute
///  - ExternalVariable
///  - DeclareLocal
///  - Local
///  - Value<double/float/int32_t/int16_t/int64_t/bool>
///  - Value<std::string>

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// @brief  The base abstract node which determines the interface and required
///         methods for all derived concrete nodes which comprise a valid AST.
/// @note   All AST nodes share a few common characteristics. All constructors
///         typically take pointers to the abstract (pure-virtual) node types
///         and assume ownership of this data on successful construction. Deep
///         copy methods propagate down through all children of a given AST node
///         but have the unique behavior of ensuring parent data is updated to
///         the newly created parent nodes. Due to this behavior and the fact
///         that most nodes store unique pointers to other nodes, we've omitted
///         comparison and equality operators.
struct Node
{
    using Ptr = std::shared_ptr<Node>;
    using UniquePtr = std::unique_ptr<Node>;

    /// @brief  An enumerated list of node types for all concrete node types.
    ///         These can be used for faster evaluation of a given concrete node
    ///         using the virtual function table via Node::nodetype() rather
    ///         than performing a dynamic_cast/calling Node::isType.
    /// @note   This is sometimes referred to as "manual RTTI". We use this
    ///         technique combine with single dispatch due to opting for CRTP on
    ///         the main visitor and no templated virtual method support in C++.
    ///         i.e. no way to double dispatch: visit<template T>(Visitor<T>*)
    /// @note   Abstract (pure-virtual) nodes are not listed here. Node::isType
    ///         should be used to determine if a node is of a given abstract
    ///         type.
    enum NodeType {
        TreeNode,
        StatementListNode,
        BlockNode,
        ConditionalStatementNode,
        CommaOperatorNode,
        LoopNode,
        KeywordNode,
        AssignExpressionNode,
        CrementNode,
        UnaryOperatorNode,
        BinaryOperatorNode,
        TernaryOperatorNode,
        CastNode,
        AttributeNode,
        FunctionCallNode,
        ExternalVariableNode,
        DeclareLocalNode,
        ArrayPackNode,
        ArrayUnpackNode,
        LocalNode,
        ValueBoolNode,
        ValueInt16Node,
        ValueInt32Node,
        ValueInt64Node,
        ValueFloatNode,
        ValueDoubleNode,
        ValueStrNode
    };

    Node() = default;
    virtual ~Node() = default;

    /// @brief  The deep copy method for a Node
    /// @return A deep copy of the current node and all its children
    virtual Node* copy() const = 0;

    /// @name Name/Type
    /// @{

    /// @brief  Virtual method for accessing node type information
    /// @note   This method should be used when querying a concrete nodes type.
    /// @return Returns the enumerated node type from the NodeType list
    virtual NodeType nodetype() const = 0;

    /// @brief  Virtual method for accessing node name information
    /// @return Returns the node class name
    virtual const char* nodename() const = 0;

    /// @brief  Virtual method for accessing node name information
    /// @return Returns the short node class name
    virtual const char* subname() const = 0;

    /// @brief  Virtual method for accessing a node's base class. Note that if
    ///         this is called explicitly on an instance of ast::Node (the top
    ///         most base class) a nullptr is returned. This is primarily used
    ///         by the Visitor to support hierarchical visits.
    /// @return Returns the current node as its base class type.
    virtual const Node* basetype() const { return nullptr; }

    /// @brief  Query whether or not this node is of a specific (derived) type.
    ///         This method should be used to check if a node is of a particular
    ///         abstract type. When checking concrete types, it's generally
    ///         more efficient to check the return value of Node::nodetype()
    /// @tparam NodeT The node type to query against.
    /// @return True if this node is of the given type, false otherwise.
    template <typename NodeT>
    inline bool isType() const {
        return dynamic_cast<const NodeT*>(this);
    }

    /// @}

    /// @name Child Queries
    /// @{

    /// @brief  Virtual method for accessing child information. Returns the
    ///         number of children a given AST node owns.
    /// @return The number of children this node owns.
    virtual size_t children() const = 0;

    /// @brief  Virtual method for accessing child information. Returns a const
    ///         pointer to a child node at the given index. If the index is out
    ///         of range, a nullptr is returned.
    /// @note   This may still return a nullptr even if the given index is valid
    ///         if the child node has not been created.
    /// @param  index  The child index to query
    /// @return A Pointer to the child node, or a nullptr if none exists.
    virtual const Node* child(const size_t index) const = 0;

    /// @brief  Returns the child index of this node in relation to its parent,
    ///         or -1 if no valid index is found (usually representing the top
    ///         most node (i.e. Tree)
    /// @return The child index of this node
    inline int64_t childidx() const
    {
        const Node* p = this->parent();
        if (!p) return -1;
        size_t i = 0;
        const size_t count = p->children();
        for (; i < count; ++i) {
            if (p->child(i) == this) break;
        }
        if (i == count) return -1;
        return static_cast<int64_t>(i);
    }

    /// @}

    /// @name Replacement
    /// @{

    /// @brief  In place replacement. Attempts to replace this node at its
    ///         specific location within its Abstract Syntax Tree. On a
    ///         successful replacement, this node is destroyed, the provided
    ///         node is inserted in its place and ownership is transferred to the
    ///         parent node. No further calls to this node can be made on
    ///         successful replacements.
    /// @note   A replacement will fail if this node is the top most node within
    ///         an AST hierarchy or if the provided node type is not a
    ///         compatible type for the required abstract storage. For example,
    ///         if this node is an Attribute being held on a BinaryOperator,
    ///         only concrete nodes derived from an Expression can be used as a
    ///         replacement.
    /// @note   This method will dynamic_cast the provided node to check to see
    ///         if it's a compatible type.
    /// @param  node  The node to insert on a successful replacement.
    /// @return True if the replacement was successful, resulting in destruction
    ///         of this class and ownership transferal of the provided node.
    ///         False otherwise, where this and the provided node are unchanged.
    inline bool replace(Node* node)
    {
        const int64_t idx = this->childidx();
        if (idx == -1) return false; // avoid second vcall
        return this->parent()->replacechild(idx, node);
    }

    /// @brief  Virtual method that attempted to replace a child at a given
    ///         index with a provided node type.
    /// @note   See Node::replace for a more detailed description
    /// @param  index  The child index where a replacement should be attempted
    /// @param  node   The node to insert on a successful replacement.
    /// @return True if the replacement was successful, false otherwise
    inline virtual bool replacechild(const size_t index, Node* node);

    /// @}

    /// @name Parent
    /// @{

    /// @brief  Access a const pointer to this nodes parent
    /// @note   Can be a nullptr if this is the top most node in an AST (usually
    ///         a Tree)
    /// @return A const pointer to this node's parent node
    inline const Node* parent() const { return mParent; }

    /// @brief  Set this node's parent. This is used during construction of an
    ///         AST and should not be used. @todo Make this private.
    /// @param  parent  The parent to set
    inline void setParent(Node* parent) {
#ifndef NDEBUG
        bool hasChild = false;
        for (size_t i = 0; i < parent->children(); ++i)
            hasChild |= parent->child(i) == this;
        OPENVDB_ASSERT(hasChild);
#endif
        mParent = parent;
    }

private:
    /// @brief  Access a non const pointer to this nodes parent. Used by
    ///         replacement methods.
    /// @note   Can be a nullptr if this is the top most node in an AST (usually
    ///         a Tree)
    /// @return A non-const pointer to this nodes parent node
    inline Node* parent() { return mParent; }

    /// @}

    Node* mParent = nullptr;
};

inline bool Node::replacechild(const size_t, Node*) { return false; }


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// Abstract (pure-virtual) AST nodes

/// @brief Statements are anything that can make up a line, i.e. everything
///        in between semicolons. Likewise to their base ast::Node class,
///        currently every concrete AST node is either directly or indirectly
///        a derived statement type. They hold no class data.
struct Statement : public Node
{
    using UniquePtr = std::unique_ptr<Statement>;
    ~Statement() override = default;
    virtual Statement* copy() const override = 0;
    const Node* basetype() const override { return this; }
};

/// @brief Expressions are comprised of full or potentially partial parts of a
///        full statement that may not necessary make up an entire valid
///        statement on their own. For example, while a Binary Operator such as
///        "3 + 5;"" is a valid statement on its own, the full statement
///        "3 + 5 + 6;" must be broken down into two expressions which together
///        form the statement as well as determining precedence.
struct Expression : public Statement
{
    using UniquePtr = std::unique_ptr<Expression>;
    ~Expression() override = default;
    virtual Expression* copy() const override = 0;
    const Statement* basetype() const override { return this; }
};

/// @brief Variables are a base type for Locals, Attributes and
///        ExternalVariables. Unlike other abstract types, they also consolidate
///        data for the derived types.
struct Variable : public Expression
{
    using UniquePtr = std::unique_ptr<Variable>;

    Variable(const std::string& name)
        : Expression(), mName(name) {}
    Variable(const Variable& other)
        : Expression(), mName(other.mName) {}
    ~Variable() override = default;

    virtual Variable* copy() const override = 0;
    const Expression* basetype() const override { return this; }
    //
    size_t children() const override { return 0; }
    const Node* child(const size_t) const override { return nullptr; }
    //
    inline const std::string& name() const { return mName; }

private:
    const std::string mName;
};

/// @brief ValueBases are a base class for anything that holds a value (literal).
/// Derived classes store the actual typed values
struct ValueBase : public Expression
{
    using UniquePtr = std::unique_ptr<ValueBase>;
    ~ValueBase() override = default;
    virtual Expression* copy() const override = 0;
    const Expression* basetype() const override { return this; }
    //
    size_t children() const override { return 0; }
    const Node* child(const size_t) const override { return nullptr; }
};


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

/// Concrete AST nodes

/// @brief  A StatementList is derived from a Statement and comprises of
///         combinations of multiple statements. This could represent either
///         a list of statements of different types but in practice will likely
///         represent a ',' separated list of the same type i.e.
///         'int i = 1, j = 1;'.
/// @note   Statements held by the list are guaranteed to be valid (non null).
///         nullptrs added to the list are implicitly dropped.
/// @todo   Consider combination with Block
struct StatementList : public Statement
{
    using UniquePtr = std::unique_ptr<StatementList>;

    /// @brief Construct a new StatementList with an empty list
    StatementList() : mList() {}
    /// @brief  Construct a new StatementList with a single statement,
    ///         transferring ownership of the statement to the statement list
    ///         and updating parent data on the statement. If the statement is a
    ///         nullptr, it is ignored.
    /// @param  statement  The statement to construct from
    StatementList(Statement* statement)
        : mList() {
        this->addStatement(statement);
    }
    /// @brief Construct a new StatementList from a vector of statements,
    ///        transferring ownership of all valid statements to the statement
    ///        list and updating parent data on the statement. Only valid (non
    ///        null) statements are added to the statement list.
    /// @param  statements  The vector of statements to construct from
    StatementList(const std::vector<Statement*>& statements)
        : mList() {
        for (Statement* statement : statements) {
            this->addStatement(statement);
        }
    }
    /// @brief  Deep copy constructor for a StatementList, performing a deep
    ///         copy on every held statement, ensuring parent information is
    ///         updated.
    /// @param  other  A const reference to another statement list to deep copy
    StatementList(const StatementList& other) : mList() {
        for (const Statement::UniquePtr& stmnt : other.mList) {
            this->addStatement(stmnt->copy());
        }
    }
    ~StatementList() override = default;

    /// @copybrief Node::copy()
    StatementList* copy() const override { return new StatementList(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::StatementListNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "statement list"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "stml"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return this->size(); }
    /// @copybrief Node::child()
    const Statement* child(const size_t i) const override final {
        if (i >= mList.size()) return nullptr;
        return mList[i].get();
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (mList.size() <= i) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        mList[i].reset(expr);
        mList[i]->setParent(this);
        return true;
    }

    /// @brief  Alias for StatementList::children
    inline size_t size() const { return mList.size(); }
    /// @brief  Adds a statement to this statement list, transferring ownership to the
    ///         statement list and updating parent data on the statement. If the
    ///         statement is a nullptr, it is ignored.
    inline void addStatement(Statement* stmnt) {
        if (stmnt) {
            mList.emplace_back(stmnt);
            stmnt->setParent(this);
        }
    }
private:
    std::vector<Statement::UniquePtr> mList;
};

/// @brief  A Block node represents a scoped list of statements. It may comprise
///         of 0 or more statements, and specifically indicates that a new scope
///         is activated, typically represented by curly braces. Note that a
///         block does not alway have to be encapsulated by curly braces, but
///         always represents a new scope.
/// @note   Statements held by the block are guaranteed to be valid (non null).
///         nullptrs added to the block are implicitly dropped.
/// @note   While closely linked, it's important to differentiate between this
///         class and an llvm::BasicBlock.
/// @todo   Consider combination with StatementList
struct Block : public Statement
{
    using UniquePtr = std::unique_ptr<Block>;

    /// @brief Construct a new Block with an empty list
    Block() : mList() {}
    /// @brief  Construct a new Block with a single statement, transferring
    ///         ownership of the statement to the block and updating parent
    ///         data on the statement. If the statement is a nullptr, it is
    ///         ignored.
    /// @param  statement  The statement to construct from
    Block(Statement* statement)
        : mList() {
        this->addStatement(statement);
    }
    /// @brief Construct a new Block from a vector of statements, transferring
    ///        ownership of all valid statements to the block and updating
    ///        parent data on the statement. Only valid (non null) statements
    ///        are added to the block.
    /// @param  statements  The vector of statements to construct from
    Block(const std::vector<Statement*>& statements)
        : mList() {
        for (Statement* statement : statements) {
            this->addStatement(statement);
        }
    }
    /// @brief  Deep copy constructor for a Block, performing a deep copy on
    ///         every held statement, ensuring parent information is updated.
    /// @param  other  A const reference to another block to deep copy
    Block(const Block& other) : mList() {
        for (const Statement::UniquePtr& stmnt : other.mList) {
            this->addStatement(stmnt->copy());
        }
    }
    ~Block() override = default;

    /// @copybrief Node::copy()
    Block* copy() const override final { return new Block(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::BlockNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "scoped block"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "blk"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return this->size(); }
    /// @copybrief Node::child()
    const Statement* child(const size_t i) const override final {
        if (i >= mList.size()) return nullptr;
        return mList[i].get();
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (mList.size() <= i) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        mList[i].reset(expr);
        mList[i]->setParent(this);
        return true;
    }

    /// @brief  Alias for Block::children
    inline size_t size() const { return mList.size(); }
    /// @brief  Adds a statement to this block, transferring ownership to the
    ///         block and updating parent data on the statement. If the
    ///         statement is a nullptr, it is ignored.
    inline void addStatement(Statement* stmnt) {
        if (stmnt) {
            mList.emplace_back(stmnt);
            stmnt->setParent(this);
        }
    }
private:
    std::vector<Statement::UniquePtr> mList;
};

/// @brief  A Tree is the highest concrete (non-abstract) node in the entire AX
///         AST hierarchy. It represents an entire conversion of a valid AX
///         string.
/// @note   A tree is the only node type which has typedefs for use as a shared
///         pointer. All other nodes are expected to be handled through unique
///         pointers to infer ownership.
/// @todo   Replace block with StatementList
struct Tree : public Node
{
    using Ptr = std::shared_ptr<Tree>;
    using ConstPtr = std::shared_ptr<const Tree>;
    using UniquePtr = std::unique_ptr<Tree>;

    /// @brief  Construct a new Tree from a given Block, transferring ownership
    ///         of the Block to the tree and updating parent data on the Block.
    /// @note   The provided Block must be a valid pointer (non-null)
    /// @param  block  The Block to construct from
    Tree(Block* block = new Block())
        : mBlock(block) {
            mBlock->setParent(this);
    }
    /// @brief  Deep copy constructor for a Tree, performing a deep copy on
    ///         the held Block, ensuring parent information is updated.
    /// @param  other  A const reference to another Tree to deep copy
    Tree(const Tree& other)
        : mBlock(new Block(*other.mBlock)) {
            mBlock->setParent(this);
        }
    ~Tree() override = default;

    /// @copybrief Node::copy()
    Tree* copy() const override final { return new Tree(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::TreeNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "tree"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "tree"; }
    /// @copybrief Node::basetype()
    const Node* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return 1; }
    /// @copybrief Node::child()
    const Block* child(const size_t i) const override final {
        if (i == 0) return mBlock.get();
        return nullptr;
    }
private:
    Block::UniquePtr mBlock;
};

struct CommaOperator : public Expression
{
    using UniquePtr = std::unique_ptr<CommaOperator>;

    /// @brief  Construct a new CommaOperator with an expr set
    CommaOperator() : mExpressions() {}
    /// @brief  Construct a new CommaOperator with a single expression,
    ///         transferring ownership of the expression to the CommaOperator
    ///         and updating parent data on the expression. If the expression is
    ///         a nullptr, it is ignored.
    /// @param  expression  The Expression to construct from
    CommaOperator(Expression* expression)
        : mExpressions() {
            this->append(expression);
        }
    /// @brief Construct a new CommaOperator from a vector of expression,
    ///        transferring ownership of all valid expression to the
    ///        CommaOperator and updating parent data on the statement. Only
    ///        valid (non null) expression are added to the block.
    /// @param  expressions  The vector of expressions to construct from
    CommaOperator(const std::vector<Expression*>& expressions)
        : mExpressions() {
            mExpressions.reserve(expressions.size());
            for (Expression* expression : expressions) {
                this->append(expression);
            }
        }
    /// @brief  Deep copy constructor for an CommaOperator, performing a deep
    ///         copy on every held expression, ensuring parent information is
    ///         updated.
    /// @param  other  A const reference to another CommaOperator to deep copy
    CommaOperator(const CommaOperator& other)
        : mExpressions() {
            mExpressions.reserve(other.mExpressions.size());
            for (const Expression::UniquePtr& expr : other.mExpressions) {
                this->append(expr->copy());
            }
        }
    ~CommaOperator() override = default;

    /// @copybrief Node::copy()
    CommaOperator* copy() const override final {
        return new CommaOperator(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::CommaOperatorNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "comma"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "comma"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return this->size(); }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i >= mExpressions.size()) return nullptr;
        return mExpressions[i].get();
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (mExpressions.size() <= i) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        mExpressions[i].reset(expr);
        mExpressions[i]->setParent(this);
        return true;
    }

    /// @brief  Alias for CommaOperator::children
    inline size_t size() const { return mExpressions.size(); }
    /// @brief  Query whether this Expression list holds any valid expressions
    /// @return True if this node if empty, false otherwise
    inline bool empty() const { return mExpressions.empty(); }
    /// @brief  Append an expression to this CommaOperator, transferring
    ///         ownership to the CommaOperator and updating parent data on the
    ///         expression. If the expression is a nullptr, it is ignored.
    inline void append(Expression* expr) {
        if (expr) {
            mExpressions.emplace_back(expr);
            expr->setParent(this);
        }
    }
private:
    std::vector<Expression::UniquePtr> mExpressions;
};

/// @brief  Loops represent for, while and do-while loop constructs.
///         These all consist of a condition - evaluated to determine if loop
///         iteration should continue, and a body which is the logic to be
///         repeated. For loops also have initial statements which are evaluated
///         prior to loop execution (at loop scope) and commonly used to
///         set up iterators, and iteration expressions which are evaluated
///         between iterations after the body and before the condition.
///         Both conditions and initial statements can be declarations or
///         expressions, so are Statements, and iteration expressions can
///         consist of multiple expressions. The loop body is a Block defining
///         its own scope (encapsulated by initial statement scope for for-loops).
/// @note   Only for-loops should have initial statements and/or iteration
///         expressions. Also for-loops allow empty conditions to be given by
///         the user, this is replaced with a 'true' expression in the parser.
struct Loop : public Statement
{
    using UniquePtr = std::unique_ptr<Loop>;

    /// @brief  Construct a new Loop with the type defined by a
    ///         tokens::LoopToken, a condition Statement, a Block representing
    ///         the body and for for-loops an optional initial Statement and
    ///         iteration Expression. Ownership of all arguments is
    ///         transferred to the Loop. All arguments have their parent data
    ///         updated.
    /// @param  loopType   The type of loop - for, while or do-while.
    /// @param  condition  The condition Statement to determine loop repetition
    /// @param  body       The Block to be repeated
    /// @param  init       The (optional) for-loop initial Statement.
    /// @param  iter       The (optional) for-loop iteration Expression.
    Loop(const tokens::LoopToken loopType,
         Statement* condition,
         Block* body,
         Statement* init = nullptr,
         Expression* iter = nullptr)
        : mLoopType(loopType)
        , mConditional(condition)
        , mBody(body)
        , mInitial(init)
        , mIteration(iter) {
            OPENVDB_ASSERT(mConditional);
            OPENVDB_ASSERT(mBody);
            mConditional->setParent(this);
            mBody->setParent(this);
            if (mInitial) {
                OPENVDB_ASSERT(mLoopType == tokens::LoopToken::FOR);
                mInitial->setParent(this);
            }
            if (mIteration) {
                OPENVDB_ASSERT(mLoopType == tokens::LoopToken::FOR);
                 mIteration->setParent(this);
            }
        }
    /// @brief  Deep copy constructor for an Loop, performing a deep copy on the
    ///         condition, body and initial Statement/iteration Expression
    ///         if they exist, ensuring parent information is updated.
    /// @param  other  A const reference to another Loop to deep copy
    Loop(const Loop& other)
        : mLoopType(other.mLoopType)
        , mConditional(other.mConditional->copy())
        , mBody(other.mBody->copy())
        , mInitial(other.hasInit() ? other.mInitial->copy() : nullptr)
        , mIteration(other.hasIter() ? other.mIteration->copy() : nullptr) {
            mConditional->setParent(this);
            mBody->setParent(this);
            if (mInitial) {
                OPENVDB_ASSERT(mLoopType == tokens::LoopToken::FOR);
                mInitial->setParent(this);
            }
            if (mIteration) {
                OPENVDB_ASSERT(mLoopType == tokens::LoopToken::FOR);
                 mIteration->setParent(this);
            }
        }
    ~Loop() override = default;

    /// @copybrief Node::copy()
    Loop* copy() const override final { return new Loop(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::LoopNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "loop"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "loop"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return 4; }
    /// @copybrief Node::child()
    const Statement* child(const size_t i) const override final {
        if (i == 0) return mConditional.get();
        if (i == 1) return mBody.get();
        if (i == 2) return mInitial.get();
        if (i == 3) return mIteration.get();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final
    {
        if (i == 0 || i == 2) {
            Statement* stmt = dynamic_cast<Statement*>(node);
            if (!stmt) return false;
            if (i == 0) {
                mConditional.reset(stmt);
                mConditional->setParent(this);
            }
            else {
                mInitial.reset(stmt);
                mInitial->setParent(this);
            }
            return true;
        }
        else if (i == 1) {
            Block* blk = dynamic_cast<Block*>(node);
            if (!blk) return false;
            mBody.reset(blk);
            mBody->setParent(this);
            return true;
        }
        else if (i == 3) {
            Expression* expr = dynamic_cast<Expression*>(node);
            if (!expr) return false;
            mIteration.reset(expr);
            mIteration->setParent(expr);
            return true;
        }
        return false;
    }

    /// @brief  Query the type of loop held on this node.
    /// @return The loop type as a tokens::LoopToken
    inline tokens::LoopToken loopType() const { return mLoopType; }
    /// @brief  Query if this Loop has a valid initial statement
    /// @return True if a valid initial statement exists, false otherwise
    inline bool hasInit() const { return static_cast<bool>(this->initial()); }
    /// @brief  Query if this Loop has a valid iteration expression list
    /// @return True if a valid iteration list exists, false otherwise
    inline bool hasIter() const { return static_cast<bool>(this->iteration()); }
    /// @brief  Access a const pointer to the Loop condition as an abstract
    ///         statement.
    /// @return A const pointer to the condition as a statement
    const Statement* condition() const { return mConditional.get(); }
    /// @brief  Access a const pointer to the Loop body as a Block.
    /// @return A const pointer to the body Block
    const Block* body() const { return mBody.get(); }
    /// @brief  Access a const pointer to the Loop initial statement as an
    ///         abstract statement.
    /// @return A const pointer to the initial statement as a statement
    const Statement* initial() const { return mInitial.get(); }
    /// @brief  Access a const pointer to the Loop iteration Expression
    /// @return A const pointer to the iteration Expression
    const Expression* iteration() const { return mIteration.get(); }

private:
    const tokens::LoopToken mLoopType;
    Statement::UniquePtr    mConditional;
    Block::UniquePtr        mBody;
    Statement::UniquePtr    mInitial;
    Expression::UniquePtr   mIteration;
};

/// @brief  ConditionalStatements represents all combinations of 'if', 'else'
///         and 'else if' syntax and semantics. A single ConditionalStatement
///         only ever represents up to two branches; an 'if' (true) and an
///         optional 'else' (false). ConditionalStatements are nested within
///         the second 'else' branch to support 'else if' logic. As well as both
///         'if' and 'else' branches, a ConditionalStatement also holds an
///         Expression related to its primary condition.
/// @note   The first 'if' branch is referred to as the 'true' branch. The
///         second 'else' branch is referred to as the 'false' branch.
struct ConditionalStatement : public Statement
{
    using UniquePtr = std::unique_ptr<ConditionalStatement>;

    /// @brief  Construct a new ConditionalStatement with an Expression
    ///         representing the primary condition, a Block representing the
    ///         'true' branch and an optional Block representing the 'false'
    ///         branch. Ownership of all arguments is transferred to the
    ///         ConditionalStatement. All arguments have their parent data
    ///         updated.
    /// @param  conditional The Expression to construct the condition from
    /// @param  trueBlock   The Block to construct the true branch from
    /// @param  falseBlock  The (optional) Block to construct the false branch
    ///                     from
    ConditionalStatement(Expression* conditional,
                         Block* trueBlock,
                         Block* falseBlock = nullptr)
        : mConditional(conditional)
        , mTrueBranch(trueBlock)
        , mFalseBranch(falseBlock) {
            OPENVDB_ASSERT(mConditional);
            OPENVDB_ASSERT(mTrueBranch);
            mConditional->setParent(this);
            mTrueBranch->setParent(this);
            if (mFalseBranch) mFalseBranch->setParent(this);
        }
    /// @brief  Deep copy constructor for an ConditionalStatement, performing a
    ///         deep copy on the condition and both held branches (Blocks),
    ///         ensuring parent information is updated.
    /// @param  other  A const reference to another ConditionalStatement to deep
    ///         copy
    ConditionalStatement(const ConditionalStatement& other)
        : mConditional(other.mConditional->copy())
        , mTrueBranch(other.mTrueBranch->copy())
        , mFalseBranch(other.hasFalse() ? other.mFalseBranch->copy() : nullptr) {
            mConditional->setParent(this);
            mTrueBranch->setParent(this);
            if (mFalseBranch) mFalseBranch->setParent(this);
        }
    ~ConditionalStatement() override = default;

    /// @copybrief Node::copy()
    ConditionalStatement* copy() const override final {
        return new ConditionalStatement(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::ConditionalStatementNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "conditional statement"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "cond"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }

    /// @copybrief Node::children()
    size_t children() const override final { return 3; }
    /// @copybrief Node::child()
    const Statement* child(const size_t i) const override final {
        if (i == 0) return this->condition();
        if (i == 1) return this->trueBranch();
        if (i == 2) return this->falseBranch();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final
    {
        if (i == 0) {
            Expression* expr = dynamic_cast<Expression*>(node);
            if (!expr) return false;
            mConditional.reset(expr);
            mConditional->setParent(this);
            return true;
        }
        else if (i == 1 || i == 2) {
            Block* blk = dynamic_cast<Block*>(node);
            if (!blk) return false;
            if (i == 1) {
                mTrueBranch.reset(blk);
                mTrueBranch->setParent(this);
            }
            else {
                mFalseBranch.reset(blk);
                mFalseBranch->setParent(this);
            }
            return true;
        }
        return false;
    }

    /// @brief  Query if this ConditionalStatement has a valid 'false' branch
    /// @return True if a valid 'false' branch exists, false otherwise
    inline bool hasFalse() const {
        return static_cast<bool>(this->falseBranch());
    }
    /// @brief  Query the number of branches held by this ConditionalStatement.
    ///         This is only ever 1 or 2.
    /// @return 2 if a valid 'true' and 'false' branch exist, 1 otherwise
    size_t branchCount() const {
        return this->hasFalse() ? 2 : 1;
    }
    /// @brief  Access a const pointer to the ConditionalStatements condition
    ///         as an abstract expression.
    /// @return A const pointer to the condition as an expression
    const Expression* condition() const { return mConditional.get(); }
    /// @brief  Access a const pointer to the ConditionalStatements 'true'
    ///         branch as a Block
    /// @return A const pointer to the 'true' branch
    const Block* trueBranch() const { return mTrueBranch.get(); }
    /// @brief  Access a const pointer to the ConditionalStatements 'false'
    ///         branch as a Block
    /// @return A const pointer to the 'false' branch
    const Block* falseBranch() const { return mFalseBranch.get(); }
private:
    Expression::UniquePtr mConditional;
    Block::UniquePtr mTrueBranch;
    Block::UniquePtr mFalseBranch;
};

/// @brief  A BinaryOperator represents a single binary operation between a
///         left hand side (LHS) and right hand side (RHS) expression. The
///         operation type is stored as a tokens::OperatorToken enumerated type
///         on the node. AX grammar guarantees that this token will only ever
///         be a valid binary operator token type when initialized by the
///         parser.
struct BinaryOperator : public Expression
{
    using UniquePtr = std::unique_ptr<BinaryOperator>;

    /// @brief  Construct a new BinaryOperator with a given
    ///         tokens::OperatorToken and a valid LHS and RHS expression,
    ///         transferring ownership of the expressions to the BinaryOperator
    ///         and updating parent data on the expressions.
    /// @param  left   The left hand side of the binary expression
    /// @param  right  The right hand side of the binary expression
    /// @param  op     The binary token representing the operation to perform.
    ///                Should not be an assignment token.
    BinaryOperator(Expression* left,
            Expression* right,
            const tokens::OperatorToken op)
        : mLeft(left)
        , mRight(right)
        , mOperation(op) {
            OPENVDB_ASSERT(mLeft);
            OPENVDB_ASSERT(mRight);
            mLeft->setParent(this);
            mRight->setParent(this);
        }
    /// @brief Construct a new BinaryOperator with a string, delegating
    ///        construction to the above BinaryOperator constructor.
    /// @param  left   The left hand side of the binary expression
    /// @param  right  The right hand side of the binary expression
    /// @param  op     A string representing the binary operation to perform
    BinaryOperator(Expression* left,
            Expression* right,
            const std::string& op)
        : BinaryOperator(left, right, tokens::operatorTokenFromName(op)) {}
    /// @brief  Deep copy constructor for a BinaryOperator, performing a
    ///         deep copy on both held expressions, ensuring parent information
    ///         is updated.
    /// @param  other  A const reference to another BinaryOperator to deep copy
    BinaryOperator(const BinaryOperator& other)
        : mLeft(other.mLeft->copy())
        , mRight(other.mRight->copy())
        , mOperation(other.mOperation) {
            mLeft->setParent(this);
            mRight->setParent(this);
        }
    ~BinaryOperator() override = default;

    /// @copybrief Node::copy()
    BinaryOperator* copy() const override final {
        return new BinaryOperator(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::BinaryOperatorNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "binary"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "bin"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 2; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return mLeft.get();
        if (i == 1) return mRight.get();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i > 1) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        if (i == 0) {
            mLeft.reset(expr);
            mLeft->setParent(this);
        }
        else if (i == 1) {
            mRight.reset(expr);
            mRight->setParent(this);
        }
        return true;
    }

    /// @brief  Query the type of binary operation held on this node.
    /// @return The binary operation as a tokens::OperatorToken
    inline tokens::OperatorToken operation() const { return mOperation; }
    /// @brief  Access a const pointer to the BinaryOperator LHS as an abstract
    ///         expression
    /// @return A const pointer to the LHS expression
    const Expression* lhs() const { return mLeft.get(); }
    /// @brief  Access a const pointer to the BinaryOperator RHS as an abstract
    ///         expression
    /// @return A const pointer to the RHS expression
    const Expression* rhs() const { return mRight.get(); }
private:
    Expression::UniquePtr mLeft;
    Expression::UniquePtr mRight;
    const tokens::OperatorToken mOperation;
};

/// @brief  A TernaryOperator represents a ternary (conditional) expression
///         'a ? b : c' which evaluates to 'b' if 'a' is true and 'c' if 'a' is false.
///         Requires 'b' and 'c' to be convertibly typed expressions, or both void.
///         The 'true' expression ('b') is optional with the conditional expression 'a'
///         returned if it evaluates to true, otherwise returning 'c'. Note that 'a'
///         will only be evaluated once in this case.
struct TernaryOperator : public Expression
{
    using UniquePtr = std::unique_ptr<TernaryOperator>;

    /// @brief  Construct a new TernaryOperator with a conditional expression
    ///         and true (optional) and false expressions, transferring
    ///         ownership of the expressions to the TernaryOperator
    ///         and updating parent data on the expressions.
    /// @param  conditional      The conditional expression determining the expression
    ///                          selection
    /// @param  trueExpression   The (optional) expression evaluated if the condition
    ///                          is true
    /// @param  falseExpression  The expression evaluated if the condition is false
    TernaryOperator(Expression* conditional,
                    Expression* trueExpression,
                    Expression* falseExpression)
        : mConditional(conditional)
        , mTrueBranch(trueExpression)
        , mFalseBranch(falseExpression) {
            OPENVDB_ASSERT(mConditional);
            OPENVDB_ASSERT(mFalseBranch);
            mConditional->setParent(this);
            if (mTrueBranch) mTrueBranch->setParent(this);
            mFalseBranch->setParent(this);
        }
    /// @brief  Deep copy constructor for a TernaryOperator, performing a
    ///         deep copy on held expressions, ensuring parent information
    ///         is updated.
    /// @param  other  A const reference to another TernaryOperator to deep copy
    TernaryOperator(const TernaryOperator& other)
        : mConditional(other.mConditional->copy())
        , mTrueBranch(other.hasTrue() ? other.mTrueBranch->copy() : nullptr)
        , mFalseBranch(other.mFalseBranch->copy()) {
            mConditional->setParent(this);
            if (mTrueBranch) mTrueBranch->setParent(this);
            mFalseBranch->setParent(this);
        }
    ~TernaryOperator() override = default;

    /// @copybrief Node::copy()
    TernaryOperator* copy() const override final {
        return new TernaryOperator(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::TernaryOperatorNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "ternary"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "tern"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 3; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return mConditional.get();
        if (i == 1) return mTrueBranch.get();
        if (i == 2) return mFalseBranch.get();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i > 2) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        if (i == 0) {
            mConditional.reset(expr);
            mConditional->setParent(this);
        }
        else if (i == 1) {
            mTrueBranch.reset(expr);
            mTrueBranch->setParent(this);
        }
        else if (i == 2) {
            mFalseBranch.reset(expr);
            mFalseBranch->setParent(this);
        }
        return true;
    }

    /// @brief  Query whether or not this has an optional if-true branch.
    bool hasTrue() const { return static_cast<bool>(this->trueBranch()); }
    /// @brief  Access a const pointer to the TernaryOperator conditional as
    ///         an abstract expression
    /// @return A const pointer to the conditional expression
    const Expression* condition() const { return mConditional.get(); }
    /// @brief  Access a const pointer to the TernaryOperator true expression as
    ///         an abstract expression
    /// @return A const pointer to the true expression
    const Expression* trueBranch() const { return mTrueBranch.get(); }
    /// @brief  Access a const pointer to the TernaryOperator false expression as
    ///         an abstract expression
    /// @return A const pointer to the false expression
    const Expression* falseBranch() const { return mFalseBranch.get(); }
private:
    Expression::UniquePtr mConditional;
    Expression::UniquePtr mTrueBranch;
    Expression::UniquePtr mFalseBranch;
};

/// @brief  AssignExpressions represents a similar object construction to a
///         BinaryOperator. AssignExpressions can be chained together and are
///         thus derived as Expressions rather than Statements.
/// @note   AssignExpressions can either be direct or compound assignments. The
///         latter is represented by the last argument in the primary
///         constructor which is expected to be a valid binary token.
struct AssignExpression : public Expression
{
    using UniquePtr = std::unique_ptr<AssignExpression>;

    /// @brief  Construct a new AssignExpression with valid LHS and RHS
    ///         expressions, transferring ownership of the expressions to the
    ///         AssignExpression and updating parent data on the expressions.
    /// @param  lhs  The left hand side of the assign expression
    /// @param  rhs  The right hand side of the assign expression
    /// @param  op   The compound assignment token, if any
    AssignExpression(Expression* lhs, Expression* rhs,
        const tokens::OperatorToken op = tokens::EQUALS)
        : mLHS(lhs)
        , mRHS(rhs)
        , mOperation(op) {
            OPENVDB_ASSERT(mLHS);
            OPENVDB_ASSERT(mRHS);
            mLHS->setParent(this);
            mRHS->setParent(this);
        }
    /// @brief  Deep copy constructor for an AssignExpression, performing a
    ///         deep copy on both held expressions, ensuring parent information
    ///         is updated.
    /// @param  other  A const reference to another AssignExpression to deep
    ///                copy
    AssignExpression(const AssignExpression& other)
        : mLHS(other.mLHS->copy())
        , mRHS(other.mRHS->copy())
        , mOperation(other.mOperation) {
            mLHS->setParent(this);
            mRHS->setParent(this);
        }
    ~AssignExpression() override = default;

    /// @copybrief Node::copy()
    AssignExpression* copy() const override final {
        return new AssignExpression(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::AssignExpressionNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "assignment expression"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "asgn"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 2; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return this->lhs();
        if (i == 1) return this->rhs();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i > 1) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        if (i == 0) {
            mLHS.reset(expr);
            mLHS->setParent(this);
        }
        else if (i == 1) {
            mRHS.reset(expr);
            mRHS->setParent(this);
        }
        return true;
    }

    /// @brief  Query whether or not this is a compound AssignExpression.
    ///         Compound AssignExpressions are assignments which read and write
    ///         to the LHS value. i.e. +=, -=, *= etc
    /// @return The binary operation as a tokens::OperatorToken
    inline bool isCompound() const { return mOperation != tokens::EQUALS; }
    /// @brief  Query the actual operational type of this AssignExpression. For
    ///         simple (non-compound) AssignExpressions, tokens::EQUALS is
    ///         returned.
    inline tokens::OperatorToken operation() const { return mOperation; }
    /// @brief  Access a const pointer to the AssignExpression LHS as an
    ///         abstract expression
    /// @return A const pointer to the LHS expression
    const Expression* lhs() const { return mLHS.get(); }
    /// @brief  Access a const pointer to the AssignExpression RHS as an
    ////        abstract expression
    /// @return A const pointer to the RHS expression
    const Expression* rhs() const { return mRHS.get(); }
private:
    Expression::UniquePtr mLHS;
    Expression::UniquePtr mRHS;
    const tokens::OperatorToken mOperation;
};

/// @brief  A Crement node represents a single increment '++' and decrement '--'
///         operation. As well as it's crement type, it also stores whether
///         the semantics constructed a post or pre-crement i.e. ++a or a++.
struct Crement : public Expression
{
    using UniquePtr = std::unique_ptr<Crement>;

    /// @brief  A simple enum representing the crement type.
    enum Operation {
        Increment,
        Decrement
    };

    /// @brief  Construct a new Crement with a valid expression, transferring
    ///         ownership of the expression to the Crement node and updating
    ///         parent data on the expression.
    /// @param  expr   The expression to crement
    /// @param  op     The type of crement operation; Increment or Decrement
    /// @param  post   True if the crement operation is a post crement i.e. a++,
    ///                false if the operation is a pre crement i.e. ++a
    Crement(Expression* expr, const Operation op, bool post)
        : mExpression(expr)
        , mOperation(op)
        , mPost(post) {
            mExpression->setParent(this);
        }
    /// @brief  Deep copy constructor for a Crement, performing a deep copy on
    ///         the underlying expressions, ensuring parent information is
    ///         updated.
    /// @param  other  A const reference to another Crement to deep copy
    Crement(const Crement& other)
        : mExpression(other.mExpression->copy())
        , mOperation(other.mOperation)
        , mPost(other.mPost) {
            mExpression->setParent(this);
        }
    ~Crement() override = default;

    /// @copybrief Node::copy()
    Crement* copy() const override final { return new Crement(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::CrementNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "crement"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "crmt"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    //
    /// @copybrief Node::children()
    size_t children() const override final { return 1; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return this->expression();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i != 0) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        mExpression.reset(expr);
        mExpression->setParent(this);
        return true;
    }

    /// @brief  Query the type of the Crement operation. This does not hold
    ///         post or pre-crement information.
    /// @return The Crement operation being performed. This is either an
    ///         Crement::Increment or Crement::Decrement.
    inline Operation operation() const { return mOperation; }
    /// @brief  Query if this Crement node represents an incrementation ++
    /// @return True if this node is performing an increment
    inline bool increment() const { return mOperation == Increment; }
    /// @brief  Query if this Crement node represents an decrement --
    /// @return True if this node is performing an decrement
    inline bool decrement() const { return mOperation == Decrement; }
    /// @brief  Query if this Crement node represents a pre crement ++a
    /// @return True if this node is performing a pre crement
    inline bool pre() const { return !mPost; }
    /// @brief  Query if this Crement node represents a post crement a++
    /// @return True if this node is performing a post crement
    inline bool post() const { return mPost; }
    /// @brief  Access a const pointer to the expression being crements as an
    ///         abstract Expression
    /// @return A const pointer to the expression
    const Expression* expression() const { return mExpression.get(); }
private:
    Expression::UniquePtr mExpression;
    const Operation mOperation;
    const bool mPost;
};

/// @brief  A UnaryOperator represents a single unary operation on an
///         expression. The operation type is stored as a tokens::OperatorToken
///         enumerated type on the node. AX grammar guarantees that this token
///         will only every be a valid unary operator token type when
///         initialized by the parser.
struct UnaryOperator : public Expression
{
    using UniquePtr = std::unique_ptr<UnaryOperator>;

    /// @brief  Construct a new UnaryOperator with a given tokens::OperatorToken
    ///         and a valid expression, transferring ownership of the expression
    ///         to the UnaryOperator and updating parent data on the expression.
    /// @param  expr  The expression to perform the unary operator on
    /// @param  op    The unary token representing the operation to perform.
    UnaryOperator(Expression* expr, const tokens::OperatorToken op)
        : mExpression(expr)
        , mOperation(op) {
            OPENVDB_ASSERT(mExpression);
            mExpression->setParent(this);
        }
    /// @brief Construct a new UnaryOperator with a string, delegating
    ///        construction to the above UnaryOperator constructor.
    /// @param  op    A string representing the unary operation to perform
    /// @param  expr  The expression to perform the unary operator on
    UnaryOperator(Expression* expr, const std::string& op)
        : UnaryOperator(expr, tokens::operatorTokenFromName(op)) {}
    /// @brief  Deep copy constructor for a UnaryOperator, performing a deep
    ///         copy on the underlying expressions, ensuring parent information
    ///         is updated.
    /// @param  other  A const reference to another UnaryOperator to deep copy
    UnaryOperator(const UnaryOperator& other)
        : mExpression(other.mExpression->copy())
        , mOperation(other.mOperation) {
            mExpression->setParent(this);
        }
    ~UnaryOperator() override = default;

    /// @copybrief Node::copy()
    UnaryOperator* copy() const override final { return new UnaryOperator(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::UnaryOperatorNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "unary"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "unry"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 1; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return this->expression();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i != 0) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        mExpression.reset(expr);
        mExpression->setParent(this);
        return true;
    }

    /// @brief  Query the type of unary operation held on this node.
    /// @return The unary operation as a tokens::OperatorToken
    inline tokens::OperatorToken operation() const { return mOperation; }
    /// @brief  Access a const pointer to the UnaryOperator expression as an
    ///         abstract expression
    /// @return A const pointer to the expression
    const Expression* expression() const { return mExpression.get(); }
private:
    Expression::UniquePtr mExpression;
    const tokens::OperatorToken mOperation;
};

/// @brief  Cast nodes represent the conversion of an underlying expression to
///         a target type. Cast nodes are typically constructed from functional
///         notation and do not represent construction of the target type,
///         rather a type-casted conversion.
struct Cast : public Expression
{
    using UniquePtr = std::unique_ptr<Cast>;

    /// @brief  Construct a new Cast with a valid expression and a target
    ///         tokens::CoreType, transferring ownership of the expression to
    ///         the Cast and updating parent data on the expression.
    /// @param  expr  The expression to perform the cast operator on
    /// @param  type  The target cast type
    Cast(Expression* expr, const tokens::CoreType type)
        : Expression()
        , mType(type)
        , mExpression(expr) {
            OPENVDB_ASSERT(mExpression);
            mExpression->setParent(this);
        }
    /// @brief  Deep copy constructor for a Cast node, performing a deep copy on
    ///         the underlying expressions, ensuring parent information is
    ///         updated.
    /// @param  other  A const reference to another Cast node to deep copy
    Cast(const Cast& other)
        : Expression()
        , mType(other.mType)
        , mExpression(other.mExpression->copy()) {
            mExpression->setParent(this);
        }
    ~Cast() override = default;

    /// @copybrief Node::copy()
    Cast* copy() const override final { return new Cast(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::CastNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "cast"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "cast"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 1; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return this->expression();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i != 0) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        mExpression.reset(expr);
        mExpression->setParent(this);
        return true;
    }

    /// @brief  Access to the target type
    /// @return a tokens::CoreType enumerable type therepresenting the target type
    inline tokens::CoreType type() const { return mType; }
    /// @brief  Get the target type as a front end AX type/token string
    /// @note   This returns the associated token to the type, not necessarily
    ///         equal to the OpenVDB type string
    /// @return A string representing the type/token
    inline std::string typestr() const {
        return ast::tokens::typeStringFromToken(mType);
    }
    /// @brief  Access a const pointer to the Cast node's expression as an
    ///         abstract expression
    /// @return A const pointer to the expression
    const Expression* expression() const { return mExpression.get(); }
private:
    const tokens::CoreType mType;
    Expression::UniquePtr mExpression;
};

/// @brief FunctionCalls represent a single call to a function and any provided
///        arguments. The argument list can be empty. The function name is
///        expected to exist in the AX function registry.
struct FunctionCall : public Expression
{
    using UniquePtr = std::unique_ptr<FunctionCall>;

    /// @brief  Construct a new FunctionCall with a given function identifier
    ///         and an optional argument, transferring ownership of any
    ///         provided argument to the FunctionCall and updating parent data
    ///         on the arguments.
    /// @param  function   The name/identifier of the function
    /// @param  argument  Function argument
    FunctionCall(const std::string& function,
        Expression* argument = nullptr)
        : mFunctionName(function)
        , mArguments() {
            this->append(argument);
        }
    /// @brief  Construct a new FunctionCall with a given function identifier
    ///         and optional argument list, transferring ownership of any
    ///         provided arguments to the FunctionCall and updating parent data
    ///         on the arguments.
    /// @param  function   The name/identifier of the function
    /// @param  arguments  Function arguments
    FunctionCall(const std::string& function,
        const std::vector<Expression*>& arguments)
        : mFunctionName(function)
        , mArguments() {
            mArguments.reserve(arguments.size());
            for (Expression* arg : arguments) {
                this->append(arg);
            }
        }
    /// @brief  Deep copy constructor for a FunctionCall, performing a deep copy
    ///         on all held function arguments, ensuring parent information is
    ///         updated.
    /// @param  other  A const reference to another FunctionCall to deep copy
    FunctionCall(const FunctionCall& other)
        : mFunctionName(other.mFunctionName)
        , mArguments() {
            mArguments.reserve(other.mArguments.size());
            for (const Expression::UniquePtr& expr : other.mArguments) {
                this->append(expr->copy());
            }
        }
    ~FunctionCall() override = default;

    /// @copybrief Node::copy()
    FunctionCall* copy() const override final { return new FunctionCall(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::FunctionCallNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "function call"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "call"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return this->size(); }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i >= mArguments.size()) return nullptr;
        return mArguments[i].get();
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (mArguments.size() <= i) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        mArguments[i].reset(expr);
        mArguments[i]->setParent(this);
        return true;
    }

    /// @brief  Access the function name/identifier
    /// @return A const reference to the function name
    inline const std::string& name() const { return mFunctionName; }
    /// @brief  Query the total number of arguments stored on this function
    /// @return The number of arguments. Can be 0
    inline size_t numArgs() const { return mArguments.size(); }

    /// @brief  Alias for FunctionCall::children
    inline size_t size() const { return mArguments.size(); }
    /// @brief  Query whether this Expression list holds any valid expressions
    /// @return True if this node if empty, false otherwise
    inline bool empty() const { return mArguments.empty(); }
    /// @brief  Appends an argument to this function call, transferring
    ///         ownership to the FunctionCall and updating parent data on the
    ///         expression. If the expression is a nullptr, it is ignored.
    inline void append(Expression* expr) {
        if (expr) {
            mArguments.emplace_back(expr);
            expr->setParent(this);
        }
    }
private:
    const std::string mFunctionName;
    std::vector<Expression::UniquePtr> mArguments;
};

/// @brief  Keywords represent keyword statements defining changes in execution.
///         These include those that define changes in loop execution such as
///         break and continue, as well as return statements.
struct Keyword : public Statement
{
    using UniquePtr = std::unique_ptr<Keyword>;

    /// @brief  Construct a new Keyword with a given tokens::KeywordToken.
    /// @param  keyw   The keyword token.
    Keyword(const tokens::KeywordToken keyw)
        : mKeyword(keyw) {}
    /// @brief  Deep copy constructor for a Keyword.
    /// @param  other  A const reference to another Keyword to deep copy
    Keyword(const Keyword& other)
        : mKeyword(other.mKeyword) {}
    ~Keyword() override = default;

    /// @copybrief Node::copy()
    Keyword* copy() const override final { return new Keyword(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::KeywordNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "keyword"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "keyw"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 0; }
    /// @copybrief Node::child()
    const Node* child(const size_t) const override final {
        return nullptr;
    }
    /// @brief  Query the keyword held on this node.
    /// @return The keyword as a tokens::KeywordToken
    inline tokens::KeywordToken keyword() const { return mKeyword; }

private:
    const tokens::KeywordToken mKeyword;
};

/// @brief  ArrayUnpack represent indexing operations into AX container types,
///         primarily vectors and matrices indexed by the square brackets []
///         syntax. Multiple levels of indirection (multiple components) can
///         be specified but current construction is limited to either a single
///         or double component lookup. Providing two components infers a matrix
///         indexing operation.
/// @note   Single indexing operations are still valid for matrix indexing
struct ArrayUnpack : public Expression
{
    using UniquePtr = std::unique_ptr<ArrayUnpack>;

    /// @brief  Construct a new ArrayUnpack with a valid expression, an initial
    ///         component (as an expression) to the first access and an optional
    ///         second component (as an expression) to a second access.
    /// @note   Providing a second component automatically infers this
    ///         ArrayUnpack as a matrix indexing operation. Ownership is
    ///         transferred and parent data is updated for all arguments.
    /// @param  expr  The expression to perform the unpacking operation on
    /// @param  component0  The first component access
    /// @param  component1  The second component access
    ArrayUnpack(Expression* expr,
        Expression* component0,
        Expression* component1 = nullptr)
        : mIdx0(component0)
        , mIdx1(component1)
        , mExpression(expr) {
            OPENVDB_ASSERT(mIdx0);
            OPENVDB_ASSERT(mExpression);
            mIdx0->setParent(this);
            if(mIdx1) mIdx1->setParent(this);
            mExpression->setParent(this);
        }
    /// @brief  Deep copy constructor for a ArrayUnpack, performing a deep
    ///         copy on the expression being indexed and all held components,
    ///         ensuring parent information is updated.
    /// @param  other  A const reference to another ArrayUnpack to deep copy
    ArrayUnpack(const ArrayUnpack& other)
        : ArrayUnpack(other.mExpression->copy(),
            other.mIdx0->copy(),
            other.mIdx1 ? other.mIdx1->copy() : nullptr) {}

    ~ArrayUnpack() override = default;

    /// @copybrief Node::copy()
    ArrayUnpack* copy() const override final { return new ArrayUnpack(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::ArrayUnpackNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "array unpack"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "unpk"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 3; }
    /// @copybrief Node::child()
    const Statement* child(const size_t i) const override final {
        if (i == 0) return this->component0();
        if (i == 1) return this->component1();
        if (i == 2) return this->expression();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i > 2) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        if (!expr) return false;
        if (i == 0) mIdx0.reset(expr);
        if (i == 1) mIdx1.reset(expr);
        if (i == 2) mExpression.reset(expr);
        expr->setParent(this);
        return true;
    }

    /// @brief  Access a const pointer to the first component being used as an
    ///         abstract Expression
    /// @return A const pointer to the first component
    inline const Expression* component0() const { return mIdx0.get(); }
    /// @brief  Access a const pointer to the second component being used as an
    ///         abstract Expression
    /// @note   This can be a nullptr for single indexing operations
    /// @return A const pointer to the second component
    inline const Expression* component1() const { return mIdx1.get(); }
    /// @brief  Access a const pointer to the expression being indexed as an
    ///         abstract Expression
    /// @return A const pointer to the expression
    inline const Expression* expression() const { return mExpression.get(); }
    /// @brief  Query whether this ArrayUnpack operation must be a matrix
    ///         indexing operation by checking the presence of a second
    ///         component access.
    /// @note   This method only guarantees that the indexing operation must be
    ///         a matrix index. Single indexing is also valid for matrices and
    ///         other multi dimensional containers
    /// @return True if this is a double indexing operation, only valid for
    ///         matrices
    inline bool isMatrixIndex() const {
        // assumes that component0 is always valid
        return static_cast<bool>(this->component1());
    }
private:
    Expression::UniquePtr mIdx0, mIdx1;
    Expression::UniquePtr mExpression;
};

/// @brief  ArrayPacks represent temporary container creations of arbitrary
///         sizes, typically generated through the use of curly braces {}.
struct ArrayPack : public Expression
{
    using UniquePtr = std::unique_ptr<ArrayPack>;

    /// @brief  Construct a new ArrayPack with a single expression, transferring
    ///         ownership of the expression to the ArrayPack and updating parent
    ///         data on the expression. If the expression is a nullptr, it is
    ///         ignored.
    /// @param  expression  The Expression to construct from
    ArrayPack(Expression* expression)
        : mExpressions() {
            this->append(expression);
        }
    /// @brief  Construct a new ArrayPack transferring ownership of any
    ///         provided arguments to the ArrayPack and updating parent data
    ///         on the arguments.
    /// @param  arguments  ArrayPack arguments
    ArrayPack(const std::vector<Expression*>& arguments)
        : mExpressions() {
            mExpressions.reserve(arguments.size());
            for (Expression* arg : arguments) {
                this->append(arg);
            }
        }
    /// @brief  Deep copy constructor for a ArrayPack, performing a deep copy
    ///         on all held arguments, ensuring parent information is updated.
    /// @param  other  A const reference to another ArrayPack to deep copy
    ArrayPack(const ArrayPack& other)
        : mExpressions() {
            mExpressions.reserve(other.mExpressions.size());
            for (const Expression::UniquePtr& expr : other.mExpressions) {
                this->append(expr->copy());
            }
        }
    ~ArrayPack() override = default;

    /// @copybrief Node::copy()
    ArrayPack* copy() const override final { return new ArrayPack(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::ArrayPackNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "array pack"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "pack"; }
    /// @copybrief Node::basetype()
    const Expression* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return this->size(); }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i >= mExpressions.size()) return nullptr;
        return mExpressions[i].get();
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (mExpressions.size() <= i) return false;
        Expression* expr = dynamic_cast<Expression*>(node);
        mExpressions[i].reset(expr);
        mExpressions[i]->setParent(this);
        return true;
    }

    /// @brief  Alias for ArrayPack::children
    inline size_t size() const { return mExpressions.size(); }
    /// @brief  Query whether this Expression list holds any valid expressions
    /// @return True if this node if empty, false otherwise
    inline bool empty() const { return mExpressions.empty(); }
    /// @brief  Appends an argument to this ArrayPack, transferring ownership
    ///         to the ArrayPack and updating parent data on the expression.
    ///         If the expression is a nullptr, it is ignored.
    inline void append(Expression* expr) {
        if (expr) {
            mExpressions.emplace_back(expr);
            expr->setParent(this);
        }
    }
private:
    std::vector<Expression::UniquePtr> mExpressions;
};

/// @brief  Attributes represent any access to a primitive value, typically
///         associated with the '@' symbol syntax. Note that the AST does not
///         store any additional information on the given attribute other than
///         its name and type, which together form a unique Attribute identifier
///         known as the Attribute 'token'. A 'primitive value' in this instance
///         refers to a value on an OpenVDB Volume or OpenVDB Points tree.
/// @note   The ExternalVariable AST node works in a similar way
/// @note   An Attribute is a complete "leaf-level" AST node. It has no children
///         and nothing derives from it.
struct Attribute : public Variable
{
    using UniquePtr = std::unique_ptr<Attribute>;

    /// @brief  Construct a new Attribute with a given name and type. Optionally
    ///         also mark it as inferred type creation (no type was directly
    ///         specified)
    /// @param  name      The name of the attribute
    /// @param  type      The type of the attribute
    /// @param  inferred  Whether the provided type was directly specified
    ///                   (false).
    Attribute(const std::string& name, const tokens::CoreType type,
              const bool inferred = false)
        : Variable(name)
        , mType(type)
        , mTypeInferred(inferred) {}
    /// @brief  Construct a new Attribute with a given name and type/token
    ///         string, delegating construction to the above Attribute
    ///         constructor.
    /// @param  name      The name of the attribute
    /// @param  token     The type/token string of the attribute
    /// @param  inferred  Whether the provided type was directly specified
    ///                   (false).
    Attribute(const std::string& name, const std::string& token,
              const bool inferred = false)
        : Attribute(name, tokens::tokenFromTypeString(token), inferred) {}
    /// @brief  Deep copy constructor for a Attribute
    /// @note   No parent information needs updating as an Attribute is a
    ///         "leaf level" node (contains no children)
    /// @param  other  A const reference to another Attribute to deep copy
    Attribute(const Attribute& other)
        : Variable(other)
        , mType(other.mType)
        , mTypeInferred(other.mTypeInferred) {}
    ~Attribute() override = default;

    /// @copybrief Node::copy()
    Attribute* copy() const override final { return new Attribute(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::AttributeNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "attribute"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "atr"; }
    /// @copybrief Node::basetype()
    const Variable* basetype() const override { return this; }

    /// @brief  Query whether this attribute was accessed via inferred syntax
    ///         i.e. \@P or \@myattribute
    /// @return True if inferred syntax was used
    inline bool inferred() const { return mTypeInferred; }
    /// @brief  Access the type that was used to access this attribute
    /// @return The type used to access this attribute as a tokens::CoreType
    inline tokens::CoreType type() const { return mType; }
    /// @brief  Get the access type as a front end AX type/token string
    /// @note   This returns the associated token to the type, not necessarily
    ///         equal to the OpenVDB type string
    /// @return A string representing the type/token
    inline std::string typestr() const {
        return ast::tokens::typeStringFromToken(mType);
    }
    /// @brief  Construct and return the full attribute token identifier. See
    ///         Attribute::tokenFromNameType
    /// @return A string representing the attribute token.
    inline std::string tokenname() const {
        return Attribute::tokenFromNameType(this->name(), this->type());
    }

    /// @brief  Static method returning the symbol associated with an Attribute
    ///         access as defined by AX Grammar
    /// @return The '@' character as a char
    static inline char symbolseparator() { return '@'; }
    /// @brief  Static method returning the full unique attribute token
    ///         identifier by consolidating its name and type such that
    ///         token = tokenstr + '\@' + name, where tokenstr is the AX type
    ///         token as a string, converted from the provided CoreType.
    /// @note   This identifier is unique for accesses to the same attribute
    /// @note   Due to inferred and single character accesses in AX, this return
    ///         value does not necessarily represent the original syntax used to
    ///         access this attribute. For example, \@myattrib will be stored
    ///         and returned as float\@myattrib.
    /// @param  name  The name of the attribute
    /// @param  type  The CoreType of the attribute
    /// @return A string representing the attribute token.
    static inline std::string
    tokenFromNameType(const std::string& name, const tokens::CoreType type) {
        return ast::tokens::typeStringFromToken(type) +
            Attribute::symbolseparator() + name;
    }
    /// @brief  Static method which splits a valid attribute token into its name
    ///         and type counterparts. If the token cannot be split, neither
    ///         name or type are updated and false is returned.
    /// @param  token The token to split.
    /// @param  name  Set to the second part of the attribute token,
    ///               representing the name. If a nullptr, it is ignored
    /// @param  type  Set to the first part of the attribute token,
    ///               representing the type. If a nullptr, it is ignored. Note
    ///               that this can be empty if the attribute token has an
    ///               inferred type or a single character.
    /// @return True if the provided attribute token could be split
    static inline bool
    nametypeFromToken(const std::string& token, std::string* name, std::string* type) {
        const size_t at = token.find(symbolseparator());
        if (at == std::string::npos) return false;
        if (type) {
            *type = token.substr(0, at);
            if (type->empty()) {
                *type = ast::tokens::typeStringFromToken(tokens::CoreType::FLOAT);
            }
        }
        if (name) *name = token.substr(at + 1, token.size());
        return true;
    }
private:
    const tokens::CoreType mType;
    const bool mTypeInferred;
};

/// @brief  ExternalVariable represent any access to external (custom) data,
///         typically associated with the '$' symbol syntax. Note that the AST
///         does not store any additional information on the given external
///         other than its name and type, which together form a unique external
///         identifier known as the ExternalVariable 'token'. This token is used
///         by the compiler to map user provided values to these external
///         values.
/// @note   The Attribute AST node works in a similar way
/// @note   An ExternalVariable is a complete "leaf-level" AST node. It has no
///         children and nothing derives from it.
struct ExternalVariable : public Variable
{
    using UniquePtr = std::unique_ptr<ExternalVariable>;

    /// @brief  Construct a new ExternalVariable with a given name and type
    /// @param  name      The name of the attribute
    /// @param  type      The type of the attribute
    ExternalVariable(const std::string& name, const tokens::CoreType type)
        : Variable(name)
        , mType(type) {}
    /// @brief  Construct a new ExternalVariable with a given name and type/token
    ///         string, delegating construction to the above ExternalVariable
    ///         constructor.
    /// @param  name      The name of the attribute
    /// @param  token     The type/token string of the attribute
    ExternalVariable(const std::string& name, const std::string& token)
        : ExternalVariable(name, tokens::tokenFromTypeString(token)) {}
    /// @brief  Deep copy constructor for a ExternalVariable
    /// @note   No parent information needs updating as an ExternalVariable is a
    ///         "leaf level" node (contains no children)
    /// @param  other  A const reference to another ExternalVariable to deep
    ///                copy
    ExternalVariable(const ExternalVariable& other)
        : Variable(other)
        , mType(other.mType) {}
    ~ExternalVariable() override = default;

    /// @copybrief Node::copy()
    ExternalVariable* copy() const override final {
        return new ExternalVariable(*this);
    }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::ExternalVariableNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "external"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "ext"; }
    /// @copybrief Node::basetype()
    const Variable* basetype() const override { return this; }

    /// @brief  Access the type that was used to access this external variable
    /// @return The type used to access this external as a tokens::CoreType
    inline tokens::CoreType type() const { return mType; }
    /// @brief  Get the access type as a front end AX type/token string
    /// @note   This returns the associated token to the type, not necessarily
    ///         equal to the OpenVDB type string
    /// @return A string representing the type/token
    inline std::string typestr() const {
        return ast::tokens::typeStringFromToken(mType);
    }
    /// @brief  Construct and return the full external token identifier. See
    ///         ExternalVariable::tokenFromNameType
    /// @return A string representing the external variable token.
    inline const std::string tokenname() const {
        return ExternalVariable::tokenFromNameType(this->name(), this->type());
    }

    /// @brief  Static method returning the symbol associated with an
    ///         ExternalVariable access as defined by AX Grammar
    /// @return The '$' character as a char
    static inline char symbolseparator() { return '$'; }
    /// @brief  Static method returning the full unique external token
    ///         identifier by consolidating its name and type such that
    ///         token = tokenstr + '$' + name, where tokenstr is the AX type
    ///         token as a string, converted from the provided CoreType.
    /// @note   This identifier is unique for accesses to the same external
    /// @note   Due to inferred and single character accesses in AX, this return
    ///         value does not necessarily represent the original syntax used to
    ///         access this external. For example, v$data will be stored and
    ///         returned as vec3f$data.
    /// @param  name  The name of the external
    /// @param  type  The CoreType of the external
    /// @return A string representing the external token.
    static inline std::string
    tokenFromNameType(const std::string& name, const tokens::CoreType type) {
        return ast::tokens::typeStringFromToken(type) +
            ExternalVariable::symbolseparator() + name;
    }
    /// @brief  Static method which splits a valid external token into its name
    ///         and type counterparts. If the token cannot be split, neither
    ///         name or type are updated and false is returned.
    /// @param  token The token to split.
    /// @param  name  Set to the second part of the external token,
    ///               representing the name. If a nullptr, it is ignored
    /// @param  type  Set to the first part of the external token,
    ///               representing the type. If a nullptr, it is ignored. Note
    ///               that this can be empty if the external token has an
    ///               inferred type or a single character.
    /// @return True if the provided external token could be split
    static inline bool
    nametypeFromToken(const std::string& token, std::string* name, std::string* type) {
        const size_t at = token.find(symbolseparator());
        if (at == std::string::npos) return false;
        if (type) {
            *type = token.substr(0, at);
            if (type->empty()) {
                *type = ast::tokens::typeStringFromToken(tokens::CoreType::FLOAT);
            }
        }
        if (name) *name = token.substr(at + 1, token.size());
        return true;
    }
private:
    const tokens::CoreType mType;
};

/// @brief  Local AST nodes represent a single accesses to a local variable.
///         The only store the name of the variable being accessed.
/// @note   A Local is a complete "leaf-level" AST node. It has no children and
///         nothing derives from it.
struct Local : public Variable
{
    using UniquePtr = std::unique_ptr<Local>;

    /// @brief  Construct a Local with a given name
    /// @param  name  The name of the local variable being accessed
    Local(const std::string& name)
        : Variable(name) {}
    ~Local() override = default;

    /// @copybrief Node::copy()
    Local* copy() const override final { return new Local(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::LocalNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "local"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "lcl"; }
    /// @copybrief Node::basetype()
    const Variable* basetype() const override { return this; }
};

/// @brief  DeclareLocal AST nodes symbolize a single type declaration of a
///         local variable. These store the local variables that They also however store its
///         specified type. These have the important distinction of representing
///         the initial creation and allocation of a variable, in comparison to
///         a Local node which only represents access.
struct DeclareLocal : public Statement
{
    using UniquePtr = std::unique_ptr<DeclareLocal>;

    /// @brief  Construct a new DeclareLocal with a given name and type
    /// @param  type  The type of the declaration
    /// @param  local The local variable being declared
    /// @param  init  The initialiser expression of the local
    DeclareLocal(const tokens::CoreType type, Local* local, Expression* init = nullptr)
        : mType(type)
        , mLocal(local)
        , mInit(init) {
            OPENVDB_ASSERT(mLocal);
            mLocal->setParent(this);
            if (mInit) mInit->setParent(this);
        }
    /// @brief  Deep copy constructor for a DeclareLocal
    /// @note   No parent information needs updating as an DeclareLocal is a
    ///         "leaf level" node (contains no children)
    /// @param  other  A const reference to another DeclareLocal to deep copy
    DeclareLocal(const DeclareLocal& other)
        : mType(other.mType)
        , mLocal(other.mLocal->copy())
        , mInit(other.hasInit() ? other.mInit->copy() : nullptr) {
            mLocal->setParent(this);
            if (mInit) mInit->setParent(this);
        }
    ~DeclareLocal() override = default;

    /// @copybrief Node::copy()
    DeclareLocal* copy() const override final { return new DeclareLocal(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override { return Node::DeclareLocalNode; }
    /// @copybrief Node::nodename()
    const char* nodename() const override { return "declaration"; }
    /// @copybrief Node::subname()
    const char* subname() const override { return "dcl"; }
    /// @copybrief Node::basetype()
    const Statement* basetype() const override { return this; }
    /// @copybrief Node::children()
    size_t children() const override final { return 2; }
    /// @copybrief Node::child()
    const Expression* child(const size_t i) const override final {
        if (i == 0) return this->local();
        if (i == 1) return this->init();
        return nullptr;
    }
    /// @copybrief Node::replacechild()
    inline bool replacechild(const size_t i, Node* node) override final {
        if (i > 1) return false;
        if (i == 0) {
            Local* local = dynamic_cast<Local*>(node);
            if (!local) return false;
            mLocal.reset(local);
            mLocal->setParent(this);
        }
        else {
            Expression* init = dynamic_cast<Expression*>(node);
            if (!init) return false;
            mInit.reset(init);
            mInit->setParent(this);
        }
        return true;
    }

    /// @brief  Access the type that was specified at which to create the given
    ///         local
    /// @return The declaration type
    inline tokens::CoreType type() const { return mType; }
    /// @brief  Get the declaration type as a front end AX type/token string
    /// @note   This returns the associated token to the type, not necessarily
    ///         equal to the OpenVDB type string
    /// @return A string representing the type/token
    inline std::string typestr() const {
        return ast::tokens::typeStringFromToken(mType);
    }
    /// @brief  Query if this declaration has an initialiser
    /// @return True if an initialiser exists, false otherwise
    inline bool hasInit() const { return static_cast<bool>(this->init()); }

    /// @brief  Access a const pointer to the Local
    /// @return A const pointer to the local
    const Local* local() const { return mLocal.get(); }
    /// @brief  Access a const pointer to the initialiser
    /// @return A const pointer to the initialiser
    const Expression* init() const { return mInit.get(); }

private:
    const tokens::CoreType mType;
    Local::UniquePtr       mLocal; // could be Variable for attribute declaration
    Expression::UniquePtr  mInit;
};


/// @brief  A Value (literal) AST node holds either literal text or absolute
///         value information on all numerical, string and boolean constants.
///         A single instance of a Value is templated on the requested scalar,
///         boolean or string type. If scalar or boolean value is constructed
///         from a string (as typically is the case in the parser), the value is
///         automatically converted to its numerical representation. If this
///         fails, the original text is stored instead.
/// @note   All numerical values are stored as their highest possible precision
///         type to support overflowing without storing the original string
///         data. The original string data is only required if the value is too
///         large to be stored in these highest precision types (usually a
///         uint64_t for scalars or double for floating points).
/// @note   Numerical values are guaranteed to be positive (if constructed from
///         the AX parser). Negative values are represented by a combination of
///         a UnaryOperator holding a Value AST node.
/// @note   Note that Value AST nodes representing strings are specialized and
///         are guranteed to be "well-formed" (there is no numerical conversion)
/// @note   A Value is a complete "leaf-level" AST node. It has no children and
///         nothing derives from it.
template <typename T>
struct Value : public ValueBase
{
    using UniquePtr = std::unique_ptr<Value<T>>;

    using Type = T;
    /// @brief  Integers and Floats store their value as ContainerType, which is
    ///         guaranteed to be at least large enough to represent the maximum
    ///         possible supported type for the requested precision.
    using ContainerType = typename std::conditional<
        std::is_integral<T>::value, uint64_t, T>::type;

    /// @brief  The list of supported numerical constants.
    /// @note   Strings are specialized and handled separately
    static constexpr bool IsSupported =
        std::is_same<T, bool>::value ||
        std::is_same<T, int16_t>::value ||
        std::is_same<T, int32_t>::value ||
        std::is_same<T, int64_t>::value ||
        std::is_same<T, float>::value ||
        std::is_same<T, double>::value;
    static_assert(IsSupported, "Incompatible ast::Value node instantiated.");

    /// @brief  Directly construct a Value from a source integer, float or
    ///         boolean, guaranteeing valid construction. Note that the provided
    ///         argument should not be negative
    Value(const ContainerType value)
        : mValue(value) {}
    /// @brief  Deep copy constructor for a Value
    /// @note   No parent information needs updating as a Value is a "leaf
    ///         level" node (contains no children)
    /// @param  other  A const reference to another Value to deep copy
    Value(const Value<T>& other)
        : mValue(other.mValue)  {}
    ~Value() override = default;

    /// @copybrief Node::copy()
    Value<Type>* copy() const override final { return new Value<Type>(*this); }
    /// @copybrief Node::nodetype()
    NodeType nodetype() const override {
        if (std::is_same<T, bool>::value)    return Node::ValueBoolNode;
        if (std::is_same<T, int16_t>::value) return Node::ValueInt16Node;
        if (std::is_same<T, int32_t>::value) return Node::ValueInt32Node;
        if (std::is_same<T, int64_t>::value) return Node::ValueInt64Node;
        if (std::is_same<T, float>::value)   return Node::ValueFloatNode;
        if (std::is_same<T, double>::value)  return Node::ValueDoubleNode;
    }
    /// @copybrief Node::nodename()
    const char* nodename() const override {
        if (std::is_same<T, bool>::value)    return "boolean literal";
        if (std::is_same<T, int16_t>::value) return "int16 literal";
        if (std::is_same<T, int32_t>::value) return "int32 literal";
        if (std::is_same<T, int64_t>::value) return "int64 literal";
        if (std::is_same<T, float>::value)   return "float (32bit) literal";
        if (std::is_same<T, double>::value)  return "double (64bit) literal";
    }
    /// @copybrief Node::subname()
    const char* subname() const override {
        if (std::is_same<T, bool>::value)    return "bool";
        if (std::is_same<T, int16_t>::value) return "i16";
        if (std::is_same<T, int32_t>::value) return "i32";
        if (std::is_same<T, int64_t>::value) return "i64";
        if (std::is_same<T, float>::value)   return "flt";
        if (std::is_same<T, double>::value)  return "dbl";
    }
    /// @copybrief Node::basetype()
    const ValueBase* basetype() const override { return this; }

    /// @brief  Access the value as its stored type
    /// @return The value as its stored ContainerType
    inline ContainerType asContainerType() const { return mValue; }
    /// @brief  Access the value as its requested (templated) type
    /// @return The value as its templed type T
    inline T value() const { return static_cast<T>(mValue); }

private:
    // A container of a max size defined by LiteralValueContainer to hold values
    // which may be out of scope. This is only used for warnings
    const ContainerType mValue;
};

/// @brief  Specialization of Values for strings
template <>
struct Value<std::string> : public ValueBase
{
    using UniquePtr = std::unique_ptr<Value<std::string>>;

    using Type = std::string;
    /// @brief  Construct a new Value string from a string
    /// @param  value The string to copy onto this Value
    Value(const Type& value) : mValue(value) {}
    /// @brief  Deep copy constructor for a Value string
    /// @note   No parent information needs updating as a Value is a "leaf
    ///         level" node (contains no children)
    /// @param  other  A const reference to another Value string to deep copy
    Value(const Value<Type>& other) : mValue(other.mValue) {}
    ~Value() override = default;

    Value<Type>* copy() const override final { return new Value<Type>(*this); }
    NodeType nodetype() const override { return Node::ValueStrNode; }
    const char* nodename() const override { return "string value"; }
    const char* subname() const override { return "str"; }
    const ValueBase* basetype() const override { return this; }

    /// @brief  Access the string
    /// @return A const reference to the string
    inline const std::string& value() const { return mValue; }
private:
    const Type mValue;
};

} // namespace ast
} // namespace ax

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_HAS_BEEN_INCLUDED

