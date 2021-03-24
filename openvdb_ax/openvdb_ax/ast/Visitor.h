// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Visitor.h
///
/// @authors Nick Avramoussis
///
/// @brief  Contains the AX AST Node Visitor, providing default and
///   customizable traversal and visitation methods on a AST hierarchy.
///   Using the visitor pattern is the recommended way to implement
///   custom operations on AST nodes.
///

#ifndef OPENVDB_AX_AST_VISITOR_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_VISITOR_HAS_BEEN_INCLUDED

#include "AST.h"
#include "Tokens.h"

#include <openvdb/version.h>

#include <type_traits>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

/// @brief   The Visitor class uses the Curiously Recursive Template Pattern
///          (CRTP) to provide a customizable interface intended to be used by
///          clients wishing to perform custom operations over an AX Abstract
///          Syntax Tree (AST). By default the Visitor implements simple
///          traversal of all nodes, ensuring that each node on a well formed
///          AST is visited at least once. By deriving from the Visitor, users
///          are able to customize this default behavior and further manually
///          override specific node behavior to their needs. The function
///          options at the top of visitor can be overridden using CRTP to
///          control the prior default behavior, with the ability to override
///          the traverse() and visit() methods for the latter more granular
///          control.
///
/// @details To commence a full visit of an AST, begin by calling traverse() on
///          a Node pointer. A visit is defined as one of the visit() methods
///          being called and accepting a Node type. Each node is is guaranteed
///          to be visited exactly once at its lowest concrete derived type.
///          Node inheritance hierarchies can also be visited (disable by
///          default, see Visitor::visitNodeHierarchies) The traverse() methods
///          define how each AST node accesses its children. The default
///          implementation is for each node to traverses its child pointers in
///          the order returned by the derived Node::child() method
///          (see Visitor::reverseChildVisits). You'll typically only require
///          overriding of the visit() methods for achieving most goals, however
///          you can utilize the traverse methods if you find that you require
///          more control over how the node hierarchy is accessed. The default
///          visit order is post order, where by nodes traverse and visit their
///          children first (see Visitor::postOrderNodes). Each visit method
///          returns a boolean value which, if false, allows for early
///          termination of the traversal. In the below example, we show a
///          Visitor capable of visiting every Local node type exactly once,
///          terminating if the Local variable is called "var".
///
///               @par Example:
///               @code
///               struct LocalVisitor : public Visitor<LocalVisitor>
///               {
///                   // Bring in all base methods to avoid hiding
///                   using ast::Visitor<LocalVisitor>::traverse;
///                   using ast::Visitor<LocalVisitor>::visit;
///
///                   // override the visit for Local AST nodes
///                   inline bool visit(const Local* node) {
///                       if (!node) return true;
///                       if (node->name() == "var") return false;
///                       return true;
///                   }
///               };
///
///               LocalVisitor visitor;
///               visitor.traverse(&tree);
///               @endcode
///
/// @note  The second template argument, ConstVisit, allows you to perform
///        non-const traversals over the AST. In this case, the visit and
///        traversal function signatures change to non-const pointers.
/// @note  This design is heavily influenced by Clang's RecursiveVisitor.
///
/// @tparam Derived     The derived visitor to template on the base visitor,
///                     using CRTP
/// @tparam ConstVisit  Whether to visit const or non-const versions of the AST
///                     nodes. Note that this value changes the class function
///                     signatures.
template <typename Derived, bool ConstVisit=true>
struct Visitor
{
    /// @brief  Templated conditional which resolves to a const NodeT if
    ///         ConstVisit is true, or a non-const NodeT if ConstVisit is false
    template <typename NodeT>
    using NodeType = typename std::conditional<ConstVisit, const NodeT, NodeT>::type;

    /// @brief  Accesses the derived class by static casting the current object.
    ///         Assumes use of the Curiously Recursive Template Pattern (CRTP).
    inline Derived& derived() {
        return *static_cast<Derived*>(this);
    }

    /// @name Options
    /// @{

    /// @brief  Default behavior option. If true, this results in post-order
    ///         traversal, where node children are traversed and visited before
    ///         their parent node. If false, this results in pre-order
    ///         traversal, where by the current node is visited before the
    ///         node's children.
    /// @details Post-order traversal (for each node):
    ///            1. Traverse all children.
    ///            2. Visit the current node.
    ///          Pre-order traversal (for each node):
    ///            1. Visit the current node.
    ///            2. Traverse all children.
    inline bool postOrderNodes() const { return true; }

    /// @brief  Default behavior option. Reverses the traversal order of child
    ///         nodes. If true, child nodes are accessed from last to first
    ///         index .i.e. Node::children() -> 0. If false, child nodes are
    ///         accessed from first to last .i.e. 0 -> Node::children()
    inline bool reverseChildVisits() const { return false; }

    /// @brief  Default behavior option. Controls whether nodes visit themselves
    ///         at each stage of their class hierarchy. If true, nodes perform
    ///         multiple visits on their potentially abstract base classes. If
    ///         false, only the concrete derived types are visited.
    /// @details When disabled, abstract node visitor methods are never accessed
    ///          directly through the default Visitor implementation. These
    ///          types include Node, Statement, Expression, etc AST nodes.
    ///          If true, for each linearly inherited AST node, a visit is
    ///          performed on the entire hierarchy. For example, for a Local AST
    ///          node which derives from Variable -> Expression -> Statement ->
    ///          Node, 5 visits will be performed at each level.
    inline bool visitNodeHierarchies() const { return false; }

    /// @brief  Default behavior option. Reverses the traversal order of node
    ///         hierarchies. If true, hierarchical visits start at the very top
    ///         of their inheritance structure (always a Node AST node) and
    ///         visit downwards until the lowest derived concrete node is
    ///         reached. If false, hierarchical visits start at the lowest
    ///         derived concrete node and visit upwards until the very top of
    ///         their inheritance structure (always a Node AST node) is reached.
    /// @note   Has no effect if visitNodeHierarchies() is false
    inline bool reverseHierarchyVisits() const { return false; }

    /// @}

    /// @name Traversals
    /// @{

    /// @brief  Default traversals for a given concrete AST node type
    /// @return True if traversal should continue, false to terminate

    bool traverse(NodeType<ast::Tree>* tree) {
        return this->defaultTraversal<ast::Tree>(tree);
    }

    bool traverse(NodeType<ast::StatementList>* cond) {
        return this->defaultTraversal<ast::StatementList>(cond);
    }

    bool traverse(NodeType<ast::Block>* block) {
        return this->defaultTraversal<ast::Block>(block);
    }

    bool traverse(NodeType<ast::CommaOperator>* comma) {
        return this->defaultTraversal<ast::CommaOperator>(comma);
    }

    bool traverse(NodeType<ast::Loop>* loop) {
        return this->defaultTraversal<ast::Loop>(loop);
    }

    bool traverse(NodeType<ast::Keyword>* keyw) {
        return this->defaultTraversal<ast::Keyword>(keyw);
    }

    bool traverse(NodeType<ast::ConditionalStatement>* cond) {
        return this->defaultTraversal<ast::ConditionalStatement>(cond);
    }

    bool traverse(NodeType<ast::AssignExpression>* asgn) {
        return this->defaultTraversal<ast::AssignExpression>(asgn);
    }

    bool traverse(NodeType<ast::Crement>* crmt) {
        return this->defaultTraversal<ast::Crement>(crmt);
    }

    bool traverse(NodeType<ast::UnaryOperator>* unry) {
        return this->defaultTraversal<ast::UnaryOperator>(unry);
    }

    bool traverse(NodeType<ast::BinaryOperator>* bin) {
        return this->defaultTraversal<ast::BinaryOperator>(bin);
    }

    bool traverse(NodeType<ast::TernaryOperator>* tern) {
        return this->defaultTraversal<ast::TernaryOperator>(tern);
    }

    bool traverse(NodeType<ast::Cast>* cast) {
        return this->defaultTraversal<ast::Cast>(cast);
    }

    bool traverse(NodeType<ast::FunctionCall>* call) {
        return this->defaultTraversal<ast::FunctionCall>(call);
    }

    bool traverse(NodeType<ast::Attribute>* attr) {
        return this->defaultTraversal<ast::Attribute>(attr);
    }

    bool traverse(NodeType<ast::ExternalVariable>* ext) {
        return this->defaultTraversal<ast::ExternalVariable>(ext);
    }

    bool traverse(NodeType<ast::DeclareLocal>* decl) {
        return this->defaultTraversal<ast::DeclareLocal>(decl);
    }

    bool traverse(NodeType<ast::Local>* loc) {
        return this->defaultTraversal<ast::Local>(loc);
    }

    bool traverse(NodeType<ast::ArrayPack>* pack) {
        return this->defaultTraversal<ast::ArrayPack>(pack);
    }

    bool traverse(NodeType<ast::ArrayUnpack>* pack) {
        return this->defaultTraversal<ast::ArrayUnpack>(pack);
    }

    bool traverse(NodeType<ast::Value<bool>>* val) {
        return this->defaultTraversal<ast::Value<bool>>(val);
    }

    bool traverse(NodeType<ast::Value<int16_t>>* val) {
        return this->defaultTraversal<ast::Value<int16_t>>(val);
    }

    bool traverse(NodeType<ast::Value<int32_t>>* val) {
        return this->defaultTraversal<ast::Value<int32_t>>(val);
    }

    bool traverse(NodeType<ast::Value<int64_t>>* val) {
        return this->defaultTraversal<ast::Value<int64_t>>(val);
    }

    bool traverse(NodeType<ast::Value<float>>* val) {
        return this->defaultTraversal<ast::Value<float>>(val);
    }

    bool traverse(NodeType<ast::Value<double>>* val) {
        return this->defaultTraversal<ast::Value<double>>(val);
    }

    bool traverse(NodeType<ast::Value<std::string>>* val) {
        return this->defaultTraversal<ast::Value<std::string>>(val);
    }

    /// @brief  The default traversal method which is hit for all child
    ///         traversals. The correct derived traversal scheme is selected by
    ///         using the node enumerated type.
    /// @note   Only handles traversal on concrete node types.
    bool traverse(NodeType<ast::Node>* node) {
        if (!node) return true;
        switch (node->nodetype()) {
            case Node::TreeNode : return this->derived().traverse(static_cast<NodeType<ast::Tree>*>(node));
            case Node::StatementListNode : return this->derived().traverse(static_cast<NodeType<ast::StatementList>*>(node));
            case Node::BlockNode : return this->derived().traverse(static_cast<NodeType<ast::Block>*>(node));
            case Node::CommaOperatorNode : return this->derived().traverse(static_cast<NodeType<ast::CommaOperator>*>(node));
            case Node::LoopNode : return this->derived().traverse(static_cast<NodeType<ast::Loop>*>(node));
            case Node::KeywordNode : return this->derived().traverse(static_cast<NodeType<ast::Keyword>*>(node));
            case Node::ConditionalStatementNode : return this->derived().traverse(static_cast<NodeType<ast::ConditionalStatement>*>(node));
            case Node::AssignExpressionNode : return this->derived().traverse(static_cast<NodeType<ast::AssignExpression>*>(node));
            case Node::CrementNode : return this->derived().traverse(static_cast<NodeType<ast::Crement>*>(node));
            case Node::UnaryOperatorNode : return this->derived().traverse(static_cast<NodeType<ast::UnaryOperator>*>(node));
            case Node::BinaryOperatorNode : return this->derived().traverse(static_cast<NodeType<ast::BinaryOperator>*>(node));
            case Node::TernaryOperatorNode : return this->derived().traverse(static_cast<NodeType<ast::TernaryOperator>*>(node));
            case Node::CastNode : return this->derived().traverse(static_cast<NodeType<ast::Cast>*>(node));
            case Node::AttributeNode : return this->derived().traverse(static_cast<NodeType<ast::Attribute>*>(node));
            case Node::FunctionCallNode : return this->derived().traverse(static_cast<NodeType<ast::FunctionCall>*>(node));
            case Node::ExternalVariableNode : return this->derived().traverse(static_cast<NodeType<ast::ExternalVariable>*>(node));
            case Node::DeclareLocalNode : return this->derived().traverse(static_cast<NodeType<ast::DeclareLocal>*>(node));
            case Node::ArrayPackNode : return this->derived().traverse(static_cast<NodeType<ast::ArrayPack>*>(node));
            case Node::ArrayUnpackNode : return this->derived().traverse(static_cast<NodeType<ast::ArrayUnpack>*>(node));
            case Node::LocalNode : return this->derived().traverse(static_cast<NodeType<ast::Local>*>(node));
            case Node::ValueBoolNode : return this->derived().traverse(static_cast<NodeType<ast::Value<bool>>*>(node));
            case Node::ValueInt16Node : return this->derived().traverse(static_cast<NodeType<ast::Value<int16_t>>*>(node));
            case Node::ValueInt32Node : return this->derived().traverse(static_cast<NodeType<ast::Value<int32_t>>*>(node));
            case Node::ValueInt64Node : return this->derived().traverse(static_cast<NodeType<ast::Value<int64_t>>*>(node));
            case Node::ValueFloatNode : return this->derived().traverse(static_cast<NodeType<ast::Value<float>>*>(node));
            case Node::ValueDoubleNode : return this->derived().traverse(static_cast<NodeType<ast::Value<double>>*>(node));
            case Node::ValueStrNode : return this->derived().traverse(static_cast<NodeType<ast::Value<std::string>>*>(node));
            default : return true;
        }
    }

    /// @}

    /// @name Visits
    /// @{

    /// @brief  Visits for abstract (pure-virtual) Node types.
    /// @note   These are only hit through the default behavior if
    ///         Visitor::visitNodeHierarchies is enabled.
    /// @return True if traversal should continue, false to terminate
    inline bool visit(NodeType<ast::Node>*) { return true; }
    inline bool visit(NodeType<ast::Statement>*) { return true; }
    inline bool visit(NodeType<ast::Expression>*) { return true; }
    inline bool visit(NodeType<ast::Variable>*) { return true; }
    inline bool visit(NodeType<ast::ValueBase>*) { return true; }

    /// @brief  Visits for concrete Node types.
    /// @return True if traversal should continue, false to terminate
    inline bool visit(NodeType<ast::Tree>*) { return true; }
    inline bool visit(NodeType<ast::StatementList>*) { return true; }
    inline bool visit(NodeType<ast::Block>*) { return true; }
    inline bool visit(NodeType<ast::CommaOperator>*) { return true; }
    inline bool visit(NodeType<ast::Loop>*) { return true; }
    inline bool visit(NodeType<ast::Keyword>*) { return true; }
    inline bool visit(NodeType<ast::ConditionalStatement>*) { return true; }
    inline bool visit(NodeType<ast::AssignExpression>*) { return true; }
    inline bool visit(NodeType<ast::Crement>*) { return true; }
    inline bool visit(NodeType<ast::UnaryOperator>*) { return true; }
    inline bool visit(NodeType<ast::BinaryOperator>*) { return true; }
    inline bool visit(NodeType<ast::TernaryOperator>*) { return true; }
    inline bool visit(NodeType<ast::Cast>*) { return true; }
    inline bool visit(NodeType<ast::FunctionCall>*) { return true; }
    inline bool visit(NodeType<ast::Attribute>*) { return true; }
    inline bool visit(NodeType<ast::ExternalVariable>*) { return true; }
    inline bool visit(NodeType<ast::DeclareLocal>*) { return true; }
    inline bool visit(NodeType<ast::Local>*) { return true; }
    inline bool visit(NodeType<ast::ArrayPack>*) { return true; }
    inline bool visit(NodeType<ast::ArrayUnpack>*) { return true; }
    inline bool visit(NodeType<ast::Value<bool>>*) { return true; }
    inline bool visit(NodeType<ast::Value<int16_t>>*) { return true; }
    inline bool visit(NodeType<ast::Value<int32_t>>*) { return true; }
    inline bool visit(NodeType<ast::Value<int64_t>>*) { return true; }
    inline bool visit(NodeType<ast::Value<float>>*) { return true; }
    inline bool visit(NodeType<ast::Value<double>>*) { return true; }
    inline bool visit(NodeType<ast::Value<std::string>>*) { return true; }

    /// @}

private:
    /// @brief  Enabled for const traversals, where by the node pointer is
    ///         returned
    /// @param  Const reference to an AST node
    /// @return Const pointer to the node
    template <bool V, typename NodeT>
    inline typename std::enable_if<V, const NodeT*>::type
    strip(const NodeT* node) {
        return node;
    }

    /// @brief  Enabled for non-const traversals, where by a const stripped node
    ///         pointer is returned
    /// @param  Const reference to an AST node
    /// @return Non-const pointer to the node
    template <bool V, typename NodeT>
    inline typename std::enable_if<!V, typename std::remove_const<NodeT>::type*>::type
    strip(const NodeT* node) {
        return const_cast<NodeT*>(node);
    }

    /// @brief  Implements recursive hierarchical visits to a given AST node
    /// @tparam NodeT The node type
    /// @param  node  The node to perform class hierarchy visits on
    /// @return True if traversal should continue, false to terminate
    template <typename NodeT>
    bool hierarchyVisits(NodeT& node)
    {
        if (this->derived().reverseHierarchyVisits()) {
            if (auto base = node.NodeT::basetype()) {
                if (!hierarchyVisits(*base)) return false;
            }
            if (!this->derived().visit(this->strip<ConstVisit>(&node))) return false;
        }
        else {
            if (!this->derived().visit(this->strip<ConstVisit>(&node))) return false;
            if (auto base = node.NodeT::basetype()) {
                return hierarchyVisits(*base);
            }
        }
        return true;
    }

    /// @brief  Implements the default behavior for a traversal to a given AST
    ///         node
    /// @tparam NodeT The node type
    /// @param  node  The node to traverse
    /// @return True if traversal should continue, false to terminate
    template <typename NodeT>
    inline bool defaultTraversal(NodeType<NodeT>* node)
    {
        if (!node) return true;
        const size_t children = node->children();

        if (this->derived().postOrderNodes()) {
            if (this->derived().reverseChildVisits()) {
                if (children != 0) {
                    for (int64_t i = static_cast<int64_t>(children - 1); i >= 0; --i) {
                        auto child = this->strip<ConstVisit>(node->child(i));
                        if (!this->derived().traverse(child)) {
                            return false;
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < children; ++i) {
                    auto child = this->strip<ConstVisit>(node->child(i));
                    if (!this->derived().traverse(child)) {
                        return false;
                    }
                }
            }
            if (this->derived().visitNodeHierarchies()) {
                return this->hierarchyVisits(*node);
            }
            else {
                return this->derived().visit(node);
            }
        }
        else {
            if (this->derived().visitNodeHierarchies()) {
                if (!this->hierarchyVisits(*node)) return false;
            }
            else {
                if (!this->derived().visit(node)) return false;
            }
            if (this->derived().reverseChildVisits()) {
                if (children != 0) {
                    for (int64_t i = static_cast<int64_t>(children - 1); i >= 0; --i) {
                        auto child = this->strip<ConstVisit>(node->child(i));
                        if (!this->derived().traverse(child)) {
                            return false;
                        }
                    }
                }
            }
            else {
                for (size_t i = 0; i < children; ++i) {
                    auto child = this->strip<ConstVisit>(node->child(i));
                    if (!this->derived().traverse(child)) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
};

} // namespace ast
} // namespace ax

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_VISITOR_HAS_BEEN_INCLUDED

