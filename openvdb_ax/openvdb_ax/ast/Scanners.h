// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Scanners.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Retrieve intrinsic information from AX AST by performing
///   various traversal algorithms.
///

#ifndef OPENVDB_AX_COMPILER_AST_SCANNERS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_AST_SCANNERS_HAS_BEEN_INCLUDED

#include "AST.h"
#include "Visitor.h"

#include <openvdb/version.h>

#include <string>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

/// @brief  Returns whether or not a given branch of an AST reads from or writes
///   to a given attribute.
///
/// @param node  The AST to analyze
/// @param name  the name of the attribute to search for
/// @param type  the type of the attribute to search for. If UNKNOWN, any
///              attribute with the given name is checked.
///
OPENVDB_AX_API bool usesAttribute(const ast::Node& node,
    const std::string& name,
    const tokens::CoreType type = tokens::UNKNOWN);

/// @brief  Returns whether or not a given branch of an AST writes to a given
///   attribute.
///
/// @param node  The AST to analyze
/// @param name  the name of the attribute to search for
/// @param type  the type of the attribute to search for. If UNKNOWN, the first
///              attribute encountered with the given name is checked.
///
OPENVDB_AX_API bool writesToAttribute(const ast::Node& node,
    const std::string& name,
    const tokens::CoreType type = tokens::UNKNOWN);

/// @brief  Returns whether or not a given branch of an AST calls a function
///
/// @param node  The AST to analyze
/// @param name  the name of the function to search for
///
OPENVDB_AX_API bool callsFunction(const ast::Node& node, const std::string& name);

/// @brief todo
OPENVDB_AX_API void catalogueVariables(const ast::Node& node,
        std::vector<const ast::Variable*>* readOnly,
        std::vector<const ast::Variable*>* writeOnly,
        std::vector<const ast::Variable*>* readWrite,
        const bool locals = true,
        const bool attributes = true);

/// @brief  Parse all attributes into three unique vectors which represent how they
///         are accessed within the syntax tree. Read only attributes are stored
///         within the 'readOnly' container (for example @code int a=@a; @endcode),
///         write only attributes in the 'writeOnly' container @code @a=1; @endcode
///         and readWrite attributes in the 'readWrite' container @code @a+=1; @endcode
/// @note   Note that the code generator is able to do this far more efficiently, however
///         this provides simple front-end support for detecting these types of operations
///
/// @param node       The AST to analyze
/// @param readOnly   The unique list of attributes which are only read from
/// @param writeOnly  The unique list of attributes which are only written too
/// @param readWrite  The unique list of attributes which both read from and written too
///
OPENVDB_AX_API void catalogueAttributeTokens(const ast::Node& node,
        std::vector<std::string>* readOnly,
        std::vector<std::string>* writeOnly,
        std::vector<std::string>* readWrite);

/// @brief  Populate a list of attribute names which the given attribute depends on
OPENVDB_AX_API void attributeDependencyTokens(const ast::Tree& tree,
        const std::string& name,
        const tokens::CoreType type,
        std::vector<std::string>& dependencies);

/// @brief  For an AST node of a given type, search for and call a custom
///         const operator() which takes a const reference to every occurrence
///         of the specified node type.
///
/// @param node  The AST to run over
/// @param op    The operator to call on every found AST node of type NodeT
///
template <typename NodeT, typename OpT>
inline void visitNodeType(const ast::Node& node, const OpT& op);

/// @brief  Visit all nodes of a given type and store pointers to them in a
///         provided compatible container
template<typename NodeT, typename ContainerType = std::vector<const NodeT*>>
inline void collectNodeType(const ast::Node& node, ContainerType& array);

/// @brief  Visit all nodes of the given types and store pointers to them in a
///         container of base ast::Node pointers
/// @note   NodeTypeList is expected to be a an openvdb::TypeList object with a
///         list of node types. For example, to collect all Attribute and
///         External Variable ast Nodes:
///
///            using ListT = openvdb::TypeList<ast::Attribute, ast::ExternalVariable>;
///            std::vector<const ast::Node*> nodes;
///            ast::collectNodeTypes<ListT>(tree, nodes);
///
template <typename NodeTypeList, typename ContainerType = std::vector<const Node*>>
inline void collectNodeTypes(const ast::Node& node, ContainerType& array);

/// @brief  Flatten the provided AST branch into a linear list using post order traversal
///
OPENVDB_AX_API void linearize(const ast::Node& node, std::vector<const ast::Node*>& list);

OPENVDB_AX_API const ast::Variable* firstUse(const ast::Node& node, const std::string& token);
OPENVDB_AX_API const ast::Variable* lastUse(const ast::Node& node, const std::string& token);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/// @cond OPENVDB_DOCS_INTERNAL

namespace internal {
template<typename ContainerType, typename T, typename ...Ts>
struct CollectForEach {
    static void exec(const ast::Node&, ContainerType&) {}
};

template<typename ContainerType, typename T, typename ...Ts>
struct CollectForEach<ContainerType, TypeList<T, Ts...>> {
    static void exec(const ast::Node& node, ContainerType& C) {
        collectNodeType<T, ContainerType>(node, C);
        CollectForEach<ContainerType, TypeList<Ts...>>::exec(node, C);
    }
};
}

// @endcond

template<typename NodeT, typename ContainerType>
inline void collectNodeType(const ast::Node& node, ContainerType& array)
{
    visitNodeType<NodeT>(node, [&](const NodeT& node) -> bool {
        array.push_back(&node);
        return true;
    });
}

template <typename NodeTypeList, typename ContainerType>
inline void collectNodeTypes(const ast::Node& node, ContainerType& array)
{
    internal::CollectForEach<ContainerType, NodeTypeList>::exec(node, array);
}

template <typename NodeT, typename OpT, typename Derived = void>
struct VisitNodeType :
    public ast::Visitor<typename std::conditional<
        std::is_same<Derived, void>::value,
        VisitNodeType<NodeT, OpT>,
        Derived>::type>
{
    using VisitorT = typename std::conditional<
        std::is_same<Derived, void>::value,
        VisitNodeType<NodeT, OpT>,
        Derived>::type;

    using ast::Visitor<VisitorT>::traverse;
    using ast::Visitor<VisitorT>::visit;

    inline bool visitNodeHierarchies() const {
        return std::is_abstract<NodeT>::value;
    }

    VisitNodeType(const OpT& op) : mOp(op) {}
    ~VisitNodeType() = default;
    inline bool visit(const NodeT* node) {
        if (node) return mOp(*node);
        return true;
    }
private:
    const OpT& mOp;
};

template <typename NodeT, typename OpT>
inline void visitNodeType(const ast::Node& node, const OpT& op)
{
    VisitNodeType<NodeT, OpT> visitOp(op);
    visitOp.traverse(&node);
}

} // namespace ast
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_AST_SCANNERS_HAS_BEEN_INCLUDED


