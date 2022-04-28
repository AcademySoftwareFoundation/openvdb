// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/PrintTree.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Various tools which traverse an AX AST and report information
///   back to a std::ostream.
///

#ifndef OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <iostream>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

struct Node;

/// @brief Writes a descriptive printout of a Node hierarchy into a target
///        stream
/// @param node  Node to print
/// @param numberStatements  Whether to number the line statements
/// @param os  Stream to write into
/// @param indent  The indent to print on each child traversal
OPENVDB_AX_API void print(const ast::Node& node,
    const bool numberStatements = true,
    std::ostream& os = std::cout,
    const char* indent = "    ");

/// @brief Using the provided AST, print corresponding AX code which
///        may have been used to create it.
/// @note  The result is not guaranteed to be exactly equal to the
///        code that was original parsed. A few potential key differences worth
///        mentioning include whitespace matching, component indexing and inferred
///        attribute types.
/// @param node  Node to print
/// @param os    Stream to write into
/// @param indent  The indent to print on each child traversal
OPENVDB_AX_API void reprint(const ast::Node& node,
    std::ostream& os = std::cout,
    const char* indent = "    ");

} // namespace ast
} // namespace ax

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED

