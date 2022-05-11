// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Parse.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief Parsing methods for creating abstract syntax trees out of AX code
///

#ifndef OPENVDB_AX_PARSE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_PARSE_HAS_BEEN_INCLUDED

#include "AST.h"
#include "../compiler/Logger.h"

#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

/// @brief   Construct an abstract syntax tree from a code snippet.
/// @details This method parses the provided null terminated code snippet and
///   attempts to construct a complete abstract syntax tree (AST) which can be
///   passed to the AX Compiler. If the code is not well formed (as defined by
///   the AX grammar) a nullptr is returned and instances of any errors
///   encoutered are stored to the provided logger.
/// @note   The returned AST is const as the logger uses this to determine line
///   and column numbers of errors/warnings in later stages. If you need to
///   modify the tree, take a copy.
///
/// @return A shared pointer to a valid const AST. Can be a nullptr on error.
/// @todo  In the future it may be useful for ::parse to return as much of
///   the valid AST that exists.
///
/// @param code    The code to parse
/// @param logger  The logger to collect syntax errors
///
OPENVDB_AX_API openvdb::ax::ast::Tree::ConstPtr
parse(const char* code, ax::Logger& logger);

/// @brief   Construct an abstract syntax tree from a code snippet.
/// @details A runtime exception will be thrown with the first syntax error.
///
/// @return A shared pointer to a valid AST.
///
/// @param code The code to parse
///
OPENVDB_AX_API openvdb::ax::ast::Tree::Ptr
parse(const char* code);

} // namespace ast
} // namespace ax

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_HAS_BEEN_INCLUDED

