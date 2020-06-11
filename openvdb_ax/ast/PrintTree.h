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

/// @file ast/PrintTree.h
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  Various tools which traverse an AX AST and report information
///   back to a std::ostream.
///

#ifndef OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED

#include <openvdb_ax/version.h>

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
void print(const ast::Node& node,
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
void reprint(const ast::Node& node,
    std::ostream& os = std::cout,
    const char* indent = "    ");

} // namespace ast
} // namespace ax

}
} // namespace openvdb

#endif // OPENVDB_AX_AST_PRINT_TREE_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
