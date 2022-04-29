// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/Logger.h
///
/// @authors Richard Jones
///
/// @brief Logging system to collect errors and warnings throughout the
///   different stages of parsing and compilation.
///

#ifndef OPENVDB_AX_COMPILER_LOGGER_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_LOGGER_HAS_BEEN_INCLUDED

#include "../ast/AST.h"

#include <openvdb/version.h>

#include <functional>
#include <string>
#include <unordered_map>

class TestLogger;

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

/// @brief Logger for collecting errors and warnings that occur during AX
///   compilation.
///
/// @details Error and warning output can be customised using the function
///   pointer arguments. These require a function that takes the formatted error
///   or warning string and handles the output, returning void.
///   e.g.
///      void streamCerr(const std::string& message) {
///          std::cerr << message << std::endl;
///      }
///
///   The Logger handles formatting of messages, tracking of number of errors or
///   warnings and retrieval of errored lines of code to be printed if needed.
///   Use of the Logger to track new errors or warnings can be done either with
///   the line/column numbers directly (e.g during lexing and parsing where the
///   code is being iterated through) or referring to the AST node using its
///   position in the Tree (e.g. during codegen where only the AST node is known
///   directly, not the corresponding line/column numbers). To find the line or
///   column numbers for events logged using AST nodes, the Logger stores a map
///   of Node* to line and column numbers. This must be populated e.g. during
///   parsing, to allow resolution of code locations when they are not
///   explicitly available. The Logger also stores a pointer to the AST Tree
///   that these nodes belong to and the code used to create it.
///
/// @warning  The logger is not thread safe. A unique instance of the Logger
///   should be used for unique invocations of ax pipelines.
class OPENVDB_AX_API Logger
{
public:
    using Ptr = std::shared_ptr<Logger>;

    using CodeLocation = std::pair<size_t, size_t>;
    using OutputFunction = std::function<void(const std::string&)>;

    /// @brief Construct a Logger with optional error and warning output
    ///   functions, defaults stream errors to std::cerr and suppress warnings
    /// @param errors   Optional error output function
    /// @param warnings Optional warning output function
    Logger(const OutputFunction& errors =
        [](const std::string& msg){
            std::cerr << msg << std::endl;
        },
        const OutputFunction& warnings = [](const std::string&){});
    ~Logger();

    /// @brief Log a compiler error and its offending code location. If the
    ///   offending location is (0,0), the message is treated as not having an
    ///   associated code location.
    /// @param message The error message
    /// @param lineCol The line/column number of the offending code
    /// @return true if can continue to capture future messages.
    bool error(const std::string& message, const CodeLocation& lineCol = CodeLocation(0,0));

    /// @brief Log a compiler error using the offending AST node. Used in AST
    ///   traversal.
    /// @param message The error message
    /// @param node The offending AST node causing the error
    /// @return true if can continue to capture future messages.
    bool error(const std::string& message, const ax::ast::Node* node);

    /// @brief Log a compiler warning and its offending code location. If the
    ///   offending location is (0,0), the message is treated as not having an
    ///   associated code location.
    /// @param message The warning message
    /// @param lineCol The line/column number of the offending code
    /// @return true if can continue to capture future messages.
    bool warning(const std::string& message, const CodeLocation& lineCol = CodeLocation(0,0));

    /// @brief Log a compiler warning using the offending AST node. Used in AST
    ///   traversal.
    /// @param message The warning message
    /// @param node The offending AST node causing the warning
    /// @return true if can continue to capture future messages.
    bool warning(const std::string& message, const ax::ast::Node* node);

    ///

    /// @brief Returns the number of errors that have been encountered
    inline size_t errors() const { return mNumErrors; }
    /// @brief Returns the number of warnings that have been encountered
    inline size_t warnings() const { return mNumWarnings; }

    /// @brief Returns true if an error has been found, false otherwise
    inline bool hasError() const { return this->errors() > 0; }
    /// @brief Returns true if a warning has been found, false otherwise
    inline bool hasWarning() const { return this->warnings() > 0; }
    /// @brief Returns true if it has errored and the max errors has been hit
    inline bool atErrorLimit() const  {
        return this->getMaxErrors() > 0 && this->errors() >= this->getMaxErrors();
    }

    /// @brief Clear the tree-code mapping and reset the number of errors/warnings
    /// @note  The tree-code mapping must be repopulated to retrieve line and
    ///   column numbers during AST traversal i.e. code generation. The
    ///   openvdb::ax::ast::parse() function does this for a given input code
    ///   string.
    void clear();

    /// @brief Set any warnings that are encountered to be promoted to errors
    /// @param warnAsError If true, warnings will be treated as errors
    void setWarningsAsErrors(const bool warnAsError = false);
    /// @brief Returns if warning are promoted to errors
    bool getWarningsAsErrors() const;

    /// @brief Sets the maximum number of errors that are allowed before
    ///   compilation should exit.
    /// @note The logger will continue to increment the error counter beyond
    ///   this value but, once reached, it will not invoke the error callback.
    /// @param maxErrors The number of allowed errors
    void setMaxErrors(const size_t maxErrors = 0);
    /// @brief Returns the number of allowed errors
    size_t getMaxErrors() const;

    /// Error/warning formatting options

    /// @brief Set whether the output should number the errors/warnings
    /// @param numbered If true, messages will be numbered
    void setNumberedOutput(const bool numbered = true);
    /// @brief Number of spaces to indent every new line before the message is formatted
    void setIndent(const size_t ident = 0);
    /// @brief Set a prefix for each warning message
    void setErrorPrefix(const char* prefix = "error: ");
    /// @brief Set a prefix for each warning message
    void setWarningPrefix(const char* prefix = "warning: ");
    /// @brief Set whether the output should include the offending line of code
    /// @param print If true, offending lines of code will be appended to the
    ///   output message
    void setPrintLines(const bool print = true);

    /// @brief Returns whether the messages will be numbered
    bool getNumberedOutput() const;
    /// @brief Returns the number of spaces to be printed before every new line
    size_t getIndent() const;
    /// @brief Returns the prefix for each error message
    const char* getErrorPrefix() const;
    /// @brief Returns the prefix for each warning message
    const char* getWarningPrefix() const;
    /// @brief Returns whether the messages will include the line of offending code
    bool getPrintLines() const;

    /// @brief Set the source code that lines can be printed from if an error or
    ///   warning is raised
    /// @param code The AX code as a c-style string
    void setSourceCode(const char* code);

    /// These functions are only to be used during parsing to allow line and
    /// column number retrieval during later stages of compilation when working
    /// solely with an AST

    /// @brief Set the AST source tree which will be used as reference for the
    ///   locations of nodes when resolving line and column numbers during AST
    ///    traversal
    /// @note  To be used just by ax::parse before any AST modifications to
    ///   ensure traversal of original source tree is possible, when adding
    ///   messages using Node* which may correspond to modified trees
    /// @param tree Pointer to const AST
    void setSourceTree(openvdb::ax::ast::Tree::ConstPtr tree);

    /// @brief Add a node to the code location map
    /// @param node     Pointer to AST node
    /// @param location Line and column number in code
    void addNodeLocation(const ax::ast::Node* node, const CodeLocation& location);

    // forward declaration
    struct Settings;
    struct SourceCode;

private:

    friend class ::TestLogger;

    OutputFunction mErrorOutput;
    OutputFunction mWarningOutput;

    size_t mNumErrors;
    size_t mNumWarnings;

    std::unique_ptr<Settings> mSettings;

    // components needed for verbose error info i.e. line/column numbers and
    // lines from source code
    std::unique_ptr<SourceCode> mCode;
    ax::ast::Tree::ConstPtr mTreePtr;
    std::unordered_map<const ax::ast::Node*, CodeLocation> mNodeToLineColMap;
};

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_LOGGER_HAS_BEEN_INCLUDED

