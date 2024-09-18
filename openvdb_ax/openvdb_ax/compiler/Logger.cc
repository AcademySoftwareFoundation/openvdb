// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file compiler/Logger.cc

#include "Logger.h"

#include <openvdb/util/Assert.h>

#include <stack>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {

struct Logger::Settings
{
    size_t      mMaxErrors = 0;
    bool        mWarningsAsErrors = false;
    // message formatting settings
    bool        mNumbered = true;
    const char* mErrorPrefix = "error: ";
    const char* mWarningPrefix = "warning: ";
    bool        mPrintLines = false;
    std::string mIndent;
};


/// @brief Wrapper around a code snippet to print individual lines from a multi
///   line string
/// @note  Assumes a null terminated c-style string input
struct Logger::SourceCode
{
    SourceCode(const char* string = nullptr)
        : mString(string)
        , mOffsets()
        , mLines() {
        reset(string);
    }

    /// @brief Print a line of the multi-line string to the stream
    /// @note  If no string hs been provided, will do nothing
    /// @param line  Line number to print
    /// @param os    Output stream
    void getLine(const size_t num, std::ostream* os)
    {
        if (num < 1) return;
        if (mOffsets.empty()) getLineOffsets();
        if (num > mLines) return;
        const size_t start = mOffsets[num - 1];
        const size_t end = mOffsets[num];
        for (size_t i = start; i < end - 1; ++i) *os << mString[i];
    }

    void reset(const char* string)
    {
        mString = string;
        mOffsets.clear();
        mLines = 0;
    }

    bool hasString() const { return static_cast<bool>(mString); }

private:

    void getLineOffsets()
    {
        if (!mString) return;
        mOffsets.emplace_back(0);
        size_t offset = 1;
        const char* iter = mString;
        while (*iter != '\0') {
            if (*iter == '\n') mOffsets.emplace_back(offset);
            ++iter; ++offset;
        }
        mOffsets.emplace_back(offset);
        mLines = mOffsets.size();
    }

    const char*         mString;
    std::vector<size_t> mOffsets;
    size_t              mLines;
};

namespace {

/// @brief Return a stack denoting the position in the tree of this node
///        Where each node is represented by its childidx of its parent
///        This gives a branching path to follow to reach this node from the
///        tree root
/// @parm  node   Node pointer to create position stack for
inline std::stack<size_t> pathStackFromNode(const ast::Node* node)
{
    std::stack<size_t> path;
    const ast::Node* child = node;
    const ast::Node* parent = node->parent();
    while (parent) {
        path.emplace(child->childidx());
        child = parent;
        parent = child->parent();
    }
    return path;
}

/// @brief Iterate through a tree, following the branch numbers from the path
///   stack, returning a Node* to the node at this position
/// @parm path   Stack of child branches to follow
/// @parm tree   Tree containing node to return
inline const ast::Node* nodeFromPathStack(std::stack<size_t>& path,
                                          const ast::Tree& tree)
{
    const ast::Node* node = &tree;
    while (node) {
        if (path.empty()) return node;
        node = node->child(path.top());
        path.pop();
    }
    return nullptr;
}

/// @brief Given any node and a tree and node to location map, return the line
///   and column number for the nodes equivalent (in terms of position in the
///   tree) from the supplied tree
/// @note  Requires the map to have been populated for all nodes in the supplied
///   tree, otherwise will return 0:0
inline const Logger::CodeLocation
nodeToCodeLocation(const ast::Node* node,
                const ast::Tree::ConstPtr tree,
                const std::unordered_map
                    <const ax::ast::Node*, Logger::CodeLocation>& map)
{
    if (!tree) return Logger::CodeLocation(0,0);
    OPENVDB_ASSERT(node);
    std::stack<size_t> pathStack = pathStackFromNode(node);
    const ast::Node* nodeInMap = nodeFromPathStack(pathStack, *tree);
    const auto locationIter = map.find(nodeInMap);
    if (locationIter == map.end()) return Logger::CodeLocation(0,0);
    return locationIter->second;
}

std::string format(const std::string& message,
                   const Logger::CodeLocation& loc,
                   const size_t numMessage,
                   const bool numbered,
                   const bool printLines,
                   const std::string& indent,
                   Logger::SourceCode* sourceCode)
{
    std::stringstream ss;
    ss << indent;
    if (numbered) ss << "[" << numMessage << "] ";
    for (auto c : message) {
        ss << c;
        if (c == '\n') ss << indent;
    }
    if (loc.first > 0) {
        ss << " " << loc.first << ":" << loc.second;
        if (printLines && sourceCode) {
            ss << '\n' << indent;
            sourceCode->getLine(loc.first, &ss);
            ss << '\n' << indent;
            for (size_t i = 0; i < loc.second - 1; ++i) ss << '-';
            ss << '^';
        }
    }
    return ss.str();
}

}

Logger::Logger(const Logger::OutputFunction& errors,
               const Logger::OutputFunction& warnings)
    : mErrorOutput(errors)
    , mWarningOutput(warnings)
    , mNumErrors(0)
    , mNumWarnings(0)
    , mSettings(new Logger::Settings())
    , mCode() {}

Logger::~Logger() {}

void Logger::setSourceCode(const char* code)
{
    mCode.reset(new SourceCode(code));
}

bool Logger::error(const std::string& message,
                   const Logger::CodeLocation& lineCol)
{
    // check if we've already exceeded the error limit
    const bool limit = this->atErrorLimit();
    // Always increment the error counter
    ++mNumErrors;
    if (limit) return false;
    mErrorOutput(format(this->getErrorPrefix() + message,
                        lineCol,
                        this->errors(),
                        this->getNumberedOutput(),
                        this->getPrintLines(),
                        this->mSettings->mIndent,
                        this->mCode.get()));
    return !this->atErrorLimit();
}

bool Logger::error(const std::string& message,
                   const ax::ast::Node* node)
{
    return this->error(message, nodeToCodeLocation(node, mTreePtr, mNodeToLineColMap));
}

bool Logger::warning(const std::string& message,
                   const Logger::CodeLocation& lineCol)
{
    if (this->getWarningsAsErrors()) {
        return this->error(message + " [warning-as-error]", lineCol);
    }
    else {
        ++mNumWarnings;
        mWarningOutput(format(this->getWarningPrefix() + message,
                              lineCol,
                              this->warnings(),
                              this->getNumberedOutput(),
                              this->getPrintLines(),
                              this->mSettings->mIndent,
                              this->mCode.get()));
        return true;
    }
}

bool Logger::warning(const std::string& message,
                   const ax::ast::Node* node)
{
    return this->warning(message, nodeToCodeLocation(node, mTreePtr, mNodeToLineColMap));
}

void Logger::setWarningsAsErrors(const bool warnAsError)
{
    mSettings->mWarningsAsErrors = warnAsError;
}

bool Logger::getWarningsAsErrors() const
{
    return mSettings->mWarningsAsErrors;
}

void Logger::setMaxErrors(const size_t maxErrors)
{
    mSettings->mMaxErrors = maxErrors;
}

size_t Logger::getMaxErrors() const
{
    return mSettings->mMaxErrors;
}

void Logger::setNumberedOutput(const bool numbered)
{
    mSettings->mNumbered = numbered;
}

void Logger::setIndent(const size_t ident)
{
    mSettings->mIndent = std::string(ident, ' ');
}

void Logger::setErrorPrefix(const char* prefix)
{
    mSettings->mErrorPrefix = prefix;
}

void Logger::setWarningPrefix(const char* prefix)
{
    mSettings->mWarningPrefix = prefix;
}

void Logger::setPrintLines(const bool print)
{
    mSettings->mPrintLines = print;
}

bool Logger::getNumberedOutput() const
{
    return mSettings->mNumbered;
}

size_t Logger::getIndent() const
{
    return mSettings->mIndent.size();
}

const char* Logger::getErrorPrefix() const
{
    return mSettings->mErrorPrefix;
}

const char* Logger::getWarningPrefix() const
{
    return mSettings->mWarningPrefix;
}

bool Logger::getPrintLines() const
{
    return mSettings->mPrintLines;
}

void Logger::clear()
{
    mCode.reset();
    mNumErrors = 0;
    mNumWarnings = 0;
    mNodeToLineColMap.clear();
    mTreePtr = nullptr;
}

void Logger::setSourceTree(openvdb::ax::ast::Tree::ConstPtr tree)
{
    mTreePtr = tree;
}

void Logger::addNodeLocation(const ax::ast::Node* node, const Logger::CodeLocation& location)
{
    mNodeToLineColMap.emplace(node, location);
}

} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

