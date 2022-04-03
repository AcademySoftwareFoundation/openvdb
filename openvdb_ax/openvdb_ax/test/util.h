// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file test/util.h
///
/// @author Nick Avramoussis
///
/// @brief  Test utilities

#ifndef OPENVDB_AX_UNITTEST_UTIL_HAS_BEEN_INCLUDED
#define OPENVDB_AX_UNITTEST_UTIL_HAS_BEEN_INCLUDED

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/PrintTree.h>
#include <openvdb_ax/ast/Parse.h>
#include <openvdb_ax/ast/Tokens.h>
#include <openvdb_ax/compiler/Logger.h>

#include <openvdb/Types.h>

#include <memory>
#include <vector>
#include <utility>
#include <string>
#include <type_traits>
#include <map>

#define ERROR_MSG(Msg, Code) Msg + std::string(": \"") + Code + std::string("\"")

#define TEST_SYNTAX_PASSES(Tests) \
{ \
    openvdb::ax::Logger logger;\
    for (const auto& test : Tests) { \
        logger.clear();\
        const std::string& code = test.first; \
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(code.c_str(), logger);\
        std::stringstream str; \
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Unexpected parsing error(s)\n", str.str()), tree && !logger.hasError()); \
    } \
} \

#define TEST_SYNTAX_FAILS(Tests) \
{ \
    openvdb::ax::Logger logger([](const std::string&) {});\
    for (const auto& test : Tests) { \
        logger.clear();\
        const std::string& code = test.first; \
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse(code.c_str(), logger);\
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Expected parsing error", code), !tree && logger.hasError()); \
    } \
} \

namespace unittest_util
{
// Use shared pointers rather than unique pointers so initializer lists can easily
// be used. Could easily introduce some move semantics to work around this if
// necessary.
using CodeTests = std::vector<std::pair<std::string, openvdb::ax::ast::Node::Ptr>>;
// Ordered map for consistency across platforms
using ConfigMap = std::map<std::string, std::map<std::string, std::string>>;

//

// Find + Replace all string helper
inline void replace(std::string& str, const std::string& oldStr, const std::string& newStr)
{
    std::string::size_type pos = 0u;
    while ((pos = str.find(oldStr, pos)) != std::string::npos) {
        str.replace(pos, oldStr.length(), newStr);
        pos += newStr.length();
    }
}

//

inline bool compareLinearTrees(const std::vector<const openvdb::ax::ast::Node*>& a,
    const std::vector<const openvdb::ax::ast::Node*>& b, const bool allowEmpty = false)
{
    if (!allowEmpty && (a.empty() || b.empty())) return false;
    if (a.size() != b.size()) return false;
    const size_t size = a.size();
    for (size_t i = 0; i < size; ++i) {
        if ((a[i] == nullptr) ^ (b[i] == nullptr)) return false;
        if (a[i] == nullptr) continue;
        if (a[i]->nodetype() != b[i]->nodetype()) return false;

        // Specific handling of various node types to compare child data
        // @todo generalize this
        // @note  Value methods does not compare child text data

        if (a[i]->nodetype() == openvdb::ax::ast::Node::AssignExpressionNode) {
            if (static_cast<const openvdb::ax::ast::AssignExpression*>(a[i])->operation() !=
                static_cast<const openvdb::ax::ast::AssignExpression*>(b[i])->operation()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::BinaryOperatorNode) {
            if (static_cast<const openvdb::ax::ast::BinaryOperator*>(a[i])->operation() !=
                static_cast<const openvdb::ax::ast::BinaryOperator*>(b[i])->operation()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::CrementNode) {
            if (static_cast<const openvdb::ax::ast::Crement*>(a[i])->operation() !=
                static_cast<const openvdb::ax::ast::Crement*>(b[i])->operation()) {
                return false;
            }
            if (static_cast<const openvdb::ax::ast::Crement*>(a[i])->post() !=
                static_cast<const openvdb::ax::ast::Crement*>(b[i])->post()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::CastNode) {
            if (static_cast<const openvdb::ax::ast::Cast*>(a[i])->type() !=
                static_cast<const openvdb::ax::ast::Cast*>(b[i])->type()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::FunctionCallNode) {
            if (static_cast<const openvdb::ax::ast::FunctionCall*>(a[i])->name() !=
                static_cast<const openvdb::ax::ast::FunctionCall*>(b[i])->name()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::LoopNode) {
            if (static_cast<const openvdb::ax::ast::Loop*>(a[i])->loopType() !=
                static_cast<const openvdb::ax::ast::Loop*>(b[i])->loopType()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::KeywordNode) {
            if (static_cast<const openvdb::ax::ast::Keyword*>(a[i])->keyword() !=
                static_cast<const openvdb::ax::ast::Keyword*>(b[i])->keyword()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::AttributeNode) {
            if (static_cast<const openvdb::ax::ast::Attribute*>(a[i])->type() !=
                static_cast<const openvdb::ax::ast::Attribute*>(b[i])->type()) {
                return false;
            }
            if (static_cast<const openvdb::ax::ast::Attribute*>(a[i])->name() !=
                static_cast<const openvdb::ax::ast::Attribute*>(b[i])->name()) {
                return false;
            }
            if (static_cast<const openvdb::ax::ast::Attribute*>(a[i])->inferred() !=
                static_cast<const openvdb::ax::ast::Attribute*>(b[i])->inferred()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ExternalVariableNode) {
            if (static_cast<const openvdb::ax::ast::ExternalVariable*>(a[i])->type() !=
                static_cast<const openvdb::ax::ast::ExternalVariable*>(b[i])->type()) {
                return false;
            }
            if (static_cast<const openvdb::ax::ast::ExternalVariable*>(a[i])->name() !=
                static_cast<const openvdb::ax::ast::ExternalVariable*>(b[i])->name()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::DeclareLocalNode) {
            if (static_cast<const openvdb::ax::ast::DeclareLocal*>(a[i])->type() !=
                static_cast<const openvdb::ax::ast::DeclareLocal*>(b[i])->type()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::LocalNode) {
            if (static_cast<const openvdb::ax::ast::Local*>(a[i])->name() !=
                static_cast<const openvdb::ax::ast::Local*>(b[i])->name()) {
                return false;
            }
        }
        // @note  Value methods does not compare child text data
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueBoolNode) {
            if (static_cast<const openvdb::ax::ast::Value<bool>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<bool>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueInt16Node) {
            if (static_cast<const openvdb::ax::ast::Value<int16_t>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<int16_t>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueInt32Node) {
            if (static_cast<const openvdb::ax::ast::Value<int32_t>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<int32_t>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueInt64Node) {
            if (static_cast<const openvdb::ax::ast::Value<int64_t>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<int64_t>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueFloatNode) {
            if (static_cast<const openvdb::ax::ast::Value<float>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<float>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueDoubleNode) {
            if (static_cast<const openvdb::ax::ast::Value<double>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<double>*>(b[i])->value()) {
                return false;
            }
        }
        else if (a[i]->nodetype() == openvdb::ax::ast::Node::ValueStrNode) {
            if (static_cast<const openvdb::ax::ast::Value<std::string>*>(a[i])->value() !=
                static_cast<const openvdb::ax::ast::Value<std::string>*>(b[i])->value()) {
                return false;
            }
        }
    }
    return true;
}

inline std::vector<std::string>
nameSequence(const std::string& base, const size_t number)
{
    std::vector<std::string> names;
    if (number <= 0) return names;
    names.reserve(number);

    for (size_t i = 1; i <= number; i++) {
        names.emplace_back(base + std::to_string(i));
    }

    return names;
}
}

#endif // OPENVDB_AX_UNITTEST_UTIL_HAS_BEEN_INCLUDED

