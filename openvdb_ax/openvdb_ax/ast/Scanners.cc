// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Scanners.cc

#include "Scanners.h"
#include "Visitor.h"

#include <openvdb/util/Assert.h>

#include <string>
#include <map>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

namespace {

template <typename NodeT, typename OpT>
struct VariableDependencyVisitor :
    public ast::VisitNodeType<NodeT, OpT,
        VariableDependencyVisitor<NodeT, OpT>>
{
    using BaseT = ast::VisitNodeType<NodeT, OpT,
        VariableDependencyVisitor<NodeT, OpT>>;
    using BaseT::traverse;
    using BaseT::visit;

    VariableDependencyVisitor(const OpT& op) : BaseT(op) {}
    ~VariableDependencyVisitor() = default;

    bool traverse(const ast::Loop* loop)
    {
        if (!loop) return true;
        if (!this->traverse(loop->initial())) return false;
        if (!this->traverse(loop->condition())) return false;
        if (!this->traverse(loop->iteration())) return false;
        if (!this->traverse(loop->body())) return false;
        if (!this->visit(loop)) return false;
        return true;
    }
};

/// @brief  For a given variable at a particular position in an AST, find all
///   attributes, locals and external variables which it depends on (i.e. any
///   Attribute, Local or ExternalVariable AST nodes which impacts the given
///   variables value) by recursively traversing through all connected  paths.
///   This includes both direct and indirect influences; for example, a direct
///   assignment "@b = @a;" and an indirect code branch "if (@a) @b = 1";
/// @note  This is position dependent in regards to the given variables location.
///   Any code which writes to this variable after the given usage will not be
///   cataloged in the output dependency vector.
/// @warning  This does not currently handle scoped local variable re-declarations
///   and instead will end up adding matching names as extra dependencies
/// @todo: fix this for scoped variables, capturing of all instances, and not adding
///   dependencies between different branches of conditionals
void variableDependencies(const ast::Variable& var,
        std::vector<const ast::Variable*>& dependencies)
{
    // external variables are read-only i.e. have no dependencies
    if (var.nodetype() == ast::Node::ExternalVariableNode) return;

    // Get the root node
    const ast::Node* root = &var;
    while (const ast::Node* parent = root->parent()) {
        root = parent;
    }

    // collect all occurrences of this var up to and including
    // it's current usage, terminating traversal afterwards
    const bool attributeVisit =
        (var.nodetype() == ast::Node::AttributeNode);

    std::vector<const ast::Variable*> usage;

    auto collect =
        [&var, &usage, attributeVisit]
        (const ast::Variable& use) -> bool
    {
        if (attributeVisit) {
            if (use.nodetype() != ast::Node::AttributeNode) return true;
            const auto& attrib = static_cast<const ast::Attribute&>(var);
            const auto& useAttrib = static_cast<const ast::Attribute&>(use);
            if (attrib.tokenname() != useAttrib.tokenname()) return true;
        }
        else {
            if (use.nodetype() != ast::Node::LocalNode) return true;
            if (use.name() != var.name()) return true;
        }
        usage.emplace_back(&use);
        return &use != &var;
    };

    VariableDependencyVisitor<ast::Variable, decltype(collect)>
        depVisitor(collect);
    depVisitor.traverse(root);

    // The list of nodes which can be considered dependencies to collect
    using ListT = openvdb::TypeList<
        ast::Attribute,
        ast::Local,
        ast::ExternalVariable>;

    // small lambda to check to see if a dep is already being tracked
    auto hasDep = [&](const ast::Variable* dep) -> bool {
        return (std::find(dependencies.cbegin(), dependencies.cend(), dep) !=
            dependencies.cend());
    };
    // recursively traverse all usages and resolve dependencies
    for (const auto& use : usage)
    {
        const ast::Node* child = use;
        // track writable for conditionals
        bool written = false;
        while (const ast::Node* parent = child->parent()) {
            const ast::Node::NodeType type = parent->nodetype();
            if (type == ast::Node::CrementNode) {
                written = true;
                if (!hasDep(use)) {
                    dependencies.emplace_back(use);
                }
            }
            else if (type == ast::Node::ConditionalStatementNode) {
                const ast::ConditionalStatement* conditional =
                    static_cast<const ast::ConditionalStatement*>(parent);
                const ast::Expression* condition = conditional->condition();
                // traverse down and collect variables
                if (child != condition){
                    std::vector<const ast::Variable*> vars;
                    collectNodeTypes<ListT>(*condition, vars);
                    // find next deps
                    for (const ast::Variable* dep : vars) {
                        // don't add this dep if it's not being written to. Unlike
                        // all other visits, the conditionals dictate program flow.
                        // Values in the conditional expression only link to the
                        // current usage if the current usage is being modified
                        if (!written || hasDep(dep)) continue;
                        dependencies.emplace_back(dep);
                        variableDependencies(*dep, dependencies);
                    }
                }
            }
            else if (type == ast::Node::TernaryOperatorNode) {
                const ast::TernaryOperator* ternary =
                    static_cast<const ast::TernaryOperator*>(parent);
                const ast::Expression* condition = ternary->condition();
                // traverse down and collect variables
                if (child != condition) {
                    std::vector<const ast::Variable*> vars;
                    collectNodeTypes<ListT>(*condition, vars);
                    // find next deps
                    for (const ast::Variable* dep : vars) {
                        // don't add this dep if it's not being written to. Unlike
                        // all other visits, the conditionals dictate program flow.
                        // Values in the conditional expression only link to the
                        // current usage if the current usage is being modified
                        if (!written || hasDep(dep)) continue;
                        dependencies.emplace_back(dep);
                        variableDependencies(*dep, dependencies);
                    }
                }
            }
            else if (type == ast::Node::LoopNode) {
                const ast::Loop* loop =
                    static_cast<const ast::Loop*>(parent);
                const ast::Statement* condition = loop->condition();
                // traverse down and collect variables
                if (child != condition) {
                    std::vector<const ast::Variable*> vars;
                    // if the condition is a comma operator the last element determines flow
                    if (condition->nodetype() == ast::Node::NodeType::CommaOperatorNode) {
                        const ast::CommaOperator*
                            comma = static_cast<const ast::CommaOperator*>(condition);
                        if (!comma->empty()) {
                            const ast::Expression* lastExpression = comma->child(comma->size()-1);
                            collectNodeTypes<ListT>(*lastExpression, vars);
                        }
                    }
                    else {
                        collectNodeTypes<ListT>(*condition, vars);
                    }
                    // find next deps
                    for (const ast::Variable* dep : vars) {
                        // don't add this dep if it's not being written to. Unlike
                        // all other visits, the conditionals dictate program flow.
                        // Values in the conditional expression only link to the
                        // current usage if the current usage is being modified
                        if (!written || hasDep(dep)) continue;
                        dependencies.emplace_back(dep);
                        variableDependencies(*dep, dependencies);
                    }
                }

            }
            else if (type == ast::Node::AssignExpressionNode) {
                const ast::AssignExpression* assignment =
                    static_cast<const ast::AssignExpression*>(parent);
                if (assignment->lhs() == child) {
                    written = true;
                    // add self dependency if compound
                    if (assignment->isCompound()) {
                        if (!hasDep(use)) {
                            dependencies.emplace_back(use);
                        }
                    }
                    // traverse down and collect variables
                    std::vector<const ast::Variable*> vars;
                    collectNodeTypes<ListT>(*assignment->rhs(), vars);
                    // find next deps
                    for (const ast::Variable* dep : vars) {
                        if (hasDep(dep)) continue;
                        dependencies.emplace_back(dep);
                        variableDependencies(*dep, dependencies);
                    }
                }
            }
            else if (type == ast::Node::DeclareLocalNode) {
                const ast::DeclareLocal* declareLocal =
                    static_cast<const ast::DeclareLocal*>(parent);
                if (declareLocal->local() == child && declareLocal->hasInit()) {
                    std::vector<const ast::Variable*> vars;
                    written = true;
                    // traverse down and collect variables
                    collectNodeTypes<ListT>(*declareLocal->init(), vars);
                    for (const ast::Variable* dep : vars) {
                        if (hasDep(dep)) continue;
                        dependencies.emplace_back(dep);
                        variableDependencies(*dep, dependencies);
                    }
                }
            }
            else if (type == ast::Node::FunctionCallNode) {
                written = true;
                // @todo  We currently can't detect if attributes are being passed by
                //   pointer and being modified automatically. We have to link this
                //   attribute to any other attribute passes into the function
                const ast::FunctionCall* call =
                    static_cast<const ast::FunctionCall*>(parent);
                // traverse down and collect variables
                std::vector<const ast::Variable*> vars;
                for (size_t i = 0; i < call->children(); ++i) {
                    collectNodeTypes<ListT>(*call->child(i), vars);
                }
                // only append dependencies here if they haven't already been visited
                // due to recursion issues
                for (const ast::Variable* dep : vars) {
                    // make sure the dep doesn't already exist in the container, otherwise
                    // we can get into issues where functions with multiple arguments
                    // constantly try to check themselves
                    // @note  should be removed with function refactoring
                    if (hasDep(dep)) continue;
                    dependencies.emplace_back(dep);
                    variableDependencies(*dep, dependencies);
                }
            }
            child = parent;
        }
    }
}


} // anonymous namespace

bool usesAttribute(const ast::Node& node,
    const std::string& name,
    const tokens::CoreType type)
{
    bool found = false;
    visitNodeType<ast::Attribute>(node,
        [&](const ast::Attribute& attrib) -> bool {
            OPENVDB_ASSERT(!found);
            if (type != tokens::UNKNOWN) {
                if (attrib.type() != type) return true;
            }
            if (attrib.name() != name) return true;
            found = true;
            return false;
        });

    return found;
}

bool writesToAttribute(const ast::Node& node,
    const std::string& name,
    const tokens::CoreType type)
{
    std::vector<const ast::Variable*> vars;
    catalogueVariables(node, nullptr, &vars, &vars, false, true);

    // See if any attributes in the result vec match the given name/type
    for (const ast::Variable* var : vars) {
        OPENVDB_ASSERT(var->isType<ast::Attribute>());
        const ast::Attribute* attrib = static_cast<const ast::Attribute*>(var);
        if (type != tokens::UNKNOWN) {
            if (attrib->type() != type) continue;
        }
        if (attrib->name() != name) continue;
        return true;
    }

    return false;
}

void catalogueVariables(const ast::Node& node,
        std::vector<const ast::Variable*>* readOnly,
        std::vector<const ast::Variable*>* writeOnly,
        std::vector<const ast::Variable*>* readWrite,
        const bool locals,
        const bool attributes)
{
    std::vector<const ast::Variable*> vars;

    if (locals) {
        collectNodeTypes<ast::Local>(node, vars);
    }
    if (attributes) {
        collectNodeType<ast::Attribute>(node, vars);
    }

    for (const ast::Variable* var : vars) {
        // traverse upwards, see if we're embedded in an assign or crement expression
        const ast::Node* child = var;
        const ast::Node* parent = child->parent();
        bool read = false, write = false;
        while (parent && !(write && read)) {
            const ast::Node::NodeType type = parent->nodetype();
            // crement operations read and write
            if (type == ast::Node::CrementNode) {
                read = write = true;
            }
            else if (type == ast::Node::AssignExpressionNode) {
                const ast::AssignExpression* assignment =
                    static_cast<const ast::AssignExpression*>(parent);
                if (assignment->lhs() == child) {
                    if (assignment->isCompound()) {
                        // +=, *=, /= etc
                        read = write = true;
                    }
                    else {
                        // op = op
                        write = true;
                    }
                }
                else {
                    read = true;
                }
            }
            else if (type == ast::Node::DeclareLocalNode) {
                const ast::DeclareLocal* declareLocal =
                    static_cast<const ast::DeclareLocal*>(parent);
                if (declareLocal->local() == child) {
                    if (declareLocal->hasInit()) {
                        write = true;
                    }
                }
            }
            else if (type == ast::Node::FunctionCallNode) {
                // @todo  We currently can't detect if attributes are being passed by
                //   pointer and being modified automatically. This is a major limitation
                //   as it means any attribute passed into any function directly must
                //   be marked as writeable
                read = write = true;
            }
            else {
                read = true;
            }
            child = parent;
            parent = child->parent();
        }

        OPENVDB_ASSERT(read || write);
        if (readWrite && read && write)  readWrite->emplace_back(var);
        if (readOnly && read && !write)  readOnly->emplace_back(var);
        if (writeOnly && !read && write) writeOnly->emplace_back(var);
    }
}

void catalogueAttributeTokens(const ast::Node& node,
        std::vector<std::string>* readOnly,
        std::vector<std::string>* writeOnly,
        std::vector<std::string>* readWrite)
{
    std::vector<const ast::Variable*> readOnlyVars;
    std::vector<const ast::Variable*> writeOnlyVars;
    std::vector<const ast::Variable*> readWriteVars;
    catalogueVariables(node,
        (readOnly ? &readOnlyVars : nullptr),
        (writeOnly ? &writeOnlyVars : nullptr),
        (readWrite ? &readWriteVars : nullptr),
        false, // locals
        true); // attributes

    // fill a single map with the access patterns for all attributes
    // .first = read, .second = write
    // @note  use a map rather than an unordered_map to preserve order
    //   of the output vectors on different platforms (the AX compiler
    //   doesn't care about the order but it's reasonable to expect
    //   an attribute has the same index from one platform to the next).
    std::map<std::string, std::pair<bool,bool>> accessmap;

    auto addAccesses = [&](const std::vector<const ast::Variable*>& vars,
        const bool read,
        const bool write)
    {
        for (const ast::Variable* var : vars) {
            OPENVDB_ASSERT(var->isType<ast::Attribute>());
            const ast::Attribute* attrib = static_cast<const ast::Attribute*>(var);
            auto& access = accessmap[attrib->tokenname()];
            access.first |= read;
            access.second |= write;
        }
    };

    addAccesses(readWriteVars, true, true);
    addAccesses(writeOnlyVars, false, true);
    addAccesses(readOnlyVars, true, false);

    // set the results from the access map
    for (const auto& result : accessmap) {
        const std::pair<bool,bool>& pair = result.second;
        if (readWrite && pair.first && pair.second) {
            readWrite->emplace_back(result.first);
        }
        else if (writeOnly && !pair.first && pair.second) {
            writeOnly->emplace_back(result.first);
        }
        else if (readOnly && pair.first && !pair.second) {
            readOnly->emplace_back(result.first);
        }
    }
}

template <bool First>
struct UseVisitor :
    public ast::Visitor<UseVisitor<First>>
{
    using ast::Visitor<UseVisitor<First>>::traverse;
    using ast::Visitor<UseVisitor<First>>::visit;

    // reverse the ast traversal if !First
    inline bool reverseChildVisits() const { return !First; }

    UseVisitor(const std::string& tokenOrName)
        : mToken(tokenOrName)
        , mAttribute(false)
        , mVar(nullptr) {
            // rebuild the expected token if necessary
            std::string name, type;
            mAttribute = ast::Attribute::nametypeFromToken(mToken, &name, &type);
            if (mAttribute) {
                mToken = type + ast::Attribute::symbolseparator() + name;
            }
        }
    ~UseVisitor() = default;

    bool traverse(const ast::Loop* loop)
    {
        if (!loop) return true;
        const ast::tokens::LoopToken type = loop->loopType();
        if (type == ast::tokens::DO) {
            if (!this->reverseChildVisits()) {
                if (!this->traverse(loop->body())) return false;
                if (!this->traverse(loop->condition())) return false;
            }
            else {
                if (!this->traverse(loop->condition())) return false;
                if (!this->traverse(loop->body())) return false;
            }
            OPENVDB_ASSERT(!loop->initial());
            OPENVDB_ASSERT(!loop->iteration());
        }
        else {
            if (!this->reverseChildVisits()) {
                if (!this->traverse(loop->initial())) return false;
                if (!this->traverse(loop->condition())) return false;
                if (!this->traverse(loop->iteration())) return false;
                if (!this->traverse(loop->body())) return false;
            }
            else {
                if (!this->traverse(loop->body())) return false;
                if (!this->traverse(loop->iteration())) return false;
                if (!this->traverse(loop->condition())) return false;
                if (!this->traverse(loop->initial())) return false;
            }
        }

        if (!this->visit(loop)) return false;
        return true;
    }

    inline bool visit(const ast::Attribute* node) {
        if (!mAttribute) return true;
        if (node->tokenname() != mToken) return true;
        mVar = node;
        return false;
    }
    inline bool visit(const ast::Local* node) {
        if (mAttribute) return true;
        if (node->name() != mToken) return true;
        mVar = node;
        return false;
    }

    const ast::Variable* var() const { return mVar; }
private:
    std::string mToken;
    bool mAttribute;
    const ast::Variable* mVar;
};

void attributeDependencyTokens(const ast::Tree& tree,
        const std::string& name,
        const tokens::CoreType type,
        std::vector<std::string>& dependencies)
{
    const std::string token = ast::Attribute::tokenFromNameType(name, type);
    const ast::Variable* var = lastUse(tree, token);
    if (!var) return;
    OPENVDB_ASSERT(var->isType<ast::Attribute>());

    std::vector<const ast::Variable*> deps;
    variableDependencies(*var, deps);

    for (const auto& dep : deps) {
        if (dep->nodetype() != ast::Node::AttributeNode) continue;
        dependencies.emplace_back(static_cast<const ast::Attribute*>(dep)->tokenname());
    }

    std::sort(dependencies.begin(), dependencies.end());
    auto iter = std::unique(dependencies.begin(), dependencies.end());
    dependencies.erase(iter, dependencies.end());
}

const ast::Variable* firstUse(const ast::Node& node, const std::string& tokenOrName)
{
    UseVisitor<true> visitor(tokenOrName);
    visitor.traverse(&node);
    return visitor.var();
}

const ast::Variable* lastUse(const ast::Node& node, const std::string& tokenOrName)
{
    UseVisitor<false> visitor(tokenOrName);
    visitor.traverse(&node);
    return visitor.var();
}

bool callsFunction(const ast::Node& node, const std::string& name)
{
    bool found = false;
    visitNodeType<ast::FunctionCall>(node,
        [&](const ast::FunctionCall& call) -> bool {
            if (call.name() != name) return true;
            found = true;
            return false;
        });

    return found;
}

void linearize(const ast::Node& node, std::vector<const ast::Node*>& list)
{
    collectNodeType<ast::Node>(node, list);
}

} // namespace ast
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


