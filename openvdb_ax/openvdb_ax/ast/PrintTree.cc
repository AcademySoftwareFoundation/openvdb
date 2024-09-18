// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/PrintTree.cc

#include "AST.h"
#include "PrintTree.h"
#include "Tokens.h"
#include "Visitor.h"

#include <openvdb/util/Assert.h>

#include <ostream>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

struct PrintVisitor : public ast::Visitor<PrintVisitor>
{
    PrintVisitor(std::ostream& os,
        const bool numberStatements = true,
        const char* indent = "    ")
        : mOs(os)
        , mLevel(0)
        , mStatementNumber(numberStatements ? 0 : -1)
        , mIndent(indent) {}
    ~PrintVisitor() = default;

    using ast::Visitor<PrintVisitor>::traverse;
    using ast::Visitor<PrintVisitor>::visit;

    inline bool postOrderNodes() const { return false; }

    inline void indent() {
        for (int i(0); i < mLevel; ++i) mOs << mIndent;
    }

    bool traverse(NodeType<ast::StatementList>* stmntl) {
        this->visit(stmntl);

        const size_t children = stmntl->children();
        ++mLevel;
        if (children == 0) {
            indent();
            mOs << "<empty>\n";
        }
        else {
            for (size_t i = 0; i < children; ++i) {
                indent();
                if (mStatementNumber >= 0) {
                    mOs << '[' << mStatementNumber++ << "] ";
                }
                this->derived().traverse(stmntl->child(i));
                mOs << '\n';
            }
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::Block>* block) {
        indent();
        this->visit(block);

        const size_t children = block->children();
        ++mLevel;
        if (children == 0) {
            indent();
            mOs << "<empty>\n";
        }
        else {
            for (size_t i = 0; i < children; ++i) {
                indent();
                if (mStatementNumber >= 0) {
                    mOs << '[' << mStatementNumber++ << "] ";
                }
                this->derived().traverse(block->child(i));
                mOs << '\n';
            }
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::CommaOperator>* comma) {
        this->visit(comma);
        const size_t children = comma->children();
        ++mLevel;
        if (children == 0) {
            indent();
            mOs << "<empty>\n";
        }
        else {
            for (size_t i = 0; i < children; ++i) {
                indent();
                this->derived().traverse(comma->child(i));
            }
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::ConditionalStatement>* cond) {
        this->visit(cond);
        ++mLevel;

        indent();
        mOs << "condition:\n";

        ++mLevel;
        indent();
        this->traverse(cond->condition());
        --mLevel;

        indent();
        mOs << "branch [true]:\n";

        this->traverse(cond->trueBranch());

        if (cond->hasFalse()) {
            indent();
            mOs << "branch [false]:\n";

            this->traverse(cond->falseBranch());
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::TernaryOperator>* tern) {
        this->visit(tern);
        ++mLevel;

        indent();
        mOs << "condition:\n";

        ++mLevel;
        indent();
        this->traverse(tern->condition());
        --mLevel;

        indent();
        mOs << "true:\n";
        if (tern->hasTrue()) {
            ++mLevel;
            indent();
            this->traverse(tern->trueBranch());
            --mLevel;
        }

        indent();
        mOs << "false:\n";
        ++mLevel;
        indent();
        this->traverse(tern->falseBranch());
        --mLevel;

        --mLevel;
        return true;
    }


    bool traverse(NodeType<ast::Loop>* loop) {
        this->visit(loop);
        ++mLevel;

        if (loop->hasInit()) {
            indent();
            mOs << "init:\n";

            ++mLevel;
            indent();
            this->traverse(loop->initial());
            --mLevel;
        }

        indent();
        mOs << "conditional:\n";

        ++mLevel;
        indent();
        this->traverse(loop->condition());
        --mLevel;

        if (loop->hasIter()) {
            indent();
            mOs << "iterator:\n";
            ++mLevel;
            indent();
            this->traverse(loop->iteration());
            --mLevel;
        }

        indent();
        mOs << "body:\n";
        this->traverse(loop->body());

        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::AssignExpression>* asgn) {
        this->visit(asgn);
        ++mLevel;
        indent();
        this->traverse(asgn->lhs());
        indent();
        this->traverse(asgn->rhs());
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::DeclareLocal>* asgn) {
        this->visit(asgn);
        ++mLevel;
        indent();
        this->traverse(asgn->local());
        if(asgn->hasInit()) {
            indent();
            mOs << "init:\n";
            ++mLevel;
            indent();
            this->traverse(asgn->init());
            --mLevel;
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::Crement>* crmt) {
        this->visit(crmt);
        ++mLevel;
        indent();
        this->traverse(crmt->expression());
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::UnaryOperator>* unry) {
        this->visit(unry);
        ++mLevel;
        indent();
        this->traverse(unry->expression());
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::BinaryOperator>* bin) {
        this->visit(bin);
        ++mLevel;
        indent();
        this->traverse(bin->lhs());
        indent();
        this->traverse(bin->rhs());
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::Cast>* cast) {
        this->visit(cast);
        ++mLevel;
        indent();
        this->traverse(cast->expression());
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::FunctionCall>* call) {
        this->visit(call);
        const size_t children = call->children();
        ++mLevel;
        if (children == 0) {
            indent();
            mOs << "<empty>\n";
        }
        else {
            for (size_t i = 0; i < children; ++i) {
                indent();
                this->derived().traverse(call->child(i));
            }
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::ArrayPack>* pack) {
        this->visit(pack);
        const size_t children = pack->children();
        ++mLevel;
        if (children == 0) {
            indent();
            mOs << "<empty>\n";
        }
        else {
            for (size_t i = 0; i < children; ++i) {
                indent();
                this->derived().traverse(pack->child(i));
            }
        }
        --mLevel;
        return true;
    }

    bool traverse(NodeType<ast::ArrayUnpack>* pack) {
        this->visit(pack);
        ++mLevel;
        indent();
        mOs << "expression: ";
        this->traverse(pack->expression());
        indent();
        mOs << "component(s) : ";
        this->traverse(pack->component0());
        this->traverse(pack->component1());
        --mLevel;
        return true;
    }

    bool visit(const ast::StatementList* node);
    bool visit(const ast::Block* node);
    bool visit(const ast::ConditionalStatement* node);
    bool visit(const ast::Loop* node);
    bool visit(const ast::Keyword* node);
    bool visit(const ast::AssignExpression* node);
    bool visit(const ast::Crement* node);
    bool visit(const ast::CommaOperator* node);
    bool visit(const ast::UnaryOperator* node);
    bool visit(const ast::BinaryOperator* node);
    bool visit(const ast::TernaryOperator* node);
    bool visit(const ast::Cast* node);
    bool visit(const ast::FunctionCall* node);
    bool visit(const ast::Attribute* node);
    bool visit(const ast::ExternalVariable* node);
    bool visit(const ast::DeclareLocal* node);
    bool visit(const ast::Local* node);
    bool visit(const ast::ArrayUnpack* node);
    bool visit(const ast::ArrayPack* node);

    bool visit(const ast::Value<bool>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int16_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int32_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int64_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<float>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<double>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<std::string>* node) {
        return this->visitValue(node);
    }

protected:
    template <typename T>
    bool visitValue(const ast::Value<T>* node);

private:
    std::ostream& mOs;
    int mLevel;
    int64_t mStatementNumber;
    const char* mIndent;
};

bool PrintVisitor::visit(const ast::StatementList* node)
{
    mOs << node->nodename() << ": " << node->size() << " statement(s)" << '\n';
    return true;
}


bool PrintVisitor::visit(const ast::Block* node)
{
    mOs << node->nodename() << ": " << node->size() << " statement(s)" << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::ConditionalStatement* node)
{
    mOs << node->nodename() << ": " << (node->hasFalse() ? "two branches " : "one branch") << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::AssignExpression* node)
{
    mOs << node->nodename() << ": " << tokens::operatorNameFromToken(node->operation());
    if (node->isCompound()) mOs << '=';
    mOs << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::Loop* node)
{
    mOs << tokens::loopNameFromToken(node->loopType()) <<" "<< node->nodename() << ":" << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::Keyword* node)
{
    mOs << node->nodename() << ": " << tokens::keywordNameFromToken(node->keyword()) << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::Crement* node)
{
    mOs << node->nodename() << ':';
    if (node->post()) mOs << " post-";
    else              mOs << " pre-";
    if (node->increment()) mOs << "increment";
    else                   mOs << "decrement";
    mOs << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::CommaOperator* node)
{
    mOs << node->nodename() << ": " << node->size() << " element(s)" << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::UnaryOperator* node)
{
    mOs << node->nodename() << ": " <<  tokens::operatorNameFromToken(node->operation()) << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::BinaryOperator* node)
{
    mOs << node->nodename() << ": " <<  tokens::operatorNameFromToken(node->operation()) << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::TernaryOperator* node)
{
    mOs << node->nodename() << ":\n";
    return true;
}

bool PrintVisitor::visit(const ast::Cast* node)
{
    mOs << node->nodename() << ": " << tokens::typeStringFromToken(node->type()) << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::FunctionCall* node)
{
    mOs << node->nodename() << ": " << node->name() << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::Attribute* node)
{
    mOs << node->nodename() << ": " << node->symbolseparator()
        << '(' << node->typestr() << (node->inferred() ? " - inferred": "")
        << ") " << node->name() << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::DeclareLocal* node)
{
    mOs << node->nodename() << ": "<<  node->typestr() << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::Local* node)
{
    mOs << node->nodename() << ' ' << node->name() << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::ArrayUnpack* node)
{
    mOs << node->nodename() << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::ArrayPack* node)
{
    mOs << node->nodename() << ": " << node->children() << " element(s)" << '\n';
    return true;
}

bool PrintVisitor::visit(const ast::ExternalVariable* node)
{
    mOs << node->nodename() << ": " << node->symbolseparator()
        << '(' << node->typestr() << ") " << node->name() << '\n';
    return true;
}

template <typename T>
bool PrintVisitor::visitValue(const ast::Value<T>* node)
{
    mOs << node->nodename() << ": " << node->value() << '\n';
    return true;
}


////////////////////////////////////////////////////////////////////////////////


void print(const ast::Node& node,
    const bool numberStatements,
    std::ostream& os,
    const char* indent)
{
    PrintVisitor visitor(os, numberStatements, indent);
    visitor.traverse(&node);
}


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

struct ReprintVisitor : public ast::Visitor<ReprintVisitor>
{
    ReprintVisitor(std::ostream& os, const char* indent = "    ")
        : mOs(os)
        , mLevel(0)
        , mIndent(indent) {}
    ~ReprintVisitor() = default;

    using ast::Visitor<ReprintVisitor>::traverse;
    using ast::Visitor<ReprintVisitor>::visit;

    inline bool postOrderNodes() const { return false; }

    inline void indent() {
        for (int i = 0; i < mLevel; ++i) mOs << mIndent;
    }

    bool traverse(NodeType<ast::Block>* block) {
        const size_t children = block->children();
        indent();
        mOs << '{' << '\n';
        ++mLevel;
        for (size_t i = 0; i < children; ++i) {
            indent();
            this->derived().traverse(block->child(i));
            const auto type = block->child(i)->nodetype();
            if (type != ast::Node::ConditionalStatementNode &&
                type != ast::Node::LoopNode) {
                mOs << ';' << '\n';
            }
        }
        --mLevel;
        indent();
        mOs << '}' << '\n';
        return true;
    }

    bool traverse(NodeType<ast::StatementList>* stmtl) {
        const size_t children = stmtl->children();
        if (children == 0) return true;
        if (children == 1) {
            this->derived().traverse(stmtl->child(0));
            mOs << ';';
            return true;
        }

        // multiple statments

        if (stmtl->child(0)->nodetype() == ast::Node::DeclareLocalNode) {
            // it's a declaration list, manually handle the child nodes.
            // This is to handle declarations within loop inits such as
            // "for (int a = 0, b = 1;;)". Without this, it would be
            // reprinted as "for (int a=0; int b=1; ;;)"

            // visit the first child completely
            this->derived().traverse(stmtl->child(0));

            for (size_t i = 1; i < children; ++i) {
                // all child statements should be declare locals
                OPENVDB_ASSERT(stmtl->child(i)->nodetype() ==
                    ast::Node::DeclareLocalNode);

                mOs << ", ";
                this->derived().traverse(stmtl->child(i)->child(0)); // local
                auto* init = stmtl->child(i)->child(1); // init
                if (init) {
                    mOs << " = ";
                    this->derived().traverse(init);
                }
            }
            return true;
        }

        // otherwise traverse as normal

        for (size_t i = 0; i < children; ++i) {
            this->derived().traverse(stmtl->child(i));
            if (i != children-1) mOs << ';' << ' ';
        }
        return true;
    }

    bool traverse(NodeType<ast::CommaOperator>* exprl) {
        mOs << '(';
        const size_t children = exprl->children();
        for (size_t i = 0; i < children; ++i) {
            this->derived().traverse(exprl->child(i));
            if (i != children-1) mOs << ',' << ' ';
        }
        mOs << ')';
        return true;
    }

    bool traverse(NodeType<ast::ConditionalStatement>* cond) {
        mOs << "if (";
        this->traverse(cond->condition());
        mOs << ")\n";
        this->traverse(cond->trueBranch());

        if (cond->hasFalse()) {
            indent();
            mOs << "else\n";
            this->traverse(cond->falseBranch());
        }
        return true;
    }

    bool traverse(NodeType<ast::Loop>* loop) {
        const ast::tokens::LoopToken loopType = loop->loopType();
        if (loopType == ast::tokens::LoopToken::FOR) {
            mOs << "for (";
            if (loop->hasInit()) this->traverse(loop->initial());
            mOs << "; ";
            this->traverse(loop->condition());
            mOs << "; ";
            if (loop->hasIter()) this->traverse(loop->iteration());
            mOs << ")\n";
            this->traverse(loop->body());
        }
        else if (loopType == ast::tokens::LoopToken::DO) {
            mOs << "do\n";
            this->traverse(loop->body());
            indent();
            mOs << "while (";
            this->traverse(loop->condition());
            mOs << ")\n";
        }
        else  {
            mOs << "while (";
            this->traverse(loop->condition());
            mOs << ")\n";
            this->traverse(loop->body());
        }
        return true;
    }

    bool traverse(NodeType<ast::AssignExpression>* asgn) {
        this->traverse(asgn->lhs());
        this->visit(asgn);
        this->traverse(asgn->rhs());
        return true;
    }

    bool traverse(NodeType<ast::DeclareLocal>* decl) {
        this->visit(decl);
        this->visit(decl->local());
        if (decl->hasInit()) {
            mOs <<" = ";
            this->traverse(decl->init());
        }
        return true;
    }

    bool traverse(NodeType<ast::Crement>* crmt) {
        if (crmt->pre()) this->visit(crmt);
        this->traverse(crmt->expression());
        if (crmt->post()) this->visit(crmt);
        return true;
    }

    bool traverse(NodeType<ast::UnaryOperator>* unry) {
        this->visit(unry);
        this->traverse(unry->expression());
        return true;
    }

    bool traverse(NodeType<ast::BinaryOperator>* bin) {
        // The AST currently doesn't store what expressions were encapsulated
        // by parenthesis and instead infers precedences from the token order
        // set out in the parser. This means that traversal determines precedence.
        // Unfortunately, statements like "a+b+c" and "a+(b+c)" both get re-printed
        // as "a+b+c". For now, every binary expression is encapsulated to
        // ensure the operation order is preserved, resulting in (a+(b+c)) for
        // the above example.
        mOs << '(';
        this->traverse(bin->lhs());
        this->visit(bin);
        this->traverse(bin->rhs());
        mOs << ')';
        return true;
    }

    bool traverse(NodeType<ast::TernaryOperator>* tern) {
        this->traverse(tern->condition());
        mOs << " ?";
        if (tern->hasTrue()) {
            mOs << ' ';
            this->traverse(tern->trueBranch());
            mOs << ' ';
        }
        mOs << ": ";
        this->traverse(tern->falseBranch());
        return true;
    }

    bool traverse(NodeType<ast::Cast>* cast) {
        this->visit(cast);
        mOs << '(';
        this->traverse(cast->expression());
        mOs << ')';
        return true;
    }

    bool traverse(NodeType<ast::FunctionCall>* call) {
        this->visit(call);
        mOs << '(';
        const size_t children = call->children();
        for (size_t i = 0; i < children; ++i) {
            this->derived().traverse(call->child(i));
            if (i != children-1) mOs << ',' << ' ';
        }
        mOs << ')';
        return true;
    }

    bool traverse(NodeType<ast::ArrayPack>* pack) {
        mOs << '{';
        const size_t children = pack->children();
        for (size_t i = 0; i < children; ++i) {
            this->derived().traverse(pack->child(i));
            if (i != children-1) mOs << ',' << ' ';
        }
        mOs << '}';
        return true;
    }

    bool traverse(NodeType<ast::ArrayUnpack>* pack) {
        this->traverse(pack->expression());
        mOs << '[';
        this->traverse(pack->component0());
        if (pack->component1()) {
            mOs << ',' << ' ';
            this->traverse(pack->component1());
        }
        mOs << ']';
        return true;
    }

    bool visit(const ast::AssignExpression* node);
    bool visit(const ast::Crement* node);
    bool visit(const ast::UnaryOperator* node);
    bool visit(const ast::BinaryOperator* node);
    bool visit(const ast::Cast* node);
    bool visit(const ast::FunctionCall* node);
    bool visit(const ast::Attribute* node);
    bool visit(const ast::ExternalVariable* node);
    bool visit(const ast::DeclareLocal* node);
    bool visit(const ast::Local* node);
    bool visit(const ast::Keyword* node);

    bool visit(const ast::Value<bool>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int16_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int32_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<int64_t>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<float>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<double>* node) {
        return this->visitValue(node);
    }
    bool visit(const ast::Value<std::string>* node) {
        return this->visitValue(node);
    }

protected:
    template <typename T>
    bool visitValue(const ast::Value<T>* node);
private:
    std::ostream& mOs;
    int mLevel;
    const char* mIndent;
};


bool ReprintVisitor::visit(const ast::AssignExpression* node)
{
    mOs << ' ' << tokens::operatorNameFromToken(node->operation());
    if (node->isCompound()) mOs << '=';
    mOs << ' ';
    return true;
}

bool ReprintVisitor::visit(const ast::Crement* node)
{
    if (node->increment()) mOs << "++";
    if (node->decrement()) mOs << "--";
    return true;
}

bool ReprintVisitor::visit(const ast::UnaryOperator* node)
{
    mOs << tokens::operatorNameFromToken(node->operation());
    return true;
}

bool ReprintVisitor::visit(const ast::BinaryOperator* node)
{
    mOs << ' ' << tokens::operatorNameFromToken(node->operation()) << ' ';
    return true;
}

bool ReprintVisitor::visit(const ast::Cast* node)
{
    mOs << node->typestr();
    return true;
}

bool ReprintVisitor::visit(const ast::FunctionCall* node)
{
    mOs << node->name();
    return true;
}

bool ReprintVisitor::visit(const ast::Attribute* node)
{
    mOs << node->typestr() << node->symbolseparator() << node->name();
    return true;
}

bool ReprintVisitor::visit(const ast::DeclareLocal* node)
{
    mOs << node->typestr() << " ";
    return true;
}

bool ReprintVisitor::visit(const ast::Local* node)
{
    mOs << node->name();
    return true;
}

bool ReprintVisitor::visit(const ast::Keyword* node)
{
    mOs << tokens::keywordNameFromToken(node->keyword());
    return true;
}

bool ReprintVisitor::visit(const ast::ExternalVariable* node)
{
    mOs << node->typestr() << node->symbolseparator() << node->name();
    return true;
}

template <typename T>
bool ReprintVisitor::visitValue(const ast::Value<T>* node)
{
    if (std::is_same<T, bool>::value) mOs << std::boolalpha;
    if (std::is_same<T, std::string>::value) mOs << '"';
    mOs << node->value();
    if (std::is_same<T, bool>::value) mOs << std::noboolalpha;
    if (std::is_same<T, std::string>::value) mOs << '"';
    if (std::is_same<T, int16_t>::value) mOs << 's';
    if (std::is_same<T, int64_t>::value) mOs << 'l';
    if (std::is_same<T, float>::value)   mOs << 'f';
    return true;
}

////////////////////////////////////////////////////////////////////////////////


void reprint(const ast::Node& node, std::ostream& os, const char* indent)
{
    ReprintVisitor visitor(os, indent);
    visitor.traverse(&node);
}


} // namespace ast
} // namespace ax

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


