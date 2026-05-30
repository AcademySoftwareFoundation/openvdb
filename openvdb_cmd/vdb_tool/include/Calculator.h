// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Calculator.h
///
/// @brief A compact bytecode interpreter for user-supplied math expressions
///        over an arbitrary number of named float inputs, intended for the
///        per-voxel hot path of the forAllValues / forOnValues / forOffValues
///        actions. Compiled once, then evaluated many times &mdash; like a
///        pocket calculator, hence the name.
///
/// @details Accepts three syntaxes that compile to a single shared bytecode:
///          - Reverse Polish Notation (RPN), colon-separated, e.g.
///              "$x:sin:$x:pow2:2:*:+"
///            (the same language as the vdb_tool string Processor).
///          - A single infix expression, e.g.
///              "sin(x) + 2*x*x"
///          - A semicolon-separated infix program with assignments, e.g.
///              "t = x*x; t + sin(t)"
///            where intermediate statements declare local slots reused by
///            subsequent statements.
///          The compiler dispatches automatically: '=' or ';' &rarr; multi-
///          statement infix; ':' or '$' &rarr; RPN; otherwise single infix.
///          The eval() loop is shared across all three. No external dependencies.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED
#define VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED

#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Util.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_E
#define M_E  2.71828182845904523536
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace vdb_tool {

/// @brief Compiled, reusable bytecode representation of a math expression with
///        an arbitrary number of named float variables. Re-entrant and thread-
///        safe at eval() time.
///
/// @details Variable names are discovered automatically by compile(): any
///          identifier in the expression that is not a built-in constant
///          (`pi`, `e`) and not the name of a function (`sin`, `abs`, ...) is
///          recorded as a variable. The full list, in order of first
///          appearance, is accessible via variables(). The general eval()
///          overload takes a parallel array of float values bound positionally
///          to variables(); two convenience overloads handle the common
///          single-variable and `(x,y,z)` cases and throw if the compiled
///          expression references a name they cannot bind.
///
/// @details (Multi-statement, infix only.) The expression may also contain
///          semicolon-separated statements with assignment, where intermediate
///          statements declare local slots reused by subsequent ones, e.g.
///              "t = x*x; t + sin(t)"
///          binds `t` once, then reads it twice. Slot names are *local* (not
///          exposed by variables()) and shadow any input variable of the same
///          name from the point of their first assignment onward. The final
///          statement may be either a plain expression or an assignment; its
///          right-hand-side value is the value returned by eval(). Each
///          intermediate statement must be an assignment, since a plain
///          expression would strand a value on the stack.
class Calculator
{
public:

    /// @brief Compile an expression into bytecode.
    /// @param expr One of:
    ///        - an RPN expression with colon-separated tokens (e.g.
    ///          `"$x:sin:$x:pow2:2:*:+"`); single statement, no `=`/`;`,
    ///        - a single infix expression (e.g. `"sin(theta) + radius*radius"`),
    ///        - or an infix multi-statement program (e.g.
    ///          `"t = x*x; t + sin(t)"`). Any identifier that is not the
    ///          constant `pi`/`e`, not a known function, and not a previously
    ///          declared slot becomes an input variable, queryable via
    ///          variables().
    /// @throw std::invalid_argument on syntax errors, unknown function names,
    ///        mismatched parentheses, stack underflow, an intermediate plain
    ///        expression (which would strand a value), or a final stack
    ///        depth other than 1.
    void compile(const std::string &expr);

    /// @brief Variable names referenced by the compiled expression, in the
    ///        order in which they first appear. The general eval() overload
    ///        binds values positionally to this list.
    const std::vector<std::string>& variables() const { return mVariables; }

    /// @brief Position of @a name in variables(), or -1 if the compiled
    ///        expression doesn't reference it. Intended for callers that
    ///        want to build the positional `values` buffer for the general
    ///        eval() overload without doing a linear scan each call &mdash;
    ///        look the indices up once after compile() and reuse them.
    inline int variableIndex(const std::string &name) const;

    /// @brief Evaluate the compiled expression by binding @a values positionally
    ///        to variables(). The pointer must address at least
    ///        variables().size() floats; the buffer is not modified.
    /// @note  Re-entrant; the working stack is allocated on the C stack inside
    ///        eval(), so a single Calculator instance can be shared across threads.
    inline float eval(const float* values) const;

    /// @brief Convenience overload for hand-written value lists. Throws when the
    ///        number of supplied values doesn't match variables().size().
    inline float eval(std::initializer_list<float> values) const;

    /// @brief Convenience overload that binds variables by name. Slower per
    ///        call than the positional form (one map lookup per variable);
    ///        intended for non-hot-path use. Throws if any of the compiled
    ///        variables is missing from @a bindings.
    inline float eval(const std::unordered_map<std::string, float> &bindings) const;

    /// @brief Convenience overload for single-variable expressions. Throws if
    ///        the compiled expression references any variable other than `x`.
    inline float eval(float x) const;

    // ---------------------------------------------------------------------
    //  Persistent memory: evalAndRemember() + accessors
    // ---------------------------------------------------------------------
    //
    // Identical bytecode and numeric result as eval(), but additionally
    // snapshots into mMemory every input value, every intermediate slot
    // value, and the final expression result keyed by the trailing LHS
    // (if any). Intended for REPL/test/debug use where the caller wants to
    // inspect named variables after the call.
    //
    // Mutates mMemory, so evalAndRemember() is NOT thread-safe and is not
    // marked const. eval() remains const and lock-free; use it on the hot
    // path (e.g. parallel forValues) where memory inspection isn't needed.

    /// @brief Evaluate and write every input + slot + named result into
    ///        mMemory; retrieve via get()/has()/memory(). Not thread-safe.
    inline float evalAndRemember(const float* values);

    /// @brief Map-based binding form of evalAndRemember().
    inline float evalAndRemember(const std::unordered_map<std::string, float> &bindings);

    /// @brief Value stored in mMemory under @a name (set by the most recent
    ///        evalAndRemember()). Throws if no such entry exists.
    inline float get(const std::string &name) const;

    /// @brief True iff mMemory contains @a name.
    bool has(const std::string &name) const { return mMemory.find(name) != mMemory.end(); }

    /// @brief Whole-memory accessor.
    const std::unordered_map<std::string, float>& memory() const { return mMemory; }

    /// @brief Name from a trailing `name = ...` assignment in the compiled
    ///        program, or "" if the final statement is a plain expression.
    ///        Useful in combination with memory()/get() to learn which entry
    ///        corresponds to the value returned by eval().
    const std::string& resultName() const { return mResultName; }

    /// @brief Returns true if compile() has not yet been called (or failed).
    bool empty() const { return mCode.empty(); }
    /// @brief Number of bytecode instructions; useful for tests and debugging.
    size_t size() const { return mCode.size(); }

private:

    /// @brief Opcode set. PushLit/PushVar/PushSlot push to the stack; Store
    ///        pops the top and writes it to a slot; the rest consume one or
    ///        two stack entries and replace the top with the result.
    enum class Op : uint8_t {
        PushLit, PushVar, PushSlot,
        Store,
        Add, Sub, Mul, Div, Pow,
        Neg, Abs, Inv, Sqrt,
        Sin, Cos, Tan, Asin, Acos, Atan,
        Exp, Ln, Log, Floor, Ceil,
        Pow2, Pow3, Min, Max
    };

    /// @brief A single bytecode instruction. 4 bytes; cache-friendly.
    struct Instr {
        Op       op;
        uint16_t arg;  ///< Index into mConstants (PushLit), mVariables
                       ///< (PushVar), or mSlotNames (PushSlot/Store);
                       ///< unused otherwise.
    };

    /// @brief Hard limit on the number of distinct local slot names declared
    ///        by intermediate assignments. Used to size the eval-time slot
    ///        buffer on the C stack.
    static constexpr int kSlotsMax = 16;

    std::vector<Instr>       mCode;
    std::vector<float>       mConstants;
    std::vector<std::string> mVariables;
    std::vector<std::string> mSlotNames;
    /// @brief Trailing `name = ...` from the compiled program, if any.
    /// @details Pure documentation in eval(), but evalAndRemember() uses
    ///          this name to record the program's return value in mMemory.
    std::string                              mResultName;
    /// @brief Populated by evalAndRemember(); never read by eval().
    std::unordered_map<std::string, float>   mMemory;

    void     compileRPN(const std::string &expr);
    void     compileInfix(const std::string &expr);
    void     compileStatements(const std::string &expr);
    void     verify();
    void     emitConst(float v);
    void     emitName(const std::string &name);
    uint16_t allocateSlot(const std::string &name);

    /// @brief Bytecode interpreter shared by all eval()/evalAndRemember()
    ///        forms. If @a slotsOut is non-null, the final values of all
    ///        named slots are copied into it (used to snapshot mMemory).
    inline float evalImpl(const float* values, float* slotsOut) const;

    static bool isIdentifier(const std::string &s);

    static const std::unordered_map<std::string, Op>& unaryOps();
    static const std::unordered_map<std::string, Op>& binaryOps();
};

// ====================================================================
//  Inline implementation
// ====================================================================

inline const std::unordered_map<std::string, Calculator::Op>& Calculator::unaryOps()
{
    // Operations that consume one stack entry and replace it with the result.
    static const std::unordered_map<std::string, Op> t = {
        {"neg",   Op::Neg },  {"abs",   Op::Abs },  {"inv",   Op::Inv },
        {"sqrt",  Op::Sqrt},
        {"sin",   Op::Sin },  {"cos",   Op::Cos },  {"tan",   Op::Tan },
        {"asin",  Op::Asin}, {"acos",  Op::Acos}, {"atan",  Op::Atan},
        {"exp",   Op::Exp },  {"ln",    Op::Ln  }, {"log",   Op::Log },
        {"floor", Op::Floor},{"ceil",  Op::Ceil},
        {"pow2",  Op::Pow2}, {"pow3",  Op::Pow3}
    };
    return t;
}

inline const std::unordered_map<std::string, Calculator::Op>& Calculator::binaryOps()
{
    // Operations that consume two stack entries and push one result.
    // The four standard operators and '^' are listed here so the infix
    // emitter and the RPN tokenizer share a single lookup table.
    static const std::unordered_map<std::string, Op> t = {
        {"+",   Op::Add}, {"-",   Op::Sub}, {"*",   Op::Mul}, {"/",   Op::Div},
        {"^",   Op::Pow}, {"pow", Op::Pow}, {"min", Op::Min}, {"max", Op::Max}
    };
    return t;
}

inline void Calculator::emitConst(float v)
{
    mCode.push_back({Op::PushLit, static_cast<uint16_t>(mConstants.size())});
    mConstants.push_back(v);
}

inline void Calculator::emitName(const std::string &name)
{
    // A previously declared slot shadows any input of the same name from this
    // point onward — match scripting-language scoping for `x = ...; x + 1`.
    for (size_t i = 0; i < mSlotNames.size(); ++i) {
        if (mSlotNames[i] == name) {
            mCode.push_back({Op::PushSlot, static_cast<uint16_t>(i)});
            return;
        }
    }
    // Otherwise the name is an input variable (linear scan; the number of
    // distinct names in any realistic expression is tiny).
    for (size_t i = 0; i < mVariables.size(); ++i) {
        if (mVariables[i] == name) {
            mCode.push_back({Op::PushVar, static_cast<uint16_t>(i)});
            return;
        }
    }
    mVariables.push_back(name);
    mCode.push_back({Op::PushVar, static_cast<uint16_t>(mVariables.size() - 1)});
}

inline uint16_t Calculator::allocateSlot(const std::string &name)
{
    // Reuse an existing slot of the same name so `t = 1; t = t + 1; t` writes
    // into the same storage and `mSlotNames` doesn't grow on reassignment.
    for (size_t i = 0; i < mSlotNames.size(); ++i) {
        if (mSlotNames[i] == name) return static_cast<uint16_t>(i);
    }
    if (mSlotNames.size() >= static_cast<size_t>(kSlotsMax)) {
        throw std::invalid_argument(
            "Calculator: too many distinct slot names (max " +
            std::to_string(kSlotsMax) + ")");
    }
    mSlotNames.push_back(name);
    return static_cast<uint16_t>(mSlotNames.size() - 1);
}

inline bool Calculator::isIdentifier(const std::string &s)
{
    // C-style identifier: [A-Za-z_][A-Za-z0-9_]*
    if (s.empty()) return false;
    const unsigned char c0 = static_cast<unsigned char>(s[0]);
    if (!std::isalpha(c0) && c0 != '_') return false;
    for (size_t i = 1; i < s.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (!std::isalnum(c) && c != '_') return false;
    }
    return true;
}

inline void Calculator::compile(const std::string &expr)
{
    mCode.clear();
    mConstants.clear();
    mVariables.clear();
    mSlotNames.clear();
    mResultName.clear();
    mMemory.clear();
    if (expr.empty()) {
        throw std::invalid_argument("Calculator: empty expression");
    }
    // Dispatch:
    //   '=' or ';' present  -> multi-statement infix (assignments / slots)
    //   ':' or '$' present  -> classic RPN, single statement
    //   otherwise           -> classic infix, single statement
    const bool hasAssign    = expr.find('=') != std::string::npos;
    const bool hasSemicolon = expr.find(';') != std::string::npos;
    const bool hasRPNMarker = expr.find(':') != std::string::npos ||
                              expr.find('$') != std::string::npos;
    if (hasAssign || hasSemicolon) {
        if (hasRPNMarker) {
            throw std::invalid_argument(
                "Calculator: assignment ('=') and multi-statement (';') "
                "require infix syntax; remove ':'/'$' or drop the assignment");
        }
        this->compileStatements(expr);
    } else if (hasRPNMarker) {
        this->compileRPN(expr);
    } else {
        this->compileInfix(expr);
    }
    this->verify();
}

inline void Calculator::compileStatements(const std::string &expr)
{
    // Split on ';'. '(' / ')' never contain a ';' in this grammar, so a flat
    // scan is sufficient — no nesting tracking needed.
    auto trim = [](std::string s) -> std::string {
        size_t a = 0, b = s.size();
        while (a < b && std::isspace(static_cast<unsigned char>(s[a])))     ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        return s.substr(a, b - a);
    };

    std::vector<std::string> stmts;
    {
        size_t start = 0;
        for (size_t i = 0; i <= expr.size(); ++i) {
            if (i == expr.size() || expr[i] == ';') {
                std::string s = trim(expr.substr(start, i - start));
                if (!s.empty()) stmts.push_back(std::move(s));
                start = i + 1;
            }
        }
    }
    if (stmts.empty()) {
        throw std::invalid_argument("Calculator: no non-empty statements");
    }

    for (size_t idx = 0; idx < stmts.size(); ++idx) {
        const std::string &s = stmts[idx];
        const bool isLast = (idx + 1 == stmts.size());

        // Detect 'lhs = rhs'. The grammar has no '==' or any other use of '='.
        std::string lhs, rhs;
        const size_t eq = s.find('=');
        if (eq != std::string::npos) {
            lhs = trim(s.substr(0, eq));
            rhs = trim(s.substr(eq + 1));
            if (rhs.find('=') != std::string::npos) {
                throw std::invalid_argument(
                    "Calculator: chained '=' is not supported (statement: \"" + s + "\")");
            }
            if (!isIdentifier(lhs)) {
                throw std::invalid_argument(
                    "Calculator: invalid assignment target \"" + lhs + "\"");
            }
            if (lhs == "pi" || lhs == "e") {
                throw std::invalid_argument(
                    "Calculator: cannot assign to constant \"" + lhs + "\"");
            }
        } else {
            rhs = s;
        }
        if (rhs.empty()) {
            throw std::invalid_argument("Calculator: empty right-hand side");
        }

        // Compile the RHS as a single infix expression. compileInfix appends
        // to mCode, so each statement's instructions land in order.
        this->compileInfix(rhs);

        if (!lhs.empty() && !isLast) {
            // Intermediate assignment: pop the value into the named slot.
            const uint16_t slot = this->allocateSlot(lhs);
            mCode.push_back({Op::Store, slot});
        } else if (lhs.empty() && !isLast) {
            // A plain expression in non-final position would leave a value
            // on the stack with no consumer, so verify() would reject it
            // anyway; reporting the structural issue here is friendlier.
            throw std::invalid_argument(
                "Calculator: intermediate statement must be an assignment "
                "(\"name = expression\"); plain expressions are only allowed "
                "as the final statement");
        } else if (!lhs.empty() && isLast) {
            // Final assignment: the RHS value stays on the stack (eval's
            // return value). Record the LHS name so evalAndRemember() can
            // expose the result under it in mMemory. No Store opcode is
            // emitted — see the docstring on mResultName.
            mResultName = lhs;
        }
        // Final plain expression: RHS value stays on the stack; nothing else.
    }
}

inline void Calculator::compileRPN(const std::string &expr)
{
    // Allow optional surrounding "{...}" for symmetry with the Processor's
    // command-line syntax; the user usually omits them since options are
    // already string-valued.
    std::string s = expr;
    if (!s.empty() && s.front() == '{') s.erase(0, 1);
    if (!s.empty() && s.back()  == '}') s.pop_back();

    const auto tokens = tokenize(s, ":");
    const auto &unary  = unaryOps();
    const auto &binary = binaryOps();

    for (const std::string &tok : tokens) {
        if (tok.empty()) continue;

        // Numeric literal (a leading '$' rules this out automatically).
        float v;
        if (isFloat(tok, v)) { this->emitConst(v); continue; }

        // Operations are matched against the bare token (no '$' prefix
        // allowed) so they cannot be shadowed by similarly-named variables.
        auto u = unary.find(tok);
        if (u != unary.end()) { mCode.push_back({u->second, 0}); continue; }
        auto b = binary.find(tok);
        if (b != binary.end()) { mCode.push_back({b->second, 0}); continue; }

        // Identifier: variable or named constant. Optional leading '$' is
        // stripped to match the Processor's convention ($x, $pi, ...).
        std::string name = tok;
        if (!name.empty() && name[0] == '$') name.erase(0, 1);
        if (!isIdentifier(name)) {
            throw std::invalid_argument("Calculator: unknown token \"" + tok + "\"");
        }
        if (name == "pi") { this->emitConst(static_cast<float>(M_PI)); continue; }
        if (name == "e" ) { this->emitConst(static_cast<float>(M_E )); continue; }
        this->emitName(name);
    }
}

namespace calculator_detail {
    enum class Kind { End, Number, Name, BinOp, UnaryOp, LParen, RParen, Comma };
    struct Token {
        Kind        kind = Kind::End;
        std::string text;
        float       value      = 0.0f;
        int         precedence = 0;
        bool        rightAssoc = false;
    };
}

inline void Calculator::compileInfix(const std::string &expr)
{
    using namespace calculator_detail;

    // Tokenize.
    std::vector<Token> tokens;
    const size_t N = expr.size();
    Kind prev = Kind::End;
    size_t i = 0;
    while (i < N) {
        const char c = expr[i];
        if (std::isspace(static_cast<unsigned char>(c))) { ++i; continue; }
        Token t;
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && i+1 < N && std::isdigit(static_cast<unsigned char>(expr[i+1])))) {
            // Number literal, with optional decimal point and exponent.
            size_t j = i;
            while (j < N && (std::isdigit(static_cast<unsigned char>(expr[j])) || expr[j] == '.')) ++j;
            if (j < N && (expr[j] == 'e' || expr[j] == 'E')) {
                ++j;
                if (j < N && (expr[j] == '+' || expr[j] == '-')) ++j;
                while (j < N && std::isdigit(static_cast<unsigned char>(expr[j]))) ++j;
            }
            t.kind  = Kind::Number;
            t.text  = expr.substr(i, j - i);
            t.value = std::stof(t.text);
            i = j;
        } else if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            // Identifier: variable, named constant, or function call.
            size_t j = i;
            while (j < N && (std::isalnum(static_cast<unsigned char>(expr[j])) || expr[j] == '_')) ++j;
            t.kind = Kind::Name;
            t.text = expr.substr(i, j - i);
            i = j;
        } else if (c == '(') { t.kind = Kind::LParen; ++i; }
          else if (c == ')') { t.kind = Kind::RParen; ++i; }
          else if (c == ',') { t.kind = Kind::Comma;  ++i; }
          else if (c == '+' || c == '-' || c == '*' || c == '/' || c == '^') {
            // Unary +/- is signalled by what came before: nothing, a binary op,
            // another unary op, an opening paren, or a comma.
            const bool isUnary = (c == '+' || c == '-') &&
                                 (prev == Kind::End    || prev == Kind::BinOp ||
                                  prev == Kind::UnaryOp|| prev == Kind::LParen||
                                  prev == Kind::Comma);
            t.text = std::string(1, c);
            if (isUnary) {
                t.kind       = Kind::UnaryOp;
                t.precedence = 5;
                t.rightAssoc = true;
            } else {
                t.kind = Kind::BinOp;
                switch (c) {
                    case '+': t.precedence = 2; break;
                    case '-': t.precedence = 2; break;
                    case '*': t.precedence = 3; break;
                    case '/': t.precedence = 3; break;
                    case '^': t.precedence = 4; t.rightAssoc = true; break;
                }
            }
            ++i;
        } else {
            throw std::invalid_argument(
                "Calculator: unexpected character '" + std::string(1, c) + "' in expression");
        }
        tokens.push_back(t);
        prev = t.kind;
    }

    // Shunting-yard, emitting straight into mCode/mConstants.
    std::vector<Token> opStack;
    const auto &unary  = unaryOps();
    const auto &binary = binaryOps();

    auto emit = [&](const Token &op) {
        if (op.kind == Kind::UnaryOp) {
            if (op.text == "-") mCode.push_back({Op::Neg, 0});
            // unary '+' is a no-op
            return;
        }
        if (op.kind == Kind::BinOp) {
            auto it = binary.find(op.text);
            if (it == binary.end())
                throw std::invalid_argument("Calculator: unknown operator \"" + op.text + "\"");
            mCode.push_back({it->second, 0});
            return;
        }
        if (op.kind == Kind::Name) {
            auto u = unary.find(op.text);
            if (u != unary.end()) { mCode.push_back({u->second, 0}); return; }
            auto b = binary.find(op.text);
            if (b != binary.end()) { mCode.push_back({b->second, 0}); return; }
            throw std::invalid_argument("Calculator: unknown function \"" + op.text + "\"");
        }
        throw std::invalid_argument("Calculator: internal: cannot emit this token");
    };

    for (size_t k = 0; k < tokens.size(); ++k) {
        const Token &t = tokens[k];
        switch (t.kind) {
        case Kind::Number:
            this->emitConst(t.value);
            break;
        case Kind::Name: {
            // If the next token is '(' this is a function call; otherwise a
            // variable or named constant. Any identifier that is neither a
            // built-in constant nor a function name becomes a variable, so
            // the dispatch below recognizes only the constants explicitly.
            const bool isFunction = (k + 1 < tokens.size() && tokens[k+1].kind == Kind::LParen);
            if (isFunction) {
                opStack.push_back(t);
            } else if (t.text == "pi") {
                this->emitConst(static_cast<float>(M_PI));
            } else if (t.text == "e") {
                this->emitConst(static_cast<float>(M_E));
            } else {
                this->emitName(t.text);
            }
            break;
        }
        case Kind::UnaryOp:
            opStack.push_back(t);
            break;
        case Kind::BinOp:
            while (!opStack.empty()) {
                const Token &top = opStack.back();
                if (top.kind == Kind::LParen) break;
                bool popTop;
                if (top.kind == Kind::Name || top.kind == Kind::UnaryOp) {
                    popTop = true;// functions and unary ops bind tighter
                } else {
                    popTop = (top.precedence > t.precedence) ||
                             (top.precedence == t.precedence && !t.rightAssoc);
                }
                if (!popTop) break;
                emit(top);
                opStack.pop_back();
            }
            opStack.push_back(t);
            break;
        case Kind::LParen:
            opStack.push_back(t);
            break;
        case Kind::Comma:
            while (!opStack.empty() && opStack.back().kind != Kind::LParen) {
                emit(opStack.back());
                opStack.pop_back();
            }
            if (opStack.empty())
                throw std::invalid_argument("Calculator: ',' outside function call");
            break;
        case Kind::RParen:
            while (!opStack.empty() && opStack.back().kind != Kind::LParen) {
                emit(opStack.back());
                opStack.pop_back();
            }
            if (opStack.empty())
                throw std::invalid_argument("Calculator: mismatched closing parenthesis");
            opStack.pop_back();// discard the LParen
            if (!opStack.empty() && opStack.back().kind == Kind::Name) {
                emit(opStack.back());
                opStack.pop_back();
            }
            break;
        default:
            throw std::invalid_argument("Calculator: internal: unexpected infix token");
        }
    }
    while (!opStack.empty()) {
        const Token &top = opStack.back();
        if (top.kind == Kind::LParen)
            throw std::invalid_argument("Calculator: mismatched opening parenthesis");
        emit(top);
        opStack.pop_back();
    }
}

inline void Calculator::verify()
{
    // Walk the bytecode with a virtual stack-depth counter so eval() doesn't
    // have to check anything at runtime. Any well-formed expression must leave
    // exactly one value on the stack; anything else is a compile-time error.
    int depth = 0;
    constexpr int kStackMax = 32;
    for (const Instr &i : mCode) {
        switch (i.op) {
        case Op::PushLit: case Op::PushVar: case Op::PushSlot:
            ++depth;
            break;
        case Op::Store:
            if (depth < 1) throw std::invalid_argument("Calculator: stack underflow on store");
            --depth;
            break;
        case Op::Neg: case Op::Abs: case Op::Inv: case Op::Sqrt:
        case Op::Sin: case Op::Cos: case Op::Tan:
        case Op::Asin: case Op::Acos: case Op::Atan:
        case Op::Exp: case Op::Ln:  case Op::Log:
        case Op::Floor: case Op::Ceil:
        case Op::Pow2:  case Op::Pow3:
            if (depth < 1) throw std::invalid_argument("Calculator: stack underflow on unary op");
            break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Pow:
        case Op::Min: case Op::Max:
            if (depth < 2) throw std::invalid_argument("Calculator: stack underflow on binary op");
            --depth;
            break;
        }
        if (depth > kStackMax)
            throw std::invalid_argument("Calculator: stack overflow (depth > 32)");
    }
    if (depth != 1)
        throw std::invalid_argument(
            "Calculator: expression must produce exactly one value (final depth=" +
            std::to_string(depth) + ")");
}

inline float Calculator::eval(const float* values) const
{
    return this->evalImpl(values, nullptr);
}

inline float Calculator::evalImpl(const float* values, float* slotsOut) const
{
    // 32 stack slots is far more than any reasonable expression needs; verify()
    // enforces the bound at compile time so this loop performs zero checks.
    // Callers must ensure @a values addresses at least mVariables.size()
    // floats; if mVariables is empty, no PushVar opcode references it and
    // a null pointer is safe. The slot buffer is sized to kSlotsMax (also
    // enforced at compile time by allocateSlot()), so PushSlot/Store never
    // run off the end.
    constexpr int kStackMax = 32;
    float s[kStackMax];
    float slots[kSlotsMax];
    int sp = 0;
    for (const Instr &i : mCode) {
        switch (i.op) {
        case Op::PushLit:  s[sp++] = mConstants[i.arg]; break;
        case Op::PushVar:  s[sp++] = values[i.arg];     break;
        case Op::PushSlot: s[sp++] = slots[i.arg];      break;
        case Op::Store:    --sp; slots[i.arg] = s[sp];  break;
        case Op::Add:  --sp; s[sp-1] += s[sp];                    break;
        case Op::Sub:  --sp; s[sp-1] -= s[sp];                    break;
        case Op::Mul:  --sp; s[sp-1] *= s[sp];                    break;
        case Op::Div:  --sp; s[sp-1] /= s[sp];                    break;
        case Op::Pow:  --sp; s[sp-1] = std::pow(s[sp-1], s[sp]);  break;
        case Op::Neg:        s[sp-1] = -s[sp-1];                  break;
        case Op::Abs:        s[sp-1] = std::fabs(s[sp-1]);        break;
        case Op::Inv:        s[sp-1] = 1.0f / s[sp-1];            break;
        case Op::Sqrt:       s[sp-1] = std::sqrt(s[sp-1]);        break;
        case Op::Sin:        s[sp-1] = std::sin(s[sp-1]);         break;
        case Op::Cos:        s[sp-1] = std::cos(s[sp-1]);         break;
        case Op::Tan:        s[sp-1] = std::tan(s[sp-1]);         break;
        case Op::Asin:       s[sp-1] = std::asin(s[sp-1]);        break;
        case Op::Acos:       s[sp-1] = std::acos(s[sp-1]);        break;
        case Op::Atan:       s[sp-1] = std::atan(s[sp-1]);        break;
        case Op::Exp:        s[sp-1] = std::exp(s[sp-1]);         break;
        case Op::Ln:         s[sp-1] = std::log(s[sp-1]);         break;
        case Op::Log:        s[sp-1] = std::log10(s[sp-1]);       break;
        case Op::Floor:      s[sp-1] = std::floor(s[sp-1]);       break;
        case Op::Ceil:       s[sp-1] = std::ceil(s[sp-1]);        break;
        case Op::Pow2:       s[sp-1] *= s[sp-1];                  break;
        case Op::Pow3:     { float v = s[sp-1]; s[sp-1] = v*v*v; }break;
        case Op::Min:  --sp; s[sp-1] = std::fmin(s[sp-1], s[sp]); break;
        case Op::Max:  --sp; s[sp-1] = std::fmax(s[sp-1], s[sp]); break;
        }
    }
    if (slotsOut != nullptr) {
        for (size_t i = 0; i < mSlotNames.size(); ++i) slotsOut[i] = slots[i];
    }
    return s[0];
}

inline float Calculator::eval(std::initializer_list<float> values) const
{
    if (values.size() != mVariables.size()) {
        throw std::invalid_argument(
            "Calculator::eval: expected " + std::to_string(mVariables.size()) +
            " value(s), got " + std::to_string(values.size()));
    }
    return this->eval(values.begin());
}

inline int Calculator::variableIndex(const std::string &name) const
{
    for (size_t i = 0; i < mVariables.size(); ++i) {
        if (mVariables[i] == name) return static_cast<int>(i);
    }
    return -1;
}

inline float Calculator::eval(const std::unordered_map<std::string, float> &bindings) const
{
    // Realistic expressions have a handful of variables, so a small stack
    // buffer suffices and avoids a heap allocation on every call.
    constexpr size_t kSmall = 16;
    float small[kSmall];
    std::vector<float> heap;
    float *values = small;
    if (mVariables.size() > kSmall) {
        heap.resize(mVariables.size());
        values = heap.data();
    }
    for (size_t i = 0; i < mVariables.size(); ++i) {
        const auto it = bindings.find(mVariables[i]);
        if (it == bindings.end()) {
            throw std::invalid_argument(
                "Calculator::eval: missing binding for variable \"" + mVariables[i] + "\"");
        }
        values[i] = it->second;
    }
    return this->eval(values);
}

inline float Calculator::eval(float x) const
{
    if (mVariables.empty()) return this->eval(&x);// @a x is ignored, but provides a valid address
    if (mVariables.size() == 1 && mVariables[0] == "x") return this->eval(&x);
    // Pick the first offending name for a useful error message.
    for (const std::string &v : mVariables) {
        if (v != "x") {
            throw std::invalid_argument(
                "Calculator::eval(x): expression references undefined variable \"" + v + "\"");
        }
    }
    throw std::invalid_argument(
        "Calculator::eval(x): expression references " + std::to_string(mVariables.size()) +
        " variables, but only \"x\" is bindable");
}

// ---------------------------------------------------------------------
//  Persistent-memory variant
// ---------------------------------------------------------------------

inline float Calculator::evalAndRemember(const float* values)
{
    float slots[kSlotsMax];
    const float result = this->evalImpl(values, slots);
    // Snapshot: inputs + slots + named result. mMemory is rebuilt each call
    // so stale entries from a previous run don't linger across compile()
    // boundaries.
    mMemory.clear();
    for (size_t i = 0; i < mVariables.size(); ++i) mMemory[mVariables[i]] = values[i];
    for (size_t i = 0; i < mSlotNames.size(); ++i) mMemory[mSlotNames[i]] = slots[i];
    if (!mResultName.empty()) mMemory[mResultName] = result;
    return result;
}

inline float Calculator::evalAndRemember(const std::unordered_map<std::string, float> &bindings)
{
    // Build a positional values buffer; small expressions get a stack array,
    // larger ones spill to the heap (same pattern as the const eval(map) overload).
    constexpr size_t kSmall = 16;
    float small[kSmall];
    std::vector<float> heap;
    float *values = small;
    if (mVariables.size() > kSmall) {
        heap.resize(mVariables.size());
        values = heap.data();
    }
    for (size_t i = 0; i < mVariables.size(); ++i) {
        const auto it = bindings.find(mVariables[i]);
        if (it == bindings.end()) {
            throw std::invalid_argument(
                "Calculator::evalAndRemember: missing binding for variable \"" +
                mVariables[i] + "\"");
        }
        values[i] = it->second;
    }
    return this->evalAndRemember(values);
}

inline float Calculator::get(const std::string &name) const
{
    const auto it = mMemory.find(name);
    if (it == mMemory.end()) {
        throw std::invalid_argument(
            "Calculator::get: no memory entry for \"" + name +
            "\" (call evalAndRemember() first, or the name is neither an "
            "input, a slot, nor the trailing result)");
    }
    return it->second;
}

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED
