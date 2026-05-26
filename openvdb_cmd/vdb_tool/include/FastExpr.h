// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file FastExpr.h
///
/// @brief A compact bytecode interpreter for user-supplied math expressions,
///        intended for the per-voxel hot path of the forAllValues / forOnValues
///        / forOffValues actions. Compiled once, then evaluated many times.
///
/// @details Accepts two syntaxes that compile to identical bytecode:
///          - Reverse Polish Notation (RPN), colon-separated, e.g.
///              "$x:sin:$x:pow2:2:*:+"
///            (the same language as the vdb_tool string Processor).
///          - Infix, e.g.
///              "sin(x) + 2*x*x"
///          The compiler dispatches on the presence of a ':' character. The
///          eval() loop is identical for both. No external dependencies.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_FASTEXPR_HAS_BEEN_INCLUDED
#define VDB_TOOL_FASTEXPR_HAS_BEEN_INCLUDED

#include <cctype>
#include <cmath>
#include <cstdint>
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

/// @brief Compiled, reusable bytecode representation of a math expression of a
///        single input variable `x`. Re-entrant and thread-safe at eval() time.
class FastExpr
{
public:

    /// @brief Compile an expression into bytecode.
    /// @param expr Either an RPN expression with colon-separated tokens
    ///             (e.g. "$x:sin:$x:pow2:2:*:+") or an infix expression
    ///             (e.g. "sin(x) + 2*x*x"). Dispatch is automatic.
    /// @throw std::invalid_argument on syntax errors, unknown operations,
    ///        mismatched parentheses, stack underflow, or a final stack
    ///        depth other than 1.
    void compile(const std::string &expr);

    /// @brief Evaluate the compiled expression with @a x as the input.
    /// @note  Re-entrant; the working stack is allocated on the C stack inside
    ///        eval(), so a single FastExpr instance can be shared across threads.
    inline float eval(float x) const;

    /// @brief Returns true if compile() has not yet been called (or failed).
    bool empty() const { return mCode.empty(); }
    /// @brief Number of bytecode instructions; useful for tests and debugging.
    size_t size() const { return mCode.size(); }

private:

    /// @brief Opcode set. PushLit/PushX push to the stack; the rest consume
    ///        one or two stack entries and replace the top with the result.
    enum class Op : uint8_t {
        PushLit, PushX,
        Add, Sub, Mul, Div, Pow,
        Neg, Abs, Inv, Sqrt,
        Sin, Cos, Tan, Asin, Acos, Atan,
        Exp, Ln, Log, Floor, Ceil,
        Pow2, Pow3, Min, Max
    };

    /// @brief A single bytecode instruction. 4 bytes; cache-friendly.
    struct Instr {
        Op       op;
        uint16_t arg;  ///< Index into mConstants for PushLit; unused otherwise.
    };

    std::vector<Instr> mCode;
    std::vector<float> mConstants;

    void compileRPN(const std::string &expr);
    void compileInfix(const std::string &expr);
    void verify();
    void emitConst(float v);

    static const std::unordered_map<std::string, Op>& unaryOps();
    static const std::unordered_map<std::string, Op>& binaryOps();
};

// ====================================================================
//  Inline implementation
// ====================================================================

inline const std::unordered_map<std::string, FastExpr::Op>& FastExpr::unaryOps()
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

inline const std::unordered_map<std::string, FastExpr::Op>& FastExpr::binaryOps()
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

inline void FastExpr::emitConst(float v)
{
    mCode.push_back({Op::PushLit, static_cast<uint16_t>(mConstants.size())});
    mConstants.push_back(v);
}

inline void FastExpr::compile(const std::string &expr)
{
    mCode.clear();
    mConstants.clear();
    if (expr.empty()) {
        throw std::invalid_argument("FastExpr: empty expression");
    }
    // RPN uses ':' as a token separator and '$' as a variable prefix; standard
    // infix math uses neither, so either character is a reliable RPN marker.
    if (expr.find(':') != std::string::npos ||
        expr.find('$') != std::string::npos) {
        this->compileRPN(expr);
    } else {
        this->compileInfix(expr);
    }
    this->verify();
}

inline void FastExpr::compileRPN(const std::string &expr)
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
        if (tok == "$x" || tok == "x") {
            mCode.push_back({Op::PushX, 0});
            continue;
        }
        if (tok == "$pi" || tok == "pi") { this->emitConst(static_cast<float>(M_PI)); continue; }
        if (tok == "$e"  || tok == "e" ) { this->emitConst(static_cast<float>(M_E )); continue; }
        float v;
        if (isFloat(tok, v)) { this->emitConst(v); continue; }
        auto u = unary.find(tok);
        if (u != unary.end()) { mCode.push_back({u->second, 0}); continue; }
        auto b = binary.find(tok);
        if (b != binary.end()) { mCode.push_back({b->second, 0}); continue; }
        throw std::invalid_argument("FastExpr: unknown token \"" + tok + "\"");
    }
}

namespace fast_expr_detail {
    enum class Kind { End, Number, Name, BinOp, UnaryOp, LParen, RParen, Comma };
    struct Token {
        Kind        kind = Kind::End;
        std::string text;
        float       value      = 0.0f;
        int         precedence = 0;
        bool        rightAssoc = false;
    };
}

inline void FastExpr::compileInfix(const std::string &expr)
{
    using namespace fast_expr_detail;

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
                "FastExpr: unexpected character '" + std::string(1, c) + "' in expression");
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
                throw std::invalid_argument("FastExpr: unknown operator \"" + op.text + "\"");
            mCode.push_back({it->second, 0});
            return;
        }
        if (op.kind == Kind::Name) {
            auto u = unary.find(op.text);
            if (u != unary.end()) { mCode.push_back({u->second, 0}); return; }
            auto b = binary.find(op.text);
            if (b != binary.end()) { mCode.push_back({b->second, 0}); return; }
            throw std::invalid_argument("FastExpr: unknown function \"" + op.text + "\"");
        }
        throw std::invalid_argument("FastExpr: internal: cannot emit this token");
    };

    for (size_t k = 0; k < tokens.size(); ++k) {
        const Token &t = tokens[k];
        switch (t.kind) {
        case Kind::Number:
            this->emitConst(t.value);
            break;
        case Kind::Name: {
            // If the next token is '(' this is a function call; otherwise a
            // variable or named constant.
            const bool isFunction = (k + 1 < tokens.size() && tokens[k+1].kind == Kind::LParen);
            if (isFunction) {
                opStack.push_back(t);
            } else if (t.text == "x") {
                mCode.push_back({Op::PushX, 0});
            } else if (t.text == "pi") {
                this->emitConst(static_cast<float>(M_PI));
            } else if (t.text == "e") {
                this->emitConst(static_cast<float>(M_E));
            } else {
                throw std::invalid_argument(
                    "FastExpr: unknown variable or constant \"" + t.text + "\"");
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
                throw std::invalid_argument("FastExpr: ',' outside function call");
            break;
        case Kind::RParen:
            while (!opStack.empty() && opStack.back().kind != Kind::LParen) {
                emit(opStack.back());
                opStack.pop_back();
            }
            if (opStack.empty())
                throw std::invalid_argument("FastExpr: mismatched closing parenthesis");
            opStack.pop_back();// discard the LParen
            if (!opStack.empty() && opStack.back().kind == Kind::Name) {
                emit(opStack.back());
                opStack.pop_back();
            }
            break;
        default:
            throw std::invalid_argument("FastExpr: internal: unexpected infix token");
        }
    }
    while (!opStack.empty()) {
        const Token &top = opStack.back();
        if (top.kind == Kind::LParen)
            throw std::invalid_argument("FastExpr: mismatched opening parenthesis");
        emit(top);
        opStack.pop_back();
    }
}

inline void FastExpr::verify()
{
    // Walk the bytecode with a virtual stack-depth counter so eval() doesn't
    // have to check anything at runtime. Any well-formed expression must leave
    // exactly one value on the stack; anything else is a compile-time error.
    int depth = 0;
    constexpr int kStackMax = 32;
    for (const Instr &i : mCode) {
        switch (i.op) {
        case Op::PushLit: case Op::PushX:
            ++depth;
            break;
        case Op::Neg: case Op::Abs: case Op::Inv: case Op::Sqrt:
        case Op::Sin: case Op::Cos: case Op::Tan:
        case Op::Asin: case Op::Acos: case Op::Atan:
        case Op::Exp: case Op::Ln:  case Op::Log:
        case Op::Floor: case Op::Ceil:
        case Op::Pow2:  case Op::Pow3:
            if (depth < 1) throw std::invalid_argument("FastExpr: stack underflow on unary op");
            break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div: case Op::Pow:
        case Op::Min: case Op::Max:
            if (depth < 2) throw std::invalid_argument("FastExpr: stack underflow on binary op");
            --depth;
            break;
        }
        if (depth > kStackMax)
            throw std::invalid_argument("FastExpr: stack overflow (depth > 32)");
    }
    if (depth != 1)
        throw std::invalid_argument(
            "FastExpr: expression must produce exactly one value (final depth=" +
            std::to_string(depth) + ")");
}

inline float FastExpr::eval(float x) const
{
    // 32 slots is far more than any reasonable expression needs; verify()
    // enforces the bound at compile time so this loop performs zero checks.
    constexpr int kStackMax = 32;
    float s[kStackMax];
    int sp = 0;
    for (const Instr &i : mCode) {
        switch (i.op) {
        case Op::PushLit: s[sp++] = mConstants[i.arg]; break;
        case Op::PushX:   s[sp++] = x;                 break;
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
    return s[0];
}

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // VDB_TOOL_FASTEXPR_HAS_BEEN_INCLUDED
