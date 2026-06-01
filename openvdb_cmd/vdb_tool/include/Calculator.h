// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

////////////////////////////////////////////////////////////////////////////////
///
/// @author Ken Museth
///
/// @file Calculator.h
///
/// @brief A compact bytecode interpreter for user-supplied math expressions
///        over an arbitrary number of named float inputs. Compiled once,
///        then evaluated many times &mdash; like a pocket calculator, hence
///        the name. Used by vdb_tool's standalone `-calc` action and by the
///        per-voxel kernels of `-forAllValues` / `-forOnValues` /
///        `-forOffValues`.
///
/// @details Accepts three syntaxes that compile to a single shared bytecode:
///          - Reverse Polish Notation (RPN), colon-separated, e.g.
///              "$x:sin:$x:pow2:2:*:+"
///            (the same language as the vdb_tool string Processor).
///          - A single infix expression, e.g.
///              "sin(x) + 2*x*x"
///          - A semicolon-separated infix program with assignments, e.g.
///              "t = x*x; t + sin(t)"
///            and optional user-defined functions, e.g.
///              "def hyp(a, b) = sqrt(a*a + b*b); hyp(3, 4)"
///          The compiler dispatches automatically: '=' or ';' &rarr; multi-
///          statement infix; ':' or '$' &rarr; RPN; otherwise single infix.
///          The eval() loop is shared across all three. No external dependencies.
///
/// @details Beyond plain expression evaluation, the implementation includes:
///          - **Constant folding** &mdash; literal-only subexpressions are
///            collapsed at compile time.
///          - **Lazy `if(cond, then, else)`** &mdash; only the taken branch is
///            evaluated, via Jump / JumpIfFalse opcodes inserted by a
///            `lazifyBranches()` post-compile pass. `switch` is currently
///            evaluated eagerly.
///          - **User-defined functions** &mdash; `def name(params) = body`
///            statements register functions whose bodies are inlined at each
///            call site (no recursion).
///          - **Persistent memory** &mdash; `evalAndRemember()` snapshots
///            every input, intermediate slot, and named result into a
///            `std::unordered_map<std::string, float>` for post-eval
///            inspection. Plain `eval()` remains `const` and thread-safe.
///          - **Batched evaluation** via `eval_n()` for vector-style transforms.
///          - **Diagnostics** &mdash; tokenizer errors include a source
///            pointer with column number, and `disassemble()` produces a
///            human-readable bytecode dump for debugging.
///
////////////////////////////////////////////////////////////////////////////////

#ifndef VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED
#define VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED

#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
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
///        an arbitrary number of named float variables. Re-entrant and
///        thread-safe at eval() time (the per-call working stack lives on the
///        C stack); `evalAndRemember()` is the one non-const exception.
///
/// @details Variable names are discovered automatically by compile(): any
///          identifier in the expression that is not a built-in constant
///          (`pi`, `tau`, `e`, `phi`, `inf`, `nan`) and not the name of a
///          known function (`sin`, `abs`, `clamp`, `if`, `switch`, ...) and
///          not a previously declared local slot or user-defined function
///          becomes an *input variable*. The full list, in order of first
///          appearance, is accessible via variables(). The general eval()
///          overload binds a parallel float buffer to variables(); other
///          overloads cover the common cases (single `x`, initializer_list,
///          by-name map, and the no-input form for constant expressions),
///          and `eval_n()` applies a single-variable kernel across an array.
///
/// @details (Multi-statement, infix only.) The expression may also contain
///          semicolon-separated statements:
///          - **Assignment** &mdash; `t = x*x` declares a local slot `t`.
///            Slot names are local (not exposed by variables()), shadow any
///            input variable of the same name from that point on, and remain
///            available to subsequent statements. The final statement may be
///            a plain expression or an assignment; either way its right-hand
///            side is what eval() returns. Each intermediate (non-final)
///            statement must be an assignment, since a plain expression
///            would strand a value on the stack.
///          - **User-defined function** &mdash; `def name(p1, p2, ...) = body`
///            registers a function whose body is inlined at each call site.
///            No recursion: a function referencing itself fails with
///            "unknown function". Free variables in the body (anything not in
///            the parameter list) are rejected at compile time. A `def`
///            statement emits no caller-side bytecode and therefore cannot
///            be the final statement.
///
/// @details Operators and built-ins (see README.md for the full table):
///          arithmetic `+ - * / % ^`; comparisons `< > <= >= == !=`
///          returning 0/1; logical `&& || !` returning 0/1; functions for
///          trig, hyperbolic, exp/log, rounding, sign, plus 2/3-arg helpers
///          `pow`, `min`, `max`, `atan2`, `hypot`, `step`, `clamp`, `lerp`
///          (alias `mix`), `smoothstep`, `if`/`select`, and the variadic
///          `switch(selector, k1, v1, ..., default)`.
///
/// @details `if(cond, then, else)` is **lazy** &mdash; only the taken branch
///          executes at runtime, so kernels can write `if(x>=0, sqrt(x), 0)`
///          without first evaluating sqrt(-x). `switch` is currently eager.
///          Compile-time passes include constant folding (literal-only
///          subexpressions collapse to a single PushLit) and lazifyBranches
///          (rewrites eager IfThenElse opcodes into Jump-based bytecode).
///
/// @details Diagnostics: tokenizer errors include a caret-and-column source
///          pointer, and disassemble() returns a human-readable bytecode
///          dump &mdash; useful when validating the effect of the optimization
///          passes or debugging kernels emitted by vdb_tool's `-calc` action
///          and the per-voxel kernels of `-forAllValues` / `-forOnValues` /
///          `-forOffValues`.
class Calculator
{
public:

    /// @brief Compile an expression into bytecode. Replaces any previous
    ///        compilation result (and clears any registered user-defined
    ///        functions and persistent memory).
    /// @param expr One of:
    ///        - an RPN expression with colon-separated tokens, e.g.
    ///          `"$x:sin:$x:pow2:2:*:+"` (single statement, no `=`/`;`);
    ///        - a single infix expression, e.g. `"sin(theta) + radius*radius"`;
    ///        - or an infix multi-statement program with `;` separators,
    ///          assignments and/or `def` declarations, e.g.
    ///          `"def sq(x) = x*x; sq(3) + sq(4)"` or
    ///          `"t = x*x; t + sin(t)"`.
    ///        Any identifier that is not one of the built-in constants
    ///        (`pi`, `tau`, `e`, `phi`, `inf`, `nan`), not a known function,
    ///        not a previously declared local slot, and not a previously
    ///        declared user-defined function becomes an input variable,
    ///        queryable via variables().
    /// @throw std::invalid_argument on syntax errors, unknown function names,
    ///        mismatched parentheses or argument counts, stack underflow,
    ///        an intermediate plain expression (which would strand a value),
    ///        a `def` body referencing free variables, a `def` as the final
    ///        statement, or a final stack depth other than 1. Tokenizer
    ///        errors include a source pointer with column number.
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

    /// @brief Convenience overload for constant expressions (no input
    ///        variables) such as `1+2+3`. Throws if the compiled expression
    ///        references any variable.
    inline float eval() const;

    /// @brief Batched evaluation for single-variable expressions. Applies the
    ///        compiled kernel to @a n inputs and writes the results into
    ///        @a out. Requires the expression to reference exactly the one
    ///        variable named by @a varName (defaults to "x") &mdash; throws
    ///        otherwise. Useful for per-element transforms over an array
    ///        without paying repeated convenience-overload validation costs.
    inline void eval_n(const float* in, float* out, size_t n,
                       const std::string &varName = "x") const;

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

    /// @brief Designate @a name as a "neighbor function": when called with
    ///        three integer-literal arguments (e.g. `v(1, 0, 0)`), the call
    ///        is rewritten at compile time to a single input variable named
    ///        `name(dx,dy,dz)`. The Calculator itself doesn't know what that
    ///        variable means; the binding layer (e.g. Tool::forValues) is
    ///        expected to parse the synthesized name and provide neighbor
    ///        values. Call BEFORE compile(); pass "" (default) to disable.
    void setNeighborFunction(const std::string &name) { mNeighborFn = name; }
    /// @brief Returns the configured neighbor-function name ("" if disabled).
    const std::string& neighborFunction() const { return mNeighborFn; }

    /// @brief Returns true if compile() has not yet been called (or failed).
    bool empty() const { return mCode.empty(); }
    /// @brief Number of bytecode instructions; useful for tests and debugging.
    size_t size() const { return mCode.size(); }

    /// @brief Multi-line, human-readable dump of the compiled bytecode.
    ///        Shows instruction index, opcode name, and any arg/constant/name
    ///        reference. Intended for debugging kernels and validating compile
    ///        passes such as constant folding.
    inline std::string disassemble() const;

private:

    /// @brief Opcode set. PushLit/PushVar/PushSlot push to the stack; Store
    ///        pops the top and writes it to a slot. The arithmetic ops are
    ///        grouped by stack arity: 1-arg pops 1 / pushes 1; 2-arg pops 2 /
    ///        pushes 1; 3-arg pops 3 / pushes 1. Comparison and logical ops
    ///        return 0.0 or 1.0.
    enum class Op : uint8_t {
        PushLit, PushVar, PushSlot,
        Store,
        // Control flow: unconditional / conditional jumps. The Instr::arg field
        // stores the absolute target index in mCode. JumpIfFalse additionally
        // pops one value (the condition) and only jumps if it equals 0.0.
        Jump, JumpIfFalse,
        // 2-arg arithmetic
        Add, Sub, Mul, Div, Mod, Pow,
        // 1-arg arithmetic
        Neg, Abs, Inv, Sqrt,
        Sin, Cos, Tan, Asin, Acos, Atan,
        Sinh, Cosh, Tanh, Asinh, Acosh, Atanh,
        Exp, Ln, Log, Floor, Ceil,
        Pow2, Pow3, Sign, Round, Trunc,
        // 2-arg arithmetic functions
        Min, Max, Atan2, Hypot, Step,
        // 2-arg comparison (return 0.0/1.0)
        Lt, Gt, Le, Ge, Eq, Ne,
        // 2-arg logical (return 0.0/1.0). Not is 1-arg.
        And, Or, Not,
        // 3-arg (pop three, push one)
        Clamp, Lerp, Smoothstep, IfThenElse,
        // Variadic: switch(selector, k1, v1, ..., kN, vN, default).
        // The Instr::arg field stores N (the number of case pairs); the
        // opcode pops (2*N + 2) values and pushes 1.
        Switch
    };

    /// @brief A single bytecode instruction. 4 bytes; cache-friendly.
    struct Instr {
        Op       op;
        uint16_t arg;  ///< Index into mConstants (PushLit), mVariables
                       ///< (PushVar), or mSlotNames (PushSlot/Store);
                       ///< case count N for Switch; unused otherwise.
    };

    /// @brief Hard limit on the number of distinct local slot names declared
    ///        by intermediate assignments AND parameter/locals introduced by
    ///        inlined user-defined function calls. Used to size the eval-time
    ///        slot buffer on the C stack.
    static constexpr int kSlotsMax = 64;

    std::vector<Instr>       mCode;
    std::vector<float>       mConstants;
    std::vector<std::string> mVariables;
    std::vector<std::string> mSlotNames;

    /// @brief A compiled user-defined function. The body's bytecode references
    ///        its parameters as PushVar opcodes indexed into @a params; at
    ///        each call site we inline a copy of @a code with parameter
    ///        references rewritten to PushSlot ops referring to caller-side
    ///        slots that hold the argument values.
    struct FunctionDef {
        std::vector<std::string> params;
        std::vector<Instr>       code;
        std::vector<float>       constants;
        std::vector<std::string> slotNames;
    };
    std::unordered_map<std::string, FunctionDef> mFunctions;
    /// @brief Name of the function whose `name(dx, dy, dz)` calls are
    ///        rewritten to synthesized neighbor-variable references.
    ///        Set via setNeighborFunction(); empty disables the feature.
    std::string                              mNeighborFn;
    /// @brief Set of function names currently being compiled (recursion guard).
    std::unordered_map<std::string, bool>        mFunctionsInProgress;
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
    void     foldConstants();// peephole: fold PushLit-only subexpressions at compile time
    void     lazifyBranches();// rewrite eager IfThenElse/Switch into Jump-based lazy bytecode
    void     inlineCall(const FunctionDef &fd);// splice a UDF body into mCode at the current position
    void     emitConst(float v);
    void     emitName(const std::string &name);
    uint16_t allocateSlot(const std::string &name);

    /// @brief Bytecode interpreter shared by all eval()/evalAndRemember()
    ///        forms. If @a slotsOut is non-null, the final values of all
    ///        named slots are copied into it (used to snapshot mMemory).
    inline float evalImpl(const float* values, float* slotsOut) const;

    static bool isIdentifier(const std::string &s);

    /// @brief Format a single-line source-pointer error message of the form
    ///        `msg\n  source\n  ^` so users see exactly where the tokenizer
    ///        choked. Returns the concatenated string suitable for passing to
    ///        std::invalid_argument.
    static inline std::string pointerError(const std::string &msg,
                                           const std::string &source,
                                           size_t column);

    /// @brief Build a "Did you mean: ...?" suggestion line for a typo'd
    ///        identifier @a name. Searches every known function and constant
    ///        (and the supplied @a extras list, e.g. user-defined function
    ///        names) ranked by Levenshtein edit distance, returning the up to
    ///        @a maxSuggest closest candidates with distance <= @a maxDist.
    ///        Empty string if no candidate is close enough.
    std::string suggestNames(const std::string &name,
                             const std::vector<std::string> &extras = {},
                             size_t maxDist = 2,
                             size_t maxSuggest = 3) const;

    static const std::unordered_map<std::string, Op>& unaryOps();
    static const std::unordered_map<std::string, Op>& binaryOps();
    static const std::unordered_map<std::string, Op>& ternaryOps();
};

// ====================================================================
//  Inline implementation
// ====================================================================

inline const std::unordered_map<std::string, Calculator::Op>& Calculator::unaryOps()
{
    // Operations that consume one stack entry and replace it with the result.
    static const std::unordered_map<std::string, Op> t = {
        {"neg",   Op::Neg },   {"abs",   Op::Abs },   {"inv",   Op::Inv },
        {"sqrt",  Op::Sqrt},
        {"sin",   Op::Sin },   {"cos",   Op::Cos },   {"tan",   Op::Tan },
        {"asin",  Op::Asin},   {"acos",  Op::Acos},   {"atan",  Op::Atan},
        {"sinh",  Op::Sinh},   {"cosh",  Op::Cosh},   {"tanh",  Op::Tanh},
        {"asinh", Op::Asinh},  {"acosh", Op::Acosh},  {"atanh", Op::Atanh},
        {"exp",   Op::Exp },   {"ln",    Op::Ln  },   {"log",   Op::Log },
        {"floor", Op::Floor},  {"ceil",  Op::Ceil},
        {"pow2",  Op::Pow2},   {"pow3",  Op::Pow3},
        {"sign",  Op::Sign},   {"round", Op::Round},  {"trunc", Op::Trunc},
        {"not",   Op::Not }
    };
    return t;
}

inline const std::unordered_map<std::string, Calculator::Op>& Calculator::binaryOps()
{
    // Operations that consume two stack entries and push one result. The
    // infix-syntax operators are spelled with their punctuation; the
    // word-spelled aliases let RPN use the same table.
    static const std::unordered_map<std::string, Op> t = {
        {"+",   Op::Add}, {"-",   Op::Sub}, {"*",   Op::Mul}, {"/",   Op::Div},
        {"%",   Op::Mod}, {"mod", Op::Mod}, {"fmod",Op::Mod},
        {"^",   Op::Pow}, {"pow", Op::Pow}, {"min", Op::Min}, {"max", Op::Max},
        {"atan2", Op::Atan2}, {"hypot", Op::Hypot}, {"step", Op::Step},
        // Comparisons return 0.0 / 1.0
        {"<",   Op::Lt}, {">",   Op::Gt}, {"<=",  Op::Le}, {">=",  Op::Ge},
        {"==",  Op::Eq}, {"!=",  Op::Ne},
        {"lt",  Op::Lt}, {"gt",  Op::Gt}, {"le",  Op::Le}, {"ge",  Op::Ge},
        {"eq",  Op::Eq}, {"ne",  Op::Ne},
        // Logical (operate on 0.0 / non-zero, return 0.0 / 1.0)
        {"&&",  Op::And}, {"||",  Op::Or},
        {"and", Op::And}, {"or",  Op::Or}
    };
    return t;
}

inline const std::unordered_map<std::string, Calculator::Op>& Calculator::ternaryOps()
{
    // Three-argument functions (clamp(x,lo,hi), lerp(a,b,t) etc.). The
    // shunting-yard handles commas and pushes the args in order; the opcode
    // pops three and pushes one.
    static const std::unordered_map<std::string, Op> t = {
        {"clamp",      Op::Clamp},
        {"lerp",       Op::Lerp},      {"mix", Op::Lerp},
        {"smoothstep", Op::Smoothstep},
        {"if",         Op::IfThenElse}, {"select", Op::IfThenElse}
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

inline std::string Calculator::pointerError(const std::string &msg,
                                            const std::string &source,
                                            size_t column)
{
    // Tabs throw the caret off; replace them with single spaces so the marker
    // lines up. Truncate very long sources to avoid one-line errors that wrap.
    std::string clean = source;
    for (char &c : clean) { if (c == '\t' || c == '\n') c = ' '; }
    constexpr size_t kMaxLen = 80;
    size_t startCol = 0;
    if (clean.size() > kMaxLen) {
        // Center the caret in a 80-char window.
        const size_t half = kMaxLen / 2;
        if (column > half) startCol = column - half;
        if (startCol + kMaxLen > clean.size()) startCol = clean.size() - kMaxLen;
        clean = (startCol > 0 ? "..." : "") + clean.substr(startCol, kMaxLen) +
                (startCol + kMaxLen < source.size() ? "..." : "");
    }
    const size_t caretCol = (column >= startCol) ? (column - startCol + (startCol > 0 ? 3 : 0)) : 0;
    std::string result = msg + "\n  " + clean + "\n  ";
    result.append(caretCol, ' ');
    result += "^  (column ";
    result += std::to_string(column + 1);
    result += ")";
    return result;
}

inline std::string Calculator::suggestNames(const std::string &name,
                                            const std::vector<std::string> &extras,
                                            size_t maxDist,
                                            size_t maxSuggest) const
{
    // Build a candidate list: every unary/binary/ternary function name, the
    // built-in constants, plus the caller-supplied `extras` (typically the
    // names of user-defined functions and currently-discovered variables).
    std::vector<std::string> candidates;
    candidates.reserve(64);
    for (const auto &kv : unaryOps())   candidates.push_back(kv.first);
    for (const auto &kv : binaryOps())  candidates.push_back(kv.first);
    for (const auto &kv : ternaryOps()) candidates.push_back(kv.first);
    candidates.insert(candidates.end(),
                      {"pi", "tau", "e", "phi", "inf", "nan", "switch", "if", "select"});
    for (const std::string &x : extras) candidates.push_back(x);

    // Rank by Levenshtein distance; keep candidates within maxDist. Sort
    // by (distance, name) and dedup so we don't show "Add" twice (the binary
    // op table has both "+" and "add" pointing at the same opcode).
    std::vector<std::pair<size_t, std::string>> ranked;
    ranked.reserve(candidates.size());
    for (const std::string &c : candidates) {
        // Skip the punctuation-only opcode names ("+", "-", "*", "/", "%",
        // "^", "<", ">", "<=", ">=", "==", "!=", "&&", "||") — they aren't
        // valid as function-call names anyway and just noise the suggestions.
        if (c.empty() || !std::isalpha(static_cast<unsigned char>(c[0]))) continue;
        const size_t d = levenshtein(name, c);
        if (d <= maxDist) ranked.emplace_back(d, c);
    }
    std::sort(ranked.begin(), ranked.end());
    // Dedup adjacent (same name, larger distance impossible after sort).
    ranked.erase(std::unique(ranked.begin(), ranked.end(),
                             [](const auto &a, const auto &b){ return a.second == b.second; }),
                 ranked.end());
    if (ranked.empty()) return {};
    std::string out = "Did you mean: ";
    const size_t n = std::min(ranked.size(), maxSuggest);
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) out += (i + 1 == n ? " or " : ", ");
        out += '"';
        out += ranked[i].second;
        out += '"';
    }
    out += '?';
    return out;
}

inline void Calculator::compile(const std::string &expr)
{
    mCode.clear();
    mConstants.clear();
    mVariables.clear();
    mSlotNames.clear();
    mFunctions.clear();
    mFunctionsInProgress.clear();
    mResultName.clear();
    mMemory.clear();
    if (expr.empty()) {
        throw std::invalid_argument("Calculator: empty expression");
    }
    // Dispatch:
    //   '=' or ';' present  -> multi-statement infix (assignments / slots).
    //                          Note: '=' may also be part of '==', '<=', '>=',
    //                          or '!='; compileStatements handles that by
    //                          searching for a *standalone* '='.
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
    this->foldConstants();// must run BEFORE lazifyBranches: fold rewrites the
                          // bytecode without preserving jump-target offsets, so
                          // it cannot safely operate on lazy (jump-containing) code.
    this->lazifyBranches();// rewrite eager IfThenElse into Jump-based bytecode
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

    // Find the first standalone '=' (not part of '==', '<=', '>=', or '!=')
    // in a string. Used by both `def`-statement parsing and the regular
    // assignment-target check below.
    auto findStandaloneEq = [](const std::string &str) -> size_t {
        for (size_t i = 0; i < str.size(); ++i) {
            if (str[i] != '=') continue;
            if (i + 1 < str.size() && str[i+1] == '=') { ++i; continue; }// skip '=='
            if (i > 0 && (str[i-1] == '<' || str[i-1] == '>' ||
                          str[i-1] == '!' || str[i-1] == '=')) continue;// part of <=, >=, !=, or trailing of ==
            return i;
        }
        return std::string::npos;
    };

    for (size_t idx = 0; idx < stmts.size(); ++idx) {
        const std::string &s = stmts[idx];
        const bool isLast = (idx + 1 == stmts.size());

        // ----------- def name(params) = body -----------
        // A `def`-statement registers a user-defined function. It does NOT
        // emit any caller-side bytecode (the body is inlined at each call
        // site). `def` can only appear as a non-final intermediate statement,
        // since it produces no value.
        if (s.compare(0, 4, "def ") == 0 || s.compare(0, 4, "def\t") == 0) {
            if (isLast) {
                throw std::invalid_argument(
                    "Calculator: `def` cannot be the final statement (it produces no value)");
            }
            // Find '(' and ')' in the LHS-equivalent portion (before the standalone '=').
            const size_t eqPos = findStandaloneEq(s);
            if (eqPos == std::string::npos) {
                throw std::invalid_argument(
                    "Calculator: `def` statement requires `= body`");
            }
            const std::string header = trim(s.substr(4, eqPos - 4));// skip "def "
            const std::string body   = trim(s.substr(eqPos + 1));
            const size_t lp = header.find('(');
            const size_t rp = header.rfind(')');
            if (lp == std::string::npos || rp == std::string::npos || rp < lp) {
                throw std::invalid_argument(
                    "Calculator: `def` must take the form `def name(p1, p2, ...) = body`");
            }
            const std::string name = trim(header.substr(0, lp));
            const std::string paramList = header.substr(lp + 1, rp - lp - 1);
            if (!isIdentifier(name)) {
                throw std::invalid_argument(
                    "Calculator: invalid function name \"" + name + "\"");
            }
            if (mFunctions.count(name)) {
                throw std::invalid_argument(
                    "Calculator: function \"" + name + "\" is already defined");
            }
            // Reject overshadowing built-ins.
            if (unaryOps().count(name) || binaryOps().count(name) || ternaryOps().count(name) ||
                name == "switch" || name == "pi" || name == "tau" || name == "e" ||
                name == "phi" || name == "inf" || name == "nan") {
                throw std::invalid_argument(
                    "Calculator: cannot redefine built-in name \"" + name + "\"");
            }
            // Parse parameter list (comma-separated identifiers, possibly empty).
            std::vector<std::string> params;
            {
                std::string acc;
                for (size_t i = 0; i <= paramList.size(); ++i) {
                    if (i == paramList.size() || paramList[i] == ',') {
                        std::string p = trim(acc);
                        if (!p.empty()) {
                            if (!isIdentifier(p)) {
                                throw std::invalid_argument(
                                    "Calculator: invalid parameter \"" + p +
                                    "\" in def \"" + name + "\"");
                            }
                            params.push_back(p);
                        }
                        acc.clear();
                    } else {
                        acc += paramList[i];
                    }
                }
            }
            if (body.empty()) {
                throw std::invalid_argument(
                    "Calculator: def body for \"" + name + "\" is empty");
            }
            // Compile body in a sub-scope: save caller's compile state, install
            // a fresh state pre-populated with params as the input variables,
            // compile, then restore.
            mFunctionsInProgress[name] = true;
            std::vector<Instr>       saved_code;      std::swap(saved_code, mCode);
            std::vector<float>       saved_consts;    std::swap(saved_consts, mConstants);
            std::vector<std::string> saved_vars;      std::swap(saved_vars, mVariables);
            std::vector<std::string> saved_slots;     std::swap(saved_slots, mSlotNames);
            std::string              saved_result;    std::swap(saved_result, mResultName);
            mVariables = params;// pre-bind so PushVar(i) inside the body refers to param i

            FunctionDef fd;
            fd.params = params;
            try {
                this->compileInfix(body);
            } catch (...) {
                std::swap(saved_code,   mCode);
                std::swap(saved_consts, mConstants);
                std::swap(saved_vars,   mVariables);
                std::swap(saved_slots,  mSlotNames);
                std::swap(saved_result, mResultName);
                mFunctionsInProgress.erase(name);
                throw;
            }
            // Free variables (not in params) are a hard error.
            if (mVariables.size() != params.size()) {
                std::vector<std::string> body_vars = std::move(mVariables);
                std::swap(saved_code,   mCode);
                std::swap(saved_consts, mConstants);
                std::swap(saved_vars,   mVariables);
                std::swap(saved_slots,  mSlotNames);
                std::swap(saved_result, mResultName);
                mFunctionsInProgress.erase(name);
                throw std::invalid_argument(
                    "Calculator: def \"" + name + "\" references free variable \"" +
                    body_vars[params.size()] + "\"; only declared parameters are allowed");
            }
            fd.code      = std::move(mCode);
            fd.constants = std::move(mConstants);
            fd.slotNames = std::move(mSlotNames);

            // Restore caller's state.
            mCode       = std::move(saved_code);
            mConstants  = std::move(saved_consts);
            mVariables  = std::move(saved_vars);
            mSlotNames  = std::move(saved_slots);
            mResultName = std::move(saved_result);
            mFunctionsInProgress.erase(name);
            mFunctions[name] = std::move(fd);
            continue;// no bytecode emitted at the caller level
        }
        // ----------- end def parsing -----------

        // Detect 'lhs = rhs'. findStandaloneEq is hoisted above the loop.
        std::string lhs, rhs;
        const size_t eq = findStandaloneEq(s);
        if (eq != std::string::npos) {
            lhs = trim(s.substr(0, eq));
            rhs = trim(s.substr(eq + 1));
            if (findStandaloneEq(rhs) != std::string::npos) {
                throw std::invalid_argument(
                    "Calculator: chained '=' is not supported (statement: \"" + s + "\")");
            }
            if (!isIdentifier(lhs)) {
                throw std::invalid_argument(
                    "Calculator: invalid assignment target \"" + lhs + "\"");
            }
            if (lhs == "pi" || lhs == "e" || lhs == "tau" ||
                lhs == "phi" || lhs == "inf" || lhs == "nan") {
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
    const auto &unary   = unaryOps();
    const auto &binary  = binaryOps();
    const auto &ternary = ternaryOps();

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
        auto tn = ternary.find(tok);
        if (tn != ternary.end()) { mCode.push_back({tn->second, 0}); continue; }

        // Identifier: variable or named constant. Optional leading '$' is
        // stripped to match the Processor's convention ($x, $pi, ...).
        std::string name = tok;
        if (!name.empty() && name[0] == '$') name.erase(0, 1);
        if (!isIdentifier(name)) {
            {
                std::vector<std::string> extras;
                for (const auto &kv : mFunctions) extras.push_back(kv.first);
                const std::string hint = this->suggestNames(name, extras);
                std::string msg = "Calculator: unknown token \"" + tok + "\"";
                if (!hint.empty()) msg += "\n  " + hint;
                throw std::invalid_argument(msg);
            }
        }
        if (name == "pi"  ) { this->emitConst(static_cast<float>(M_PI)); continue; }
        if (name == "tau" ) { this->emitConst(static_cast<float>(2.0L * M_PI)); continue; }
        if (name == "e"   ) { this->emitConst(static_cast<float>(M_E )); continue; }
        if (name == "phi" ) { this->emitConst(1.6180339887498949f); continue; }
        if (name == "inf" ) { this->emitConst(std::numeric_limits<float>::infinity()); continue; }
        if (name == "nan" ) { this->emitConst(std::numeric_limits<float>::quiet_NaN()); continue; }
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
        size_t      col        = 0;  ///< Starting column in the source expression (for error reporting).
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
        t.col = i;
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
          else if (c == '+' || c == '-' || c == '*' || c == '/' ||
                   c == '%' || c == '^') {
            // Unary +/- is signalled by what came before: nothing, a binary op,
            // another unary op, an opening paren, or a comma.
            const bool isUnary = (c == '+' || c == '-') &&
                                 (prev == Kind::End    || prev == Kind::BinOp ||
                                  prev == Kind::UnaryOp|| prev == Kind::LParen||
                                  prev == Kind::Comma);
            t.text = std::string(1, c);
            if (isUnary) {
                t.kind       = Kind::UnaryOp;
                t.precedence = 8;
                t.rightAssoc = true;
            } else {
                t.kind = Kind::BinOp;
                switch (c) {
                    case '+': t.precedence = 5; break;
                    case '-': t.precedence = 5; break;
                    case '*': t.precedence = 6; break;
                    case '/': t.precedence = 6; break;
                    case '%': t.precedence = 6; break;
                    case '^': t.precedence = 7; t.rightAssoc = true; break;
                }
            }
            ++i;
        } else if (c == '<' || c == '>') {
            // <, <=, >, >=
            t.kind       = Kind::BinOp;
            t.precedence = 4;
            if (i+1 < N && expr[i+1] == '=') {
                t.text = (c == '<') ? "<=" : ">=";
                i += 2;
            } else {
                t.text = std::string(1, c);
                ++i;
            }
        } else if (c == '=') {
            // Only '==' is valid in expressions; bare '=' is the statement
            // separator handled at the multi-statement level.
            if (i+1 < N && expr[i+1] == '=') {
                t.kind       = Kind::BinOp;
                t.precedence = 3;
                t.text       = "==";
                i += 2;
            } else {
                throw std::invalid_argument(pointerError(
                    "Calculator: unexpected '=' in expression (use '==' for equality, "
                    "and only the top-level statement may assign)", expr, i));
            }
        } else if (c == '!') {
            // '!=' (binary) or unary logical NOT
            if (i+1 < N && expr[i+1] == '=') {
                t.kind       = Kind::BinOp;
                t.precedence = 3;
                t.text       = "!=";
                i += 2;
            } else {
                t.kind       = Kind::UnaryOp;
                t.precedence = 8;
                t.rightAssoc = true;
                t.text       = "!";
                ++i;
            }
        } else if (c == '&') {
            // Only '&&' is valid.
            if (i+1 < N && expr[i+1] == '&') {
                t.kind       = Kind::BinOp;
                t.precedence = 2;
                t.text       = "&&";
                i += 2;
            } else {
                throw std::invalid_argument(pointerError(
                    "Calculator: unexpected '&' in expression (use '&&' for logical AND)",
                    expr, i));
            }
        } else if (c == '|') {
            // Only '||' is valid.
            if (i+1 < N && expr[i+1] == '|') {
                t.kind       = Kind::BinOp;
                t.precedence = 1;
                t.text       = "||";
                i += 2;
            } else {
                throw std::invalid_argument(pointerError(
                    "Calculator: unexpected '|' in expression (use '||' for logical OR)",
                    expr, i));
            }
        } else {
            throw std::invalid_argument(pointerError(
                "Calculator: unexpected character '" + std::string(1, c) + "' in expression",
                expr, i));
        }
        tokens.push_back(t);
        prev = t.kind;
    }

    // Shunting-yard, emitting straight into mCode/mConstants.
    std::vector<Token> opStack;
    // commaCounts[i] counts the commas seen at the i-th currently-open
    // paren scope (synchronized with the LParen entries on opStack).
    // Used to determine the arity of variadic functions like switch().
    std::vector<int> commaCounts;
    const auto &unary   = unaryOps();
    const auto &binary  = binaryOps();
    const auto &ternary = ternaryOps();

    auto emit = [&](const Token &op) {
        if (op.kind == Kind::UnaryOp) {
            if      (op.text == "-") mCode.push_back({Op::Neg, 0});
            else if (op.text == "!") mCode.push_back({Op::Not, 0});
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
            auto tn = ternary.find(op.text);
            if (tn != ternary.end()) { mCode.push_back({tn->second, 0}); return; }
            {
                std::vector<std::string> extras;
                for (const auto &kv : mFunctions) extras.push_back(kv.first);
                const std::string hint = this->suggestNames(op.text, extras);
                std::string msg = "Calculator: unknown function \"" + op.text + "\"";
                if (!hint.empty()) msg += "\n  " + hint;
                throw std::invalid_argument(msg);
            }
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
            } else if (t.text == "pi" ) {
                this->emitConst(static_cast<float>(M_PI));
            } else if (t.text == "tau") {
                this->emitConst(static_cast<float>(2.0L * M_PI));
            } else if (t.text == "e"  ) {
                this->emitConst(static_cast<float>(M_E));
            } else if (t.text == "phi") {
                this->emitConst(1.6180339887498949f);
            } else if (t.text == "inf") {
                this->emitConst(std::numeric_limits<float>::infinity());
            } else if (t.text == "nan") {
                this->emitConst(std::numeric_limits<float>::quiet_NaN());
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
            commaCounts.push_back(0);// new paren scope; track commas for variadic-arity dispatch
            break;
        case Kind::Comma:
            while (!opStack.empty() && opStack.back().kind != Kind::LParen) {
                emit(opStack.back());
                opStack.pop_back();
            }
            if (opStack.empty() || commaCounts.empty())
                throw std::invalid_argument(pointerError(
                    "Calculator: ',' outside function call", expr, t.col));
            ++commaCounts.back();
            break;
        case Kind::RParen: {
            while (!opStack.empty() && opStack.back().kind != Kind::LParen) {
                emit(opStack.back());
                opStack.pop_back();
            }
            if (opStack.empty())
                throw std::invalid_argument(pointerError(
                    "Calculator: mismatched closing parenthesis", expr, t.col));
            const int commas = commaCounts.empty() ? 0 : commaCounts.back();
            if (!commaCounts.empty()) commaCounts.pop_back();
            opStack.pop_back();// discard the LParen
            if (!opStack.empty() && opStack.back().kind == Kind::Name) {
                const std::string fname = opStack.back().text;
                const int argCount = commas + 1;// each comma separates two args; ≥1 if parens were non-empty
                // Neighbor-function rewrite: f(dx, dy, dz) with integer-
                // literal args -> a single PushVar with a synthesized name
                // "f(dx,dy,dz)". The binding layer is expected to interpret
                // the synthesized name as a relative neighbor lookup.
                if (!mNeighborFn.empty() && fname == mNeighborFn) {
                    if (argCount != 3) {
                        throw std::invalid_argument(
                            "Calculator: \"" + fname + "(...)\" denotes a voxel neighbor "
                            "and requires exactly 3 integer offsets, e.g. " +
                            fname + "(1, 0, 0) — got " + std::to_string(argCount) +
                            " argument(s)");
                    }
                    // Pop each argument off the back of mCode. An argument
                    // is either a single PushLit (positive literal) or a
                    // pair PushLit-then-Neg (negative literal — unary '-'
                    // compiles to Neg in this codepath since constant
                    // folding hasn't run yet). Anything else means the
                    // user wrote a runtime expression and we reject it.
                    auto popOffset = [&](int argIdx) -> int {
                        if (mCode.empty()) {
                            throw std::invalid_argument(
                                "Calculator: \"" + fname + "(dx, dy, dz)\": missing "
                                "literal for arg " + std::to_string(argIdx));
                        }
                        bool negate = false;
                        if (mCode.back().op == Op::Neg) {
                            negate = true;
                            mCode.pop_back();
                            if (mCode.empty()) {
                                throw std::invalid_argument(
                                    "Calculator: \"" + fname + "(dx, dy, dz)\": unary '-' "
                                    "without literal for arg " + std::to_string(argIdx));
                            }
                        }
                        if (mCode.back().op != Op::PushLit) {
                            throw std::invalid_argument(
                                "Calculator: \"" + fname + "(dx, dy, dz)\" requires "
                                "integer-literal offsets (got a runtime expression for "
                                "arg " + std::to_string(argIdx) + ")");
                        }
                        const float f = mConstants[mCode.back().arg];
                        if (std::isnan(f) || std::isinf(f) || f != std::floor(f)) {
                            throw std::invalid_argument(
                                "Calculator: \"" + fname + "(dx, dy, dz)\" requires "
                                "integer offsets (got " + std::to_string(f) +
                                " for arg " + std::to_string(argIdx) + ")");
                        }
                        mCode.pop_back();
                        const int v = static_cast<int>(f);
                        return negate ? -v : v;
                    };
                    // Args were pushed left-to-right, so we pop right-to-left.
                    const int dz_v = popOffset(3);
                    const int dy_v = popOffset(2);
                    const int dx_v = popOffset(1);
                    const int off[3] = {dx_v, dy_v, dz_v};
                    // Synthesize a variable name encoding the offset. The
                    // form "name(dx,dy,dz)" mirrors what the user wrote;
                    // the parentheses make it unambiguous vs. ordinary names.
                    const std::string nbrName = mNeighborFn + "(" +
                        std::to_string(off[0]) + "," +
                        std::to_string(off[1]) + "," +
                        std::to_string(off[2]) + ")";
                    this->emitName(nbrName);
                } else if (fname == "switch") {
                    if (argCount < 4 || (argCount % 2) != 0) {
                        throw std::invalid_argument(
                            "Calculator: switch() requires an even number of arguments "
                            ">= 4 (form: switch(selector, k1, v1, ..., kN, vN, default), "
                            "got " + std::to_string(argCount) + ")");
                    }
                    const uint16_t cases = static_cast<uint16_t>((argCount - 2) / 2);
                    mCode.push_back({Op::Switch, cases});
                } else {
                    // User-defined function call? Inline its body.
                    auto fnIt = mFunctions.find(fname);
                    if (fnIt != mFunctions.end()) {
                        const FunctionDef &fd = fnIt->second;
                        if (argCount != static_cast<int>(fd.params.size())) {
                            throw std::invalid_argument(
                                "Calculator: function \"" + fname + "\" expects " +
                                std::to_string(fd.params.size()) + " argument(s), got " +
                                std::to_string(argCount));
                        }
                        this->inlineCall(fd);
                    } else {
                        emit(opStack.back());
                    }
                }
                opStack.pop_back();
            }
            break;
        }
            break;
        default:
            throw std::invalid_argument("Calculator: internal: unexpected infix token");
        }
    }
    while (!opStack.empty()) {
        const Token &top = opStack.back();
        if (top.kind == Kind::LParen)
            throw std::invalid_argument(pointerError(
                "Calculator: mismatched opening parenthesis", expr, top.col));
        emit(top);
        opStack.pop_back();
    }
}

inline void Calculator::inlineCall(const FunctionDef &fd)
{
    // The call site has just pushed `fd.params.size()` argument values onto
    // the runtime stack (last param on top). Plan:
    //   1. Allocate a unique slot in mSlotNames for each parameter.
    //   2. Emit Store opcodes in REVERSE order (top-of-stack pops first into
    //      the LAST parameter's slot).
    //   3. Splice in a copy of fd.code, rewriting:
    //      - PushLit: relocate the constant pool index.
    //      - PushVar i (body-side parameter reference): rewrite as PushSlot
    //        of the caller's slot for that parameter.
    //      - PushSlot/Store: relocate the slot-name index.
    //      - Jump/JumpIfFalse: relocate to absolute caller-side positions.
    //
    // The body has been compiled with body-level lazifyBranches/foldConstants
    // already applied (compile() runs them on the sub-scope before this is
    // called), so body's jumps are already encoded as absolute targets within
    // the body's own bytecode coordinate frame; we just shift them by the
    // pre-splice mCode.size().

    // Step 1: per-call parameter slots. Use a fresh, uniquified name per call
    // so nested calls don't collide.
    static int call_counter = 0;
    const int call_id = ++call_counter;
    std::vector<uint16_t> paramSlot(fd.params.size());
    for (size_t i = 0; i < fd.params.size(); ++i) {
        paramSlot[i] = this->allocateSlot(
            "__arg" + std::to_string(call_id) + "_" + fd.params[i]);
    }

    // Step 2: store args in reverse (top-of-stack first).
    for (size_t i = fd.params.size(); i-- > 0; ) {
        mCode.push_back({Op::Store, paramSlot[i]});
    }

    // Pre-compute offsets used to rewrite the body.
    const uint16_t constOffset = static_cast<uint16_t>(mConstants.size());
    const uint16_t slotOffset  = static_cast<uint16_t>(mSlotNames.size());
    const size_t   codeOffset  = mCode.size();

    // Append body's constants.
    for (float c : fd.constants) mConstants.push_back(c);
    // Append body's slot names (the body-local ones, distinct from param slots).
    for (const std::string &n : fd.slotNames) {
        mSlotNames.push_back("__local" + std::to_string(call_id) + "_" + n);
    }

    // Step 3: copy the body's bytecode, rewriting indices.
    for (const Instr &in : fd.code) {
        Instr out = in;
        switch (in.op) {
        case Op::PushLit:
            out.arg = static_cast<uint16_t>(in.arg + constOffset);
            break;
        case Op::PushVar:
            // Body's PushVar(i) refers to fd.params[i]. Rewrite as PushSlot
            // of the caller's per-call parameter slot.
            out.op  = Op::PushSlot;
            out.arg = paramSlot[in.arg];
            break;
        case Op::PushSlot: case Op::Store:
            out.arg = static_cast<uint16_t>(in.arg + slotOffset);
            break;
        case Op::Jump: case Op::JumpIfFalse:
            out.arg = static_cast<uint16_t>(in.arg + codeOffset);
            break;
        default:
            break;
        }
        mCode.push_back(out);
    }
}

inline void Calculator::lazifyBranches()
{
    // Rewrite eager Op::IfThenElse into Jump/JumpIfFalse-based lazy form so
    // only the taken branch is evaluated. Op::Switch is left eager: lazy
    // switch needs the selector duplicated across each case test, which would
    // require a Dup opcode or anonymous slot — bigger surgery, deferred.
    //
    // Approach: forward-walk mCode, building newCode while tracking the
    // start-position of each live stack value (origin stack). When we hit
    // Op::IfThenElse, the three top origins point at the start of cond/then/
    // else spans inside newCode; splice in JumpIfFalse and Jump and patch
    // any existing jump targets that get shifted by the insertions.
    if (mCode.empty()) return;

    auto stackEffect = [](Op op, int arg) -> std::pair<int,int> {
        // returns {pops, pushes}
        switch (op) {
        case Op::PushLit: case Op::PushVar: case Op::PushSlot: return {0, 1};
        case Op::Store:                                         return {1, 0};
        case Op::Jump:                                          return {0, 0};
        case Op::JumpIfFalse:                                   return {1, 0};
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        case Op::Mod: case Op::Pow:
        case Op::Min: case Op::Max: case Op::Atan2: case Op::Hypot: case Op::Step:
        case Op::Lt: case Op::Gt: case Op::Le: case Op::Ge:
        case Op::Eq: case Op::Ne:
        case Op::And: case Op::Or:                              return {2, 1};
        case Op::Neg: case Op::Abs: case Op::Inv: case Op::Sqrt:
        case Op::Sin: case Op::Cos: case Op::Tan:
        case Op::Asin: case Op::Acos: case Op::Atan:
        case Op::Sinh: case Op::Cosh: case Op::Tanh:
        case Op::Asinh: case Op::Acosh: case Op::Atanh:
        case Op::Exp: case Op::Ln: case Op::Log:
        case Op::Floor: case Op::Ceil:
        case Op::Pow2: case Op::Pow3:
        case Op::Sign: case Op::Round: case Op::Trunc: case Op::Not:
                                                                return {1, 1};
        case Op::Clamp: case Op::Lerp: case Op::Smoothstep: case Op::IfThenElse:
                                                                return {3, 1};
        case Op::Switch:                                        return {2 * arg + 2, 1};
        }
        return {0, 0};
    };

    std::vector<Instr> newCode;
    newCode.reserve(mCode.size() + 8);
    std::vector<size_t> originStack;
    originStack.reserve(16);

    // Shift any existing Jump/JumpIfFalse target STRICTLY greater than `from`
    // by `delta`. Targets equal to `from` stay put: when inserting at position
    // P, the new instruction occupies P and the old content moves to P+1, but
    // jumps that pointed at "between P-1 and P" semantically want to land on
    // the new instruction at P. Pointing at P itself (the inserted Jump/JIF)
    // is the correct semantics for branch-end markers; we only shift jumps
    // whose targets sat beyond P.
    auto shiftTargets = [&newCode](size_t from, int delta) {
        for (Instr &j : newCode) {
            if ((j.op == Op::Jump || j.op == Op::JumpIfFalse) && j.arg > from) {
                j.arg = static_cast<uint16_t>(static_cast<int>(j.arg) + delta);
            }
        }
    };

    for (size_t k = 0; k < mCode.size(); ++k) {
        const Instr in = mCode[k];
        const auto [pops, pushes] = stackEffect(in.op, static_cast<int>(in.arg));

        if (in.op == Op::IfThenElse) {
            // Origin stack top-3 are cond/then/else start positions in newCode.
            const size_t cond_start = originStack[originStack.size() - 3];
            const size_t then_start = originStack[originStack.size() - 2];
            const size_t else_start = originStack[originStack.size() - 1];
            const size_t end_pos    = newCode.size();// position AFTER else bytecode
            (void)cond_start;// only used for clarity / documentation

            // Step 1: insert Jump (target patched later) right BEFORE else_start.
            // After insertion, any jumps with target >= else_start shift by +1.
            shiftTargets(else_start, +1);
            const size_t jump_pos = else_start;
            newCode.insert(newCode.begin() + jump_pos, Instr{Op::Jump, 0});
            const size_t new_else_start = else_start + 1;
            const size_t new_end_pos    = end_pos    + 1;

            // Step 2: insert JumpIfFalse right BEFORE then_start.
            shiftTargets(then_start, +1);
            const size_t jif_pos = then_start;
            newCode.insert(newCode.begin() + jif_pos, Instr{Op::JumpIfFalse, 0});
            const size_t new_jump_pos      = jump_pos      + 1;
            const size_t new_new_else_start = new_else_start + 1;
            const size_t new_new_end_pos    = new_end_pos    + 1;

            // Step 3: set targets.
            newCode[jif_pos ].arg = static_cast<uint16_t>(new_new_else_start);
            newCode[new_jump_pos].arg = static_cast<uint16_t>(new_new_end_pos);

            // Don't emit Op::IfThenElse — its job is done by the jumps.
            // Origin stack: pop 3, push 1 (result origin = cond_start, the
            // start of the entire branch construct).
            originStack.pop_back();
            originStack.pop_back();
            // top is now cond_start; leave it as the merged origin.
            // (It correctly identifies the start of this lazy if expression.)
            continue;
        }

        // Default: copy instruction; rewrite jump targets if it's a jump
        // (preserving relative semantics — Jump targets in mCode are unused
        // pre-lazify, but be safe in case re-running this pass).
        newCode.push_back(in);

        // Update origin stack to mirror runtime stack effects.
        const size_t resultOrigin = (pushes == 1 && pops >= 1)
            ? originStack[originStack.size() - pops]
            : newCode.size() - 1;
        for (int p = 0; p < pops; ++p) originStack.pop_back();
        if (pushes == 1) originStack.push_back(resultOrigin);
    }

    mCode = std::move(newCode);
}

inline void Calculator::foldConstants()
{
    // Single pass: walk mCode left-to-right with a parallel "constant-stack"
    // that records, for each value on the runtime stack, the position of the
    // PushLit that produced it (or -1 if non-constant). When an arithmetic op
    // sees only constants on its stack-top, we precompute the result and
    // splice the producing instructions away, replacing them with one PushLit.
    if (mCode.empty()) return;
    // Track each stack slot's producer: -1 = non-constant, else index in mCode of the PushLit.
    std::vector<int> producer;
    producer.reserve(32);
    // Helper: evaluate a 1-arg op on a constant value.
    auto eval1 = [](Op op, float a) -> float {
        switch (op) {
        case Op::Neg:   return -a;
        case Op::Abs:   return std::fabs(a);
        case Op::Inv:   return 1.0f / a;
        case Op::Sqrt:  return std::sqrt(a);
        case Op::Sin:   return std::sin(a);
        case Op::Cos:   return std::cos(a);
        case Op::Tan:   return std::tan(a);
        case Op::Asin:  return std::asin(a);
        case Op::Acos:  return std::acos(a);
        case Op::Atan:  return std::atan(a);
        case Op::Sinh:  return std::sinh(a);
        case Op::Cosh:  return std::cosh(a);
        case Op::Tanh:  return std::tanh(a);
        case Op::Asinh: return std::asinh(a);
        case Op::Acosh: return std::acosh(a);
        case Op::Atanh: return std::atanh(a);
        case Op::Exp:   return std::exp(a);
        case Op::Ln:    return std::log(a);
        case Op::Log:   return std::log10(a);
        case Op::Floor: return std::floor(a);
        case Op::Ceil:  return std::ceil(a);
        case Op::Pow2:  return a * a;
        case Op::Pow3:  return a * a * a;
        case Op::Sign:  return (a > 0.0f) ? 1.0f : ((a < 0.0f) ? -1.0f : 0.0f);
        case Op::Round: return std::round(a);
        case Op::Trunc: return std::trunc(a);
        case Op::Not:   return (a != 0.0f) ? 0.0f : 1.0f;
        default:        return std::numeric_limits<float>::quiet_NaN();// not a foldable unary
        }
    };
    auto eval2 = [](Op op, float a, float b) -> float {
        switch (op) {
        case Op::Add:   return a + b;
        case Op::Sub:   return a - b;
        case Op::Mul:   return a * b;
        case Op::Div:   return a / b;
        case Op::Mod:   return std::fmod(a, b);
        case Op::Pow:   return std::pow(a, b);
        case Op::Min:   return std::fmin(a, b);
        case Op::Max:   return std::fmax(a, b);
        case Op::Atan2: return std::atan2(a, b);
        case Op::Hypot: return std::hypot(a, b);
        case Op::Step:  return (b < a) ? 0.0f : 1.0f;
        case Op::Lt:    return (a <  b) ? 1.0f : 0.0f;
        case Op::Gt:    return (a >  b) ? 1.0f : 0.0f;
        case Op::Le:    return (a <= b) ? 1.0f : 0.0f;
        case Op::Ge:    return (a >= b) ? 1.0f : 0.0f;
        case Op::Eq:    return (a == b) ? 1.0f : 0.0f;
        case Op::Ne:    return (a != b) ? 1.0f : 0.0f;
        case Op::And:   return ((a != 0.0f) && (b != 0.0f)) ? 1.0f : 0.0f;
        case Op::Or:    return ((a != 0.0f) || (b != 0.0f)) ? 1.0f : 0.0f;
        default:        return std::numeric_limits<float>::quiet_NaN();
        }
    };

    auto isFoldableUnary = [](Op op) {
        switch (op) {
        case Op::Neg: case Op::Abs: case Op::Inv: case Op::Sqrt:
        case Op::Sin: case Op::Cos: case Op::Tan:
        case Op::Asin: case Op::Acos: case Op::Atan:
        case Op::Sinh: case Op::Cosh: case Op::Tanh:
        case Op::Asinh: case Op::Acosh: case Op::Atanh:
        case Op::Exp: case Op::Ln: case Op::Log:
        case Op::Floor: case Op::Ceil: case Op::Pow2: case Op::Pow3:
        case Op::Sign: case Op::Round: case Op::Trunc: case Op::Not:
            return true;
        default: return false;
        }
    };
    auto isFoldableBinary = [](Op op) {
        switch (op) {
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        case Op::Mod: case Op::Pow:
        case Op::Min: case Op::Max: case Op::Atan2: case Op::Hypot: case Op::Step:
        case Op::Lt: case Op::Gt: case Op::Le: case Op::Ge:
        case Op::Eq: case Op::Ne: case Op::And: case Op::Or:
            return true;
        default: return false;
        }
    };

    // Rebuild mCode by emitting either the original instruction, or a folded
    // PushLit when the top of the constant-stack admits it. We track producer
    // indices into the OUTPUT code (newCode) so we can rewrite tail entries.
    std::vector<Instr> newCode;
    newCode.reserve(mCode.size());

    for (size_t k = 0; k < mCode.size(); ++k) {
        const Instr in = mCode[k];
        if (in.op == Op::PushLit) {
            newCode.push_back(in);
            producer.push_back(static_cast<int>(newCode.size()) - 1);
            continue;
        }
        if (in.op == Op::PushVar || in.op == Op::PushSlot) {
            newCode.push_back(in);
            producer.push_back(-1);
            continue;
        }
        if (in.op == Op::Store) {
            // Pops one. Don't fold across stores; just emit.
            newCode.push_back(in);
            if (!producer.empty()) producer.pop_back();
            continue;
        }
        if (isFoldableUnary(in.op)) {
            if (!producer.empty() && producer.back() >= 0) {
                // Fold: pop the trailing PushLit, push a new PushLit with eval1.
                const int litIdx = producer.back();
                const float a = mConstants[newCode[litIdx].arg];
                const float v = eval1(in.op, a);
                // Replace newCode[litIdx] with a PushLit pointing at a new constant
                // (we append to mConstants; the old constant becomes dead but
                // harmless). Drop the unary op (don't append).
                const uint16_t cidx = static_cast<uint16_t>(mConstants.size());
                mConstants.push_back(v);
                newCode[litIdx] = {Op::PushLit, cidx};
                // producer stack unchanged (still 1 constant on top).
                continue;
            }
            newCode.push_back(in);
            // producer top stays the same (1-arg ops don't change stack size).
            // But the result is non-constant now.
            if (!producer.empty()) producer.back() = -1;
            continue;
        }
        if (isFoldableBinary(in.op)) {
            const size_t N = producer.size();
            if (N >= 2 && producer[N-1] >= 0 && producer[N-2] >= 0) {
                const int litA = producer[N-2];
                const int litB = producer[N-1];
                const float a = mConstants[newCode[litA].arg];
                const float b = mConstants[newCode[litB].arg];
                const float v = eval2(in.op, a, b);
                // Replace litA with the folded literal; erase litB. Don't
                // append the binary op.
                const uint16_t cidx = static_cast<uint16_t>(mConstants.size());
                mConstants.push_back(v);
                newCode[litA] = {Op::PushLit, cidx};
                newCode.erase(newCode.begin() + litB);
                producer.pop_back();
                producer.back() = litA;// the folded literal's index is still litA in newCode
                continue;
            }
            newCode.push_back(in);
            if (producer.size() >= 1) producer.pop_back();
            if (!producer.empty()) producer.back() = -1;
            continue;
        }
        // 3-arg, Switch, IfThenElse, etc: don't fold (safe but more work).
        // Update producer stack to reflect the net pop/push.
        newCode.push_back(in);
        switch (in.op) {
        case Op::Clamp: case Op::Lerp: case Op::Smoothstep: case Op::IfThenElse:
            if (producer.size() >= 3) { producer.pop_back(); producer.pop_back(); }
            if (!producer.empty()) producer.back() = -1;
            break;
        case Op::Switch: {
            const int n = static_cast<int>(in.arg);
            const int pops = 2 * n + 2;
            for (int j = 0; j < pops - 1 && !producer.empty(); ++j) producer.pop_back();
            if (!producer.empty()) producer.back() = -1;
            break;
        }
        default:
            break;
        }
    }
    mCode = std::move(newCode);
}

inline void Calculator::verify()
{
    // Walk the bytecode with a virtual stack-depth counter so eval() doesn't
    // have to check anything at runtime. Any well-formed expression must leave
    // exactly one value on the stack; anything else is a compile-time error.
    // Use index-based traversal (matching eval()) so we can correctly follow
    // forward Jumps emitted by lazy if()/switch(). Strategy:
    //   - Unconditional Jump: follow it (ip = arg) — the not-taken branch
    //     converges at the target by construction.
    //   - JumpIfFalse: pop the condition, fall through (verifies the "then"
    //     path). The "else" path is trusted; the parser emits it with the
    //     same compile pass as "then" so structural arity is guaranteed.
    int depth = 0;
    constexpr int kStackMax = 32;
    const size_t codeSize = mCode.size();
    for (size_t ip = 0; ip < codeSize; ) {
        const Instr &i = mCode[ip];
        switch (i.op) {
        case Op::Jump:
            // arg == codeSize is valid: "jump to end" (loop terminates).
            if (i.arg > codeSize) throw std::invalid_argument("Calculator: jump target out of range");
            ip = i.arg;
            continue;
        case Op::JumpIfFalse:
            if (depth < 1) throw std::invalid_argument("Calculator: stack underflow on conditional jump");
            if (i.arg > codeSize) throw std::invalid_argument("Calculator: conditional jump target out of range");
            --depth;
            break;
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
        case Op::Sinh: case Op::Cosh: case Op::Tanh:
        case Op::Asinh: case Op::Acosh: case Op::Atanh:
        case Op::Exp: case Op::Ln:  case Op::Log:
        case Op::Floor: case Op::Ceil:
        case Op::Pow2:  case Op::Pow3:
        case Op::Sign: case Op::Round: case Op::Trunc:
        case Op::Not:
            if (depth < 1) throw std::invalid_argument("Calculator: stack underflow on unary op");
            break;
        case Op::Add: case Op::Sub: case Op::Mul: case Op::Div:
        case Op::Mod: case Op::Pow:
        case Op::Min: case Op::Max: case Op::Atan2: case Op::Hypot: case Op::Step:
        case Op::Lt: case Op::Gt: case Op::Le: case Op::Ge:
        case Op::Eq: case Op::Ne:
        case Op::And: case Op::Or:
            if (depth < 2) throw std::invalid_argument("Calculator: stack underflow on binary op");
            --depth;
            break;
        case Op::Clamp: case Op::Lerp: case Op::Smoothstep: case Op::IfThenElse:
            if (depth < 3) throw std::invalid_argument("Calculator: stack underflow on ternary op");
            depth -= 2;
            break;
        case Op::Switch: {
            const int n = static_cast<int>(i.arg);// number of case pairs
            const int popN = 2 * n + 2;            // selector + N*(key,val) + default
            if (depth < popN)
                throw std::invalid_argument("Calculator: stack underflow on switch");
            depth -= (popN - 1);
            break;
        }
        }
        if (depth > kStackMax)
            throw std::invalid_argument("Calculator: stack overflow (depth > 32)");
        ++ip;// Jump/JumpIfFalse handle their own advancement via `continue`.
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
    const size_t codeSize = mCode.size();
    for (size_t ip = 0; ip < codeSize; ) {
        const Instr &i = mCode[ip];
        switch (i.op) {
        case Op::Jump:        ip = i.arg; continue;
        case Op::JumpIfFalse: if (s[--sp] == 0.0f) { ip = i.arg; continue; } break;
        case Op::PushLit:  s[sp++] = mConstants[i.arg]; break;
        case Op::PushVar:  s[sp++] = values[i.arg];     break;
        case Op::PushSlot: s[sp++] = slots[i.arg];      break;
        case Op::Store:    --sp; slots[i.arg] = s[sp];  break;
        case Op::Add:  --sp; s[sp-1] += s[sp];                    break;
        case Op::Sub:  --sp; s[sp-1] -= s[sp];                    break;
        case Op::Mul:  --sp; s[sp-1] *= s[sp];                    break;
        case Op::Div:  --sp; s[sp-1] /= s[sp];                    break;
        case Op::Mod:  --sp; s[sp-1] = std::fmod(s[sp-1], s[sp]); break;
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
        case Op::Sinh:       s[sp-1] = std::sinh(s[sp-1]);        break;
        case Op::Cosh:       s[sp-1] = std::cosh(s[sp-1]);        break;
        case Op::Tanh:       s[sp-1] = std::tanh(s[sp-1]);        break;
        case Op::Asinh:      s[sp-1] = std::asinh(s[sp-1]);       break;
        case Op::Acosh:      s[sp-1] = std::acosh(s[sp-1]);       break;
        case Op::Atanh:      s[sp-1] = std::atanh(s[sp-1]);       break;
        case Op::Exp:        s[sp-1] = std::exp(s[sp-1]);         break;
        case Op::Ln:         s[sp-1] = std::log(s[sp-1]);         break;
        case Op::Log:        s[sp-1] = std::log10(s[sp-1]);       break;
        case Op::Floor:      s[sp-1] = std::floor(s[sp-1]);       break;
        case Op::Ceil:       s[sp-1] = std::ceil(s[sp-1]);        break;
        case Op::Pow2:       s[sp-1] *= s[sp-1];                  break;
        case Op::Pow3:     { float v = s[sp-1]; s[sp-1] = v*v*v; }break;
        case Op::Sign:     { float v = s[sp-1]; s[sp-1] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : 0.0f); } break;
        case Op::Round:      s[sp-1] = std::round(s[sp-1]);       break;
        case Op::Trunc:      s[sp-1] = std::trunc(s[sp-1]);       break;
        case Op::Not:        s[sp-1] = (s[sp-1] != 0.0f) ? 0.0f : 1.0f; break;
        case Op::Min:    --sp; s[sp-1] = std::fmin(s[sp-1], s[sp]);  break;
        case Op::Max:    --sp; s[sp-1] = std::fmax(s[sp-1], s[sp]);  break;
        case Op::Atan2:  --sp; s[sp-1] = std::atan2(s[sp-1], s[sp]); break;
        case Op::Hypot:  --sp; s[sp-1] = std::hypot(s[sp-1], s[sp]); break;
        case Op::Step:   --sp; s[sp-1] = (s[sp] < s[sp-1]) ? 0.0f : 1.0f; break;// step(edge, x): x < edge ? 0 : 1
        case Op::Lt:     --sp; s[sp-1] = (s[sp-1] <  s[sp]) ? 1.0f : 0.0f; break;
        case Op::Gt:     --sp; s[sp-1] = (s[sp-1] >  s[sp]) ? 1.0f : 0.0f; break;
        case Op::Le:     --sp; s[sp-1] = (s[sp-1] <= s[sp]) ? 1.0f : 0.0f; break;
        case Op::Ge:     --sp; s[sp-1] = (s[sp-1] >= s[sp]) ? 1.0f : 0.0f; break;
        case Op::Eq:     --sp; s[sp-1] = (s[sp-1] == s[sp]) ? 1.0f : 0.0f; break;
        case Op::Ne:     --sp; s[sp-1] = (s[sp-1] != s[sp]) ? 1.0f : 0.0f; break;
        case Op::And:    --sp; s[sp-1] = ((s[sp-1] != 0.0f) && (s[sp] != 0.0f)) ? 1.0f : 0.0f; break;
        case Op::Or:     --sp; s[sp-1] = ((s[sp-1] != 0.0f) || (s[sp] != 0.0f)) ? 1.0f : 0.0f; break;
        // 3-arg: stack has [a, b, c] with c on top; pops 3, pushes 1.
        case Op::Clamp: {// clamp(x, lo, hi)
            const float hi = s[--sp]; const float lo = s[--sp]; float &x = s[sp-1];
            x = (x < lo) ? lo : ((x > hi) ? hi : x);
        } break;
        case Op::Lerp: {// lerp(a, b, t) -> a*(1-t) + b*t
            const float t = s[--sp]; const float b = s[--sp]; float &a = s[sp-1];
            a = a + (b - a) * t;
        } break;
        case Op::Smoothstep: {// smoothstep(e0, e1, x)
            const float x = s[--sp]; const float e1 = s[--sp]; float &e0 = s[sp-1];
            float u = (x - e0) / (e1 - e0);
            u = (u < 0.0f) ? 0.0f : ((u > 1.0f) ? 1.0f : u);
            e0 = u * u * (3.0f - 2.0f * u);
        } break;
        case Op::IfThenElse: {// if(c, t, e) -> c != 0 ? t : e
            const float e = s[--sp]; const float t = s[--sp]; float &c = s[sp-1];
            c = (c != 0.0f) ? t : e;
        } break;
        case Op::Switch: {// switch(selector, k1, v1, ..., kN, vN, default)
            // Stack layout (bottom-up): selector, k1, v1, ..., kN, vN, default.
            // All branches are eagerly evaluated; we just pick one.
            const int n     = static_cast<int>(i.arg);
            const int total = 2 * n + 2;
            const int base  = sp - total;
            const float selector   = s[base];
            const float defaultVal = s[sp - 1];
            float result = defaultVal;
            for (int k = 0; k < n; ++k) {
                if (s[base + 1 + 2*k] == selector) {
                    result = s[base + 2 + 2*k];
                    break;
                }
            }
            sp = base + 1;
            s[base] = result;
        } break;
        }
        ++ip;// fall-through path advances; explicit Jump/JumpIfFalse used `continue`
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

inline float Calculator::eval() const
{
    if (!mVariables.empty()) {
        throw std::invalid_argument(
            "Calculator::eval(): expression references " + std::to_string(mVariables.size()) +
            " variable(s) (first: \"" + mVariables[0] + "\"); use one of the bound-arg "
            "overloads (eval(x), eval(values), eval(initializer_list), or eval(map))");
    }
    // No variables → no PushVar opcodes → evalImpl never dereferences values.
    // Pass a non-null dummy to be safe.
    float dummy = 0.0f;
    return this->eval(&dummy);
}

inline void Calculator::eval_n(const float* in, float* out, size_t n,
                               const std::string &varName) const
{
    // Validate the binding shape once, outside the hot loop. Three accepted
    // shapes: no variables (broadcast a constant), or exactly one variable
    // matching @a varName, or — for backward compat — no variables and a
    // null @a in pointer is allowed (caller can pass nullptr).
    if (mVariables.empty()) {
        // Broadcast: every output is the same constant value.
        float dummy = 0.0f;
        const float v = this->eval(&dummy);
        for (size_t i = 0; i < n; ++i) out[i] = v;
        return;
    }
    if (mVariables.size() != 1 || mVariables[0] != varName) {
        throw std::invalid_argument(
            "Calculator::eval_n: expression must reference only the variable \"" +
            varName + "\" (got " + std::to_string(mVariables.size()) + " variable(s)" +
            (mVariables.empty() ? "" : ", first: \"" + mVariables[0] + "\"") + ")");
    }
    // Hot loop. Pass each input through evalImpl directly with a 1-slot buffer.
    for (size_t i = 0; i < n; ++i) {
        out[i] = this->eval(&in[i]);
    }
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

inline std::string Calculator::disassemble() const
{
    auto opName = [](Op op) -> const char* {
        switch (op) {
        case Op::PushLit:    return "PushLit";
        case Op::PushVar:    return "PushVar";
        case Op::PushSlot:   return "PushSlot";
        case Op::Store:      return "Store";
        case Op::Add:        return "Add";
        case Op::Sub:        return "Sub";
        case Op::Mul:        return "Mul";
        case Op::Div:        return "Div";
        case Op::Mod:        return "Mod";
        case Op::Pow:        return "Pow";
        case Op::Neg:        return "Neg";
        case Op::Abs:        return "Abs";
        case Op::Inv:        return "Inv";
        case Op::Sqrt:       return "Sqrt";
        case Op::Sin:        return "Sin";
        case Op::Cos:        return "Cos";
        case Op::Tan:        return "Tan";
        case Op::Asin:       return "Asin";
        case Op::Acos:       return "Acos";
        case Op::Atan:       return "Atan";
        case Op::Sinh:       return "Sinh";
        case Op::Cosh:       return "Cosh";
        case Op::Tanh:       return "Tanh";
        case Op::Asinh:      return "Asinh";
        case Op::Acosh:      return "Acosh";
        case Op::Atanh:      return "Atanh";
        case Op::Exp:        return "Exp";
        case Op::Ln:         return "Ln";
        case Op::Log:        return "Log";
        case Op::Floor:      return "Floor";
        case Op::Ceil:       return "Ceil";
        case Op::Pow2:       return "Pow2";
        case Op::Pow3:       return "Pow3";
        case Op::Sign:       return "Sign";
        case Op::Round:      return "Round";
        case Op::Trunc:      return "Trunc";
        case Op::Not:        return "Not";
        case Op::Min:        return "Min";
        case Op::Max:        return "Max";
        case Op::Atan2:      return "Atan2";
        case Op::Hypot:      return "Hypot";
        case Op::Step:       return "Step";
        case Op::Lt:         return "Lt";
        case Op::Gt:         return "Gt";
        case Op::Le:         return "Le";
        case Op::Ge:         return "Ge";
        case Op::Eq:         return "Eq";
        case Op::Ne:         return "Ne";
        case Op::And:        return "And";
        case Op::Or:         return "Or";
        case Op::Clamp:      return "Clamp";
        case Op::Lerp:       return "Lerp";
        case Op::Smoothstep: return "Smoothstep";
        case Op::IfThenElse: return "IfThenElse";
        case Op::Switch:     return "Switch";
        }
        return "???";
    };
    std::ostringstream out;
    for (size_t i = 0; i < mCode.size(); ++i) {
        const Instr &in = mCode[i];
        out.width(4); out << i << "  " << opName(in.op);
        switch (in.op) {
        case Op::PushLit:
            out << "  " << mConstants[in.arg] << "  (#const " << in.arg << ")";
            break;
        case Op::PushVar:
            if (in.arg < mVariables.size()) out << "  " << mVariables[in.arg] << "  (#var " << in.arg << ")";
            break;
        case Op::PushSlot: case Op::Store:
            if (in.arg < mSlotNames.size()) out << "  " << mSlotNames[in.arg] << "  (#slot " << in.arg << ")";
            break;
        case Op::Switch:
            out << "  cases=" << in.arg;
            break;
        default:
            break;
        }
        out << "\n";
    }
    if (!mVariables.empty()) {
        out << "Variables: ";
        for (size_t i = 0; i < mVariables.size(); ++i)
            out << (i ? ", " : "") << mVariables[i];
        out << "\n";
    }
    if (!mSlotNames.empty()) {
        out << "Slots: ";
        for (size_t i = 0; i < mSlotNames.size(); ++i)
            out << (i ? ", " : "") << mSlotNames[i];
        out << "\n";
    }
    if (!mResultName.empty()) out << "Result name: " << mResultName << "\n";
    return out.str();
}

} // namespace vdb_tool
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // VDB_TOOL_CALCULATOR_HAS_BEEN_INCLUDED
