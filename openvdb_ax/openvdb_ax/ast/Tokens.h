// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file ast/Tokens.h
///
/// @authors Nick Avramoussis
///
/// @brief  Various function and operator tokens used throughout the
///   AST and code generation
///

#ifndef OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED

#include "../Exceptions.h"

#include <openvdb/version.h>
#include <openvdb/Types.h>

#include <stdexcept>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace ast {

namespace tokens {

enum CoreType
{
    BOOL = 0,
    CHAR,
    INT16,
    INT32,
    INT64,
    FLOAT,
    DOUBLE,
    //
    VEC2I,
    VEC2F,
    VEC2D,
    //
    VEC3I,
    VEC3F,
    VEC3D,
    //
    VEC4I,
    VEC4F,
    VEC4D,
    //
    MAT3F,
    MAT3D,
    //
    MAT4F,
    MAT4D,
    //
    QUATF,
    QUATD,
    //
    STRING,
    UNKNOWN
};

inline CoreType tokenFromTypeString(const std::string& type)
{
    if (type[0] == 'v') {
        if (type == "vec2i") return VEC2I;
        if (type == "vec2f") return VEC2F;
        if (type == "vec2d") return VEC2D;
        if (type == "vec3i") return VEC3I;
        if (type == "vec3f") return VEC3F;
        if (type == "vec3d") return VEC3D;
        if (type == "vec4i") return VEC4I;
        if (type == "vec4f") return VEC4F;
        if (type == "vec4d") return VEC4D;
    }
    else if (type[0] == 'm') {
        if (type == "mat3f") return MAT3F;
        if (type == "mat3d") return MAT3D;
        if (type == "mat4f") return MAT4F;
        if (type == "mat4d") return MAT4D;
    }
    else if (type[0] == 'q') {
        if (type == "quatf") return QUATF;
        if (type == "quatd") return QUATD;
    }
    else if (type[0] == 'i') {
        if (type == "int16") return INT16;
        if (type == "int")   return INT32;
        if (type == "int32") return INT32;
        if (type == "int64") return INT64;
    }
    else if (type == "bool")   return BOOL;
    else if (type == "char")   return CHAR;
    else if (type == "float")  return FLOAT;
    else if (type == "double") return DOUBLE;
    else if (type == "string") return STRING;

    // also handle vdb types that have different type strings to our tokens
    // @todo  These should probably be separated out. The executables currently
    //   use this function to guarantee conversion
    if (type[0] == 'v') {
        if (type == "vec2s") return VEC2F;
        if (type == "vec3s") return VEC3F;
        if (type == "vec4s") return VEC4F;
    }
    else if (type[0] == 'm') {
        if (type == "mat3s") return MAT3F;
        if (type == "mat4s") return MAT4F;
    }
    else if (type == "quats") return QUATF;

    return UNKNOWN;
}

inline std::string typeStringFromToken(const CoreType type)
{
    switch (type) {
        case BOOL    : return "bool";
        case CHAR    : return "char";
        case INT16   : return "int16";
        case INT32   : return "int32";
        case INT64   : return "int64";
        case FLOAT   : return "float";
        case DOUBLE  : return "double";
        case VEC2I   : return "vec2i";
        case VEC2F   : return "vec2f";
        case VEC2D   : return "vec2d";
        case VEC3I   : return "vec3i";
        case VEC3F   : return "vec3f";
        case VEC3D   : return "vec3d";
        case VEC4I   : return "vec4i";
        case VEC4F   : return "vec4f";
        case VEC4D   : return "vec4d";
        case MAT3F   : return "mat3f";
        case MAT3D   : return "mat3d";
        case MAT4F   : return "mat4f";
        case MAT4D   : return "mat4d";
        case QUATF   : return "quatf";
        case QUATD   : return "quatd";
        case STRING  : return "string";
        case UNKNOWN :
        default      :
            return "unknown";
    }
}

enum OperatorToken
{
    ////////////////////////////////////////////////////////////////
    ///  ARITHMETIC
    ////////////////////////////////////////////////////////////////
    PLUS = 0,
    MINUS,
    MULTIPLY,
    DIVIDE,
    MODULO,
    ////////////////////////////////////////////////////////////////
    ///  LOGICAL
    ////////////////////////////////////////////////////////////////
    AND,
    OR,
    NOT,
    ////////////////////////////////////////////////////////////////
    ///  RELATIONAL
    ////////////////////////////////////////////////////////////////
    EQUALSEQUALS,
    NOTEQUALS,
    MORETHAN,
    LESSTHAN,
    MORETHANOREQUAL,
    LESSTHANOREQUAL,
    ////////////////////////////////////////////////////////////////
    ///  BITWISE
    ////////////////////////////////////////////////////////////////
    SHIFTLEFT,
    SHIFTRIGHT,
    BITAND,
    BITOR,
    BITXOR,
    BITNOT,
    ////////////////////////////////////////////////////////////////
    ///  ASSIGNMENT
    ////////////////////////////////////////////////////////////////
    EQUALS,
    PLUSEQUALS,
    MINUSEQUALS,
    MULTIPLYEQUALS,
    DIVIDEEQUALS,
    MODULOEQUALS,
    SHIFTLEFTEQUALS,
    SHIFTRIGHTEQUALS,
    BITANDEQUALS,
    BITXOREQUALS,
    BITOREQUALS
};

enum OperatorType
{
    ARITHMETIC = 0,
    LOGICAL,
    RELATIONAL,
    BITWISE,
    ASSIGNMENT,
    UNKNOWN_OPERATOR
};

inline OperatorType operatorType(const OperatorToken token)
{
    const size_t idx = static_cast<size_t>(token);
    if (idx <= static_cast<size_t>(MODULO))          return ARITHMETIC;
    if (idx <= static_cast<size_t>(NOT))             return LOGICAL;
    if (idx <= static_cast<size_t>(LESSTHANOREQUAL)) return RELATIONAL;
    if (idx <= static_cast<size_t>(BITNOT))          return BITWISE;
    if (idx <= static_cast<size_t>(BITOREQUALS))     return ASSIGNMENT;
    return UNKNOWN_OPERATOR;
}

inline OperatorToken operatorTokenFromName(const std::string& name)
{
    if (name == "+")    return PLUS;
    if (name == "-")    return MINUS;
    if (name == "*")    return MULTIPLY;
    if (name == "/")    return DIVIDE;
    if (name == "%")    return MODULO;
    if (name == "&&")   return AND;
    if (name == "||")   return OR;
    if (name == "!")    return NOT;
    if (name == "==")   return EQUALSEQUALS;
    if (name == "!=")   return NOTEQUALS;
    if (name == ">")    return MORETHAN;
    if (name == "<")    return LESSTHAN;
    if (name == ">=")   return MORETHANOREQUAL;
    if (name == "<=")   return LESSTHANOREQUAL;
    if (name == "<<")   return SHIFTLEFT;
    if (name == ">>")   return SHIFTRIGHT;
    if (name == "&")    return BITAND;
    if (name == "|")    return BITOR;
    if (name == "^")    return BITXOR;
    if (name == "~")    return BITNOT;
    if (name == "=")    return EQUALS;
    if (name == "+=")   return PLUSEQUALS;
    if (name == "-=")   return MINUSEQUALS;
    if (name == "*=")   return MULTIPLYEQUALS;
    if (name == "/=")   return DIVIDEEQUALS;
    if (name == "%=")   return MODULOEQUALS;
    if (name == "<<=")  return SHIFTLEFTEQUALS;
    if (name == ">>=")  return SHIFTRIGHTEQUALS;
    if (name == "&=")   return BITANDEQUALS;
    if (name == "^=")   return BITXOREQUALS;
    if (name == "|=")   return BITOREQUALS;
    OPENVDB_THROW(AXTokenError, "Unsupported op \"" + name + "\"");
}

inline std::string operatorNameFromToken(const OperatorToken token)
{
    switch (token) {
        case PLUS              : return "+";
        case MINUS             : return "-";
        case MULTIPLY          : return "*";
        case DIVIDE            : return "/";
        case MODULO            : return "%";
        case AND               : return "&&";
        case OR                : return "||";
        case NOT               : return "!";
        case EQUALSEQUALS      : return "==";
        case NOTEQUALS         : return "!=";
        case MORETHAN          : return ">";
        case LESSTHAN          : return "<";
        case MORETHANOREQUAL   : return ">=";
        case LESSTHANOREQUAL   : return "<=";
        case SHIFTLEFT         : return "<<";
        case SHIFTRIGHT        : return ">>";
        case BITAND            : return "&";
        case BITOR             : return "|";
        case BITXOR            : return "^";
        case BITNOT            : return "~";
        case EQUALS            : return "=";
        case PLUSEQUALS        : return "+=";
        case MINUSEQUALS       : return "-=";
        case MULTIPLYEQUALS    : return "*=";
        case DIVIDEEQUALS      : return "/=";
        case MODULOEQUALS      : return "%=";
        case SHIFTLEFTEQUALS   : return "<<=";
        case SHIFTRIGHTEQUALS  : return ">>=";
        case BITANDEQUALS      : return "&=";
        case BITXOREQUALS      : return "^=";
        case BITOREQUALS       : return "|=";
        default                :
            OPENVDB_THROW(AXTokenError, "Unsupported op");
    }
}

enum LoopToken
{
    FOR = 0,
    DO,
    WHILE
};

inline std::string loopNameFromToken(const LoopToken loop)
{
    switch (loop) {
        case FOR      : return "for";
        case DO       : return "do";
        case WHILE    : return "while";
        default       :
            OPENVDB_THROW(AXTokenError, "Unsupported loop");
    }
}

enum KeywordToken
{
    RETURN = 0,
    BREAK,
    CONTINUE
};

inline std::string keywordNameFromToken(const KeywordToken keyw)
{
    switch (keyw) {
        case RETURN      : return "return";
        case BREAK       : return "break";
        case CONTINUE    : return "continue";
        default          :
            OPENVDB_THROW(AXTokenError, "Unsupported keyword");
    }
}


} // namespace tokens

} // namespace ast
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED

