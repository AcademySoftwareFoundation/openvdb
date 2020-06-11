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

/// @file ast/Tokens.h
///
/// @authors Nick Avramoussis
///
/// @brief  Various function and operator tokens used throughout the
///   AST and code generation
///

#ifndef OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED
#define OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED

#include <openvdb/Types.h>
#include <openvdb_ax/version.h>
#include <openvdb_ax/Exceptions.h>
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
    SHORT,
    INT,
    LONG,
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
    else if (type == "bool")   return BOOL;
    else if (type == "char")   return CHAR;
    else if (type == "short")  return SHORT;
    else if (type == "int")    return INT;
    else if (type == "long")   return LONG;
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
    else if (type[0] == 'i') {
        if (type == "int16") return SHORT;
        if (type == "int32") return INT;
        if (type == "int64") return LONG;
    }

    return UNKNOWN;
}

inline std::string typeStringFromToken(const CoreType type)
{
    switch (type) {
        case BOOL    : return "bool";
        case CHAR    : return "char";
        case SHORT   : return "short";
        case INT     : return "int";
        case LONG    : return "long";
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

template<typename T> CoreType tokenFromType()                    { return UNKNOWN; }
template<> inline CoreType tokenFromType<bool>()                 { return BOOL; }
template<> inline CoreType tokenFromType<char>()                 { return CHAR; }
template<> inline CoreType tokenFromType<int16_t>()              { return SHORT; }
template<> inline CoreType tokenFromType<int32_t>()              { return INT; }
template<> inline CoreType tokenFromType<int64_t>()              { return LONG; }
template<> inline CoreType tokenFromType<float>()                { return FLOAT; }
template<> inline CoreType tokenFromType<double>()               { return DOUBLE; }
template<> inline CoreType tokenFromType<math::Vec2<int32_t>>()  { return VEC2I; }
template<> inline CoreType tokenFromType<math::Vec2<float>>()    { return VEC2F; }
template<> inline CoreType tokenFromType<math::Vec2<double>>()   { return VEC2D; }
template<> inline CoreType tokenFromType<math::Vec3<int32_t>>()  { return VEC3I; }
template<> inline CoreType tokenFromType<math::Vec3<float>>()    { return VEC3F; }
template<> inline CoreType tokenFromType<math::Vec3<double>>()   { return VEC3D; }
template<> inline CoreType tokenFromType<math::Vec4<int32_t>>()  { return VEC4I; }
template<> inline CoreType tokenFromType<math::Vec4<float>>()    { return VEC4F; }
template<> inline CoreType tokenFromType<math::Vec4<double>>()   { return VEC4D; }
template<> inline CoreType tokenFromType<math::Mat3<float>>()    { return MAT3F; }
template<> inline CoreType tokenFromType<math::Mat3<double>>()   { return MAT3D; }
template<> inline CoreType tokenFromType<math::Mat4<float>>()    { return MAT4F; }
template<> inline CoreType tokenFromType<math::Mat4<double>>()   { return MAT4D; }
template<> inline CoreType tokenFromType<std::string>()          { return STRING; }

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
    OPENVDB_THROW(AXSyntaxError, "Unsupported op \"" + name + "\"");
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
            OPENVDB_THROW(AXSyntaxError, "Unsupported op");
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
            OPENVDB_THROW(AXSyntaxError, "Unsupported loop");
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
            OPENVDB_THROW(AXSyntaxError, "Unsupported keyword");
    }
}


}

}
}
}
}

#endif // OPENVDB_AX_AST_TOKENS_HAS_BEEN_INCLUDED

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
