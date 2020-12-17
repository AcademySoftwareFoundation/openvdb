/* A Bison parser, made by GNU Bison 3.0.5.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_AX_OPENVDB_AX_GRAMMAR_AXPARSER_H_INCLUDED
# define YY_AX_OPENVDB_AX_GRAMMAR_AXPARSER_H_INCLUDED
/* Debug traces.  */
#ifndef AXDEBUG
# if defined YYDEBUG
#if YYDEBUG
#   define AXDEBUG 1
#  else
#   define AXDEBUG 0
#  endif
# else /* ! defined YYDEBUG */
#  define AXDEBUG 0
# endif /* ! defined YYDEBUG */
#endif  /* ! defined AXDEBUG */
#if AXDEBUG
extern int axdebug;
#endif

/* Token type.  */
#ifndef AXTOKENTYPE
# define AXTOKENTYPE
  enum axtokentype
  {
    TRUE = 258,
    FALSE = 259,
    SEMICOLON = 260,
    AT = 261,
    DOLLAR = 262,
    IF = 263,
    ELSE = 264,
    FOR = 265,
    DO = 266,
    WHILE = 267,
    RETURN = 268,
    BREAK = 269,
    CONTINUE = 270,
    LCURLY = 271,
    RCURLY = 272,
    LSQUARE = 273,
    RSQUARE = 274,
    STRING = 275,
    DOUBLE = 276,
    FLOAT = 277,
    INT32 = 278,
    INT64 = 279,
    BOOL = 280,
    VEC2I = 281,
    VEC2F = 282,
    VEC2D = 283,
    VEC3I = 284,
    VEC3F = 285,
    VEC3D = 286,
    VEC4I = 287,
    VEC4F = 288,
    VEC4D = 289,
    F_AT = 290,
    I_AT = 291,
    V_AT = 292,
    S_AT = 293,
    I16_AT = 294,
    MAT3F = 295,
    MAT3D = 296,
    MAT4F = 297,
    MAT4D = 298,
    M3F_AT = 299,
    M4F_AT = 300,
    F_DOLLAR = 301,
    I_DOLLAR = 302,
    V_DOLLAR = 303,
    S_DOLLAR = 304,
    DOT_X = 305,
    DOT_Y = 306,
    DOT_Z = 307,
    L_INT32 = 308,
    L_INT64 = 309,
    L_FLOAT = 310,
    L_DOUBLE = 311,
    L_STRING = 312,
    IDENTIFIER = 313,
    COMMA = 314,
    QUESTION = 315,
    COLON = 316,
    EQUALS = 317,
    PLUSEQUALS = 318,
    MINUSEQUALS = 319,
    MULTIPLYEQUALS = 320,
    DIVIDEEQUALS = 321,
    MODULOEQUALS = 322,
    BITANDEQUALS = 323,
    BITXOREQUALS = 324,
    BITOREQUALS = 325,
    SHIFTLEFTEQUALS = 326,
    SHIFTRIGHTEQUALS = 327,
    OR = 328,
    AND = 329,
    BITOR = 330,
    BITXOR = 331,
    BITAND = 332,
    EQUALSEQUALS = 333,
    NOTEQUALS = 334,
    MORETHAN = 335,
    LESSTHAN = 336,
    MORETHANOREQUAL = 337,
    LESSTHANOREQUAL = 338,
    SHIFTLEFT = 339,
    SHIFTRIGHT = 340,
    PLUS = 341,
    MINUS = 342,
    MULTIPLY = 343,
    DIVIDE = 344,
    MODULO = 345,
    UMINUS = 346,
    NOT = 347,
    BITNOT = 348,
    PLUSPLUS = 349,
    MINUSMINUS = 350,
    LPARENS = 351,
    RPARENS = 352,
    LOWER_THAN_ELSE = 353
  };
#endif

/* Value type.  */
#if ! defined AXSTYPE && ! defined AXSTYPE_IS_DECLARED

union AXSTYPE
{


    /// @brief Temporary storage for comma separated expressions
    using ExpList = std::vector<openvdb::ax::ast::Expression*>;

    const char* string;
    uint64_t index;
    double flt;

    openvdb::ax::ast::Tree* tree;
    openvdb::ax::ast::ValueBase* value;
    openvdb::ax::ast::Statement* statement;
    openvdb::ax::ast::StatementList* statementlist;
    openvdb::ax::ast::Block* block;
    openvdb::ax::ast::Expression* expression;
    openvdb::ax::ast::FunctionCall* function;
    openvdb::ax::ast::ArrayPack* arraypack;
    openvdb::ax::ast::CommaOperator* comma;
    openvdb::ax::ast::Variable* variable;
    openvdb::ax::ast::ExternalVariable* external;
    openvdb::ax::ast::Attribute* attribute;
    openvdb::ax::ast::DeclareLocal* declare_local;
    openvdb::ax::ast::Local* local;
    ExpList* explist;


};

typedef union AXSTYPE AXSTYPE;
# define AXSTYPE_IS_TRIVIAL 1
# define AXSTYPE_IS_DECLARED 1
#endif

/* Location type.  */
#if ! defined AXLTYPE && ! defined AXLTYPE_IS_DECLARED
typedef struct AXLTYPE AXLTYPE;
struct AXLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
};
# define AXLTYPE_IS_DECLARED 1
# define AXLTYPE_IS_TRIVIAL 1
#endif


extern AXSTYPE axlval;
extern AXLTYPE axlloc;
int axparse (openvdb::ax::ast::Tree** tree);

#endif /* !YY_AX_OPENVDB_AX_GRAMMAR_AXPARSER_H_INCLUDED  */
