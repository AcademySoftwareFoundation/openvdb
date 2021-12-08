/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

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

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_AX_AXPARSER_H_INCLUDED
# define YY_AX_AXPARSER_H_INCLUDED
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

/* Token kinds.  */
#ifndef AXTOKENTYPE
# define AXTOKENTYPE
  enum axtokentype
  {
    AXEMPTY = -2,
    AXEOF = 0,                     /* "end of file"  */
    AXerror = 256,                 /* error  */
    AXUNDEF = 257,                 /* "invalid token"  */
    TRUE = 258,                    /* TRUE  */
    FALSE = 259,                   /* FALSE  */
    SEMICOLON = 260,               /* SEMICOLON  */
    AT = 261,                      /* AT  */
    DOLLAR = 262,                  /* DOLLAR  */
    IF = 263,                      /* IF  */
    ELSE = 264,                    /* ELSE  */
    FOR = 265,                     /* FOR  */
    DO = 266,                      /* DO  */
    WHILE = 267,                   /* WHILE  */
    RETURN = 268,                  /* RETURN  */
    BREAK = 269,                   /* BREAK  */
    CONTINUE = 270,                /* CONTINUE  */
    LCURLY = 271,                  /* LCURLY  */
    RCURLY = 272,                  /* RCURLY  */
    LSQUARE = 273,                 /* LSQUARE  */
    RSQUARE = 274,                 /* RSQUARE  */
    STRING = 275,                  /* STRING  */
    DOUBLE = 276,                  /* DOUBLE  */
    FLOAT = 277,                   /* FLOAT  */
    INT32 = 278,                   /* INT32  */
    INT64 = 279,                   /* INT64  */
    BOOL = 280,                    /* BOOL  */
    VEC2I = 281,                   /* VEC2I  */
    VEC2F = 282,                   /* VEC2F  */
    VEC2D = 283,                   /* VEC2D  */
    VEC3I = 284,                   /* VEC3I  */
    VEC3F = 285,                   /* VEC3F  */
    VEC3D = 286,                   /* VEC3D  */
    VEC4I = 287,                   /* VEC4I  */
    VEC4F = 288,                   /* VEC4F  */
    VEC4D = 289,                   /* VEC4D  */
    F_AT = 290,                    /* F_AT  */
    I_AT = 291,                    /* I_AT  */
    V_AT = 292,                    /* V_AT  */
    S_AT = 293,                    /* S_AT  */
    I16_AT = 294,                  /* I16_AT  */
    MAT3F = 295,                   /* MAT3F  */
    MAT3D = 296,                   /* MAT3D  */
    MAT4F = 297,                   /* MAT4F  */
    MAT4D = 298,                   /* MAT4D  */
    M3F_AT = 299,                  /* M3F_AT  */
    M4F_AT = 300,                  /* M4F_AT  */
    F_DOLLAR = 301,                /* F_DOLLAR  */
    I_DOLLAR = 302,                /* I_DOLLAR  */
    V_DOLLAR = 303,                /* V_DOLLAR  */
    S_DOLLAR = 304,                /* S_DOLLAR  */
    DOT_X = 305,                   /* DOT_X  */
    DOT_Y = 306,                   /* DOT_Y  */
    DOT_Z = 307,                   /* DOT_Z  */
    L_INT32 = 308,                 /* L_INT32  */
    L_INT64 = 309,                 /* L_INT64  */
    L_FLOAT = 310,                 /* L_FLOAT  */
    L_DOUBLE = 311,                /* L_DOUBLE  */
    L_STRING = 312,                /* L_STRING  */
    IDENTIFIER = 313,              /* IDENTIFIER  */
    COMMA = 314,                   /* COMMA  */
    QUESTION = 315,                /* QUESTION  */
    COLON = 316,                   /* COLON  */
    EQUALS = 317,                  /* EQUALS  */
    PLUSEQUALS = 318,              /* PLUSEQUALS  */
    MINUSEQUALS = 319,             /* MINUSEQUALS  */
    MULTIPLYEQUALS = 320,          /* MULTIPLYEQUALS  */
    DIVIDEEQUALS = 321,            /* DIVIDEEQUALS  */
    MODULOEQUALS = 322,            /* MODULOEQUALS  */
    BITANDEQUALS = 323,            /* BITANDEQUALS  */
    BITXOREQUALS = 324,            /* BITXOREQUALS  */
    BITOREQUALS = 325,             /* BITOREQUALS  */
    SHIFTLEFTEQUALS = 326,         /* SHIFTLEFTEQUALS  */
    SHIFTRIGHTEQUALS = 327,        /* SHIFTRIGHTEQUALS  */
    OR = 328,                      /* OR  */
    AND = 329,                     /* AND  */
    BITOR = 330,                   /* BITOR  */
    BITXOR = 331,                  /* BITXOR  */
    BITAND = 332,                  /* BITAND  */
    EQUALSEQUALS = 333,            /* EQUALSEQUALS  */
    NOTEQUALS = 334,               /* NOTEQUALS  */
    MORETHAN = 335,                /* MORETHAN  */
    LESSTHAN = 336,                /* LESSTHAN  */
    MORETHANOREQUAL = 337,         /* MORETHANOREQUAL  */
    LESSTHANOREQUAL = 338,         /* LESSTHANOREQUAL  */
    SHIFTLEFT = 339,               /* SHIFTLEFT  */
    SHIFTRIGHT = 340,              /* SHIFTRIGHT  */
    PLUS = 341,                    /* PLUS  */
    MINUS = 342,                   /* MINUS  */
    MULTIPLY = 343,                /* MULTIPLY  */
    DIVIDE = 344,                  /* DIVIDE  */
    MODULO = 345,                  /* MODULO  */
    UMINUS = 346,                  /* UMINUS  */
    NOT = 347,                     /* NOT  */
    BITNOT = 348,                  /* BITNOT  */
    PLUSPLUS = 349,                /* PLUSPLUS  */
    MINUSMINUS = 350,              /* MINUSMINUS  */
    LPARENS = 351,                 /* LPARENS  */
    RPARENS = 352,                 /* RPARENS  */
    LOWER_THAN_ELSE = 353          /* LOWER_THAN_ELSE  */
  };
  typedef enum axtokentype axtoken_kind_t;
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


#endif /* !YY_AX_AXPARSER_H_INCLUDED  */
