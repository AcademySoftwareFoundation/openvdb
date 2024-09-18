/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output, and Bison version.  */
#define YYBISON 30802

/* Bison version string.  */
#define YYBISON_VERSION "3.8.2"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* "%code top" blocks.  */

    #include "openvdb_ax/ast/AST.h"
    #include "openvdb_ax/ast/Parse.h"
    #include "openvdb_ax/ast/Tokens.h"
    #include "openvdb_ax/compiler/Logger.h"
    #include <openvdb/Platform.h> // for OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
    #include <vector>

    /// @note  Bypasses bison conversion warnings in yyparse
    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN

    extern int axlex();
    extern openvdb::ax::Logger* axlog;

    using namespace openvdb::ax::ast;
    using namespace openvdb::ax;

    void axerror(Tree** tree, const char* s);

    using ExpList = std::vector<openvdb::ax::ast::Expression*>;

/* Substitute the type names.  */
#define YYSTYPE         AXSTYPE
#define YYLTYPE         AXLTYPE
/* Substitute the variable and function names.  */
#define yyparse         axparse
#define yylex           axlex
#define yyerror         axerror
#define yydebug         axdebug
#define yynerrs         axnerrs
#define yylval          axlval
#define yychar          axchar
#define yylloc          axlloc


# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

#include "axparser.h"
/* Symbol kind.  */
enum yysymbol_kind_t
{
  YYSYMBOL_YYEMPTY = -2,
  YYSYMBOL_YYEOF = 0,                      /* "end of file"  */
  YYSYMBOL_YYerror = 1,                    /* error  */
  YYSYMBOL_YYUNDEF = 2,                    /* "invalid token"  */
  YYSYMBOL_TRUE = 3,                       /* TRUE  */
  YYSYMBOL_FALSE = 4,                      /* FALSE  */
  YYSYMBOL_SEMICOLON = 5,                  /* SEMICOLON  */
  YYSYMBOL_AT = 6,                         /* AT  */
  YYSYMBOL_DOLLAR = 7,                     /* DOLLAR  */
  YYSYMBOL_IF = 8,                         /* IF  */
  YYSYMBOL_ELSE = 9,                       /* ELSE  */
  YYSYMBOL_FOR = 10,                       /* FOR  */
  YYSYMBOL_DO = 11,                        /* DO  */
  YYSYMBOL_WHILE = 12,                     /* WHILE  */
  YYSYMBOL_RETURN = 13,                    /* RETURN  */
  YYSYMBOL_BREAK = 14,                     /* BREAK  */
  YYSYMBOL_CONTINUE = 15,                  /* CONTINUE  */
  YYSYMBOL_LCURLY = 16,                    /* LCURLY  */
  YYSYMBOL_RCURLY = 17,                    /* RCURLY  */
  YYSYMBOL_LSQUARE = 18,                   /* LSQUARE  */
  YYSYMBOL_RSQUARE = 19,                   /* RSQUARE  */
  YYSYMBOL_STRING = 20,                    /* STRING  */
  YYSYMBOL_DOUBLE = 21,                    /* DOUBLE  */
  YYSYMBOL_FLOAT = 22,                     /* FLOAT  */
  YYSYMBOL_INT32 = 23,                     /* INT32  */
  YYSYMBOL_INT64 = 24,                     /* INT64  */
  YYSYMBOL_BOOL = 25,                      /* BOOL  */
  YYSYMBOL_VEC2I = 26,                     /* VEC2I  */
  YYSYMBOL_VEC2F = 27,                     /* VEC2F  */
  YYSYMBOL_VEC2D = 28,                     /* VEC2D  */
  YYSYMBOL_VEC3I = 29,                     /* VEC3I  */
  YYSYMBOL_VEC3F = 30,                     /* VEC3F  */
  YYSYMBOL_VEC3D = 31,                     /* VEC3D  */
  YYSYMBOL_VEC4I = 32,                     /* VEC4I  */
  YYSYMBOL_VEC4F = 33,                     /* VEC4F  */
  YYSYMBOL_VEC4D = 34,                     /* VEC4D  */
  YYSYMBOL_F_AT = 35,                      /* F_AT  */
  YYSYMBOL_I_AT = 36,                      /* I_AT  */
  YYSYMBOL_V_AT = 37,                      /* V_AT  */
  YYSYMBOL_S_AT = 38,                      /* S_AT  */
  YYSYMBOL_I16_AT = 39,                    /* I16_AT  */
  YYSYMBOL_MAT3F = 40,                     /* MAT3F  */
  YYSYMBOL_MAT3D = 41,                     /* MAT3D  */
  YYSYMBOL_MAT4F = 42,                     /* MAT4F  */
  YYSYMBOL_MAT4D = 43,                     /* MAT4D  */
  YYSYMBOL_M3F_AT = 44,                    /* M3F_AT  */
  YYSYMBOL_M4F_AT = 45,                    /* M4F_AT  */
  YYSYMBOL_F_DOLLAR = 46,                  /* F_DOLLAR  */
  YYSYMBOL_I_DOLLAR = 47,                  /* I_DOLLAR  */
  YYSYMBOL_V_DOLLAR = 48,                  /* V_DOLLAR  */
  YYSYMBOL_S_DOLLAR = 49,                  /* S_DOLLAR  */
  YYSYMBOL_DOT_X = 50,                     /* DOT_X  */
  YYSYMBOL_DOT_Y = 51,                     /* DOT_Y  */
  YYSYMBOL_DOT_Z = 52,                     /* DOT_Z  */
  YYSYMBOL_L_INT32 = 53,                   /* L_INT32  */
  YYSYMBOL_L_INT64 = 54,                   /* L_INT64  */
  YYSYMBOL_L_FLOAT = 55,                   /* L_FLOAT  */
  YYSYMBOL_L_DOUBLE = 56,                  /* L_DOUBLE  */
  YYSYMBOL_L_STRING = 57,                  /* L_STRING  */
  YYSYMBOL_IDENTIFIER = 58,                /* IDENTIFIER  */
  YYSYMBOL_COMMA = 59,                     /* COMMA  */
  YYSYMBOL_QUESTION = 60,                  /* QUESTION  */
  YYSYMBOL_COLON = 61,                     /* COLON  */
  YYSYMBOL_EQUALS = 62,                    /* EQUALS  */
  YYSYMBOL_PLUSEQUALS = 63,                /* PLUSEQUALS  */
  YYSYMBOL_MINUSEQUALS = 64,               /* MINUSEQUALS  */
  YYSYMBOL_MULTIPLYEQUALS = 65,            /* MULTIPLYEQUALS  */
  YYSYMBOL_DIVIDEEQUALS = 66,              /* DIVIDEEQUALS  */
  YYSYMBOL_MODULOEQUALS = 67,              /* MODULOEQUALS  */
  YYSYMBOL_BITANDEQUALS = 68,              /* BITANDEQUALS  */
  YYSYMBOL_BITXOREQUALS = 69,              /* BITXOREQUALS  */
  YYSYMBOL_BITOREQUALS = 70,               /* BITOREQUALS  */
  YYSYMBOL_SHIFTLEFTEQUALS = 71,           /* SHIFTLEFTEQUALS  */
  YYSYMBOL_SHIFTRIGHTEQUALS = 72,          /* SHIFTRIGHTEQUALS  */
  YYSYMBOL_OR = 73,                        /* OR  */
  YYSYMBOL_AND = 74,                       /* AND  */
  YYSYMBOL_BITOR = 75,                     /* BITOR  */
  YYSYMBOL_BITXOR = 76,                    /* BITXOR  */
  YYSYMBOL_BITAND = 77,                    /* BITAND  */
  YYSYMBOL_EQUALSEQUALS = 78,              /* EQUALSEQUALS  */
  YYSYMBOL_NOTEQUALS = 79,                 /* NOTEQUALS  */
  YYSYMBOL_MORETHAN = 80,                  /* MORETHAN  */
  YYSYMBOL_LESSTHAN = 81,                  /* LESSTHAN  */
  YYSYMBOL_MORETHANOREQUAL = 82,           /* MORETHANOREQUAL  */
  YYSYMBOL_LESSTHANOREQUAL = 83,           /* LESSTHANOREQUAL  */
  YYSYMBOL_SHIFTLEFT = 84,                 /* SHIFTLEFT  */
  YYSYMBOL_SHIFTRIGHT = 85,                /* SHIFTRIGHT  */
  YYSYMBOL_PLUS = 86,                      /* PLUS  */
  YYSYMBOL_MINUS = 87,                     /* MINUS  */
  YYSYMBOL_MULTIPLY = 88,                  /* MULTIPLY  */
  YYSYMBOL_DIVIDE = 89,                    /* DIVIDE  */
  YYSYMBOL_MODULO = 90,                    /* MODULO  */
  YYSYMBOL_UMINUS = 91,                    /* UMINUS  */
  YYSYMBOL_NOT = 92,                       /* NOT  */
  YYSYMBOL_BITNOT = 93,                    /* BITNOT  */
  YYSYMBOL_PLUSPLUS = 94,                  /* PLUSPLUS  */
  YYSYMBOL_MINUSMINUS = 95,                /* MINUSMINUS  */
  YYSYMBOL_LPARENS = 96,                   /* LPARENS  */
  YYSYMBOL_RPARENS = 97,                   /* RPARENS  */
  YYSYMBOL_LOWER_THAN_ELSE = 98,           /* LOWER_THAN_ELSE  */
  YYSYMBOL_YYACCEPT = 99,                  /* $accept  */
  YYSYMBOL_tree = 100,                     /* tree  */
  YYSYMBOL_body = 101,                     /* body  */
  YYSYMBOL_block = 102,                    /* block  */
  YYSYMBOL_statement = 103,                /* statement  */
  YYSYMBOL_expressions = 104,              /* expressions  */
  YYSYMBOL_comma_operator = 105,           /* comma_operator  */
  YYSYMBOL_expression = 106,               /* expression  */
  YYSYMBOL_declaration = 107,              /* declaration  */
  YYSYMBOL_declaration_list = 108,         /* declaration_list  */
  YYSYMBOL_declarations = 109,             /* declarations  */
  YYSYMBOL_block_or_statement = 110,       /* block_or_statement  */
  YYSYMBOL_conditional_statement = 111,    /* conditional_statement  */
  YYSYMBOL_loop_condition = 112,           /* loop_condition  */
  YYSYMBOL_loop_condition_optional = 113,  /* loop_condition_optional  */
  YYSYMBOL_loop_init = 114,                /* loop_init  */
  YYSYMBOL_loop_iter = 115,                /* loop_iter  */
  YYSYMBOL_loop = 116,                     /* loop  */
  YYSYMBOL_function_start_expression = 117, /* function_start_expression  */
  YYSYMBOL_function_call_expression = 118, /* function_call_expression  */
  YYSYMBOL_assign_expression = 119,        /* assign_expression  */
  YYSYMBOL_binary_expression = 120,        /* binary_expression  */
  YYSYMBOL_ternary_expression = 121,       /* ternary_expression  */
  YYSYMBOL_unary_expression = 122,         /* unary_expression  */
  YYSYMBOL_pre_crement = 123,              /* pre_crement  */
  YYSYMBOL_post_crement = 124,             /* post_crement  */
  YYSYMBOL_variable_reference = 125,       /* variable_reference  */
  YYSYMBOL_array = 126,                    /* array  */
  YYSYMBOL_variable = 127,                 /* variable  */
  YYSYMBOL_attribute = 128,                /* attribute  */
  YYSYMBOL_external = 129,                 /* external  */
  YYSYMBOL_local = 130,                    /* local  */
  YYSYMBOL_literal = 131,                  /* literal  */
  YYSYMBOL_type = 132,                     /* type  */
  YYSYMBOL_matrix_type = 133,              /* matrix_type  */
  YYSYMBOL_scalar_type = 134,              /* scalar_type  */
  YYSYMBOL_vector_type = 135               /* vector_type  */
};
typedef enum yysymbol_kind_t yysymbol_kind_t;



/* Unqualified %code blocks.  */

    template<typename T, typename... Args>
    T* newNode(AXLTYPE* loc, const Args&... args) {
        T* ptr = new T(args...);
        OPENVDB_ASSERT(axlog);
        axlog->addNodeLocation(ptr, {loc->first_line, loc->first_column});
        return ptr;
    }


#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

/* Work around bug in HP-UX 11.23, which defines these macros
   incorrectly for preprocessor constants.  This workaround can likely
   be removed in 2023, as HPE has promised support for HP-UX 11.23
   (aka HP-UX 11i v2) only through the end of 2022; see Table 2 of
   <https://h20195.www2.hpe.com/V2/getpdf.aspx/4AA4-7673ENW.pdf>.  */
#ifdef __hpux
# undef UINT_LEAST8_MAX
# undef UINT_LEAST16_MAX
# define UINT_LEAST8_MAX 255
# define UINT_LEAST16_MAX 65535
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))


/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif


#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YY_USE(E) ((void) (E))
#else
# define YY_USE(E) /* empty */
#endif

/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
#if defined __GNUC__ && ! defined __ICC && 406 <= __GNUC__ * 100 + __GNUC_MINOR__
# if __GNUC__ * 100 + __GNUC_MINOR__ < 407
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")
# else
#  define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                           \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# endif
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if 1

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* 1 */

#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined AXLTYPE_IS_TRIVIAL && AXLTYPE_IS_TRIVIAL \
             && defined AXSTYPE_IS_TRIVIAL && AXSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE) \
             + YYSIZEOF (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  126
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   898

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  99
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  37
/* YYNRULES -- Number of rules.  */
#define YYNRULES  155
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  263

/* YYMAXUTOK -- Last valid token kind.  */
#define YYMAXUTOK   353


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK                     \
   ? YY_CAST (yysymbol_kind_t, yytranslate[YYX])        \
   : YYSYMBOL_YYUNDEF)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    89,    90,    91,    92,    93,    94,
      95,    96,    97,    98
};

#if AXDEBUG
/* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   211,   211,   214,   220,   221,   222,   225,   231,   232,
     238,   239,   240,   241,   242,   243,   244,   245,   248,   250,
     255,   256,   262,   263,   264,   265,   266,   267,   268,   269,
     270,   271,   272,   277,   279,   285,   290,   295,   302,   313,
     314,   319,   320,   326,   327,   332,   333,   337,   338,   343,
     344,   345,   350,   351,   356,   358,   359,   364,   365,   370,
     371,   372,   377,   378,   379,   380,   381,   382,   383,   384,
     385,   386,   387,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   406,   407,   408,   409,
     410,   414,   415,   420,   421,   422,   423,   427,   428,   432,
     433,   438,   439,   440,   441,   442,   443,   444,   458,   464,
     465,   470,   471,   472,   473,   474,   475,   476,   477,   478,
     483,   484,   485,   486,   487,   488,   495,   502,   503,   504,
     505,   506,   507,   508,   512,   513,   514,   515,   520,   521,
     522,   523,   528,   529,   530,   531,   532,   537,   538,   539,
     540,   541,   542,   543,   544,   545
};
#endif

/** Accessing symbol of state STATE.  */
#define YY_ACCESSING_SYMBOL(State) YY_CAST (yysymbol_kind_t, yystos[State])

#if 1
/* The user-facing name of the symbol whose (internal) number is
   YYSYMBOL.  No bounds checking.  */
static const char *yysymbol_name (yysymbol_kind_t yysymbol) YY_ATTRIBUTE_UNUSED;

/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "\"end of file\"", "error", "\"invalid token\"", "TRUE", "FALSE",
  "SEMICOLON", "AT", "DOLLAR", "IF", "ELSE", "FOR", "DO", "WHILE",
  "RETURN", "BREAK", "CONTINUE", "LCURLY", "RCURLY", "LSQUARE", "RSQUARE",
  "STRING", "DOUBLE", "FLOAT", "INT32", "INT64", "BOOL", "VEC2I", "VEC2F",
  "VEC2D", "VEC3I", "VEC3F", "VEC3D", "VEC4I", "VEC4F", "VEC4D", "F_AT",
  "I_AT", "V_AT", "S_AT", "I16_AT", "MAT3F", "MAT3D", "MAT4F", "MAT4D",
  "M3F_AT", "M4F_AT", "F_DOLLAR", "I_DOLLAR", "V_DOLLAR", "S_DOLLAR",
  "DOT_X", "DOT_Y", "DOT_Z", "L_INT32", "L_INT64", "L_FLOAT", "L_DOUBLE",
  "L_STRING", "IDENTIFIER", "COMMA", "QUESTION", "COLON", "EQUALS",
  "PLUSEQUALS", "MINUSEQUALS", "MULTIPLYEQUALS", "DIVIDEEQUALS",
  "MODULOEQUALS", "BITANDEQUALS", "BITXOREQUALS", "BITOREQUALS",
  "SHIFTLEFTEQUALS", "SHIFTRIGHTEQUALS", "OR", "AND", "BITOR", "BITXOR",
  "BITAND", "EQUALSEQUALS", "NOTEQUALS", "MORETHAN", "LESSTHAN",
  "MORETHANOREQUAL", "LESSTHANOREQUAL", "SHIFTLEFT", "SHIFTRIGHT", "PLUS",
  "MINUS", "MULTIPLY", "DIVIDE", "MODULO", "UMINUS", "NOT", "BITNOT",
  "PLUSPLUS", "MINUSMINUS", "LPARENS", "RPARENS", "LOWER_THAN_ELSE",
  "$accept", "tree", "body", "block", "statement", "expressions",
  "comma_operator", "expression", "declaration", "declaration_list",
  "declarations", "block_or_statement", "conditional_statement",
  "loop_condition", "loop_condition_optional", "loop_init", "loop_iter",
  "loop", "function_start_expression", "function_call_expression",
  "assign_expression", "binary_expression", "ternary_expression",
  "unary_expression", "pre_crement", "post_crement", "variable_reference",
  "array", "variable", "attribute", "external", "local", "literal", "type",
  "matrix_type", "scalar_type", "vector_type", YY_NULLPTR
};

static const char *
yysymbol_name (yysymbol_kind_t yysymbol)
{
  return yytname[yysymbol];
}
#endif

#define YYPACT_NINF (-225)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
static const yytype_int16 yypact[] =
{
     525,  -225,  -225,  -225,   -54,   -51,   -85,   -62,   525,   -49,
      48,    72,    83,   337,  -225,  -225,  -225,  -225,  -225,  -225,
    -225,  -225,  -225,  -225,  -225,  -225,  -225,  -225,  -225,    31,
      34,    35,    36,    40,  -225,  -225,  -225,  -225,    41,    64,
      65,    66,    85,    86,  -225,  -225,  -225,  -225,  -225,    -6,
     713,   713,   713,   713,   790,   790,   713,   145,   525,  -225,
    -225,   154,    87,   233,   102,   103,   158,  -225,  -225,   -56,
    -225,  -225,  -225,  -225,  -225,  -225,  -225,   533,  -225,    -8,
    -225,  -225,  -225,  -225,    20,  -225,    70,  -225,  -225,  -225,
     713,   713,  -225,  -225,   152,   713,  -225,  -225,  -225,  -225,
     431,   -11,  -225,  -225,  -225,  -225,  -225,  -225,  -225,  -225,
    -225,  -225,  -225,   242,   713,   -57,    22,  -225,  -225,  -225,
    -225,  -225,   161,  -225,  -225,    73,  -225,  -225,  -225,  -225,
     713,   713,   619,   713,   713,   713,   713,   713,   713,   713,
     713,   713,   713,   713,   713,   713,   713,   713,   713,   713,
     713,   111,   113,  -225,   713,  -225,   713,   713,   713,   713,
     713,   713,   713,   713,   713,   713,   713,  -225,  -225,   713,
    -225,  -225,  -225,   114,   115,   112,   713,    78,  -225,  -225,
     171,    81,  -225,  -225,   106,  -225,  -225,  -225,   421,   -11,
     233,  -225,   421,   421,   713,   327,   607,   697,   773,   787,
     808,    68,    68,   -70,   -70,   -70,   -70,    -7,    -7,   -57,
     -57,  -225,  -225,  -225,   116,   137,   421,   421,   421,   421,
     421,   421,   421,   421,   421,   421,   421,   421,   -14,  -225,
    -225,   713,   108,   525,   713,   713,   525,   421,   713,   713,
     713,  -225,   713,   421,  -225,   195,  -225,   201,   110,  -225,
     421,   421,   421,   141,   525,   713,  -225,  -225,  -225,  -225,
     135,   525,  -225
};

/* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
   Performed when YYTABLE does not specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,   132,   133,    17,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   137,   146,   145,   143,   144,   142,
     147,   148,   149,   150,   151,   152,   153,   154,   155,     0,
       0,     0,     0,     0,   138,   139,   140,   141,     0,     0,
       0,     0,     0,     0,   127,   128,   129,   130,   131,   126,
       0,     0,     0,     0,     0,     0,     0,     0,     3,     7,
       6,     0,    19,    18,    39,    40,     0,    12,    13,     0,
      26,    25,    22,    24,    23,   102,    29,    31,    30,   101,
     109,    28,   110,    27,     0,   136,   134,   135,   119,   125,
       0,    51,    41,    42,     0,     0,    14,    15,    16,     9,
       0,    19,   114,   113,   115,   116,   112,   117,   118,   122,
     121,   123,   124,     0,     0,    93,     0,    94,    96,    95,
     126,    97,     0,   134,    98,     0,     1,     5,     4,    10,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    11,     0,    60,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    99,   100,     0,
     103,   104,   105,     0,     0,    33,     0,     0,    49,    50,
       0,     0,    45,    46,     0,     8,   108,    59,    57,     0,
       0,    32,    21,    20,     0,     0,    84,    83,    81,    82,
      80,    85,    86,    87,    88,    89,    90,    78,    79,    73,
      74,    75,    76,    77,    36,    38,    58,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    71,    72,     0,   111,
     120,     0,     0,     0,    48,     0,     0,    92,     0,     0,
       0,   106,     0,    34,    61,    43,    47,     0,     0,    56,
      91,    35,    37,     0,     0,    53,    55,   107,    44,    52,
       0,     0,    54
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -225,  -225,   199,    38,    39,   -55,    12,   -29,   -93,  -225,
     117,  -224,  -225,  -185,  -225,  -225,  -225,  -225,  -225,  -225,
    -225,  -225,  -225,  -225,  -225,  -225,     2,  -225,  -225,  -225,
    -225,  -225,  -225,     0,  -225,    32,  -225
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
       0,    57,    58,    92,    93,    61,    62,    63,    64,    65,
      66,    94,    67,   184,   247,   180,   260,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,   116,    85,    86,    87
};

/* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule whose
   number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
      84,   125,   183,   154,    88,   241,   186,    89,    84,   245,
     169,    90,   249,    84,   144,   145,   146,   147,   148,   149,
     150,   115,   117,   118,   119,   101,   173,   174,   173,   174,
     258,   148,   149,   150,    91,   177,   178,   262,    59,    60,
     182,   155,   170,   171,   172,   242,   132,    95,   130,   246,
     248,    59,    60,    96,   122,   122,   121,   124,    84,   133,
     134,   135,   136,   137,   138,   139,   140,   141,   142,   143,
     144,   145,   146,   147,   148,   149,   150,    97,   175,   146,
     147,   148,   149,   150,   188,   190,   123,   123,    98,   102,
     113,    84,   103,   104,   105,    84,   127,   128,   106,   107,
      84,   192,   193,   195,   196,   197,   198,   199,   200,   201,
     202,   203,   204,   205,   206,   207,   208,   209,   210,   211,
     212,   213,   108,   109,   110,   216,   189,   217,   218,   219,
     220,   221,   222,   223,   224,   225,   226,   227,   127,   128,
     228,   183,   183,   111,   112,   126,   130,   232,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   129,
     257,   151,   152,   153,   181,   237,   176,   173,   132,   214,
     191,   215,   229,   230,   231,   233,   234,   235,   239,   182,
     182,   133,   134,   135,   136,   137,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150,   240,
     259,   132,   243,   236,   254,   244,   255,   256,   179,   250,
     251,   252,   100,   253,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,   261,    84,    84,    84,    84,     0,     0,     0,
       0,     0,     0,     0,     0,     1,     2,     0,     4,     5,
       0,     0,     0,     0,    84,     0,     0,     0,   114,     0,
       0,    84,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    40,    41,
      42,    43,   131,   132,     0,    44,    45,    46,    47,    48,
      49,     0,     0,     0,     0,     0,   133,   134,   135,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,     0,     0,     0,     0,    50,    51,
       0,     0,     0,     0,    52,    53,    54,    55,    56,   187,
       1,     2,     3,     4,     5,     6,     0,     7,     8,     9,
      10,    11,    12,    13,    99,     0,     0,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,   132,   238,     0,
      44,    45,    46,    47,    48,    49,     0,     0,     0,     0,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,     0,     0,
       0,     0,     0,    50,    51,     0,     0,     0,     0,    52,
      53,    54,    55,    56,     1,     2,     3,     4,     5,     6,
       0,     7,     8,     9,    10,    11,    12,    13,   185,     0,
       0,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,   132,     0,     0,    44,    45,    46,    47,    48,    49,
       0,     0,     0,     0,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,     0,     0,     0,     0,     0,    50,    51,     0,
       0,     0,     0,    52,    53,    54,    55,    56,     1,     2,
       3,     4,     5,     6,     0,     7,     8,     9,    10,    11,
      12,    13,     0,     0,     0,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,     0,     0,     0,    44,    45,
      46,    47,    48,    49,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   156,   157,   158,   159,   160,
     161,   162,   163,   164,   165,   166,     0,     0,     0,     0,
       0,    50,    51,     0,     0,     0,     0,    52,    53,    54,
      55,    56,     1,     2,     0,     4,     5,   167,   168,     0,
       0,     0,     0,     0,     0,   114,     0,     0,     0,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,     0,
       0,     0,    44,    45,    46,    47,    48,    49,     0,     0,
     194,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,     0,     0,
       0,     0,     0,     0,     0,    50,    51,     0,     0,     0,
       0,    52,    53,    54,    55,    56,     1,     2,     0,     4,
       5,     0,     0,     0,     0,     0,     0,     0,     0,   114,
       0,     0,     0,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,     0,     0,     0,    44,    45,    46,    47,
      48,    49,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,     0,     0,
       0,     0,     0,     0,     0,     0,     4,     0,     0,    50,
      51,     0,     0,     0,     0,    52,    53,    54,    55,    56,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    37,    38,    39,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   120,   136,
     137,   138,   139,   140,   141,   142,   143,   144,   145,   146,
     147,   148,   149,   150,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146,   147,   148,   149,   150,     0,     0,
       0,     0,     0,     0,    54,    55,   138,   139,   140,   141,
     142,   143,   144,   145,   146,   147,   148,   149,   150
};

static const yytype_int16 yycheck[] =
{
       0,    56,    95,    59,    58,    19,    17,    58,     8,   233,
      18,    96,   236,    13,    84,    85,    86,    87,    88,    89,
      90,    50,    51,    52,    53,    13,     6,     7,     6,     7,
     254,    88,    89,    90,    96,    90,    91,   261,     0,     0,
      95,    97,    50,    51,    52,    59,    60,    96,    59,   234,
     235,    13,    13,     5,    54,    55,    54,    55,    58,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    83,
      84,    85,    86,    87,    88,    89,    90,     5,    58,    86,
      87,    88,    89,    90,   113,   114,    54,    55,     5,    58,
      96,    91,    58,    58,    58,    95,    58,    58,    58,    58,
     100,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,   140,   141,   142,   143,   144,   145,   146,   147,   148,
     149,   150,    58,    58,    58,   154,   114,   156,   157,   158,
     159,   160,   161,   162,   163,   164,   165,   166,   100,   100,
     169,   234,   235,    58,    58,     0,    59,   176,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,     5,
      19,    59,    59,     5,    12,   194,    96,     6,    60,    58,
      97,    58,    58,    58,    62,    97,     5,    96,    62,   234,
     235,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90,    62,
     255,    60,   231,    97,     9,    97,     5,    97,    91,   238,
     239,   240,    13,   242,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    97,   233,   234,   235,   236,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,     3,     4,    -1,     6,     7,
      -1,    -1,    -1,    -1,   254,    -1,    -1,    -1,    16,    -1,
      -1,   261,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    39,    40,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    59,    60,    -1,    53,    54,    55,    56,    57,
      58,    -1,    -1,    -1,    -1,    -1,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    -1,    -1,    -1,    -1,    86,    87,
      -1,    -1,    -1,    -1,    92,    93,    94,    95,    96,    97,
       3,     4,     5,     6,     7,     8,    -1,    10,    11,    12,
      13,    14,    15,    16,    17,    -1,    -1,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    37,    38,    39,    40,    41,    42,
      43,    44,    45,    46,    47,    48,    49,    60,    61,    -1,
      53,    54,    55,    56,    57,    58,    -1,    -1,    -1,    -1,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    -1,    -1,    86,    87,    -1,    -1,    -1,    -1,    92,
      93,    94,    95,    96,     3,     4,     5,     6,     7,     8,
      -1,    10,    11,    12,    13,    14,    15,    16,    17,    -1,
      -1,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    37,    38,
      39,    40,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    60,    -1,    -1,    53,    54,    55,    56,    57,    58,
      -1,    -1,    -1,    -1,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      89,    90,    -1,    -1,    -1,    -1,    -1,    86,    87,    -1,
      -1,    -1,    -1,    92,    93,    94,    95,    96,     3,     4,
       5,     6,     7,     8,    -1,    10,    11,    12,    13,    14,
      15,    16,    -1,    -1,    -1,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    -1,    -1,    -1,    53,    54,
      55,    56,    57,    58,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    62,    63,    64,    65,    66,
      67,    68,    69,    70,    71,    72,    -1,    -1,    -1,    -1,
      -1,    86,    87,    -1,    -1,    -1,    -1,    92,    93,    94,
      95,    96,     3,     4,    -1,     6,     7,    94,    95,    -1,
      -1,    -1,    -1,    -1,    -1,    16,    -1,    -1,    -1,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    37,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    -1,
      -1,    -1,    53,    54,    55,    56,    57,    58,    -1,    -1,
      61,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    86,    87,    -1,    -1,    -1,
      -1,    92,    93,    94,    95,    96,     3,     4,    -1,     6,
       7,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    16,
      -1,    -1,    -1,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      37,    38,    39,    40,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    -1,    -1,    -1,    53,    54,    55,    56,
      57,    58,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,     6,    -1,    -1,    86,
      87,    -1,    -1,    -1,    -1,    92,    93,    94,    95,    96,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    39,
      40,    41,    42,    43,    44,    45,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    58,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    -1,    -1,
      -1,    -1,    -1,    -1,    94,    95,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    90
};

/* YYSTOS[STATE-NUM] -- The symbol kind of the accessing symbol of
   state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,    10,    11,    12,
      13,    14,    15,    16,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    37,    38,    39,    40,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    53,    54,    55,    56,    57,    58,
      86,    87,    92,    93,    94,    95,    96,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   111,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,    58,    58,
      96,    96,   102,   103,   110,    96,     5,     5,     5,    17,
     101,   105,    58,    58,    58,    58,    58,    58,    58,    58,
      58,    58,    58,    96,    16,   106,   132,   106,   106,   106,
      58,   125,   132,   134,   125,   104,     0,   102,   103,     5,
      59,    59,    60,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      90,    59,    59,     5,    59,    97,    62,    63,    64,    65,
      66,    67,    68,    69,    70,    71,    72,    94,    95,    18,
      50,    51,    52,     6,     7,    58,    96,   104,   104,   109,
     114,    12,   104,   107,   112,    17,    17,    97,   106,   105,
     106,    97,   106,   106,    61,   106,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   106,   106,   106,
     106,   106,   106,   106,    58,    58,   106,   106,   106,   106,
     106,   106,   106,   106,   106,   106,   106,   106,   106,    58,
      58,    62,   106,    97,     5,    96,    97,   106,    61,    62,
      62,    19,    59,   106,    97,   110,   112,   113,   112,   110,
     106,   106,   106,   106,     9,     5,    97,    19,   110,   104,
     115,    97,   110
};

/* YYR1[RULE-NUM] -- Symbol kind of the left-hand side of rule RULE-NUM.  */
static const yytype_uint8 yyr1[] =
{
       0,    99,   100,   100,   101,   101,   101,   101,   102,   102,
     103,   103,   103,   103,   103,   103,   103,   103,   104,   104,
     105,   105,   106,   106,   106,   106,   106,   106,   106,   106,
     106,   106,   106,   107,   107,   108,   108,   108,   108,   109,
     109,   110,   110,   111,   111,   112,   112,   113,   113,   114,
     114,   114,   115,   115,   116,   116,   116,   117,   117,   118,
     118,   118,   119,   119,   119,   119,   119,   119,   119,   119,
     119,   119,   119,   120,   120,   120,   120,   120,   120,   120,
     120,   120,   120,   120,   120,   120,   120,   120,   120,   120,
     120,   121,   121,   122,   122,   122,   122,   123,   123,   124,
     124,   125,   125,   125,   125,   125,   125,   125,   126,   127,
     127,   128,   128,   128,   128,   128,   128,   128,   128,   128,
     129,   129,   129,   129,   129,   129,   130,   131,   131,   131,
     131,   131,   131,   131,   132,   132,   132,   132,   133,   133,
     133,   133,   134,   134,   134,   134,   134,   135,   135,   135,
     135,   135,   135,   135,   135,   135
};

/* YYR2[RULE-NUM] -- Number of symbols on the right-hand side of rule RULE-NUM.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     0,     1,     2,     2,     1,     1,     3,     2,
       2,     2,     1,     1,     2,     2,     2,     1,     1,     1,
       3,     3,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     3,     2,     4,     5,     3,     5,     3,     1,
       1,     1,     1,     5,     7,     1,     1,     1,     0,     1,
       1,     0,     1,     0,     9,     6,     5,     3,     3,     3,
       2,     4,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     5,     4,     2,     2,     2,     2,     2,     2,     2,
       2,     1,     1,     2,     2,     2,     4,     6,     3,     1,
       1,     3,     2,     2,     2,     2,     2,     2,     2,     2,
       3,     2,     2,     2,     2,     2,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1
};


enum { YYENOMEM = -2 };

#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = AXEMPTY)

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab
#define YYNOMEM         goto yyexhaustedlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == AXEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (tree, YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Backward compatibility with an undocumented macro.
   Use AXerror or AXUNDEF. */
#define YYERRCODE AXUNDEF

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)                                \
    do                                                                  \
      if (N)                                                            \
        {                                                               \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;        \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;      \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;         \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;       \
        }                                                               \
      else                                                              \
        {                                                               \
          (Current).first_line   = (Current).last_line   =              \
            YYRHSLOC (Rhs, 0).last_line;                                \
          (Current).first_column = (Current).last_column =              \
            YYRHSLOC (Rhs, 0).last_column;                              \
        }                                                               \
    while (0)
#endif

#define YYRHSLOC(Rhs, K) ((Rhs)[K])


/* Enable debugging if requested.  */
#if AXDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)


/* YYLOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

# ifndef YYLOCATION_PRINT

#  if defined YY_LOCATION_PRINT

   /* Temporary convenience wrapper in case some people defined the
      undocumented and private YY_LOCATION_PRINT macros.  */
#   define YYLOCATION_PRINT(File, Loc)  YY_LOCATION_PRINT(File, *(Loc))

#  elif defined AXLTYPE_IS_TRIVIAL && AXLTYPE_IS_TRIVIAL

/* Print *YYLOCP on YYO.  Private, do not rely on its existence. */

YY_ATTRIBUTE_UNUSED
static int
yy_location_print_ (FILE *yyo, YYLTYPE const * const yylocp)
{
  int res = 0;
  int end_col = 0 != yylocp->last_column ? yylocp->last_column - 1 : 0;
  if (0 <= yylocp->first_line)
    {
      res += YYFPRINTF (yyo, "%d", yylocp->first_line);
      if (0 <= yylocp->first_column)
        res += YYFPRINTF (yyo, ".%d", yylocp->first_column);
    }
  if (0 <= yylocp->last_line)
    {
      if (yylocp->first_line < yylocp->last_line)
        {
          res += YYFPRINTF (yyo, "-%d", yylocp->last_line);
          if (0 <= end_col)
            res += YYFPRINTF (yyo, ".%d", end_col);
        }
      else if (0 <= end_col && yylocp->first_column < end_col)
        res += YYFPRINTF (yyo, "-%d", end_col);
    }
  return res;
}

#   define YYLOCATION_PRINT  yy_location_print_

    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT(File, Loc)  YYLOCATION_PRINT(File, &(Loc))

#  else

#   define YYLOCATION_PRINT(File, Loc) ((void) 0)
    /* Temporary convenience wrapper in case some people defined the
       undocumented and private YY_LOCATION_PRINT macros.  */
#   define YY_LOCATION_PRINT  YYLOCATION_PRINT

#  endif
# endif /* !defined YYLOCATION_PRINT */


# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Kind, Value, Location, tree); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo,
                       yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, openvdb::ax::ast::Tree** tree)
{
  FILE *yyoutput = yyo;
  YY_USE (yyoutput);
  YY_USE (yylocationp);
  YY_USE (tree);
  if (!yyvaluep)
    return;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YY_USE (yykind);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo,
                 yysymbol_kind_t yykind, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp, openvdb::ax::ast::Tree** tree)
{
  YYFPRINTF (yyo, "%s %s (",
             yykind < YYNTOKENS ? "token" : "nterm", yysymbol_name (yykind));

  YYLOCATION_PRINT (yyo, yylocationp);
  YYFPRINTF (yyo, ": ");
  yy_symbol_value_print (yyo, yykind, yyvaluep, yylocationp, tree);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, YYLTYPE *yylsp,
                 int yyrule, openvdb::ax::ast::Tree** tree)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       YY_ACCESSING_SYMBOL (+yyssp[yyi + 1 - yynrhs]),
                       &yyvsp[(yyi + 1) - (yynrhs)],
                       &(yylsp[(yyi + 1) - (yynrhs)]), tree);
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, yylsp, Rule, tree); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !AXDEBUG */
# define YYDPRINTF(Args) ((void) 0)
# define YY_SYMBOL_PRINT(Title, Kind, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !AXDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


/* Context of a parse error.  */
typedef struct
{
  yy_state_t *yyssp;
  yysymbol_kind_t yytoken;
  YYLTYPE *yylloc;
} yypcontext_t;

/* Put in YYARG at most YYARGN of the expected tokens given the
   current YYCTX, and return the number of tokens stored in YYARG.  If
   YYARG is null, return the number of expected tokens (guaranteed to
   be less than YYNTOKENS).  Return YYENOMEM on memory exhaustion.
   Return 0 if there are more than YYARGN expected tokens, yet fill
   YYARG up to YYARGN. */
static int
yypcontext_expected_tokens (const yypcontext_t *yyctx,
                            yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  int yyn = yypact[+*yyctx->yyssp];
  if (!yypact_value_is_default (yyn))
    {
      /* Start YYX at -YYN if negative to avoid negative indexes in
         YYCHECK.  In other words, skip the first -YYN actions for
         this state because they are default actions.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;
      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yyx;
      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
        if (yycheck[yyx + yyn] == yyx && yyx != YYSYMBOL_YYerror
            && !yytable_value_is_error (yytable[yyx + yyn]))
          {
            if (!yyarg)
              ++yycount;
            else if (yycount == yyargn)
              return 0;
            else
              yyarg[yycount++] = YY_CAST (yysymbol_kind_t, yyx);
          }
    }
  if (yyarg && yycount == 0 && 0 < yyargn)
    yyarg[0] = YYSYMBOL_YYEMPTY;
  return yycount;
}




#ifndef yystrlen
# if defined __GLIBC__ && defined _STRING_H
#  define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
# else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
# endif
#endif

#ifndef yystpcpy
# if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#  define yystpcpy stpcpy
# else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
# endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;
      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
#endif


static int
yy_syntax_error_arguments (const yypcontext_t *yyctx,
                           yysymbol_kind_t yyarg[], int yyargn)
{
  /* Actual size of YYARG. */
  int yycount = 0;
  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yyctx->yytoken != YYSYMBOL_YYEMPTY)
    {
      int yyn;
      if (yyarg)
        yyarg[yycount] = yyctx->yytoken;
      ++yycount;
      yyn = yypcontext_expected_tokens (yyctx,
                                        yyarg ? yyarg + 1 : yyarg, yyargn - 1);
      if (yyn == YYENOMEM)
        return YYENOMEM;
      else
        yycount += yyn;
    }
  return yycount;
}

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return -1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return YYENOMEM if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                const yypcontext_t *yyctx)
{
  enum { YYARGS_MAX = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  yysymbol_kind_t yyarg[YYARGS_MAX];
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* Actual size of YYARG. */
  int yycount = yy_syntax_error_arguments (yyctx, yyarg, YYARGS_MAX);
  if (yycount == YYENOMEM)
    return YYENOMEM;

  switch (yycount)
    {
#define YYCASE_(N, S)                       \
      case N:                               \
        yyformat = S;                       \
        break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
#undef YYCASE_
    }

  /* Compute error message size.  Don't count the "%s"s, but reserve
     room for the terminator.  */
  yysize = yystrlen (yyformat) - 2 * yycount + 1;
  {
    int yyi;
    for (yyi = 0; yyi < yycount; ++yyi)
      {
        YYPTRDIFF_T yysize1
          = yysize + yytnamerr (YY_NULLPTR, yytname[yyarg[yyi]]);
        if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
          yysize = yysize1;
        else
          return YYENOMEM;
      }
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return -1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yytname[yyarg[yyi++]]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg,
            yysymbol_kind_t yykind, YYSTYPE *yyvaluep, YYLTYPE *yylocationp, openvdb::ax::ast::Tree** tree)
{
  YY_USE (yyvaluep);
  YY_USE (yylocationp);
  YY_USE (tree);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yykind, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  switch (yykind)
    {
    case YYSYMBOL_L_INT32: /* L_INT32  */
            { }
        break;

    case YYSYMBOL_L_INT64: /* L_INT64  */
            { }
        break;

    case YYSYMBOL_L_FLOAT: /* L_FLOAT  */
            { }
        break;

    case YYSYMBOL_L_DOUBLE: /* L_DOUBLE  */
            { }
        break;

    case YYSYMBOL_L_STRING: /* L_STRING  */
            { free(const_cast<char*>(((*yyvaluep).string))); }
        break;

    case YYSYMBOL_IDENTIFIER: /* IDENTIFIER  */
            { free(const_cast<char*>(((*yyvaluep).string))); }
        break;

    case YYSYMBOL_tree: /* tree  */
            { }
        break;

    case YYSYMBOL_body: /* body  */
            { delete ((*yyvaluep).block); }
        break;

    case YYSYMBOL_block: /* block  */
            { delete ((*yyvaluep).block); }
        break;

    case YYSYMBOL_statement: /* statement  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_expressions: /* expressions  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_comma_operator: /* comma_operator  */
            { for (auto& ptr : *((*yyvaluep).explist)) delete ptr; delete ((*yyvaluep).explist); }
        break;

    case YYSYMBOL_expression: /* expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_declaration: /* declaration  */
            { delete ((*yyvaluep).declare_local); }
        break;

    case YYSYMBOL_declaration_list: /* declaration_list  */
            { delete ((*yyvaluep).statementlist); }
        break;

    case YYSYMBOL_declarations: /* declarations  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_block_or_statement: /* block_or_statement  */
            { delete ((*yyvaluep).block); }
        break;

    case YYSYMBOL_conditional_statement: /* conditional_statement  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_loop_condition: /* loop_condition  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_loop_condition_optional: /* loop_condition_optional  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_loop_init: /* loop_init  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_loop_iter: /* loop_iter  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_loop: /* loop  */
            { delete ((*yyvaluep).statement); }
        break;

    case YYSYMBOL_function_start_expression: /* function_start_expression  */
            { delete ((*yyvaluep).function); }
        break;

    case YYSYMBOL_function_call_expression: /* function_call_expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_assign_expression: /* assign_expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_binary_expression: /* binary_expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_ternary_expression: /* ternary_expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_unary_expression: /* unary_expression  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_pre_crement: /* pre_crement  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_post_crement: /* post_crement  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_variable_reference: /* variable_reference  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_array: /* array  */
            { delete ((*yyvaluep).expression); }
        break;

    case YYSYMBOL_variable: /* variable  */
            { delete ((*yyvaluep).variable); }
        break;

    case YYSYMBOL_attribute: /* attribute  */
            { delete ((*yyvaluep).attribute); }
        break;

    case YYSYMBOL_external: /* external  */
            { delete ((*yyvaluep).external); }
        break;

    case YYSYMBOL_local: /* local  */
            { delete ((*yyvaluep).local); }
        break;

    case YYSYMBOL_literal: /* literal  */
            { delete ((*yyvaluep).value); }
        break;

    case YYSYMBOL_type: /* type  */
            { }
        break;

    case YYSYMBOL_matrix_type: /* matrix_type  */
            { }
        break;

    case YYSYMBOL_scalar_type: /* scalar_type  */
            { }
        break;

    case YYSYMBOL_vector_type: /* vector_type  */
            { }
        break;

      default:
        break;
    }
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/* Lookahead token kind.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Location data for the lookahead symbol.  */
YYLTYPE yylloc
# if defined AXLTYPE_IS_TRIVIAL && AXLTYPE_IS_TRIVIAL
  = { 1, 1, 1, 1 }
# endif
;
/* Number of syntax errors so far.  */
int yynerrs;




/*----------.
| yyparse.  |
`----------*/

int
yyparse (openvdb::ax::ast::Tree** tree)
{
    yy_state_fast_t yystate = 0;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus = 0;

    /* Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* Their size.  */
    YYPTRDIFF_T yystacksize = YYINITDEPTH;

    /* The state stack: array, bottom, top.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss = yyssa;
    yy_state_t *yyssp = yyss;

    /* The semantic value stack: array, bottom, top.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp = yyvs;

    /* The location stack: array, bottom, top.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls = yylsa;
    YYLTYPE *yylsp = yyls;

  int yyn;
  /* The return value of yyparse.  */
  int yyresult;
  /* Lookahead symbol kind.  */
  yysymbol_kind_t yytoken = YYSYMBOL_YYEMPTY;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

  /* The locations where the error started and ended.  */
  YYLTYPE yyerror_range[3];

  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yychar = AXEMPTY; /* Cause a token to be read.  */

  yylsp[0] = yylloc;
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END
  YY_STACK_PRINT (yyss, yyssp);

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    YYNOMEM;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;
        YYLTYPE *yyls1 = yyls;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yyls1, yysize * YYSIZEOF (*yylsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
        yyls = yyls1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        YYNOMEM;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          YYNOMEM;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
        YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */


  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either empty, or end-of-input, or a valid lookahead.  */
  if (yychar == AXEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token\n"));
      yychar = yylex ();
    }

  if (yychar <= AXEOF)
    {
      yychar = AXEOF;
      yytoken = YYSYMBOL_YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else if (yychar == AXerror)
    {
      /* The scanner already issued an error message, process directly
         to error recovery.  But do not keep the error token as
         lookahead, it is too special and may lead us to an endless
         loop in error recovery. */
      yychar = AXUNDEF;
      yytoken = YYSYMBOL_YYerror;
      yyerror_range[1] = yylloc;
      goto yyerrlab1;
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END
  *++yylsp = yylloc;

  /* Discard the shifted token.  */
  yychar = AXEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location. */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  yyerror_range[1] = yyloc;
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2: /* tree: %empty  */
                         {  *tree = newNode<Tree>(&(yyloc));
                    (yyval.tree) = *tree;
                 }
    break;

  case 3: /* tree: body  */
                 {  *tree = newNode<Tree>(&(yylsp[0]), (yyvsp[0].block));
                    (yyval.tree) = *tree;
                 }
    break;

  case 4: /* body: body statement  */
                      { (yyvsp[-1].block)->addStatement((yyvsp[0].statement)); (yyval.block) = (yyvsp[-1].block); }
    break;

  case 5: /* body: body block  */
                      { (yyvsp[-1].block)->addStatement((yyvsp[0].block)); (yyval.block) = (yyvsp[-1].block); }
    break;

  case 6: /* body: statement  */
                      { (yyval.block) = newNode<Block>(&(yyloc));
                        (yyval.block)->addStatement((yyvsp[0].statement));
                      }
    break;

  case 7: /* body: block  */
                      { (yyval.block) = newNode<Block>(&(yyloc));
                        (yyval.block)->addStatement((yyvsp[0].block));
                      }
    break;

  case 8: /* block: LCURLY body RCURLY  */
                            { (yyval.block) = (yyvsp[-1].block); }
    break;

  case 9: /* block: LCURLY RCURLY  */
                            { (yyval.block) = newNode<Block>(&(yyloc)); }
    break;

  case 10: /* statement: expressions SEMICOLON  */
                              { (yyval.statement) = (yyvsp[-1].expression); }
    break;

  case 11: /* statement: declarations SEMICOLON  */
                              { (yyval.statement) = (yyvsp[-1].statement); }
    break;

  case 12: /* statement: conditional_statement  */
                              { (yyval.statement) = (yyvsp[0].statement); }
    break;

  case 13: /* statement: loop  */
                              { (yyval.statement) = (yyvsp[0].statement); }
    break;

  case 14: /* statement: RETURN SEMICOLON  */
                              { (yyval.statement) = newNode<Keyword>(&(yyloc), tokens::RETURN); }
    break;

  case 15: /* statement: BREAK SEMICOLON  */
                              { (yyval.statement) = newNode<Keyword>(&(yyloc), tokens::BREAK); }
    break;

  case 16: /* statement: CONTINUE SEMICOLON  */
                              { (yyval.statement) = newNode<Keyword>(&(yyloc), tokens::CONTINUE); }
    break;

  case 17: /* statement: SEMICOLON  */
                              { (yyval.statement) = nullptr; }
    break;

  case 18: /* expressions: expression  */
                      { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 19: /* expressions: comma_operator  */
                      { (yyval.expression) = newNode<CommaOperator>(&(yyloc), *static_cast<ExpList*>((yyvsp[0].explist))); delete (yyvsp[0].explist); }
    break;

  case 20: /* comma_operator: expression COMMA expression  */
                                      { (yyval.explist) = new ExpList(); (yyval.explist)->assign({(yyvsp[-2].expression), (yyvsp[0].expression)}); }
    break;

  case 21: /* comma_operator: comma_operator COMMA expression  */
                                      { (yyvsp[-2].explist)->emplace_back((yyvsp[0].expression)); (yyval.explist) = (yyvsp[-2].explist); }
    break;

  case 22: /* expression: binary_expression  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 23: /* expression: unary_expression  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 24: /* expression: ternary_expression  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 25: /* expression: assign_expression  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 26: /* expression: function_call_expression  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 27: /* expression: literal  */
                                   { (yyval.expression) = (yyvsp[0].value); }
    break;

  case 28: /* expression: external  */
                                   { (yyval.expression) = (yyvsp[0].external); }
    break;

  case 29: /* expression: post_crement  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 30: /* expression: array  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 31: /* expression: variable_reference  */
                                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 32: /* expression: LPARENS expressions RPARENS  */
                                   { (yyval.expression) = (yyvsp[-1].expression); }
    break;

  case 33: /* declaration: type IDENTIFIER  */
                                         { (yyval.declare_local)  = newNode<DeclareLocal>(&(yylsp[-1]), static_cast<tokens::CoreType>((yyvsp[-1].index)), newNode<Local>(&(yylsp[0]), (yyvsp[0].string)));
                                            free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 34: /* declaration: type IDENTIFIER EQUALS expression  */
                                         { (yyval.declare_local) = newNode<DeclareLocal>(&(yylsp[-3]), static_cast<tokens::CoreType>((yyvsp[-3].index)), newNode<Local>(&(yylsp[-2]), (yyvsp[-2].string)), (yyvsp[0].expression));
                                            free(const_cast<char*>((yyvsp[-2].string))); }
    break;

  case 35: /* declaration_list: declaration COMMA IDENTIFIER EQUALS expression  */
                                                            { (yyval.statementlist) = newNode<StatementList>(&(yyloc), (yyvsp[-4].declare_local));
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>((yyvsp[-4].declare_local))->type();
                                                              (yyval.statementlist)->addStatement(newNode<DeclareLocal>(&(yylsp[-4]), type, newNode<Local>(&(yylsp[-2]), (yyvsp[-2].string)), (yyvsp[0].expression)));
                                                              free(const_cast<char*>((yyvsp[-2].string)));
                                                            }
    break;

  case 36: /* declaration_list: declaration COMMA IDENTIFIER  */
                                                            { (yyval.statementlist) = newNode<StatementList>(&(yyloc), (yyvsp[-2].declare_local));
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>((yyvsp[-2].declare_local))->type();
                                                              (yyval.statementlist)->addStatement(newNode<DeclareLocal>(&(yylsp[-2]), type, newNode<Local>(&(yylsp[0]), (yyvsp[0].string))));
                                                              free(const_cast<char*>((yyvsp[0].string)));
                                                            }
    break;

  case 37: /* declaration_list: declaration_list COMMA IDENTIFIER EQUALS expression  */
                                                            { const auto firstNode = (yyvsp[-4].statementlist)->child(0);
                                                              OPENVDB_ASSERT(firstNode);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>(firstNode)->type();
                                                              (yyval.statementlist)->addStatement(newNode<DeclareLocal>(&(yylsp[-4]), type, newNode<Local>(&(yylsp[-2]), (yyvsp[-2].string)), (yyvsp[0].expression)));
                                                              (yyval.statementlist) = (yyvsp[-4].statementlist);
                                                              free(const_cast<char*>((yyvsp[-2].string)));
                                                            }
    break;

  case 38: /* declaration_list: declaration_list COMMA IDENTIFIER  */
                                                            { const auto firstNode = (yyvsp[-2].statementlist)->child(0);
                                                              OPENVDB_ASSERT(firstNode);
                                                              const tokens::CoreType type =  static_cast<const DeclareLocal*>(firstNode)->type();
                                                              (yyval.statementlist)->addStatement(newNode<DeclareLocal>(&(yylsp[-2]), type, newNode<Local>(&(yylsp[0]), (yyvsp[0].string))));
                                                              free(const_cast<char*>((yyvsp[0].string)));
                                                              (yyval.statementlist) = (yyvsp[-2].statementlist);
                                                            }
    break;

  case 39: /* declarations: declaration  */
                        { (yyval.statement) = (yyvsp[0].declare_local); }
    break;

  case 40: /* declarations: declaration_list  */
                        { (yyval.statement) = (yyvsp[0].statementlist); }
    break;

  case 41: /* block_or_statement: block  */
                { (yyval.block) = (yyvsp[0].block); }
    break;

  case 42: /* block_or_statement: statement  */
                { (yyval.block) = newNode<Block>(&(yyloc)); (yyval.block)->addStatement((yyvsp[0].statement)); }
    break;

  case 43: /* conditional_statement: IF LPARENS expressions RPARENS block_or_statement  */
                                                                                { (yyval.statement) = newNode<ConditionalStatement>(&(yyloc), (yyvsp[-2].expression), (yyvsp[0].block)); }
    break;

  case 44: /* conditional_statement: IF LPARENS expressions RPARENS block_or_statement ELSE block_or_statement  */
                                                                                { (yyval.statement) = newNode<ConditionalStatement>(&(yyloc), (yyvsp[-4].expression), (yyvsp[-2].block), (yyvsp[0].block)); }
    break;

  case 45: /* loop_condition: expressions  */
                                { (yyval.statement) = (yyvsp[0].expression); }
    break;

  case 46: /* loop_condition: declaration  */
                                { (yyval.statement) = (yyvsp[0].declare_local); }
    break;

  case 47: /* loop_condition_optional: loop_condition  */
                      { (yyval.statement) = (yyvsp[0].statement); }
    break;

  case 48: /* loop_condition_optional: %empty  */
                              { (yyval.statement) = nullptr; }
    break;

  case 49: /* loop_init: expressions  */
                    { (yyval.statement) = (yyvsp[0].expression); }
    break;

  case 50: /* loop_init: declarations  */
                    { (yyval.statement) = (yyvsp[0].statement); }
    break;

  case 51: /* loop_init: %empty  */
                            { (yyval.statement) = nullptr; }
    break;

  case 52: /* loop_iter: expressions  */
                   { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 53: /* loop_iter: %empty  */
                           { (yyval.expression) = nullptr; }
    break;

  case 54: /* loop: FOR LPARENS loop_init SEMICOLON loop_condition_optional SEMICOLON loop_iter RPARENS block_or_statement  */
                                                                    { (yyval.statement) = newNode<Loop>(&(yyloc), tokens::FOR, ((yyvsp[-4].statement) ? (yyvsp[-4].statement) : newNode<Value<bool>>(&(yyloc), true)), (yyvsp[0].block), (yyvsp[-6].statement), (yyvsp[-2].expression)); }
    break;

  case 55: /* loop: DO block_or_statement WHILE LPARENS loop_condition RPARENS  */
                                                                    { (yyval.statement) = newNode<Loop>(&(yyloc), tokens::DO, (yyvsp[-1].statement), (yyvsp[-4].block)); }
    break;

  case 56: /* loop: WHILE LPARENS loop_condition RPARENS block_or_statement  */
                                                                    { (yyval.statement) = newNode<Loop>(&(yyloc), tokens::WHILE, (yyvsp[-2].statement), (yyvsp[0].block)); }
    break;

  case 57: /* function_start_expression: IDENTIFIER LPARENS expression  */
                                                  { (yyval.function) = newNode<FunctionCall>(&(yylsp[-2]), (yyvsp[-2].string)); (yyval.function)->append((yyvsp[0].expression)); free(const_cast<char*>((yyvsp[-2].string))); }
    break;

  case 58: /* function_start_expression: function_start_expression COMMA expression  */
                                                  { (yyvsp[-2].function)->append((yyvsp[0].expression)); (yyval.function) = (yyvsp[-2].function); }
    break;

  case 59: /* function_call_expression: IDENTIFIER LPARENS RPARENS  */
                                              { (yyval.expression) = newNode<FunctionCall>(&(yylsp[-2]), (yyvsp[-2].string)); free(const_cast<char*>((yyvsp[-2].string))); }
    break;

  case 60: /* function_call_expression: function_start_expression RPARENS  */
                                              { (yyval.expression) = (yyvsp[-1].function); }
    break;

  case 61: /* function_call_expression: scalar_type LPARENS expression RPARENS  */
                                              { (yyval.expression) = newNode<Cast>(&(yylsp[-3]), (yyvsp[-1].expression), static_cast<tokens::CoreType>((yyvsp[-3].index))); }
    break;

  case 62: /* assign_expression: variable_reference EQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression)); }
    break;

  case 63: /* assign_expression: variable_reference PLUSEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::PLUS); }
    break;

  case 64: /* assign_expression: variable_reference MINUSEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MINUS); }
    break;

  case 65: /* assign_expression: variable_reference MULTIPLYEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MULTIPLY); }
    break;

  case 66: /* assign_expression: variable_reference DIVIDEEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::DIVIDE); }
    break;

  case 67: /* assign_expression: variable_reference MODULOEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MODULO); }
    break;

  case 68: /* assign_expression: variable_reference BITANDEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITAND); }
    break;

  case 69: /* assign_expression: variable_reference BITXOREQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITXOR); }
    break;

  case 70: /* assign_expression: variable_reference BITOREQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITOR); }
    break;

  case 71: /* assign_expression: variable_reference SHIFTLEFTEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::SHIFTLEFT); }
    break;

  case 72: /* assign_expression: variable_reference SHIFTRIGHTEQUALS expression  */
                                                      { (yyval.expression) = newNode<AssignExpression>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::SHIFTRIGHT); }
    break;

  case 73: /* binary_expression: expression PLUS expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::PLUS); }
    break;

  case 74: /* binary_expression: expression MINUS expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MINUS); }
    break;

  case 75: /* binary_expression: expression MULTIPLY expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MULTIPLY); }
    break;

  case 76: /* binary_expression: expression DIVIDE expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::DIVIDE); }
    break;

  case 77: /* binary_expression: expression MODULO expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MODULO); }
    break;

  case 78: /* binary_expression: expression SHIFTLEFT expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::SHIFTLEFT); }
    break;

  case 79: /* binary_expression: expression SHIFTRIGHT expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::SHIFTRIGHT); }
    break;

  case 80: /* binary_expression: expression BITAND expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITAND); }
    break;

  case 81: /* binary_expression: expression BITOR expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITOR); }
    break;

  case 82: /* binary_expression: expression BITXOR expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::BITXOR); }
    break;

  case 83: /* binary_expression: expression AND expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::AND); }
    break;

  case 84: /* binary_expression: expression OR expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::OR); }
    break;

  case 85: /* binary_expression: expression EQUALSEQUALS expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::EQUALSEQUALS); }
    break;

  case 86: /* binary_expression: expression NOTEQUALS expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::NOTEQUALS); }
    break;

  case 87: /* binary_expression: expression MORETHAN expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MORETHAN); }
    break;

  case 88: /* binary_expression: expression LESSTHAN expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::LESSTHAN); }
    break;

  case 89: /* binary_expression: expression MORETHANOREQUAL expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::MORETHANOREQUAL); }
    break;

  case 90: /* binary_expression: expression LESSTHANOREQUAL expression  */
                                             { (yyval.expression) = newNode<BinaryOperator>(&(yylsp[-2]), (yyvsp[-2].expression), (yyvsp[0].expression), tokens::LESSTHANOREQUAL); }
    break;

  case 91: /* ternary_expression: expression QUESTION expression COLON expression  */
                                                      { (yyval.expression) = newNode<TernaryOperator>(&(yylsp[-4]), (yyvsp[-4].expression), (yyvsp[-2].expression), (yyvsp[0].expression)); }
    break;

  case 92: /* ternary_expression: expression QUESTION COLON expression  */
                                                      { (yyval.expression) = newNode<TernaryOperator>(&(yylsp[-3]), (yyvsp[-3].expression), nullptr, (yyvsp[0].expression)); }
    break;

  case 93: /* unary_expression: PLUS expression  */
                                     { (yyval.expression) = newNode<UnaryOperator>(&(yylsp[-1]), (yyvsp[0].expression), tokens::PLUS); }
    break;

  case 94: /* unary_expression: MINUS expression  */
                                     { (yyval.expression) = newNode<UnaryOperator>(&(yylsp[-1]), (yyvsp[0].expression), tokens::MINUS); }
    break;

  case 95: /* unary_expression: BITNOT expression  */
                                     { (yyval.expression) = newNode<UnaryOperator>(&(yylsp[-1]), (yyvsp[0].expression), tokens::BITNOT); }
    break;

  case 96: /* unary_expression: NOT expression  */
                                     { (yyval.expression) = newNode<UnaryOperator>(&(yylsp[-1]), (yyvsp[0].expression), tokens::NOT); }
    break;

  case 97: /* pre_crement: PLUSPLUS variable_reference  */
                                     { (yyval.expression) = newNode<Crement>(&(yylsp[-1]), (yyvsp[0].expression), Crement::Increment, /*post*/false); }
    break;

  case 98: /* pre_crement: MINUSMINUS variable_reference  */
                                     { (yyval.expression) = newNode<Crement>(&(yylsp[-1]), (yyvsp[0].expression), Crement::Decrement, /*post*/false); }
    break;

  case 99: /* post_crement: variable_reference PLUSPLUS  */
                                     { (yyval.expression) = newNode<Crement>(&(yylsp[-1]), (yyvsp[-1].expression), Crement::Increment, /*post*/true); }
    break;

  case 100: /* post_crement: variable_reference MINUSMINUS  */
                                     { (yyval.expression) = newNode<Crement>(&(yylsp[-1]), (yyvsp[-1].expression), Crement::Decrement, /*post*/true); }
    break;

  case 101: /* variable_reference: variable  */
                                                            { (yyval.expression) = (yyvsp[0].variable); }
    break;

  case 102: /* variable_reference: pre_crement  */
                                                            { (yyval.expression) = (yyvsp[0].expression); }
    break;

  case 103: /* variable_reference: variable DOT_X  */
                                                            { (yyval.expression) = newNode<ArrayUnpack>(&(yylsp[-1]), (yyvsp[-1].variable), newNode<Value<int32_t>>(&(yylsp[0]), 0));  }
    break;

  case 104: /* variable_reference: variable DOT_Y  */
                                                            { (yyval.expression) = newNode<ArrayUnpack>(&(yylsp[-1]), (yyvsp[-1].variable), newNode<Value<int32_t>>(&(yylsp[0]), 1)); }
    break;

  case 105: /* variable_reference: variable DOT_Z  */
                                                            { (yyval.expression) = newNode<ArrayUnpack>(&(yylsp[-1]), (yyvsp[-1].variable), newNode<Value<int32_t>>(&(yylsp[0]), 2));  }
    break;

  case 106: /* variable_reference: variable LSQUARE expression RSQUARE  */
                                                            { (yyval.expression) = newNode<ArrayUnpack>(&(yylsp[-3]), (yyvsp[-3].variable), (yyvsp[-1].expression)); }
    break;

  case 107: /* variable_reference: variable LSQUARE expression COMMA expression RSQUARE  */
                                                            { (yyval.expression) = newNode<ArrayUnpack>(&(yylsp[-5]), (yyvsp[-5].variable), (yyvsp[-3].expression), (yyvsp[-1].expression));  }
    break;

  case 108: /* array: LCURLY comma_operator RCURLY  */
                                   { (yyval.expression) = newNode<ArrayPack>(&(yylsp[-2]), *(yyvsp[-1].explist)); delete (yyvsp[-1].explist); }
    break;

  case 109: /* variable: attribute  */
                 { (yyval.variable) = (yyvsp[0].attribute); }
    break;

  case 110: /* variable: local  */
                 { (yyval.variable) = (yyvsp[0].local); }
    break;

  case 111: /* attribute: type AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), static_cast<tokens::CoreType>((yyvsp[-2].index))); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 112: /* attribute: I16_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::INT16); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 113: /* attribute: I_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::INT32); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 114: /* attribute: F_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::FLOAT); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 115: /* attribute: V_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::VEC3F); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 116: /* attribute: S_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::STRING); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 117: /* attribute: M3F_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::MAT3F); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 118: /* attribute: M4F_AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::MAT4F); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 119: /* attribute: AT IDENTIFIER  */
                             { (yyval.attribute) = newNode<Attribute>(&(yyloc), (yyvsp[0].string), tokens::FLOAT, true); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 120: /* external: type DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), static_cast<tokens::CoreType>((yyvsp[-2].index))); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 121: /* external: I_DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), tokens::INT32); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 122: /* external: F_DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), tokens::FLOAT); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 123: /* external: V_DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), tokens::VEC3F); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 124: /* external: S_DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), tokens::STRING); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 125: /* external: DOLLAR IDENTIFIER  */
                              { (yyval.external) = newNode<ExternalVariable>(&(yyloc), (yyvsp[0].string), tokens::FLOAT); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 126: /* local: IDENTIFIER  */
                { (yyval.local) = newNode<Local>(&(yyloc), (yyvsp[0].string)); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 127: /* literal: L_INT32  */
                { (yyval.value) = newNode<Value<int32_t>>(&(yylsp[0]), (yyvsp[0].index)); }
    break;

  case 128: /* literal: L_INT64  */
                { (yyval.value) = newNode<Value<int64_t>>(&(yylsp[0]), (yyvsp[0].index)); }
    break;

  case 129: /* literal: L_FLOAT  */
                { (yyval.value) = newNode<Value<float>>(&(yylsp[0]), static_cast<float>((yyvsp[0].flt))); }
    break;

  case 130: /* literal: L_DOUBLE  */
                { (yyval.value) = newNode<Value<double>>(&(yylsp[0]), (yyvsp[0].flt)); }
    break;

  case 131: /* literal: L_STRING  */
                { (yyval.value) = newNode<Value<std::string>>(&(yylsp[0]), (yyvsp[0].string)); free(const_cast<char*>((yyvsp[0].string))); }
    break;

  case 132: /* literal: TRUE  */
                { (yyval.value) = newNode<Value<bool>>(&(yylsp[0]), true); }
    break;

  case 133: /* literal: FALSE  */
                { (yyval.value) = newNode<Value<bool>>(&(yylsp[0]), false); }
    break;

  case 134: /* type: scalar_type  */
                    { (yyval.index) = (yyvsp[0].index); }
    break;

  case 135: /* type: vector_type  */
                    { (yyval.index) = (yyvsp[0].index); }
    break;

  case 136: /* type: matrix_type  */
                    { (yyval.index) = (yyvsp[0].index); }
    break;

  case 137: /* type: STRING  */
                    { (yyval.index) = tokens::STRING; }
    break;

  case 138: /* matrix_type: MAT3F  */
              { (yyval.index) = tokens::MAT3F; }
    break;

  case 139: /* matrix_type: MAT3D  */
              { (yyval.index) = tokens::MAT3D; }
    break;

  case 140: /* matrix_type: MAT4F  */
              { (yyval.index) = tokens::MAT4F; }
    break;

  case 141: /* matrix_type: MAT4D  */
              { (yyval.index) = tokens::MAT4D; }
    break;

  case 142: /* scalar_type: BOOL  */
              { (yyval.index) = tokens::BOOL; }
    break;

  case 143: /* scalar_type: INT32  */
              { (yyval.index) = tokens::INT32; }
    break;

  case 144: /* scalar_type: INT64  */
              { (yyval.index) = tokens::INT64; }
    break;

  case 145: /* scalar_type: FLOAT  */
              { (yyval.index) = tokens::FLOAT; }
    break;

  case 146: /* scalar_type: DOUBLE  */
              { (yyval.index) = tokens::DOUBLE; }
    break;

  case 147: /* vector_type: VEC2I  */
              { (yyval.index) = tokens::VEC2I; }
    break;

  case 148: /* vector_type: VEC2F  */
              { (yyval.index) = tokens::VEC2F; }
    break;

  case 149: /* vector_type: VEC2D  */
              { (yyval.index) = tokens::VEC2D; }
    break;

  case 150: /* vector_type: VEC3I  */
              { (yyval.index) = tokens::VEC3I; }
    break;

  case 151: /* vector_type: VEC3F  */
              { (yyval.index) = tokens::VEC3F; }
    break;

  case 152: /* vector_type: VEC3D  */
              { (yyval.index) = tokens::VEC3D; }
    break;

  case 153: /* vector_type: VEC4I  */
              { (yyval.index) = tokens::VEC4I; }
    break;

  case 154: /* vector_type: VEC4F  */
              { (yyval.index) = tokens::VEC4F; }
    break;

  case 155: /* vector_type: VEC4D  */
              { (yyval.index) = tokens::VEC4D; }
    break;



      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", YY_CAST (yysymbol_kind_t, yyr1[yyn]), &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == AXEMPTY ? YYSYMBOL_YYEMPTY : YYTRANSLATE (yychar);
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
      {
        yypcontext_t yyctx
          = {yyssp, yytoken, &yylloc};
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == -1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *,
                             YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (yymsg)
              {
                yysyntax_error_status
                  = yysyntax_error (&yymsg_alloc, &yymsg, &yyctx);
                yymsgp = yymsg;
              }
            else
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = YYENOMEM;
              }
          }
        yyerror (tree, yymsgp);
        if (yysyntax_error_status == YYENOMEM)
          YYNOMEM;
      }
    }

  yyerror_range[1] = yylloc;
  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= AXEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == AXEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval, &yylloc, tree);
          yychar = AXEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;
  ++yynerrs;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  /* Pop stack until we find a state that shifts the error token.  */
  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYSYMBOL_YYerror;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYSYMBOL_YYerror)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;

      yyerror_range[1] = *yylsp;
      yydestruct ("Error: popping",
                  YY_ACCESSING_SYMBOL (yystate), yyvsp, yylsp, tree);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  yyerror_range[2] = yylloc;
  ++yylsp;
  YYLLOC_DEFAULT (*yylsp, yyerror_range, 2);

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", YY_ACCESSING_SYMBOL (yyn), yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturnlab;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturnlab;


/*-----------------------------------------------------------.
| yyexhaustedlab -- YYNOMEM (memory exhaustion) comes here.  |
`-----------------------------------------------------------*/
yyexhaustedlab:
  yyerror (tree, YY_("memory exhausted"));
  yyresult = 2;
  goto yyreturnlab;


/*----------------------------------------------------------.
| yyreturnlab -- parsing is finished, clean up and return.  |
`----------------------------------------------------------*/
yyreturnlab:
  if (yychar != AXEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval, &yylloc, tree);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  YY_ACCESSING_SYMBOL (+*yyssp), yyvsp, yylsp, tree);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
  return yyresult;
}



OPENVDB_NO_TYPE_CONVERSION_WARNING_END

