// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file grammar/axparser.y
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  OpenVDB AX Grammar Rules
///

%code top {
    #include "openvdb_ax/ast/AST.h"
    #include "openvdb_ax/ast/Parse.h"
    #include "openvdb_ax/ast/Tokens.h"
    #include "openvdb_ax/compiler/Logger.h"
    #include <openvdb/util/Assert.h>
    #include <vector>

    extern int axlex();
    extern openvdb::ax::Logger* axlog;

    using namespace openvdb::ax::ast;
    using namespace openvdb::ax;

    void axerror(Tree** tree, const char* s);

    using ExpList = std::vector<openvdb::ax::ast::Expression*>;
}

/* Option 'parse.error verbose' tells bison to output verbose parsing errors
 * as a char* array to yyerror (axerror). Note that this is in lieu of doing
 * more specific error handling ourselves, as the actual tokens are printed
 * which is confusing.
 * @todo Implement a proper error handler
 */
%define parse.error verbose

/* Option 'api.prefix {ax}' matches the prefix option in the lexer to produce
 * prefixed C++ symbols (where 'yy' is replaced with 'ax') so we can link
 * with other flex-generated lexers in the same application.
 */
%define api.prefix {ax}

/* Tell bison to track grammar locations
 */
%locations

/* Our collection of strongly typed semantic values. Whilst nodes could all
   be represented by ast::Node pointers, specific types allow for compiler
   failures on incorrect usage within the parser.
 */
%union
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
}


%code
{
    template<typename T, typename... Args>
    T* newNode(AXLTYPE* loc, const Args&... args) {
        T* ptr = new T(args...);
        OPENVDB_ASSERT(axlog);
        axlog->addNodeLocation(ptr, {loc->first_line, loc->first_column});
        return ptr;
    }
}

/* AX token type names/terminal symbols
 */

%token TRUE FALSE
%token SEMICOLON AT DOLLAR
%token IF ELSE
%token FOR DO WHILE
%token RETURN BREAK CONTINUE
%token LCURLY RCURLY
%token LSQUARE RSQUARE
%token STRING DOUBLE FLOAT INT32 INT64 BOOL
%token VEC2I VEC2F VEC2D VEC3I VEC3F VEC3D VEC4I VEC4F VEC4D
%token F_AT I_AT V_AT S_AT I16_AT
%token MAT3F MAT3D MAT4F MAT4D M3F_AT M4F_AT
%token F_DOLLAR I_DOLLAR V_DOLLAR S_DOLLAR
%token DOT_X DOT_Y DOT_Z
%token <index> L_INT32 L_INT64
%token <flt> L_FLOAT
%token <flt> L_DOUBLE
%token <string> L_STRING IDENTIFIER

/* AX nonterminal symbols and their union types
 */
%type <tree>  tree
%type <block> block
%type <block> body
%type <block> block_or_statement
%type <statement> statement
%type <statement> conditional_statement
%type <statement> loop loop_init loop_condition loop_condition_optional
%type <expression> loop_iter

%type <statement> declarations
%type <statementlist> declaration_list

%type <function> function_start_expression

%type <expression> assign_expression
%type <expression> function_call_expression
%type <expression> binary_expression
%type <expression> unary_expression
%type <expression> ternary_expression
%type <expression> array
%type <expression> variable_reference
%type <expression> pre_crement
%type <expression> post_crement
%type <expression> expression
%type <expression> expressions
%type <explist> comma_operator
%type <variable> variable
%type <attribute> attribute
%type <external> external

%type <declare_local> declaration

%type <local> local
%type <value> literal
%type <index> type
%type <index> scalar_type
%type <index> vector_type
%type <index> matrix_type

/* Destructors designed for deallocation of discarded symbols during
 * error recovery. Note that the start symbol, tree, is additionally
 * deallocated by bison on successfully completion. It is only ever
 * constructed on a successfully parse, thus we avoid ever explicitly
 * deallocating it.
 */
%destructor { } <index> // nothing to do
%destructor { } <flt> // nothing to do
%destructor { } <tree>
%destructor { free(const_cast<char*>($$)); } <string>
%destructor { for (auto& ptr : *$$) delete ptr; delete $$; } <explist>
%destructor { delete $$; } <*> // all other AST nodes

/*  Operator Precedence Definitions. Precendence goes from lowest to
 *  highest, e.g. assignment operations are generally lowest. Note
 *  that this precedence and associativity is heavily based off of C++:
 *  https://en.cppreference.com/w/cpp/language/operator_precedence
 */
%left COMMA
%right QUESTION COLON EQUALS PLUSEQUALS MINUSEQUALS MULTIPLYEQUALS DIVIDEEQUALS MODULOEQUALS BITANDEQUALS BITXOREQUALS BITOREQUALS SHIFTLEFTEQUALS SHIFTRIGHTEQUALS
%left OR
%left AND
%left BITOR
%left BITXOR
%left BITAND
%left EQUALSEQUALS NOTEQUALS
%left MORETHAN LESSTHAN MORETHANOREQUAL LESSTHANOREQUAL
%left SHIFTLEFT SHIFTRIGHT
%left PLUS MINUS
%left MULTIPLY DIVIDE MODULO
/*  UMINUS exists for contextual precedence with negation. Note that
 *  whilst the evaluation of (-a)*b == -(a*b) and (-a)/b == -(a/b)
 *  (due to truncated division), AX implements a floored modulus.
 *  This means thats (-a)%b != -(a%b) and, as such, this precedence
 *  must be respected. Note that in general it makes sense to adhere
 *  to this anyway (i.e. reprint -a * b rather than -(a*b))
 */
%left UMINUS // for contextual precedence with negation
%left NOT BITNOT PLUSPLUS MINUSMINUS
%left LCURLY RCURLY
%left LPARENS RPARENS

%nonassoc LOWER_THAN_ELSE // null token to force no associativity in conditional statement
%nonassoc ELSE

/*  The start token from AX for bison, represents a fully constructed AST.
 */
%parse-param {openvdb::ax::ast::Tree** tree}

%start tree

/* Begin grammar
 */
%%

tree:
    /*empty*/     %empty {  *tree = newNode<Tree>(&@$);
                    $$ = *tree;
                 }
    | body       {  *tree = newNode<Tree>(&@1, $1);
                    $$ = *tree;
                 }
;

body:
      body statement  { $1->addStatement($2); $$ = $1; }
    | body block      { $1->addStatement($2); $$ = $1; }
    | statement       { $$ = newNode<Block>(&@$);
                        $$->addStatement($1);
                      }
    | block           { $$ = newNode<Block>(&@$);
                        $$->addStatement($1);
                      }
;

block:
      LCURLY body RCURLY    { $$ = $2; }
    | LCURLY RCURLY         { $$ = newNode<Block>(&@$); }
;

/// @brief  Syntax for a statement; a line followed by a semicolon, a
///         conditional statement or a loop
statement:
      expressions SEMICOLON   { $$ = $1; }
    | declarations SEMICOLON  { $$ = $1; }
    | conditional_statement   { $$ = $1; }
    | loop                    { $$ = $1; }
    | RETURN SEMICOLON        { $$ = newNode<Keyword>(&@$, tokens::RETURN); }
    | BREAK SEMICOLON         { $$ = newNode<Keyword>(&@$, tokens::BREAK); }
    | CONTINUE SEMICOLON      { $$ = newNode<Keyword>(&@$, tokens::CONTINUE); }
    | SEMICOLON               { $$ = nullptr; }

expressions:
      expression      { $$ = $1; }
    /// delete the ExpList after constructing a CommaOperator (ownership of the contents are taken, not the container)
    | comma_operator  { $$ = newNode<CommaOperator>(&@$, *static_cast<ExpList*>($1)); delete $1; }
;

/// @brief  Comma operator
comma_operator:
      expression COMMA expression     { $$ = new ExpList(); $$->assign({$1, $3}); }
    | comma_operator COMMA expression { $1->emplace_back($3); $$ = $1; }
;

/// @brief  Syntax for a combination of all numerical expressions which can return
///         an rvalue.
expression:
      binary_expression            { $$ = $1; }
    | unary_expression             { $$ = $1; }
    | ternary_expression           { $$ = $1; }
    | assign_expression            { $$ = $1; }
    | function_call_expression     { $$ = $1; }
    | literal                      { $$ = $1; }
    | external                     { $$ = $1; }
    | post_crement                 { $$ = $1; }
    | array                        { $$ = $1; }
    | variable_reference           { $$ = $1; }
    | LPARENS expressions RPARENS  { $$ = $2; }
;

/// @brief  Syntax for the declaration of supported local variable types
declaration:
      type IDENTIFIER                    { $$  = newNode<DeclareLocal>(&@1, static_cast<tokens::CoreType>($1), newNode<Local>(&@2, $2));
                                            free(const_cast<char*>($2)); }
    | type IDENTIFIER EQUALS expression  { $$ = newNode<DeclareLocal>(&@1, static_cast<tokens::CoreType>($1), newNode<Local>(&@2, $2), $4);
                                            free(const_cast<char*>($2)); }
;

/// @brief  A declaration list of at least size 2
declaration_list:
     declaration COMMA IDENTIFIER EQUALS expression         { $$ = newNode<StatementList>(&@$, $1);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>($1)->type();
                                                              $$->addStatement(newNode<DeclareLocal>(&@1, type, newNode<Local>(&@3, $3), $5));
                                                              free(const_cast<char*>($3));
                                                            }
    | declaration COMMA IDENTIFIER                          { $$ = newNode<StatementList>(&@$, $1);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>($1)->type();
                                                              $$->addStatement(newNode<DeclareLocal>(&@1, type, newNode<Local>(&@3, $3)));
                                                              free(const_cast<char*>($3));
                                                            }
    | declaration_list COMMA IDENTIFIER EQUALS expression   { const auto firstNode = $1->child(0);
                                                              OPENVDB_ASSERT(firstNode);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>(firstNode)->type();
                                                              $$->addStatement(newNode<DeclareLocal>(&@1, type, newNode<Local>(&@3, $3), $5));
                                                              $$ = $1;
                                                              free(const_cast<char*>($3));
                                                            }
    | declaration_list COMMA IDENTIFIER                     { const auto firstNode = $1->child(0);
                                                              OPENVDB_ASSERT(firstNode);
                                                              const tokens::CoreType type =  static_cast<const DeclareLocal*>(firstNode)->type();
                                                              $$->addStatement(newNode<DeclareLocal>(&@1, type, newNode<Local>(&@3, $3)));
                                                              free(const_cast<char*>($3));
                                                              $$ = $1;
                                                            }
;

/// @brief  Variable numbers of declarations, either creating a single declaration or a list
declarations:
      declaration       { $$ = $1; }
    | declaration_list  { $$ = $1; }
;

/// @brief  A single line scope or a scoped block
block_or_statement:
      block     { $$ = $1; }
    | statement { $$ = newNode<Block>(&@$); $$->addStatement($1); }
;

/// @brief  Syntax for a conditional statement, capable of supporting a single if
///         and an optional single else. Multiple else ifs are handled by this.
conditional_statement:
      IF LPARENS expressions RPARENS block_or_statement %prec LOWER_THAN_ELSE   { $$ = newNode<ConditionalStatement>(&@$, $3, $5); }
    | IF LPARENS expressions RPARENS block_or_statement ELSE block_or_statement { $$ = newNode<ConditionalStatement>(&@$, $3, $5, $7); }
;

/// @brief  A loop condition statement, either an initialized declaration or a list of expressions
loop_condition:
      expressions               { $$ = $1; }
    | declaration               { $$ = $1; }
;

loop_condition_optional:
      loop_condition  { $$ = $1; }
    | /*empty*/        %empty { $$ = nullptr; }
;

/// @brief A for loop initial statement, an optional list of declarations/list of expressions
loop_init:
      expressions   { $$ = $1; }
    | declarations  { $$ = $1; }
    | /*empty*/      %empty { $$ = nullptr; }
;

/// @brief A for loop iteration statement, an optional list of expressions
loop_iter:
      expressions  { $$ = $1; }
    | /* empty */   %empty { $$ = nullptr; }
;

/// @brief  For loops, while loops and do-while loops.
loop:
      FOR LPARENS loop_init SEMICOLON loop_condition_optional SEMICOLON loop_iter RPARENS block_or_statement
                                                                    { $$ = newNode<Loop>(&@$, tokens::FOR, ($5 ? $5 : newNode<Value<bool>>(&@$, true)), $9, $3, $7); }
    | DO block_or_statement WHILE LPARENS loop_condition RPARENS    { $$ = newNode<Loop>(&@$, tokens::DO, $5, $2); }
    | WHILE LPARENS loop_condition RPARENS block_or_statement       { $$ = newNode<Loop>(&@$, tokens::WHILE, $3, $5); }
;

/// @brief  Beginning/builder syntax for function calls with arguments
function_start_expression:
      IDENTIFIER LPARENS expression               { $$ = newNode<FunctionCall>(&@1, $1); $$->append($3); free(const_cast<char*>($1)); }
    | function_start_expression COMMA expression  { $1->append($3); $$ = $1; }
;

/// @brief  A function call, taking zero or a comma separated list of arguments
function_call_expression:
      IDENTIFIER LPARENS RPARENS              { $$ = newNode<FunctionCall>(&@1, $1); free(const_cast<char*>($1)); }
    | function_start_expression RPARENS       { $$ = $1; }
    | scalar_type LPARENS expression RPARENS  { $$ = newNode<Cast>(&@1, $3, static_cast<tokens::CoreType>($1)); }
;

/// @brief  Assign expressions for attributes and local variables
assign_expression:
      variable_reference EQUALS expression            { $$ = newNode<AssignExpression>(&@1, $1, $3); }
    | variable_reference PLUSEQUALS expression        { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::PLUS); }
    | variable_reference MINUSEQUALS expression       { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::MINUS); }
    | variable_reference MULTIPLYEQUALS expression    { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::MULTIPLY); }
    | variable_reference DIVIDEEQUALS expression      { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::DIVIDE); }
    | variable_reference MODULOEQUALS expression      { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::MODULO); }
    | variable_reference BITANDEQUALS expression      { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::BITAND); }
    | variable_reference BITXOREQUALS expression      { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::BITXOR); }
    | variable_reference BITOREQUALS expression       { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::BITOR); }
    | variable_reference SHIFTLEFTEQUALS expression   { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::SHIFTLEFT); }
    | variable_reference SHIFTRIGHTEQUALS expression  { $$ = newNode<AssignExpression>(&@1, $1, $3, tokens::SHIFTRIGHT); }
;

/// @brief  A binary expression which takes a left and right hand side expression
///         and returns an expression
binary_expression:
      expression PLUS expression             { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::PLUS); }
    | expression MINUS expression            { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::MINUS); }
    | expression MULTIPLY expression         { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::MULTIPLY); }
    | expression DIVIDE expression           { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::DIVIDE); }
    | expression MODULO expression           { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::MODULO); }
    | expression SHIFTLEFT expression        { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::SHIFTLEFT); }
    | expression SHIFTRIGHT expression       { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::SHIFTRIGHT); }
    | expression BITAND expression           { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::BITAND); }
    | expression BITOR expression            { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::BITOR); }
    | expression BITXOR expression           { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::BITXOR); }
    | expression AND expression              { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::AND); }
    | expression OR expression               { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::OR); }
    | expression EQUALSEQUALS expression     { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::EQUALSEQUALS); }
    | expression NOTEQUALS expression        { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::NOTEQUALS); }
    | expression MORETHAN expression         { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::MORETHAN); }
    | expression LESSTHAN expression         { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::LESSTHAN); }
    | expression MORETHANOREQUAL expression  { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::MORETHANOREQUAL); }
    | expression LESSTHANOREQUAL expression  { $$ = newNode<BinaryOperator>(&@1, $1, $3, tokens::LESSTHANOREQUAL); }
;

ternary_expression:
      expression QUESTION expression COLON expression { $$ = newNode<TernaryOperator>(&@1, $1, $3, $5); }
    | expression QUESTION COLON expression            { $$ = newNode<TernaryOperator>(&@1, $1, nullptr, $4); }
;

/// @brief  A unary expression which takes an expression and returns an expression
unary_expression:
      PLUS expression                { $$ = newNode<UnaryOperator>(&@1, $2, tokens::PLUS); }
    | MINUS expression %prec UMINUS  { $$ = newNode<UnaryOperator>(&@1, $2, tokens::MINUS); }
    | BITNOT expression              { $$ = newNode<UnaryOperator>(&@1, $2, tokens::BITNOT); }
    | NOT expression                 { $$ = newNode<UnaryOperator>(&@1, $2, tokens::NOT); }
;

pre_crement:
      PLUSPLUS variable_reference    { $$ = newNode<Crement>(&@1, $2, Crement::Increment, /*post*/false); }
    | MINUSMINUS variable_reference  { $$ = newNode<Crement>(&@1, $2, Crement::Decrement, /*post*/false); }
;

post_crement:
      variable_reference PLUSPLUS    { $$ = newNode<Crement>(&@1, $1, Crement::Increment, /*post*/true); }
    | variable_reference MINUSMINUS  { $$ = newNode<Crement>(&@1, $1, Crement::Decrement, /*post*/true); }
;

/// @brief  Syntax which can return a valid variable lvalue
variable_reference:
      variable                                              { $$ = $1; }
    | pre_crement                                           { $$ = $1; }
    | variable DOT_X                                        { $$ = newNode<ArrayUnpack>(&@1, $1, newNode<Value<int32_t>>(&@2, 0));  }
    | variable DOT_Y                                        { $$ = newNode<ArrayUnpack>(&@1, $1, newNode<Value<int32_t>>(&@2, 1)); }
    | variable DOT_Z                                        { $$ = newNode<ArrayUnpack>(&@1, $1, newNode<Value<int32_t>>(&@2, 2));  }
    | variable LSQUARE expression RSQUARE                   { $$ = newNode<ArrayUnpack>(&@1, $1, $3); }
    | variable LSQUARE expression COMMA expression RSQUARE  { $$ = newNode<ArrayUnpack>(&@1, $1, $3, $5);  }
;

/// @brief  Terminating syntax for containers
/// @todo   due to terminal conflicts, the array pack must be represented as
///         a single rule. Consider:
///           if (a) {a,a};  vs  if (a) { a,a; }
///         there must be a way to use contextual precedence to solve this.
///         For now, the entire non-terminal for arrays is defined up front.
///         This requires it to take in a comma_operator which is temporarily
///         represented as an vector of non-owned expressions.
/// @note   delete the ExpList after constructing an ArrayPack (ownership of
///         the contents are taken, not the container)
array:
      LCURLY comma_operator RCURLY { $$ = newNode<ArrayPack>(&@1, *$2); delete $2; }
;

/// @brief  Objects which are assignable are considered variables. Importantly,
///         externals are not classified in this rule as they are read only.
variable:
      attribute  { $$ = $1; }
    | local      { $$ = $1; }
;

/// @brief  Syntax for supported attribute access
attribute:
      type AT IDENTIFIER     { $$ = newNode<Attribute>(&@$, $3, static_cast<tokens::CoreType>($1)); free(const_cast<char*>($3)); }
    | I16_AT IDENTIFIER      { $$ = newNode<Attribute>(&@$, $2, tokens::INT16); free(const_cast<char*>($2)); }
    | I_AT IDENTIFIER        { $$ = newNode<Attribute>(&@$, $2, tokens::INT32); free(const_cast<char*>($2)); }
    | F_AT IDENTIFIER        { $$ = newNode<Attribute>(&@$, $2, tokens::FLOAT); free(const_cast<char*>($2)); }
    | V_AT IDENTIFIER        { $$ = newNode<Attribute>(&@$, $2, tokens::VEC3F); free(const_cast<char*>($2)); }
    | S_AT IDENTIFIER        { $$ = newNode<Attribute>(&@$, $2, tokens::STRING); free(const_cast<char*>($2)); }
    | M3F_AT IDENTIFIER      { $$ = newNode<Attribute>(&@$, $2, tokens::MAT3F); free(const_cast<char*>($2)); }
    | M4F_AT IDENTIFIER      { $$ = newNode<Attribute>(&@$, $2, tokens::MAT4F); free(const_cast<char*>($2)); }
    | AT IDENTIFIER          { $$ = newNode<Attribute>(&@$, $2, tokens::FLOAT, true); free(const_cast<char*>($2)); }
;

/// @brief  Syntax for supported external variable access
external:
      type DOLLAR IDENTIFIER  { $$ = newNode<ExternalVariable>(&@$, $3, static_cast<tokens::CoreType>($1)); free(const_cast<char*>($3)); }
    | I_DOLLAR IDENTIFIER     { $$ = newNode<ExternalVariable>(&@$, $2, tokens::INT32); free(const_cast<char*>($2)); }
    | F_DOLLAR IDENTIFIER     { $$ = newNode<ExternalVariable>(&@$, $2, tokens::FLOAT); free(const_cast<char*>($2)); }
    | V_DOLLAR IDENTIFIER     { $$ = newNode<ExternalVariable>(&@$, $2, tokens::VEC3F); free(const_cast<char*>($2)); }
    | S_DOLLAR IDENTIFIER     { $$ = newNode<ExternalVariable>(&@$, $2, tokens::STRING); free(const_cast<char*>($2)); }
    | DOLLAR IDENTIFIER       { $$ = newNode<ExternalVariable>(&@$, $2, tokens::FLOAT); free(const_cast<char*>($2)); }
;

/// @brief  Syntax for text identifiers which resolves to a local. Types have
///         have their own tokens which do not evaluate to a local variable
/// @note   Anything which uses an IDENTIFIER must free the returned char array
local:
    IDENTIFIER  { $$ = newNode<Local>(&@$, $1); free(const_cast<char*>($1)); }
;

/// @brief  Syntax numerical and boolean literal values
/// @note   Anything which uses one of the below tokens must free the returned char
///         array (aside from TRUE and FALSE tokens)
literal:
      L_INT32   { $$ = newNode<Value<int32_t>>(&@1, $1); }
    | L_INT64   { $$ = newNode<Value<int64_t>>(&@1, $1); }
    | L_FLOAT   { $$ = newNode<Value<float>>(&@1, static_cast<float>($1)); }
    | L_DOUBLE  { $$ = newNode<Value<double>>(&@1, $1); }
    | L_STRING  { $$ = newNode<Value<std::string>>(&@1, $1); free(const_cast<char*>($1)); }
    | TRUE      { $$ = newNode<Value<bool>>(&@1, true); }
    | FALSE     { $$ = newNode<Value<bool>>(&@1, false); }
;

type:
      scalar_type   { $$ = $1; }
    | vector_type   { $$ = $1; }
    | matrix_type   { $$ = $1; }
    | STRING        { $$ = tokens::STRING; }
;

/// @brief  Matrix types
matrix_type:
      MAT3F   { $$ = tokens::MAT3F; }
    | MAT3D   { $$ = tokens::MAT3D; }
    | MAT4F   { $$ = tokens::MAT4F; }
    | MAT4D   { $$ = tokens::MAT4D; }
;

/// @brief  Scalar types
scalar_type:
      BOOL    { $$ = tokens::BOOL; }
    | INT32   { $$ = tokens::INT32; }
    | INT64   { $$ = tokens::INT64; }
    | FLOAT   { $$ = tokens::FLOAT; }
    | DOUBLE  { $$ = tokens::DOUBLE; }
;

/// @brief  Vector types
vector_type:
      VEC2I   { $$ = tokens::VEC2I; }
    | VEC2F   { $$ = tokens::VEC2F; }
    | VEC2D   { $$ = tokens::VEC2D; }
    | VEC3I   { $$ = tokens::VEC3I; }
    | VEC3F   { $$ = tokens::VEC3F; }
    | VEC3D   { $$ = tokens::VEC3D; }
    | VEC4I   { $$ = tokens::VEC4I; }
    | VEC4F   { $$ = tokens::VEC4F; }
    | VEC4D   { $$ = tokens::VEC4D; }
;

%%
