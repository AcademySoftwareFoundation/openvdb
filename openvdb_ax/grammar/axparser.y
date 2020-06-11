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

/// @file grammar/axparser.y
///
/// @authors Nick Avramoussis, Richard Jones
///
/// @brief  OpenVDB AX Grammar Rules
///

%{
    #include <stdio.h>
    #include <vector>

    #include <openvdb/Platform.h> // for OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN
    #include <openvdb_ax/ast/AST.h>
    #include <openvdb_ax/ast/Tokens.h>

    /// @note  Bypasses bison conversion warnings in yyparse
    OPENVDB_NO_TYPE_CONVERSION_WARNING_BEGIN

    extern int axlex();

    using namespace openvdb::ax::ast;

    void yyerror(Tree** tree, const char* s);

    using ExpList = std::vector<openvdb::ax::ast::Expression*>;
%}

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

/* AX token type names/terminal symbols
 */
%token TRUE FALSE
%token SEMICOLON AT DOLLAR
%token IF ELSE
%token FOR DO WHILE
%token RETURN BREAK CONTINUE
%token LCURLY RCURLY
%token LSQUARE RSQUARE
%token STRING DOUBLE FLOAT LONG INT SHORT BOOL VOID
%token VEC2I VEC2F VEC2D VEC3I VEC3F VEC3D VEC4I VEC4F VEC4D
%token F_AT I_AT V_AT S_AT
%token MAT3F MAT3D MAT4F MAT4D M3F_AT M4F_AT
%token F_DOLLAR I_DOLLAR V_DOLLAR S_DOLLAR
%token DOT_X DOT_Y DOT_Z
%token <string> L_SHORT
%token <string> L_INT
%token <string> L_LONG
%token <string> L_FLOAT
%token <string> L_DOUBLE
%token <string> L_STRING
%token <string> IDENTIFIER

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
    /*empty*/    { *tree = new Tree(); $$ = *tree; }
    | body       { *tree = new Tree($1); $$ = *tree; }
;

body:
      body statement  { $1->addStatement($2); $$ = $1; }
    | body block      { $1->addStatement($2); $$ = $1; }
    | statement       { $$ = new Block(); $$->addStatement($1); }
    | block           { $$ = new Block(); $$->addStatement($1); }
;

block:
      LCURLY body RCURLY    { $$ = $2; }
    | LCURLY RCURLY         { $$ = new Block(); }
;

/// @brief  Syntax for a statement; a line followed by a semicolon, a
///         conditional statement or a loop
statement:
      expressions SEMICOLON   { $$ = $1; }
    | declarations SEMICOLON  { $$ = $1; }
    | conditional_statement   { $$ = $1; }
    | loop                    { $$ = $1; }
    | RETURN SEMICOLON        { $$ = new Keyword(tokens::RETURN); }
    | BREAK SEMICOLON         { $$ = new Keyword(tokens::BREAK); }
    | CONTINUE SEMICOLON      { $$ = new Keyword(tokens::CONTINUE); }
    | SEMICOLON               { $$ = nullptr; }
;

expressions:
      expression      { $$ = $1; }
    | comma_operator  { $$ = new CommaOperator(*static_cast<ExpList*>($1)); }
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
      type IDENTIFIER    { $$ = new DeclareLocal(static_cast<tokens::CoreType>($1), new Local($2)); free(const_cast<char*>($2)); }
    | type IDENTIFIER EQUALS expression  { $$ = new DeclareLocal(static_cast<tokens::CoreType>($1),
                                                                new Local($2),
                                                                $4); free(const_cast<char*>($2)); }
;

/// @brief  A declaration list of at least size 2
declaration_list:
     declaration COMMA IDENTIFIER EQUALS expression         { $$ = new StatementList($1);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>($1)->type();
                                                              $$->addStatement(
                                                                  new DeclareLocal(type, new Local($3), $5));
                                                              free(const_cast<char*>($3));
                                                            }
    | declaration COMMA IDENTIFIER                          { $$ = new StatementList($1);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>($1)->type();
                                                              $$->addStatement(new DeclareLocal(type, new Local($3)));
                                                              free(const_cast<char*>($3));
                                                            }
    | declaration_list COMMA IDENTIFIER EQUALS expression   { const auto firstNode = $1->child(0);
                                                              assert(firstNode);
                                                              const tokens::CoreType type = static_cast<const DeclareLocal*>(firstNode)->type();
                                                              $$->addStatement(
                                                                  new DeclareLocal(type, new Local($3), $5));
                                                              free(const_cast<char*>($3));
                                                              $$ = $1;
                                                            }
    | declaration_list COMMA IDENTIFIER                     { const auto firstNode = $1->child(0);
                                                              assert(firstNode);
                                                              const tokens::CoreType type =  static_cast<const DeclareLocal*>(firstNode)->type();
                                                              $$->addStatement(new DeclareLocal(type, new Local($3)));
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
    | statement { $$ = new Block(); $$->addStatement($1); }
;

/// @brief  Syntax for a conditional statement, capable of supporting a single if
///         and an optional single else. Multiple else ifs are handled by this.
conditional_statement:
      IF LPARENS expressions RPARENS block_or_statement %prec LOWER_THAN_ELSE   { $$ = new ConditionalStatement($3, $5); }
    | IF LPARENS expressions RPARENS block_or_statement ELSE block_or_statement { $$ = new ConditionalStatement($3, $5, $7); }
;

/// @brief  A loop condition statement, either an initialized declaration or a list of expressions
loop_condition:
      expressions               { $$ = $1; }
    | declaration               { $$ = $1; }
;

loop_condition_optional:
      loop_condition  { $$ = $1; }
    | /*empty*/       { $$ = nullptr; }
;

/// @brief A for loop initial statement, an optional list of declarations/list of expressions
loop_init:
      expressions   { $$ = $1; }
    | declarations  { $$ = $1; }
    | /*empty*/     { $$ = nullptr; }
;

/// @brief A for loop iteration statement, an optional list of expressions
loop_iter:
      expressions  { $$ = $1; }
    | /* empty */  { $$ = nullptr; }
;

/// @brief  For loops, while loops and do-while loops.
loop:
      FOR LPARENS loop_init SEMICOLON loop_condition_optional SEMICOLON loop_iter RPARENS block_or_statement
                                                                    { $$ = new Loop(tokens::FOR, ($5 ? $5 : new Value<bool>(true)), $9, $3, $7); }
    | DO block_or_statement WHILE LPARENS loop_condition RPARENS    { $$ = new Loop(tokens::DO, $5, $2); }
    | WHILE LPARENS loop_condition RPARENS block_or_statement       { $$ = new Loop(tokens::WHILE, $3, $5); }
;

/// @brief  Beginning/builder syntax for function calls with arguments
function_start_expression:
      IDENTIFIER LPARENS expression               { $$ = new FunctionCall($1); $$->append($3); free(const_cast<char*>($1)); }
    | function_start_expression COMMA expression  { $1->append($3); $$ = $1; }
;

/// @brief  A function call, taking zero or a comma separated list of arguments
function_call_expression:
      IDENTIFIER LPARENS RPARENS              { $$ = new FunctionCall($1); free(const_cast<char*>($1)); }
    | function_start_expression RPARENS       { $$ = $1; }
    | scalar_type LPARENS expression RPARENS  { $$ = new Cast($3, static_cast<tokens::CoreType>($1)); }
;

/// @brief  Assign expressions for attributes and local variables
assign_expression:
      variable_reference EQUALS expression            { $$ = new AssignExpression($1, $3); }
    | variable_reference PLUSEQUALS expression        { $$ = new AssignExpression($1, $3, tokens::PLUS); }
    | variable_reference MINUSEQUALS expression       { $$ = new AssignExpression($1, $3, tokens::MINUS); }
    | variable_reference MULTIPLYEQUALS expression    { $$ = new AssignExpression($1, $3, tokens::MULTIPLY); }
    | variable_reference DIVIDEEQUALS expression      { $$ = new AssignExpression($1, $3, tokens::DIVIDE); }
    | variable_reference MODULOEQUALS expression      { $$ = new AssignExpression($1, $3, tokens::MODULO); }
    | variable_reference BITANDEQUALS expression      { $$ = new AssignExpression($1, $3, tokens::BITAND); }
    | variable_reference BITXOREQUALS expression      { $$ = new AssignExpression($1, $3, tokens::BITXOR); }
    | variable_reference BITOREQUALS expression       { $$ = new AssignExpression($1, $3, tokens::BITOR); }
    | variable_reference SHIFTLEFTEQUALS expression   { $$ = new AssignExpression($1, $3, tokens::SHIFTLEFT); }
    | variable_reference SHIFTRIGHTEQUALS expression  { $$ = new AssignExpression($1, $3, tokens::SHIFTRIGHT); }
;

/// @brief  A binary expression which takes a left and right hand side expression
///         and returns an expression
binary_expression:
      expression PLUS expression             { $$ = new BinaryOperator($1, $3, tokens::PLUS); }
    | expression MINUS expression            { $$ = new BinaryOperator($1, $3, tokens::MINUS); }
    | expression MULTIPLY expression         { $$ = new BinaryOperator($1, $3, tokens::MULTIPLY); }
    | expression DIVIDE expression           { $$ = new BinaryOperator($1, $3, tokens::DIVIDE); }
    | expression MODULO expression           { $$ = new BinaryOperator($1, $3, tokens::MODULO); }
    | expression SHIFTLEFT expression        { $$ = new BinaryOperator($1, $3, tokens::SHIFTLEFT); }
    | expression SHIFTRIGHT expression       { $$ = new BinaryOperator($1, $3, tokens::SHIFTRIGHT); }
    | expression BITAND expression           { $$ = new BinaryOperator($1, $3, tokens::BITAND); }
    | expression BITOR expression            { $$ = new BinaryOperator($1, $3, tokens::BITOR); }
    | expression BITXOR expression           { $$ = new BinaryOperator($1, $3, tokens::BITXOR); }
    | expression AND expression              { $$ = new BinaryOperator($1, $3, tokens::AND); }
    | expression OR expression               { $$ = new BinaryOperator($1, $3, tokens::OR); }
    | expression EQUALSEQUALS expression     { $$ = new BinaryOperator($1, $3, tokens::EQUALSEQUALS); }
    | expression NOTEQUALS expression        { $$ = new BinaryOperator($1, $3, tokens::NOTEQUALS); }
    | expression MORETHAN expression         { $$ = new BinaryOperator($1, $3, tokens::MORETHAN); }
    | expression LESSTHAN expression         { $$ = new BinaryOperator($1, $3, tokens::LESSTHAN); }
    | expression MORETHANOREQUAL expression  { $$ = new BinaryOperator($1, $3, tokens::MORETHANOREQUAL); }
    | expression LESSTHANOREQUAL expression  { $$ = new BinaryOperator($1, $3, tokens::LESSTHANOREQUAL); }
;

ternary_expression:
      expression QUESTION expression COLON expression { $$ = new TernaryOperator($1, $3, $5); }
    | expression QUESTION COLON expression            { $$ = new TernaryOperator($1, nullptr, $4); }
;

/// @brief  A unary expression which takes an expression and returns an expression
unary_expression:
      PLUS expression        { $$ = new UnaryOperator($2, tokens::PLUS); }
    | MINUS expression       { $$ = new UnaryOperator($2, tokens::MINUS); }
    | BITNOT expression      { $$ = new UnaryOperator($2, tokens::BITNOT); }
    | NOT expression         { $$ = new UnaryOperator($2, tokens::NOT); }
;

pre_crement:
      PLUSPLUS variable_reference    { $$ = new Crement($2, Crement::Increment, /*post*/false); }
    | MINUSMINUS variable_reference  { $$ = new Crement($2, Crement::Decrement, /*post*/false); }
;

post_crement:
      variable_reference PLUSPLUS    { $$ = new Crement($1, Crement::Increment, /*post*/true); }
    | variable_reference MINUSMINUS  { $$ = new Crement($1, Crement::Decrement, /*post*/true); }
;

/// @brief  Syntax which can return a valid variable lvalue
variable_reference:
      variable                                              { $$ = $1; }
    | pre_crement                                           { $$ = $1; }
    | variable DOT_X                                        { $$ = new ArrayUnpack($1, new Value<int32_t>(0)); }
    | variable DOT_Y                                        { $$ = new ArrayUnpack($1, new Value<int32_t>(1)); }
    | variable DOT_Z                                        { $$ = new ArrayUnpack($1, new Value<int32_t>(2)); }
    | variable LSQUARE expression RSQUARE                   { $$ = new ArrayUnpack($1, $3); }
    | variable LSQUARE expression COMMA expression RSQUARE  { $$ = new ArrayUnpack($1, $3, $5); }
;

/// @brief  Terminating syntax for containers
/// @todo   due to terminal conflicts, the array pack must be represented as
///         a single rule. Consider:
///           if (a) {a,a};  vs  if (a) { a,a; }
///         there must be a way to use contextual precedence to solve this.
///         For now, the entire non-terminal for arrays is defined up front.
///         This requires it to take in a comma_operator which is temporarily
///         represented as an vector of non-owned expressions.
array:
      LCURLY comma_operator RCURLY { $$ = new ArrayPack(*$2); }
;

/// @brief  Objects which are assignable are considered variables. Importantly,
///         externals are not classified in this rule as they are read only.
variable:
      attribute  { $$ = $1; }
    | local      { $$ = $1; }
;

/// @brief  Syntax for supported attribute access
attribute:
      type AT IDENTIFIER     { $$ = new Attribute($3, static_cast<tokens::CoreType>($1)); free(const_cast<char*>($3)); }
    | I_AT IDENTIFIER        { $$ = new Attribute($2, tokens::INT); free(const_cast<char*>($2)); }
    | F_AT IDENTIFIER        { $$ = new Attribute($2, tokens::FLOAT); free(const_cast<char*>($2)); }
    | V_AT IDENTIFIER        { $$ = new Attribute($2, tokens::VEC3F); free(const_cast<char*>($2)); }
    | S_AT IDENTIFIER        { $$ = new Attribute($2, tokens::STRING); free(const_cast<char*>($2)); }
    | M3F_AT IDENTIFIER      { $$ = new Attribute($2, tokens::MAT3F); free(const_cast<char*>($2)); }
    | M4F_AT IDENTIFIER      { $$ = new Attribute($2, tokens::MAT4F); free(const_cast<char*>($2)); }
    | AT IDENTIFIER          { $$ = new Attribute($2, tokens::FLOAT, true); free(const_cast<char*>($2)); }
;

/// @brief  Syntax for supported external variable access
external:
      type DOLLAR IDENTIFIER  { $$ = new ExternalVariable($3, static_cast<tokens::CoreType>($1)); free(const_cast<char*>($3)); }
    | I_DOLLAR IDENTIFIER     { $$ = new ExternalVariable($2, tokens::INT); free(const_cast<char*>($2)); }
    | F_DOLLAR IDENTIFIER     { $$ = new ExternalVariable($2, tokens::FLOAT); free(const_cast<char*>($2)); }
    | V_DOLLAR IDENTIFIER     { $$ = new ExternalVariable($2, tokens::VEC3F); free(const_cast<char*>($2)); }
    | S_DOLLAR IDENTIFIER     { $$ = new ExternalVariable($2, tokens::STRING); free(const_cast<char*>($2)); }
    | DOLLAR IDENTIFIER       { $$ = new ExternalVariable($2, tokens::FLOAT); free(const_cast<char*>($2)); }
;

/// @brief  Syntax for text identifiers which resolves to a local. Types have
///         have their own tokens which do not evaluate to a local variable
/// @note   Anything which uses an IDENTIFIER must free the returned char array
local:
    IDENTIFIER  { $$ = new Local($1); free(const_cast<char*>($1)); }
;

/// @brief  Syntax numerical and boolean literal values
/// @note   Anything which uses one of the below tokens must free the returned char
///         array (aside from TRUE and FALSE tokens)
literal:
      L_SHORT         { $$ = new Value<int16_t>($1); free(const_cast<char*>($1)); }
    | L_INT           { $$ = new Value<int32_t>($1); free(const_cast<char*>($1)); }
    | L_LONG          { $$ = new Value<int64_t>($1); free(const_cast<char*>($1)); }
    | L_FLOAT         { $$ = new Value<float>($1); free(const_cast<char*>($1)); }
    | L_DOUBLE        { $$ = new Value<double>($1); free(const_cast<char*>($1)); }
    | L_STRING        { $$ = new Value<std::string>($1); free(const_cast<char*>($1)); }
    | TRUE            { $$ = new Value<bool>(true); }
    | FALSE           { $$ = new Value<bool>(false); }
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
    | SHORT   { $$ = tokens::SHORT; }
    | INT     { $$ = tokens::INT; }
    | LONG    { $$ = tokens::LONG; }
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

OPENVDB_NO_TYPE_CONVERSION_WARNING_END

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
