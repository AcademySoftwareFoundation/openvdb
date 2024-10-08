// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "../util.h"

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>

#include <gtest/gtest.h>

#include <string>

using namespace openvdb::ax::ast;
using namespace openvdb::ax::ast::tokens;

namespace {

// No dependencies
// - use @a once (read), no other variables
const std::vector<std::string> none = {
    "@a;",
    "@a+1;",
    "@a=1;",
    "@a=func(5);",
    "-@a;",
    "@a[0]=1;",
    "if(true) @a = 1;",
    "if(@a) a=1;",
    "if (@e) if (@b) if (@c) @a;",
    "@b = c ? : @a;",
    "@a ? : c = 1;",
    "for (@a; @b; @c) ;",
    "while (@a) ;",
    "for(;true;) @a = 1;"
};

// Self dependencies
// - use @a potentially multiple times (read/write), no other variables
const std::vector<std::string> self = {
    "@a=@a;",
    "@a+=1;",
    "++@a;",
    "@a--;",
    "func(@a);",
    "--@a + 1;",
    "if(@a) @a = 1;",
    "if(@a) ; else @a = 1;",
    "@a ? : @a = 2;",
    "for (@b;@a;@c) @a = 0;",
    "while(@a) @a = 1;"
};

// Code where @a should have a direct dependency on @b only
// - use @a once (read/write), use @b once
const std::vector<std::string> direct = {
    "@a=@b;",
    "@a=-@b;",
    "@a=@b;",
    "@a=1+@b;",
    "@a=func(@b);",
    "@a=++@b;",
    "if(@b) @a=1;",
    "if(@b) {} else { @a=1; }",
    "@b ? @a = 1 : 0;",
    "@b ? : @a = 1;",
    "for (;@b;) @a = 1;",
    "while (@b) @a = 1;"
};

// Code where @a should have a direct dependency on @b only
// - use @a once (read/write), use @b once, b a vector
const std::vector<std::string> directvec = {
    "@a=v@b.x;",
    "@a=v@b[0];",
    "@a=v@b.x + 1;",
    "@a=v@b[0] * 3;",
    "if (v@b[0]) @a = 3;",
};

// Code where @a should have dependencies on @b and c (c always first)
const std::vector<std::string> indirect = {
    "c=@b; @a=c;",
    "c=@b; @a=c[0];",
    "c = {@b,1,2}; @a=c;",
    "int c=@b; @a=c;",
    "int c; c=@b; @a=c;",
    "if (c = @b) @a = c;",
    "(c = @b) ? @a=c : 0;",
    "(c = @b) ? : @a=c;",
    "for(int c = @b; true; e) @a = c;",
    "for(int c = @b;; @a = c) ;",
    "for(int c = @b; c; e) @a = c;",
    "for(c = @b; c; e) @a = c;",
    "int c; for(c = @b; c; e) @a = c;",
    "for(; c = @b;) @a = c;",
    "while(int c = @b) @a = c;",
};

}

class TestScanners : public ::testing::Test
{
};

TEST_F(TestScanners, testVisitNodeType)
{
    size_t count = 0;
    auto counter = [&](const Node&) -> bool {
        ++count; return true;
    };

    // "int64@a;"
    Node::Ptr node(new Attribute("a", CoreType::INT64));

    visitNodeType<Node>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    ASSERT_EQ(size_t(0), count);

    count = 0;
    visitNodeType<Variable>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    count = 0;
    visitNodeType<Attribute>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    // "{1.0f, 2.0, 3};"
    node.reset(new ArrayPack( {
        new Value<float>(1.0f),
        new Value<double>(2.0),
        new Value<int64_t>(3)
    }));

    count = 0;
    visitNodeType<Node>(*node, counter);
    ASSERT_EQ(size_t(4), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    ASSERT_EQ(size_t(0), count);

    count = 0;
    visitNodeType<ValueBase>(*node, counter);
    ASSERT_EQ(size_t(3), count);

    count = 0;
    visitNodeType<ArrayPack>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    count = 0;
    visitNodeType<Expression>(*node, counter);
    ASSERT_EQ(size_t(4), count);

    count = 0;
    visitNodeType<Statement>(*node, counter);
    ASSERT_EQ(size_t(4), count);

    // "@a += v@b.x = x %= 1;"
    // @note 9 explicit nodes
    node.reset(new AssignExpression(
        new Attribute("a", CoreType::FLOAT, true),
        new AssignExpression(
            new ArrayUnpack(
                new Attribute("b", CoreType::VEC3F),
                new Value<int32_t>(0)
            ),
            new AssignExpression(
                new Local("x"),
                new Value<int32_t>(1),
                OperatorToken::MODULO
                )
            ),
        OperatorToken::PLUS
    ));

    count = 0;
    visitNodeType<Node>(*node, counter);
    ASSERT_EQ(size_t(9), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    count = 0;
    visitNodeType<Attribute>(*node, counter);
    ASSERT_EQ(size_t(2), count);

    count = 0;
    visitNodeType<Value<int>>(*node, counter);
    ASSERT_EQ(size_t(2), count);

    count = 0;
    visitNodeType<ArrayUnpack>(*node, counter);
    ASSERT_EQ(size_t(1), count);

    count = 0;
    visitNodeType<AssignExpression>(*node, counter);
    ASSERT_EQ(size_t(3), count);

    count = 0;
    visitNodeType<Expression>(*node, counter);
    ASSERT_EQ(size_t(9), count);
}

TEST_F(TestScanners, testFirstLastLocation)
{
    // The list of above code sets which are expected to have the same
    // first and last use of @a.
    const std::vector<const std::vector<std::string>*> snippets {
        &none,
        &direct,
        &indirect
    };

    for (const auto& samples : snippets) {
        for (const std::string& code : *samples) {
            const Tree::ConstPtr tree = parse(code.c_str());
            ASSERT_TRUE(tree);
            const Variable* first = firstUse(*tree, "@a");
            const Variable* last = lastUse(*tree, "@a");
            ASSERT_TRUE(first) << ERROR_MSG("Unable to locate first @a AST node", code);
            ASSERT_TRUE(last) << ERROR_MSG("Unable to locate last @a AST node", code);
            ASSERT_TRUE(first == last) << ERROR_MSG("Invalid first/last AST node comparison", code);
        }
    }

    // Test some common edge cases

    // @a = @a;
    Node::Ptr node(new AssignExpression(
        new Attribute("a", CoreType::FLOAT),
        new Attribute("a", CoreType::FLOAT)));

    const Node* expectedFirst =
        static_cast<AssignExpression*>(node.get())->lhs();
    const Node* expectedLast =
        static_cast<AssignExpression*>(node.get())->rhs();
    ASSERT_TRUE(expectedFirst != expectedLast);

    const Node* first = firstUse(*node, "@a");
    const Node* last = lastUse(*node, "@a");

    ASSERT_EQ(first, expectedFirst) << ERROR_MSG("Unexpected location of @a AST node", "@a=@a");
    ASSERT_EQ(last, expectedLast) << ERROR_MSG("Unexpected location of @a AST node", "@a=@a");

    // for(@a;@a;@a) { @a; }
    node.reset(new Loop(
        tokens::FOR,
        new Attribute("a", CoreType::FLOAT),
        new Block(new Attribute("a", CoreType::FLOAT)),
        new Attribute("a", CoreType::FLOAT),
        new Attribute("a", CoreType::FLOAT)
        ));

    expectedFirst = static_cast<Loop*>(node.get())->initial();
    expectedLast = static_cast<Loop*>(node.get())->body()->child(0);
    ASSERT_TRUE(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    ASSERT_EQ(first, expectedFirst) << ERROR_MSG("Unexpected location of @a AST node", "for(@a;@a;@a) { @a; }");
    ASSERT_EQ(last, expectedLast) << ERROR_MSG("Unexpected location of @a AST node", "for(@a;@a;@a) { @a; }");

    // do { @a; } while(@a);
    node.reset(new Loop(
        tokens::DO,
        new Attribute("a", CoreType::FLOAT),
        new Block(new Attribute("a", CoreType::FLOAT)),
        nullptr,
        nullptr
        ));

    expectedFirst = static_cast<Loop*>(node.get())->body()->child(0);
    expectedLast = static_cast<Loop*>(node.get())->condition();
    ASSERT_TRUE(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    ASSERT_EQ(first, expectedFirst) << ERROR_MSG("Unexpected location of @a AST node", "do { @a; } while(@a);");
    ASSERT_EQ(last, expectedLast) << ERROR_MSG("Unexpected location of @a AST node", "do { @a; } while(@a);");

    // if (@a) {} else if (@a) {} else { @a; }
    node.reset(new ConditionalStatement(
        new Attribute("a", CoreType::FLOAT),
        new Block(),
        new Block(
            new ConditionalStatement(
                new Attribute("a", CoreType::FLOAT),
                new Block(),
                new Block(new Attribute("a", CoreType::FLOAT))
                )
            )
        ));

    expectedFirst = static_cast<ConditionalStatement*>(node.get())->condition();
    expectedLast =
        static_cast<const ConditionalStatement*>(
            static_cast<ConditionalStatement*>(node.get())
                ->falseBranch()->child(0))
                    ->falseBranch()->child(0);

    ASSERT_TRUE(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    ASSERT_EQ(first, expectedFirst) << ERROR_MSG("Unexpected location of @a AST node", "if (@a) {} else if (1) {} else { @a; }");
    ASSERT_EQ(last, expectedLast) << ERROR_MSG("Unexpected location of @a AST node", "if (@a) {} else if (1) {} else { @a; }");
}

TEST_F(TestScanners, testAttributeDependencyTokens)
{
    for (const std::string& code : none) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);

        ASSERT_TRUE(dependencies.empty()) << ERROR_MSG("Expected 0 deps", code);
    }

    for (const std::string& code : self) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        ASSERT_TRUE(!dependencies.empty());
        ASSERT_EQ(dependencies.front(), std::string("float@a")) << ERROR_MSG("Invalid variable dependency", code);
    }

    for (const std::string& code : direct) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        ASSERT_TRUE(!dependencies.empty());
        ASSERT_EQ(dependencies.front(), std::string("float@b")) << ERROR_MSG("Invalid variable dependency", code);
    }

    for (const std::string& code : directvec) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        ASSERT_TRUE(!dependencies.empty());
        ASSERT_EQ(dependencies.front(), std::string("vec3f@b")) << ERROR_MSG("Invalid variable dependency", code);
    }

    for (const std::string& code : indirect) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        ASSERT_TRUE(!dependencies.empty());
        ASSERT_EQ(dependencies.front(), std::string("float@b")) << ERROR_MSG("Invalid variable dependency", code);
    }

    // Test a more complicated code snippet. Note that this also checks the
    // order which isn't strictly necessary

    const std::string complex =
        "int a = func(1,@e);"
        "pow(@d, a);"
        "mat3f m = 0;"
        "scale(m, v@v);"
        ""
        "float f1 = 0;"
        "float f2 = 0;"
        "float f3 = 0;"
        ""
        "f3 = @f;"
        "f2 = f3;"
        "f1 = f2;"
        "if (@a - @e > f1) {"
        "    @b = func(m);"
        "    if (true) {"
        "        ++@c[0] = a;"
        "    }"
        "}";

    const Tree::ConstPtr tree = parse(complex.c_str());
    ASSERT_TRUE(tree);
    std::vector<std::string> dependencies;
    attributeDependencyTokens(*tree, "b", tokens::CoreType::FLOAT, dependencies);
    // @b should depend on: @a, @e, @f, v@v
    ASSERT_EQ(size_t(4), dependencies.size());
    ASSERT_EQ(dependencies[0], std::string("float@a"));
    ASSERT_EQ(dependencies[1], std::string("float@e"));
    ASSERT_EQ(dependencies[2], std::string("float@f"));
    ASSERT_EQ(dependencies[3], std::string("vec3f@v"));

    // @c should depend on: @a, @c, @d, @e, @f
    dependencies.clear();
    attributeDependencyTokens(*tree, "c", tokens::CoreType::FLOAT, dependencies);
    ASSERT_EQ(size_t(5), dependencies.size());
    ASSERT_EQ(dependencies[0], std::string("float@a"));
    ASSERT_EQ(dependencies[1], std::string("float@c"));
    ASSERT_EQ(dependencies[2], std::string("float@d"));
    ASSERT_EQ(dependencies[3], std::string("float@e"));
    ASSERT_EQ(dependencies[4], std::string("float@f"));


    // @d should depend on: @d, @e
    dependencies.clear();
    attributeDependencyTokens(*tree, "d", tokens::CoreType::FLOAT, dependencies);
    ASSERT_EQ(size_t(2), dependencies.size());
    ASSERT_EQ(dependencies[0], std::string("float@d"));
    ASSERT_EQ(dependencies[1], std::string("float@e"));

    // @e should depend on itself
    dependencies.clear();
    attributeDependencyTokens(*tree, "e", tokens::CoreType::FLOAT, dependencies);
    ASSERT_EQ(size_t(1), dependencies.size());
    ASSERT_EQ(dependencies[0], std::string("float@e"));

    // @f should depend on nothing
    dependencies.clear();
    attributeDependencyTokens(*tree, "f", tokens::CoreType::FLOAT, dependencies);
    ASSERT_TRUE(dependencies.empty());

    // @v should depend on: v@v
    dependencies.clear();
    attributeDependencyTokens(*tree, "v", tokens::CoreType::VEC3F, dependencies);
    ASSERT_EQ(size_t(1), dependencies.size());
    ASSERT_EQ(dependencies[0], std::string("vec3f@v"));
}

/*
TEST_F(TestScanners, testVariableDependencies)
{
    for (const std::string& code : none) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        const Variable* last = lastUse(*tree, "@a");
        ASSERT_TRUE(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);
        ASSERT_TRUE(vars.empty()) << ERROR_MSG("Expected 0 deps", code);
    }

    for (const std::string& code : self) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        const Variable* last = lastUse(*tree, "@a");
        ASSERT_TRUE(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);
        const Variable* var = vars.front();
        ASSERT_TRUE(var->isType<Attribute>()) << ERROR_MSG("Invalid variable dependency", code);
        const Attribute* attrib = static_cast<const Attribute*>(var);
        ASSERT_EQ(std::string("float@a"), attrib->tokenname()) << ERROR_MSG("Invalid variable dependency", code);
    }

    for (const std::string& code : direct) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        const Variable* last = lastUse(*tree, "@a");
        ASSERT_TRUE(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);

        const Variable* var = vars.front();
        ASSERT_TRUE(var->isType<Attribute>()) << ERROR_MSG("Invalid variable dependency", code);
        ASSERT_EQ(std::string("b"), var->name()) << ERROR_MSG("Invalid variable dependency", code);
    }

    for (const std::string& code : indirect) {
        const Tree::ConstPtr tree = parse(code.c_str());
        ASSERT_TRUE(tree);
        const Variable* last = lastUse(*tree, "@a");
        ASSERT_TRUE(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);

        // check c
        const Variable* var = vars[0];
        ASSERT_TRUE(var->isType<Local>()) << ERROR_MSG("Invalid variable dependency", code);
        ASSERT_EQ(std::string("c"), var->name()) << ERROR_MSG("Invalid variable dependency", code);

        // check @b
        var = vars[1];
        ASSERT_TRUE(var->isType<Attribute>()) << ERROR_MSG("Invalid variable dependency", code);
        ASSERT_EQ(std::string("b"), var->name()) << ERROR_MSG("Invalid variable dependency", code);
    }

    // Test a more complicated code snippet. Note that this also checks the
    // order which isn't strictly necessary

    const std::string complex =
        "int a = func(1,@e);"
        "pow(@d, a);"
        "mat3f m = 0;"
        "scale(m, v@v);"
        ""
        "float f1 = 0;"
        "float f2 = 0;"
        "float f3 = 0;"
        ""
        "f3 = @f;"
        "f2 = f3;"
        "f1 = f2;"
        "if (@a - @e > f1) {"
        "    @b = func(m);"
        "    if (true) {"
        "        ++@c[0] = a;"
        "    }"
        "}";

    const Tree::ConstPtr tree = parse(complex.c_str());
    ASSERT_TRUE(tree);
    const Variable* lasta = lastUse(*tree, "@a");
    const Variable* lastb = lastUse(*tree, "@b");
    const Variable* lastc = lastUse(*tree, "@c");
    const Variable* lastd = lastUse(*tree, "@d");
    const Variable* laste = lastUse(*tree, "@e");
    const Variable* lastf = lastUse(*tree, "@f");
    const Variable* lastv = lastUse(*tree, "vec3f@v");
    ASSERT_TRUE(lasta);
    ASSERT_TRUE(lastb);
    ASSERT_TRUE(lastc);
    ASSERT_TRUE(lastd);
    ASSERT_TRUE(laste);
    ASSERT_TRUE(lastf);
    ASSERT_TRUE(lastv);

    std::vector<const Variable*> vars;
    variableDependencies(*lasta, vars);
    ASSERT_TRUE(vars.empty());

    // @b should depend on: m, m, v@v, @a, @e, @e, f1, f2, f3, @f
    variableDependencies(*lastb, vars);
    ASSERT_EQ(10ul, vars.size());
    ASSERT_TRUE(vars[0]->isType<Local>());
    ASSERT_TRUE(vars[0]->name() == "m");
    ASSERT_TRUE(vars[1]->isType<Local>());
    ASSERT_TRUE(vars[1]->name() == "m");
    ASSERT_TRUE(vars[2]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[2])->tokenname() == "vec3f@v");
    ASSERT_TRUE(vars[3]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[3])->tokenname() == "float@a");
    ASSERT_TRUE(vars[4]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[4])->tokenname() == "float@e");
    ASSERT_TRUE(vars[5]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[5])->tokenname() == "float@e");
    ASSERT_TRUE(vars[6]->isType<Local>());
    ASSERT_TRUE(vars[6]->name() == "f1");
    ASSERT_TRUE(vars[7]->isType<Local>());
    ASSERT_TRUE(vars[7]->name() == "f2");
    ASSERT_TRUE(vars[8]->isType<Local>());
    ASSERT_TRUE(vars[8]->name() == "f3");
    ASSERT_TRUE(vars[9]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[9])->tokenname() == "float@f");

    // @c should depend on: @c, a, @e, @d, a, @e, @a, @e, f1, f2, f3, @f
    vars.clear();
    variableDependencies(*lastc, vars);
    ASSERT_EQ(11ul, vars.size());
    ASSERT_TRUE(vars[0]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[0])->tokenname() == "float@c");
    ASSERT_TRUE(vars[1]->isType<Local>());
    ASSERT_TRUE(vars[1]->name() == "a");
    ASSERT_TRUE(vars[2]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[2])->tokenname() == "float@e");
    ASSERT_TRUE(vars[3]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[3])->tokenname() == "float@d");
    ASSERT_TRUE(vars[4]->isType<Local>());
    ASSERT_TRUE(vars[4]->name() == "a");
    ASSERT_TRUE(vars[5]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[5])->tokenname() == "float@a");
    ASSERT_TRUE(vars[6]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[6])->tokenname() == "float@e");
    ASSERT_TRUE(vars[7]->isType<Local>());
    ASSERT_TRUE(vars[7]->name() == "f1");
    ASSERT_TRUE(vars[8]->isType<Local>());
    ASSERT_TRUE(vars[8]->name() == "f2");
    ASSERT_TRUE(vars[9]->isType<Local>());
    ASSERT_TRUE(vars[9]->name() == "f3");
    ASSERT_TRUE(vars[10]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[10])->tokenname() == "float@f");

    // @d should depend on: @d, a, @e
    vars.clear();
    variableDependencies(*lastd, vars);
    ASSERT_EQ(3ul, vars.size());
    ASSERT_TRUE(vars[0]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[0])->tokenname() == "float@d");
    ASSERT_TRUE(vars[1]->isType<Local>());
    ASSERT_TRUE(vars[1]->name() == "a");
    ASSERT_TRUE(vars[2]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[2])->tokenname() == "float@e");

    // @e should depend on itself
    vars.clear();
    variableDependencies(*laste, vars);
    ASSERT_EQ(1ul, vars.size());
    ASSERT_TRUE(vars[0]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[0])->tokenname() == "float@e");

    // @f should depend on nothing
    vars.clear();
    variableDependencies(*lastf, vars);
    ASSERT_TRUE(vars.empty());

    // @v should depend on: m, v@v
    vars.clear();
    variableDependencies(*lastv, vars);
    ASSERT_EQ(2ul, vars.size());
    ASSERT_TRUE(vars[0]->isType<Local>());
    ASSERT_TRUE(vars[0]->name() == "m");
    ASSERT_TRUE(vars[1]->isType<Attribute>());
    ASSERT_TRUE(static_cast<const Attribute*>(vars[1])->tokenname() == "vec3f@v");
}
*/

