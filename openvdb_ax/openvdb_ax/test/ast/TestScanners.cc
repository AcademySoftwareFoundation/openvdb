// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "../util.h"

#include <openvdb_ax/ast/AST.h>
#include <openvdb_ax/ast/Scanners.h>

#include <cppunit/extensions/HelperMacros.h>

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

class TestScanners : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestScanners);
    CPPUNIT_TEST(testVisitNodeType);
    CPPUNIT_TEST(testFirstLastLocation);
    CPPUNIT_TEST(testAttributeDependencyTokens);
    // CPPUNIT_TEST(testVariableDependencies);
    CPPUNIT_TEST_SUITE_END();

    void testVisitNodeType();
    void testFirstLastLocation();
    void testAttributeDependencyTokens();
    // void testVariableDependencies();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestScanners);

void TestScanners::testVisitNodeType()
{
    size_t count = 0;
    auto counter = [&](const Node&) -> bool {
        ++count; return true;
    };

    // "int64@a;"
    Node::Ptr node(new Attribute("a", CoreType::INT64));

    visitNodeType<Node>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(0), count);

    count = 0;
    visitNodeType<Variable>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    count = 0;
    visitNodeType<Attribute>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    // "{1.0f, 2.0, 3};"
    node.reset(new ArrayPack( {
        new Value<float>(1.0f),
        new Value<double>(2.0),
        new Value<int64_t>(3)
    }));

    count = 0;
    visitNodeType<Node>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(4), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(0), count);

    count = 0;
    visitNodeType<ValueBase>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(3), count);

    count = 0;
    visitNodeType<ArrayPack>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    count = 0;
    visitNodeType<Expression>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(4), count);

    count = 0;
    visitNodeType<Statement>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(4), count);

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
    CPPUNIT_ASSERT_EQUAL(size_t(9), count);

    count = 0;
    visitNodeType<Local>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    count = 0;
    visitNodeType<Attribute>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(2), count);

    count = 0;
    visitNodeType<Value<int>>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(2), count);

    count = 0;
    visitNodeType<ArrayUnpack>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(1), count);

    count = 0;
    visitNodeType<AssignExpression>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(3), count);

    count = 0;
    visitNodeType<Expression>(*node, counter);
    CPPUNIT_ASSERT_EQUAL(size_t(9), count);
}

void TestScanners::testFirstLastLocation()
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
            CPPUNIT_ASSERT(tree);
            const Variable* first = firstUse(*tree, "@a");
            const Variable* last = lastUse(*tree, "@a");
            CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Unable to locate first @a AST node", code), first);
            CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Unable to locate last @a AST node", code), last);
            CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid first/last AST node comparison", code),
                first == last);
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
    CPPUNIT_ASSERT(expectedFirst != expectedLast);

    const Node* first = firstUse(*node, "@a");
    const Node* last = lastUse(*node, "@a");

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node", "@a=@a"),
        first, expectedFirst);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node", "@a=@a"),
        last, expectedLast);

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
    CPPUNIT_ASSERT(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "for(@a;@a;@a) { @a; }"), first, expectedFirst);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "for(@a;@a;@a) { @a; }"), last, expectedLast);

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
    CPPUNIT_ASSERT(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "do { @a; } while(@a);"), first, expectedFirst);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "do { @a; } while(@a);"), last, expectedLast);

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

    CPPUNIT_ASSERT(expectedFirst != expectedLast);

    first = firstUse(*node, "@a");
    last = lastUse(*node, "@a");

    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "if (@a) {} else if (1) {} else { @a; }"), first, expectedFirst);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Unexpected location of @a AST node",
        "if (@a) {} else if (1) {} else { @a; }"), last, expectedLast);
}

void TestScanners::testAttributeDependencyTokens()
{
    for (const std::string& code : none) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);

        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Expected 0 deps", code),
            dependencies.empty());
    }

    for (const std::string& code : self) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        CPPUNIT_ASSERT(!dependencies.empty());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            dependencies.front(), std::string("float@a"));
    }

    for (const std::string& code : direct) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        CPPUNIT_ASSERT(!dependencies.empty());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            dependencies.front(), std::string("float@b"));
    }

    for (const std::string& code : directvec) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        CPPUNIT_ASSERT(!dependencies.empty());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            dependencies.front(), std::string("vec3f@b"));
    }

    for (const std::string& code : indirect) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        std::vector<std::string> dependencies;
        attributeDependencyTokens(*tree, "a", tokens::CoreType::FLOAT, dependencies);
        CPPUNIT_ASSERT(!dependencies.empty());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            dependencies.front(), std::string("float@b"));
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
    CPPUNIT_ASSERT(tree);
    std::vector<std::string> dependencies;
    attributeDependencyTokens(*tree, "b", tokens::CoreType::FLOAT, dependencies);
    // @b should depend on: @a, @e, @f, v@v
    CPPUNIT_ASSERT_EQUAL(size_t(4), dependencies.size());
    CPPUNIT_ASSERT_EQUAL(dependencies[0], std::string("float@a"));
    CPPUNIT_ASSERT_EQUAL(dependencies[1], std::string("float@e"));
    CPPUNIT_ASSERT_EQUAL(dependencies[2], std::string("float@f"));
    CPPUNIT_ASSERT_EQUAL(dependencies[3], std::string("vec3f@v"));

    // @c should depend on: @a, @c, @d, @e, @f
    dependencies.clear();
    attributeDependencyTokens(*tree, "c", tokens::CoreType::FLOAT, dependencies);
    CPPUNIT_ASSERT_EQUAL(size_t(5), dependencies.size());
    CPPUNIT_ASSERT_EQUAL(dependencies[0], std::string("float@a"));
    CPPUNIT_ASSERT_EQUAL(dependencies[1], std::string("float@c"));
    CPPUNIT_ASSERT_EQUAL(dependencies[2], std::string("float@d"));
    CPPUNIT_ASSERT_EQUAL(dependencies[3], std::string("float@e"));
    CPPUNIT_ASSERT_EQUAL(dependencies[4], std::string("float@f"));


    // @d should depend on: @d, @e
    dependencies.clear();
    attributeDependencyTokens(*tree, "d", tokens::CoreType::FLOAT, dependencies);
    CPPUNIT_ASSERT_EQUAL(size_t(2), dependencies.size());
    CPPUNIT_ASSERT_EQUAL(dependencies[0], std::string("float@d"));
    CPPUNIT_ASSERT_EQUAL(dependencies[1], std::string("float@e"));

    // @e should depend on itself
    dependencies.clear();
    attributeDependencyTokens(*tree, "e", tokens::CoreType::FLOAT, dependencies);
    CPPUNIT_ASSERT_EQUAL(size_t(1), dependencies.size());
    CPPUNIT_ASSERT_EQUAL(dependencies[0], std::string("float@e"));

    // @f should depend on nothing
    dependencies.clear();
    attributeDependencyTokens(*tree, "f", tokens::CoreType::FLOAT, dependencies);
    CPPUNIT_ASSERT(dependencies.empty());

    // @v should depend on: v@v
    dependencies.clear();
    attributeDependencyTokens(*tree, "v", tokens::CoreType::VEC3F, dependencies);
    CPPUNIT_ASSERT_EQUAL(size_t(1), dependencies.size());
    CPPUNIT_ASSERT_EQUAL(dependencies[0], std::string("vec3f@v"));
}

/*
void TestScanners::testVariableDependencies()
{
    for (const std::string& code : none) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        const Variable* last = lastUse(*tree, "@a");
        CPPUNIT_ASSERT(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Expected 0 deps", code),
            vars.empty());
    }

    for (const std::string& code : self) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        const Variable* last = lastUse(*tree, "@a");
        CPPUNIT_ASSERT(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);
        const Variable* var = vars.front();
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            var->isType<Attribute>());
        const Attribute* attrib = static_cast<const Attribute*>(var);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            std::string("float@a"), attrib->tokenname());
    }

    for (const std::string& code : direct) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        const Variable* last = lastUse(*tree, "@a");
        CPPUNIT_ASSERT(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);

        const Variable* var = vars.front();
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            var->isType<Attribute>());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            std::string("b"), var->name());
    }

    for (const std::string& code : indirect) {
        const Tree::ConstPtr tree = parse(code.c_str());
        CPPUNIT_ASSERT(tree);
        const Variable* last = lastUse(*tree, "@a");
        CPPUNIT_ASSERT(last);

        std::vector<const Variable*> vars;
        variableDependencies(*last, vars);

        // check c
        const Variable* var = vars[0];
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            var->isType<Local>());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            std::string("c"), var->name());

        // check @b
        var = vars[1];
        CPPUNIT_ASSERT_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            var->isType<Attribute>());
        CPPUNIT_ASSERT_EQUAL_MESSAGE(ERROR_MSG("Invalid variable dependency", code),
            std::string("b"), var->name());
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
    CPPUNIT_ASSERT(tree);
    const Variable* lasta = lastUse(*tree, "@a");
    const Variable* lastb = lastUse(*tree, "@b");
    const Variable* lastc = lastUse(*tree, "@c");
    const Variable* lastd = lastUse(*tree, "@d");
    const Variable* laste = lastUse(*tree, "@e");
    const Variable* lastf = lastUse(*tree, "@f");
    const Variable* lastv = lastUse(*tree, "vec3f@v");
    CPPUNIT_ASSERT(lasta);
    CPPUNIT_ASSERT(lastb);
    CPPUNIT_ASSERT(lastc);
    CPPUNIT_ASSERT(lastd);
    CPPUNIT_ASSERT(laste);
    CPPUNIT_ASSERT(lastf);
    CPPUNIT_ASSERT(lastv);

    std::vector<const Variable*> vars;
    variableDependencies(*lasta, vars);
    CPPUNIT_ASSERT(vars.empty());

    // @b should depend on: m, m, v@v, @a, @e, @e, f1, f2, f3, @f
    variableDependencies(*lastb, vars);
    CPPUNIT_ASSERT_EQUAL(10ul, vars.size());
    CPPUNIT_ASSERT(vars[0]->isType<Local>());
    CPPUNIT_ASSERT(vars[0]->name() == "m");
    CPPUNIT_ASSERT(vars[1]->isType<Local>());
    CPPUNIT_ASSERT(vars[1]->name() == "m");
    CPPUNIT_ASSERT(vars[2]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[2])->tokenname() == "vec3f@v");
    CPPUNIT_ASSERT(vars[3]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[3])->tokenname() == "float@a");
    CPPUNIT_ASSERT(vars[4]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[4])->tokenname() == "float@e");
    CPPUNIT_ASSERT(vars[5]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[5])->tokenname() == "float@e");
    CPPUNIT_ASSERT(vars[6]->isType<Local>());
    CPPUNIT_ASSERT(vars[6]->name() == "f1");
    CPPUNIT_ASSERT(vars[7]->isType<Local>());
    CPPUNIT_ASSERT(vars[7]->name() == "f2");
    CPPUNIT_ASSERT(vars[8]->isType<Local>());
    CPPUNIT_ASSERT(vars[8]->name() == "f3");
    CPPUNIT_ASSERT(vars[9]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[9])->tokenname() == "float@f");

    // @c should depend on: @c, a, @e, @d, a, @e, @a, @e, f1, f2, f3, @f
    vars.clear();
    variableDependencies(*lastc, vars);
    CPPUNIT_ASSERT_EQUAL(11ul, vars.size());
    CPPUNIT_ASSERT(vars[0]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[0])->tokenname() == "float@c");
    CPPUNIT_ASSERT(vars[1]->isType<Local>());
    CPPUNIT_ASSERT(vars[1]->name() == "a");
    CPPUNIT_ASSERT(vars[2]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[2])->tokenname() == "float@e");
    CPPUNIT_ASSERT(vars[3]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[3])->tokenname() == "float@d");
    CPPUNIT_ASSERT(vars[4]->isType<Local>());
    CPPUNIT_ASSERT(vars[4]->name() == "a");
    CPPUNIT_ASSERT(vars[5]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[5])->tokenname() == "float@a");
    CPPUNIT_ASSERT(vars[6]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[6])->tokenname() == "float@e");
    CPPUNIT_ASSERT(vars[7]->isType<Local>());
    CPPUNIT_ASSERT(vars[7]->name() == "f1");
    CPPUNIT_ASSERT(vars[8]->isType<Local>());
    CPPUNIT_ASSERT(vars[8]->name() == "f2");
    CPPUNIT_ASSERT(vars[9]->isType<Local>());
    CPPUNIT_ASSERT(vars[9]->name() == "f3");
    CPPUNIT_ASSERT(vars[10]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[10])->tokenname() == "float@f");

    // @d should depend on: @d, a, @e
    vars.clear();
    variableDependencies(*lastd, vars);
    CPPUNIT_ASSERT_EQUAL(3ul, vars.size());
    CPPUNIT_ASSERT(vars[0]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[0])->tokenname() == "float@d");
    CPPUNIT_ASSERT(vars[1]->isType<Local>());
    CPPUNIT_ASSERT(vars[1]->name() == "a");
    CPPUNIT_ASSERT(vars[2]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[2])->tokenname() == "float@e");

    // @e should depend on itself
    vars.clear();
    variableDependencies(*laste, vars);
    CPPUNIT_ASSERT_EQUAL(1ul, vars.size());
    CPPUNIT_ASSERT(vars[0]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[0])->tokenname() == "float@e");

    // @f should depend on nothing
    vars.clear();
    variableDependencies(*lastf, vars);
    CPPUNIT_ASSERT(vars.empty());

    // @v should depend on: m, v@v
    vars.clear();
    variableDependencies(*lastv, vars);
    CPPUNIT_ASSERT_EQUAL(2ul, vars.size());
    CPPUNIT_ASSERT(vars[0]->isType<Local>());
    CPPUNIT_ASSERT(vars[0]->name() == "m");
    CPPUNIT_ASSERT(vars[1]->isType<Attribute>());
    CPPUNIT_ASSERT(static_cast<const Attribute*>(vars[1])->tokenname() == "vec3f@v");
}
*/

