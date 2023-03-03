// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/PointExecutable.h>
#include <openvdb_ax/util/x86.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointGroup.h>

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

using namespace openvdb;

class TestPointExecutable : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestPointExecutable);
    CPPUNIT_TEST(testConstructionDestruction);
    CPPUNIT_TEST(testCreateMissingAttributes);
    CPPUNIT_TEST(testGroupExecution);
    CPPUNIT_TEST(testCompilerCases);
    CPPUNIT_TEST(testExecuteBindings);
    CPPUNIT_TEST(testAttributeCodecs);
    CPPUNIT_TEST(testCLI);
    CPPUNIT_TEST_SUITE_END();

    void testConstructionDestruction();
    void testCreateMissingAttributes();
    void testGroupExecution();
    void testCompilerCases();
    void testExecuteBindings();
    void testAttributeCodecs();
    void testCLI();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointExecutable);

void
TestPointExecutable::testConstructionDestruction()
{
    // Test the building and teardown of executable objects. This is primarily to test
    // the destruction of Context and ExecutionEngine LLVM objects. These must be destructed
    // in the correct order (ExecutionEngine, then Context) otherwise LLVM will crash

    // must be initialized, otherwise construction/destruction of llvm objects won't
    // exhibit correct behaviour

    CPPUNIT_ASSERT(openvdb::ax::isInitialized());

    std::shared_ptr<llvm::LLVMContext> C(new llvm::LLVMContext);
    std::unique_ptr<llvm::Module> M(new llvm::Module("test_module", *C));
    std::shared_ptr<const llvm::ExecutionEngine> E(llvm::EngineBuilder(std::move(M))
            .setEngineKind(llvm::EngineKind::JIT)
            .create());

    CPPUNIT_ASSERT(!M);
    CPPUNIT_ASSERT(E);

    std::weak_ptr<llvm::LLVMContext> wC = C;
    std::weak_ptr<const llvm::ExecutionEngine> wE = E;

    // Basic construction

    openvdb::ax::ast::Tree tree;
    openvdb::ax::AttributeRegistry::ConstPtr emptyReg =
        openvdb::ax::AttributeRegistry::create(tree);
    openvdb::ax::PointExecutable::Ptr pointExecutable
        (new openvdb::ax::PointExecutable(C, E, emptyReg, nullptr, {}, tree));

    CPPUNIT_ASSERT_EQUAL(2, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(2, int(wC.use_count()));

    C.reset();
    E.reset();

    CPPUNIT_ASSERT_EQUAL(1, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(1, int(wC.use_count()));

    // test destruction

    pointExecutable.reset();

    CPPUNIT_ASSERT_EQUAL(0, int(wE.use_count()));
    CPPUNIT_ASSERT_EQUAL(0, int(wC.use_count()));
}

void
TestPointExecutable::testCreateMissingAttributes()
{
    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();

    const std::vector<openvdb::Vec3d> singlePointZero = {openvdb::Vec3d::zero()};
    openvdb::points::PointDataGrid::Ptr
        grid = openvdb::points::createPointDataGrid
            <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::PointExecutable::Ptr executable =
        compiler->compile<openvdb::ax::PointExecutable>("@a=v@b.x;");
    CPPUNIT_ASSERT(executable);

    executable->setCreateMissing(false);
    CPPUNIT_ASSERT_THROW(executable->execute(*grid), openvdb::AXExecutionError);

    executable->setCreateMissing(true);
    executable->execute(*grid);

    const auto leafIter = grid->tree().cbeginLeaf();
    const auto& descriptor = leafIter->attributeSet().descriptor();

    CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor.size());
    const size_t bIdx = descriptor.find("b");
    CPPUNIT_ASSERT(bIdx != openvdb::points::AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(descriptor.valueType(bIdx) == openvdb::typeNameAsString<openvdb::Vec3f>());
    openvdb::points::AttributeHandle<openvdb::Vec3f>::Ptr
        bHandle = openvdb::points::AttributeHandle<openvdb::Vec3f>::create(leafIter->constAttributeArray(bIdx));
    CPPUNIT_ASSERT(bHandle->get(0) == openvdb::Vec3f::zero());

    const size_t aIdx = descriptor.find("a");
    CPPUNIT_ASSERT(aIdx != openvdb::points::AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(descriptor.valueType(aIdx) == openvdb::typeNameAsString<float>());
    openvdb::points::AttributeHandle<float>::Ptr
        aHandle = openvdb::points::AttributeHandle<float>::create(leafIter->constAttributeArray(aIdx));
    CPPUNIT_ASSERT(aHandle->get(0) == 0.0f);
}

void
TestPointExecutable::testGroupExecution()
{
    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform(0.1);

    // 4 points in 4 leaf nodes
    const std::vector<openvdb::Vec3d> positions = {
        {0,0,0},
        {1,1,1},
        {2,2,2},
        {3,3,3},
    };

    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid
            <openvdb::points::NullCodec, openvdb::points::PointDataGrid>
                (positions, *defaultTransform);

    // check the values of "a"
    auto checkValues = [&](const int expected)
    {
        auto leafIter = grid->tree().cbeginLeaf();
        CPPUNIT_ASSERT(leafIter);

        const auto& descriptor = leafIter->attributeSet().descriptor();
        const size_t aIdx = descriptor.find("a");
        CPPUNIT_ASSERT(aIdx != openvdb::points::AttributeSet::INVALID_POS);

        for (; leafIter; ++leafIter) {
            openvdb::points::AttributeHandle<int> handle(leafIter->constAttributeArray(aIdx));
            CPPUNIT_ASSERT(handle.size() == 1);
            CPPUNIT_ASSERT_EQUAL(expected, handle.get(0));
        }
    };

    openvdb::points::appendAttribute<int>(grid->tree(), "a", 0);

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::PointExecutable::Ptr executable =
        compiler->compile<openvdb::ax::PointExecutable>("i@a=1;");
    CPPUNIT_ASSERT(executable);

    const std::string group = "test";

    // non existent group
    executable->setGroupExecution(group);
    CPPUNIT_ASSERT_THROW(executable->execute(*grid), openvdb::AXExecutionError);
    checkValues(0);

    openvdb::points::appendGroup(grid->tree(), group);

    // false group
    executable->execute(*grid);
    checkValues(0);

    openvdb::points::setGroup(grid->tree(), group, true);

    // true group
    executable->execute(*grid);
    checkValues(1);
}

void
TestPointExecutable::testCompilerCases()
{
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    CPPUNIT_ASSERT(compiler);
    {
        // with string only
        CPPUNIT_ASSERT(static_cast<bool>(compiler->compile<openvdb::ax::PointExecutable>("int i;")));
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::PointExecutable>("i;"), openvdb::AXCompilerError);
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::PointExecutable>("i"), openvdb::AXSyntaxError);
        // with AST only
        auto ast = openvdb::ax::ast::parse("i;");
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::PointExecutable>(*ast), openvdb::AXCompilerError);
    }

    openvdb::ax::Logger logger([](const std::string&) {});

    // using string and logger
    {
        openvdb::ax::PointExecutable::Ptr executable =
        compiler->compile<openvdb::ax::PointExecutable>("", logger); // empty
        CPPUNIT_ASSERT(executable);
    }
    logger.clear();
    {
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("i;", logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
        CPPUNIT_ASSERT(logger.hasError());
        logger.clear();
        openvdb::ax::PointExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::PointExecutable>("i", logger); // expected ; error (parser)
        CPPUNIT_ASSERT(!executable2);
        CPPUNIT_ASSERT(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("int i = 18446744073709551615;", logger); // warning
        CPPUNIT_ASSERT(executable);
        CPPUNIT_ASSERT(logger.hasWarning());
    }

    // using syntax tree and logger
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // empty
        CPPUNIT_ASSERT(executable);
        logger.clear(); // no tree for line col numbers
        openvdb::ax::PointExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // empty
        CPPUNIT_ASSERT(executable2);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
        CPPUNIT_ASSERT(logger.hasError());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::PointExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable2);
        CPPUNIT_ASSERT(logger.hasError());
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        CPPUNIT_ASSERT(tree);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // warning
        CPPUNIT_ASSERT(executable);
        CPPUNIT_ASSERT(logger.hasWarning());
        logger.clear(); // no tree for line col numbers
        openvdb::ax::PointExecutable::Ptr executable2 =
            compiler->compile<openvdb::ax::PointExecutable>(*tree, logger); // warning
        CPPUNIT_ASSERT(executable2);
        CPPUNIT_ASSERT(logger.hasWarning());
    }
    logger.clear();

    // with copied tree
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*copy, logger); // empty
        CPPUNIT_ASSERT(executable);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("i;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*copy, logger); // undeclared variable error
        CPPUNIT_ASSERT(!executable);
    }
    logger.clear();
    {
        openvdb::ax::ast::Tree::ConstPtr tree = openvdb::ax::ast::parse("int i = 18446744073709551615;", logger);
        std::unique_ptr<openvdb::ax::ast::Tree> copy(tree->copy());
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>(*copy, logger); // warning
        CPPUNIT_ASSERT(executable);
    }
    logger.clear();
}

void
TestPointExecutable::testExecuteBindings()
{
    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();
    const std::vector<openvdb::Vec3d> singlePointZero = {openvdb::Vec3d::zero()};

    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();

    // binding to different name existing attribute
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::points::appendAttribute<float>(points->tree(), "a");
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b", "a"); // bind @b to attribute a
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        const auto& descriptor = leafIter->attributeSet().descriptor();

        // check value set via binding is correct
        CPPUNIT_ASSERT_EQUAL(size_t(2), descriptor.size());
        const size_t aidx = descriptor.find("a");
        CPPUNIT_ASSERT(aidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(aidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle(leafIter->constAttributeArray(aidx));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));
    }

    // binding to existing attribute AND default bind other attribute
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::points::appendAttribute<float>(points->tree(), "a");
        openvdb::points::appendAttribute<float>(points->tree(), "c");
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f; @c = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        const auto& descriptor = leafIter->attributeSet().descriptor();

        // check value set via binding
        CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor.size());
        const size_t aidx = descriptor.find("a");
        CPPUNIT_ASSERT(aidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(aidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle(leafIter->constAttributeArray(aidx));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));

        // check value set not using binding
        const size_t cidx = descriptor.find("c");
        CPPUNIT_ASSERT(cidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(cidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle2(leafIter->constAttributeArray(cidx));
        CPPUNIT_ASSERT_EQUAL(2.0f, handle2.get(0));
    }

    // bind to created attribute AND not binding to created attribute
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f; @c = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b", "a"); // bind b to a
        executable->setAttributeBindings(bindings);
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        const auto& descriptor = leafIter->attributeSet().descriptor();

        // check value set via binding
        CPPUNIT_ASSERT_EQUAL(size_t(3), descriptor.size());
        const size_t aidx = descriptor.find("a");
        CPPUNIT_ASSERT(aidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(aidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle(leafIter->constAttributeArray(aidx));
        CPPUNIT_ASSERT_EQUAL(1.0f, handle.get(0));

        // check value set not using binding
        const size_t cidx = descriptor.find("c");
        CPPUNIT_ASSERT(cidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(cidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle2(leafIter->constAttributeArray(cidx));
        CPPUNIT_ASSERT_EQUAL(2.0f, handle2.get(0));
    }

    // binding to non existent attribute, error
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        executable->setAttributeBindings(bindings);
        executable->setCreateMissing(false);
        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        CPPUNIT_ASSERT_THROW(executable->execute(*points), openvdb::AXExecutionError);
    }

    // trying to bind to an attribute and use the original attribute name at same time
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f; @a = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        CPPUNIT_ASSERT_THROW(executable->setAttributeBindings(bindings), openvdb::AXExecutionError);
    }

    // swap ax and data attributes with bindings
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f; @a = 2.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        bindings.set("a","b"); // bind a to b

        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));
    }


    // bind P away from world space position to some other float attribute
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("f@P = 1.25f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("P","a"); // bind float a to P

        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        const auto& descriptor = leafIter->attributeSet().descriptor();

        // check value set via binding
        CPPUNIT_ASSERT_EQUAL(size_t(2), descriptor.size());
        const size_t aidx = descriptor.find("a");
        CPPUNIT_ASSERT(aidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(aidx) == openvdb::typeNameAsString<float>());
        openvdb::points::AttributeHandle<float> handle(leafIter->constAttributeArray(aidx));
        CPPUNIT_ASSERT_EQUAL(1.25f, handle.get(0));
    }

    // bind P away from world space position to some other attribute, defaulting to vec3f (as P does)
    {
        openvdb::points::PointDataGrid::Ptr
            points = openvdb::points::createPointDataGrid
                <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@P = 1.25f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("P","a"); // bind float a to P

        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        const auto& descriptor = leafIter->attributeSet().descriptor();

        // check value set via binding
        CPPUNIT_ASSERT_EQUAL(size_t(2), descriptor.size());
        const size_t aidx = descriptor.find("a");
        CPPUNIT_ASSERT(aidx != openvdb::points::AttributeSet::INVALID_POS);
        CPPUNIT_ASSERT(descriptor.valueType(aidx) == openvdb::typeNameAsString<openvdb::Vec3f>());
        openvdb::points::AttributeHandle<openvdb::Vec3f> handle(leafIter->constAttributeArray(aidx));
        CPPUNIT_ASSERT_EQUAL(openvdb::Vec3f(1.25f), handle.get(0));
    }

    // test setting bindings and then resetting some of those bindings on the same executable
    {
        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>("@b = 1.0f; @a = 2.0f; @c = 3.0f;");
        CPPUNIT_ASSERT(executable);
        openvdb::ax::AttributeBindings bindings;
        bindings.set("b","a"); // bind b to a
        bindings.set("c","b"); // bind c to b
        bindings.set("a","c"); // bind a to c
        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));

        bindings.set("a","b"); // bind a to b
        bindings.set("b","a"); // bind a to b
        CPPUNIT_ASSERT(!bindings.dataNameBoundTo("c")); // c should be unbound
        // check that the set call resets c to c
        CPPUNIT_ASSERT_NO_THROW(executable->setAttributeBindings(bindings));
        const openvdb::ax::AttributeBindings& bindingsOnExecutable = executable->getAttributeBindings();
        CPPUNIT_ASSERT(bindingsOnExecutable.isBoundAXName("c"));
        CPPUNIT_ASSERT_EQUAL(*bindingsOnExecutable.dataNameBoundTo("c"), std::string("c"));
    }
}

void
TestPointExecutable::testAttributeCodecs()
{
    math::Transform::Ptr defaultTransform =
        math::Transform::createLinearTransform(5.0f);
    const std::vector<Vec3d> twoPoints = {Vec3d::ones(), Vec3d::zero()};

    ax::Compiler::UniquePtr compiler = ax::Compiler::create();

    // test supported truncated codecs
    {
        points::PointDataGrid::Ptr
            points = points::createPointDataGrid
                <points::NullCodec, points::PointDataGrid>
                    (twoPoints, *defaultTransform);
        CPPUNIT_ASSERT_EQUAL(points->tree().leafCount(), Index32(1));

        // collapsed uniform 0 attributes
        points::appendAttribute<float, points::NullCodec>(points->tree(), "f");
        points::appendAttribute<float, points::TruncateCodec>(points->tree(), "t");
        points::appendAttribute<int32_t, points::NullCodec>(points->tree(), "i");
        points::appendAttribute<Vec3f, points::TruncateCodec>(points->tree(), "vu");
        points::appendAttribute<Vec3f, points::TruncateCodec>(points->tree(), "vnu");

        // assert the inputs are expected as we specifically test certain states
        auto leafIter = points->tree().beginLeaf();
        points::AttributeHandle<float> handle0(leafIter->constAttributeArray("f"));
        points::AttributeHandle<float> handle1(leafIter->constAttributeArray("t"));
        points::AttributeHandle<int32_t> handle2(leafIter->constAttributeArray("i"));
        points::AttributeHandle<Vec3f> handle3(leafIter->constAttributeArray("vu"));
        CPPUNIT_ASSERT(handle0.isUniform());
        CPPUNIT_ASSERT(handle1.isUniform());
        CPPUNIT_ASSERT(handle2.isUniform());
        CPPUNIT_ASSERT(handle3.isUniform());
        CPPUNIT_ASSERT_EQUAL(0.0f, handle0.get(0));
        CPPUNIT_ASSERT_EQUAL(float(math::half(0.0f)), handle1.get(0));
        CPPUNIT_ASSERT_EQUAL(int32_t(0), handle2.get(0));
        CPPUNIT_ASSERT_EQUAL(Vec3f(math::half(0)), handle3.get(0));

        // non uniform codec compressed inputs
        points::AttributeWriteHandle<Vec3f> handle4(leafIter->attributeArray("vnu"));
        handle4.set(0, Vec3f(1.0f));
        handle4.set(1, Vec3f(2.0f));
        CPPUNIT_ASSERT(!handle4.isUniform());

        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>
                ("if (v@P.x > 0.5) { @f = 3.245e-7f; }"
                 "else             { @f = 9.28e-12f; }"
                 "if (v@P.x > 0.5) { @t = 3.245e-7f; }"
                 "else             { @t = 0.0f; }"
                 "if (v@P.x > 0.5) { i@i = 3; }"
                 "if (v@P.x > 0.5) { v@vu[0]  = 3.245e-7f; v@vu[1]  = 100000.0f; v@vu[2]  = -1e-2f; }"
                 "else             { v@vu[0]  = 6.1e-3f;   v@vu[1]  = 0.0f;      v@vu[2]  = -9.367e-6f; }"
                 "if (v@P.x > 0.5) { v@vnu[0] = 7.135e-7f; v@vnu[1] = 200000.0f; v@vnu[2] = -5e-3f; }"
                 "else             { v@vnu[0] = -1.0f;     v@vnu[1] = 80123.14f; v@vnu[2] = 9019.53123f; }");

#if defined(__i386__) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(_M_X64)
    if (openvdb::ax::x86::CheckX86Feature("f16c") ==
        openvdb::ax::x86::CpuFlagStatus::Unsupported)
    {
        CPPUNIT_ASSERT(!executable->usesAcceleratedKernel(points->tree()));
    }
    else {
        CPPUNIT_ASSERT(executable->usesAcceleratedKernel(points->tree()));
    }
#else
        CPPUNIT_ASSERT(executable->usesAcceleratedKernel(points->tree()));
#endif

        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        CPPUNIT_ASSERT_EQUAL(3.245e-7f, handle0.get(0));
        CPPUNIT_ASSERT_EQUAL(9.28e-12f, handle0.get(1));
        CPPUNIT_ASSERT_EQUAL(float(math::half(3.245e-7f)), handle1.get(0));
        CPPUNIT_ASSERT_EQUAL(float(math::half(0.0f)), handle1.get(1));
        CPPUNIT_ASSERT_EQUAL(int32_t(3), handle2.get(0));
        CPPUNIT_ASSERT_EQUAL(int32_t(0), handle2.get(1));

        CPPUNIT_ASSERT_EQUAL(float(math::half(3.245e-7f)),  handle3.get(0).x());
        CPPUNIT_ASSERT_EQUAL(float(math::half(100000.0f)),  handle3.get(0).y());
        CPPUNIT_ASSERT_EQUAL(float(math::half(-1e-2f)),     handle3.get(0).z());
        CPPUNIT_ASSERT_EQUAL(float(math::half(6.1e-3f)),    handle3.get(1).x());
        CPPUNIT_ASSERT_EQUAL(float(math::half(0.0f)),       handle3.get(1).y());
        CPPUNIT_ASSERT_EQUAL(float(math::half(-9.367e-6f)), handle3.get(1).z());

        CPPUNIT_ASSERT_EQUAL(float(math::half(7.135e-7f)),   handle4.get(0).x());
        CPPUNIT_ASSERT_EQUAL(float(math::half(200000.0f)),   handle4.get(0).y());
        CPPUNIT_ASSERT_EQUAL(float(math::half(-5e-3f)),      handle4.get(0).z());
        CPPUNIT_ASSERT_EQUAL(float(math::half(-1.0f)),       handle4.get(1).x());
        CPPUNIT_ASSERT_EQUAL(float(math::half(80123.14f)),   handle4.get(1).y());
        CPPUNIT_ASSERT_EQUAL(float(math::half(9019.53123f)), handle4.get(1).z());
    }

    // compress/decompress val according to op and return it as the same type as val
    auto compress = [](const auto op, const auto val) {
        using InputT = decltype(val);
        typename decltype(op)::template Storage<InputT>::Type tmp;
        typename std::remove_const<InputT>::type out;
        op.encode(val, tmp);
        op.decode(tmp, out);
        return out;
    };

    // test supported fixed point codecs
    {
        points::PointDataGrid::Ptr
            points = points::createPointDataGrid
                <points::NullCodec, points::PointDataGrid>
                    (twoPoints, *defaultTransform);
        CPPUNIT_ASSERT_EQUAL(points->tree().leafCount(), Index32(1));

        // collapsed uniform 0 attributes
        points::appendAttribute<Vec3f, points::FixedPointCodec<true, points::UnitRange>>(points->tree(), "fpu8");
        points::appendAttribute<float, points::NullCodec>(points->tree(), "f");
        points::appendAttribute<Vec3f, points::FixedPointCodec<true, points::PositionRange>>(points->tree(), "fpr8");
        points::appendAttribute<Vec3f, points::FixedPointCodec<false, points::UnitRange>>(points->tree(), "fpu16");
        points::appendAttribute<Vec3f, points::FixedPointCodec<false, points::PositionRange>>(points->tree(), "fpr16");

        // assert the inputs are expected as we specifically test certain states
        auto leafIter = points->tree().beginLeaf();
        points::AttributeHandle<Vec3f> handle0(leafIter->constAttributeArray("fpu8"));
        points::AttributeHandle<float> handle1(leafIter->constAttributeArray("f"));
        points::AttributeHandle<Vec3f> handle2(leafIter->constAttributeArray("fpr8"));
        points::AttributeHandle<Vec3f> handle3(leafIter->constAttributeArray("fpu16"));
        CPPUNIT_ASSERT(handle0.isUniform());
        CPPUNIT_ASSERT(handle1.isUniform());
        CPPUNIT_ASSERT(handle2.isUniform());
        CPPUNIT_ASSERT(handle3.isUniform());

        const float fpr8zero = compress(points::FixedPointCodec<true, points::PositionRange>(), 0.0f);
        CPPUNIT_ASSERT_EQUAL(Vec3f(0.0f), handle0.get(0));
        CPPUNIT_ASSERT_EQUAL(float(0.0f), handle1.get(0));
        CPPUNIT_ASSERT_EQUAL(Vec3f(fpr8zero), handle2.get(0));
        CPPUNIT_ASSERT_EQUAL(Vec3f(0.0f), handle3.get(0));

        // non uniform codec compressed inputs
        points::AttributeWriteHandle<Vec3f> handle4(leafIter->attributeArray("fpr16"));
        handle4.set(0, Vec3f(0.49f));
        handle4.set(1, Vec3f(1e-9f));
        CPPUNIT_ASSERT(!handle4.isUniform());

        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>
                ("if (v@P.x > 0.5) { v@fpu8[0] = 0.924599f;  v@fpu8[1] = 0.0f;     v@fpu8[2] = -7e-2f; }"
                 "else             { v@fpu8[0] = 9.9e-9f;    v@fpu8[1] = -0.9999f; v@fpu8[2] = 7.2134e-4f; }"
                 "if (v@P.x > 0.5) { @f = 3.245e-7f; }"
                 "else             { @f = 0.0f; }"
                 "if (v@P.x > 0.5) { v@fpr8[0] = 3.245e-7f;  v@fpr8[1] = 0.0f;   v@fpr8[2] = -1e-12f; }"
                 "else             { v@fpr8[0] = -1.245e-9f; v@fpr8[1] = -0.49f; v@fpr8[2] = 0.078918f; }"
                 "if (v@P.x > 0.5) { v@fpu16[0] = 0.999999f; v@fpu16[1] = -0.0f;      v@fpu16[2] = 7.66e-2f; }"
                 "else             { v@fpu16[0] = 0.0f;      v@fpu16[1] = -0.999999f; v@fpu16[2] = 5.9811e-14f; }"
                 "if (v@P.x > 0.5) { v@fpr16[0] = 7.135e-7f; v@fpr16[1] = 200000.0f; v@fpr16[2] = -5e-3f; }"
                 "else             { v@fpr16[0] = -0.5f;     v@fpr16[1] = 0.0f;      v@fpr16[2] = 0.5f; }");

        CPPUNIT_ASSERT(executable->usesAcceleratedKernel(points->tree()));
        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));


        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), 0.924599f),  handle0.get(0).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), 0.0f),       handle0.get(0).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), -7e-2f),     handle0.get(0).z());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), 9.9e-9f),    handle0.get(1).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), -0.9999f),   handle0.get(1).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::UnitRange>(), 7.2134e-4f), handle0.get(1).z());

        CPPUNIT_ASSERT_EQUAL(float(3.245e-7f), handle1.get(0));
        CPPUNIT_ASSERT_EQUAL(float(0.0f),      handle1.get(1));

        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), 3.245e-7f),  handle2.get(0).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), 0.0f),       handle2.get(0).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), -1e-12f),    handle2.get(0).z());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), -1.245e-9f), handle2.get(1).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), -0.49f),     handle2.get(1).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), 0.078918f),  handle2.get(1).z());

        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), 0.999999f),   handle3.get(0).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), -0.0f),       handle3.get(0).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), 7.66e-2f),    handle3.get(0).z());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), 0.0f),        handle3.get(1).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), -0.999999f),  handle3.get(1).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::UnitRange>(), 5.9811e-14f), handle3.get(1).z());

        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), 7.135e-7f), handle4.get(0).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), 200000.0f), handle4.get(0).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), -5e-3f),   handle4.get(0).z());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), -0.5f),     handle4.get(1).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), 0.0f),      handle4.get(1).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<false, points::PositionRange>(), 0.5f),      handle4.get(1).z());
    }

    // finally test position (uint8_t compression) and different codecs together
    {
        points::PointDataGrid::Ptr
            points = points::createPointDataGrid
                <points::FixedPointCodec<true, points::PositionRange>, points::PointDataGrid>
                    (twoPoints, *defaultTransform);
        CPPUNIT_ASSERT_EQUAL(points->tree().leafCount(), Index32(1));

        points::appendAttribute<float, points::TruncateCodec>(points->tree(), "t");
        points::appendAttribute<Vec3f, points::FixedPointCodec<false, points::PositionRange>>(points->tree(), "f");

        openvdb::ax::PointExecutable::Ptr executable =
            compiler->compile<openvdb::ax::PointExecutable>
                ("@t = 8908410.12384910f;"
                 "vec3f@f = 245e-9f;"
                 "v@P.x += 1.0f;"
                 "v@P.y -= 1.0f;"
                 "v@P.z += 2.0f;");

#if defined(__i386__) || defined(_M_IX86) || \
    defined(__x86_64__) || defined(_M_X64)
    if (openvdb::ax::x86::CheckX86Feature("f16c") ==
        openvdb::ax::x86::CpuFlagStatus::Unsupported)
    {
        CPPUNIT_ASSERT(!executable->usesAcceleratedKernel(points->tree()));
    }
    else {
        CPPUNIT_ASSERT(executable->usesAcceleratedKernel(points->tree()));
    }
#else
        CPPUNIT_ASSERT(executable->usesAcceleratedKernel(points->tree()));
#endif

        CPPUNIT_ASSERT_NO_THROW(executable->execute(*points));

        const auto leafIter = points->tree().cbeginLeaf();
        points::AttributeHandle<Vec3f> handle0(leafIter->constAttributeArray("P"));
        points::AttributeHandle<float> handle1(leafIter->constAttributeArray("t"));
        points::AttributeHandle<Vec3f> handle2(leafIter->constAttributeArray("f"));

        Vec3f pos(compress(points::FixedPointCodec<true, points::PositionRange>(), 0.0f));
        pos.x() += 1.0f;
        pos.y() -= 1.0f;
        pos.z() += 2.0f;

        const math::Coord coord = leafIter->cbeginValueOn().getCoord();
        pos = Vec3f(defaultTransform->worldToIndex(pos));
        pos -= coord.asVec3s();

        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), pos.x()), handle0.get(0).x());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), pos.y()), handle0.get(0).y());
        CPPUNIT_ASSERT_EQUAL(compress(points::FixedPointCodec<true, points::PositionRange>(), pos.z()), handle0.get(0).z());

        CPPUNIT_ASSERT_EQUAL(float(math::half(8908410.12384910f)), handle1.get(0));
        CPPUNIT_ASSERT_EQUAL(Vec3f(compress(points::FixedPointCodec<false, points::PositionRange>(), 245e-9f)), handle2.get(0));
    }
}

void
TestPointExecutable::testCLI()
{
    using CLI = openvdb::ax::PointExecutable::CLI;

    struct UnusedCLIParam : public openvdb::Exception {
        UnusedCLIParam() noexcept: Exception( "UnusedCLIParam" ) {} \
        explicit UnusedCLIParam(const std::string& msg) noexcept: Exception( "UnusedCLIParam" , &msg) {}
    };

    auto CreateCLI = [](const char* c, bool throwIfUnused = true)
    {
        std::vector<std::string> strs;
        const char* s = c;
        while (*c != '\0') {
            if (*c == ' ') {
                strs.emplace_back(std::string(s, c-s));
                ++c;
                s = c;
            }
            else {
                ++c;
            }
        }
        if (*s != '\0') strs.emplace_back(std::string(s, c-s));

        std::vector<const char*> args;
        for (auto& str : strs) args.emplace_back(str.c_str());

        std::unique_ptr<bool[]> flags(new bool[args.size()]);
        std::fill(flags.get(), flags.get()+args.size(), false);

        auto cli = CLI::create(args.size(), args.data(), flags.get());
        if (throwIfUnused) {
            for (size_t i = 0; i < args.size(); ++i) {
                if (!flags[i]) OPENVDB_THROW(UnusedCLIParam, "unused param");
            }
        }
        return cli;
    };

    ax::Compiler::UniquePtr compiler = ax::Compiler::create();

    auto defaultExe = compiler->compile<openvdb::ax::PointExecutable>("");
    const auto defaultGroup = defaultExe->getGroupExecution();
    const auto defaultCreateMissing = defaultExe->getCreateMissing();
    const auto defaultGrain = defaultExe->getGrainSize();
    const auto defaultBindings = defaultExe->getAttributeBindings();

    CPPUNIT_ASSERT_THROW(CreateCLI("--unknown"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-unknown"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("--"), UnusedCLIParam);
    CPPUNIT_ASSERT_THROW(CreateCLI("-- "), UnusedCLIParam);

    {
        CLI cli = CreateCLI("");
        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(defaultGroup, exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --create-missing
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing invalid"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--create-missing --group test"), openvdb::CLIError);

        CLI cli = CreateCLI("--create-missing ON");
        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(true, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultGroup, exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --group
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--group"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--group --create-missing ON"), openvdb::CLIError);

        CLI cli = CreateCLI("--group test");
        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(std::string("test"), exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --grain
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--points-grain"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--points-grain nan"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--points-grain -1"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--points-grain --create-missing ON"), openvdb::CLIError);

        CLI cli = CreateCLI("--points-grain 0");
        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultGroup, exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(size_t(0), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    // --bindings
    {
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings :"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings ,"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings a:"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings a,b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings :b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings ,a:b"), openvdb::CLIError);
        CPPUNIT_ASSERT_THROW(CreateCLI("--bindings --create-missing ON"), openvdb::CLIError);

        CLI cli = CreateCLI("--bindings a:b,c:d,12:13");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");
        bindings.set("c", "d");
        bindings.set("12", "13");

        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(defaultCreateMissing, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultGroup, exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(defaultGrain, exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(bindings, exe->getAttributeBindings());
    }

    // multiple
    {
        CLI cli = CreateCLI("--points-grain 5 --create-missing OFF");
        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(false, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(defaultGroup, exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(size_t(5), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(defaultBindings, exe->getAttributeBindings());
    }

    {
        CLI cli = CreateCLI("--group 123 --points-grain 128 --create-missing OFF --bindings a:b");
        ax::AttributeBindings bindings;
        bindings.set("a", "b");

        auto exe = compiler->compile<openvdb::ax::PointExecutable>("");
        CPPUNIT_ASSERT_NO_THROW(exe->setSettingsFromCLI(cli));
        CPPUNIT_ASSERT_EQUAL(false, exe->getCreateMissing());
        CPPUNIT_ASSERT_EQUAL(std::string("123"), exe->getGroupExecution());
        CPPUNIT_ASSERT_EQUAL(size_t(128), exe->getGrainSize());
        CPPUNIT_ASSERT_EQUAL(bindings, exe->getAttributeBindings());
    }
}
