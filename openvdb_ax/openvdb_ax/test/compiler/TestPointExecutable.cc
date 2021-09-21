// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb_ax/compiler/Compiler.h>
#include <openvdb_ax/compiler/PointExecutable.h>

#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointGroup.h>

#include <cppunit/extensions/HelperMacros.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>

class TestPointExecutable : public CppUnit::TestCase
{
public:

    CPPUNIT_TEST_SUITE(TestPointExecutable);
    CPPUNIT_TEST(testConstructionDestruction);
    CPPUNIT_TEST(testCreateMissingAttributes);
    CPPUNIT_TEST(testGroupExecution);
    CPPUNIT_TEST(testCompilerCases);
    CPPUNIT_TEST(testExecuteBindings);
    CPPUNIT_TEST_SUITE_END();

    void testConstructionDestruction();
    void testCreateMissingAttributes();
    void testGroupExecution();
    void testCompilerCases();
    void testExecuteBindings();
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
        CPPUNIT_ASSERT_THROW(compiler->compile<openvdb::ax::PointExecutable>("i"), openvdb::AXCompilerError);
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
}
