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
    CPPUNIT_TEST_SUITE_END();

    void testConstructionDestruction();
    void testCreateMissingAttributes();
    void testGroupExecution();
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
        (new openvdb::ax::PointExecutable(E, C, emptyReg, nullptr, {}));

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
    openvdb::ax::Compiler::UniquePtr compiler = openvdb::ax::Compiler::create();
    openvdb::ax::PointExecutable::Ptr executable =
        compiler->compile<openvdb::ax::PointExecutable>("@a=v@b.x;");

    openvdb::math::Transform::Ptr defaultTransform =
        openvdb::math::Transform::createLinearTransform();

    const std::vector<openvdb::Vec3d> singlePointZero = {openvdb::Vec3d::zero()};
    openvdb::points::PointDataGrid::Ptr
        grid = openvdb::points::createPointDataGrid
            <openvdb::points::NullCodec, openvdb::points::PointDataGrid>(singlePointZero, *defaultTransform);
    CPPUNIT_ASSERT_THROW(executable->execute(*grid, nullptr, false),
        openvdb::LookupError);

    executable->execute(*grid, nullptr, true);

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

    const std::string group = "test";

    // non existent group
    CPPUNIT_ASSERT_THROW(executable->execute(*grid, &group), openvdb::LookupError);
    checkValues(0);

    openvdb::points::appendGroup(grid->tree(), group);

    // false group
    executable->execute(*grid, &group);
    checkValues(0);

    openvdb::points::setGroup(grid->tree(), group, true);

    // true group
    executable->execute(*grid, &group);
    checkValues(1);
}

// Copyright (c) 2015-2020 DNEG
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
