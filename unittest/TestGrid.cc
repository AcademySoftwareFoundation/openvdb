///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2013 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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

#include <cppunit/extensions/HelperMacros.h>
#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/util/Name.h>
#include <openvdb/math/Transform.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>


#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    CPPUNIT_ASSERT_DOUBLES_EQUAL((expected), (actual), /*tolerance=*/0.0);

class TestGrid: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestGrid);
    CPPUNIT_TEST(testGridRegistry);
    CPPUNIT_TEST(testConstPtr);
    CPPUNIT_TEST(testGetGrid);
    CPPUNIT_TEST(testIsType);
    CPPUNIT_TEST(testTransform);
    CPPUNIT_TEST(testCopyGrid);
    CPPUNIT_TEST_SUITE_END();

    void testGridRegistry();
    void testConstPtr();
    void testGetGrid();
    void testIsType();
    void testTransform();
    void testCopyGrid();
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestGrid);


////////////////////////////////////////


class ProxyTree: public openvdb::TreeBase
{
public:
    typedef int ValueType;
    typedef void ValueAllCIter;
    typedef void ValueAllIter;
    typedef void ValueOffCIter;
    typedef void ValueOffIter;
    typedef void ValueOnCIter;
    typedef void ValueOnIter;
    typedef openvdb::TreeBase::Ptr TreeBasePtr;
    typedef boost::shared_ptr<ProxyTree> Ptr;
    typedef boost::shared_ptr<const ProxyTree> ConstPtr;

    static const openvdb::Index DEPTH = 0;
    static const ValueType backg = 0;

    ProxyTree() {}
    ProxyTree(const ValueType&) {}
    virtual ~ProxyTree() {}

    static const openvdb::Name& treeType() { static const openvdb::Name s("proxy"); return s; }
    virtual const openvdb::Name& type() const { return treeType(); }
    virtual openvdb::Name valueType() const { return "proxy"; }
    const ValueType& background() const { return backg; }

    virtual TreeBasePtr copy() const { return TreeBasePtr(new ProxyTree(*this)); }

    virtual void readTopology(std::istream& is, bool = false) { is.seekg(0, std::ios::beg); }
    virtual void writeTopology(std::ostream& os, bool = false) const { os.seekp(0); }

    virtual void readBuffers(std::istream& is, bool /*saveFloatAsHalf*/=false) { is.seekg(0); }
    virtual void writeBuffers(std::ostream& os, bool /*saveFloatAsHalf*/=false) const
        { os.seekp(0, std::ios::beg); }

    bool empty() const { return true; }
    void clear() {}
    void prune(const ValueType& = 0) {}

    virtual void getIndexRange(openvdb::CoordBBox&) const {}
    virtual bool evalLeafBoundingBox(openvdb::CoordBBox& bbox) const
        { bbox.min() = bbox.max() = openvdb::Coord(0, 0, 0); return false; }
    virtual bool evalActiveVoxelBoundingBox(openvdb::CoordBBox& bbox) const
        { bbox.min() = bbox.max() = openvdb::Coord(0, 0, 0); return false; }
    virtual bool evalActiveVoxelDim(openvdb::Coord& dim) const
        { dim = openvdb::Coord(0, 0, 0); return false; }
    virtual bool evalLeafDim(openvdb::Coord& dim) const
        { dim = openvdb::Coord(0, 0, 0); return false; }

    virtual openvdb::Index treeDepth() const { return 0; }
    virtual openvdb::Index leafCount() const { return 0; }
    virtual openvdb::Index nonLeafCount() const { return 0; }
    virtual openvdb::Index64 activeVoxelCount() const { return 0UL; }
    virtual openvdb::Index64 inactiveVoxelCount() const { return 0UL; }
    virtual openvdb::Index64 activeLeafVoxelCount() const { return 0UL; }
    virtual openvdb::Index64 inactiveLeafVoxelCount() const { return 0UL; }
};

typedef openvdb::Grid<ProxyTree> ProxyGrid;


////////////////////////////////////////

void
TestGrid::testGridRegistry()
{
    using namespace openvdb::tree;

    typedef Tree<RootNode<InternalNode<LeafNode<float, 3>, 2> > > TreeType;
    typedef openvdb::Grid<TreeType> GridType;

    openvdb::GridBase::clearRegistry();

    CPPUNIT_ASSERT(!GridType::isRegistered());
    GridType::registerGrid();
    CPPUNIT_ASSERT(GridType::isRegistered());
    CPPUNIT_ASSERT_THROW(GridType::registerGrid(), openvdb::KeyError);
    GridType::unregisterGrid();
    CPPUNIT_ASSERT(!GridType::isRegistered());
    CPPUNIT_ASSERT_NO_THROW(GridType::unregisterGrid());
    CPPUNIT_ASSERT(!GridType::isRegistered());
    CPPUNIT_ASSERT_NO_THROW(GridType::registerGrid());
    CPPUNIT_ASSERT(GridType::isRegistered());

    openvdb::GridBase::clearRegistry();
}


void
TestGrid::testConstPtr()
{
    using namespace openvdb;

    GridBase::ConstPtr constgrid = ProxyGrid::create();

    CPPUNIT_ASSERT_EQUAL(Name("proxy"), constgrid->type());
}


void
TestGrid::testGetGrid()
{
    using namespace openvdb;

    GridBase::Ptr grid = FloatGrid::create(/*bg=*/0.0);
    GridBase::ConstPtr constGrid = grid;

    CPPUNIT_ASSERT(grid->baseTreePtr());

    CPPUNIT_ASSERT(!gridPtrCast<DoubleGrid>(grid));
    CPPUNIT_ASSERT(!gridPtrCast<DoubleGrid>(grid));

    CPPUNIT_ASSERT(gridConstPtrCast<FloatGrid>(constGrid));
    CPPUNIT_ASSERT(!gridConstPtrCast<DoubleGrid>(constGrid));
}


void
TestGrid::testIsType()
{
    using namespace openvdb;

    GridBase::Ptr grid = FloatGrid::create();
    CPPUNIT_ASSERT(grid->isType<FloatGrid>());
    CPPUNIT_ASSERT(!grid->isType<DoubleGrid>());
}


void
TestGrid::testTransform()
{
    ProxyGrid grid;

    // Verify that the grid has a valid default transform.
    CPPUNIT_ASSERT(grid.transformPtr());

    // Verify that a null transform pointer is not allowed.
    CPPUNIT_ASSERT_THROW(grid.setTransform(openvdb::math::Transform::Ptr()),
        openvdb::ValueError);

    grid.setTransform(openvdb::math::Transform::createLinearTransform());

    CPPUNIT_ASSERT(grid.transformPtr());

    // Verify that calling Transform-related Grid methods (Grid::voxelSize(), etc.)
    // is the same as calling those methods on the Transform.

    CPPUNIT_ASSERT(grid.transform().voxelSize().eq(grid.voxelSize()));
    CPPUNIT_ASSERT(grid.transform().voxelSize(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.voxelSize(openvdb::Vec3d(0.1, 0.2, 0.3))));

    CPPUNIT_ASSERT(grid.transform().indexToWorld(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.indexToWorld(openvdb::Vec3d(0.1, 0.2, 0.3))));
    CPPUNIT_ASSERT(grid.transform().indexToWorld(openvdb::Coord(1, 2, 3)).eq(
        grid.indexToWorld(openvdb::Coord(1, 2, 3))));
    CPPUNIT_ASSERT(grid.transform().worldToIndex(openvdb::Vec3d(0.1, 0.2, 0.3)).eq(
        grid.worldToIndex(openvdb::Vec3d(0.1, 0.2, 0.3))));
}


void
TestGrid::testCopyGrid()
{
    using namespace openvdb;

    // set up a grid
    const float fillValue1=5.0f;
    FloatGrid::Ptr grid1 = createGrid<FloatGrid>(/*bg=*/fillValue1);
    FloatTree& tree1 = grid1->tree();
    tree1.setValue(Coord(-10,40,845), 3.456f);
    tree1.setValue(Coord(1,-50,-8), 1.0f);

    // create a new grid, copying the first grid
    GridBase::Ptr grid2 = grid1->deepCopy();

    // cast down to the concrete type to query values
    FloatTree& tree2 = gridPtrCast<FloatGrid>(grid2)->tree();

    // compare topology
    CPPUNIT_ASSERT(tree1.hasSameTopology(tree2));
    CPPUNIT_ASSERT(tree2.hasSameTopology(tree1));

    // trees should be equal
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue1, tree2.getValue(Coord(1,2,3)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(3.456f, tree2.getValue(Coord(-10,40,845)));
    ASSERT_DOUBLES_EXACTLY_EQUAL(1.0f, tree2.getValue(Coord(1,-50,-8)));

    // change 1 value in tree2
    Coord changeCoord(1, -500, -8);
    tree2.setValue(changeCoord, 1.0f);

    // topology should no longer match
    CPPUNIT_ASSERT(!tree1.hasSameTopology(tree2));
    CPPUNIT_ASSERT(!tree2.hasSameTopology(tree1));

    // query changed value and make sure it's different between trees
    ASSERT_DOUBLES_EXACTLY_EQUAL(fillValue1, tree1.getValue(changeCoord));
    ASSERT_DOUBLES_EXACTLY_EQUAL(1.0f, tree2.getValue(changeCoord));
}

// Copyright (c) 2012-2013 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
