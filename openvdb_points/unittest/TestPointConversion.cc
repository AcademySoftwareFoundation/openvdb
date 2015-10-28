///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
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


#include <cppunit/extensions/HelperMacros.h>

#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointAttribute.h>
#include <openvdb_points/tools/PointConversion.h>
#include <openvdb_points/openvdb.h>

#include <boost/ptr_container/ptr_vector.hpp>

using namespace openvdb;
using namespace openvdb::tools;

class TestPointConversion: public CppUnit::TestCase
{
public:

    virtual void setUp() { openvdb::initialize(); openvdb::points::initialize(); }
    virtual void tearDown() { openvdb::uninitialize(); openvdb::points::uninitialize(); }

    CPPUNIT_TEST_SUITE(TestPointConversion);
    CPPUNIT_TEST(testPointConversion);

    CPPUNIT_TEST_SUITE_END();

    void testPointConversion();

}; // class TestPointConversion

CPPUNIT_TEST_SUITE_REGISTRATION(TestPointConversion);

// simple data structure for storing a point with position and id

struct Point
{
    Point(const Vec3f pos, const int id)
        : position(pos), id(id) { }

    Vec3f position;
    int id;
};

// sort points by id

inline bool operator<(const Point& lhs, const Point& rhs) {
    return lhs.id < rhs.id;
}

typedef boost::ptr_vector<Point> PointData;


// wrapper to retrieve position
class PointPosition
{
public:
    typedef openvdb::Vec3f value_type;

    PointPosition(const PointData& pointData)
        : mPointData(pointData) { }

    size_t size() const { return mPointData.size(); }
    void getPos(size_t n, openvdb::Vec3f& xyz) const { xyz = mPointData[n].position; }

private:
    const PointData& mPointData;
};

// wrapper to retrieve id
class PointId
{
public:
    typedef int value_type;

    PointId(const PointData& pointData)
        : mPointData(pointData) { }

    size_t size() const { return mPointData.size(); }

    template <typename T>
    void get(size_t n, T& value) const { value = mPointData[n].id; }

private:
    const PointData& mPointData;
};

// Generate random points by uniformly distributing points
// on a unit-sphere.
void genPoints(const int numPoints, const double scale, PointData& points)
{
    // init
    openvdb::math::Random01 randNumber(0);
    const int n = int(std::sqrt(double(numPoints)));
    const double xScale = (2.0 * M_PI) / double(n);
    const double yScale = M_PI / double(n);

    double x, y, theta, phi;
    openvdb::Vec3f pos;

    points.reserve(n*n);

    int id = 0;

    // loop over a [0 to n) x [0 to n) grid.
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < n; ++b) {

            // jitter, move to random pos. inside the current cell
            x = double(a) + randNumber();
            y = double(b) + randNumber();

            // remap to a lat/long map
            theta = y * yScale; // [0 to PI]
            phi   = x * xScale; // [0 to 2PI]

            // convert to cartesian coordinates on a unit sphere.
            // spherical coordinate triplet (r=1, theta, phi)
            pos[0] = std::sin(theta)*std::cos(phi)*scale;
            pos[1] = std::sin(theta)*std::sin(phi)*scale;
            pos[2] = std::cos(theta)*scale;

            points.push_back(new Point(pos, id++));
        }
    }
}

class PointAttribute
{
public:
    typedef boost::shared_ptr<PointAttribute> Ptr;

    static Ptr create(  const Name& name,
                        const NamePair& type,
                        PointData& points) { return Ptr(new PointAttribute(name, type, points)); }

    PointAttribute(const Name& name,
                   const NamePair& type,
                   PointData& points)
        : mName(name)
        , mType(type)
        , mPoints(points) { }

    const Name& name() const { return mName; }
    const NamePair& type() const { return mType; }

    size_t size() const { return mPoints.size(); }

    // comparison function to enable sorting by name

    bool
    operator<(const PointAttribute& rhs) {
        return this->name() < rhs.name();
    }

public:
    template <typename attributeT>
    struct Accessor
    {
        typedef boost::shared_ptr<Accessor<attributeT> > Ptr;

        friend class PointAttribute;

    public:
        // a point can only store P (vector attr) and id (scalar attr)
        // so we just hard code these into the access functions

        template<typename T> typename boost::enable_if_c<VecTraits<T>::IsVec, void>::type
        getValue(size_t n, T& value) const {
            for (unsigned i = 0; i < VecTraits<T>::Size; ++i) {
                value[i] = mPoints[n].position[i];
            }
        }

        template<typename T> typename boost::disable_if_c<VecTraits<T>::IsVec, void>::type
        getValue(size_t n, T& value) const {
            value = mPoints[n].id;
        }

    protected:
        Accessor(const PointData& points)
            : mPoints(points) { }

    private:
        const PointData& mPoints;
    }; // Accessor

    template <typename AttributeT>
    typename Accessor<AttributeT>::Ptr
    getAccessor() const
    {
        return typename Accessor<AttributeT>::Ptr(new Accessor<AttributeT>(mPoints));
    }

private:
    const Name mName;
    const NamePair mType;
    const PointData& mPoints;
}; // PointAttribute

// comparison function to enable sorting PointAttribute shared pointers

inline bool
operator<(const PointAttribute::Ptr& lhs, const PointAttribute::Ptr& rhs) {
    return lhs->name() < rhs->name();
}


////////////////////////////////////////


void
TestPointConversion::testPointConversion()
{
    // Define and register some common attribute types
    typedef TypedAttributeArray<int32_t>        AttributeI;
    typedef TypedAttributeArray<openvdb::Vec3s> AttributeVec3s;

    AttributeI::registerType();
    AttributeVec3s::registerType();

    // generate points

    PointData data;

    const unsigned long count(40000);

    genPoints(count, /*scale=*/ 100.0, data);

    CPPUNIT_ASSERT_EQUAL(data.size(), count);

    PointPosition pointPos(data);

    // convert point positions into a Point Data Grid

    const float voxelSize = 1.0f;
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(voxelSize));

    PointIndexGrid::Ptr pointIndexGrid = createPointIndexGrid<PointIndexGrid>(pointPos, *transform);
    PointDataGrid::Ptr pointDataGrid = createPointDataGrid<PointDataGrid>(*pointIndexGrid, pointPos,
                                            AttributeVec3s::attributeType(), *transform);

    // add id and populate

    AttributeSet::Util::NameAndType nameAndType("id", AttributeI::attributeType());

    appendAttribute(pointDataGrid->tree(), nameAndType);

    PointId pointId(data);

    populateAttribute(pointDataGrid->tree(), pointIndexGrid->tree(), "id", pointId);

    CPPUNIT_ASSERT_EQUAL(pointIndexGrid->tree().leafCount(), pointDataGrid->tree().leafCount());

    // create accessor and iterator for Point Data Tree

    const PointDataTree& tree = pointDataGrid->tree();
    PointDataAccessor<PointDataTree> acc(tree);

    PointDataTree::LeafCIter leafIter = tree.cbeginLeaf();

    // convert points back into original data structure

    CPPUNIT_ASSERT_EQUAL((unsigned long) 2, leafIter->attributeSet().size());

    CPPUNIT_ASSERT(leafIter->attributeSet().find("id") != AttributeSet::INVALID_POS);
    CPPUNIT_ASSERT(leafIter->attributeSet().find("P") != AttributeSet::INVALID_POS);

    PointData newData;

    for (; leafIter; ++leafIter) {

        AttributeHandle<Vec3f>::Ptr posHandle = AttributeHandle<Vec3f>::create(leafIter->attributeArray("P"));
        AttributeHandle<int32_t>::Ptr idHandle = AttributeHandle<int32_t>::create(leafIter->attributeArray("id"));

        for (PointDataTree::LeafNodeType::ValueOnCIter valueIter = leafIter->cbeginValueOn(); valueIter; ++valueIter) {

            Coord ijk = valueIter.getCoord();
            Vec3d xyz = ijk.asVec3d();

            PointDataAccessor<PointDataTree>::PointDataIndex pointIndexRange = acc.get(ijk);

            for (Index64 n = pointIndexRange.first, N = pointIndexRange.second; n < N; ++n) {

                // retrieve position in index space and translate into world space

                Vec3d pos = Vec3d(posHandle->get(n)) + xyz;
                pos = transform->indexToWorld(pos);

                // retrieve id

                const int id = idHandle->get(n);

                newData.push_back(new Point(pos, id));
            }
        }
    }

    CPPUNIT_ASSERT_EQUAL(data.size(), newData.size());

    // sort new point array by id

    std::sort(newData.begin(), newData.end());

    // confirm values match

    for (unsigned int i = 0; i < count; i++)
    {
        CPPUNIT_ASSERT_EQUAL(data[i].id, newData[i].id);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[i].position.x(), newData[i].position.x(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[i].position.y(), newData[i].position.y(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(data[i].position.z(), newData[i].position.z(), /*tolerance=*/1e-6);
    }
}

// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
