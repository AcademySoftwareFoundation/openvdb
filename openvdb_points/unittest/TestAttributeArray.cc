///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
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
#include <openvdb_points/tools/AttributeArray.h>
#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/io/File.h>

#include "ProfileTimer.h"

#include <sstream>
#include <iostream>

// Boost.Interprocess uses a header-only portion of Boost.DateTime
#define BOOST_DATE_TIME_NO_LIB
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/system/error_code.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/version.hpp> // for BOOST_VERSION

#include <tbb/atomic.h>

#ifdef _MSC_VER
#include <boost/interprocess/detail/os_file_functions.hpp> // open_existing_file(), close_file()
extern "C" __declspec(dllimport) bool __stdcall GetFileTime(
    void* fh, void* ctime, void* atime, void* mtime);
// boost::interprocess::detail was renamed to boost::interprocess::ipcdetail in Boost 1.48.
// Ensure that both namespaces exist.
namespace boost { namespace interprocess { namespace detail {} namespace ipcdetail {} } }
#else
#include <sys/types.h> // for struct stat
#include <sys/stat.h> // for stat()
#include <unistd.h> // for unlink()
#endif

/// @brief io::MappedFile has a private constructor, so this unit tests uses a matching proxy
class ProxyMappedFile
{
public:
    explicit ProxyMappedFile(const std::string& filename)
        : mImpl(new Impl(filename)) { }

private:
    class Impl
    {
    public:
        Impl(const std::string& filename)
            : mMap(filename.c_str(), boost::interprocess::read_only)
            , mRegion(mMap, boost::interprocess::read_only)
            , mAutoDelete(false)
        {
            mLastWriteTime = 0;
            const char* regionFilename = mMap.get_name();
            struct stat info;
            if (0 == ::stat(regionFilename, &info)) {
                mLastWriteTime = openvdb::Index64(info.st_mtime);
            }
        }

        typedef boost::function<void(std::string /*filename*/)> Notifier;
        boost::interprocess::file_mapping mMap;
        boost::interprocess::mapped_region mRegion;
        bool mAutoDelete;
        Notifier mNotifier;
        mutable tbb::atomic<openvdb::Index64> mLastWriteTime;
    }; // class Impl
    boost::scoped_ptr<Impl> mImpl;
}; // class ProxyMappedFile


using namespace openvdb;
using namespace openvdb::tools;

class TestAttributeArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testFixedPointConversion);
    CPPUNIT_TEST(testCompression);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAttributeHandle);
    CPPUNIT_TEST(testDelayedLoad);
    CPPUNIT_TEST(testProfile);

    CPPUNIT_TEST_SUITE_END();

    void testFixedPointConversion();
    void testCompression();
    void testAttributeArray();
    void testAttributeHandle();
    void testDelayedLoad();
    void testProfile();
}; // class TestAttributeArray

CPPUNIT_TEST_SUITE_REGISTRATION(TestAttributeArray);


////////////////////////////////////////


namespace {

bool
matchingNamePairs(const openvdb::NamePair& lhs,
                  const openvdb::NamePair& rhs)
{
    if (lhs.first != rhs.first)     return false;
    if (lhs.second != rhs.second)     return false;

    return true;
}

} // namespace


////////////////////////////////////////

void
TestAttributeArray::testFixedPointConversion()
{
    openvdb::math::Transform::Ptr transform(openvdb::math::Transform::createLinearTransform(/*voxelSize=*/0.1));

    const float value = 33.5688040469035f;

    {
        // convert to fixed-point value

        const openvdb::Vec3f worldSpaceValue(value);
        const openvdb::Vec3f indexSpaceValue = transform->worldToIndex(worldSpaceValue);
        const float voxelSpaceValue = indexSpaceValue.x() - math::Round(indexSpaceValue.x()) + 0.5f;
        const uint32_t intValue = floatingPointToFixedPoint<uint32_t>(voxelSpaceValue);

        // convert back to floating-point value

        const float newVoxelSpaceValue = fixedPointToFloatingPoint<float>(intValue);
        const openvdb::Vec3f newIndexSpaceValue(newVoxelSpaceValue + math::Round(indexSpaceValue.x()) - 0.5f);
        const openvdb::Vec3f newWorldSpaceValue = transform->indexToWorld(newIndexSpaceValue);

        const float newValue = newWorldSpaceValue.x();

        CPPUNIT_ASSERT_DOUBLES_EQUAL(value, newValue, /*tolerance=*/1e-6);
    }

    {
        // convert to fixed-point value (vector)

        const openvdb::Vec3f worldSpaceValue(value, value+1, value+2);
        const openvdb::Vec3f indexSpaceValue = transform->worldToIndex(worldSpaceValue);
        const float voxelSpaceValueX = indexSpaceValue.x() - math::Round(indexSpaceValue.x()) + 0.5f;
        const float voxelSpaceValueY = indexSpaceValue.y() - math::Round(indexSpaceValue.y()) + 0.5f;
        const float voxelSpaceValueZ = indexSpaceValue.z() - math::Round(indexSpaceValue.z()) + 0.5f;
        const openvdb::Vec3f voxelSpaceValue(voxelSpaceValueX, voxelSpaceValueY, voxelSpaceValueZ);
        const openvdb::math::Vec3<uint32_t> intValue = floatingPointToFixedPoint<openvdb::math::Vec3<uint32_t> >(voxelSpaceValue);

        // convert back to floating-point value (vector)

        const openvdb::Vec3f newVoxelSpaceValue = fixedPointToFloatingPoint<openvdb::Vec3f>(intValue);
        const float newIndexSpaceValueX = newVoxelSpaceValue.x() + math::Round(indexSpaceValue.x()) - 0.5f;
        const float newIndexSpaceValueY = newVoxelSpaceValue.y() + math::Round(indexSpaceValue.y()) - 0.5f;
        const float newIndexSpaceValueZ = newVoxelSpaceValue.z() + math::Round(indexSpaceValue.z()) - 0.5f;
        const openvdb::Vec3f newIndexSpaceValue(newIndexSpaceValueX, newIndexSpaceValueY, newIndexSpaceValueZ);
        const openvdb::Vec3f newWorldSpaceValue = transform->indexToWorld(newIndexSpaceValue);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.x(), newWorldSpaceValue.x(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.y(), newWorldSpaceValue.y(), /*tolerance=*/1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(worldSpaceValue.z(), newWorldSpaceValue.z(), /*tolerance=*/1e-6);

    }
}

void
TestAttributeArray::testCompression()
{
    using namespace attribute_compression;

    const int count = 256;

    { // with cleanup
        // compress

        int* uncompressedBuffer = new int[count];

        for (int i = 0; i < count; i++) {
            uncompressedBuffer[i] = i / 2;
        }

        int uncompressedBytes = 256 * sizeof(int);
        int compressedBytes;

        char* compressedBuffer = compress(  reinterpret_cast<char*>(uncompressedBuffer), sizeof(int),
                                            uncompressedBytes, compressedBytes, /*cleanup=*/ true);

#ifdef OPENVDB_USE_BLOSC
        CPPUNIT_ASSERT(compressedBytes < uncompressedBytes);
        CPPUNIT_ASSERT(compressedBuffer);

        // uncompressedSize

        CPPUNIT_ASSERT_EQUAL(uncompressedBytes, uncompressedSize(compressedBuffer));

        // decompress

        char* newUncompressedBuffer = decompress(compressedBuffer, uncompressedBytes, /*cleanup=*/ true);

        CPPUNIT_ASSERT(newUncompressedBuffer);

        delete[] newUncompressedBuffer;
#else
        CPPUNIT_ASSERT(!compressedBuffer);

        // uncompressedSize

        CPPUNIT_ASSERT_THROW(uncompressedSize(compressedBuffer), openvdb::RuntimeError);

        // decompress

        char* newUncompressedBuffer = 0;
        CPPUNIT_ASSERT_THROW(   newUncompressedBuffer = decompress(compressedBuffer, uncompressedBytes,
                                                        /*cleanup=*/ true), openvdb::RuntimeError);

        CPPUNIT_ASSERT(!newUncompressedBuffer);

        delete[] uncompressedBuffer;
#endif
    }

    { // without cleanup
        // compress

        int* uncompressedBuffer = new int[count];

        for (int i = 0; i < count; i++) {
            uncompressedBuffer[i] = i / 2;
        }

        int uncompressedBytes = 256 * sizeof(int);
        int compressedBytes;

        const char* compressedBuffer = compress(reinterpret_cast<const char*>(uncompressedBuffer), sizeof(int),
                                                uncompressedBytes, compressedBytes);

#ifdef OPENVDB_USE_BLOSC
        CPPUNIT_ASSERT(compressedBytes < uncompressedBytes);
        CPPUNIT_ASSERT(compressedBuffer);

        // uncompressedSize

        CPPUNIT_ASSERT_EQUAL(uncompressedBytes, uncompressedSize(compressedBuffer));

        // decompress

        const char* newUncompressedBuffer = decompress(compressedBuffer, uncompressedBytes);

        CPPUNIT_ASSERT(newUncompressedBuffer);

        for (int i = 0; i < count; i++) {
            CPPUNIT_ASSERT_EQUAL(uncompressedBuffer[i], reinterpret_cast<const int*>(newUncompressedBuffer)[i]);
        }

        delete[] uncompressedBuffer;
        delete[] compressedBuffer;
        delete[] newUncompressedBuffer;
#else
        CPPUNIT_ASSERT(!compressedBuffer);

        // uncompressedSize

        CPPUNIT_ASSERT_THROW(uncompressedSize(compressedBuffer), openvdb::RuntimeError);

        // decompress

        char* newUncompressedBuffer = 0;
        CPPUNIT_ASSERT_THROW(   newUncompressedBuffer = decompress(compressedBuffer, uncompressedBytes),
                                                        openvdb::RuntimeError);

        CPPUNIT_ASSERT(!newUncompressedBuffer);

        delete[] uncompressedBuffer;
#endif
    }
}

void
TestAttributeArray::testAttributeArray()
{
    typedef openvdb::tools::TypedAttributeArray<float> AttributeArrayF;
    typedef openvdb::tools::TypedAttributeArray<double> AttributeArrayD;

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(attr->size(), size_t(50));
    }

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(size_t(50), attr->size());

        AttributeArrayD& typedAttr = static_cast<AttributeArrayD&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0));

        // test unsafe methods for get() and set()

        typedAttr.setUnsafe(0, 1.5);
        typedAttr.getUnsafe(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(1.5), value, /*tolerance=*/double(0.0));
    }

    typedef openvdb::tools::FixedPointAttributeCodec<uint16_t> FixedPointCodec;
    typedef openvdb::tools::TypedAttributeArray<double, FixedPointCodec> AttributeArrayC;

    { // test hasValueType()
        openvdb::tools::AttributeArray::Ptr attrC(new AttributeArrayC(50));
        openvdb::tools::AttributeArray::Ptr attrD(new AttributeArrayD(50));
        openvdb::tools::AttributeArray::Ptr attrF(new AttributeArrayF(50));

        CPPUNIT_ASSERT(attrD->hasValueType<double>());
        CPPUNIT_ASSERT(attrC->hasValueType<double>());
        CPPUNIT_ASSERT(!attrF->hasValueType<double>());

        CPPUNIT_ASSERT(!attrD->hasValueType<float>());
        CPPUNIT_ASSERT(!attrC->hasValueType<float>());
        CPPUNIT_ASSERT(attrF->hasValueType<float>());
    }

    {
        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayC(50));

        AttributeArrayC& typedAttr = static_cast<AttributeArrayC&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), value, /*tolerance=*/double(0.0001));

        // test unsafe methods for get() and set()

        double value2 = 0.0;
        typedAttr.setUnsafe(0, double(0.2));
        typedAttr.getUnsafe(0, value2);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.2), value2, /*tolerance=*/double(0.0001));
    }

    typedef openvdb::tools::TypedAttributeArray<int32_t> AttributeArrayI;

    { // Base class API

        openvdb::tools::AttributeArray::Ptr attr(new AttributeArrayI(50));

        CPPUNIT_ASSERT_EQUAL(size_t(50), attr->size());

        CPPUNIT_ASSERT_EQUAL((sizeof(AttributeArrayI) + sizeof(int)), attr->memUsage());

        CPPUNIT_ASSERT(attr->isType<AttributeArrayI>());
        CPPUNIT_ASSERT(!attr->isType<AttributeArrayD>());

        CPPUNIT_ASSERT(*attr == *attr);
    }

    { // Typed class API

        const size_t count = 50;
        const size_t uniformMemUsage = sizeof(AttributeArrayI) + sizeof(int);
        const size_t expandedMemUsage = sizeof(AttributeArrayI) + count * sizeof(int);

        AttributeArrayI attr(count);

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 0);
        CPPUNIT_ASSERT_EQUAL(attr.get(10), 0);

        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        attr.set(0, 10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        AttributeArrayI attr2(count);
        attr2.set(0, 10);

        CPPUNIT_ASSERT(attr == attr2);

        attr.set(1, 5);

        CPPUNIT_ASSERT(!attr.compact());
        CPPUNIT_ASSERT(!attr.isUniform());

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 10);
        CPPUNIT_ASSERT_EQUAL(attr.get(1), 5);
        CPPUNIT_ASSERT_EQUAL(attr.get(2), 0);

        attr.collapse(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 5);
        CPPUNIT_ASSERT_EQUAL(attr.get(20), 5);
        CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(20), 5);

        attr.expand(/*fill=*/false);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        attr.collapse(5);

        CPPUNIT_ASSERT(attr.isUniform());

        attr.expand();

        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 5);
        }

        CPPUNIT_ASSERT(attr.compact());
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT(attr.compact());

        attr.expand();

        attr.fill(10);
        CPPUNIT_ASSERT(!attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 10);
        }

        attr.collapse(7);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        CPPUNIT_ASSERT_EQUAL(attr.get(0), 7);
        CPPUNIT_ASSERT_EQUAL(attr.get(20), 7);

        attr.fill(5);
        CPPUNIT_ASSERT(attr.isUniform());
        CPPUNIT_ASSERT_EQUAL(uniformMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attr.get(i), 5);
        }

        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setTransient(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setHidden(true);
        CPPUNIT_ASSERT(attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        attr.setTransient(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(attr.isHidden());

        { // test copy construction
            AttributeArrayI attrB(attr);
            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }

        // attribute array must not be uniform for compression

        attr.set(1, 7);
        attr.set(2, 8);
        attr.set(6, 100);

        { // test compressed copy construction
            attr.compress();

#ifdef OPENVDB_USE_BLOSC
            CPPUNIT_ASSERT(attr.isCompressed());
#endif

            AttributeArray::Ptr attrCopy = attr.copy();
            AttributeArrayI& attrB(AttributeArrayI::cast(*attrCopy));

            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }

        { // test compressed copy construction (uncompress on copy)
            attr.compress();

#ifdef OPENVDB_USE_BLOSC
            CPPUNIT_ASSERT(attr.isCompressed());
#endif

            AttributeArray::Ptr attrCopy = attr.copyUncompressed();
            AttributeArrayI& attrB(AttributeArrayI::cast(*attrCopy));

            CPPUNIT_ASSERT(!attrB.isCompressed());

            attr.decompress();

            CPPUNIT_ASSERT(matchingNamePairs(attr.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attr.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attr.memUsage(), attrB.memUsage());
            CPPUNIT_ASSERT_EQUAL(attr.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attr.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attr.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attr.isCompressed(), attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attr.get(i), attrB.getUnsafe(i));
                CPPUNIT_ASSERT_EQUAL(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }
    }

    typedef openvdb::tools::FixedPointAttributeCodec<uint16_t> FixedPointCodec;

    { // Fixed codec range
        openvdb::tools::AttributeArray::Ptr attr1(new AttributeArrayC(50));

        AttributeArrayC& fixedPoint = static_cast<AttributeArrayC&>(*attr1);

        // fixed point range is -0.5 => 0.5

        fixedPoint.set(0, -0.6);
        fixedPoint.set(1, -0.4);
        fixedPoint.set(2, 0.4);
        fixedPoint.set(3, 0.6);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.5), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(-0.4), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.4), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(0.5), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    typedef openvdb::tools::TypedAttributeArray<openvdb::Vec3f, openvdb::tools::UnitVecAttributeCodec> AttributeArrayU;

    { // UnitVec codec test
        openvdb::tools::AttributeArray::Ptr attr1(new AttributeArrayU(50));

        AttributeArrayU& unitVec = static_cast<AttributeArrayU&>(*attr1);

        // all vectors must be unit length

        const openvdb::Vec3f vec1(1.0, 0.0, 0.0);
        const openvdb::Vec3f vec2(openvdb::Vec3f(1.0, 2.0, 3.0).unit());
        const openvdb::Vec3f vec3(openvdb::Vec3f(1.0, 2.0, 300000.0).unit());

        unitVec.set(0, vec1);
        unitVec.set(1, vec2);
        unitVec.set(2, vec3);

        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.x()), unitVec.get(0).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.y()), unitVec.get(0).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec1.z()), unitVec.get(0).z(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.x()), unitVec.get(1).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.y()), unitVec.get(1).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec2.z()), unitVec.get(1).z(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.x()), unitVec.get(2).x(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.y()), unitVec.get(2).y(), /*tolerance=*/double(0.0001));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(double(vec3.z()), unitVec.get(2).z(), /*tolerance=*/double(0.0001));
    }

    { // IO
        const size_t count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        io::setDataCompression(ostr, io::COMPRESS_BLOSC);

        attrA.write(ostr);

        AttributeArrayI attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
        CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
        CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
        CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
        CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
        CPPUNIT_ASSERT_EQUAL(attrA.isCompressed(), attrB.isCompressed());
        CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
        }

        AttributeArrayI attrC(count, 3);
        attrC.setTransient(true);

        std::ostringstream ostrC(std::ios_base::binary);
        attrC.write(ostrC);

        CPPUNIT_ASSERT_EQUAL(ostrC.str().size(), size_t(0));
    }

    // Registry
    AttributeArrayI::registerType();

    openvdb::tools::AttributeArray::Ptr attr =
        openvdb::tools::AttributeArray::create(
            AttributeArrayI::attributeType(), 34);

    { // Casting
        using namespace openvdb;
        using namespace openvdb::tools;

        AttributeArray::Ptr array = TypedAttributeArray<float>::create(0);
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*array));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*array), TypeError);

        AttributeArray::ConstPtr constArray = array;
        CPPUNIT_ASSERT_NO_THROW(TypedAttributeArray<float>::cast(*constArray));
        CPPUNIT_ASSERT_THROW(TypedAttributeArray<int>::cast(*constArray), TypeError);
    }
}


void
TestAttributeArray::testAttributeHandle()
{
    using namespace openvdb;
    using namespace openvdb::tools;
    using namespace openvdb::math;

    typedef TypedAttributeArray<int>                                                          AttributeI;
    typedef TypedAttributeArray<float, NullAttributeCodec<half> >                             AttributeFH;
    typedef TypedAttributeArray<Vec3f>                                                        AttributeVec3f;

    typedef AttributeWriteHandle<int> AttributeHandleRWI;

    AttributeI::registerType();
    AttributeFH::registerType();
    AttributeVec3f::registerType();

    // create a Descriptor and AttributeSet

    typedef AttributeSet::Descriptor Descriptor;

    Descriptor::Ptr descr = Descriptor::create(Descriptor::Inserter()
        .add("pos", AttributeVec3f::attributeType())
        .add("truncate", AttributeFH::attributeType())
        .add("int", AttributeI::attributeType())
        .vec);

    unsigned count = 50;
    AttributeSet attrSet(descr, /*arrayLength=*/count);

    // check uniform value implementation

    {
        AttributeArray* array = attrSet.get(2);

        AttributeHandleRWI handle(*array);

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 0);
        CPPUNIT_ASSERT_EQUAL(handle.get(10), 0);

        CPPUNIT_ASSERT(handle.isUniform());

        handle.set(0, 10);
        CPPUNIT_ASSERT(!handle.isUniform());

        handle.collapse(5);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 5);
        CPPUNIT_ASSERT_EQUAL(handle.get(20), 5);

        handle.expand();
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 5);
        }

        CPPUNIT_ASSERT(handle.compact());
        CPPUNIT_ASSERT(handle.isUniform());

        handle.expand();

        handle.fill(10);
        CPPUNIT_ASSERT(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 10);
        }

        handle.collapse(7);
        CPPUNIT_ASSERT(handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 7);
        CPPUNIT_ASSERT_EQUAL(handle.get(20), 7);

        handle.fill(5);
        CPPUNIT_ASSERT(handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            CPPUNIT_ASSERT_EQUAL(handle.get(i), 5);
        }

        CPPUNIT_ASSERT(handle.isUniform());
    }

    {
        AttributeArray* array = attrSet.get(0);

        AttributeHandleRWVec3f handle(*array);

        handle.set(5, Vec3f(10));

        CPPUNIT_ASSERT_EQUAL(handle.get(5), Vec3f(10));
    }

    {
        AttributeArray* array = attrSet.get(1);

        array->compress();

        AttributeHandleRWF handle(*array);

        handle.set(6, float(11));

        CPPUNIT_ASSERT_EQUAL(handle.get(6), float(11));

        CPPUNIT_ASSERT(!array->isCompressed());

#ifdef OPENVDB_USE_BLOSC
        array->compress();

        CPPUNIT_ASSERT(array->isCompressed());

        {
            AttributeHandleROF handleRO(*array);

            CPPUNIT_ASSERT(array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(handleRO.get(6), float(11));

            CPPUNIT_ASSERT(array->isCompressed());
        }

        CPPUNIT_ASSERT(array->isCompressed());

        {
            AttributeHandleROF handleRO(*array, /*preserveCompression=*/false);

            // AttributeHandle uncompresses data on construction

            CPPUNIT_ASSERT(!array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(handleRO.get(6), float(11));

            CPPUNIT_ASSERT(!array->isCompressed());
        }

        CPPUNIT_ASSERT(!array->isCompressed());
#endif
    }

    // check values have been correctly set without using handles

    {
        AttributeVec3f* array = static_cast<AttributeVec3f*>(attrSet.get(0));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(array->get(5), Vec3f(10));
    }

    {
        AttributeFH* array = static_cast<AttributeFH*>(attrSet.get(1));

        CPPUNIT_ASSERT(array);

        CPPUNIT_ASSERT_EQUAL(array->get(6), float(11));
    }
}

void
TestAttributeArray::testDelayedLoad()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<int>    AttributeArrayI;

    AttributeArrayI::registerType();

    std::string tempDir(std::getenv("TMPDIR"));
    if (tempDir.empty())    tempDir = P_tmpdir;

    { // IO
        const size_t count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        std::string filename;

        // write out attribute array to a temp file
        {
            std::ofstream fileout;
            filename = tempDir + "/openvdb_delayed1";
            fileout.open(filename.c_str());
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrA.write(fileout);

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        ProxyMappedFile* proxy = new ProxyMappedFile(filename);
        boost::shared_ptr<io::MappedFile> mappedFile(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

            CPPUNIT_ASSERT(matchingNamePairs(attrA.type(), attrB.type()));
            CPPUNIT_ASSERT_EQUAL(attrA.size(), attrB.size());
            CPPUNIT_ASSERT_EQUAL(attrA.isUniform(), attrB.isUniform());
            CPPUNIT_ASSERT_EQUAL(attrA.isTransient(), attrB.isTransient());
            CPPUNIT_ASSERT_EQUAL(attrA.isHidden(), attrB.isHidden());
            CPPUNIT_ASSERT_EQUAL(attrA.isCompressed(), attrB.isCompressed());

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
            attrB.loadData();
#endif
            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            attrB.get(0);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check implicit load through compress()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            CPPUNIT_ASSERT(!attrB.isCompressed());

            attrB.compress();

#ifdef OPENVDB_USE_BLOSC
            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrB.isCompressed());
#else
#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif
            CPPUNIT_ASSERT(!attrB.isCompressed());
#endif
        }

        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrC.isOutOfCore());
            CPPUNIT_ASSERT(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();
#endif

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrC.isOutOfCore());
            CPPUNIT_ASSERT(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrC.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            AttributeHandle<int> handle(attrB);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
        }

        // read in using delayed load and check detaching of file (using collapse())
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.collapse();

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrB.isUniform());

            CPPUNIT_ASSERT_EQUAL(attrB.get(0), 0);
        }

        // cleanup temp files

        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out compressed attribute array to a temp file
        {
            std::ofstream fileout;
            filename = tempDir + "/openvdb_delayed2";
            fileout.open(filename.c_str());
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrA.compress();
            attrA.write(fileout);

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifdef OPENVDB_USE_BLOSC
            CPPUNIT_ASSERT(attrB.isCompressed());
#endif

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
            attrB.loadData();
#endif

            CPPUNIT_ASSERT_EQUAL(attrA.memUsage(), attrB.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            attrB.get(0);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
            }
        }

#ifdef OPENVDB_USE_BLOSC
        // read in using delayed load and check no implicit load through compress()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif
            CPPUNIT_ASSERT(attrB.isCompressed());

            attrB.compress();

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif
            CPPUNIT_ASSERT(attrB.isCompressed());
        }

        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrC.isOutOfCore());
            CPPUNIT_ASSERT(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();
#endif

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(!attrC.isOutOfCore());
            CPPUNIT_ASSERT(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrB.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrC.get(i));
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            CPPUNIT_ASSERT(attrB.isCompressed());

            AttributeHandle<int> handle(attrB);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());
            CPPUNIT_ASSERT(attrB.isCompressed());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(attrA.get(i), handle.get(i));
            }

            AttributeHandle<int> handle2(attrB, /*preserveCompression=*/false);
            CPPUNIT_ASSERT(!attrB.isCompressed());
        }
#endif

        // cleanup temp files

        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());
    }
}

namespace profile {

typedef openvdb::util::ProfileTimer ProfileTimer;

template <typename AttrT>
void expand(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": expand");
    attr.expand();
}

template <typename AttrT>
void set(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": set");
    for (size_t i = 0; i < attr.size(); i++) {
        attr.set(i, typename AttrT::ValueType(i));
    }
}

template <typename AttrT>
void setH(const Name& prefix, AttrT& attr)
{
    typedef typename AttrT::ValueType ValueType;
    ProfileTimer timer(prefix + ": setHandle");
    AttributeWriteHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        handle.set(i, ValueType(i));
    }
}

template <typename AttrT>
void sum(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sum");
    typename AttrT::ValueType sum = 0;
    for (size_t i = 0; i < attr.size(); i++) {
        sum += attr.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename AttrT>
void sumH(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumHandle");
    typedef typename AttrT::ValueType ValueType;
    ValueType sum = 0;
    AttributeHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename AttrT>
void sumWH(const Name& prefix, AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumWriteHandle");
    typedef typename AttrT::ValueType ValueType;
    ValueType sum = 0;
    AttributeWriteHandle<ValueType> handle(attr);
    for (size_t i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

} // namespace profile

void
TestAttributeArray::testProfile()
{
    using namespace openvdb;
    using namespace openvdb::util;
    using namespace openvdb::math;
    using namespace openvdb::tools;

    typedef TypedAttributeArray<float>                                      AttributeArrayF;
    typedef TypedAttributeArray<float,
                                FixedPointAttributeCodec<uint16_t> >        AttributeArrayF16;
    typedef TypedAttributeArray<float,
                                FixedPointAttributeCodec<uint8_t> >         AttributeArrayF8;

    ///////////////////////////////////////////////////

#ifdef PROFILE
    const int elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const int elements(10 * 1000 * 1000);
#endif

    // std::vector

    {
        std::vector<float> values;
        {
            ProfileTimer timer("Vector<float>: resize");
            values.resize(elements);
        }
        {
            ProfileTimer timer("Vector<float>: set");
            for (int i = 0; i < elements; i++) {
                values[i] = float(i);
            }
        }
        {
            ProfileTimer timer("Vector<float>: sum");
            float sum = 0;
            for (int i = 0; i < elements; i++) {
                sum += values[i];
            }
            // to prevent optimisation clean up
            CPPUNIT_ASSERT(sum);
        }
    }

    // AttributeArray

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeArray<float>", attr);
        profile::set("AttributeArray<float>", attr);
        profile::sum("AttributeArray<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeArray<float, fp16>", attr);
        profile::set("AttributeArray<float, fp16>", attr);
        profile::sum("AttributeArray<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeArray<float, fp8>", attr);
        profile::set("AttributeArray<float, fp8>", attr);
        profile::sum("AttributeArray<float, fp8>", attr);
    }

    // AttributeHandle

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH("AttributeHandle<float>", attr);
        profile::sumH("AttributeHandle<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH("AttributeHandle<float, fp16>", attr);
        profile::sumH("AttributeHandle<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH("AttributeHandle<float, fp8>", attr);
        profile::sumH("AttributeHandle<float, fp8>", attr);
    }
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
