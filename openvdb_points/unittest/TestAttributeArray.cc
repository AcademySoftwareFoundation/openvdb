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
        {
            mLastWriteTime = 0;
            const char* regionFilename = mMap.get_name();
            struct stat info;
            if (0 == ::stat(regionFilename, &info)) {
                mLastWriteTime = openvdb::Index64(info.st_mtime);
            }
        }

        using Notifier = std::function<void(std::string /*filename*/)>;
        boost::interprocess::file_mapping mMap;
        boost::interprocess::mapped_region mRegion;
        bool mAutoDelete = false;
        Notifier mNotifier;
        mutable tbb::atomic<openvdb::Index64> mLastWriteTime;
    }; // class Impl
    std::unique_ptr<Impl> mImpl;
}; // class ProxyMappedFile


using namespace openvdb;
using namespace openvdb::tools;

class TestAttributeArray: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestAttributeArray);
    CPPUNIT_TEST(testFixedPointConversion);
    CPPUNIT_TEST(testCompression);
    CPPUNIT_TEST(testRegistry);
    CPPUNIT_TEST(testAttributeArray);
    CPPUNIT_TEST(testAccessorEval);
    CPPUNIT_TEST(testAttributeHandle);
    CPPUNIT_TEST(testStrided);
    CPPUNIT_TEST(testDelayedLoad);
    CPPUNIT_TEST(testQuaternions);
    CPPUNIT_TEST(testMatrices);
    CPPUNIT_TEST(testProfile);

    CPPUNIT_TEST_SUITE_END();

    void testFixedPointConversion();
    void testCompression();
    void testRegistry();
    void testAttributeArray();
    void testAccessorEval();
    void testAttributeHandle();
    void testStrided();
    void testDelayedLoad();
    void testQuaternions();
    void testMatrices();
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
        const openvdb::math::Vec3<uint32_t> intValue = floatingPointToFixedPoint<openvdb::math::Vec3<uint32_t>>(voxelSpaceValue);

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

namespace {

static AttributeArray::Ptr factory1(size_t, Index) { return AttributeArray::Ptr(); }
static AttributeArray::Ptr factory2(size_t, Index) { return AttributeArray::Ptr(); }

} // namespace

void
TestAttributeArray::testRegistry()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    using AttributeF = TypedAttributeArray<float>;

    AttributeArray::clearRegistry();

    { // cannot create AttributeArray that is not registered
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
        CPPUNIT_ASSERT_THROW(AttributeArray::create(AttributeF::attributeType(), size_t(5)), LookupError);
    }

    // manually register the type and factory

    AttributeArray::registerType(AttributeF::attributeType(), factory1);

    { // cannot re-register an already registered AttributeArray
        CPPUNIT_ASSERT(AttributeArray::isRegistered(AttributeF::attributeType()));
        CPPUNIT_ASSERT_THROW(AttributeArray::registerType(AttributeF::attributeType(), factory2), KeyError);
    }

    { // un-registering
        AttributeArray::unregisterType(AttributeF::attributeType());
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
    }

    { // clearing registry
        AttributeArray::registerType(AttributeF::attributeType(), factory1);
        AttributeArray::clearRegistry();
        CPPUNIT_ASSERT(!AttributeArray::isRegistered(AttributeF::attributeType()));
    }
}

void
TestAttributeArray::testCompression()
{
    using namespace attribute_compression;

    const int count = 256;

    { // invalid buffer (out of range)

        // compress

        int* smallBuffer = new int[256];
        size_t invalidBytes = INT_MAX - 1;

        size_t testCompressedBytes = compressedSize(reinterpret_cast<char*>(smallBuffer), sizeof(int), invalidBytes);

        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        char* buffer = compress(reinterpret_cast<char*>(smallBuffer), sizeof(int), invalidBytes, testCompressedBytes, /*cleanup=*/ false);

        CPPUNIT_ASSERT(!buffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        // decompress

#ifdef OPENVDB_USE_BLOSC
        for (int i = 0; i < 256; i++)   smallBuffer[i] = i;

        char* compressedBuffer = compress(reinterpret_cast<char*>(smallBuffer), sizeof(int), 256 * sizeof(int), testCompressedBytes, /*cleanup=*/ true);

        buffer = decompress(reinterpret_cast<char*>(compressedBuffer), invalidBytes - 16, /*cleanup=*/ false);

        CPPUNIT_ASSERT(!buffer);

        CPPUNIT_ASSERT_THROW(decompress(reinterpret_cast<char*>(compressedBuffer), 256 * sizeof(int) + 1, /*cleanup=*/ true), openvdb::RuntimeError);
#endif
    }

    { // with cleanup
        // compress

        int* uncompressedBuffer = new int[count];

        for (int i = 0; i < count; i++) {
            uncompressedBuffer[i] = i / 2;
        }

        size_t uncompressedBytes = 256 * sizeof(int);
        size_t compressedBytes;

        size_t testCompressedBytes = compressedSize(reinterpret_cast<char*>(uncompressedBuffer), sizeof(int), uncompressedBytes);

        char* compressedBuffer = compress(  reinterpret_cast<char*>(uncompressedBuffer), sizeof(int),
                                            uncompressedBytes, compressedBytes, /*cleanup=*/ true);

#ifdef OPENVDB_USE_BLOSC
        CPPUNIT_ASSERT(compressedBytes < uncompressedBytes);
        CPPUNIT_ASSERT(compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, compressedBytes);

        // uncompressedSize

        CPPUNIT_ASSERT_EQUAL(uncompressedBytes, uncompressedSize(compressedBuffer));

        // decompress

        char* newUncompressedBuffer = decompress(compressedBuffer, uncompressedBytes, /*cleanup=*/ true);

        CPPUNIT_ASSERT(newUncompressedBuffer);

        delete[] newUncompressedBuffer;
#else
        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

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

        size_t uncompressedBytes = 256 * sizeof(int);
        size_t compressedBytes;

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
    using AttributeArrayF = openvdb::tools::TypedAttributeArray<float>;
    using AttributeArrayD = openvdb::tools::TypedAttributeArray<double>;

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

        // out-of-range get() and set()
        CPPUNIT_ASSERT_THROW(typedAttr.set(100, 0.5), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.set(100, 1), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.get(100, value), openvdb::IndexError);
        CPPUNIT_ASSERT_THROW(typedAttr.get(100), openvdb::IndexError);
    }

#ifdef NDEBUG
    { // test setUnsafe and getUnsafe on uniform arrays
        AttributeArrayD::Ptr attr(new AttributeArrayD(50));

        CPPUNIT_ASSERT_EQUAL(size_t(50), attr->size());
        attr->collapse(5.0);
        CPPUNIT_ASSERT(attr->isUniform());

        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(10), 5.0, /*tolerance=*/double(0.0));
        CPPUNIT_ASSERT(attr->isUniform());

        // this is expected behaviour because for performance reasons, array is not implicitly expanded

        attr->setUnsafe(10, 15.0);
        CPPUNIT_ASSERT(attr->isUniform());
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));

        attr->expand();
        CPPUNIT_ASSERT(!attr->isUniform());
        attr->setUnsafe(10, 25.0);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));
        CPPUNIT_ASSERT_DOUBLES_EQUAL(attr->getUnsafe(10), 25.0, /*tolerance=*/double(0.0));
    }
#endif

    using FixedPointCodec = openvdb::tools::FixedPointCodec<false>;
    using AttributeArrayC = openvdb::tools::TypedAttributeArray<double, FixedPointCodec>;

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

    using AttributeArrayI = openvdb::tools::TypedAttributeArray<int32_t>;

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

        CPPUNIT_ASSERT_EQUAL(attr.size(), size_t(count));

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

        attr.setHidden(false);
        CPPUNIT_ASSERT(!attr.isTransient());
        CPPUNIT_ASSERT(!attr.isHidden());

        attr.setHidden(true);

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
    
    using FixedPointCodec = openvdb::tools::FixedPointCodec<false>;

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

    using AttributeArrayU = openvdb::tools::TypedAttributeArray<openvdb::Vec3f, openvdb::tools::UnitVecCodec>;

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

        std::ostringstream ostrD(std::ios_base::binary);
        attrC.write(ostrD, /*transient=*/true);

        CPPUNIT_ASSERT(ostrD.str().size() != size_t(0));
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
TestAttributeArray::testAccessorEval()
{
    using AttributeF = TypedAttributeArray<float>;

    struct TestAccessor
    {
        static float getterError(const AttributeArray* /*array*/, const Index /*n*/) {
            OPENVDB_THROW(NotImplementedError, "");
        }
        static void setterError(AttributeArray* /*array*/, const Index /*n*/, const float& /*value*/) {
            OPENVDB_THROW(NotImplementedError, "");
        }

        static float testGetter(const AttributeArray* array, const Index n) {
            return AccessorEval<UnknownCodec, float>::get(&getterError, array, n);
        }
        static void testSetter(AttributeArray* array, const Index n, const float& value) {
            AccessorEval<UnknownCodec, float>::set(&setterError, array, n, value);
        }
    };

    { // test get and set (NullCodec)
        AttributeF::Ptr attr = AttributeF::create(10);
        attr->collapse(5.0f);
        attr->expand();

        AttributeArray& array = *attr;

        // explicit codec is used here so getter and setter are not called

        AttributeWriteHandle<float, NullCodec> writeHandle(array);

        writeHandle.mSetter = TestAccessor::setterError;

        writeHandle.set(4, 15.0f);

        AttributeHandle<float, NullCodec> handle(array);

        handle.mGetter = TestAccessor::getterError;

        const float result1 = handle.get(4);
        const float result2 = handle.get(6);

        CPPUNIT_ASSERT_EQUAL(result1, 15.0f);
        CPPUNIT_ASSERT_EQUAL(result2, 5.0f);
    }

    { // test get and set (UnknownCodec)
        AttributeF::Ptr attr = AttributeF::create(10);
        attr->collapse(5.0f);
        attr->expand();

        AttributeArray& array = *attr;

        // unknown codec is used here so getter and setter are called

        AttributeWriteHandle<float, UnknownCodec> writeHandle(array);

        writeHandle.mSetter = TestAccessor::setterError;

        CPPUNIT_ASSERT_THROW(writeHandle.set(4, 15.0f), NotImplementedError);

        AttributeHandle<float, UnknownCodec> handle(array);

        handle.mGetter = TestAccessor::getterError;

        CPPUNIT_ASSERT_THROW(handle.get(4), NotImplementedError);
    }
}


void
TestAttributeArray::testAttributeHandle()
{
    using namespace openvdb;
    using namespace openvdb::tools;
    using namespace openvdb::math;

    using AttributeI            = TypedAttributeArray<int>;
    using AttributeFH           = TypedAttributeArray<float, TruncateCodec>;
    using AttributeVec3f        = TypedAttributeArray<Vec3f>;

    using AttributeHandleRWI    = AttributeWriteHandle<int>;

    AttributeI::registerType();
    AttributeFH::registerType();
    AttributeVec3f::registerType();

    // create a Descriptor and AttributeSet

    using Descriptor = AttributeSet::Descriptor;
    Descriptor::Ptr descr = Descriptor::create(AttributeVec3f::attributeType());

    unsigned count = 50;
    AttributeSet attrSet(descr, /*arrayLength=*/count);

    attrSet.appendAttribute("truncate", AttributeFH::attributeType());
    attrSet.appendAttribute("int", AttributeI::attributeType());

    // check uniform value implementation

    {
        AttributeArray* array = attrSet.get(2);

        AttributeHandleRWI nonExpandingHandle(*array, /*expand=*/false);
        CPPUNIT_ASSERT(nonExpandingHandle.isUniform());

        AttributeHandleRWI handle(*array);
        CPPUNIT_ASSERT(!handle.isUniform());

        CPPUNIT_ASSERT_EQUAL(handle.size(), array->size());

        CPPUNIT_ASSERT_EQUAL(handle.get(0), 0);
        CPPUNIT_ASSERT_EQUAL(handle.get(10), 0);

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

        AttributeWriteHandle<Vec3f> handle(*array);

        handle.set(5, Vec3f(10));

        CPPUNIT_ASSERT_EQUAL(handle.get(5), Vec3f(10));
    }

    {
        AttributeArray* array = attrSet.get(1);

        array->compress();

        AttributeWriteHandle<float> handle(*array);

        handle.set(6, float(11));

        CPPUNIT_ASSERT_EQUAL(handle.get(6), float(11));

        CPPUNIT_ASSERT(!array->isCompressed());

#ifdef OPENVDB_USE_BLOSC
        array->compress();

        CPPUNIT_ASSERT(array->isCompressed());

        {
            AttributeHandle<float> handleRO(*array);

            CPPUNIT_ASSERT(array->isCompressed());

            CPPUNIT_ASSERT_EQUAL(handleRO.get(6), float(11));

            CPPUNIT_ASSERT(array->isCompressed());
        }

        CPPUNIT_ASSERT(array->isCompressed());

        {
            AttributeHandle<float> handleRO(*array, /*preserveCompression=*/false);

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
TestAttributeArray::testStrided()
{
    using AttributeArrayI           = openvdb::tools::TypedAttributeArray<int>;
    using NonStridedHandle          = AttributeHandle<int>;
    using StridedHandle             = AttributeHandle<int, /*CodecType=*/UnknownCodec, /*Strided=*/true>;
    using StridedWriteHandle        = AttributeWriteHandle<int, /*CodecType=*/UnknownCodec, /*Strided=*/true>;
    using InterleavedHandle         = AttributeHandle<int, /*CodecType=*/UnknownCodec, /*Strided=*/true, /*Interleaved=*/true>;
    using InterleavedWriteHandle    = AttributeWriteHandle<int, /*CodecType=*/UnknownCodec, /*Strided=*/true, /*Interleaved=*/true>;

    { // non-strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/1);
        CPPUNIT_ASSERT(!array->isStrided());
        CPPUNIT_ASSERT_EQUAL(array->stride(), Index(1));
        CPPUNIT_ASSERT_EQUAL(array->size(), size_t(2));
        // cannot create a StridedAttributeHandle with a stride of 1
        CPPUNIT_ASSERT_THROW(StridedHandle::create(*array), openvdb::TypeError);
        CPPUNIT_ASSERT_THROW(StridedWriteHandle::create(*array), openvdb::TypeError);
    }

    { // strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/3);

        CPPUNIT_ASSERT(array->isStrided());
        CPPUNIT_ASSERT_EQUAL(array->stride(), Index(3));
        CPPUNIT_ASSERT_EQUAL(array->size(), size_t(2));
        CPPUNIT_ASSERT(array->isUniform());

        CPPUNIT_ASSERT_EQUAL(array->get(0), 0);
        CPPUNIT_ASSERT_EQUAL(array->get(5), 0);
        CPPUNIT_ASSERT_THROW(array->get(6), IndexError); // out-of-range

        // cannot create a non-strided AttributeHandle for a strided array
        CPPUNIT_ASSERT_THROW(NonStridedHandle::create(*array), TypeError);
        CPPUNIT_ASSERT_THROW(InterleavedHandle::create(*array), TypeError);
        CPPUNIT_ASSERT_THROW(InterleavedWriteHandle::create(*array), TypeError);
        CPPUNIT_ASSERT_NO_THROW(StridedHandle::create(*array));
        CPPUNIT_ASSERT_NO_THROW(StridedWriteHandle::create(*array));

        array->collapse(10);

        CPPUNIT_ASSERT_EQUAL(array->get(0), int(10));
        CPPUNIT_ASSERT_EQUAL(array->get(5), int(10));

        array->expand();

        CPPUNIT_ASSERT_EQUAL(array->get(0), int(10));
        CPPUNIT_ASSERT_EQUAL(array->get(5), int(10));

        array->collapse(0);

        CPPUNIT_ASSERT_EQUAL(array->get(0), int(0));
        CPPUNIT_ASSERT_EQUAL(array->get(5), int(0));

        StridedWriteHandle writeHandle(*array);

        writeHandle.set(0, 2, 5);
        writeHandle.set(1, 1, 10);

        CPPUNIT_ASSERT_EQUAL(writeHandle.stride(), Index(3));
        CPPUNIT_ASSERT_EQUAL(writeHandle.size(), size_t(2));

        // non-interleaved: 0 0 5 0 10 0

        CPPUNIT_ASSERT_EQUAL(array->get(2), 5);
        CPPUNIT_ASSERT_EQUAL(array->get(4), 10);

        CPPUNIT_ASSERT_EQUAL(writeHandle.get(0, 2), 5);
        CPPUNIT_ASSERT_EQUAL(writeHandle.get(1, 1), 10);

        StridedHandle handle(*array);

        CPPUNIT_ASSERT_EQUAL(handle.get(0, 2), 5);
        CPPUNIT_ASSERT_EQUAL(handle.get(1, 1), 10);

        CPPUNIT_ASSERT_EQUAL(handle.stride(), Index(3));
        CPPUNIT_ASSERT_EQUAL(handle.size(), size_t(2));

#ifdef OPENVDB_2_ABI_COMPATIBLE
        size_t arrayMem = 48;
#else
        size_t arrayMem = 64;
#endif

        CPPUNIT_ASSERT_EQUAL(array->memUsage(), sizeof(int) * /*size*/3 * /*stride*/2 + arrayMem);
    }

    { // strided, interleaved array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/3);
        array->setInterleaved(true);

        CPPUNIT_ASSERT(array->isStrided());
        CPPUNIT_ASSERT(array->isInterleaved());
        CPPUNIT_ASSERT_EQUAL(array->stride(), Index(3));
        CPPUNIT_ASSERT_EQUAL(array->size(), size_t(2));
        CPPUNIT_ASSERT(array->isUniform());

        array->setInterleaved(false);

        CPPUNIT_ASSERT(!array->isInterleaved());

        array->setInterleaved(true);

        CPPUNIT_ASSERT_EQUAL(array->get(0), 0);
        CPPUNIT_ASSERT_EQUAL(array->get(5), 0);
        CPPUNIT_ASSERT_THROW(array->get(6), IndexError); // out-of-range

        CPPUNIT_ASSERT_EQUAL(array->get(4), 0);
        CPPUNIT_ASSERT_EQUAL(array->get(3), 0);

        CPPUNIT_ASSERT_THROW(StridedHandle::create(*array), TypeError);
        CPPUNIT_ASSERT_THROW(StridedWriteHandle::create(*array), TypeError);
        CPPUNIT_ASSERT_NO_THROW(InterleavedHandle::create(*array));
        CPPUNIT_ASSERT_NO_THROW(InterleavedWriteHandle::create(*array));

        InterleavedWriteHandle writeHandle(*array);

        CPPUNIT_ASSERT_EQUAL(array->get(4), 0);
        CPPUNIT_ASSERT_EQUAL(array->get(3), 0);

        CPPUNIT_ASSERT_EQUAL(array->get(4), 0);
        CPPUNIT_ASSERT_EQUAL(array->get(3), 0);

        writeHandle.set(0, 2, 5);
        writeHandle.set(1, 1, 10);

        // interleaved: 0 0 0 10 5 0

        CPPUNIT_ASSERT_EQUAL(array->get(4), 5);
        CPPUNIT_ASSERT_EQUAL(array->get(3), 10);

        CPPUNIT_ASSERT_EQUAL(writeHandle.get(0, 2), 5);
        CPPUNIT_ASSERT_EQUAL(writeHandle.get(1, 1), 10);

        InterleavedHandle handle(*array);

        CPPUNIT_ASSERT_EQUAL(handle.get(0, 2), 5);
        CPPUNIT_ASSERT_EQUAL(handle.get(1, 1), 10);

#ifdef OPENVDB_2_ABI_COMPATIBLE
        size_t arrayMem = 48;
#else
        size_t arrayMem = 64;
#endif

        CPPUNIT_ASSERT_EQUAL(array->memUsage(), sizeof(int) * /*size*/3 * /*stride*/2 + arrayMem);

        std::ostringstream ostr(std::ios_base::binary);
        io::setDataCompression(ostr, io::COMPRESS_BLOSC);

        array->write(ostr);

        AttributeArrayI attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        InterleavedHandle handle2(attrB);

        CPPUNIT_ASSERT_EQUAL(handle2.get(0, 2), 5);
        CPPUNIT_ASSERT_EQUAL(handle2.get(1, 1), 10);
    }
}

void
TestAttributeArray::testDelayedLoad()
{
    using namespace openvdb;
    using namespace openvdb::tools;

    using AttributeArrayI = TypedAttributeArray<int>;

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
        SharedPtr<io::MappedFile> mappedFile(reinterpret_cast<io::MappedFile*>(proxy));

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

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            CPPUNIT_ASSERT(!attrB.isUniform());

            attrB.fill(5);

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(5, attrB.get(i));
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

        AttributeArrayI attrUniform(count);

        // write out uniform attribute array to a temp file
        {
            std::ofstream fileout;
            filename = tempDir + "/openvdb_delayed2";
            fileout.open(filename.c_str());
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrUniform.write(fileout);

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

#ifndef OPENVDB_2_ABI_COMPATIBLE
            CPPUNIT_ASSERT(attrB.isOutOfCore());
#endif

            CPPUNIT_ASSERT(attrB.isUniform());

            attrB.fill(5);

            CPPUNIT_ASSERT(attrB.isUniform());

            CPPUNIT_ASSERT(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                CPPUNIT_ASSERT_EQUAL(5, attrB.get(i));
            }
        }

        AttributeArrayI attrStrided(count, /*stride=*/3);

        CPPUNIT_ASSERT(attrStrided.isStrided());
        CPPUNIT_ASSERT_EQUAL(attrStrided.stride(), Index(3));

        // write out strided attribute array to a temp file
        {
            std::ofstream fileout;
            filename = tempDir + "/openvdb_delayed3";
            fileout.open(filename.c_str());
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrStrided.write(fileout);

            fileout.close();
        }

        // abuse File being a friend of MappedFile to get around the private constructor

        proxy = new ProxyMappedFile(filename);
        mappedFile.reset(reinterpret_cast<io::MappedFile*>(proxy));

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.read(filein);

            CPPUNIT_ASSERT(attrB.isStrided());
            CPPUNIT_ASSERT_EQUAL(attrB.stride(), Index(3));
        }

        // write out compressed attribute array to a temp file
        {
            std::ofstream fileout;
            filename = tempDir + "/openvdb_delayed4";
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


void
TestAttributeArray::testQuaternions()
{
    using namespace openvdb;

    using AttributeQF = TypedAttributeArray<math::Quat<float>>;
    using AttributeQD = TypedAttributeArray<QuatR>;

    AttributeQF::registerType();
    AttributeQD::registerType();

    CPPUNIT_ASSERT(AttributeQF::attributeType().first == "quats");
    CPPUNIT_ASSERT(AttributeQD::attributeType().first == "quatd");

    AttributeQF test(/*size=*/5);

    AttributeQD orient(/*size=*/10);

    { // set some quaternion values
        AttributeWriteHandle<QuatR> orientHandle(orient);

        orientHandle.set(4, QuatR(1, 2, 3, 4));
        orientHandle.set(7, QuatR::zero());
    }

    { // get some quaternion values
        AttributeHandle<QuatR> orientHandle(orient);

        CPPUNIT_ASSERT_EQUAL(orientHandle.get(3), QuatR::identity());
        CPPUNIT_ASSERT_EQUAL(orientHandle.get(4), QuatR(1, 2, 3, 4));
        CPPUNIT_ASSERT_EQUAL(orientHandle.get(7), QuatR::zero());
    }

    { // create a quaternion array with a zero uniform value
        AttributeQD zero(/*size=*/10, /*stride=*/1, QuatR::zero());

        CPPUNIT_ASSERT_EQUAL(zero.get(5), QuatR::zero());
    }
}


void
TestAttributeArray::testMatrices()
{
    using namespace openvdb;

    using AttributeM = TypedAttributeArray<Mat4d>;

    AttributeM::registerType();

    CPPUNIT_ASSERT(AttributeM::attributeType().first == "mat4d");

    AttributeM matrix(/*size=*/10);

    Mat4d testMatrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    { // set some matrix values
        AttributeWriteHandle<Mat4d> matrixHandle(matrix);

        matrixHandle.set(4, testMatrix);
        matrixHandle.set(7, Mat4d::zero());
    }

    { // get some matrix values
        AttributeHandle<Mat4d> matrixHandle(matrix);

        CPPUNIT_ASSERT_EQUAL(matrixHandle.get(3), Mat4d::identity());
        CPPUNIT_ASSERT_EQUAL(matrixHandle.get(4), testMatrix);
        CPPUNIT_ASSERT_EQUAL(matrixHandle.get(7), Mat4d::zero());
    }

    { // create a matrix array with a zero uniform value
        AttributeM zero(/*size=*/10, /*stride=*/1, Mat4d::zero());

        CPPUNIT_ASSERT_EQUAL(zero.get(5), Mat4d::zero());
    }
}


namespace profile {

using ProfileTimer = openvdb::util::ProfileTimer;

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
    const size_t size = attr.size();
    for (size_t i = 0; i < size; i++) {
        attr.setUnsafe(i, typename AttrT::ValueType(i));
    }
}

template <typename CodecT, typename AttrT>
void setH(const Name& prefix, AttrT& attr)
{
    using ValueType = typename AttrT::ValueType;
    ProfileTimer timer(prefix + ": setHandle");
    AttributeWriteHandle<ValueType, CodecT> handle(attr);
    const size_t size = attr.size();
    for (size_t i = 0; i < size; i++) {
        handle.set(i, ValueType(i));
    }
}

template <typename AttrT>
void sum(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sum");
    typename AttrT::ValueType sum = 0;
    const size_t size = attr.size();
    for (size_t i = 0; i < size; i++) {
        sum += attr.getUnsafe(i);
    }
    // prevent compiler optimisations removing computation
    CPPUNIT_ASSERT(sum);
}

template <typename CodecT, typename AttrT>
void sumH(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumHandle");
    using ValueType = typename AttrT::ValueType;
    ValueType sum = 0;
    AttributeHandle<ValueType, CodecT> handle(attr);
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

    using AttributeArrayF   = TypedAttributeArray<float>;
    using AttributeArrayF16 = TypedAttributeArray<float, FixedPointCodec<false>>;
    using AttributeArrayF8  = TypedAttributeArray<float, FixedPointCodec<true>>;

    ///////////////////////////////////////////////////

#ifdef PROFILE
    const size_t elements(1000 * 1000 * 1000);

    std::cerr << std::endl;
#else
    const size_t elements(10 * 1000 * 1000);
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
            for (size_t i = 0; i < elements; i++) {
                values[i] = float(i);
            }
        }
        {
            ProfileTimer timer("Vector<float>: sum");
            float sum = 0;
            for (size_t i = 0; i < elements; i++) {
                sum += float(values[i]);
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

    // AttributeHandle (UnknownCodec)

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float, fp16>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float, fp16>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH<UnknownCodec>("AttributeHandle<float, fp8>", attr);
        profile::sumH<UnknownCodec>("AttributeHandle<float, fp8>", attr);
    }

    // AttributeHandle (explicit codec)

    {
        AttributeArrayF attr(elements);
        profile::expand("AttributeHandle<float>", attr);
        profile::setH<NullCodec>("AttributeHandle<float, Codec>", attr);
        profile::sumH<NullCodec>("AttributeHandle<float, Codec>", attr);
    }

    {
        AttributeArrayF16 attr(elements);
        profile::expand("AttributeHandle<float, fp16>", attr);
        profile::setH<FixedPointCodec<false>>("AttributeHandle<float, fp16, Codec>", attr);
        profile::sumH<FixedPointCodec<false>>("AttributeHandle<float, fp16, Codec>", attr);
    }

    {
        AttributeArrayF8 attr(elements);
        profile::expand("AttributeHandle<float, fp8>", attr);
        profile::setH<FixedPointCodec<true>>("AttributeHandle<float, fp8, Codec>", attr);
        profile::sumH<FixedPointCodec<true>>("AttributeHandle<float, fp8, Codec>", attr);
    }
}

// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
