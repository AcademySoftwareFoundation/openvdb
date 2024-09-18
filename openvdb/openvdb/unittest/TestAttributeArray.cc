// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/AttributeSet.h>
#include <openvdb/Types.h>
#include <openvdb/math/Transform.h>
#include <openvdb/io/File.h>
#include <openvdb/util/Assert.h>

#include <gtest/gtest.h>

#ifdef OPENVDB_USE_DELAYED_LOADING
#ifdef __clang__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-macros"
#endif
// Boost.Interprocess uses a header-only portion of Boost.DateTime
#define BOOST_DATE_TIME_NO_LIB
#ifdef __clang__
#pragma GCC diagnostic pop
#endif
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#ifdef _WIN32
#include <boost/interprocess/detail/os_file_functions.hpp> // open_existing_file(), close_file()
// boost::interprocess::detail was renamed to boost::interprocess::ipcdetail in Boost 1.48.
// Ensure that both namespaces exist.
namespace boost { namespace interprocess { namespace detail {} namespace ipcdetail {} } }
#include <windows.h>
#else
#include <sys/types.h> // for struct stat
#include <sys/stat.h> // for stat()
#endif
#endif // OPENVDB_USE_DELAYED_LOADING

#include <tbb/tick_count.h>
#include <atomic>

#include <cstdio> // for std::remove()
#include <fstream>
#include <sstream>
#include <iostream>


#ifdef OPENVDB_USE_DELAYED_LOADING
/// @brief io::MappedFile has a private constructor, so declare a class that acts as the friend
class TestMappedFile
{
public:
    static openvdb::io::MappedFile::Ptr create(const std::string& filename)
    {
        return openvdb::SharedPtr<openvdb::io::MappedFile>(new openvdb::io::MappedFile(filename));
    }
};
#endif


/// @brief Functionality similar to openvdb::util::CpuTimer except with prefix padding and no decimals.
///
/// @code
///    ProfileTimer timer("algorithm 1");
///    // code to be timed goes here
///    timer.stop();
/// @endcode
class ProfileTimer
{
public:
    /// @brief Prints message and starts timer.
    ///
    /// @note Should normally be followed by a call to stop()
    ProfileTimer(const std::string& msg)
    {
        (void)msg;
#ifdef PROFILE
        // padd string to 50 characters
        std::string newMsg(msg);
        if (newMsg.size() < 50)     newMsg.insert(newMsg.end(), 50 - newMsg.size(), ' ');
        std::cerr << newMsg << " ... ";
#endif
        mT0 = tbb::tick_count::now();
    }

    ~ProfileTimer() { this->stop(); }

    /// Return Time diference in milliseconds since construction or start was called.
    inline double delta() const
    {
        tbb::tick_count::interval_t dt = tbb::tick_count::now() - mT0;
        return 1000.0*dt.seconds();
    }

    /// @brief Print time in milliseconds since construction or start was called.
    inline void stop() const
    {
#ifdef PROFILE
        std::stringstream ss;
        ss << std::setw(6) << ::round(this->delta());
        std::cerr << "completed in " << ss.str() << " ms\n";
#endif
    }

private:
    tbb::tick_count mT0;
};// ProfileTimer


struct ScopedFile
{
    explicit ScopedFile(const std::string& s): pathname(s) {}
    ~ScopedFile() { if (!pathname.empty()) std::remove(pathname.c_str()); }
    const std::string pathname;
};


using namespace openvdb;
using namespace openvdb::points;

class TestAttributeArray: public ::testing::Test
{
public:
    void SetUp() override { AttributeArray::clearRegistry(); }
    void TearDown() override { AttributeArray::clearRegistry(); }

    void testRegistry();
    void testAccessorEval();
    void testDelayedLoad();
}; // class TestAttributeArray


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


TEST_F(TestAttributeArray, testFixedPointConversion)
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

        EXPECT_NEAR(value, newValue, /*tolerance=*/1e-6);
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

        EXPECT_NEAR(worldSpaceValue.x(), newWorldSpaceValue.x(), /*tolerance=*/1e-6);
        EXPECT_NEAR(worldSpaceValue.y(), newWorldSpaceValue.y(), /*tolerance=*/1e-6);
        EXPECT_NEAR(worldSpaceValue.z(), newWorldSpaceValue.z(), /*tolerance=*/1e-6);

    }
}

namespace
{
// use a dummy factory as TypedAttributeArray::factory is private
static AttributeArray::Ptr factoryInt(Index n, Index strideOrTotalSize, bool constantStride, const Metadata*)
{
    return TypedAttributeArray<int>::create(n, strideOrTotalSize, constantStride);
}
} // namespace

void
TestAttributeArray::testRegistry()
{
    using AttributeF = TypedAttributeArray<float>;
    using AttributeFTrnc = TypedAttributeArray<float, TruncateCodec>;

    AttributeArray::clearRegistry();

    { // cannot create AttributeArray that is not registered
        EXPECT_TRUE(!AttributeArray::isRegistered(AttributeF::attributeType()));
        EXPECT_THROW(AttributeArray::create(AttributeF::attributeType(), Index(5)), LookupError);
    }

    { // throw when attempting to register a float type with an integer factory
        EXPECT_THROW(AttributeArray::registerType(
            AttributeF::attributeType(), factoryInt), KeyError);
    }

    // register the attribute array

    AttributeF::registerType();

    { // can register an AttributeArray with the same value type but different codec
        EXPECT_NO_THROW(AttributeFTrnc::registerType());
        EXPECT_TRUE(AttributeArray::isRegistered(AttributeF::attributeType()));
        EXPECT_TRUE(AttributeArray::isRegistered(AttributeFTrnc::attributeType()));
    }

    { // un-registering
        AttributeArray::unregisterType(AttributeF::attributeType());
        EXPECT_TRUE(!AttributeArray::isRegistered(AttributeF::attributeType()));
        EXPECT_TRUE(AttributeArray::isRegistered(AttributeFTrnc::attributeType()));
    }

    { // clearing registry
        AttributeF::registerType();
        AttributeArray::clearRegistry();
        EXPECT_TRUE(!AttributeArray::isRegistered(AttributeF::attributeType()));
    }
}
TEST_F(TestAttributeArray, testRegistry) { testRegistry(); }


TEST_F(TestAttributeArray, testAttributeArray)
{
    using AttributeArrayF = TypedAttributeArray<float>;
    using AttributeArrayD = TypedAttributeArray<double>;

    {
        AttributeArray::Ptr attr(new AttributeArrayD(50));

        EXPECT_EQ(Index(50), attr->size());
    }

    {
        AttributeArray::Ptr attr(new AttributeArrayD(50));

        EXPECT_EQ(Index(50), attr->size());

        AttributeArrayD& typedAttr = static_cast<AttributeArrayD&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        EXPECT_NEAR(double(0.5), value, /*tolerance=*/double(0.0));

        // test unsafe methods for get() and set()

        typedAttr.setUnsafe(0, 1.5);
        typedAttr.getUnsafe(0, value);

        EXPECT_NEAR(double(1.5), value, /*tolerance=*/double(0.0));

        // out-of-range get() and set()
        EXPECT_THROW(typedAttr.set(100, 0.5), openvdb::IndexError);
        EXPECT_THROW(typedAttr.set(100, 1), openvdb::IndexError);
        EXPECT_THROW(typedAttr.get(100, value), openvdb::IndexError);
        EXPECT_THROW(typedAttr.get(100), openvdb::IndexError);
    }

    { // test copy constructor and copy assignment operator
        AttributeArrayD attr1(10);
        AttributeArrayD attr2(5);

        attr1.set(9, 4.6);

        // copy constructor

        AttributeArrayD attr3(attr1);

        EXPECT_EQ(Index(10), attr3.size());
        EXPECT_EQ(4.6, attr3.get(9));

        // copy assignment operator

        attr2 = attr1;

        EXPECT_EQ(Index(10), attr2.size());
        EXPECT_EQ(4.6, attr2.get(9));
    }

#ifdef NDEBUG
    { // test setUnsafe and getUnsafe on uniform arrays
        AttributeArrayD::Ptr attr(new AttributeArrayD(50));

        EXPECT_EQ(Index(50), attr->size());
        attr->collapse(5.0);
        EXPECT_TRUE(attr->isUniform());

        EXPECT_NEAR(attr->getUnsafe(10), 5.0, /*tolerance=*/double(0.0));
        EXPECT_TRUE(attr->isUniform());

        // this is expected behaviour because for performance reasons, array is not implicitly expanded

        attr->setUnsafe(10, 15.0);
        EXPECT_TRUE(attr->isUniform());
        EXPECT_NEAR(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));

        attr->expand();
        EXPECT_TRUE(!attr->isUniform());
        attr->setUnsafe(10, 25.0);
        EXPECT_NEAR(attr->getUnsafe(5), 15.0, /*tolerance=*/double(0.0));
        EXPECT_NEAR(attr->getUnsafe(10), 25.0, /*tolerance=*/double(0.0));
    }
#endif

    using AttributeArrayC = TypedAttributeArray<double, FixedPointCodec<false>>;

    { // test hasValueType()
        AttributeArray::Ptr attrC(new AttributeArrayC(50));
        AttributeArray::Ptr attrD(new AttributeArrayD(50));
        AttributeArray::Ptr attrF(new AttributeArrayF(50));

        EXPECT_TRUE(attrD->hasValueType<double>());
        EXPECT_TRUE(attrC->hasValueType<double>());
        EXPECT_TRUE(!attrF->hasValueType<double>());

        EXPECT_TRUE(!attrD->hasValueType<float>());
        EXPECT_TRUE(!attrC->hasValueType<float>());
        EXPECT_TRUE(attrF->hasValueType<float>());
    }

    { // lots of type checking
        Index size(50);
        {
            TypedAttributeArray<bool> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("bool"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(1), attr.valueTypeSize());
            EXPECT_EQ(Index(1), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int8_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("int8"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(1), attr.valueTypeSize());
            EXPECT_EQ(Index(1), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int16_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("int16"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(2), attr.valueTypeSize());
            EXPECT_EQ(Index(2), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int32_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("int32"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(4), attr.valueTypeSize());
            EXPECT_EQ(Index(4), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<int64_t> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("int64"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(8), attr.valueTypeSize());
            EXPECT_EQ(Index(8), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            // half is not registered by default, but for complete-ness
            TypedAttributeArray<math::half> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("half"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(2), attr.valueTypeSize());
            EXPECT_EQ(Index(2), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("float"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(4), attr.valueTypeSize());
            EXPECT_EQ(Index(4), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<double> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("double"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(8), attr.valueTypeSize());
            EXPECT_EQ(Index(8), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Vec3<int32_t>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("vec3i"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(12), attr.valueTypeSize());
            EXPECT_EQ(Index(12), attr.storageTypeSize());
            EXPECT_TRUE(!attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(attr.valueTypeIsClass());
            EXPECT_TRUE(attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Vec3<double>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("vec3d"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(24), attr.valueTypeSize());
            EXPECT_EQ(Index(24), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(attr.valueTypeIsClass());
            EXPECT_TRUE(attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Mat3<float>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("mat3s"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(36), attr.valueTypeSize());
            EXPECT_EQ(Index(36), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Mat4<double>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("mat4d"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(128), attr.valueTypeSize());
            EXPECT_EQ(Index(128), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<math::Quat<float>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("quats"), attr.valueType());
            EXPECT_EQ(Name("null"), attr.codecType());
            EXPECT_EQ(Index(16), attr.valueTypeSize());
            EXPECT_EQ(Index(16), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float, TruncateCodec> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("float"), attr.valueType());
            EXPECT_EQ(Name("trnc"), attr.codecType());
            EXPECT_EQ(Index(4), attr.valueTypeSize());
            EXPECT_EQ(Index(2), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float, FixedPointCodec<false, UnitRange>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("float"), attr.valueType());
            EXPECT_EQ(Name("ufxpt16"), attr.codecType());
            EXPECT_EQ(Index(4), attr.valueTypeSize());
            EXPECT_EQ(Index(2), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
        {
            TypedAttributeArray<float, FixedPointCodec<true, UnitRange>> typedAttr(size);
            AttributeArray& attr(typedAttr);
            EXPECT_EQ(Name("float"), attr.valueType());
            EXPECT_EQ(Name("ufxpt8"), attr.codecType());
            EXPECT_EQ(Index(4), attr.valueTypeSize());
            EXPECT_EQ(Index(1), attr.storageTypeSize());
            EXPECT_TRUE(attr.valueTypeIsFloatingPoint());
            EXPECT_TRUE(!attr.valueTypeIsClass());
            EXPECT_TRUE(!attr.valueTypeIsVector());
            EXPECT_TRUE(!attr.valueTypeIsQuaternion());
            EXPECT_TRUE(!attr.valueTypeIsMatrix());
        }
    }

    {
        AttributeArray::Ptr attr(new AttributeArrayC(50));

        AttributeArrayC& typedAttr = static_cast<AttributeArrayC&>(*attr);

        typedAttr.set(0, 0.5);

        double value = 0.0;
        typedAttr.get(0, value);

        EXPECT_NEAR(double(0.5), value, /*tolerance=*/double(0.0001));

        // test unsafe methods for get() and set()

        double value2 = 0.0;
        typedAttr.setUnsafe(0, double(0.2));
        typedAttr.getUnsafe(0, value2);

        EXPECT_NEAR(double(0.2), value2, /*tolerance=*/double(0.0001));
    }

    using AttributeArrayI = TypedAttributeArray<int32_t>;

    { // Base class API

        AttributeArray::Ptr attr(new AttributeArrayI(50));

        EXPECT_EQ(Index(50), attr->size());

        EXPECT_EQ((sizeof(AttributeArrayI) + sizeof(int)), attr->memUsage());

        EXPECT_TRUE(attr->isType<AttributeArrayI>());
        EXPECT_TRUE(!attr->isType<AttributeArrayD>());

        EXPECT_TRUE(*attr == *attr);
    }

    { // Typed class API

        const Index count = 50;
        const size_t uniformMemUsage = sizeof(AttributeArrayI) + sizeof(int);
        const size_t expandedMemUsage = sizeof(AttributeArrayI) + count * sizeof(int);

        AttributeArrayI attr(count);

        EXPECT_EQ(Index(count), attr.size());

        EXPECT_EQ(0, attr.get(0));
        EXPECT_EQ(0, attr.get(10));

        EXPECT_TRUE(attr.isUniform());
        EXPECT_EQ(uniformMemUsage, attr.memUsage());

        attr.set(0, 10);
        EXPECT_TRUE(!attr.isUniform());
        EXPECT_EQ(expandedMemUsage, attr.memUsage());

        AttributeArrayI attr2(count);
        attr2.set(0, 10);

        EXPECT_TRUE(attr == attr2);

        attr.set(1, 5);

        EXPECT_TRUE(!attr.compact());
        EXPECT_TRUE(!attr.isUniform());

        EXPECT_EQ(10, attr.get(0));
        EXPECT_EQ(5, attr.get(1));
        EXPECT_EQ(0, attr.get(2));

        attr.collapse(5);
        EXPECT_TRUE(attr.isUniform());
        EXPECT_EQ(uniformMemUsage, attr.memUsage());

        EXPECT_EQ(5, attr.get(0));
        EXPECT_EQ(5, attr.get(20));
        EXPECT_EQ(5, attr.getUnsafe(20));

        attr.expand(/*fill=*/false);
        EXPECT_TRUE(!attr.isUniform());
        EXPECT_EQ(expandedMemUsage, attr.memUsage());

        attr.collapse(5);

        EXPECT_TRUE(attr.isUniform());

        attr.expand();

        EXPECT_TRUE(!attr.isUniform());
        EXPECT_EQ(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(5, attr.get(i));
        }

        EXPECT_TRUE(attr.compact());
        EXPECT_TRUE(attr.isUniform());
        EXPECT_TRUE(attr.compact());

        attr.expand();

        attr.fill(10);
        EXPECT_TRUE(!attr.isUniform());
        EXPECT_EQ(expandedMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(10, attr.get(i));
        }

        attr.collapse(7);
        EXPECT_TRUE(attr.isUniform());
        EXPECT_EQ(uniformMemUsage, attr.memUsage());

        EXPECT_EQ(7, attr.get(0));
        EXPECT_EQ(7, attr.get(20));

        attr.fill(5);
        EXPECT_TRUE(attr.isUniform());
        EXPECT_EQ(uniformMemUsage, attr.memUsage());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(5, attr.get(i));
        }

        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());

        attr.setTransient(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());

        attr.setHidden(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(attr.isHidden());

        attr.setTransient(false);
        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(attr.isHidden());

        attr.setHidden(false);
        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());

        attr.setHidden(true);

        { // test copy construction
            AttributeArrayI attrB(attr);
            EXPECT_TRUE(matchingNamePairs(attr.type(), attrB.type()));
            EXPECT_EQ(attr.size(), attrB.size());
            EXPECT_EQ(attr.memUsage(), attrB.memUsage());
            EXPECT_EQ(attr.isUniform(), attrB.isUniform());
            EXPECT_EQ(attr.isTransient(), attrB.isTransient());
            EXPECT_EQ(attr.isHidden(), attrB.isHidden());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attr.get(i), attrB.get(i));
                EXPECT_EQ(attr.get(i), attrB.getUnsafe(i));
                EXPECT_EQ(attr.getUnsafe(i), attrB.getUnsafe(i));
            }
        }

        { // Equality using an unregistered attribute type
            TypedAttributeArray<math::half> attr1(50);
            TypedAttributeArray<math::half> attr2(50);

            EXPECT_TRUE(attr1 == attr2);
        }

        // attribute array must not be uniform for compression

        attr.set(1, 7);
        attr.set(2, 8);
        attr.set(6, 100);
    }

    { // Fixed codec (position range)
        AttributeArray::Ptr attr1(new AttributeArrayC(50));

        AttributeArrayC& fixedPoint = static_cast<AttributeArrayC&>(*attr1);

        // position range is -0.5 => 0.5

        fixedPoint.set(0, -0.6);
        fixedPoint.set(1, -0.4);
        fixedPoint.set(2, 0.4);
        fixedPoint.set(3, 0.6);

        EXPECT_NEAR(double(-0.5), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(-0.4), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.4), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.5), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using UnitFixedPointCodec8 = FixedPointCodec<false, UnitRange>;
    using AttributeArrayUFxpt8 = TypedAttributeArray<float, UnitFixedPointCodec8>;

    { // 8-bit fixed codec (unit range)
        AttributeArray::Ptr attr1(new AttributeArrayUFxpt8(50));

        AttributeArrayUFxpt8& fixedPoint = static_cast<AttributeArrayUFxpt8&>(*attr1);

        // unit range is 0.0 => 1.0

        fixedPoint.set(0, -0.2);
        fixedPoint.set(1, 0.3);
        fixedPoint.set(2, 0.6);
        fixedPoint.set(3, 1.1);

        EXPECT_NEAR(double(0.0), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.3), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.6), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(1.0), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using UnitFixedPointCodec16 = FixedPointCodec<false, UnitRange>;
    using AttributeArrayUFxpt16 = TypedAttributeArray<float, UnitFixedPointCodec16>;

    { // 16-bit fixed codec (unit range)
        AttributeArray::Ptr attr1(new AttributeArrayUFxpt16(50));

        AttributeArrayUFxpt16& fixedPoint = static_cast<AttributeArrayUFxpt16&>(*attr1);

        // unit range is 0.0 => 1.0

        fixedPoint.set(0, -0.2);
        fixedPoint.set(1, 0.3);
        fixedPoint.set(2, 0.6);
        fixedPoint.set(3, 1.1);

        EXPECT_NEAR(double(0.0), fixedPoint.get(0), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.3), fixedPoint.get(1), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(0.6), fixedPoint.get(2), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(1.0), fixedPoint.get(3), /*tolerance=*/double(0.0001));
    }

    using AttributeArrayU = TypedAttributeArray<openvdb::Vec3f, UnitVecCodec>;

    { // UnitVec codec test
        AttributeArray::Ptr attr1(new AttributeArrayU(50));

        AttributeArrayU& unitVec = static_cast<AttributeArrayU&>(*attr1);

        // all vectors must be unit length

        const openvdb::Vec3f vec1(1.0, 0.0, 0.0);
        const openvdb::Vec3f vec2(openvdb::Vec3f(1.0, 2.0, 3.0).unit());
        const openvdb::Vec3f vec3(openvdb::Vec3f(1.0, 2.0, 300000.0).unit());

        unitVec.set(0, vec1);
        unitVec.set(1, vec2);
        unitVec.set(2, vec3);

        EXPECT_NEAR(double(vec1.x()), unitVec.get(0).x(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec1.y()), unitVec.get(0).y(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec1.z()), unitVec.get(0).z(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec2.x()), unitVec.get(1).x(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec2.y()), unitVec.get(1).y(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec2.z()), unitVec.get(1).z(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec3.x()), unitVec.get(2).x(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec3.y()), unitVec.get(2).y(), /*tolerance=*/double(0.0001));
        EXPECT_NEAR(double(vec3.z()), unitVec.get(2).z(), /*tolerance=*/double(0.0001));
    }

    { // IO
        const Index count = 50;
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

        EXPECT_TRUE(attrA == attrB);

        AttributeArrayI attrC(count, 3);
        attrC.setTransient(true);

        std::ostringstream ostrC(std::ios_base::binary);
        attrC.write(ostrC);

        EXPECT_TRUE(ostrC.str().empty());

        std::ostringstream ostrD(std::ios_base::binary);
        attrC.write(ostrD, /*transient=*/true);

        EXPECT_TRUE(!ostrD.str().empty());
    }

    // Registry
    AttributeArrayI::registerType();

    AttributeArray::Ptr attr =
        AttributeArray::create(
            AttributeArrayI::attributeType(), 34);

    { // Casting
        AttributeArray::Ptr array = TypedAttributeArray<float>::create(0);
        EXPECT_NO_THROW(TypedAttributeArray<float>::cast(*array));
        EXPECT_THROW(TypedAttributeArray<int>::cast(*array), TypeError);

        AttributeArray::ConstPtr constArray = array;
        EXPECT_NO_THROW(TypedAttributeArray<float>::cast(*constArray));
        EXPECT_THROW(TypedAttributeArray<int>::cast(*constArray), TypeError);
    }
}

struct VectorWrapper
{
    using T = std::vector<std::pair<Index, Index>>;

    VectorWrapper(const T& _data) : data(_data) { }
    operator bool() const { return index < data.size(); }
    VectorWrapper& operator++() { index++; return *this; }
    Index sourceIndex() const { OPENVDB_ASSERT(*this); return data[index].first; }
    Index targetIndex() const { OPENVDB_ASSERT(*this); return data[index].second; }

private:
    const T& data;
    T::size_type index = 0;
}; // struct VectorWrapper

TEST_F(TestAttributeArray, testAttributeArrayCopy)
{
    using AttributeArrayD = TypedAttributeArray<double>;

    Index size(50);

    // initialize some test data

    AttributeArrayD sourceTypedAttr(size);
    AttributeArray& sourceAttr(sourceTypedAttr);
    EXPECT_EQ(size, sourceAttr.size());

    sourceAttr.expand();
    for (Index i = 0; i < size; i++) {
        sourceTypedAttr.set(i, double(i)/2);
    }

    // initialize source -> target pairs that reverse the order

    std::vector<std::pair<Index, Index>> indexPairs;
    for (Index i = 0; i < size; i++) {
        indexPairs.push_back(std::make_pair(i, size-i-1));
    }

    // create a new index pair wrapper

    VectorWrapper wrapper(indexPairs);

    // build a target attribute array

    AttributeArrayD targetTypedAttr(size);
    AttributeArray& targetAttr(targetTypedAttr);
    for (const auto& pair : indexPairs) {
        targetTypedAttr.set(pair.second, sourceTypedAttr.get(pair.first));
    }

    using AttributeArrayF = TypedAttributeArray<float>;

    { // use std::vector<std::pair<Index, Index>>::begin() as iterator to AttributeArray::copy()
        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        attr.copyValues(sourceAttr, wrapper);

        EXPECT_TRUE(targetAttr == attr);
    }

    { // attempt to copy values between attribute arrays with different storage sizes
        AttributeArrayF typedAttr(size);
        AttributeArray& attr(typedAttr);

        EXPECT_THROW(attr.copyValues(sourceAttr, wrapper), TypeError);
    }

    { // attempt to copy values between integer and float attribute arrays
        AttributeArrayF typedAttr(size);
        AttributeArray& attr(typedAttr);

        EXPECT_THROW(attr.copyValues(sourceAttr, wrapper), TypeError);
    }

    { // copy values between attribute arrays with different value types, but the same storage type
        // target half array
        TypedAttributeArray<math::half> targetTypedAttr1(size);
        AttributeArray& targetAttr1(targetTypedAttr1);
        for (Index i = 0; i < size; i++) {
            targetTypedAttr1.set(i,
                io::RealToHalf<double>::convert(sourceTypedAttr.get(i)));
        }

        // truncated float array
        TypedAttributeArray<float, TruncateCodec> targetTypedAttr2(size);
        AttributeArray& targetAttr2(targetTypedAttr2);

        targetAttr2.copyValues(targetAttr1, wrapper);

        // equality fails as attribute types are not the same
        EXPECT_TRUE(targetAttr2 != targetAttr);
        EXPECT_TRUE(targetAttr2.type() != targetAttr.type());
        // however testing value equality succeeds
        for (Index i = 0; i < size; i++) {
            EXPECT_TRUE(targetTypedAttr2.get(i) == targetTypedAttr.get(i));
        }
    }

    { // out-of-range checking
        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        decltype(indexPairs) rangeIndexPairs(indexPairs);

        rangeIndexPairs[10].first = size+1;

        VectorWrapper rangeWrapper(rangeIndexPairs);

        EXPECT_THROW(attr.copyValues(sourceAttr, rangeWrapper), IndexError);

        rangeIndexPairs[10].first = 0;

        EXPECT_NO_THROW(attr.copyValues(sourceAttr, rangeWrapper));

        rangeIndexPairs[10].second = size+1;

        EXPECT_THROW(attr.copyValues(sourceAttr, rangeWrapper), IndexError);
    }

    { // source attribute array is uniform
        AttributeArrayD uniformTypedAttr(size);
        AttributeArray& uniformAttr(uniformTypedAttr);

        uniformTypedAttr.collapse(5.3);

        EXPECT_TRUE(uniformAttr.isUniform());

        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        EXPECT_TRUE(attr.isUniform());

        attr.copyValues(uniformAttr, wrapper);

        EXPECT_TRUE(attr.isUniform());

        attr.copyValues(uniformAttr, wrapper, /*preserveUniformity=*/false);

        EXPECT_TRUE(!attr.isUniform());

        typedAttr.collapse(1.4);

        EXPECT_TRUE(attr.isUniform());

        // resize the vector to be smaller than the size of the array

        decltype(indexPairs) subsetIndexPairs(indexPairs);
        subsetIndexPairs.resize(size-1);

        decltype(wrapper) subsetWrapper(subsetIndexPairs);

        // now copy the values attempting to preserve uniformity

        attr.copyValues(uniformAttr, subsetWrapper, /*preserveUniformity=*/true);

        // verify that the array cannot be kept uniform

        EXPECT_TRUE(!attr.isUniform());
    }

    { // target attribute array is uniform
        AttributeArrayD uniformTypedAttr(size);
        AttributeArray& uniformAttr(uniformTypedAttr);

        uniformTypedAttr.collapse(5.3);

        EXPECT_TRUE(uniformAttr.isUniform());

        AttributeArrayD typedAttr(size);
        AttributeArray& attr(typedAttr);

        typedAttr.set(5, 1.2);
        typedAttr.set(10, 3.1);

        EXPECT_TRUE(!attr.isUniform());

        std::vector<std::pair<Index, Index>> uniformIndexPairs;
        uniformIndexPairs.push_back(std::make_pair(10, 0));
        uniformIndexPairs.push_back(std::make_pair(5, 0));
        VectorWrapper uniformWrapper(uniformIndexPairs);

        // note that calling copyValues() will implicitly expand the uniform target

        EXPECT_NO_THROW(uniformAttr.copyValuesUnsafe(attr, uniformWrapper));

        EXPECT_TRUE(uniformAttr.isUniform());
        EXPECT_TRUE(uniformTypedAttr.get(0) == typedAttr.get(5));
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
        static void setterError [[noreturn]] (AttributeArray* /*array*/,
            const Index /*n*/, const float& /*value*/)
        {
            OPENVDB_THROW(NotImplementedError, "");
        }

        //static float testGetter(const AttributeArray* array, const Index n) {
        //    return AccessorEval<UnknownCodec, float>::get(&getterError, array, n);
        //}
        //static void testSetter(AttributeArray* array, const Index n, const float& value) {
        //    AccessorEval<UnknownCodec, float>::set(&setterError, array, n, value);
        //}
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

        const AttributeArray& constArray(array);
        EXPECT_EQ(&constArray, &handle.array());

        handle.mGetter = TestAccessor::getterError;

        const float result1 = handle.get(4);
        const float result2 = handle.get(6);

        EXPECT_EQ(15.0f, result1);
        EXPECT_EQ(5.0f, result2);
    }

    { // test get and set (UnknownCodec)
        AttributeF::Ptr attr = AttributeF::create(10);
        attr->collapse(5.0f);
        attr->expand();

        AttributeArray& array = *attr;

        // unknown codec is used here so getter and setter are called

        AttributeWriteHandle<float, UnknownCodec> writeHandle(array);

        EXPECT_EQ(&array, &writeHandle.array());

        writeHandle.mSetter = TestAccessor::setterError;

        EXPECT_THROW(writeHandle.set(4, 15.0f), NotImplementedError);

        AttributeHandle<float, UnknownCodec> handle(array);

        handle.mGetter = TestAccessor::getterError;

        EXPECT_THROW(handle.get(4), NotImplementedError);
    }
}
TEST_F(TestAttributeArray, testAccessorEval) { testAccessorEval(); }


TEST_F(TestAttributeArray, testAttributeHandle)
{
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

    unsigned count = 500;
    AttributeSet attrSet(descr, /*arrayLength=*/count);

    attrSet.appendAttribute("truncate", AttributeFH::attributeType());
    attrSet.appendAttribute("int", AttributeI::attributeType());

    // check uniform value implementation

    {
        AttributeArray* array = attrSet.get(2);

        AttributeHandleRWI nonExpandingHandle(*array, /*expand=*/false);
        EXPECT_TRUE(nonExpandingHandle.isUniform());

        AttributeHandleRWI handle(*array);
        EXPECT_TRUE(!handle.isUniform());

        EXPECT_EQ(array->size(), handle.size());

        EXPECT_EQ(0, handle.get(0));
        EXPECT_EQ(0, handle.get(10));

        handle.set(0, 10);
        EXPECT_TRUE(!handle.isUniform());

        handle.collapse(5);
        EXPECT_TRUE(handle.isUniform());

        EXPECT_EQ(5, handle.get(0));
        EXPECT_EQ(5, handle.get(20));

        handle.expand();
        EXPECT_TRUE(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(5, handle.get(i));
        }

        EXPECT_TRUE(handle.compact());
        EXPECT_TRUE(handle.isUniform());

        handle.expand();

        handle.fill(10);
        EXPECT_TRUE(!handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(10, handle.get(i));
        }

        handle.collapse(7);
        EXPECT_TRUE(handle.isUniform());

        EXPECT_EQ(7, handle.get(0));
        EXPECT_EQ(7, handle.get(20));

        handle.fill(5);
        EXPECT_TRUE(handle.isUniform());

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(5, handle.get(i));
        }

        EXPECT_TRUE(handle.isUniform());
    }

    {
        AttributeArray* array = attrSet.get(0);

        AttributeWriteHandle<Vec3f> handle(*array);

        handle.set(5, Vec3f(10));

        EXPECT_EQ(Vec3f(10), handle.get(5));
    }

    {
        AttributeArray* array = attrSet.get(1);

        AttributeWriteHandle<float> handle(*array);

        handle.set(6, float(11));

        EXPECT_EQ(float(11), handle.get(6));

        {
            AttributeHandle<float> handleRO(*array);

            EXPECT_EQ(float(11), handleRO.get(6));
        }
    }

    // check values have been correctly set without using handles

    {
        AttributeVec3f* array = static_cast<AttributeVec3f*>(attrSet.get(0));

        EXPECT_TRUE(array);

        EXPECT_EQ(Vec3f(10), array->get(5));
    }

    {
        AttributeFH* array = static_cast<AttributeFH*>(attrSet.get(1));

        EXPECT_TRUE(array);

        EXPECT_EQ(float(11), array->get(6));
    }
}

TEST_F(TestAttributeArray, testStrided)
{
    using AttributeArrayI       = TypedAttributeArray<int>;
    using StridedHandle         = AttributeHandle<int, /*CodecType=*/UnknownCodec>;
    using StridedWriteHandle    = AttributeWriteHandle<int, /*CodecType=*/UnknownCodec>;

    { // non-strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/1);
        EXPECT_TRUE(array->hasConstantStride());
        EXPECT_EQ(Index(1), array->stride());
        EXPECT_EQ(Index(2), array->size());
        EXPECT_EQ(Index(2), array->dataSize());
    }

    { // strided array
        AttributeArrayI::Ptr array = AttributeArrayI::create(/*n=*/2, /*stride=*/3);

        EXPECT_TRUE(array->hasConstantStride());

        EXPECT_EQ(Index(3), array->stride());
        EXPECT_EQ(Index(2), array->size());
        EXPECT_EQ(Index(6), array->dataSize());
        EXPECT_TRUE(array->isUniform());

        EXPECT_EQ(0, array->get(0));
        EXPECT_EQ(0, array->get(5));
        EXPECT_THROW(array->get(6), IndexError); // out-of-range

        EXPECT_NO_THROW(StridedHandle::create(*array));
        EXPECT_NO_THROW(StridedWriteHandle::create(*array));

        array->collapse(10);

        EXPECT_EQ(int(10), array->get(0));
        EXPECT_EQ(int(10), array->get(5));

        array->expand();

        EXPECT_EQ(int(10), array->get(0));
        EXPECT_EQ(int(10), array->get(5));

        array->collapse(0);

        EXPECT_EQ(int(0), array->get(0));
        EXPECT_EQ(int(0), array->get(5));

        StridedWriteHandle writeHandle(*array);

        writeHandle.set(0, 2, 5);
        writeHandle.set(1, 1, 10);

        EXPECT_EQ(Index(3), writeHandle.stride());
        EXPECT_EQ(Index(2), writeHandle.size());

        // non-interleaved: 0 0 5 0 10 0

        EXPECT_EQ(5, array->get(2));
        EXPECT_EQ(10, array->get(4));

        EXPECT_EQ(5, writeHandle.get(0, 2));
        EXPECT_EQ(10, writeHandle.get(1, 1));

        StridedHandle handle(*array);
        EXPECT_TRUE(handle.hasConstantStride());

        EXPECT_EQ(5, handle.get(0, 2));
        EXPECT_EQ(10, handle.get(1, 1));

        EXPECT_EQ(Index(3), handle.stride());
        EXPECT_EQ(Index(2), handle.size());

        size_t arrayMem = 40;
        EXPECT_EQ(sizeof(int) * /*size*/3 * /*stride*/2 + arrayMem, array->memUsage());
    }

    { // dynamic stride
        AttributeArrayI::Ptr array = AttributeArrayI::create(
            /*n=*/2, /*stride=*/7, /*constantStride=*/false);

        EXPECT_TRUE(!array->hasConstantStride());

        // zero indicates dynamic striding
        EXPECT_EQ(Index(0), array->stride());
        EXPECT_EQ(Index(2), array->size());
        // the actual array size
        EXPECT_EQ(Index(7), array->dataSize());
        EXPECT_TRUE(array->isUniform());

        EXPECT_EQ(0, array->get(0));
        EXPECT_EQ(0, array->get(6));
        EXPECT_THROW(array->get(7), IndexError); // out-of-range

        EXPECT_NO_THROW(StridedHandle::create(*array));
        EXPECT_NO_THROW(StridedWriteHandle::create(*array));

        // handle is bound as if a linear array with stride 1
        StridedHandle handle(*array);
        EXPECT_TRUE(!handle.hasConstantStride());
        EXPECT_EQ(Index(1), handle.stride());
        EXPECT_EQ(array->dataSize(), handle.size());
    }

    { // IO
        const Index count = 50, total = 100;
        AttributeArrayI attrA(count, total, /*constantStride=*/false);

        for (unsigned i = 0; i < unsigned(total); ++i) {
            attrA.set(i, int(i));
        }

        std::ostringstream ostr(std::ios_base::binary);
        io::setDataCompression(ostr, io::COMPRESS_BLOSC);
        attrA.write(ostr);

        AttributeArrayI attrB;
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        EXPECT_TRUE(attrA == attrB);
    }
}

#ifdef OPENVDB_USE_DELAYED_LOADING
void
TestAttributeArray::testDelayedLoad()
{
    using AttributeArrayI = TypedAttributeArray<int>;
    using AttributeArrayF = TypedAttributeArray<float>;

    AttributeArrayI::registerType();
    AttributeArrayF::registerType();

    SharedPtr<io::MappedFile> mappedFile;

    io::StreamMetadata::Ptr streamMetadata(new io::StreamMetadata);

    std::string tempDir;
    if (const char* dir = std::getenv("TMPDIR")) tempDir = dir;
#ifdef _WIN32
    if (tempDir.empty()) {
        char tempDirBuffer[MAX_PATH+1];
        int tempDirLen = GetTempPath(MAX_PATH+1, tempDirBuffer);
        EXPECT_TRUE(tempDirLen > 0 && tempDirLen <= MAX_PATH);
        tempDir = tempDirBuffer;
    }
#else
    if (tempDir.empty()) tempDir = P_tmpdir;
#endif

    { // IO
        const Index count = 50;
        AttributeArrayI attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        AttributeArrayF attrA2(count);

        std::string filename;

        // write out attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed1";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrA.writeMetadata(fileout, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrA.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrA.writePagedBuffers(outputStream, false);
            outputStream.flush();

            attrA2.writeMetadata(fileout, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize2(fileout);
            outputStreamSize2.setSizeOnly(true);
            attrA2.writePagedBuffers(outputStreamSize2, false);
            outputStreamSize2.flush();
            compression::PagedOutputStream outputStream2(fileout);
            outputStream2.setSizeOnly(false);
            attrA2.writePagedBuffers(outputStream2, false);
            outputStream2.flush();

            fileout.close();
        }

        mappedFile = TestMappedFile::create(filename);

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;
            AttributeArrayF attrB2;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(matchingNamePairs(attrA.type(), attrB.type()));
            EXPECT_EQ(attrA.size(), attrB.size());
            EXPECT_EQ(attrA.isUniform(), attrB.isUniform());
            EXPECT_EQ(attrA.isTransient(), attrB.isTransient());
            EXPECT_EQ(attrA.isHidden(), attrB.isHidden());

            AttributeArrayI attrBcopy(attrB);
            AttributeArrayI attrBequal = attrB;

            EXPECT_TRUE(attrB.isOutOfCore());
            EXPECT_TRUE(attrBcopy.isOutOfCore());
            EXPECT_TRUE(attrBequal.isOutOfCore());

            EXPECT_TRUE(!static_cast<AttributeArray&>(attrB).isDataLoaded());
            EXPECT_TRUE(!static_cast<AttributeArray&>(attrBcopy).isDataLoaded());
            EXPECT_TRUE(!static_cast<AttributeArray&>(attrBequal).isDataLoaded());

            attrB.loadData();
            attrBcopy.loadData();
            attrBequal.loadData();

            EXPECT_TRUE(!attrB.isOutOfCore());
            EXPECT_TRUE(!attrBcopy.isOutOfCore());
            EXPECT_TRUE(!attrBequal.isOutOfCore());

            EXPECT_TRUE(static_cast<AttributeArray&>(attrB).isDataLoaded());
            EXPECT_TRUE(static_cast<AttributeArray&>(attrBcopy).isDataLoaded());
            EXPECT_TRUE(static_cast<AttributeArray&>(attrBequal).isDataLoaded());

            EXPECT_EQ(attrA.memUsage(), attrB.memUsage());
            EXPECT_EQ(attrA.memUsage(), attrBcopy.memUsage());
            EXPECT_EQ(attrA.memUsage(), attrBequal.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
                EXPECT_EQ(attrA.get(i), attrBcopy.get(i));
                EXPECT_EQ(attrA.get(i), attrBequal.get(i));
            }

            attrB2.readMetadata(filein);
            compression::PagedInputStream inputStream2(filein);
            inputStream2.setSizeOnly(true);
            attrB2.readPagedBuffers(inputStream2);
            inputStream2.setSizeOnly(false);
            attrB2.readPagedBuffers(inputStream2);

            EXPECT_TRUE(matchingNamePairs(attrA2.type(), attrB2.type()));
            EXPECT_EQ(attrA2.size(), attrB2.size());
            EXPECT_EQ(attrA2.isUniform(), attrB2.isUniform());
            EXPECT_EQ(attrA2.isTransient(), attrB2.isTransient());
            EXPECT_EQ(attrA2.isHidden(), attrB2.isHidden());

            AttributeArrayF attrB2copy(attrB2);
            AttributeArrayF attrB2equal = attrB2;

            EXPECT_TRUE(attrB2.isOutOfCore());
            EXPECT_TRUE(attrB2copy.isOutOfCore());
            EXPECT_TRUE(attrB2equal.isOutOfCore());
            attrB2.loadData();
            attrB2copy.loadData();
            attrB2equal.loadData();

            EXPECT_TRUE(!attrB2.isOutOfCore());
            EXPECT_TRUE(!attrB2copy.isOutOfCore());
            EXPECT_TRUE(!attrB2equal.isOutOfCore());

            EXPECT_EQ(attrA2.memUsage(), attrB2.memUsage());
            EXPECT_EQ(attrA2.memUsage(), attrB2copy.memUsage());
            EXPECT_EQ(attrA2.memUsage(), attrB2equal.memUsage());

            EXPECT_EQ(attrA2.get(0), attrB2.get(0));
            EXPECT_EQ(attrA2.get(0), attrB2copy.get(0));
            EXPECT_EQ(attrA2.get(0), attrB2equal.get(0));
        }

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            EXPECT_TRUE(!attrB.isUniform());

            attrB.fill(5);

            EXPECT_TRUE(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(5, attrB.get(i));
            }
        }

        // read in using delayed load and check streaming (write handle)
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            EXPECT_TRUE(!attrB.isUniform());

            attrB.setStreaming(true);

            {
                AttributeWriteHandle<int> handle(attrB);
                EXPECT_TRUE(!attrB.isOutOfCore());
                EXPECT_TRUE(!attrB.isUniform());
            }

            EXPECT_TRUE(!attrB.isUniform());
        }

        // read in using delayed load and check streaming (read handle)
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            EXPECT_TRUE(!attrB.isUniform());

            attrB.setStreaming(true);

            {
                AttributeHandle<int> handle(attrB);
                EXPECT_TRUE(!attrB.isOutOfCore());
                EXPECT_TRUE(!attrB.isUniform());
            }

            EXPECT_TRUE(attrB.isUniform());
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            attrB.get(0);

            EXPECT_TRUE(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check implicit load through compress()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());
        }

        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

            EXPECT_TRUE(attrB.isOutOfCore());
            EXPECT_TRUE(attrC.isOutOfCore());
            EXPECT_TRUE(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();

            EXPECT_TRUE(!attrB.isOutOfCore());
            EXPECT_TRUE(!attrC.isOutOfCore());
            EXPECT_TRUE(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
                EXPECT_EQ(attrA.get(i), attrC.get(i));
                EXPECT_EQ(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            AttributeHandle<int> handle(attrB);

            EXPECT_TRUE(!attrB.isOutOfCore());
        }

        // read in using delayed load and check detaching of file (using collapse())
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            EXPECT_TRUE(!attrB.isUniform());

            attrB.collapse();

            EXPECT_TRUE(!attrB.isOutOfCore());
            EXPECT_TRUE(attrB.isUniform());

            EXPECT_EQ(0, attrB.get(0));
        }

        // read in and write out using delayed load to check writing out-of-core attributes
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            std::string filename2 = tempDir + "/openvdb_delayed5";
            std::ofstream fileout2(filename2.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout2, streamMetadata);
            io::setDataCompression(fileout2, io::COMPRESS_BLOSC);

            attrB.writeMetadata(fileout2, false, /*paged=*/true);
            compression::PagedOutputStream outputStreamSize(fileout2);
            outputStreamSize.setSizeOnly(true);
            attrB.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout2);
            outputStream.setSizeOnly(false);
            attrB.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout2.close();

            AttributeArrayI attrB2;

            std::ifstream filein2(filename2.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein2, streamMetadata);
            io::setMappedFilePtr(filein2, mappedFile);

            attrB2.readMetadata(filein2);
            compression::PagedInputStream inputStream2(filein2);
            inputStream2.setSizeOnly(true);
            attrB2.readPagedBuffers(inputStream2);
            inputStream2.setSizeOnly(false);
            attrB2.readPagedBuffers(inputStream2);

            EXPECT_TRUE(attrB2.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrB.get(i), attrB2.get(i));
            }

            filein2.close();
        }

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        AttributeArrayI attrUniform(count);

        // write out uniform attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed2";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrUniform.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrUniform.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrUniform.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        mappedFile = TestMappedFile::create(filename);

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isUniform());

            attrB.fill(5);

            EXPECT_TRUE(attrB.isUniform());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(5, attrB.get(i));
            }
        }

        AttributeArrayI attrStrided(count, /*stride=*/3);

        EXPECT_EQ(Index(3), attrStrided.stride());

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out strided attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed3";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrStrided.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrStrided.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrStrided.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        mappedFile = TestMappedFile::create(filename);

        // read in using delayed load and check fill()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_EQ(Index(3), attrB.stride());
        }

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out compressed attribute array to a temp file
        {
            filename = tempDir + "/openvdb_delayed4";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            attrA.writeMetadata(fileout, false, /*paged=*/true);

            compression::PagedOutputStream outputStreamSize(fileout);
            outputStreamSize.setSizeOnly(true);
            attrA.writePagedBuffers(outputStreamSize, false);
            outputStreamSize.flush();
            compression::PagedOutputStream outputStream(fileout);
            outputStream.setSizeOnly(false);
            attrA.writePagedBuffers(outputStream, false);
            outputStream.flush();

            fileout.close();
        }

        mappedFile = TestMappedFile::create(filename);

        // read in using delayed load and check manual loading of data
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());
            attrB.loadData();
            EXPECT_TRUE(!attrB.isOutOfCore());

            EXPECT_EQ(attrA.memUsage(), attrB.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
            }
        }

        // read in using delayed load and check partial read state
        {
            std::unique_ptr<AttributeArrayI> attrB(new AttributeArrayI);

            EXPECT_TRUE(!(attrB->flags() & AttributeArray::PARTIALREAD));

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB->readMetadata(filein);

            // PARTIALREAD flag should now be set
            EXPECT_TRUE(attrB->flags() & AttributeArray::PARTIALREAD);

            // copy-construct and assign AttributeArray
            AttributeArrayI attrC(*attrB);
            EXPECT_TRUE(attrC.flags() & AttributeArray::PARTIALREAD);
            AttributeArrayI attrD = *attrB;
            EXPECT_TRUE(attrD.flags() & AttributeArray::PARTIALREAD);

            // verify deleting attrB is safe
            attrB.reset();

            // verify data is not valid
            EXPECT_TRUE(!attrC.validData());

            { // attempting to write a partially-read AttributeArray throws
                std::string filename = tempDir + "/openvdb_partial1";
                ScopedFile f(filename);
                std::ofstream fileout(filename.c_str(), std::ios_base::binary);
                io::setStreamMetadataPtr(fileout, streamMetadata);
                io::setDataCompression(fileout, io::COMPRESS_BLOSC);

                EXPECT_THROW(attrC.writeMetadata(fileout, false, /*paged=*/true), IoError);
            }

            // continue loading with copy-constructed AttributeArray

            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrC.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrC.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrC.isOutOfCore());
            attrC.loadData();
            EXPECT_TRUE(!attrC.isOutOfCore());

            // verify data is now valid
            EXPECT_TRUE(attrC.validData());

            EXPECT_EQ(attrA.memUsage(), attrC.memUsage());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrC.get(i));
            }
        }

        // read in using delayed load and check implicit load through get()
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            attrB.get(0);

            EXPECT_TRUE(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
            }
        }

#ifdef OPENVDB_USE_BLOSC
        // read in using delayed load and check copy and assignment constructors
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            AttributeArrayI attrC(attrB);
            AttributeArrayI attrD = attrB;

            EXPECT_TRUE(attrB.isOutOfCore());
            EXPECT_TRUE(attrC.isOutOfCore());
            EXPECT_TRUE(attrD.isOutOfCore());

            attrB.loadData();
            attrC.loadData();
            attrD.loadData();

            EXPECT_TRUE(!attrB.isOutOfCore());
            EXPECT_TRUE(!attrC.isOutOfCore());
            EXPECT_TRUE(!attrD.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), attrB.get(i));
                EXPECT_EQ(attrA.get(i), attrC.get(i));
                EXPECT_EQ(attrA.get(i), attrD.get(i));
            }
        }

        // read in using delayed load and check implicit load through AttributeHandle
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            attrB.readMetadata(filein);
            compression::PagedInputStream inputStream(filein);
            inputStream.setSizeOnly(true);
            attrB.readPagedBuffers(inputStream);
            inputStream.setSizeOnly(false);
            attrB.readPagedBuffers(inputStream);

            EXPECT_TRUE(attrB.isOutOfCore());

            AttributeHandle<int> handle(attrB);

            EXPECT_TRUE(!attrB.isOutOfCore());

            for (unsigned i = 0; i < unsigned(count); ++i) {
                EXPECT_EQ(attrA.get(i), handle.get(i));
            }
        }
#endif

        // Clean up temp files.
        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());

        // write out invalid serialization flags as metadata to a temp file
        {
            filename = tempDir + "/openvdb_delayed5";
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);
            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, io::COMPRESS_BLOSC);

            // write out unknown serialization flags to check forwards-compatibility

            Index64 bytes(0);
            uint8_t flags(0);
            uint8_t serializationFlags(Int16(0x10));
            Index size(0);

            fileout.write(reinterpret_cast<const char*>(&bytes), sizeof(Index64));
            fileout.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));
            fileout.write(reinterpret_cast<const char*>(&serializationFlags), sizeof(uint8_t));
            fileout.write(reinterpret_cast<const char*>(&size), sizeof(Index));

            fileout.close();
        }

        mappedFile = TestMappedFile::create(filename);

        // read in using delayed load and check metadata fail due to serialization flags
        {
            AttributeArrayI attrB;

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            EXPECT_THROW(attrB.readMetadata(filein), openvdb::IoError);
        }

        // cleanup temp files

        std::remove(mappedFile->filename().c_str());
        std::remove(filename.c_str());
    }
}
TEST_F(TestAttributeArray, testDelayedLoad) { testDelayedLoad(); }
#endif


TEST_F(TestAttributeArray, testDefaultValue)
{
    using AttributeArrayF = TypedAttributeArray<float>;
    using AttributeArrayI = TypedAttributeArray<int>;

    AttributeArrayI::registerType();
    AttributeArrayF::registerType();

    TypedMetadata<float> defaultValue(5.4f);
    Metadata& baseDefaultValue = defaultValue;

    // default value is same value type

    AttributeArray::Ptr attr =
        AttributeArrayF::create(10, 1, true, &baseDefaultValue);
    EXPECT_TRUE(attr);
    EXPECT_EQ(5.4f, AttributeArrayF::cast(*attr).get(0));

    // default value is different value type, so not used

    attr = AttributeArrayI::create(10, 1, true, &baseDefaultValue);
    EXPECT_TRUE(attr);
    EXPECT_EQ(0, AttributeArrayI::cast(*attr).get(0));
}


TEST_F(TestAttributeArray, testQuaternions)
{
    using AttributeQF = TypedAttributeArray<math::Quat<float>>;
    using AttributeQD = TypedAttributeArray<QuatR>;

    AttributeQF::registerType();
    AttributeQD::registerType();

    EXPECT_TRUE(AttributeQF::attributeType().first == "quats");
    EXPECT_TRUE(AttributeQD::attributeType().first == "quatd");

    AttributeQF test(/*size=*/5);

    AttributeQD orient(/*size=*/10);

    { // set some quaternion values
        AttributeWriteHandle<QuatR> orientHandle(orient);

        orientHandle.set(4, QuatR(1, 2, 3, 4));
        orientHandle.set(7, QuatR::identity());
    }

    { // get some quaternion values
        AttributeHandle<QuatR> orientHandle(orient);

        EXPECT_EQ(QuatR::zero(), orientHandle.get(3));
        EXPECT_EQ(QuatR(1, 2, 3, 4), orientHandle.get(4));
        EXPECT_EQ(QuatR::identity(), orientHandle.get(7));
    }

    { // create a quaternion array with a zero uniform value
        AttributeQD zero(/*size=*/10, /*stride=*/1, /*constantStride=*/true, QuatR::zero());

        EXPECT_EQ(QuatR::zero(), zero.get(5));
    }
}


TEST_F(TestAttributeArray, testMatrices)
{
    typedef TypedAttributeArray<Mat4d>      AttributeM;

    AttributeM::registerType();

    EXPECT_TRUE(AttributeM::attributeType().first == "mat4d");

    AttributeM matrix(/*size=*/10);

    Mat4d testMatrix(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    { // set some matrix values
        AttributeWriteHandle<Mat4d> matrixHandle(matrix);

        matrixHandle.set(4, testMatrix);
        matrixHandle.set(7, Mat4d::zero());
    }

    { // get some matrix values
        AttributeHandle<Mat4d> matrixHandle(matrix);

        EXPECT_EQ(Mat4d::zero(), matrixHandle.get(3));
        EXPECT_EQ(testMatrix, matrixHandle.get(4));
        EXPECT_EQ(Mat4d::zero(), matrixHandle.get(7));
    }

    { // create a matrix array with a zero uniform value
        AttributeM zero(/*size=*/10, /*stride=*/1, /*constantStride=*/true, Mat4d::zero());

        EXPECT_EQ(Mat4d::zero(), zero.get(5));
    }
}


namespace profile {

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
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        attr.setUnsafe(i, typename AttrT::ValueType(i));
    }
}

template <typename CodecT, typename AttrT>
void setH(const Name& prefix, AttrT& attr)
{
    using ValueType = typename AttrT::ValueType;
    ProfileTimer timer(prefix + ": setHandle");
    AttributeWriteHandle<ValueType, CodecT> handle(attr);
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        handle.set(i, ValueType(i));
    }
}

template <typename AttrT>
void sum(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sum");
    using ValueType = typename AttrT::ValueType;
    ValueType sum = 0;
    const Index size = attr.size();
    for (Index i = 0; i < size; i++) {
        sum += attr.getUnsafe(i);
    }
    // prevent compiler optimisations removing computation
    EXPECT_TRUE(sum!=ValueType());
}

template <typename CodecT, typename AttrT>
void sumH(const Name& prefix, const AttrT& attr)
{
    ProfileTimer timer(prefix + ": sumHandle");
    using ValueType = typename AttrT::ValueType;
    ValueType sum = 0;
    AttributeHandle<ValueType, CodecT> handle(attr);
    for (Index i = 0; i < attr.size(); i++) {
        sum += handle.get(i);
    }
    // prevent compiler optimisations removing computation
    EXPECT_TRUE(sum!=ValueType());
}

} // namespace profile

TEST_F(TestAttributeArray, testProfile)
{
    using namespace openvdb::util;
    using namespace openvdb::math;

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
            EXPECT_TRUE(sum!=0.0f);
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
