// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/Metadata.h>

#include <gtest/gtest.h>

#include <sstream>


class TestMetadata: public ::testing::Test
{
public:
    void SetUp() override { openvdb::Metadata::clearRegistry(); }
    void TearDown() override { openvdb::Metadata::clearRegistry(); }
};


TEST_F(TestMetadata, testMetadataRegistry)
{
    using namespace openvdb;

    Int32Metadata::registerType();

    StringMetadata strMetadata;

    EXPECT_TRUE(!Metadata::isRegisteredType(strMetadata.typeName()));

    StringMetadata::registerType();

    EXPECT_TRUE(Metadata::isRegisteredType(strMetadata.typeName()));
    EXPECT_TRUE(Metadata::isRegisteredType(Int32Metadata::staticTypeName()));

    Metadata::Ptr stringMetadata = Metadata::createMetadata(strMetadata.typeName());

    EXPECT_TRUE(stringMetadata->typeName() == strMetadata.typeName());

    StringMetadata::unregisterType();

    EXPECT_THROW(Metadata::createMetadata(strMetadata.typeName()), openvdb::LookupError);
}

TEST_F(TestMetadata, testMetadataAsBool)
{
    using namespace openvdb;

    {
        FloatMetadata meta(0.0);
        EXPECT_TRUE(!meta.asBool());
        meta.setValue(1.0);
        EXPECT_TRUE(meta.asBool());
        meta.setValue(-1.0);
        EXPECT_TRUE(meta.asBool());
        meta.setValue(999.0);
        EXPECT_TRUE(meta.asBool());
    }
    {
        Int32Metadata meta(0);
        EXPECT_TRUE(!meta.asBool());
        meta.setValue(1);
        EXPECT_TRUE(meta.asBool());
        meta.setValue(-1);
        EXPECT_TRUE(meta.asBool());
        meta.setValue(999);
        EXPECT_TRUE(meta.asBool());
    }
    {
        StringMetadata meta("");
        EXPECT_TRUE(!meta.asBool());
        meta.setValue("abc");
        EXPECT_TRUE(meta.asBool());
    }
    {
        Vec3IMetadata meta(Vec3i(0));
        EXPECT_TRUE(!meta.asBool());
        meta.setValue(Vec3i(-1, 0, 1));
        EXPECT_TRUE(meta.asBool());
    }
    {
        Vec3SMetadata meta(Vec3s(0.0));
        EXPECT_TRUE(!meta.asBool());
        meta.setValue(Vec3s(-1.0, 0.0, 1.0));
        EXPECT_TRUE(meta.asBool());
    }
    {
        Vec4DMetadata meta(Vec4d(0.0));
        EXPECT_TRUE(!meta.asBool());
        meta.setValue(Vec4d(1.0));
        EXPECT_TRUE(meta.asBool());
    }
}


TEST_F(TestMetadata, testCustomMetadata)
{
    using namespace openvdb;

    const Vec3i expected(1, 2, 3);
    std::ostringstream ostr(std::ios_base::binary);
    {
        Vec3IMetadata::registerType();
        Vec3IMetadata meta(expected);

        // Write Vec3I metadata to a byte string.
        meta.write(ostr);
    }

    // Unregister Vec3I metadata.
    Metadata::clearRegistry();

    {
        std::istringstream istr(ostr.str(), std::ios_base::binary);

        UnknownMetadata meta;
        // Verify that metadata of an unregistered type can be read successfully.
        EXPECT_NO_THROW(meta.read(istr));

        // Verify that the metadata matches the original vector value.
        EXPECT_EQ(sizeof(Vec3i), size_t(meta.size()));
        EXPECT_TRUE(meta.value().size() == size_t(meta.size()));
        EXPECT_EQ(expected, *reinterpret_cast<const Vec3i*>(&meta.value()[0]));

        ostr.str("");
        meta.write(ostr);

        // Verify that UnknownMetadata can be copied.
        auto metaPtr = meta.copy();
        EXPECT_TRUE(metaPtr.get() != nullptr);
        EXPECT_TRUE(meta == *metaPtr);

        // Verify that typed metadata can be copied into UnknownMetadata.
        meta.copy(Vec3IMetadata(expected));
        EXPECT_EQ(sizeof(expected), size_t(meta.size()));
        const auto* ptr = reinterpret_cast<const uint8_t*>(&expected);
        EXPECT_TRUE(UnknownMetadata::ByteVec(ptr, ptr + sizeof(expected)) == meta.value());
    }

    Vec3IMetadata::registerType();

    {
        std::istringstream istr(ostr.str(), std::ios_base::binary);
        Vec3IMetadata meta;
        meta.read(istr);

        EXPECT_EQ(expected, meta.value());
    }
}
