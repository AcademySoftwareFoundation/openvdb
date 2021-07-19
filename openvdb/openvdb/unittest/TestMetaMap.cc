// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/util/logging.h>
#include <openvdb/Metadata.h>
#include <openvdb/MetaMap.h>

#include <gtest/gtest.h>


class TestMetaMap: public ::testing::Test
{
};


TEST_F(TestMetaMap, testInsert)
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    MetaMap::MetaIterator iter = meta.beginMeta();
    int i = 1;
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        if(i == 1) {
            EXPECT_TRUE(iter->first.compare("meta1") == 0);
            std::string val = meta.metaValue<std::string>("meta1");
            EXPECT_TRUE(val == "testing");
        } else if(i == 2) {
            EXPECT_TRUE(iter->first.compare("meta2") == 0);
            int32_t val = meta.metaValue<int32_t>("meta2");
            EXPECT_TRUE(val == 20);
        } else if(i == 3) {
            EXPECT_TRUE(iter->first.compare("meta3") == 0);
            float val = meta.metaValue<float>("meta3");
            //EXPECT_TRUE(val == 2.0);
            EXPECT_NEAR(2.0f,val,0);
        }
    }
}

TEST_F(TestMetaMap, testRemove)
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    meta.removeMeta("meta2");

    MetaMap::MetaIterator iter = meta.beginMeta();
    int i = 1;
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        if(i == 1) {
            EXPECT_TRUE(iter->first.compare("meta1") == 0);
            std::string val = meta.metaValue<std::string>("meta1");
            EXPECT_TRUE(val == "testing");
        } else if(i == 2) {
            EXPECT_TRUE(iter->first.compare("meta3") == 0);
            float val = meta.metaValue<float>("meta3");
            //EXPECT_TRUE(val == 2.0);
            EXPECT_NEAR(2.0f,val,0);
        }
    }

    meta.removeMeta("meta1");

    iter = meta.beginMeta();
    for( ; iter != meta.endMeta(); ++iter, ++i) {
        EXPECT_TRUE(iter->first.compare("meta3") == 0);
        float val = meta.metaValue<float>("meta3");
        //EXPECT_TRUE(val == 2.0);
        EXPECT_NEAR(2.0f,val,0);
    }

    meta.removeMeta("meta3");

    EXPECT_EQ(0, int(meta.metaCount()));
}

TEST_F(TestMetaMap, testGetMetadata)
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", DoubleMetadata(2.0));

    Metadata::Ptr metadata = meta["meta2"];
    EXPECT_TRUE(metadata);
    EXPECT_TRUE(metadata->typeName().compare("int32") == 0);

    DoubleMetadata::Ptr dm = meta.getMetadata<DoubleMetadata>("meta3");
    //EXPECT_TRUE(dm->value() == 2.0);
    EXPECT_NEAR(2.0,dm->value(),0);

    const DoubleMetadata::Ptr cdm = meta.getMetadata<DoubleMetadata>("meta3");
    //EXPECT_TRUE(dm->value() == 2.0);
    EXPECT_NEAR(2.0,cdm->value(),0);

    EXPECT_TRUE(!meta.getMetadata<StringMetadata>("meta2"));

    EXPECT_THROW(meta.metaValue<int32_t>("meta3"),
                         openvdb::TypeError);

    EXPECT_THROW(meta.metaValue<double>("meta5"),
                         openvdb::LookupError);
}

TEST_F(TestMetaMap, testIO)
{
    using namespace openvdb;

    logging::LevelScope suppressLogging{logging::Level::Fatal};

    Metadata::clearRegistry();

    // Write some metadata using unregistered types.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", DoubleMetadata(2.0));
    std::ostringstream ostr(std::ios_base::binary);
    meta.writeMeta(ostr);

    // Verify that reading metadata of unregistered types is possible,
    // though the values cannot be retrieved.
    MetaMap meta2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    EXPECT_NO_THROW(meta2.readMeta(istr));
    EXPECT_EQ(3, int(meta2.metaCount()));

    // Verify that writing metadata of unknown type (i.e., UnknownMetadata) is possible.
    std::ostringstream ostrUnknown(std::ios_base::binary);
    meta2.writeMeta(ostrUnknown);

    // Register just one of the three types, then reread and verify that
    // the value of the registered type can be retrieved.
    Int32Metadata::registerType();
    istr.seekg(0, std::ios_base::beg);
    EXPECT_NO_THROW(meta2.readMeta(istr));
    EXPECT_EQ(3, int(meta2.metaCount()));
    EXPECT_EQ(meta.metaValue<int>("meta2"), meta2.metaValue<int>("meta2"));

    // Register the remaining types.
    StringMetadata::registerType();
    DoubleMetadata::registerType();

    {
        // Now seek to beginning and read again.
        istr.seekg(0, std::ios_base::beg);
        meta2.clearMetadata();

        EXPECT_NO_THROW(meta2.readMeta(istr));
        EXPECT_EQ(meta.metaCount(), meta2.metaCount());

        std::string val = meta.metaValue<std::string>("meta1");
        std::string val2 = meta2.metaValue<std::string>("meta1");
        EXPECT_EQ(0, val.compare(val2));

        int intval = meta.metaValue<int>("meta2");
        int intval2 = meta2.metaValue<int>("meta2");
        EXPECT_EQ(intval, intval2);

        double dval = meta.metaValue<double>("meta3");
        double dval2 = meta2.metaValue<double>("meta3");
        EXPECT_NEAR(dval, dval2,0);
    }
    {
        // Verify that metadata that was written as UnknownMetadata can
        // be read as typed metadata once the underlying types are registered.
        std::istringstream istrUnknown(ostrUnknown.str(), std::ios_base::binary);

        meta2.clearMetadata();
        EXPECT_NO_THROW(meta2.readMeta(istrUnknown));

        EXPECT_EQ(meta.metaCount(), meta2.metaCount());
        EXPECT_EQ(
            meta.metaValue<std::string>("meta1"), meta2.metaValue<std::string>("meta1"));
        EXPECT_EQ(meta.metaValue<int>("meta2"), meta2.metaValue<int>("meta2"));
        EXPECT_NEAR(
            meta.metaValue<double>("meta3"), meta2.metaValue<double>("meta3"), 0.0);
    }

    // Clear the registry once the test is done.
    Metadata::clearRegistry();
}

TEST_F(TestMetaMap, testEmptyIO)
{
    using namespace openvdb;

    MetaMap meta;

    // Write out an empty metadata
    std::ostringstream ostr(std::ios_base::binary);

    // Read in the metadata;
    MetaMap meta2;
    std::istringstream istr(ostr.str(), std::ios_base::binary);
    EXPECT_NO_THROW(meta2.readMeta(istr));

    EXPECT_TRUE(meta2.metaCount() == 0);
}

TEST_F(TestMetaMap, testCopyConstructor)
{
    using namespace openvdb;

    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    // copy constructor
    MetaMap meta2(meta);

    EXPECT_TRUE(meta.metaCount() == meta2.metaCount());

    std::string str = meta.metaValue<std::string>("meta1");
    std::string str2 = meta2.metaValue<std::string>("meta1");
    EXPECT_TRUE(str == str2);

    EXPECT_TRUE(meta.metaValue<int32_t>("meta2") ==
            meta2.metaValue<int32_t>("meta2"));

    EXPECT_NEAR(meta.metaValue<float>("meta3"),
                                 meta2.metaValue<float>("meta3"),0);
    //EXPECT_TRUE(meta.metaValue<float>("meta3") ==
    //        meta2.metaValue<float>("meta3"));
}


TEST_F(TestMetaMap, testCopyConstructorEmpty)
{
    using namespace openvdb;

    MetaMap meta;

    MetaMap meta2(meta);

    EXPECT_TRUE(meta.metaCount() == 0);
    EXPECT_TRUE(meta2.metaCount() == meta.metaCount());
}


TEST_F(TestMetaMap, testAssignment)
{
    using namespace openvdb;

    // Populate a map with data.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(2.0));

    // Create an empty map.
    MetaMap meta2;
    EXPECT_EQ(0, int(meta2.metaCount()));

    // Copy the first map to the second.
    meta2 = meta;
    EXPECT_EQ(meta.metaCount(), meta2.metaCount());

    // Verify that the contents of the two maps are the same.
    EXPECT_EQ(
        meta.metaValue<std::string>("meta1"), meta2.metaValue<std::string>("meta1"));
    EXPECT_EQ(meta.metaValue<int32_t>("meta2"), meta2.metaValue<int32_t>("meta2"));
    EXPECT_NEAR(
        meta.metaValue<float>("meta3"), meta2.metaValue<float>("meta3"), /*tolerance=*/0);

    // Verify that changing one map doesn't affect the other.
    meta.insertMeta("meta1", StringMetadata("changed"));
    std::string str = meta.metaValue<std::string>("meta1");
    EXPECT_EQ(std::string("testing"), meta2.metaValue<std::string>("meta1"));
}


TEST_F(TestMetaMap, testEquality)
{
    using namespace openvdb;

    // Populate a map with data.
    MetaMap meta;
    meta.insertMeta("meta1", StringMetadata("testing"));
    meta.insertMeta("meta2", Int32Metadata(20));
    meta.insertMeta("meta3", FloatMetadata(3.14159f));

    // Create an empty map.
    MetaMap meta2;

    // Verify that the two maps differ.
    EXPECT_TRUE(meta != meta2);
    EXPECT_TRUE(meta2 != meta);

    // Copy the first map to the second.
    meta2 = meta;

    // Verify that the two maps are equivalent.
    EXPECT_TRUE(meta == meta2);
    EXPECT_TRUE(meta2 == meta);

    // Modify the first map.
    meta.removeMeta("meta1");
    meta.insertMeta("abc", DoubleMetadata(2.0));

    // Verify that the two maps differ.
    EXPECT_TRUE(meta != meta2);
    EXPECT_TRUE(meta2 != meta);

    // Modify the second map and verify that the two maps differ.
    meta2 = meta;
    meta2.insertMeta("meta2", Int32Metadata(42));
    EXPECT_TRUE(meta != meta2);
    EXPECT_TRUE(meta2 != meta);

    meta2 = meta;
    meta2.insertMeta("meta3", FloatMetadata(2.0001f));
    EXPECT_TRUE(meta != meta2);
    EXPECT_TRUE(meta2 != meta);

    meta2 = meta;
    meta2.insertMeta("abc", DoubleMetadata(2.0001));
    EXPECT_TRUE(meta != meta2);
    EXPECT_TRUE(meta2 != meta);
}
