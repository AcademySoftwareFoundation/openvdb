// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/AttributeArrayString.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/openvdb.h>

#include <gtest/gtest.h>

#include <iostream>

using namespace openvdb;
using namespace openvdb::points;

class TestAttributeArrayString: public ::testing::Test
{
public:
    void SetUp() override { openvdb::initialize(); }
    void TearDown() override { openvdb::uninitialize(); }
}; // class TestAttributeArrayString


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


TEST_F(TestAttributeArrayString, testStringMetaCache)
{
    { // cache with manual insertion
        StringMetaCache cache;
        EXPECT_TRUE(cache.empty());
        EXPECT_EQ(size_t(0), cache.size());

        cache.insert("test", 1);

        EXPECT_TRUE(!cache.empty());
        EXPECT_EQ(size_t(1), cache.size());

        auto it = cache.map().find("test");
        EXPECT_TRUE(it != cache.map().end());
    }

    { // cache with metadata insertion and reset
        MetaMap metadata;

        StringMetaInserter inserter(metadata);
        inserter.insert("test1");
        inserter.insert("test2");

        StringMetaCache cache(metadata);
        EXPECT_TRUE(!cache.empty());
        EXPECT_EQ(size_t(2), cache.size());

        auto it = cache.map().find("test1");
        EXPECT_TRUE(it != cache.map().end());
        EXPECT_EQ(Name("test1"), it->first);
        EXPECT_EQ(Index(1), it->second);
        it = cache.map().find("test2");
        EXPECT_TRUE(it != cache.map().end());
        EXPECT_EQ(Name("test2"), it->first);
        EXPECT_EQ(Index(2), it->second);

        MetaMap metadata2;

        StringMetaInserter inserter2(metadata2);
        inserter2.insert("test3");

        cache.reset(metadata2);
        EXPECT_EQ(size_t(1), cache.size());
        it = cache.map().find("test3");
        EXPECT_TRUE(it != cache.map().end());
    }
}


TEST_F(TestAttributeArrayString, testStringMetaInserter)
{
    MetaMap metadata;

    StringMetaInserter inserter(metadata);

    { // insert one value
        Index index = inserter.insert("test");
        EXPECT_EQ(metadata.metaCount(), size_t(1));
        EXPECT_EQ(Index(1), index);
        EXPECT_TRUE(inserter.hasIndex(1));
        EXPECT_TRUE(inserter.hasKey("test"));
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test"));
    }

    { // insert another value
        Index index = inserter.insert("test2");
        EXPECT_EQ(metadata.metaCount(), size_t(2));
        EXPECT_EQ(Index(2), index);
        EXPECT_TRUE(inserter.hasIndex(1));
        EXPECT_TRUE(inserter.hasKey("test"));
        EXPECT_TRUE(inserter.hasIndex(2));
        EXPECT_TRUE(inserter.hasKey("test2"));
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test"));
        meta = metadata.getMetadata<StringMetadata>("string:1");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test2"));
    }

    // remove a value and reset the cache

    metadata.removeMeta("string:1");
    inserter.resetCache();

    { // re-insert value
        Index index = inserter.insert("test3");
        EXPECT_EQ(metadata.metaCount(), size_t(2));
        EXPECT_EQ(Index(2), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test"));
        meta = metadata.getMetadata<StringMetadata>("string:1");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test3"));
    }

    { // insert and remove to create a gap
        Index index = inserter.insert("test4");
        EXPECT_EQ(metadata.metaCount(), size_t(3));
        EXPECT_EQ(Index(3), index);
        metadata.removeMeta("string:1");
        inserter.resetCache();
        EXPECT_EQ(metadata.metaCount(), size_t(2));
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test"));
        meta = metadata.getMetadata<StringMetadata>("string:2");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test4"));
    }

    { // insert to fill gap
        Index index = inserter.insert("test10");
        EXPECT_EQ(metadata.metaCount(), size_t(3));
        EXPECT_EQ(Index(2), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test"));
        meta = metadata.getMetadata<StringMetadata>("string:1");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test10"));
        meta = metadata.getMetadata<StringMetadata>("string:2");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test4"));
    }

    { // insert existing value
        EXPECT_EQ(metadata.metaCount(), size_t(3));
        Index index = inserter.insert("test10");
        EXPECT_EQ(metadata.metaCount(), size_t(3));
        EXPECT_EQ(Index(2), index);
    }

    metadata.removeMeta("string:0");
    metadata.removeMeta("string:2");
    inserter.resetCache();

    { // insert other value and string metadata
        metadata.insertMeta("int:1", Int32Metadata(5));
        metadata.insertMeta("irrelevant", StringMetadata("irrelevant"));
        inserter.resetCache();
        EXPECT_EQ(metadata.metaCount(), size_t(3));
        Index index = inserter.insert("test15");
        EXPECT_EQ(metadata.metaCount(), size_t(4));
        EXPECT_EQ(Index(1), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:0");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test15"));
        meta = metadata.getMetadata<StringMetadata>("string:1");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test10"));
    }

    { // insert using a hint
        Index index = inserter.insert("test1000", 1000);
        EXPECT_EQ(metadata.metaCount(), size_t(5));
        EXPECT_EQ(Index(1000), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:999");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test1000"));
    }

    { // insert using same hint (fail to use hint this time)
        Index index = inserter.insert("test1001", 1000);
        EXPECT_EQ(metadata.metaCount(), size_t(6));
        EXPECT_EQ(Index(3), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:2");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test1001"));
    }

    { // insert using next adjacent hint
        Index index = inserter.insert("test1002", 1001);
        EXPECT_EQ(metadata.metaCount(), size_t(7));
        EXPECT_EQ(Index(1001), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:1000");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test1002"));
    }

    { // insert using previous adjacent hint
        Index index = inserter.insert("test999", 999);
        EXPECT_EQ(metadata.metaCount(), size_t(8));
        EXPECT_EQ(Index(999), index);
        StringMetadata::Ptr meta = metadata.getMetadata<StringMetadata>("string:998");
        EXPECT_TRUE(meta);
        EXPECT_EQ(meta->value(), openvdb::Name("test999"));
    }
}


TEST_F(TestAttributeArrayString, testStringAttribute)
{
    { // Typed class API

        const Index count = 50;
        StringAttributeArray attr(count);

        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());
        EXPECT_TRUE(isString(attr));

        attr.setTransient(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(!attr.isHidden());
        EXPECT_TRUE(isString(attr));

        attr.setHidden(true);
        EXPECT_TRUE(attr.isTransient());
        EXPECT_TRUE(attr.isHidden());
        EXPECT_TRUE(isString(attr));

        attr.setTransient(false);
        EXPECT_TRUE(!attr.isTransient());
        EXPECT_TRUE(attr.isHidden());
        EXPECT_TRUE(isString(attr));

        StringAttributeArray attrB(attr);

        EXPECT_TRUE(matchingNamePairs(attr.type(), attrB.type()));
        EXPECT_EQ(attr.size(), attrB.size());
        EXPECT_EQ(attr.memUsage(), attrB.memUsage());
        EXPECT_EQ(attr.isUniform(), attrB.isUniform());
        EXPECT_EQ(attr.isTransient(), attrB.isTransient());
        EXPECT_EQ(attr.isHidden(), attrB.isHidden());
        EXPECT_EQ(isString(attr), isString(attrB));

        AttributeArray& baseAttr(attr);
        EXPECT_EQ(Name(typeNameAsString<Index>()), baseAttr.valueType());
        EXPECT_EQ(Name("str"), baseAttr.codecType());
        EXPECT_EQ(Index(4), baseAttr.valueTypeSize());
        EXPECT_EQ(Index(4), baseAttr.storageTypeSize());
        EXPECT_TRUE(!baseAttr.valueTypeIsFloatingPoint());
    }

    { // IO
        const Index count = 50;
        StringAttributeArray attrA(count);

        for (unsigned i = 0; i < unsigned(count); ++i) {
            attrA.set(i, int(i));
        }

        attrA.setHidden(true);

        std::ostringstream ostr(std::ios_base::binary);
        attrA.write(ostr);

        StringAttributeArray attrB;

        std::istringstream istr(ostr.str(), std::ios_base::binary);
        attrB.read(istr);

        EXPECT_TRUE(matchingNamePairs(attrA.type(), attrB.type()));
        EXPECT_EQ(attrA.size(), attrB.size());
        EXPECT_EQ(attrA.memUsage(), attrB.memUsage());
        EXPECT_EQ(attrA.isUniform(), attrB.isUniform());
        EXPECT_EQ(attrA.isTransient(), attrB.isTransient());
        EXPECT_EQ(attrA.isHidden(), attrB.isHidden());
        EXPECT_EQ(isString(attrA), isString(attrB));

        for (unsigned i = 0; i < unsigned(count); ++i) {
            EXPECT_EQ(attrA.get(i), attrB.get(i));
        }
    }
}


TEST_F(TestAttributeArrayString, testStringAttributeHandle)
{
    MetaMap metadata;

    StringAttributeArray attr(4);
    StringAttributeHandle handle(attr, metadata);

    EXPECT_EQ(handle.size(), Index(4));
    EXPECT_EQ(handle.size(), attr.size());
    EXPECT_EQ(Index(1), handle.stride());
    EXPECT_TRUE(handle.hasConstantStride());

    { // index 0 should always be an empty string
        Name value = handle.get(0);

        EXPECT_EQ(value, Name(""));
    }

    // set first element to 101

    EXPECT_TRUE(handle.isUniform());

    attr.set(2, 102);

    EXPECT_TRUE(!handle.isUniform());

    { // index 101 does not exist as metadata is empty
        EXPECT_EQ(handle.get(0), Name(""));
        EXPECT_THROW(handle.get(2), LookupError);
    }

    { // add an element to the metadata for 101
        metadata.insertMeta("string:101", StringMetadata("test101"));

        EXPECT_EQ(handle.get(0), Name(""));

        EXPECT_NO_THROW(handle.get(2));
        EXPECT_EQ(handle.get(2), Name("test101"));

        Name name;
        handle.get(name, 2);

        EXPECT_EQ(name, Name("test101"));
    }

    { // add a second element to the metadata
        metadata.insertMeta("string:102", StringMetadata("test102"));

        EXPECT_EQ(handle.get(0), Name(""));

        EXPECT_NO_THROW(handle.get(2));
        EXPECT_EQ(handle.get(2), Name("test101"));

        Name name;
        handle.get(name, 2);

        EXPECT_EQ(name, Name("test101"));
    }

    { // set two more values in the array
        attr.set(0, 103);
        attr.set(1, 103);

        EXPECT_EQ(handle.get(0), Name("test102"));
        EXPECT_EQ(handle.get(1), Name("test102"));
        EXPECT_EQ(handle.get(2), Name("test101"));
        EXPECT_EQ(handle.get(3), Name(""));
    }

    { // change a value
        attr.set(1, 102);

        EXPECT_EQ(handle.get(0), Name("test102"));
        EXPECT_EQ(handle.get(1), Name("test101"));
        EXPECT_EQ(handle.get(2), Name("test101"));
        EXPECT_EQ(handle.get(3), Name(""));
    }

    { // cannot use a StringAttributeHandle with a non-string attribute
        TypedAttributeArray<float> invalidAttr(50);
        EXPECT_THROW(StringAttributeHandle(invalidAttr, metadata), TypeError);
    }

    // Test stride and hasConstantStride methods for string handles

    {
        StringAttributeArray attr(3, 2, true);
        StringAttributeHandle handle(attr, metadata);

        EXPECT_EQ(Index(3), handle.size());
        EXPECT_EQ(handle.size(), attr.size());
        EXPECT_EQ(Index(2), handle.stride());
        EXPECT_TRUE(handle.hasConstantStride());
    }

    {
        StringAttributeArray attr(4, 10, false);
        StringAttributeHandle handle(attr, metadata);

        EXPECT_EQ(Index(10), handle.size());
        EXPECT_EQ(Index(4), attr.size());
        EXPECT_EQ(Index(1), handle.stride());
        EXPECT_TRUE(!handle.hasConstantStride());
    }
}


TEST_F(TestAttributeArrayString, testStringAttributeWriteHandle)
{
    MetaMap metadata;

    StringAttributeArray attr(4);
    StringAttributeWriteHandle handle(attr, metadata);

    { // add some values to metadata
        metadata.insertMeta("string:45", StringMetadata("testA"));
        metadata.insertMeta("string:90", StringMetadata("testB"));
        metadata.insertMeta("string:1000", StringMetadata("testC"));
    }

    { // no string values set
        EXPECT_EQ(handle.get(0), Name(""));
        EXPECT_EQ(handle.get(1), Name(""));
        EXPECT_EQ(handle.get(2), Name(""));
        EXPECT_EQ(handle.get(3), Name(""));
    }

    { // cache not reset since metadata changed
        EXPECT_THROW(handle.set(1, "testB"), LookupError);
    }

    { // empty string always has index 0
        EXPECT_TRUE(handle.contains(""));
    }

    { // cache won't contain metadata until it has been reset
        EXPECT_TRUE(!handle.contains("testA"));
        EXPECT_TRUE(!handle.contains("testB"));
        EXPECT_TRUE(!handle.contains("testC"));
    }

    handle.resetCache();

    { // empty string always has index 0 regardless of cache reset
        EXPECT_TRUE(handle.contains(""));
    }

    { // cache now reset
        EXPECT_TRUE(handle.contains("testA"));
        EXPECT_TRUE(handle.contains("testB"));
        EXPECT_TRUE(handle.contains("testC"));

        EXPECT_NO_THROW(handle.set(1, "testB"));

        EXPECT_EQ(handle.get(0), Name(""));
        EXPECT_EQ(handle.get(1), Name("testB"));
        EXPECT_EQ(handle.get(2), Name(""));
        EXPECT_EQ(handle.get(3), Name(""));
    }

    { // add another value
        handle.set(2, "testC");

        EXPECT_EQ(handle.get(0), Name(""));
        EXPECT_EQ(handle.get(1), Name("testB"));
        EXPECT_EQ(handle.get(2), Name("testC"));
        EXPECT_EQ(handle.get(3), Name(""));
    }

    handle.resetCache();

    { // compact tests
        EXPECT_TRUE(!handle.compact());
        handle.set(0, "testA");
        handle.set(1, "testA");
        handle.set(2, "testA");
        handle.set(3, "testA");
        EXPECT_TRUE(handle.compact());
        EXPECT_TRUE(handle.isUniform());
    }

    { // expand tests
        EXPECT_TRUE(handle.isUniform());
        handle.expand();
        EXPECT_TRUE(!handle.isUniform());
        EXPECT_EQ(handle.get(0), Name("testA"));
        EXPECT_EQ(handle.get(1), Name("testA"));
        EXPECT_EQ(handle.get(2), Name("testA"));
        EXPECT_EQ(handle.get(3), Name("testA"));
    }

    { // fill tests
        EXPECT_TRUE(!handle.isUniform());
        handle.set(3, "testB");
        handle.fill("testC");
        EXPECT_TRUE(!handle.isUniform());
        EXPECT_EQ(handle.get(0), Name("testC"));
        EXPECT_EQ(handle.get(1), Name("testC"));
        EXPECT_EQ(handle.get(2), Name("testC"));
        EXPECT_EQ(handle.get(3), Name("testC"));
    }

    { // collapse tests
        handle.set(2, "testB");
        handle.collapse("testA");
        EXPECT_TRUE(handle.isUniform());
        EXPECT_EQ(handle.get(0), Name("testA"));
        handle.expand();
        handle.set(2, "testB");
        EXPECT_TRUE(!handle.isUniform());
        handle.collapse();
        EXPECT_EQ(handle.get(0), Name(""));
    }

    { // empty string tests
        handle.collapse("");
        EXPECT_EQ(handle.get(0), Name(""));
    }
}


TEST_F(TestAttributeArrayString, testProfile)
{
#ifdef PROFILE
    struct Timer : public openvdb::util::CpuTimer {};
    const size_t elements = 1000000;
#else
    struct Timer {
        void start(const std::string&) {}
        void stop() {}
    };
    const size_t elements = 10000;
#endif

    MetaMap metadata;
    StringMetaInserter inserter(metadata);

    Timer timer;
    timer.start("StringMetaInserter initialise");

    for (size_t i = 0; i < elements; ++i) {
        inserter.insert("test_string_" + std::to_string(i));
    }

    timer.stop();

    for (size_t i = 0; i < elements/2; ++i) {
        metadata.removeMeta("test_string_" + std::to_string(i*2));
    }

    timer.start("StringMetaInserter resetCache()");

    inserter.resetCache();

    timer.stop();
    timer.start("StringMetaInserter insert duplicates");

    for (size_t i = 0; i < elements; ++i) {
        inserter.insert("test_string_" + std::to_string(i));
    }

    timer.stop();

    openvdb::points::StringAttributeArray attr(elements);
    for (size_t i = 0; i < elements; ++i) {
        attr.set(Index(i), Index(i));
    }

    timer.start("StringAttributeWriteHandle construction");

    openvdb::points::StringAttributeWriteHandle handle(attr, metadata);

    timer.stop();
    timer.start("StringAttributeWriteHandle contains()");

    // half the calls will miss caches
    volatile bool result = false;
    (void)result;
    for (size_t i = 0; i < elements/2; ++i) {
        result = handle.contains("test_string_" + std::to_string(i*4)) || result;
    }

    timer.stop();
}
