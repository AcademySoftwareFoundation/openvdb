// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/io/DelayedLoadMetadata.h>

#include <gtest/gtest.h>

class TestDelayedLoadMetadata : public ::testing::Test
{
};


TEST_F(TestDelayedLoadMetadata, test)
{
    using namespace openvdb::io;

    // registration

    EXPECT_TRUE(!DelayedLoadMetadata::isRegisteredType());

    DelayedLoadMetadata::registerType();

    EXPECT_TRUE(DelayedLoadMetadata::isRegisteredType());

    DelayedLoadMetadata::unregisterType();

    EXPECT_TRUE(!DelayedLoadMetadata::isRegisteredType());

    openvdb::initialize();

    EXPECT_TRUE(DelayedLoadMetadata::isRegisteredType());

    // construction

    DelayedLoadMetadata metadata;

    EXPECT_TRUE(metadata.empty());

    metadata.resizeMask(size_t(2));

    EXPECT_TRUE(!metadata.empty());

    metadata.setMask(0, DelayedLoadMetadata::MaskType(5));
    metadata.setMask(1, DelayedLoadMetadata::MaskType(-3));

    EXPECT_EQ(metadata.getMask(0), DelayedLoadMetadata::MaskType(5));
    EXPECT_EQ(metadata.getMask(1), DelayedLoadMetadata::MaskType(-3));

    metadata.resizeCompressedSize(size_t(3));

    metadata.setCompressedSize(0, DelayedLoadMetadata::CompressedSizeType(6));
    metadata.setCompressedSize(1, DelayedLoadMetadata::CompressedSizeType(101));
    metadata.setCompressedSize(2, DelayedLoadMetadata::CompressedSizeType(-13522));

    EXPECT_EQ(metadata.getCompressedSize(0), DelayedLoadMetadata::CompressedSizeType(6));
    EXPECT_EQ(metadata.getCompressedSize(1), DelayedLoadMetadata::CompressedSizeType(101));
    EXPECT_EQ(metadata.getCompressedSize(2), DelayedLoadMetadata::CompressedSizeType(-13522));

    // copy construction

    DelayedLoadMetadata metadataCopy1(metadata);

    EXPECT_TRUE(!metadataCopy1.empty());

    EXPECT_EQ(metadataCopy1.getMask(0), DelayedLoadMetadata::MaskType(5));
    EXPECT_EQ(metadataCopy1.getCompressedSize(2), DelayedLoadMetadata::CompressedSizeType(-13522));

    openvdb::Metadata::Ptr baseMetadataCopy2 = metadata.copy();
    DelayedLoadMetadata::Ptr metadataCopy2 =
        openvdb::StaticPtrCast<DelayedLoadMetadata>(baseMetadataCopy2);

    EXPECT_EQ(metadataCopy2->getMask(0), DelayedLoadMetadata::MaskType(5));
    EXPECT_EQ(metadataCopy2->getCompressedSize(2), DelayedLoadMetadata::CompressedSizeType(-13522));

    // I/O

    metadata.clear();
    EXPECT_TRUE(metadata.empty());

    const size_t headerInitialSize(sizeof(openvdb::Index32));
    const size_t headerCountSize(sizeof(openvdb::Index32));
    const size_t headerMaskSize(sizeof(openvdb::Index32));
    const size_t headerCompressedSize(sizeof(openvdb::Index32));
    const size_t headerTotalSize(headerInitialSize + headerCountSize + headerMaskSize + headerCompressedSize);

    { // empty buffer
        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        metadata.write(ss);
        EXPECT_EQ(ss.tellp(), std::streampos(headerInitialSize));

        DelayedLoadMetadata newMetadata;
        newMetadata.read(ss);
        EXPECT_TRUE(newMetadata.empty());
    }

    { // single value, no compressed sizes
        metadata.clear();
        metadata.resizeMask(size_t(1));
        metadata.setMask(0, DelayedLoadMetadata::MaskType(5));

        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        metadata.write(ss);
        std::streampos expectedPos(headerTotalSize + sizeof(int8_t));
        EXPECT_EQ(ss.tellp(), expectedPos);
        EXPECT_EQ(static_cast<size_t>(expectedPos)-headerInitialSize, size_t(metadata.size()));

        DelayedLoadMetadata newMetadata;
        newMetadata.read(ss);
        EXPECT_TRUE(!newMetadata.empty());
        EXPECT_EQ(newMetadata.getMask(0), DelayedLoadMetadata::MaskType(5));
    }

    { // single value, with compressed sizes
        metadata.clear();
        metadata.resizeMask(size_t(1));
        metadata.setMask(0, DelayedLoadMetadata::MaskType(5));

        metadata.resizeCompressedSize(size_t(1));
        metadata.setCompressedSize(0, DelayedLoadMetadata::CompressedSizeType(-10322));

        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        metadata.write(ss);
        std::streampos expectedPos(headerTotalSize + sizeof(int8_t) + sizeof(int64_t));

        EXPECT_EQ(expectedPos, ss.tellp());
        EXPECT_EQ(static_cast<size_t>(ss.tellp())-headerInitialSize, size_t(metadata.size()));

        DelayedLoadMetadata newMetadata;
        newMetadata.read(ss);
        EXPECT_TRUE(!newMetadata.empty());
        EXPECT_EQ(newMetadata.getMask(0), DelayedLoadMetadata::MaskType(5));
        EXPECT_EQ(newMetadata.getCompressedSize(0), DelayedLoadMetadata::CompressedSizeType(-10322));
    }

    { // larger, but compressible buffer
        metadata.clear();

        const size_t size = 1000;

        const size_t uncompressedBufferSize = (sizeof(int8_t)+sizeof(int64_t))*size;

        metadata.resizeMask(size);
        metadata.resizeCompressedSize(size);
        for (size_t i = 0; i < size; i++) {
            metadata.setMask(i,
                DelayedLoadMetadata::MaskType(static_cast<int8_t>((i%32)*2)));
            metadata.setCompressedSize(i,
                DelayedLoadMetadata::CompressedSizeType(static_cast<int64_t>((i%64)*200)));
        }

        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);
        metadata.write(ss);

        EXPECT_EQ(static_cast<size_t>(ss.tellp())-headerInitialSize, size_t(metadata.size()));

        std::streampos uncompressedSize(uncompressedBufferSize + headerTotalSize);
#ifdef OPENVDB_USE_BLOSC
        // expect a compression ratio of more than 10x
        EXPECT_TRUE(ss.tellp() * 10 < uncompressedSize);
#else
        EXPECT_TRUE(ss.tellp() == uncompressedSize);
#endif

        DelayedLoadMetadata newMetadata;
        newMetadata.read(ss);
        EXPECT_EQ(metadata.size(), newMetadata.size());
        for (size_t i = 0; i < size; i++) {
            EXPECT_EQ(metadata.getMask(i), newMetadata.getMask(i));
        }
    }

    // when read as unknown metadata should be treated as temporary metadata

    {
        metadata.clear();
        metadata.resizeMask(size_t(1));
        metadata.setMask(0, DelayedLoadMetadata::MaskType(5));

        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

        openvdb::MetaMap metamap;
        metamap.insertMeta("delayload", metadata);

        EXPECT_EQ(size_t(1), metamap.metaCount());

        metamap.writeMeta(ss);

        {
            openvdb::MetaMap newMetamap;
            newMetamap.readMeta(ss);

            EXPECT_EQ(size_t(1), newMetamap.metaCount());
        }

        {
            DelayedLoadMetadata::unregisterType();

            openvdb::MetaMap newMetamap;
            newMetamap.readMeta(ss);

            EXPECT_EQ(size_t(0), newMetamap.metaCount());
        }
    }
}
