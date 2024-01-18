// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/points/StreamCompression.h>
#include <openvdb/io/Compression.h> // io::COMPRESS_BLOSC

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
#include <boost/iostreams/device/array.hpp>
#include <boost/iostreams/stream.hpp>

#ifdef _WIN32
#include <boost/interprocess/detail/os_file_functions.hpp> // open_existing_file(), close_file()
#include <windows.h>
#else
#include <sys/types.h> // for struct stat
#include <sys/stat.h> // for stat()
#include <unistd.h> // for unlink()
#endif
#endif // OPENVDB_USE_DELAYED_LOADING

#include <atomic>
#include <fstream>
#include <numeric> // for std::iota()

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

#ifdef OPENVDB_USE_DELAYED_LOADING
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
#ifdef _WIN32
            using namespace boost::interprocess::ipcdetail;
            using openvdb::Index64;

            if (void* fh = open_existing_file(regionFilename, boost::interprocess::read_only)) {
                FILETIME mtime;
                if (GetFileTime(fh, nullptr, nullptr, &mtime)) {
                    mLastWriteTime = (Index64(mtime.dwHighDateTime) << 32) | mtime.dwLowDateTime;
                }
                close_file(fh);
            }
#else
            struct stat info;
            if (0 == ::stat(regionFilename, &info)) {
                mLastWriteTime = openvdb::Index64(info.st_mtime);
            }
#endif
        }

        using Notifier = std::function<void(std::string /*filename*/)>;
        boost::interprocess::file_mapping mMap;
        boost::interprocess::mapped_region mRegion;
        bool mAutoDelete = false;
        Notifier mNotifier;
        mutable std::atomic<openvdb::Index64> mLastWriteTime;
    }; // class Impl
    std::unique_ptr<Impl> mImpl;
}; // class ProxyMappedFile
#endif // OPENVDB_USE_DELAYED_LOADING

using namespace openvdb;
using namespace openvdb::compression;

class TestStreamCompression: public ::testing::Test
{
public:
    void testPagedStreams();
}; // class TestStreamCompression


////////////////////////////////////////


TEST_F(TestStreamCompression, testBlosc)
{
    // ensure that the library and unit tests are both built with or without Blosc enabled
#ifdef OPENVDB_USE_BLOSC
    EXPECT_TRUE(bloscCanCompress());
#else
    EXPECT_TRUE(!bloscCanCompress());
#endif

    const int count = 256;

    { // valid buffer
        // compress
        std::unique_ptr<int[]> uncompressedBuffer(new int[count]);

        for (int i = 0; i < count; i++) {
            uncompressedBuffer.get()[i] = i / 2;
        }

        size_t uncompressedBytes = count * sizeof(int);
        size_t compressedBytes;

        size_t testCompressedBytes = bloscCompressedSize(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes);

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes, compressedBytes);

#ifdef OPENVDB_USE_BLOSC
        EXPECT_TRUE(compressedBytes < uncompressedBytes);
        EXPECT_TRUE(compressedBuffer);
        EXPECT_EQ(testCompressedBytes, compressedBytes);

        // uncompressedSize

        EXPECT_EQ(uncompressedBytes, bloscUncompressedSize(compressedBuffer.get()));

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer =
            bloscDecompress(compressedBuffer.get(), uncompressedBytes);

        // incorrect number of expected bytes
        EXPECT_THROW(newUncompressedBuffer =
            bloscDecompress(compressedBuffer.get(), 1), openvdb::RuntimeError);

        EXPECT_TRUE(newUncompressedBuffer);
#else
        EXPECT_TRUE(!compressedBuffer);
        EXPECT_EQ(testCompressedBytes, size_t(0));

        // uncompressedSize

        EXPECT_THROW(bloscUncompressedSize(compressedBuffer.get()), openvdb::RuntimeError);

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer;
        EXPECT_THROW(
            newUncompressedBuffer = bloscDecompress(compressedBuffer.get(), uncompressedBytes),
            openvdb::RuntimeError);

        EXPECT_TRUE(!newUncompressedBuffer);
#endif
    }

    { // one value (below minimum bytes)
        std::unique_ptr<int[]> uncompressedBuffer(new int[1]);
        uncompressedBuffer.get()[0] = 10;

        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), sizeof(int), compressedBytes);

        EXPECT_TRUE(!compressedBuffer);
        EXPECT_EQ(compressedBytes, size_t(0));
    }

    { // padded buffer
        std::unique_ptr<char[]> largeBuffer(new char[2048]);

        for (int paddedCount = 1; paddedCount < 256; paddedCount++) {

            std::unique_ptr<char[]> newTest(new char[paddedCount]);
            for (int i = 0; i < paddedCount; i++)  newTest.get()[i] = char(0);

#ifdef OPENVDB_USE_BLOSC
            size_t compressedBytes;
            std::unique_ptr<char[]> compressedBuffer = bloscCompress(
                newTest.get(), paddedCount, compressedBytes);

            // compress into a large buffer to check for any padding issues
            size_t compressedSizeBytes;
            bloscCompress(largeBuffer.get(), compressedSizeBytes, size_t(2048),
                newTest.get(), paddedCount);

            // regardless of compression, these numbers should always match
            EXPECT_EQ(compressedSizeBytes, compressedBytes);

            // no compression performed due to buffer being too small
            if (paddedCount <= BLOSC_MINIMUM_BYTES) {
                EXPECT_TRUE(!compressedBuffer);
            }
            else {
                EXPECT_TRUE(compressedBuffer);
                EXPECT_TRUE(compressedBytes > 0);
                EXPECT_TRUE(int(compressedBytes) < paddedCount);

                std::unique_ptr<char[]> uncompressedBuffer = bloscDecompress(
                    compressedBuffer.get(), paddedCount);

                EXPECT_TRUE(uncompressedBuffer);

                for (int i = 0; i < paddedCount; i++) {
                    EXPECT_EQ((uncompressedBuffer.get())[i], newTest[i]);
                }
            }
#endif
        }
    }

    { // invalid buffer (out of range)

        // compress

        std::vector<int> smallBuffer;
        smallBuffer.resize(count);

        for (int i = 0; i < count; i++)     smallBuffer[i] = i;

        size_t invalidBytes = INT_MAX - 1;

        size_t testCompressedBytes = bloscCompressedSize(
            reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes);

        EXPECT_EQ(testCompressedBytes, size_t(0));

        std::unique_ptr<char[]> buffer = bloscCompress(
            reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes, testCompressedBytes);

        EXPECT_TRUE(!buffer);
        EXPECT_EQ(testCompressedBytes, size_t(0));

        // decompress

#ifdef OPENVDB_USE_BLOSC
        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(&smallBuffer[0]), count * sizeof(int), testCompressedBytes);

        EXPECT_THROW(buffer = bloscDecompress(
            reinterpret_cast<char*>(compressedBuffer.get()), invalidBytes - 16),
            openvdb::RuntimeError);

        EXPECT_TRUE(!buffer);

        EXPECT_THROW(bloscDecompress(
            reinterpret_cast<char*>(compressedBuffer.get()), count * sizeof(int) + 1),
            openvdb::RuntimeError);
#endif
    }

    { // uncompressible buffer
        const int uncompressedCount = 32;

        std::vector<int> values;
        values.reserve(uncompressedCount); // 128 bytes

        // insert a sequence of 32 integer values that cannot be compressed using Blosc

        for (int i = 0; i < uncompressedCount; i++) {
            if ((i%2) == 0) {
                values.push_back(i * 12340);
            } else {
                values.push_back(i * 56780);
            }
        }

        std::unique_ptr<int[]> uncompressedBuffer(new int[values.size()]);

        for (size_t i = 0; i < values.size(); i++)     uncompressedBuffer.get()[i] = values[i];

        size_t uncompressedBytes = values.size() * sizeof(int);
        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(
            reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes, compressedBytes);

        EXPECT_TRUE(!compressedBuffer);
        EXPECT_EQ(compressedBytes, size_t(0));
    }
}


void
TestStreamCompression::testPagedStreams()
{
    { // one small value
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        int foo = 5;
        ostream.write(reinterpret_cast<const char*>(&foo), sizeof(int));
        EXPECT_EQ(ostr.tellp(), std::streampos(0));

        ostream.flush();
        EXPECT_EQ(ostr.tellp(), std::streampos(sizeof(int)));
    }

    { // small values up to page threshold
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        for (int i = 0; i < PageSize; i++) {
            uint8_t oneByte = 255;
            ostream.write(reinterpret_cast<const char*>(&oneByte), sizeof(uint8_t));
        }
        EXPECT_EQ(ostr.tellp(), std::streampos(0));

        std::vector<uint8_t> values;
        values.assign(PageSize, uint8_t(255));
        size_t compressedSize = compression::bloscCompressedSize(
            reinterpret_cast<const char*>(&values[0]), PageSize);

        uint8_t oneMoreByte(255);
        ostream.write(reinterpret_cast<const char*>(&oneMoreByte), sizeof(char));

        if (compressedSize == 0) {
            EXPECT_EQ(ostr.tellp(), std::streampos(PageSize));
        }
        else {
            EXPECT_EQ(ostr.tellp(), std::streampos(compressedSize));
        }
    }

    { // one large block at exactly page threshold
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        std::vector<uint8_t> values;
        values.assign(PageSize, uint8_t(255));
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        EXPECT_EQ(ostr.tellp(), std::streampos(0));
    }

    { // two large blocks at page threshold + 1 byte
        std::ostringstream ostr(std::ios_base::binary);
        PagedOutputStream ostream(ostr);

        std::vector<uint8_t> values;
        values.assign(PageSize + 1, uint8_t(255));
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        size_t compressedSize = compression::bloscCompressedSize(
            reinterpret_cast<const char*>(&values[0]), values.size());

#ifndef OPENVDB_USE_BLOSC
        compressedSize = values.size();
#endif

        EXPECT_EQ(ostr.tellp(), std::streampos(compressedSize));

        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());

        EXPECT_EQ(ostr.tellp(), std::streampos(compressedSize * 2));

        uint8_t oneMoreByte(255);
        ostream.write(reinterpret_cast<const char*>(&oneMoreByte), sizeof(uint8_t));

        ostream.flush();

        EXPECT_EQ(ostr.tellp(), std::streampos(compressedSize * 2 + 1));
    }

    { // one full page
        std::stringstream ss(std::ios_base::out | std::ios_base::in | std::ios_base::binary);

        // write

        PagedOutputStream ostreamSizeOnly(ss);
        ostreamSizeOnly.setSizeOnly(true);

        EXPECT_EQ(ss.tellp(), std::streampos(0));

        std::vector<uint8_t> values;
        values.resize(PageSize);
        std::iota(values.begin(), values.end(), 0); // ascending integer values
        ostreamSizeOnly.write(reinterpret_cast<const char*>(&values[0]), values.size());
        ostreamSizeOnly.flush();

#ifdef OPENVDB_USE_BLOSC
        // two integers - compressed size and uncompressed size
        EXPECT_EQ(ss.tellp(), std::streampos(sizeof(int)*2));
#else
        // one integer - uncompressed size
        EXPECT_EQ(ss.tellp(), std::streampos(sizeof(int)));
#endif

        PagedOutputStream ostream(ss);
        ostream.write(reinterpret_cast<const char*>(&values[0]), values.size());
        ostream.flush();

#ifndef OPENVDB_USE_BLOSC
        EXPECT_EQ(ss.tellp(), std::streampos(PageSize+sizeof(int)));
#endif

        // read

        EXPECT_EQ(ss.tellg(), std::streampos(0));

        PagedInputStream istream(ss);
        istream.setSizeOnly(true);

        PageHandle::Ptr handle = istream.createHandle(values.size());

#ifdef OPENVDB_USE_BLOSC
        // two integers - compressed size and uncompressed size
        EXPECT_EQ(ss.tellg(), std::streampos(sizeof(int)*2));
#else
        // one integer - uncompressed size
        EXPECT_EQ(ss.tellg(), std::streampos(sizeof(int)));
#endif

        istream.read(handle, values.size(), false);

#ifndef OPENVDB_USE_BLOSC
        EXPECT_EQ(ss.tellg(), std::streampos(PageSize+sizeof(int)));
#endif

        std::unique_ptr<uint8_t[]> newValues(reinterpret_cast<uint8_t*>(handle->read().release()));

        EXPECT_TRUE(newValues);

        for (size_t i = 0; i < values.size(); i++) {
            EXPECT_EQ(values[i], newValues.get()[i]);
        }
    }

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

    {
        std::string filename = tempDir + "/openvdb_page1";
        io::StreamMetadata::Ptr streamMetadata(new io::StreamMetadata);

        { // ascending values up to 10 million written in blocks of PageSize/3
            std::ofstream fileout(filename.c_str(), std::ios_base::binary);

            io::setStreamMetadataPtr(fileout, streamMetadata);
            io::setDataCompression(fileout, openvdb::io::COMPRESS_BLOSC);

            std::vector<uint8_t> values;
            values.resize(10*1000*1000);
            std::iota(values.begin(), values.end(), 0); // ascending integer values

            // write page sizes

            PagedOutputStream ostreamSizeOnly(fileout);
            ostreamSizeOnly.setSizeOnly(true);

            EXPECT_EQ(fileout.tellp(), std::streampos(0));

            int increment = PageSize/3;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    ostreamSizeOnly.write(
                        reinterpret_cast<const char*>(&values[0]+i), values.size() - i);
                }
                else {
                    ostreamSizeOnly.write(reinterpret_cast<const char*>(&values[0]+i), increment);
                }
            }
            ostreamSizeOnly.flush();

#ifdef OPENVDB_USE_BLOSC
            int pages = static_cast<int>(fileout.tellp() / (sizeof(int)*2));
#else
            int pages = static_cast<int>(fileout.tellp() / (sizeof(int)));
#endif

            EXPECT_EQ(pages, 10);

            // write

            PagedOutputStream ostream(fileout);

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    ostream.write(reinterpret_cast<const char*>(&values[0]+i), values.size() - i);
                }
                else {
                    ostream.write(reinterpret_cast<const char*>(&values[0]+i), increment);
                }
            }

            ostream.flush();

#ifndef OPENVDB_USE_BLOSC
            EXPECT_EQ(fileout.tellp(), std::streampos(values.size()+sizeof(int)*pages));
#endif


#ifdef OPENVDB_USE_DELAYED_LOADING
            // abuse File being a friend of MappedFile to get around the private constructor
            ProxyMappedFile* proxy = new ProxyMappedFile(filename);
            SharedPtr<io::MappedFile> mappedFile(reinterpret_cast<io::MappedFile*>(proxy));

            // read

            std::ifstream filein(filename.c_str(), std::ios_base::in | std::ios_base::binary);
            io::setStreamMetadataPtr(filein, streamMetadata);
            io::setMappedFilePtr(filein, mappedFile);

            EXPECT_EQ(filein.tellg(), std::streampos(0));

            PagedInputStream istreamSizeOnly(filein);
            istreamSizeOnly.setSizeOnly(true);

            std::vector<PageHandle::Ptr> handles;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    handles.push_back(istreamSizeOnly.createHandle(values.size() - i));
                }
                else {
                    handles.push_back(istreamSizeOnly.createHandle(increment));
                }
            }

#ifdef OPENVDB_USE_BLOSC
            // two integers - compressed size and uncompressed size
            EXPECT_EQ(filein.tellg(), std::streampos(pages*sizeof(int)*2));
#else
            // one integer - uncompressed size
            EXPECT_EQ(filein.tellg(), std::streampos(pages*sizeof(int)));
#endif

            PagedInputStream istream(filein);

            int pageHandle = 0;

            for (size_t i = 0; i < values.size(); i += increment) {
                if (size_t(i+increment) > values.size()) {
                    istream.read(handles[pageHandle++], values.size() - i);
                }
                else {
                    istream.read(handles[pageHandle++], increment);
                }
            }

            // first three handles live in the same page

            Page& page0 = handles[0]->page();
            Page& page1 = handles[1]->page();
            Page& page2 = handles[2]->page();
            Page& page3 = handles[3]->page();

            EXPECT_TRUE(page0.isOutOfCore());
            EXPECT_TRUE(page1.isOutOfCore());
            EXPECT_TRUE(page2.isOutOfCore());
            EXPECT_TRUE(page3.isOutOfCore());

            handles[0]->read();

            // store the Page shared_ptr

            Page::Ptr page = handles[0]->mPage;

            // verify use count is four (one plus three handles)

            EXPECT_EQ(page.use_count(), long(4));

            // on reading from the first handle, all pages referenced
            // in the first three handles are in-core

            EXPECT_TRUE(!page0.isOutOfCore());
            EXPECT_TRUE(!page1.isOutOfCore());
            EXPECT_TRUE(!page2.isOutOfCore());
            EXPECT_TRUE(page3.isOutOfCore());

            handles[1]->read();

            EXPECT_TRUE(handles[0]->mPage);

            handles[2]->read();

            handles.erase(handles.begin());
            handles.erase(handles.begin());
            handles.erase(handles.begin());

            // after all three handles have been read,
            // page should have just one use count (itself)

            EXPECT_EQ(page.use_count(), long(1));

#endif // OPENVDB_USE_DELAYED_LOADING
        }
        std::remove(filename.c_str());
    }
}
TEST_F(TestStreamCompression, testPagedStreams) { testPagedStreams(); }
