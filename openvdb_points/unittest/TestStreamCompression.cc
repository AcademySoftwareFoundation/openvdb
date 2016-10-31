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
#include <openvdb_points/tools/StreamCompression.h>

using namespace openvdb;
using namespace openvdb::compression;

class TestStreamCompression: public CppUnit::TestCase
{
public:
    CPPUNIT_TEST_SUITE(TestStreamCompression);
    CPPUNIT_TEST(testBlosc);

    CPPUNIT_TEST_SUITE_END();

    void testBlosc();
}; // class TestStreamCompression

CPPUNIT_TEST_SUITE_REGISTRATION(TestStreamCompression);


////////////////////////////////////////


void
TestStreamCompression::testBlosc()
{
    const int count = 256;

    { // valid buffer
        // compress

        std::unique_ptr<int[]> uncompressedBuffer(new int[count]);

        for (int i = 0; i < count; i++) {
            uncompressedBuffer.get()[i] = i / 2;
        }

        size_t uncompressedBytes = count * sizeof(int);
        size_t compressedBytes;

        size_t testCompressedBytes = bloscCompressedSize(reinterpret_cast<char*>(uncompressedBuffer.get()), uncompressedBytes);

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(  reinterpret_cast<char*>(uncompressedBuffer.get()),
                                            uncompressedBytes, compressedBytes);

#ifdef OPENVDB_USE_BLOSC
        CPPUNIT_ASSERT(compressedBytes < uncompressedBytes);
        CPPUNIT_ASSERT(compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, compressedBytes);

        // uncompressedSize

        CPPUNIT_ASSERT_EQUAL(uncompressedBytes, bloscUncompressedSize(compressedBuffer.get()));

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer = bloscDecompress(compressedBuffer.get(), uncompressedBytes);

        // incorrect number of expected bytes
        CPPUNIT_ASSERT_THROW(newUncompressedBuffer = bloscDecompress(compressedBuffer.get(), 1), openvdb::RuntimeError);

        CPPUNIT_ASSERT(newUncompressedBuffer);
#else
        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        // uncompressedSize

        CPPUNIT_ASSERT_THROW(bloscUncompressedSize(compressedBuffer.get()), openvdb::RuntimeError);

        // decompress

        std::unique_ptr<char[]> newUncompressedBuffer;
        CPPUNIT_ASSERT_THROW(newUncompressedBuffer = bloscDecompress(compressedBuffer.get(), uncompressedBytes), openvdb::RuntimeError);

        CPPUNIT_ASSERT(!newUncompressedBuffer);
#endif
    }

    { // one value (below minimum bytes)
        std::unique_ptr<int[]> uncompressedBuffer(new int[1]);
        uncompressedBuffer.get()[0] = 10;

        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(reinterpret_cast<char*>(uncompressedBuffer.get()),
                                                            sizeof(int), compressedBytes);

        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(compressedBytes, size_t(0));
    }

    { // padded buffer
        const int paddedCount = 16;

        std::unique_ptr<int[]> newTest(new int[paddedCount]);
        for (int i = 0; i < paddedCount; i++)  newTest.get()[i] = i;

#ifdef OPENVDB_USE_BLOSC
        size_t compressedBytes;
        std::unique_ptr<char[]> compressedBuffer = bloscCompress(reinterpret_cast<char*>(newTest.get()), paddedCount*sizeof(int), compressedBytes);

        CPPUNIT_ASSERT(compressedBuffer);

        CPPUNIT_ASSERT(compressedBytes > 0 && compressedBytes < (paddedCount*sizeof(int)));

        std::unique_ptr<char[]> uncompressedBuffer = bloscDecompress(reinterpret_cast<char*>(compressedBuffer.get()), paddedCount*sizeof(int));

        CPPUNIT_ASSERT(uncompressedBuffer);

        for (int i = 0; i < paddedCount; i++) {
            CPPUNIT_ASSERT_EQUAL((reinterpret_cast<int*>(uncompressedBuffer.get()))[i], newTest[i]);
        }
#endif
    }

    { // invalid buffer (out of range)

        // compress

        std::vector<int> smallBuffer;
        smallBuffer.reserve(count);

        for (int i = 0; i < count; i++)     smallBuffer[i] = i;

        size_t invalidBytes = INT_MAX - 1;

        size_t testCompressedBytes = bloscCompressedSize(reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes);

        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        std::unique_ptr<char[]> buffer = bloscCompress(reinterpret_cast<char*>(&smallBuffer[0]), invalidBytes, testCompressedBytes);

        CPPUNIT_ASSERT(!buffer);
        CPPUNIT_ASSERT_EQUAL(testCompressedBytes, size_t(0));

        // decompress

#ifdef OPENVDB_USE_BLOSC
        std::unique_ptr<char[]> compressedBuffer = bloscCompress(reinterpret_cast<char*>(&smallBuffer[0]), count * sizeof(int), testCompressedBytes);

        CPPUNIT_ASSERT_THROW(buffer = bloscDecompress(reinterpret_cast<char*>(compressedBuffer.get()), invalidBytes - 16), openvdb::RuntimeError);

        CPPUNIT_ASSERT(!buffer);

        CPPUNIT_ASSERT_THROW(bloscDecompress(reinterpret_cast<char*>(compressedBuffer.get()), count * sizeof(int) + 1), openvdb::RuntimeError);
#endif
    }

    { // uncompressible buffer
        const int uncompressedCount = 32;

        std::vector<int> values;
        values.reserve(uncompressedCount); // 128 bytes

        for (int i = 0; i < uncompressedCount; i++)     values.push_back(i*10000);

        std::random_shuffle(values.begin(), values.end());

        std::unique_ptr<int[]> uncompressedBuffer(new int[values.size()]);

        for (size_t i = 0; i < values.size(); i++)     uncompressedBuffer.get()[i] = values[i];

        size_t uncompressedBytes = values.size() * sizeof(int);
        size_t compressedBytes;

        std::unique_ptr<char[]> compressedBuffer = bloscCompress(reinterpret_cast<char*>(uncompressedBuffer.get()),
                                                            uncompressedBytes, compressedBytes);

        CPPUNIT_ASSERT(!compressedBuffer);
        CPPUNIT_ASSERT_EQUAL(compressedBytes, size_t(0));
    }
}


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
