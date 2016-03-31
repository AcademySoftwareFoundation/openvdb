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
//
/// @file AttributeArray.cc
///
/// @authors Dan Bailey, Mihai Alden, Peter Cucka

#include <map>

#include <openvdb_points/tools/AttributeArray.h>

#ifdef OPENVDB_USE_BLOSC
#include <blosc.h>
#endif

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

////////////////////////////////////////

namespace attribute_compression {


#ifdef OPENVDB_USE_BLOSC


bool canCompress()
{
    return true;
}


int uncompressedSize(const char* buffer)
{
    size_t bytes, _1, _2;
    blosc_cbuffer_sizes(buffer, &bytes, &_1, &_2);
    return bytes;
}


int compressedSize( const char* buffer, const size_t typeSize, const int uncompressedBytes)
{
    int tempBytes = uncompressedBytes + BLOSC_MAX_OVERHEAD;
    boost::scoped_array<char> outBuf(new char[tempBytes]);

    int compressedBytes = blosc_compress_ctx(
        /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
        /*doshuffle=*/true,
        /*typesize=*/typeSize,
        /*srcsize=*/uncompressedBytes,
        /*src=*/buffer,
        /*dest=*/outBuf.get(),
        /*destsize=*/tempBytes,
        BLOSC_LZ4_COMPNAME,
        /*blocksize=*/256,
        /*numthreads=*/1);

    if (compressedBytes <= 0) {
        std::ostringstream ostr;
        ostr << "Blosc failed to compress " << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s");
        if (compressedBytes < 0) ostr << " (internal error " << compressedBytes << ")";
        OPENVDB_LOG_DEBUG(ostr.str());
        return 0;
    }

    return compressedBytes;
}


char* compress( char* buffer, const size_t typeSize,
                const int uncompressedBytes, int& compressedBytes, const bool cleanup)
{
    int tempBytes = uncompressedBytes + BLOSC_MAX_OVERHEAD;
    boost::scoped_array<char> outBuf(new char[tempBytes]);

    compressedBytes = blosc_compress_ctx(
        /*clevel=*/9, // 0 (no compression) to 9 (maximum compression)
        /*doshuffle=*/true,
        /*typesize=*/typeSize,
        /*srcsize=*/uncompressedBytes,
        /*src=*/buffer,
        /*dest=*/outBuf.get(),
        /*destsize=*/tempBytes,
        BLOSC_LZ4_COMPNAME,
        /*blocksize=*/256,
        /*numthreads=*/1);

    if (compressedBytes <= 0) {
        std::ostringstream ostr;
        ostr << "Blosc failed to compress " << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s");
        if (compressedBytes < 0) ostr << " (internal error " << compressedBytes << ")";
        OPENVDB_LOG_DEBUG(ostr.str());
        return 0;
    }

    // optionally cleanup compressed buffer if requested (prior to allocating new uncompressed buffer)

    if (cleanup)    delete[] buffer;

    char* outData = new char[compressedBytes];
    std::memcpy(outData, outBuf.get(), size_t(compressedBytes));
    return outData;
}


char* decompress(char* buffer, const int expectedBytes, const bool cleanup)
{
    int tempBytes = expectedBytes + BLOSC_MAX_OVERHEAD;
    boost::scoped_array<char> tempBuffer(new char[tempBytes]);

    const int uncompressedBytes = blosc_decompress_ctx( /*src=*/buffer,
                                                        /*dest=*/tempBuffer.get(),
                                                        tempBytes,
                                                        /*numthreads=*/1);

    if (uncompressedBytes < 1) {
        OPENVDB_LOG_DEBUG("blosc_decompress() returned error code " << uncompressedBytes);
        return 0;
    }
    if (uncompressedBytes != Int64(expectedBytes)) {
        OPENVDB_THROW(RuntimeError, "Expected to decompress " << expectedBytes
            << " byte" << (expectedBytes == 1 ? "" : "s") << ", got "
            << uncompressedBytes << " byte" << (uncompressedBytes == 1 ? "" : "s"));
    }

    // optionally cleanup compressed buffer if requested (prior to allocating new uncompressed buffer)

    if (cleanup)    delete[] buffer;

    char* newBuffer = new char[uncompressedBytes];
    std::memcpy(newBuffer, tempBuffer.get(), size_t(uncompressedBytes));

    return newBuffer;
}


#else


bool canCompress()
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return false;
}


int uncompressedSize(const char*)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


int compressedSize(const char*, const size_t, const int)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return 0;
}


char* compress(char*, const size_t, const int, int&, const bool)
{
    OPENVDB_LOG_DEBUG("Can't compress array data without the blosc library.");
    return 0;
}


char* decompress(char*, const int, const bool)
{
    OPENVDB_THROW(RuntimeError, "Can't extract compressed data without the blosc library.");
}


#endif // OPENVDB_USE_BLOSC


char* compress(  const char* buffer, const size_t typeSize,
                        const int uncompressedBytes, int& compressedBytes)
{
    return compress(const_cast<char*>(buffer), typeSize, uncompressedBytes, compressedBytes, /*cleanup=*/false);
}


char* decompress(const char* buffer, const int expectedBytes)
{
    return decompress(const_cast<char*>(buffer), expectedBytes, /*cleanup=*/false);
}


} // namespace attribute_compression


////////////////////////////////////////


namespace {

typedef std::map<NamePair, AttributeArray::FactoryMethod> AttributeFactoryMap;
typedef AttributeFactoryMap::const_iterator AttributeFactoryMapCIter;

struct LockedAttributeRegistry
{
    tbb::spin_mutex     mMutex;
    AttributeFactoryMap mMap;
};


// Declare this at file scope to ensure thread-safe initialization.
tbb::spin_mutex sInitAttributeRegistryMutex;


// Global function for accessing the registry
LockedAttributeRegistry*
getAttributeRegistry()
{
    tbb::spin_mutex::scoped_lock lock(sInitAttributeRegistryMutex);

    static LockedAttributeRegistry* registry = NULL;

    if (registry == NULL) {

#ifdef __ICC
// Disable ICC "assignment to statically allocated variable" warning.
__pragma(warning(disable:1711))
#endif
        // This assignment is mutex-protected and therefore thread-safe.
        registry = new LockedAttributeRegistry();

#ifdef __ICC
__pragma(warning(default:1711))
#endif

    }

    return registry;
}

} // unnamed namespace


////////////////////////////////////////

// AttributeArray implementation


AttributeArray::Ptr
AttributeArray::create(const NamePair& type, size_t length)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    AttributeFactoryMapCIter iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        OPENVDB_THROW(LookupError, "Cannot create attribute of unregistered type " << type.first << "_" << type.second);
    }

    return (iter->second)(length);
}


bool
AttributeArray::isRegistered(const NamePair& type)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);
    return (registry->mMap.find(type) != registry->mMap.end());
}


void
AttributeArray::clearRegistry()
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);
    registry->mMap.clear();
}


void
AttributeArray::registerType(const NamePair& type, FactoryMethod factory)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    AttributeFactoryMapCIter iter = registry->mMap.find(type);

    if (iter == registry->mMap.end()) {
        registry->mMap[type] = factory;

    } else if (iter->second != factory) {
        OPENVDB_THROW(KeyError, "Attribute type " << type.first << "_" << type.second
            << " is already registered with different factory method.");
    }
}


void
AttributeArray::unregisterType(const NamePair& type)
{
    LockedAttributeRegistry* registry = getAttributeRegistry();
    tbb::spin_mutex::scoped_lock lock(registry->mMutex);

    registry->mMap.erase(type);
}


void
AttributeArray::setTransient(bool state)
{
    if (state) mFlags |= Int16(TRANSIENT);
    else mFlags &= ~Int16(TRANSIENT);
}


void
AttributeArray::setHidden(bool state)
{
    if (state) mFlags |= Int16(HIDDEN);
    else mFlags &= ~Int16(HIDDEN);
}


IndexIter
AttributeArray::beginIndex() const
{
    return IndexIter(0, this->size());
}


bool
AttributeArray::operator==(const AttributeArray& other) const {
    if(this->mCompressedBytes != other.mCompressedBytes ||
       this->mFlags != other.mFlags) return false;
    return this->isEqual(other);
}



////////////////////////////////////////



} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
