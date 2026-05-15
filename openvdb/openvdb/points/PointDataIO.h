// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef OPENVDB_POINTS_POINT_DATA_IO_HAS_BEEN_INCLUDED
#define OPENVDB_POINTS_POINT_DATA_IO_HAS_BEEN_INCLUDED


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {


////////////////////////////////////////


namespace io
{

/// @brief openvdb::io::readCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
readCompressedValues(   std::istream& is, PointDataIndex32* destBuf, Index destCount,
                        const util::NodeMask<3>& /*valueMask*/, bool /*fromHalf*/)
{
    using compression::bloscDecompress;

    const bool seek = destBuf == nullptr;

    const size_t destBytes = destCount*sizeof(PointDataIndex32);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (destBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot read more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    uint16_t bytes16;

    const io::StreamMetadata::Ptr meta = io::getStreamMetadataPtr(is);

    if (seek && meta) {
        // buffer size temporarily stored in the StreamMetadata pass
        // to avoid having to perform an expensive disk read for 2-bytes
        bytes16 = static_cast<uint16_t>(meta->pass());
        // seek over size of the compressed buffer
        is.seekg(sizeof(uint16_t), std::ios_base::cur);
    }
    else {
        // otherwise read from disk
        is.read(reinterpret_cast<char*>(&bytes16), sizeof(uint16_t));
    }

    if (bytes16 == std::numeric_limits<uint16_t>::max()) {
        // read or seek uncompressed data
        if (seek) {
            is.seekg(destBytes, std::ios_base::cur);
        }
        else {
            is.read(reinterpret_cast<char*>(destBuf), destBytes);
        }
    }
    else {
        // read or seek uncompressed data
        if (seek) {
            is.seekg(int(bytes16), std::ios_base::cur);
        }
        else {
            // decompress into the destination buffer
            std::unique_ptr<char[]> bloscBuffer(new char[int(bytes16)]);
            is.read(bloscBuffer.get(), bytes16);
            std::unique_ptr<char[]> buffer = bloscDecompress(   bloscBuffer.get(),
                                                                destBytes,
                                                                /*resize=*/false);
            std::memcpy(destBuf, buffer.get(), destBytes);
        }
    }
}

/// @brief openvdb::io::writeCompressedValues specialized on PointDataIndex32 arrays to
/// ignore the value mask, use a larger block size and use 16-bit size instead of 64-bit
template<>
inline void
writeCompressedValues(  std::ostream& os, const PointDataIndex32* srcBuf, Index srcCount,
                        const util::NodeMask<3>& /*valueMask*/,
                        const util::NodeMask<3>& /*childMask*/, bool /*toHalf*/)
{
    using compression::bloscCompress;

    const size_t srcBytes = srcCount*sizeof(PointDataIndex32);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (srcBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot write more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    const char* charBuffer = reinterpret_cast<const char*>(srcBuf);

    size_t compressedBytes;
    std::unique_ptr<char[]> buffer = bloscCompress( charBuffer, srcBytes,
                                                    compressedBytes, /*resize=*/false);

    if (compressedBytes > 0) {
        auto bytes16 = static_cast<uint16_t>(compressedBytes); // clamp to 16-bit unsigned integer
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(buffer.get()), compressedBytes);
    }
    else {
        auto bytes16 = static_cast<uint16_t>(maximumBytes); // max value indicates uncompressed
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
        os.write(reinterpret_cast<const char*>(srcBuf), srcBytes);
    }
}

template <typename T>
inline void
writeCompressedValuesSize(std::ostream& os, const T* srcBuf, Index srcCount)
{
    using compression::bloscCompressedSize;

    const size_t srcBytes = srcCount*sizeof(T);
    const size_t maximumBytes = std::numeric_limits<uint16_t>::max();
    if (srcBytes >= maximumBytes) {
        OPENVDB_THROW(openvdb::IoError, "Cannot write more than " <<
                                maximumBytes << " bytes in voxel values.")
    }

    const char* charBuffer = reinterpret_cast<const char*>(srcBuf);

    // calculate voxel buffer size after compression
    size_t compressedBytes = bloscCompressedSize(charBuffer, srcBytes);

    if (compressedBytes > 0) {
        auto bytes16 = static_cast<uint16_t>(compressedBytes); // clamp to 16-bit unsigned integer
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
    }
    else {
        auto bytes16 = static_cast<uint16_t>(maximumBytes); // max value indicates uncompressed
        os.write(reinterpret_cast<const char*>(&bytes16), sizeof(uint16_t));
    }
}

} // namespace io


} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_POINTS_POINT_DATA_IO_HAS_BEEN_INCLUDED
