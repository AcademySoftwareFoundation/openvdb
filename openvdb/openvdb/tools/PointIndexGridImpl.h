// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include "PointIndexGrid.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

template<typename T, Index Log2Dim>
inline bool
PointIndexLeafNode<T, Log2Dim>::isEmpty(const CoordBBox& bbox) const
{
    Index xPos, pos, zStride = Index(bbox.max()[2] - bbox.min()[2]);
    Coord ijk;

    for (ijk[0] = bbox.min()[0]; ijk[0] <= bbox.max()[0]; ++ijk[0]) {
        xPos = (ijk[0] & (DIM - 1u)) << (2 * LOG2DIM);

        for (ijk[1] = bbox.min()[1]; ijk[1] <= bbox.max()[1]; ++ijk[1]) {
            pos = xPos + ((ijk[1] & (DIM - 1u)) << LOG2DIM);
            pos += (bbox.min()[2] & (DIM - 1u));

            if (this->buffer()[pos+zStride] > (pos == 0 ? T(0) : this->buffer()[pos - 1])) {
                return false;
            }
        }
    }

    return true;
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    BaseLeaf::readBuffers(is, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    mIndices.resize(size_t(numIndices));
    is.read(reinterpret_cast<char*>(mIndices.data()), numIndices * sizeof(T));
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::readBuffers(std::istream& is, const CoordBBox& bbox, bool fromHalf)
{
    // Read and clip voxel values.
    BaseLeaf::readBuffers(is, bbox, fromHalf);

    Index64 numIndices = Index64(0);
    is.read(reinterpret_cast<char*>(&numIndices), sizeof(Index64));

    const Index64 numBytes = numIndices * sizeof(T);

    if (bbox.hasOverlap(this->getNodeBoundingBox())) {
        mIndices.resize(size_t(numIndices));
        is.read(reinterpret_cast<char*>(mIndices.data()), numBytes);

        /// @todo If any voxels were deactivated as a result of clipping in the call to
        /// BaseLeaf::readBuffers(), the point index list will need to be regenerated.
    } else {
        // Read and discard voxel values.
        std::unique_ptr<char[]> buf{new char[numBytes]};
        is.read(buf.get(), numBytes);
    }

    // Reserved for future use
    Index64 auxDataBytes = Index64(0);
    is.read(reinterpret_cast<char*>(&auxDataBytes), sizeof(Index64));
    if (auxDataBytes > 0) {
        // For now, read and discard any auxiliary data.
        std::unique_ptr<char[]> auxData{new char[auxDataBytes]};
        is.read(auxData.get(), auxDataBytes);
    }
}


template<typename T, Index Log2Dim>
inline void
PointIndexLeafNode<T, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    BaseLeaf::writeBuffers(os, toHalf);

    Index64 numIndices = Index64(mIndices.size());
    os.write(reinterpret_cast<const char*>(&numIndices), sizeof(Index64));
    os.write(reinterpret_cast<const char*>(mIndices.data()), numIndices * sizeof(T));

    // Reserved for future use
    const Index64 auxDataBytes = Index64(0);
    os.write(reinterpret_cast<const char*>(&auxDataBytes), sizeof(Index64));
}


template<typename T, Index Log2Dim>
inline Index64
PointIndexLeafNode<T, Log2Dim>::memUsage() const
{
    return BaseLeaf::memUsage() + Index64((sizeof(T)*mIndices.capacity()) + sizeof(mIndices));
}

} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
