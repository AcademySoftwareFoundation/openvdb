// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/io/Compression.h> // for io::readCompressedValues(), etc.

#include "InternalNode.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {


template<typename ChildT, Index Log2Dim>
inline Index32
InternalNode<ChildT, Log2Dim>::leafCount() const
{
    if (ChildNodeType::getLevel() == 0) return mChildMask.countOn();
    Index32 sum = 0;
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        sum += iter->leafCount();
    }
    return sum;
}

template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::nodeCount(std::vector<Index32> &vec) const
{
    assert(vec.size() > ChildNodeType::LEVEL);
    const auto count = mChildMask.countOn();
    if (ChildNodeType::LEVEL > 0 && count > 0) {
        for (auto iter = this->cbeginChildOn(); iter; ++iter) iter->nodeCount(vec);
    }
    vec[ChildNodeType::LEVEL] += count;
}


template<typename ChildT, Index Log2Dim>
inline Index32
InternalNode<ChildT, Log2Dim>::nonLeafCount() const
{
    Index32 sum = 1;
    if (ChildNodeType::getLevel() == 0) return sum;
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        sum += iter->nonLeafCount();
    }
    return sum;
}


template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::onVoxelCount() const
{
    Index64 sum = ChildT::NUM_VOXELS * mValueMask.countOn();
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        sum += iter->onVoxelCount();
    }
    return sum;
}


template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::offVoxelCount() const
{
    Index64 sum = ChildT::NUM_VOXELS * (NUM_VALUES-mValueMask.countOn()-mChildMask.countOn());
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        sum += iter->offVoxelCount();
    }
    return sum;
}


template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::onLeafVoxelCount() const
{
    Index64 sum = 0;
    for (ChildOnCIter iter = this->beginChildOn(); iter; ++iter) {
        sum += mNodes[iter.pos()].getChild()->onLeafVoxelCount();
    }
    return sum;
}


template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::offLeafVoxelCount() const
{
    Index64 sum = 0;
    for (ChildOnCIter iter = this->beginChildOn(); iter; ++iter) {
        sum += mNodes[iter.pos()].getChild()->offLeafVoxelCount();
    }
    return sum;
}

template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::onTileCount() const
{
    Index64 sum = mValueMask.countOn();
    for (ChildOnCIter iter = this->cbeginChildOn(); LEVEL>1 && iter; ++iter) {
        sum += iter->onTileCount();
    }
    return sum;
}

template<typename ChildT, Index Log2Dim>
inline Index64
InternalNode<ChildT, Log2Dim>::memUsage() const
{
    Index64 sum = NUM_VALUES * sizeof(UnionType) + mChildMask.memUsage()
                + mValueMask.memUsage() + sizeof(mOrigin);
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        sum += iter->memUsage();
    }
    return sum;
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    if (bbox.isInside(this->getNodeBoundingBox())) return;

    for (ValueOnCIter i = this->cbeginValueOn(); i; ++i) {
        bbox.expand(i.getCoord(), ChildT::DIM);
    }
    for (ChildOnCIter i = this->cbeginChildOn(); i; ++i) {
        i->evalActiveBoundingBox(bbox, visitVoxels);
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::prune(const ValueType& tolerance)
{
    bool state = false;
    ValueType value = zeroVal<ValueType>();
    for (ChildOnIter iter = this->beginChildOn(); iter; ++iter) {
        const Index i = iter.pos();
        ChildT* child = mNodes[i].getChild();
        child->prune(tolerance);
        if (child->isConstant(value, state, tolerance)) {
            delete child;
            mChildMask.setOff(i);
            mValueMask.set(i, state);
            mNodes[i].setValue(value);
        }
     }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline bool
InternalNode<ChildT, Log2Dim>::isConstant(ValueType& minValue,
                                          ValueType& maxValue,
                                          bool& state,
                                          const ValueType& tolerance) const
{

    if (!mChildMask.isOff() || !mValueMask.isConstant(state)) return false;// early termination
    minValue = maxValue = mNodes[0].getValue();
    for (Index i = 1; i < NUM_VALUES; ++i) {
        const ValueType& v = mNodes[i].getValue();
        if (v < minValue) {
            if ((maxValue + math::negative(v)) > tolerance) return false;// early termination
            minValue = v;
        } else if (v > maxValue) {
            if ((v + math::negative(minValue)) > tolerance) return false;// early termination
            maxValue = v;
        }
    }
    return true;
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline bool
InternalNode<ChildT, Log2Dim>::hasActiveTiles() const
{
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_BEGIN
    const bool anyActiveTiles = !mValueMask.isOff();
    if (LEVEL==1 || anyActiveTiles) return anyActiveTiles;
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        if (iter->hasActiveTiles()) return true;
    }
    return false;
    OPENVDB_NO_UNREACHABLE_CODE_WARNING_END
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::setValuesOn()
{
    mValueMask = !mChildMask;
    for (ChildOnIter iter = this->beginChildOn(); iter; ++iter) {
        mNodes[iter.pos()].getChild()->setValuesOn();
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::clip(const CoordBBox& clipBBox, const ValueType& background)
{
    CoordBBox nodeBBox = this->getNodeBoundingBox();
    if (!clipBBox.hasOverlap(nodeBBox)) {
        // This node lies completely outside the clipping region.  Fill it with background tiles.
        this->fill(nodeBBox, background, /*active=*/false);
    } else if (clipBBox.isInside(nodeBBox)) {
        // This node lies completely inside the clipping region.  Leave it intact.
        return;
    }

    // This node isn't completely contained inside the clipping region.
    // Clip tiles and children, and replace any that lie outside the region
    // with background tiles.

    for (Index pos = 0; pos < NUM_VALUES; ++pos) {
        const Coord xyz = this->offsetToGlobalCoord(pos); // tile or child origin
        CoordBBox tileBBox(xyz, xyz.offsetBy(ChildT::DIM - 1)); // tile or child bounds
        if (!clipBBox.hasOverlap(tileBBox)) {
            // This table entry lies completely outside the clipping region.
            // Replace it with a background tile.
            this->makeChildNodeEmpty(pos, background);
            mValueMask.setOff(pos);
        } else if (!clipBBox.isInside(tileBBox)) {
            // This table entry does not lie completely inside the clipping region
            // and must be clipped.
            if (this->isChildMaskOn(pos)) {
                mNodes[pos].getChild()->clip(clipBBox, background);
            } else {
                // Replace this tile with a background tile, then fill the clip region
                // with the tile's original value.  (This might create a child branch.)
                tileBBox.intersect(clipBBox);
                const ValueType val = mNodes[pos].getValue();
                const bool on = this->isValueMaskOn(pos);
                mNodes[pos].setValue(background);
                mValueMask.setOff(pos);
                this->fill(tileBBox, val, on);
            }
        } else {
            // This table entry lies completely inside the clipping region.  Leave it intact.
        }
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    // Iterate over the fill region in axis-aligned, tile-sized chunks.
    // (The first and last chunks along each axis might be smaller than a tile.)
    Coord xyz, tileMin, tileMax;
    for (int x = clippedBBox.min().x(); x <= clippedBBox.max().x(); x = tileMax.x() + 1) {
        xyz.setX(x);
        for (int y = clippedBBox.min().y(); y <= clippedBBox.max().y(); y = tileMax.y() + 1) {
            xyz.setY(y);
            for (int z = clippedBBox.min().z(); z <= clippedBBox.max().z(); z = tileMax.z() + 1) {
                xyz.setZ(z);

                // Get the bounds of the tile that contains voxel (x, y, z).
                const Index n = this->coordToOffset(xyz);
                tileMin = this->offsetToGlobalCoord(n);
                tileMax = tileMin.offsetBy(ChildT::DIM - 1);

                if (xyz != tileMin || Coord::lessThan(clippedBBox.max(), tileMax)) {
                    // If the box defined by (xyz, clippedBBox.max()) doesn't completely enclose
                    // the tile to which xyz belongs, create a child node (or retrieve
                    // the existing one).
                    ChildT* child = nullptr;
                    if (this->isChildMaskOff(n)) {
                        // Replace the tile with a newly-created child that is initialized
                        // with the tile's value and active state.
                        child = new ChildT{xyz, mNodes[n].getValue(), this->isValueMaskOn(n)};
                        this->setChildNode(n, child);
                    } else {
                        child = mNodes[n].getChild();
                    }

                    // Forward the fill request to the child.
                    if (child) {
                        const Coord tmp = Coord::minComponent(clippedBBox.max(), tileMax);
                        child->fill(CoordBBox(xyz, tmp), value, active);
                    }

                } else {
                    // If the box given by (xyz, clippedBBox.max()) completely encloses
                    // the tile to which xyz belongs, create the tile (if it
                    // doesn't already exist) and give it the fill value.
                    this->makeChildNodeEmpty(n, value);
                    mValueMask.set(n, active);
                }
            }
        }
    }
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::denseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    auto clippedBBox = this->getNodeBoundingBox();
    clippedBBox.intersect(bbox);
    if (!clippedBBox) return;

    // Iterate over the fill region in axis-aligned, tile-sized chunks.
    // (The first and last chunks along each axis might be smaller than a tile.)
    Coord xyz, tileMin, tileMax;
    for (int x = clippedBBox.min().x(); x <= clippedBBox.max().x(); x = tileMax.x() + 1) {
        xyz.setX(x);
        for (int y = clippedBBox.min().y(); y <= clippedBBox.max().y(); y = tileMax.y() + 1) {
            xyz.setY(y);
            for (int z = clippedBBox.min().z(); z <= clippedBBox.max().z(); z = tileMax.z() + 1) {
                xyz.setZ(z);

                // Get the table index of the tile that contains voxel (x, y, z).
                const auto n = this->coordToOffset(xyz);

                // Retrieve the child node at index n, or replace the tile at index n with a child.
                ChildT* child = nullptr;
                if (this->isChildMaskOn(n)) {
                    child = mNodes[n].getChild();
                } else {
                    // Replace the tile with a newly-created child that is filled
                    // with the tile's value and active state.
                    child = new ChildT{xyz, mNodes[n].getValue(), this->isValueMaskOn(n)};
                    this->setChildNode(n, child);
                }

                // Get the bounds of the tile that contains voxel (x, y, z).
                tileMin = this->offsetToGlobalCoord(n);
                tileMax = tileMin.offsetBy(ChildT::DIM - 1);

                // Forward the fill request to the child.
                child->denseFill(CoordBBox{xyz, clippedBBox.max()}, value, active);
            }
        }
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::writeTopology(std::ostream& os, bool toHalf) const
{
    mChildMask.save(os);
    mValueMask.save(os);

    {
        // Copy all of this node's values into an array.
        std::unique_ptr<ValueType[]> valuePtr(new ValueType[NUM_VALUES]);
        ValueType* values = valuePtr.get();
        const ValueType zero = zeroVal<ValueType>();
        for (Index i = 0; i < NUM_VALUES; ++i) {
            values[i] = (mChildMask.isOff(i) ? mNodes[i].getValue() : zero);
        }
        // Compress (optionally) and write out the contents of the array.
        io::writeCompressedValues(os, values, NUM_VALUES, mValueMask, mChildMask, toHalf);
    }
    // Write out the child nodes in order.
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        iter->writeTopology(os, toHalf);
    }
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::readTopology(std::istream& is, bool fromHalf)
{
    const ValueType background = (!io::getGridBackgroundValuePtr(is) ? zeroVal<ValueType>()
        : *static_cast<const ValueType*>(io::getGridBackgroundValuePtr(is)));

    mChildMask.load(is);
    mValueMask.load(is);

    if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_INTERNALNODE_COMPRESSION) {
        for (Index i = 0; i < NUM_VALUES; ++i) {
            if (this->isChildMaskOn(i)) {
                ChildNodeType* child =
                    new ChildNodeType(PartialCreate(), offsetToGlobalCoord(i), background);
                mNodes[i].setChild(child);
                child->readTopology(is);
            } else {
                ValueType value;
                is.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
                mNodes[i].setValue(value);
            }
        }
    } else {
        const bool oldVersion =
            (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_NODE_MASK_COMPRESSION);
        const Index numValues = (oldVersion ? mChildMask.countOff() : NUM_VALUES);
        {
            // Read in (and uncompress, if necessary) all of this node's values
            // into a contiguous array.
            std::unique_ptr<ValueType[]> valuePtr(new ValueType[numValues]);
            ValueType* values = valuePtr.get();
            io::readCompressedValues(is, values, numValues, mValueMask, fromHalf);

            // Copy values from the array into this node's table.
            if (oldVersion) {
                Index n = 0;
                for (ValueAllIter iter = this->beginValueAll(); iter; ++iter) {
                    mNodes[iter.pos()].setValue(values[n++]);
                }
                assert(n == numValues);
            } else {
                for (ValueAllIter iter = this->beginValueAll(); iter; ++iter) {
                    mNodes[iter.pos()].setValue(values[iter.pos()]);
                }
            }
        }
        // Read in all child nodes and insert them into the table at their proper locations.
        for (ChildOnIter iter = this->beginChildOn(); iter; ++iter) {
            ChildNodeType* child = new ChildNodeType(PartialCreate(), iter.getCoord(), background);
            mNodes[iter.pos()].setChild(child);
            child->readTopology(is, fromHalf);
        }
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::negate()
{
    for (Index i = 0; i < NUM_VALUES; ++i) {
        if (this->isChildMaskOn(i)) {
            mNodes[i].getChild()->negate();
        } else {
            mNodes[i].setValue(math::negative(mNodes[i].getValue()));
        }
    }

}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
struct InternalNode<ChildT, Log2Dim>::VoxelizeActiveTiles
{
    VoxelizeActiveTiles(InternalNode &node) : mNode(&node) {
        //(*this)(tbb::blocked_range<Index>(0, NUM_VALUES));//single thread for debugging
        tbb::parallel_for(tbb::blocked_range<Index>(0, NUM_VALUES), *this);

        node.mChildMask |= node.mValueMask;
        node.mValueMask.setOff();
    }
    void operator()(const tbb::blocked_range<Index> &r) const
    {
        for (Index i = r.begin(), end=r.end(); i!=end; ++i) {
            if (mNode->mChildMask.isOn(i)) {// Loop over node's child nodes
                mNode->mNodes[i].getChild()->voxelizeActiveTiles(true);
            } else if (mNode->mValueMask.isOn(i)) {// Loop over node's active tiles
                const Coord &ijk = mNode->offsetToGlobalCoord(i);
                ChildNodeType *child = new ChildNodeType(ijk, mNode->mNodes[i].getValue(), true);
                child->voxelizeActiveTiles(true);
                mNode->mNodes[i].setChild(child);
            }
        }
    }
    InternalNode* mNode;
};// VoxelizeActiveTiles

template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::voxelizeActiveTiles(bool threaded)
{
    if (threaded) {
        VoxelizeActiveTiles tmp(*this);
    } else {
        for (ValueOnIter iter = this->beginValueOn(); iter; ++iter) {
            this->setChildNode(iter.pos(),
                new ChildNodeType(iter.getCoord(), iter.getValue(), true));
        }
        for (ChildOnIter iter = this->beginChildOn(); iter; ++iter)
            iter->voxelizeActiveTiles(false);
    }
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::writeBuffers(std::ostream& os, bool toHalf) const
{
    for (ChildOnCIter iter = this->cbeginChildOn(); iter; ++iter) {
        iter->writeBuffers(os, toHalf);
    }
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::readBuffers(std::istream& is, bool fromHalf)
{
    for (ChildOnIter iter = this->beginChildOn(); iter; ++iter) {
        iter->readBuffers(is, fromHalf);
    }
}


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::readBuffers(std::istream& is,
    const CoordBBox& clipBBox, bool fromHalf)
{
    for (ChildOnIter iter = this->beginChildOn(); iter; ++iter) {
        // Stream in the branch rooted at this child.
        // (We can't skip over children that lie outside the clipping region,
        // because buffers are serialized in depth-first order and need to be
        // unserialized in the same order.)
        iter->readBuffers(is, clipBBox, fromHalf);
    }

    // Get this tree's background value.
    ValueType background = zeroVal<ValueType>();
    if (const void* bgPtr = io::getGridBackgroundValuePtr(is)) {
        background = *static_cast<const ValueType*>(bgPtr);
    }
    this->clip(clipBBox, background);
}


////////////////////////////////////////


template<typename ChildT, Index Log2Dim>
inline void
InternalNode<ChildT, Log2Dim>::resetBackground(const ValueType& oldBackground,
                                               const ValueType& newBackground)
{
    if (math::isExactlyEqual(oldBackground, newBackground)) return;
    for (Index i = 0; i < NUM_VALUES; ++i) {
       if (this->isChildMaskOn(i)) {
           mNodes[i].getChild()->resetBackground(oldBackground, newBackground);
       } else if (this->isValueMaskOff(i)) {
           if (math::isApproxEqual(mNodes[i].getValue(), oldBackground)) {
               mNodes[i].setValue(newBackground);
           } else if (math::isApproxEqual(mNodes[i].getValue(), math::negative(oldBackground))) {
               mNodes[i].setValue(math::negative(newBackground));
           }
       }
    }
}

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
