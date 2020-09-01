// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

#include <openvdb/util/NodeMasks.h> // for backward compatibility only (see readTopology())

#include <tbb/parallel_for.h>

#include "RootNode.h"


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {


////////////////////////////////////////


template<typename ChildT>
inline void
RootNode<ChildT>::setBackground(const ValueType& background, bool updateChildNodes)
{
    if (math::isExactlyEqual(background, mBackground)) return;

    if (updateChildNodes) {
        // Traverse the tree, replacing occurrences of mBackground with background
        // and -mBackground with -background.
        for (MapIter iter=mTable.begin(); iter!=mTable.end(); ++iter) {
            ChildT *child = iter->second.child;
            if (child) {
                child->resetBackground(/*old=*/mBackground, /*new=*/background);
            } else {
                Tile& tile = getTile(iter);
                if (tile.active) continue;//only change inactive tiles
                if (math::isApproxEqual(tile.value, mBackground)) {
                    tile.value = background;
                } else if (math::isApproxEqual(tile.value, math::negative(mBackground))) {
                    tile.value = math::negative(background);
                }
            }
        }
    }
    mBackground = background;
}

template<typename ChildT>
inline bool
RootNode<ChildT>::isBackgroundTile(const Tile& tile) const
{
    return !tile.active && math::isApproxEqual(tile.value, mBackground);
}

template<typename ChildT>
inline bool
RootNode<ChildT>::isBackgroundTile(const MapIter& iter) const
{
    return isTileOff(iter) && math::isApproxEqual(getTile(iter).value, mBackground);
}

template<typename ChildT>
inline bool
RootNode<ChildT>::isBackgroundTile(const MapCIter& iter) const
{
    return isTileOff(iter) && math::isApproxEqual(getTile(iter).value, mBackground);
}


template<typename ChildT>
inline size_t
RootNode<ChildT>::numBackgroundTiles() const
{
    size_t count = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (this->isBackgroundTile(i)) ++count;
    }
    return count;
}


template<typename ChildT>
inline size_t
RootNode<ChildT>::eraseBackgroundTiles()
{
    std::set<Coord> keysToErase;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (this->isBackgroundTile(i)) keysToErase.insert(i->first);
    }
    for (std::set<Coord>::iterator i = keysToErase.begin(), e = keysToErase.end(); i != e; ++i) {
        mTable.erase(*i);
    }
    return keysToErase.size();
}


////////////////////////////////////////


template<typename ChildT>
inline Index64
RootNode<ChildT>::memUsage() const
{
    Index64 sum = sizeof(*this);
    for (MapCIter iter=mTable.begin(); iter!=mTable.end(); ++iter) {
        if (const ChildT *child = iter->second.child) {
            sum += child->memUsage();
        }
    }
    return sum;
}


template<typename ChildT>
inline void
RootNode<ChildT>::evalActiveBoundingBox(CoordBBox& bbox, bool visitVoxels) const
{
    for (MapCIter iter=mTable.begin(); iter!=mTable.end(); ++iter) {
        if (const ChildT *child = iter->second.child) {
            child->evalActiveBoundingBox(bbox, visitVoxels);
        } else if (isTileOn(iter)) {
            bbox.expand(iter->first, ChildT::DIM);
        }
    }
}


template<typename ChildT>
inline Index
RootNode<ChildT>::getChildCount() const {
    Index sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) ++sum;
    }
    return sum;
}


template<typename ChildT>
inline Index
RootNode<ChildT>::getTileCount() const
{
    Index sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isTile(i)) ++sum;
    }
    return sum;
}


template<typename ChildT>
inline Index
RootNode<ChildT>::getActiveTileCount() const
{
    Index sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isTileOn(i)) ++sum;
    }
    return sum;
}


template<typename ChildT>
inline Index
RootNode<ChildT>::getInactiveTileCount() const
{
    Index sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isTileOff(i)) ++sum;
    }
    return sum;
}


template<typename ChildT>
inline Index32
RootNode<ChildT>::leafCount() const
{
    Index32 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) sum += getChild(i).leafCount();
    }
    return sum;
}


template<typename ChildT>
inline Index32
RootNode<ChildT>::nonLeafCount() const
{
    Index32 sum = 1;
    if (ChildT::LEVEL != 0) {
        for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
            if (isChild(i)) sum += getChild(i).nonLeafCount();
        }
    }
    return sum;
}


template<typename ChildT>
inline Index32
RootNode<ChildT>::childCount() const
{
    return this->getChildCount();
}


template<typename ChildT>
inline Index64
RootNode<ChildT>::onVoxelCount() const
{
    Index64 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) {
            sum += getChild(i).onVoxelCount();
        } else if (isTileOn(i)) {
            sum += ChildT::NUM_VOXELS;
        }
    }
    return sum;
}


template<typename ChildT>
inline Index64
RootNode<ChildT>::offVoxelCount() const
{
    Index64 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) {
            sum += getChild(i).offVoxelCount();
        } else if (isTileOff(i) && !this->isBackgroundTile(i)) {
            sum += ChildT::NUM_VOXELS;
        }
    }
    return sum;
}


template<typename ChildT>
inline Index64
RootNode<ChildT>::onLeafVoxelCount() const
{
    Index64 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) sum += getChild(i).onLeafVoxelCount();
    }
    return sum;
}


template<typename ChildT>
inline Index64
RootNode<ChildT>::offLeafVoxelCount() const
{
    Index64 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) sum += getChild(i).offLeafVoxelCount();
    }
    return sum;
}

template<typename ChildT>
inline Index64
RootNode<ChildT>::onTileCount() const
{
    Index64 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) {
            sum += getChild(i).onTileCount();
        } else if (isTileOn(i)) {
            sum += 1;
        }
    }
    return sum;
}

template<typename ChildT>
inline void
RootNode<ChildT>::nodeCount(std::vector<Index32> &vec) const
{
    assert(vec.size() > LEVEL);
    Index32 sum = 0;
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) {
            ++sum;
            getChild(i).nodeCount(vec);
        }
    }
    vec[LEVEL] = 1;// one root node
    vec[ChildNodeType::LEVEL] = sum;
}


////////////////////////////////////////


template<typename ChildT>
inline void
RootNode<ChildT>::fill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    if (bbox.empty()) return;

    // Iterate over the fill region in axis-aligned, tile-sized chunks.
    // (The first and last chunks along each axis might be smaller than a tile.)
    Coord xyz, tileMax;
    for (int x = bbox.min().x(); x <= bbox.max().x(); x = tileMax.x() + 1) {
        xyz.setX(x);
        for (int y = bbox.min().y(); y <= bbox.max().y(); y = tileMax.y() + 1) {
            xyz.setY(y);
            for (int z = bbox.min().z(); z <= bbox.max().z(); z = tileMax.z() + 1) {
                xyz.setZ(z);

                // Get the bounds of the tile that contains voxel (x, y, z).
                Coord tileMin = coordToKey(xyz);
                tileMax = tileMin.offsetBy(ChildT::DIM - 1);

                if (xyz != tileMin || Coord::lessThan(bbox.max(), tileMax)) {
                    // If the box defined by (xyz, bbox.max()) doesn't completely enclose
                    // the tile to which xyz belongs, create a child node (or retrieve
                    // the existing one).
                    ChildT* child = nullptr;
                    MapIter iter = this->findKey(tileMin);
                    if (iter == mTable.end()) {
                        // No child or tile exists.  Create a child and initialize it
                        // with the background value.
                        child = new ChildT(xyz, mBackground);
                        mTable[tileMin] = NodeStruct(*child);
                    } else if (isTile(iter)) {
                        // Replace the tile with a newly-created child that is filled
                        // with the tile's value and active state.
                        const Tile& tile = getTile(iter);
                        child = new ChildT(xyz, tile.value, tile.active);
                        mTable[tileMin] = NodeStruct(*child);
                    } else if (isChild(iter)) {
                        child = &getChild(iter);
                    }
                    // Forward the fill request to the child.
                    if (child) {
                        const Coord tmp = Coord::minComponent(bbox.max(), tileMax);
                        child->fill(CoordBBox(xyz, tmp), value, active);
                    }
                } else {
                    // If the box given by (xyz, bbox.max()) completely encloses
                    // the tile to which xyz belongs, create the tile (if it
                    // doesn't already exist) and give it the fill value.
                    MapIter iter = this->findOrAddCoord(tileMin);
                    setTile(iter, Tile(value, active));
                }
            }
        }
    }
}


template<typename ChildT>
inline void
RootNode<ChildT>::denseFill(const CoordBBox& bbox, const ValueType& value, bool active)
{
    if (bbox.empty()) return;

    if (active && mTable.empty()) {
        // If this tree is empty, then a sparse fill followed by (threaded)
        // densification of active tiles is the more efficient approach.
        sparseFill(bbox, value, active);
        voxelizeActiveTiles(/*threaded=*/true);
        return;
    }

    // Iterate over the fill region in axis-aligned, tile-sized chunks.
    // (The first and last chunks along each axis might be smaller than a tile.)
    Coord xyz, tileMin, tileMax;
    for (int x = bbox.min().x(); x <= bbox.max().x(); x = tileMax.x() + 1) {
        xyz.setX(x);
        for (int y = bbox.min().y(); y <= bbox.max().y(); y = tileMax.y() + 1) {
            xyz.setY(y);
            for (int z = bbox.min().z(); z <= bbox.max().z(); z = tileMax.z() + 1) {
                xyz.setZ(z);

                // Get the bounds of the tile that contains voxel (x, y, z).
                tileMin = coordToKey(xyz);
                tileMax = tileMin.offsetBy(ChildT::DIM - 1);

                // Retrieve the table entry for the tile that contains xyz,
                // or, if there is no table entry, add a background tile.
                const auto iter = findOrAddCoord(tileMin);

                if (isTile(iter)) {
                    // If the table entry is a tile, replace it with a child node
                    // that is filled with the tile's value and active state.
                    const auto& tile = getTile(iter);
                    auto* child = new ChildT{tileMin, tile.value, tile.active};
                    setChild(iter, *child);
                }
                // Forward the fill request to the child.
                getChild(iter).denseFill(bbox, value, active);
            }
        }
    }
}


////////////////////////////////////////


template<typename ChildT>
inline bool
RootNode<ChildT>::addChild(ChildT* child)
{
    if (!child) return false;
    const Coord& xyz = child->origin();
    MapIter iter = this->findCoord(xyz);
    if (iter == mTable.end()) {//background
        mTable[this->coordToKey(xyz)] = NodeStruct(*child);
    } else {//child or tile
        setChild(iter, *child);//this also deletes the existing child node
    }
    return true;
}

template<typename ChildT>
inline void
RootNode<ChildT>::addTile(const Coord& xyz, const ValueType& value, bool state)
{
    MapIter iter = this->findCoord(xyz);
    if (iter == mTable.end()) {//background
        mTable[this->coordToKey(xyz)] = NodeStruct(Tile(value, state));
    } else {//child or tile
        setTile(iter, Tile(value, state));//this also deletes the existing child node
    }
}

template<typename ChildT>
inline void
RootNode<ChildT>::addTile(Index level, const Coord& xyz,
                          const ValueType& value, bool state)
{
    if (LEVEL >= level) {
        MapIter iter = this->findCoord(xyz);
        if (iter == mTable.end()) {//background
            if (LEVEL > level) {
                ChildT* child = new ChildT(xyz, mBackground, false);
                mTable[this->coordToKey(xyz)] = NodeStruct(*child);
                child->addTile(level, xyz, value, state);
            } else {
                mTable[this->coordToKey(xyz)] = NodeStruct(Tile(value, state));
            }
        } else if (isChild(iter)) {//child
            if (LEVEL > level) {
                getChild(iter).addTile(level, xyz, value, state);
            } else {
                setTile(iter, Tile(value, state));//this also deletes the existing child node
            }
        } else {//tile
            if (LEVEL > level) {
                ChildT* child = new ChildT(xyz, getTile(iter).value, isTileOn(iter));
                setChild(iter, *child);
                child->addTile(level, xyz, value, state);
            } else {
                setTile(iter, Tile(value, state));
            }
        }
    }
}


////////////////////////////////////////


template<typename ChildT>
inline void
RootNode<ChildT>::voxelizeActiveTiles(bool threaded)
{
    // There is little point in threading over the root table since each tile
    // spans a huge index space (by default 4096^3) and hence we expect few
    // active tiles if any at all.  In fact, you're very likely to run out of
    // memory if this method is called on a tree with root-level active tiles!
    for (MapIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (this->isTileOff(i)) continue;
        ChildT* child = i->second.child;
        if (child == nullptr) {
            // If this table entry is an active tile (i.e., not off and not a child node),
            // replace it with a child node filled with active tiles of the same value.
            child = new ChildT{i->first, this->getTile(i).value, true};
            i->second.child = child;
        }
        child->voxelizeActiveTiles(threaded);
    }
}


template<typename ChildT>
inline void
RootNode<ChildT>::clip(const CoordBBox& clipBBox)
{
    const Tile bgTile(mBackground, /*active=*/false);

    // Iterate over a copy of this node's table so that we can modify the original.
    // (Copying the table copies child node pointers, not the nodes themselves.)
    MapType copyOfTable(mTable);
    for (MapIter i = copyOfTable.begin(), e = copyOfTable.end(); i != e; ++i) {
        const Coord& xyz = i->first; // tile or child origin
        CoordBBox tileBBox(xyz, xyz.offsetBy(ChildT::DIM - 1)); // tile or child bounds
        if (!clipBBox.hasOverlap(tileBBox)) {
            // This table entry lies completely outside the clipping region.  Delete it.
            setTile(this->findCoord(xyz), bgTile); // delete any existing child node first
            mTable.erase(xyz);
        } else if (!clipBBox.isInside(tileBBox)) {
            // This table entry does not lie completely inside the clipping region
            // and must be clipped.
            if (isChild(i)) {
                getChild(i).clip(clipBBox, mBackground);
            } else {
                // Replace this tile with a background tile, then fill the clip region
                // with the tile's original value.  (This might create a child branch.)
                tileBBox.intersect(clipBBox);
                const Tile& origTile = getTile(i);
                setTile(this->findCoord(xyz), bgTile);
                this->sparseFill(tileBBox, origTile.value, origTile.active);
            }
        } else {
            // This table entry lies completely inside the clipping region.  Leave it intact.
        }
    }
    this->prune(); // also erases root-level background tiles
}


template<typename ChildT>
inline void
RootNode<ChildT>::prune(const ValueType& tolerance)
{
    bool state = false;
    ValueType value = zeroVal<ValueType>();
    for (MapIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (this->isTile(i)) continue;
        this->getChild(i).prune(tolerance);
        if (this->getChild(i).isConstant(value, state, tolerance)) {
            this->setTile(i, Tile(value, state));
        }
    }
    this->eraseBackgroundTiles();
}


////////////////////////////////////////


template<typename ChildT>
inline bool
RootNode<ChildT>::writeTopology(std::ostream& os, bool toHalf) const
{
    if (!toHalf) {
        os.write(reinterpret_cast<const char*>(&mBackground), sizeof(ValueType));
    } else {
        ValueType truncatedVal = io::truncateRealToHalf(mBackground);
        os.write(reinterpret_cast<const char*>(&truncatedVal), sizeof(ValueType));
    }
    io::setGridBackgroundValuePtr(os, &mBackground);

    const Index numTiles = this->getTileCount(), numChildren = this->getChildCount();
    os.write(reinterpret_cast<const char*>(&numTiles), sizeof(Index));
    os.write(reinterpret_cast<const char*>(&numChildren), sizeof(Index));

    if (numTiles == 0 && numChildren == 0) return false;

    // Write tiles.
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) continue;
        os.write(reinterpret_cast<const char*>(i->first.asPointer()), 3 * sizeof(Int32));
        os.write(reinterpret_cast<const char*>(&getTile(i).value), sizeof(ValueType));
        os.write(reinterpret_cast<const char*>(&getTile(i).active), sizeof(bool));
    }
    // Write child nodes.
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isTile(i)) continue;
        os.write(reinterpret_cast<const char*>(i->first.asPointer()), 3 * sizeof(Int32));
        getChild(i).writeTopology(os, toHalf);
    }

    return true; // not empty
}


template<typename ChildT>
inline bool
RootNode<ChildT>::readTopology(std::istream& is, bool fromHalf)
{
    // Delete the existing tree.
    this->clear();

    if (io::getFormatVersion(is) < OPENVDB_FILE_VERSION_ROOTNODE_MAP) {
        // Read and convert an older-format RootNode.

        // For backward compatibility with older file formats, read both
        // outside and inside background values.
        is.read(reinterpret_cast<char*>(&mBackground), sizeof(ValueType));
        ValueType inside;
        is.read(reinterpret_cast<char*>(&inside), sizeof(ValueType));

        io::setGridBackgroundValuePtr(is, &mBackground);

        // Read the index range.
        Coord rangeMin, rangeMax;
        is.read(reinterpret_cast<char*>(rangeMin.asPointer()), 3 * sizeof(Int32));
        is.read(reinterpret_cast<char*>(rangeMax.asPointer()), 3 * sizeof(Int32));

        this->initTable();
        Index tableSize = 0, log2Dim[4] = { 0, 0, 0, 0 };
        Int32 offset[3];
        for (int i = 0; i < 3; ++i) {
            offset[i] = rangeMin[i] >> ChildT::TOTAL;
            rangeMin[i] = offset[i] << ChildT::TOTAL;
            log2Dim[i] = 1 + util::FindHighestOn((rangeMax[i] >> ChildT::TOTAL) - offset[i]);
            tableSize += log2Dim[i];
            rangeMax[i] = (((1 << log2Dim[i]) + offset[i]) << ChildT::TOTAL) - 1;
        }
        log2Dim[3] = log2Dim[1] + log2Dim[2];
        tableSize = 1U << tableSize;

        // Read masks.
        util::RootNodeMask childMask(tableSize), valueMask(tableSize);
        childMask.load(is);
        valueMask.load(is);

        // Read child nodes/values.
        for (Index i = 0; i < tableSize; ++i) {
            // Compute origin = offset2coord(i).
            Index n = i;
            Coord origin;
            origin[0] = (n >> log2Dim[3]) + offset[0];
            n &= (1U << log2Dim[3]) - 1;
            origin[1] = (n >> log2Dim[2]) + offset[1];
            origin[2] = (n & ((1U << log2Dim[2]) - 1)) + offset[1];
            origin <<= ChildT::TOTAL;

            if (childMask.isOn(i)) {
                // Read in and insert a child node.
                ChildT* child = new ChildT(PartialCreate(), origin, mBackground);
                child->readTopology(is);
                mTable[origin] = NodeStruct(*child);
            } else {
                // Read in a tile value and insert a tile, but only if the value
                // is either active or non-background.
                ValueType value;
                is.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
                if (valueMask.isOn(i) || (!math::isApproxEqual(value, mBackground))) {
                    mTable[origin] = NodeStruct(Tile(value, valueMask.isOn(i)));
                }
            }
        }
        return true;
    }

    // Read a RootNode that was stored in the current format.

    is.read(reinterpret_cast<char*>(&mBackground), sizeof(ValueType));
    io::setGridBackgroundValuePtr(is, &mBackground);

    Index numTiles = 0, numChildren = 0;
    is.read(reinterpret_cast<char*>(&numTiles), sizeof(Index));
    is.read(reinterpret_cast<char*>(&numChildren), sizeof(Index));

    if (numTiles == 0 && numChildren == 0) return false;

    Int32 vec[3];
    ValueType value;
    bool active;

    // Read tiles.
    for (Index n = 0; n < numTiles; ++n) {
        is.read(reinterpret_cast<char*>(vec), 3 * sizeof(Int32));
        is.read(reinterpret_cast<char*>(&value), sizeof(ValueType));
        is.read(reinterpret_cast<char*>(&active), sizeof(bool));
        mTable[Coord(vec)] = NodeStruct(Tile(value, active));
    }

    // Read child nodes.
    for (Index n = 0; n < numChildren; ++n) {
        is.read(reinterpret_cast<char*>(vec), 3 * sizeof(Int32));
        Coord origin(vec);
        ChildT* child = new ChildT(PartialCreate(), origin, mBackground);
        child->readTopology(is, fromHalf);
        mTable[Coord(vec)] = NodeStruct(*child);
    }

    return true; // not empty
}


template<typename ChildT>
inline void
RootNode<ChildT>::writeBuffers(std::ostream& os, bool toHalf) const
{
    for (MapCIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) getChild(i).writeBuffers(os, toHalf);
    }
}


template<typename ChildT>
inline void
RootNode<ChildT>::readBuffers(std::istream& is, bool fromHalf)
{
    for (MapIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) getChild(i).readBuffers(is, fromHalf);
    }
}


template<typename ChildT>
inline void
RootNode<ChildT>::readBuffers(std::istream& is, const CoordBBox& clipBBox, bool fromHalf)
{
    const Tile bgTile(mBackground, /*active=*/false);

    for (MapIter i = mTable.begin(), e = mTable.end(); i != e; ++i) {
        if (isChild(i)) {
            // Stream in and clip the branch rooted at this child.
            // (We can't skip over children that lie outside the clipping region,
            // because buffers are serialized in depth-first order and need to be
            // unserialized in the same order.)
            ChildT& child = getChild(i);
            child.readBuffers(is, clipBBox, fromHalf);
        }
    }
    // Clip root-level tiles and prune children that were clipped.
    this->clip(clipBBox);
}


} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb
