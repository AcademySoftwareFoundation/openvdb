// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridBuilder.h

    \author Ken Museth

    \date June 26, 2020

    \brief Generates a NanoVDB grid from any volume or function.

    \note This is only intended as a simple tool to generate nanovdb grids without
          any dependency on openvdb.
*/

#ifndef NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED

#include "GridHandle.h"
#include "MultiThreading.h"

#include <map>
#include <limits>
#include <atomic>
#include <sstream>// for stringstream
#include <vector>
#include <cstring>// for memcpy

namespace nanovdb {

/// @brief Returns a handle to a narrow-band level set of a sphere
///
/// @param radius    Radius of sphere in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetSphere(ValueT radius = 100, 
                     const Vec3d& center = Vec3d(0),
                     ValueT voxelSize = 1.0,
                     ValueT halfWidth = 3.0,
                     const Vec3d &origin = Vec3d(0),
                     const std::string &name = "sphere_ls",
                     const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a sphere such
///        that the eterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the 
///        sphere to 1 at the halfWidth and interior of the sphere. 
///
/// @param radius    Radius of sphere in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template<typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeSphere(ValueT             radius = 100,
                      const Vec3d&       center = Vec3d(0),
                      ValueT             voxelSize = 1.0,
                      ValueT             halfWidth = 3.0,
                      const Vec3d&       origin = Vec3d(0),
                      const std::string& name = "sphere_fog",
                      const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered 
///        on the surface of a sphere. 
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param radius         Radius of sphere in world units
/// @param center         Center of sphere in world units
/// @param voxelSize      Size of a voxel in world units
/// @param origin         Origin of grid in world units
/// @param name           Name of the grid
/// @param buffer         Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointSphere(int pointsPerVoxel = 1,
                  ValueT radius = 100,
                  const Vec3d& center = Vec3d(0),
                  ValueT voxelSize = 1.0,
                  const Vec3d &origin = Vec3d(0),
                  const std::string &name = "sphere_points",
                  const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a torus in the xz-plane
///
/// @param majorRadius Major radius of torus in world units
/// @param minorRadius Minor radius of torus in world units
/// @param center      Center of sphere in world units
/// @param voxelSize   Size of a voxel in world units
/// @param halfWidth   Half-width of narrow band in voxel units
/// @param origin      Origin of grid in world units
/// @param name        Name of the grid
/// @param buffer      Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetTorus(ValueT majorRadius = 100, 
                    ValueT minorRadius = 50,
                    const Vec3d& center = Vec3d(0),
                    ValueT voxelSize = 1.0,
                    ValueT halfWidth = 3.0,
                    const Vec3d &origin = Vec3d(0), 
                    const std::string &name = "torus_ls",
                    const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a torus in the xz-plane such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the 
///        torus to 1 at the halfWidth and interior of the sphere. 
///
/// @param majorRadius Major radius of torus in world units
/// @param minorRadius Minor radius of torus in world units
/// @param center      Center of sphere in world units
/// @param voxelSize   Size of a voxel in world units
/// @param halfWidth   Half-width of narrow band in voxel units
/// @param origin      Origin of grid in world units
/// @param name        Name of the grid
/// @param buffer      Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeTorus(ValueT majorRadius = 100, 
                     ValueT minorRadius = 50,
                     const Vec3d& center = Vec3d(0),
                     ValueT voxelSize = 1.0,
                     ValueT halfWidth = 3.0,
                     const Vec3d &origin = Vec3d(0), 
                     const std::string &name = "torus_fog",
                     const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered 
///        on the surface of a torus. 
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param majorRadius    Major radius of torus in world units
/// @param minorRadius    Minor radius of torus in world units
/// @param center         Center of sphere in world units
/// @param voxelSize      Size of a voxel in world units
/// @param origin         Origin of grid in world units
/// @param name           Name of the grid
/// @param buffer         Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointTorus(int pointsPerVoxel = 1,// half-width of narrow band in voxel units
                 ValueT majorRadius = 100,// major radius of torus in world units
                 ValueT minorRadius = 50,// minor radius of torus in world units
                 const Vec3d& center = Vec3d(0), //center of sphere in world units
                 ValueT voxelSize = 1.0, // size of a voxel in world units 
                 const Vec3d& origin = Vec3d(0), // origin of grid in world units
                 const std::string& name = "torus_points",// name of grid
                 const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a box
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetBox(ValueT width = 40,
                  ValueT height = 60,
                  ValueT depth = 100,
                  const Vec3d& center = Vec3d(0),
                  ValueT voxelSize = 1.0,
                  ValueT halfWidth = 3.0, 
                  const Vec3d &origin = Vec3d(0),
                  const std::string &name = "box_ls",
                  const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a narrow-band level set of a bounding-box (= wireframe of a box)
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param thickness Thickness of the wire in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createLevelSetBBox(ValueT width = 40,
                   ValueT height = 60,
                   ValueT depth = 100,
                   ValueT thickness = 10,
                   const Vec3d& center = Vec3d(0),
                   ValueT voxelSize = 1.0,
                   ValueT halfWidth = 3.0, 
                   const Vec3d &origin = Vec3d(0),
                   const std::string &name = "bbox_ls",
                   const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a sparse fog volume of a box such
///        that the exterior is 0 and inactive, the interior is active
///        with values varying smoothly from 0 at the surface of the 
///        box to 1 at the halfWidth and interior of the sphere. 
///
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param halfWidth Half-width of narrow band in voxel units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
GridHandle<BufferT>
createFogVolumeBox(ValueT width = 40,
                   ValueT height = 60,
                   ValueT depth = 100,
                   const Vec3d& center = Vec3d(0),
                   ValueT voxelSize = 1.0,
                   ValueT halfWidth = 3.0, 
                   const Vec3d &origin = Vec3d(0),
                   const std::string &name = "box_ls",
                   const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Returns a handle to a PointDataGrid containing points scattered 
///        on the surface of a box. 
///
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param width     Width of box in world units
/// @param height    Height of box in world units
/// @param depth     Depth of box in world units
/// @param center    Center of sphere in world units
/// @param voxelSize Size of a voxel in world units
/// @param origin    Origin of grid in world units
/// @param name      Name of the grid
/// @param buffer    Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointBox(int pointsPerVoxel = 1,// half-width of narrow band in voxel units
               ValueT width = 40,// width of box in world units
               ValueT height = 60,// height of box in world units
               ValueT depth = 100,// depth of box in world units
               const Vec3d& center = Vec3d(0),//center of sphere in world units
               ValueT voxelSize = 1.0,// size of a voxel in world units
               const Vec3d &origin = Vec3d(0),// origin of grid in world units
               const std::string& name = "box_points",// name of grid
               const BufferT& buffer = BufferT());

//================================================================================================

/// @brief Given an input NanoVDB voxel grid this methods returns a GridHandle to another NanoVDB
///        PointDataGrid with points scattered in the active leaf voxels of in input grid.
///
/// @param srcGrid        Const input grid used to determine the active voxels to scatter point intp 
/// @param pointsPerVoxel Number of point per voxel on on the surface
/// @param name           Name of the grid
/// @param buffer         Buffer used for memory allocation by the handle
template <typename ValueT = float, typename BufferT = HostBuffer>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<ValueT> &srcGrid,// origin of grid in world units
                   int pointsPerVoxel = 1,// half-width of narrow band in voxel units
                   const std::string &name = "point_scatter",// name of grid
                   const BufferT& buffer = BufferT());

//================================================================================================

template<typename T>
class Extrema
{
    T mMin, mMax;
public:
    Extrema() : mMin(std::numeric_limits<T>::max()), mMax(std::numeric_limits<T>::min()) {}
    Extrema(const T &v) : mMin(v), mMax(v) {}
    Extrema(const T &a, const T &b) : mMin(a), mMax(b) {}
    Extrema& operator=(const Extrema&) = default;
    void min(const T &v) { if (v < mMin) mMin = v; }
    void max(const T &v) { if (v > mMax) mMax = v; }
    void operator()(const T &v) {
        if (v < mMin) {
            mMin = v;
        } else if (v > mMax) {
            mMax = v;
        }
    }
    const T& min() const { return mMin; }
    const T& max() const { return mMax; }
    operator bool() const { return mMin <= mMax; }
};// Extrema

// Template specialization
template<typename T>
class Extrema<Vec3<T>>
{
    Vec3<T> mMin, mMax;
public:
    Extrema() : mMin(std::numeric_limits<T>::max()), mMax(std::numeric_limits<T>::min()) {}
    Extrema(const Vec3<T> &v) : mMin(v), mMax(v) {}
    Extrema(const Vec3<T> &a, const Vec3<T> &b) : mMin(a), mMax(b) {}
    Extrema& operator=(const Extrema&) = default;
    void min(const Vec3<T> &v) { for (int i=0; i<3; ++i) if (v[i] < mMin[i]) mMin[i] = v[i]; }
    void max(const Vec3<T> &v) { for (int i=0; i<3; ++i) if (v[i] > mMax[i]) mMax[i] = v[i]; }
    void operator()(const Vec3<T> &v) {
        for (int i=0; i<3; ++i) {
            if (v[i] < mMin[i]) {
                mMin[i] = v[i];
            } else if (v[i] > mMax[i]) {
                mMax[i] = v[i];
            }
        }
    }
    const Vec3<T>& min() const { return mMin; }
    const Vec3<T>& max() const { return mMax; }
    operator bool() const { return mMin[0] <= mMax[0] && mMin[1] <= mMax[1] && mMin[2] <= mMax[2]; }
};// Extrema


/// @brief Allows for the construction of NanoVDB grids without any dependecy
template <typename ValueT, typename ExtremaOp = Extrema<ValueT> >
class GridBuilder
{
    struct Leaf;
    template<typename ChildT>
    struct Node;
    template<typename ChildT>
    struct Root;
    struct ValueAccessor;

    using SrcNode0 = Leaf;
    using SrcNode1 = Node<SrcNode0>;
    using SrcNode2 = Node<SrcNode1>;
    using SrcRootT = Root<SrcNode2>;

    using DstNode0 = nanovdb::LeafNode<ValueT>; // leaf
    using DstNode1 = nanovdb::InternalNode<DstNode0>; // lower
    using DstNode2 = nanovdb::InternalNode<DstNode1>; // upper
    using DstRootT = nanovdb::RootNode<DstNode2>;
    using DstTreeT = nanovdb::Tree<DstRootT>;
    using DstGridT = nanovdb::Grid<DstTreeT>;
    
    ValueT   mDelta;// skip node if: node.max < -mDelta || node.min > mDelta
    SrcRootT mRoot;
    uint8_t* mData;
    uint64_t mBytes[7]; // Byte offsets to from mData to: tree, root, node2, node1, leafs, meta, (total size)
    std::atomic<uint64_t> mActiveVoxelCount;
    std::vector<SrcNode0*> mArray0; // leaf nodes
    std::vector<SrcNode1*> mArray1; // lower internal nodes
    std::vector<SrcNode2*> mArray2; // upper internal nodes
    uint64_t mBlindDataSize;

    template<typename DstNodeT>
    typename DstNodeT::DataType* nodeData() const { return reinterpret_cast<typename DstNodeT::DataType*>(mData + mBytes[4 - DstNodeT::LEVEL]); }
    typename DstTreeT::DataType* treeData() const { return reinterpret_cast<typename DstTreeT::DataType*>(mData + mBytes[0]); }
    typename DstGridT::DataType* gridData() const { return reinterpret_cast<typename DstGridT::DataType*>(mData); }
    
    // Below are private methods use to serialize nodes into NanoVDB
    void processLeafs();   
    template <typename SrcNodeT, typename DstNodeT>
    void processNodes(std::vector<SrcNodeT*>&);
    void processRoot();
    void processTree();
    void processGrid(const Map&, const std::string&, GridClass);
    
    template <typename SrcNodeT>
    void update(std::vector<SrcNodeT*>&);

    template<typename T, typename FlagT>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    setFlag(const T&, const T&, FlagT& flag) const { flag &= ~FlagT(1); } // unset first bit

    template<typename T, typename FlagT>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    setFlag(const T& min, const T& max, FlagT& flag) const;

public:
    GridBuilder(ValueT background, uint64_t blindDataSize = 0) 
        : mDelta(0), mRoot(background), mData(nullptr), mBlindDataSize(blindDataSize) {}

    ValueAccessor getAccessor() { return ValueAccessor(mRoot); }

    void sdfToLevelSet();

    void sdfToFog();

    template <typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(double voxelSize = 1.0, const Vec3d &gridOrigin = Vec3d(0), const std::string &name = "", GridClass gridClass = GridClass::Unknown, const BufferT& buffer = BufferT());

    template <typename BufferT = HostBuffer>
    GridHandle<BufferT> getHandle(const Map &map, const std::string &name = "", GridClass gridClass = GridClass::Unknown, const BufferT& buffer = BufferT());

    /// @brief Sets grids values in domain of the @a bbox to those returned by the specified @a func with the 
    ///        expected signature [](const Coord&)->ValueT.
    ///
    /// @note If @a func returns a value equal to the brackground value (specified in the constructor) at a 
    ///       specefic voxel coordinate, then the active state of that coordinate is left off! Else the value
    ///       value is set and the active state is on. This is done to allow for sparse grids to be generated.
    ///
    /// @param func  Functor used to evaluate the grid values in the @a bbox 
    /// @param bbox  Coordinate bounding-box over which the grid values will be set.
    /// @param delta Specifies a lower threshold value for rendering (optiona). Typically equals the voxel size
    ///              for level sets and otherwise it's zero.
    template <typename Func>
    void operator()(const Func &func, const CoordBBox &bbox, ValueT delta = ValueT(0));

};// GridBuilder

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename Func>
void GridBuilder<ValueT, ExtremaOp>::
operator()(const Func &func, const CoordBBox &voxelBBox, ValueT delta)
{
    static_assert(is_same<ValueT, typename std::result_of<Func(const Coord&)>::type>::value, "GridBuilder: mismatched ValueType");
    mDelta = delta;// delta = voxel size for level sets, else 0
    mActiveVoxelCount = 0;

    using NodeT = Leaf;
    //using NodeT = Node<Leaf>;
    const CoordBBox nodeBBox(voxelBBox[0] >> NodeT::TOTAL, voxelBBox[1] >> NodeT::TOTAL);
    std::mutex mutex;
    auto kernel = [&](const CoordBBox &b)
    {
        uint64_t sum = 0;
        NodeT *node = nullptr;
        for (auto it = b.begin(); it; ++it) {
            Coord min(*it << NodeT::TOTAL), max(min + Coord(NodeT::DIM - 1));
            const CoordBBox bbox( min.maxComponent(voxelBBox.min()), 
                                  max.minComponent(voxelBBox.max()) );
            if (node == nullptr) {
                node = new NodeT( bbox[0], mRoot.mBackground, false );
            } else {
                node->mOrigin = bbox[0] & ~NodeT::MASK;
            }
            uint64_t count = 0;
            for (auto ijk = bbox.begin(); ijk; ++ijk) {
                const auto v = func( *ijk );
                if ( v == mRoot.mBackground ) continue;
                ++count;
                node->setValue( *ijk, v );
            }
            if (count>0) {
                sum += count;
                std::lock_guard<std::mutex> guard(mutex);
                assert(node != nullptr);
                mRoot.addNode(node); 
                assert(node == nullptr);
            }
        }
        if (node) delete node;
        mActiveVoxelCount += sum;
    };// kernel
    parallel_for(nodeBBox, kernel);
}

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename SrcNodeT>
void GridBuilder<ValueT, ExtremaOp>::
update(std::vector<SrcNodeT*> &array)
{
    const uint32_t nodeCount = mRoot.template nodeCount<SrcNodeT>();
    if (nodeCount != uint32_t(array.size())) {
        array.clear();
        array.reserve(nodeCount);
        mRoot.getNodes(array);
    }
}// GridBuilder::update

//================================================================================================

template <typename ValueT, typename ExtremaOp>
void GridBuilder<ValueT, ExtremaOp>::
sdfToLevelSet()
{
    const ValueT outside = mRoot.mBackground;
    // Note that the bottum-up flood filling is essential
    parallel_invoke([&](){this->update(mArray0);}, 
                    [&](){this->update(mArray1);}, 
                    [&](){this->update(mArray2);});
    parallel_for(0, mArray0.size(), 8,[&](const BlockedRange<size_t> &r){
        for (auto i = r.begin(); i != r.end(); ++i) mArray0[i]->signedFloodFill(outside);
    });
    parallel_for(0, mArray1.size(), 1,[&](const BlockedRange<size_t> &r){
        for (auto i = r.begin(); i != r.end(); ++i) mArray1[i]->signedFloodFill(outside);
    });
    parallel_for(0, mArray2.size(), 1,[&](const BlockedRange<size_t> &r){
        for (auto i = r.begin(); i != r.end(); ++i) mArray2[i]->signedFloodFill(outside);
    });
    mRoot.signedFloodFill(outside);
}// GridBuilder::sdfToLevelSet

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, ExtremaOp>::
getHandle(double dx,//voxel size
          const Vec3d &p0,// origin
          const std::string &name,
          GridClass gridClass,
          const BufferT& buffer)
{ 
    if (dx <= 0) {
        throw std::runtime_error("GridBuilder: voxel size is zero or negative");
    }
    Map map;// affine map
    const double Tx = p0[0], Ty = p0[1], Tz = p0[2];
    const double mat[4][4] = {
        {dx,  0.0, 0.0, 0.0},// row 0
        {0.0,  dx, 0.0, 0.0},// row 1
        {0.0, 0.0,  dx, 0.0},// row 2
        { Tx,  Ty,  Tz, 1.0},// row 3
    };
    const double invMat[4][4] = {
        {1/dx, 0.0, 0.0, 0.0},// row 0
        {0.0, 1/dx, 0.0, 0.0},// row 1
        {0.0, 0.0, 1/dx, 0.0},// row 2
        {-Tx, -Ty,  -Tz, 1.0},// row 3
    };
    map.set(mat, invMat, 1.0);
    return this->getHandle(map, name, gridClass, buffer);
}// GridBuilder::getHandle

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename BufferT>
GridHandle<BufferT> GridBuilder<ValueT, ExtremaOp>::
getHandle(const Map &map,
          const std::string &name,
          GridClass gridClass,
          const BufferT& buffer)
{ 
    if (gridClass == GridClass::LevelSet && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Level sets are expected to be floating point types");
    if (gridClass == GridClass::FogVolume && !is_floating_point<ValueT>::value)
        throw std::runtime_error("Fog volumes are expected to be floating point types");
    
    parallel_invoke([&](){this->update(mArray0);}, 
                    [&](){this->update(mArray1);}, 
                    [&](){this->update(mArray2);});

    mBytes[0] = DstGridT::memUsage(mBlindDataSize>0 ? 1 : 0); // grid + blind meta data
    mBytes[1] = DstTreeT::memUsage(); // tree
    mBytes[2] = DstRootT::memUsage(uint32_t(mRoot.mTable.size())); // root
    mBytes[3] = mArray2.size() * DstNode2::memUsage(); // upper internal nodes
    mBytes[4] = mArray1.size() * DstNode1::memUsage(); // lower internal nodes
    mBytes[5] = mArray0.size() * DstNode0::memUsage(); // leaf nodes
    mBytes[6] = mBlindDataSize;

    for (int i = 1; i < 7; ++i) {
        mBytes[i] += mBytes[i - 1]; // Byte offsets to: tree, root, node2, node1, leafs, meta, total
    }

    GridHandle<BufferT> handle(BufferT::create(mBytes[6], &buffer));
    mData = handle.data();

    this->processLeafs();
    this->template processNodes<SrcNode1, DstNode1>(mArray1);
    this->template processNodes<SrcNode2, DstNode2>(mArray2);
    this->processRoot();
    this->processTree();
    this->processGrid(map, name, gridClass);
    
    return handle; 
}// GridBuilder::getHandle

//================================================================================================

template<typename ValueT, typename ExtremaOp>
template<typename T, typename FlagT>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, ExtremaOp>::
setFlag(const T& min, const T& max, FlagT& flag) const
{
    if (mDelta > 0 && (min > mDelta || max < -mDelta)) {
        flag |= FlagT(1); // set first bit
    } else {
        flag &= ~FlagT(1); // unset first bit
    }
}

//================================================================================================

template<typename ValueT, typename ExtremaOp>
inline void GridBuilder<ValueT, ExtremaOp>::
sdfToFog()
{
    this->sdfToLevelSet();// performs signed flood fill

    const ValueT d = -mRoot.mBackground, w = 1.0f/d;
    auto op = [&](ValueT &v)->bool
    {
        if (v>ValueT(0)) {
            v = ValueT(0);
            return false;
        }
        v = v>d ? v*w : ValueT(1);
        return true; 
    };
    auto kernel0 = [&](const BlockedRange<size_t> &r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode0* node = mArray0[i];
            for (uint32_t i=0; i<SrcNode0::SIZE; ++i) node->mValueMask.set(i, op(node->mValues[i]));
        }
    };
    auto kernel1 = [&](const BlockedRange<size_t> &r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode1* node = mArray1[i];
            for (uint32_t i=0; i<SrcNode1::SIZE; ++i) {
                if (node->mChildMask.isOn(i)) {
                    SrcNode0 *leaf = node->mTable[i].child;
                    if (leaf->mValueMask.isOff()) {
                        node->mTable[i].value = leaf->getFirstValue();
                        node->mChildMask.setOff(i);
                        delete leaf;
                    }
                } else {
                    node->mValueMask.set(i, op(node->mTable[i].value));
                }
            }
        }
    };
    auto kernel2 = [&](const BlockedRange<size_t> &r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            SrcNode2* node = mArray2[i];
            for (uint32_t i=0; i<SrcNode2::SIZE; ++i) {
                if (node->mChildMask.isOn(i)) {
                    SrcNode1 *child = node->mTable[i].child;
                    if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
                        node->mTable[i].value = child->getFirstValue();
                        node->mChildMask.setOff(i);
                        delete child;
                    }
                } else {
                    node->mValueMask.set(i, op(node->mTable[i].value));
                }
            }
        }
    };
    parallel_for(0, mArray0.size(), 8, kernel0);
    parallel_for(0, mArray1.size(), 1, kernel1);
    parallel_for(0, mArray2.size(), 1, kernel2);

    for (auto it = mRoot.mTable.begin(); it != mRoot.mTable.end(); ++it) {
        SrcNode2 *child = it->second.child;
        if (child == nullptr) {
            it->second.state = op(it->second.value);
        } else if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
            it->second.value = child->getFirstValue();
            it->second.state = false;
            it->second.child = nullptr;
            delete child;
        }
    }
}// GridBuilder::sdfToFog

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridBuilder<ValueT, ExtremaOp>::
processLeafs()
{
    mActiveVoxelCount = 0;
    auto* start = this->template nodeData<DstNode0>(); // address of first leaf node
    auto kernel = [&](const BlockedRange<uint32_t> &r) {
        uint64_t sum = 0;
        auto* data = start + r.begin();
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            SrcNode0& srcLeaf = *mArray0[i];
            assert(srcLeaf.mID == i);
            sum += srcLeaf.mValueMask.countOn();
            data->mValueMask = srcLeaf.mValueMask;
            const ValueT* src = srcLeaf.mValues;
            for (ValueT *dst = data->mValues, *n = dst + SrcNode0::SIZE; dst != n; dst += 4, src += 4) {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
            auto iter = srcLeaf.mValueMask.beginOn();
            if (!iter) throw std::runtime_error("Expected at least one active voxel in every leaf node! Hint: try pruneInactive.");
            src = srcLeaf.mValues;
            ExtremaOp extrema( src[*iter] );
            CoordBBox bbox;// empty
            bbox.expand(SrcNode0::OffsetToLocalCoord(*iter));// initially use local coord for speed
            for (++iter; iter; ++iter) {
                bbox.expand(SrcNode0::OffsetToLocalCoord(*iter));
                extrema( src[*iter] );
            }
            assert(!bbox.empty());
            srcLeaf.localToGlobalCoord(bbox[0]);
            srcLeaf.localToGlobalCoord(bbox[1]);
            data->mBBoxDif[0] = uint8_t(bbox[1][0] - bbox[0][0]);
            data->mBBoxDif[1] = uint8_t(bbox[1][1] - bbox[0][1]);
            data->mBBoxDif[2] = uint8_t(bbox[1][2] - bbox[0][2]);
            data->mBBoxMin = bbox[0];
            data->mValueMin = extrema.min();
            data->mValueMax = extrema.max();
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
        }
        mActiveVoxelCount += sum;
    };
    parallel_for(BlockedRange<uint32_t>(0, uint32_t(mArray0.size()), 8), kernel);
} // GridBuilder::processLeafs

//================================================================================================

template<typename ValueT, typename ExtremaOp>
template<typename SrcNodeT, typename DstNodeT>
void GridBuilder<ValueT, ExtremaOp>::
processNodes(std::vector<SrcNodeT*>& array)
{
    using SrcChildT = typename SrcNodeT::ChildType;
    const uint32_t size = static_cast<uint32_t>(array.size());
    auto*          start = this->template nodeData<DstNodeT>();
    auto           kernel = [&](const BlockedRange<uint32_t> &r) 
    {
        auto* data = start + r.begin();
        uint64_t sum = 0;
        for (auto i = r.begin(); i != r.end(); ++i, ++data) {
            SrcNodeT& srcNode = *array[i];
            assert(srcNode.mID == i);
            sum += SrcChildT::NUM_VALUES * srcNode.mValueMask.countOn();// active tiles
            data->mValueMask = srcNode.mValueMask;
            data->mChildMask = srcNode.mChildMask;
            data->mOffset = size - i;
            auto noneChildMask = srcNode.mChildMask;//copy
            noneChildMask.toggle();// bits are on for values vs child nodes
            for (auto iter = noneChildMask.beginOn(); iter; ++iter) {
                data->mTable[*iter].value = srcNode.mTable[*iter].value;
            }
            auto onValIter = srcNode.mValueMask.beginOn();
            auto childIter = srcNode.mChildMask.beginOn();
            ExtremaOp extrema;
            if (onValIter) {
                extrema = ExtremaOp(srcNode.mTable[*onValIter].value);
                const Coord ijk = srcNode.offsetToGlobalCoord(*onValIter);
                data->mBBox[0] = ijk;
                data->mBBox[1] = ijk + Coord(int32_t(SrcChildT::DIM) - 1);
                ++onValIter;
            } else if (childIter) {
                data->mTable[*childIter].childID = srcNode.mTable[*childIter].child->mID;
                auto* dstChild = data->child(*childIter);
                extrema = ExtremaOp(dstChild->valueMin(), dstChild->valueMax());
                data->mBBox = dstChild->bbox();
                ++childIter;
            } else {
                throw std::runtime_error("Internal node with no children or active values! Hint: try pruneInactive.");
            }
            for (; onValIter; ++onValIter) { // typically there are few active tiles
                extrema( srcNode.mTable[*onValIter].value );
                const Coord ijk = srcNode.offsetToGlobalCoord(*onValIter);
                data->mBBox[0].minComponent(ijk);
                data->mBBox[1].maxComponent(ijk + Coord(int32_t(SrcChildT::DIM) - 1));
            }
            for (; childIter; ++childIter) {
                data->mTable[*childIter].childID = srcNode.mTable[*childIter].child->mID;
                auto* dstChild = data->child(*childIter);
                extrema.min( dstChild->valueMin() );
                extrema.max( dstChild->valueMax() );
                const auto& bbox = dstChild->bbox();
                data->mBBox[0].minComponent(bbox[0]);
                data->mBBox[1].maxComponent(bbox[1]);
            }
            data->mValueMin = extrema.min();
            data->mValueMax = extrema.max();
            this->setFlag(data->mValueMin, data->mValueMax, data->mFlags);
        }
        mActiveVoxelCount += sum;
    };
    parallel_for(BlockedRange<uint32_t>(0, uint32_t(array.size()), 4), kernel);
} // GridBuilder::processNodes

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridBuilder<ValueT, ExtremaOp>::
processRoot()
{
    using SrcChildT = SrcNode2;
    auto& data = *(this->template nodeData<DstRootT>());
    data.mBackground = mRoot.mBackground;
    data.mTileCount = uint32_t(mRoot.mTable.size());
    // since openvdb::RootNode internally uses a std::map for child nodes its iterator
    // visits elements in the stored order required by the nanovdb::RootNode
    if (data.mTileCount == 0) { // empty root node
        data.mValueMin = data.mValueMax = data.mBackground;
        data.mBBox[0] = Coord::max(); // set to an empty bounding box
        data.mBBox[1] = Coord::min();
        data.mActiveVoxelCount = 0;
    } else {
        ExtremaOp extrema;// invalid
        uint32_t tileID = 0;
        for (auto iter = mRoot.mTable.begin(); iter != mRoot.mTable.end(); ++iter, ++tileID) {
            auto& dstTile = data.tile(tileID);
            if (auto *srcChild = iter->second.child) {
                dstTile.setChild(srcChild->mOrigin, srcChild->mID);
                auto& dstChild = data.child(dstTile);
                if (!extrema) {
                    extrema = ExtremaOp( dstChild.valueMin(), dstChild.valueMax() );
                    assert(extrema);
                    data.mBBox = dstChild.bbox();
                } else {
                    extrema.min( dstChild.valueMin() );
                    extrema.max( dstChild.valueMax() );
                    data.mBBox[0].minComponent(dstChild.bbox()[0]);
                    data.mBBox[1].maxComponent(dstChild.bbox()[1]);
                }
            } else {
                dstTile.setValue(iter->first, iter->second.state, iter->second.value);
                if (iter->second.state) {// active tile
                    mActiveVoxelCount += SrcChildT::NUM_VALUES;
                    if (!extrema) {
                        extrema = ExtremaOp(iter->second.value);
                        assert(extrema);
                        data.mBBox[0] = iter->first;
                        data.mBBox[1] = iter->first + Coord(SrcChildT::DIM - 1);
                    } else {
                        extrema( dstTile.value );
                        data.mBBox[0].minComponent(iter->first);
                        data.mBBox[1].maxComponent(iter->first + Coord(SrcChildT::DIM - 1)); 
                    }
                }
            }
        }
        data.mValueMin = extrema.min();
        data.mValueMax = extrema.max();
        data.mActiveVoxelCount = mActiveVoxelCount;
        if (!extrema) std::cerr << "\nWarning: input tree only contained inactive root tiles! While not strictly an error it's suspecious." << std::endl;
    }
}// GridBuilder::processRoot

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridBuilder<ValueT, ExtremaOp>::
processTree()
{
    const uint64_t count[4] = {mArray0.size(), mArray1.size(), mArray2.size(), 1};
    auto& data = *this->treeData(); // data for the tree
    for (int i = 0; i < 4; ++i) {
        if (count[i] > std::numeric_limits<uint32_t>::max()) throw std::runtime_error("Node count exceeds 32 bit range");
        data.mCount[i] = static_cast<uint32_t>(count[i]);
        data.mBytes[i] = mBytes[4 - i] - mBytes[0]; // offset from the tree to the first node at each tree level
    }
}// GridBuilder::processTree

//================================================================================================

template<typename ValueT, typename ExtremaOp>
void GridBuilder<ValueT, ExtremaOp>::
processGrid(const Map &map, 
            const std::string &name, 
            GridClass gridClass)
{
    auto& data = *this->gridData();
    data.mMagic = NANOVDB_MAGIC_NUMBER;
    data.mBlindDataCount = mBlindDataSize>0 ? 1u : 0u;
    data.mGridClass = gridClass;
    if (std::is_same<ValueT, float>::value) { // resolved at compiletime
        data.mGridType = GridType::Float;
    } else if (std::is_same<ValueT, double>::value) {
        data.mGridType = GridType::Double;
    } else if (std::is_same<ValueT, int16_t>::value) {
        data.mGridType = GridType::Int16;
    } else if (std::is_same<ValueT, int32_t>::value) {
        data.mGridType = GridType::Int32;
    } else if (std::is_same<ValueT, int64_t>::value) {
        data.mGridType = GridType::Int64;
    } else if (std::is_same<ValueT, Vec3f>::value) {
        data.mGridType = GridType::Vec3f;
    } else if (std::is_same<ValueT, uint32_t>::value) {
        data.mGridType = GridType::UInt32;
    } else {
        throw std::runtime_error("Unsupported value type");
    }
    { // set grid name
        if (name.length() + 1 > GridData::MaxNameSize) {
            std::stringstream ss;
            ss << "Grid name \"" << name << "\" is more then " << nanovdb::GridData::MaxNameSize << " characters";
            throw std::runtime_error(ss.str());
        }
        memcpy(data.mGridName, name.c_str(), name.size() + 1);
    }
    data.mUniformScale = (map.applyMap(Vec3d(1,0,0))-map.applyMap(Vec3d(0))).length();
    data.mMap = map;
    { // set world space AABB
        const auto& indexBBox = this->template nodeData<DstRootT>()->mBBox;
        auto &worldBBox = data.mWorldBBox;
        worldBBox[0] = worldBBox[1] = map.applyMap(Vec3d(indexBBox[0][0], indexBBox[0][1], indexBBox[0][2]));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[0][1], indexBBox[1][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[1][1], indexBBox[0][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[0][1], indexBBox[0][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[1][1], indexBBox[0][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[0][1], indexBBox[1][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[0][0], indexBBox[1][1], indexBBox[1][2])));
        worldBBox.expand(map.applyMap(Vec3d(indexBBox[1][0], indexBBox[1][1], indexBBox[1][2])));
    }
}// GridBuilder::processGrid

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename ChildT>
struct GridBuilder<ValueT, ExtremaOp>::Root
{
    using ValueType = typename ChildT::ValueType;
    using ChildType = ChildT;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    struct Tile {
        Tile(ChildT* c = nullptr) : child(c) {}
        Tile(const ValueT& v, bool s) : child(nullptr), value(v), state(s) {}
        ChildT* child;
        ValueT  value;
        bool    state;
    };
    using MapT = std::map<Coord, Tile>;
    MapT   mTable;
    ValueT mBackground;

    Root(const ValueT& background) : mBackground(background) {}
    Root(const Root&) = delete;// disallow copy-construction
    Root(Root&&) = default;// allow move construction
    Root& operator=(const Root&) = delete;// disallow copy assignment
    Root& operator=(Root&&) = default;// allow move assignment

    ~Root() { this->clear(); }

    bool empty() const {return mTable.empty();}

    void clear() 
    {
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) delete iter->second.child;
        mTable.clear();   
    }

    static Coord CoordToKey(const Coord& ijk) { return ijk & ~ChildT::MASK; }

    template <typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end()) return false;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->isActiveAndCache(ijk, acc);
        }
        return iter->second.state;
    }

    const ValueT& getValue(const Coord& ijk) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end()) {
            return mBackground;
        } else if (iter->second.child) {
            return iter->second.child->getValue(ijk);
        } else {
            return iter->second.value;
        }
    }

    template <typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end()) return mBackground;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->getValueAndCache(ijk, acc);
        }
        return iter->second.value;
    }

    template <typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, AccT& acc)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        if (child) {
            acc.insert(ijk, child);
            child->setValueAndCache(ijk, value, acc);
        }
    }

    template <typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        uint32_t sum = 0;
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr) continue;// skip tiles
            if (is_same<NodeT,ChildT>::value) {//resolved at compile-time
                ++sum;
            } else {
                sum += iter->second.child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template <typename NodeT>
    void getNodes(std::vector<NodeT*> &array)
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr) continue;
            if (is_same<NodeT,ChildT>::value) {//resolved at compile-time
                iter->second.child->mID = static_cast<uint32_t>(array.size());
                array.push_back(reinterpret_cast<NodeT*>(iter->second.child));
            } else {
                iter->second.child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        assert(child);
        const Coord key = CoordToKey(child->mOrigin);
        auto iter = mTable.find(key);
        if (iter != mTable.end() && iter->second.child != nullptr) {// existing child node
            delete iter->second.child;
            iter->second.child = child;
        } else {
            mTable[key] = Tile(child);
        }
        child = nullptr;
    }

    template <typename NodeT>
    void addNode(NodeT*& node)
    {
        if (is_same<NodeT, ChildT>::value) {//resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else {
            ChildT* child = nullptr;
            const Coord key = CoordToKey(node->mOrigin);
            auto iter = mTable.find(key);
            if (iter == mTable.end()) {
                child = new ChildT(node->mOrigin, mBackground, false);
                mTable[key] = Tile(child);
            } else if (iter->second.child != nullptr) {
                child = iter->second.child;
            } else {
                child = new ChildT(node->mOrigin, iter->second.value, iter->second.state);
                iter->second.child = child;
            }
            child->addNode(node);
        }
    }

    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template <typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    signedFloodFill(T) {}// no-op for none floating point values
};// GridBuilder::Root

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename ChildT>
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, ExtremaOp>::Root<ChildT>::
signedFloodFill(T outside)
{
    std::map<Coord, ChildT*> nodeKeys;
    for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
        if (iter->second.child == nullptr) continue;
        nodeKeys.insert(std::pair<Coord, ChildT*>(iter->first, iter->second.child));
    }

    // We employ a simple z-scanline algorithm that inserts inactive tiles with
    // the inside value if they are sandwiched between inside child nodes only!
    auto b = nodeKeys.begin(), e = nodeKeys.end();
    if ( b == e ) return;
    //const ValueT inside = -mRoot.mBackground;
    for (auto a = b++; b != e; ++a, ++b) {
        Coord d = b->first - a->first; // delta of neighboring coordinates
        if (d[0]!=0 || d[1]!=0 || d[2]==int(ChildT::DIM)) continue;// not same z-scanline or neighbors
        const ValueT fill[] = { a->second->getLastValue(), b->second->getFirstValue() };
        if (!(fill[0] < 0) || !(fill[1] < 0)) continue; // scanline isn't inside
        Coord c = a->first + Coord(0u, 0u, ChildT::DIM);
        for (; c[2] != b->first[2]; c[2] += ChildT::DIM) {
            const Coord key = SrcRootT::CoordToKey(c);
            mTable[key] = typename SrcRootT::Tile(-outside, false);// inactive tile
        }
    }
}// Root::signedFloodFill

//================================================================================================

template <typename ValueT, typename ExtremaOp>
template <typename ChildT>
struct GridBuilder<ValueT, ExtremaOp>::
Node
{
    using ValueType = typename ChildT::ValueType;
    using ChildType = ChildT;
    static constexpr uint32_t LOG2DIM = ChildT::LOG2DIM + 1;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
    static constexpr uint32_t MASK = DIM - 1u;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL);// total voxel count represented by this node
    using MaskT = Mask<LOG2DIM>;

    struct Tile {
        Tile(ChildT* c = nullptr) : child(c) {}
        union { ChildT* child; ValueT value; };
    };
    Coord mOrigin;
    MaskT mValueMask;
    MaskT mChildMask;
    Tile  mTable[SIZE];
    uint32_t mID;

    Node(const Coord& origin, const ValueT& value, bool state)
        : mOrigin(origin & ~MASK), mValueMask(state), mChildMask()
    {
        for (uint32_t i = 0; i < SIZE; ++i) mTable[i].value = value;
    }
    Node(const Node&) = delete;// disallow copy-construction
    Node(Node&&) = delete;// disallow move construction
    Node& operator=(const Node&) = delete;// disallow copy assignment
    Node& operator=(Node&&) = delete;// disallow move assignment
    ~Node() {for (auto iter = mChildMask.beginOn(); iter; ++iter) delete mTable[*iter].child;}

    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
               (( ijk[2] & MASK) >> ChildT::TOTAL);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        assert(n < SIZE);
        const uint32_t m = n & ((1<<2*LOG2DIM)-1);
        return Coord(n >> 2*LOG2DIM, m >> LOG2DIM, m & ((1<<LOG2DIM)-1));
    }

    void localToGlobalCoord(Coord &ijk) const
    {
        ijk <<= ChildT::TOTAL;
        ijk  += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = Node::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk); 
        return ijk;
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->isActiveAndCache(ijk, acc);
        }
        return mValueMask.isOn(n);
    }

    ValueT getFirstValue() const {return mChildMask.isOn(0) ? mTable[0].child->getFirstValue() : mTable[0].value;}
    ValueT getLastValue() const {return mChildMask.isOn(SIZE-1) ? mTable[SIZE-1].child->getLastValue() : mTable[SIZE-1].value;}

    const ValueT& getValue(const Coord& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            return mTable[n].child->getValue(ijk);
        }
        return mTable[n].value;
    }

    template<typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->getValueAndCache(ijk, acc);
        }
        return mTable[n].value;
    }

    void setValue(const Coord& ijk, const ValueT& value)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT* child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        child->setValue(ijk, value);
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, AccT& acc)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT* child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        acc.insert(ijk, child);
        child->setValueAndCache(ijk, value, acc);
    }

    template <typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        assert(NodeT::LEVEL < LEVEL);
        uint32_t sum = 0;
        if (is_same<NodeT,ChildT>::value) {//resolved at compile-time
            sum += mChildMask.countOn();
        } else {
            for (auto iter = mChildMask.beginOn(); iter; ++iter) {
                sum += mTable[*iter].child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template <typename NodeT>
    void getNodes(std::vector<NodeT*> &array)
    {
        static_assert(is_same<ValueT, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        assert(NodeT::LEVEL < LEVEL);
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            if (is_same<NodeT,ChildT>::value) {//resolved at compile-time
                mTable[*iter].child->mID = static_cast<uint32_t>(array.size());
                array.push_back(reinterpret_cast<NodeT*>(mTable[*iter].child));
            } else {
                mTable[*iter].child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        assert(child && (child->mOrigin & ~MASK) == this->mOrigin);
        const uint32_t n = CoordToOffset(child->mOrigin);
        if (mChildMask.isOn(n)) {
            delete mTable[n].child;
        } else {
            mChildMask.setOn(n);
        }
        mTable[n].child = child;
        child = nullptr;
    }

    template <typename NodeT>
    void addNode(NodeT*& node)
    {
        if (is_same<NodeT, ChildT>::value) {//resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else {
            const uint32_t n = CoordToOffset(node->mOrigin);
            ChildT* child = nullptr;
            if (mChildMask.isOn(n)) {
                child = mTable[n].child;
            } else {
                child = new ChildT(node->mOrigin, mTable[n].value, mValueMask.isOn(n));
                mTable[n].child = child;
                mChildMask.setOn(n);
            }
            child->addNode(node);
        }
    }

    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template <typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    signedFloodFill(T) {}// no-op for none floating point values
};// GridBuilder::Node

//================================================================================================

template<typename ValueT, typename ExtremaOp>
template<typename ChildT>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, ExtremaOp>::Node<ChildT>::
signedFloodFill(T outside)
{
    const uint32_t first = *mChildMask.beginOn();
    if (first < NUM_VALUES) {
        bool xInside = mTable[first].child->getFirstValue()<0;
        bool yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != (1 << LOG2DIM); ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM); // offset for block(x, 0, 0)
            if (mChildMask.isOn(x00)) xInside = mTable[x00].child->getLastValue()<0;
            yInside = xInside;
            for (uint32_t y = 0; y != (1u << LOG2DIM); ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM); // offset for block(x, y, 0)
                if (mChildMask.isOn(xy0)) yInside = mTable[xy0].child->getLastValue()<0;
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // offset for block(x, y, z)
                    if (mChildMask.isOn(xyz)) {
                        zInside = mTable[xyz].child->getLastValue()<0;
                    } else {
                        mTable[xyz].value = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
}// Node::signedFloodFill

//================================================================================================

template <typename ValueT, typename ExtremaOp>
struct GridBuilder<ValueT, ExtremaOp>::
Leaf
{
    using ValueType = ValueT;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = DIM - 1u; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL);// total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    Coord mOrigin;
    Mask<LOG2DIM>  mValueMask;
    ValueT mValues[SIZE];
    uint32_t mID;

    Leaf(const Coord& ijk, const ValueT& value, bool state) : mOrigin(ijk & ~MASK), mValueMask(state)//invalid
    {
      ValueT* target = mValues;
      uint32_t n = SIZE;
      while (n--) *target++ = value;
    }
    Leaf(const Leaf&) = delete;// disallow copy-construction
    Leaf(Leaf&&) = delete;// disallow move construction
    Leaf& operator=(const Leaf&) = delete;// disallow copy assignment
    Leaf& operator=(Leaf&&) = delete;// disallow move assignment
    ~Leaf() = default;

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & MASK) << (2 * LOG2DIM)) + ((ijk[1] & MASK) << LOG2DIM) + (ijk[2] & MASK);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        assert(n < SIZE);
        const uint32_t m = n & ((1<<2*LOG2DIM)-1);
        return Coord(n >> 2*LOG2DIM, m >> LOG2DIM, m & MASK);
    }

    void localToGlobalCoord(Coord &ijk) const
    {
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = Leaf::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, const AccT&) const
    {
      return mValueMask.isOn(CoordToOffset(ijk));
    }

    ValueT getFirstValue() const {return mValues[0];}
    ValueT getLastValue() const {return mValues[SIZE-1];}

    const ValueT& getValue(const Coord& ijk) const
    {
        return mValues[CoordToOffset(ijk)];
    }

    template<typename AccT>
    const ValueT& getValueAndCache(const Coord& ijk, const AccT&) const
    {
        return mValues[CoordToOffset(ijk)];
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueT& value, const AccT&)
    {   
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues[n] = value;
    }

    void setValue(const Coord& ijk, const ValueT& value)
    {   
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues[n] = value;
    }

    template <typename NodeT>
    void getNodes(std::vector<NodeT*>&) { assert(false); }

    template <typename NodeT>
    void addNode(NodeT*&) {}

    template <typename NodeT>
    uint32_t nodeCount() const { assert(false); return 1;}

    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);
    template <typename T>
    typename std::enable_if<!std::is_floating_point<T>::value>::type
    signedFloodFill(T) {}// no-op for none floating point values
};// Leaf

//================================================================================================

template<typename ValueT, typename ExtremaOp>
template<typename T>
inline typename std::enable_if<std::is_floating_point<T>::value>::type
GridBuilder<ValueT, ExtremaOp>::Leaf::
signedFloodFill(T outside)
{
    const uint32_t first = *mValueMask.beginOn();
    if (first < SIZE) {
        bool xInside = mValues[first]<0, yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != DIM; ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM);
            if (mValueMask.isOn(x00)) xInside = mValues[x00] < 0; // element(x, 0, 0)
            yInside = xInside;
            for (uint32_t y = 0; y != DIM; ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM);
                if (mValueMask.isOn(xy0)) yInside = mValues[xy0] < 0; // element(x, y, 0)
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // element(x, y, z)
                    if (mValueMask.isOn(xyz)) {
                        zInside = mValues[xyz] < 0;
                    } else {
                        mValues[xyz] = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
}// Leaf::signedFloodFill

//================================================================================================
template <typename ValueT, typename ExtremaOp>
struct GridBuilder<ValueT, ExtremaOp>::
ValueAccessor
{
    ValueAccessor(SrcRootT& root) : mKeys{Coord(Maximum<int>::value()),Coord(Maximum<int>::value()), Coord(Maximum<int>::value())}
                                  , mNode{nullptr, nullptr, nullptr, &root}
    {
    }
    template<typename NodeT>
    bool isCached(const Coord& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][0] && 
               (ijk[1] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][1] && 
               (ijk[2] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][2];
    }
    bool isActive(const Coord& ijk)
    {
      if (this->isCached<SrcNode0>(ijk)) {
          return ((SrcNode0*)mNode[0])->isActiveAndCache(ijk, *this);
      } else if (this->isCached<SrcNode1>(ijk)) {
          return ((SrcNode1*)mNode[1])->isActiveAndCache(ijk, *this);
      } else if (this->isCached<SrcNode2>(ijk)) {
          return ((SrcNode2*)mNode[2])->isActiveAndCache(ijk, *this);
      }
      return ((SrcRootT*)mNode[3])->isActiveAndCache(ijk, *this);
    }
    const ValueT& getValue(const Coord& ijk)
    {
      if (this->isCached<SrcNode0>(ijk)) {
          return ((SrcNode0*)mNode[0])->getValueAndCache(ijk, *this);
      } else if (this->isCached<SrcNode1>(ijk)) {
          return ((SrcNode1*)mNode[1])->getValueAndCache(ijk, *this);
      } else if (this->isCached<SrcNode2>(ijk)) {
          return ((SrcNode2*)mNode[2])->getValueAndCache(ijk, *this);
      }
      return ((SrcRootT*)mNode[3])->getValueAndCache(ijk, *this);
    }
    /// @brief Sets value in a leaf node and returns it.
    SrcNode0* setValue(const Coord& ijk, const ValueT& value)
    {
        if (this->isCached<SrcNode0>(ijk)) {
           ((SrcNode0*)mNode[0])->setValueAndCache(ijk, value, *this);
        } else if (this->isCached<SrcNode1>(ijk)) {
           ((SrcNode1*)mNode[1])->setValueAndCache(ijk, value, *this);
        } else if (this->isCached<SrcNode2>(ijk)) {
           ((SrcNode2*)mNode[2])->setValueAndCache(ijk, value, *this);
        } else {
           ((SrcRootT*)mNode[3])->setValueAndCache(ijk, value, *this);
        }
        assert(this->isCached<SrcNode0>(ijk));
        return (SrcNode0*)mNode[0];
    }
    template <typename NodeT>
    void insert(const Coord& ijk, NodeT* node)
    {
        mKeys[NodeT::LEVEL] = ijk & int(~NodeT::MASK);
        mNode[NodeT::LEVEL] = node;
    }
    Coord mKeys[3];
    void* mNode[4];
}; // ValueAccessor

//================================================================================================

namespace {

/// @brief Returns a shared pointer to a GridBuilder with narrow-band SDF values for a sphere
///
/// @brief Note, this is not (yet) a valid level set SDF field since values inside sphere (and outside
///        the narrow band) are still undefined. Call GridBuilder::sdfToLevelSet() to set those
///        values or alternatively call GridBuilder::sdfToFog to generate a FOG volume.
template <typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initSphere(ValueT radius,// radius of sphere in world units 
           const Vec3d& center,//center of sphere in world units
           ValueT voxelSize,// size of a voxel in world units
           ValueT halfWidth,// half-width of narrow band in voxel units 
           const Vec3d &origin)// origin of grid in world units
{
    static_assert(is_floating_point<ValueT>::value,"Sphere: expect floating point");
    if (!(radius > 0)) throw std::runtime_error("Sphere: radius must be positive!");
    if (!(voxelSize > 0)) throw std::runtime_error("Sphere: voxelSize must be positive!");
    if (!(halfWidth > 0)) throw std::runtime_error("Sphere: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth*voxelSize);
    auto acc = builder->getAccessor();

    // Define radius of sphere and narrow-band in voxel units
    const ValueT r0 = radius/voxelSize, rmax = r0 + halfWidth;

    // Radius below the Nyquist frequency
    if (r0 < ValueT(1.5)) return builder;

    // Define center of sphere in voxel units
    const Vec3<ValueT> c(ValueT(center[0]-origin[0])/voxelSize,
                         ValueT(center[1]-origin[1])/voxelSize,
                         ValueT(center[2]-origin[2])/voxelSize);

    // Define bounds of the voxel coordinates
    const int imin = Floor(c[0]-rmax), imax = Ceil(c[0]+rmax);
    const int jmin = Floor(c[1]-rmax), jmax = Ceil(c[1]+rmax);
    const int kmin = Floor(c[2]-rmax), kmax = Ceil(c[2]+rmax);

    Coord ijk;
    int &i = ijk[0], &j = ijk[1], &k = ijk[2], m=1;
    // Compute signed distances to sphere using leapfrogging in k
    for (i = imin; i <= imax; ++i) {
        const auto x2 = Pow2(ValueT(i) - c[0]);
        for (j = jmin; j <= jmax; ++j) {
            const auto x2y2 = Pow2(ValueT(j) - c[1]) + x2;
            for (k = kmin; k <= kmax; k += m) {
                m = 1;
                const auto v = Sqrt(x2y2 + Pow2(ValueT(k)-c[2]))-r0;// Distance in voxel units
                const auto d = v<0 ? -v : v;
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(ijk, voxelSize*v);// distance in world units
                } else { // outside narrow band
                    m += Floor(d-halfWidth);// leapfrog
                }
            }//end leapfrog over k
        }//end loop over j
    }//end loop over i

    return builder;
}// initSphere

template <typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initTorus(ValueT radius1,// major radius of torus in world units 
          ValueT radius2,// minor radius of torus in world units 
          const Vec3d& center,//center of sphere in world units
          ValueT voxelSize,// size of a voxel in world units
          ValueT halfWidth,// half-width of narrow band in voxel units 
          const Vec3d &origin)// origin of grid in world units
{
    static_assert(is_floating_point<ValueT>::value,"Torus: expect floating point");
    if (!(radius2 > 0)) throw std::runtime_error("Torus: radius2 must be positive!");
    if (!(radius1 > radius2)) throw std::runtime_error("Torus: radius1 must be larger than radius2!");
    if (!(voxelSize > 0)) throw std::runtime_error("Torus: voxelSize must be positive!");
    if (!(halfWidth > 0)) throw std::runtime_error("Torus: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth*voxelSize);
    auto acc = builder->getAccessor();

    // Define radius of sphere and narrow-band in voxel units
    const ValueT r1 = radius1/voxelSize, r2 = radius2/voxelSize, rmax1 = r1 + r2 + halfWidth, rmax2 = r2 + halfWidth;

    // Radius below the Nyquist frequency
    if (r2 < ValueT(1.5)) return builder;

    // Define center of sphere in voxel units
    const Vec3<ValueT> c(ValueT(center[0]-origin[0])/voxelSize,
                         ValueT(center[1]-origin[1])/voxelSize,
                         ValueT(center[2]-origin[2])/voxelSize);

    // Define bounds of the voxel coordinates
    const int imin = Floor(c[0]-rmax1), imax = Ceil(c[0]+rmax1);
    const int jmin = Floor(c[1]-rmax2), jmax = Ceil(c[1]+rmax2);
    const int kmin = Floor(c[2]-rmax1), kmax = Ceil(c[2]+rmax1);

    Coord ijk;
    int &i = ijk[0], &j = ijk[1], &k = ijk[2], m=1;
    // Compute signed distances to sphere using leapfrogging in k
    for (i = imin; i <= imax; ++i) {
        const auto x2 = Pow2(ValueT(i) - c[0]);
        for (k = kmin; k <= kmax; ++k) {
            const auto x2z2 = Pow2(Sqrt(Pow2(ValueT(k) - c[2]) + x2) - r1);
            for (j = jmin; j <= jmax; j += m) {
                m = 1;
                const auto v = Sqrt(x2z2 + Pow2(ValueT(j)-c[1])) - r2; // Distance in voxel units
                const auto d = v<0 ? -v : v;
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(ijk, voxelSize*v);// distance in world units
                } else { // outside narrow band
                    m += Floor(d-halfWidth);// leapfrog
                }
            }//end leapfrog over k
        }//end loop over j
    }//end loop over i

    return builder;
}// initTorus

template <typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initBox(ValueT width,// major radius of torus in world units 
        ValueT height,// minor radius of torus in world units
        ValueT depth, 
        const Vec3d& center,//center of sphere in world units
        ValueT voxelSize,// size of a voxel in world units
        ValueT halfWidth,// half-width of narrow band in voxel units 
        const Vec3d &origin)// origin of grid in world units
{
    using Vec3T = Vec3<ValueT>;
    static_assert(is_floating_point<ValueT>::value,"Box: expect floating point");
    if (!(width > 0))  throw std::runtime_error("Box: width must be positive!");
    if (!(height > 0)) throw std::runtime_error("Box: height must be positive!");
    if (!(depth > 0))  throw std::runtime_error("Box: depth must be positive!");

    if (!(voxelSize > 0)) throw std::runtime_error("Box: voxelSize must be positive!");
    if (!(halfWidth > 0)) throw std::runtime_error("Box: halfWidth must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth*voxelSize);
    auto acc = builder->getAccessor();

    // Define radius of sphere and narrow-band in voxel units
    const Vec3T r(width/(2*voxelSize), height/(2*voxelSize), depth/(2*voxelSize));
                      
    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5)) return builder;

    // Define center of sphere in voxel units
    const Vec3T c(ValueT(center[0]-origin[0])/voxelSize,
                  ValueT(center[1]-origin[1])/voxelSize,
                  ValueT(center[2]-origin[2])/voxelSize);

    // Define utinity functions
    auto Pos = [](ValueT x){return x>0 ?  x : 0;};
    auto Neg = [](ValueT x){return x<0 ?  x : 0;};

    // Define bounds of the voxel coordinates
    const BBox<Vec3T> b(c - r - Vec3T(halfWidth), c + r + Vec3T(halfWidth));
    const CoordBBox bbox(Coord(Floor(b[0][0]), Floor(b[0][1]),Floor(b[0][2])), 
                         Coord( Ceil(b[1][0]),  Ceil(b[1][1]), Ceil(b[1][2])));

    // Compute signed distances to sphere using leapfrogging in k
    int m = 1;
    for (Coord p = bbox[0]; p[0]<=bbox[1][0]; ++p[0]) {
        const auto q1 = Abs(ValueT(p[0]) - c[0]) - r[0];
        const auto x2 = Pow2(Pos(q1));
        for (p[1] = bbox[0][1]; p[1]<=bbox[1][1]; ++p[1]) {
            const auto q2 = Abs(ValueT(p[1]) - c[1]) - r[1];
            const auto q0 = Max(q1, q2);
            const auto x2y2 = x2 + Pow2(Pos(q2));
            for (p[2] = bbox[0][2]; p[2]<=bbox[1][2]; p[2] += m) {
                m = 1;
                const auto q3 = Abs(ValueT(p[2]) - c[2]) - r[2];
                const auto v = Sqrt(x2y2 + Pow2(Pos(q3))) + Neg(Max(q0, q3));// Distance in voxel units
                const auto d = Abs(v);
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(p, voxelSize*v);// distance in world units
                } else { // outside narrow band
                    m += Floor(d-halfWidth);// leapfrog
                }
            }//end leapfrog over k
        }//end loop over j
    }//end loop over i

    return builder;
}// initBox

template <typename ValueT>
std::shared_ptr<GridBuilder<ValueT>>
initBBox(ValueT width,// width of the box in world units 
         ValueT height,// height of the box in world units
         ValueT depth, // depth of th ebox in world units
         ValueT thickness,// thickness of the wire in world units
         const Vec3d& center,//center of sphere in world units
         ValueT voxelSize,// size of a voxel in world units
         ValueT halfWidth,// half-width of narrow band in voxel units 
         const Vec3d &origin)// origin of grid in world units
{
    using Vec3T = Vec3<ValueT>;
    static_assert(is_floating_point<ValueT>::value,"BBox: expect floating point");
    if (!(width > 0))  throw std::runtime_error("BBox: width must be positive!");
    if (!(height > 0)) throw std::runtime_error("BBox: height must be positive!");
    if (!(depth > 0))  throw std::runtime_error("BBox: depth must be positive!");
    if (!(thickness > 0)) throw std::runtime_error("BBox: thickness must be positive!");
    if (!(voxelSize > 0)) throw std::runtime_error("BBox: voxelSize must be positive!");

    auto builder = std::make_shared<GridBuilder<ValueT>>(halfWidth*voxelSize);
    auto acc = builder->getAccessor();

    // Define radius of sphere and narrow-band in voxel units
    const Vec3T r(width/(2*voxelSize), height/(2*voxelSize), depth/(2*voxelSize));
    const ValueT e = thickness/voxelSize;
                      
    // Below the Nyquist frequency
    if (r.min() < ValueT(1.5) || e < ValueT(1.5) ) return builder;

    // Define center of sphere in voxel units
    const Vec3T c(ValueT(center[0]-origin[0])/voxelSize,
                  ValueT(center[1]-origin[1])/voxelSize,
                  ValueT(center[2]-origin[2])/voxelSize);

    // Define utinity functions
    auto Pos = [](ValueT x){return x>0 ?  x : 0;};
    auto Neg = [](ValueT x){return x<0 ?  x : 0;};

    // Define bounds of the voxel coordinates
    const BBox<Vec3T> b(c - r - Vec3T(e + halfWidth), c + r + Vec3T(e + halfWidth));
    const CoordBBox bbox(Coord(Floor(b[0][0]), Floor(b[0][1]),Floor(b[0][2])), 
                         Coord( Ceil(b[1][0]),  Ceil(b[1][1]), Ceil(b[1][2])));

    // Compute signed distances to sphere using leapfrogging in k
    int m = 1;
    for (Coord p = bbox[0]; p[0]<=bbox[1][0]; ++p[0]) {
        const ValueT px = Abs(ValueT(p[0]) - c[0]) - r[0];
        const ValueT qx = Abs(ValueT(px) + e) - e;
        const ValueT px2 = Pow2(Pos(px));
        const ValueT qx2 = Pow2(Pos(qx));
        for (p[1] = bbox[0][1]; p[1]<=bbox[1][1]; ++p[1]) {
            const ValueT py = Abs(ValueT(p[1]) - c[1]) - r[1];
            const ValueT qy = Abs(ValueT(py) + e) - e;
            const ValueT qy2 = Pow2(Pos(qy));;
            const ValueT px2qy2 = px2 + qy2;
            const ValueT qx2py2 = qx2 + Pow2(Pos(py));
            const ValueT qx2qy2 = qx2 + qy2;
            const ValueT a[3] = {Max(px, qy), Max(qx, py), Max(qx, qy)};
            for (p[2] = bbox[0][2]; p[2]<=bbox[1][2]; p[2] += m) {
                m = 1;
                const ValueT pz = Abs(ValueT(p[2]) - c[2]) - r[2];
                const ValueT qz = Abs(ValueT(pz) + e) - e;
                const ValueT qz2 = Pow2(Pos(qz));
                const ValueT s1 = Sqrt(px2qy2 + qz2) + Neg(Max(a[0], qz));
                const ValueT s2 = Sqrt(qx2py2 + qz2) + Neg(Max(a[1], qz));
                const ValueT s3 = Sqrt(qx2qy2 + Pow2(Pos(pz))) + Neg(Max(a[2], pz));
                const ValueT v = Min(s1, Min(s2, s3));// Distance in voxel units
                const ValueT d = Abs(v);
                if (d < halfWidth) { // inside narrow band
                    acc.setValue(p, voxelSize*v);// distance in world units
                } else { // outside narrow band
                    m += Floor(d-halfWidth);// leapfrog
                }
            }//end leapfrog over k
        }//end loop over j
    }//end loop over i

    return builder;
}// initBBox

}// unnamed namespace

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetSphere(ValueT radius,// radius of sphere in world units 
                     const Vec3d& center,//center of sphere in world units
                     ValueT voxelSize,// size of a voxel in world units
                     ValueT halfWidth,// half-width of narrow band in voxel units 
                     const Vec3d &origin,// origin of grid in world units
                     const std::string &name,// name of grid
                     const BufferT& buffer)
{
    auto builder = initSphere(radius, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, buffer);
    assert(handle);
    return handle;
}// createLevelSetSphere

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeSphere(ValueT             radius, // radius of sphere in world units
                      const Vec3d&       center, //center of sphere in world units
                      ValueT             voxelSize, // size of a voxel in world units
                      ValueT             halfWidth,// half-width of narrow band in voxel units 
                      const Vec3d&       origin, // origin of grid in world units
                      const std::string& name,// name of grid
                      const BufferT& buffer)
{
    auto builder = initSphere(radius, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, buffer);
    assert(handle);
    return handle;
} // createFogVolumeSphere

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointSphere(int pointsPerVoxel,// half-width of narrow band in voxel units
                  ValueT radius,// radius of sphere in world units 
                  const Vec3d& center,//center of sphere in world units
                  ValueT voxelSize,// size of a voxel in world units 
                  const Vec3d &origin,// origin of grid in world units
                  const std::string &name,// name of grid
                  const BufferT& buffer)
{
    auto sphereHandle = createLevelSetSphere(radius, center, voxelSize, 0.5f, origin, "dummy", buffer);
    assert(sphereHandle);
    auto *sphereGrid = sphereHandle.template grid<ValueT>();
    assert(sphereGrid);
    auto pointHandle = createPointScatter(*sphereGrid, pointsPerVoxel, name, buffer);
    assert(pointHandle);
    return pointHandle;
}// createPointSphere

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetTorus(ValueT majorRadius,// major radius of torus in world units
                    ValueT minorRadius,// minor radius of torus in world units 
                    const Vec3d& center,//center of sphere in world units
                    ValueT voxelSize,// size of a voxel in world units
                    ValueT halfWidth,// half-width of narrow band in voxel units 
                    const Vec3d &origin,// origin of grid in world units
                    const std::string& name,// name of grid
                    const BufferT& buffer)
{
    auto builder = initTorus(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, buffer);
    assert(handle);
    return handle;
}// createLevelSetTorus

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeTorus(ValueT majorRadius,// major radius of torus in world units
                     ValueT minorRadius,// minor radius of torus in world units
                     const Vec3d& center, //center of sphere in world units
                     ValueT voxelSize, // size of a voxel in world units
                     ValueT halfWidth,// half-width of narrow band in voxel units 
                     const Vec3d& origin, // origin of grid in world units
                     const std::string& name,// name of grid
                     const BufferT& buffer)
{
    auto builder = initTorus(majorRadius, minorRadius, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, buffer);
    assert(handle);
    return handle;
} // createFogVolumeTorus

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointTorus(int pointsPerVoxel,// half-width of narrow band in voxel units
                 ValueT majorRadius,// major radius of torus in world units
                 ValueT minorRadius,// minor radius of torus in world units
                 const Vec3d& center, //center of sphere in world units
                 ValueT voxelSize, // size of a voxel in world units 
                 const Vec3d& origin, // origin of grid in world units
                 const std::string& name,// name of grid
                 const BufferT& buffer)
{
    auto torusHandle = createLevelSetTorus(majorRadius, minorRadius, center, voxelSize, 0.5f, origin, "dummy", buffer);
    assert(torusHandle);
    auto *torusGrid = torusHandle.template grid<ValueT>();
    assert(torusGrid);
    auto pointHandle = createPointScatter(*torusGrid, pointsPerVoxel, name, buffer);
    assert(pointHandle);
    return pointHandle;
}// createPointTorus

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetBox(ValueT width,// width of box in world units
                  ValueT height,// height of box in world units
                  ValueT depth,// depth of box in world units
                  const Vec3d& center,//center of sphere in world units
                  ValueT voxelSize,// size of a voxel in world units
                  ValueT halfWidth,// half-width of narrow band in voxel units 
                  const Vec3d &origin,// origin of grid in world units
                  const std::string& name,// name of grid
                  const BufferT& buffer)
{
    auto builder = initBox(width, height, depth, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, buffer);
    assert(handle);
    return handle;
}// createLevelSetBox

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createLevelSetBBox(ValueT width,// width of box in world units
                   ValueT height,// height of box in world units
                   ValueT depth,// depth of box in world units
                   ValueT thickness,// thickness of the wire in world units
                   const Vec3d& center,//center of sphere in world units
                   ValueT voxelSize,// size of a voxel in world units
                   ValueT halfWidth,// half-width of narrow band in voxel units 
                   const Vec3d &origin,// origin of grid in world units
                   const std::string& name,// name of grid
                   const BufferT& buffer)
{
    auto builder = initBBox(width, height, depth, thickness, center, voxelSize, halfWidth, origin);
    builder->sdfToLevelSet();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::LevelSet, buffer);
    assert(handle);
    return handle;
}// createLevelSetBBox

//================================================================================================

template<typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createFogVolumeBox(ValueT width,// width of box in world units
                   ValueT height,// height of box in world units
                   ValueT depth,// depth of box in world units
                   const Vec3d&       center, //center of sphere in world units
                   ValueT             voxelSize, // size of a voxel in world units
                   ValueT             halfWidth,// half-width of narrow band in voxel units 
                   const Vec3d&       origin, // origin of grid in world units
                   const std::string& name,// name of grid
                   const BufferT& buffer)
{
    auto builder = initBox(width, height, depth, center, voxelSize, halfWidth, origin);
    builder->sdfToFog();
    auto handle = builder->template getHandle<BufferT>(double(voxelSize), origin, name, GridClass::FogVolume, buffer);
    assert(handle);
    return handle;
} // createFogVolumeBox

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointBox(int pointsPerVoxel,// half-width of narrow band in voxel units
               ValueT width,// width of box in world units
               ValueT height,// height of box in world units
               ValueT depth,// depth of box in world units
               const Vec3d& center,//center of sphere in world units
               ValueT voxelSize,// size of a voxel in world units
               const Vec3d &origin,// origin of grid in world units
               const std::string& name,// name of grid
               const BufferT& buffer)
{
    auto boxHandle = createLevelSetBox(width, height, depth, center, voxelSize, 0.5f, origin, "dummy", buffer);
    assert(boxHandle);
    auto *boxGrid = boxHandle.template grid<ValueT>();
    assert(boxGrid);
    auto pointHandle = createPointScatter(*boxGrid, pointsPerVoxel, name, buffer);
    assert(pointHandle);
    return pointHandle;
    
}// createPointBox

//================================================================================================

template <typename ValueT, typename BufferT>
inline GridHandle<BufferT>
createPointScatter(const NanoGrid<ValueT> &srcGrid,// origin of grid in world units
                   int pointsPerVoxel,// half-width of narrow band in voxel units
                   const std::string &name,// name of grid
                   const BufferT& buffer)
{
    static_assert(is_floating_point<ValueT>::value,"Sphere: expect floating point");
    using Vec3T = Vec3<ValueT>;
    if (pointsPerVoxel < 1) throw std::runtime_error("Expected at least one point per voxel");
    if (!srcGrid.isLevelSet()) throw std::runtime_error("Expected a level set grid");
    
    const uint64_t pointCount = pointsPerVoxel * srcGrid.activeVoxelCount();
    const uint64_t pointSize = AlignUp<NANOVDB_DATA_ALIGNMENT>(pointCount * sizeof(Vec3T));
    if (pointCount == 0) throw std::runtime_error("No particles to scatter");
    std::vector<Vec3T> xyz;
    xyz.reserve(pointCount);
    GridBuilder<uint32_t> builder(std::numeric_limits<uint32_t>::max(), pointSize);
    auto dstAcc = builder.getAccessor();
    std::srand(1234);
    const ValueT s = 1/(1+ValueT(RAND_MAX));
    auto op = [&](const Coord &p){
        //return Vec3T(p[0] + s*rand(), p[1] + s*rand(), p[2] + s*rand());
        return Vec3T(p[0] + s*rand() - 0.5, p[1] + s*rand() - 0.5, p[2] + s*rand() - 0.5);
        //return srcGrid.indexToWorld(Vec3T(p[0] + s*rand(), p[1] + s*rand(), p[2] + s*rand()));
    };
    const auto &srcTree = srcGrid.tree();
    for (uint32_t i=0, end = srcTree.nodeCount(0); i<end; ++i) {
        auto *srcLeaf = srcTree.template getNode<0>(i);
        auto *dstLeaf = dstAcc.setValue(srcLeaf->origin(), pointsPerVoxel);// allocates leaf node
        dstLeaf->mValueMask = srcLeaf->valueMask();
        for (uint32_t j=0, m=0; j<512; ++j) {
            if (dstLeaf->mValueMask.isOn(j)) {
                const auto ijk = srcLeaf->offsetToGlobalCoord(j);
                for (int n=0; n<pointsPerVoxel; ++n, ++m) xyz.push_back(op(ijk));
            }
            dstLeaf->mValues[j] = m;
        }
    }
    assert(pointCount == xyz.size());
    auto handle = builder.template getHandle<BufferT>(srcGrid.map(), name, GridClass::PointData, buffer);
    assert(handle);
    auto *dstGrid = handle.template grid<uint32_t>();
    assert(dstGrid);
    auto &dstTree = dstGrid->tree();
    if (dstTree.nodeCount(0) == 0) throw std::runtime_error("Expect leaf nodes!");
    auto *leafData = const_cast<typename NanoLeaf<uint32_t>::DataType*>(dstTree.template getNode<0>(0u)->data());
    leafData[0].mValueMin = 0; // start of prefix sum
    for (uint32_t i = 1, n = dstTree.nodeCount(0); i < n; ++i) {
        leafData[i].mValueMin = leafData[i - 1].mValueMin + leafData[i - 1].mValueMax;
    }
    auto &meta = const_cast<GridBlindMetaData&>(dstGrid->blindMetaData(0u));
    meta.mByteOffset = handle.size() - pointSize;// offset from Grid to blind data
    meta.mElementCount = xyz.size();
    meta.mFlags = 0;
    meta.mDataClass = GridBlindDataClass::AttributeArray;
    meta.mSemantic = GridBlindDataSemantic::PointPosition;
    if (name.length() + 1 > GridBlindMetaData::MaxNameSize) {
        std::stringstream ss;
        ss << "Point attribute name \"" << name << "\" is more then " 
           << nanovdb::GridBlindMetaData::MaxNameSize << " characters";
        throw std::runtime_error(ss.str());
    }
    memcpy(meta.mName, name.c_str(), name.size() + 1);
    if (std::is_same<ValueT, float>::value) { // resolved at compiletime
        meta.mDataType = GridType::Vec3f;
    } else if (std::is_same<ValueT, double>::value) {
        meta.mDataType = GridType::Vec3d;
    } else {
        throw std::runtime_error("Unsupported value type");
    }
    memcpy(handle.data() + meta.mByteOffset, xyz.data(), xyz.size() * sizeof(Vec3T));
    return handle;
}// createPointScatter

} // namespace nanovdb

#endif // NANOVDB_GRIDBUILDER_H_HAS_BEEN_INCLUDED
