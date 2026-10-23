// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/PointsToGrid.cuh

    \authors Greg Klar (initial version) and Ken Museth (final version)

    \brief Generates NanoVDB grids from a list of voxels or points on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <vector>
#include <tuple>
#include <cinttypes>
#include <algorithm>
#include <cmath> // for std::isfinite
#include <cstddef> // for std::byte
#include <stdexcept> // for std::invalid_argument
#include <type_traits> // for the stream-argument check
#include <iostream> // for std::cerr
#include <limits> // for std::numeric_limits
#include <string> // for std::string, std::to_string
#include <utility> // for std::move

#include <nanovdb/NanoVDB.h>
#include <nanovdb/cuda/DeviceBuffer.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/HandleStorage.h>
#include <nanovdb/cuda/GridHandle.cuh> // for merging the per-set grids in the point-set overloads
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>

/*
   Note: 4.29 billion (=2^32) coordinates of type Vec3f have a memory footprint of 48 GB!
*/

namespace nanovdb {// ================================================================================

namespace tools::cuda {// ============================================================================

/// @brief Generates a NanoGrid<Point> from a list of point coordinates on the device. This method is
///        mainly used as a means to build a BVH acceleration structure for points, e.g. for efficient rendering.
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dWorldPoints Raw or fancy pointer to list of point coordinates in world space on the device
/// @param pointCount number of point in the list @c d_world
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param type Defined the way point information is represented in the output grid (see PointType enum NanoVDB.h)
///             Must not be PointType::Disable
/// @param buffer Instance of the device buffer used for memory allocation
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return Returns a handle with a grid of type NanoGrid<Point> where point information, e.g. coordinates,
///         are represented as blind data defined by @c type.
/// @throw std::invalid_argument if the point count is negative, or the voxel size is not positive and finite, or @c type is PointType::Disable
template<typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             double voxelSize,
             PointType type = PointType::Default,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

/// @brief Generates a NanoGrid<Point> from a list of point coordinates on the device. This method is
///        mainly used as a means to build a BVH acceleration structure for points, e.g. for efficient rendering.
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dWorldPoints Raw or fancy pointer to list of point coordinates in world space on the device
/// @param pointCount total number of point in the list @c d_world
/// @param maxPointsPerVoxel Max density of points per voxel, i.e. maximum number of points in any voxel
/// @param tolerance allow for point density to vary by the specified tolerance (defaults to 1). That is, the voxel size
///                  is selected such that the max density is +/- the tolerance.
/// @param maxIterations Maximum number of iterations used to search for a voxel size that produces a point density
///                      with specified tolerance takes.
/// @param type Defined the way point information is represented in the output grid (see PointType enum in NanoVDB.h)
///             Must not be PointType::Disable
/// @param buffer Instance of the device buffer used for memory allocation
/// @param stream optional CUDA stream (defaults to CUDA stream 0)
/// @return Returns a handle with a grid of type NanoGrid<Point> where point information, e.g. coordinates,
///         are represented as blind data defined by @c type.
/// @throw std::invalid_argument if the point count is negative, or the maximum points per voxel is not positive, or @c type is PointType::Disable
template<typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             int maxPointPerVoxel,
             int tolerance = 1,
             int maxIterations = 10,
             PointType type = PointType::Default,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

/// @note An empty @c pointSet yields an empty handle, and a set without points an empty grid.
/// @throw std::invalid_argument if any set's voxel size is not positive and finite, or its point type is PointType::Disable
template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
pointsToGrid(const std::vector<std::tuple<const PtrT,size_t,double,PointType>> &pointSet,
            const BufferT &buffer = BufferT(),
            cudaStream_t stream = 0);

//-----------------------------------------------------------------------------------------------------

/// @brief Generates a NanoGrid of any type from a list of voxel coordinates on the device. Unlike @c cudaPointsToGrid
///        this method only builds the grid but does not encode the coordinates as blind data. It is mainly useful as a
///        means to generate a grid that is know to contain the voxels given in the list.
/// @tparam BuildT Template type of the return grid
/// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world space. Dereferencing should return Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam ResourceT Template type of optional resource used for internal temporary memory
/// @param dGridVoxels Raw or fancy pointer to list of voxel coordinates in grid (or index) space on the device
/// @param pointCount number of voxel in the list @c dGridVoxels
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param buffer Instance of the device buffer used for memory allocation
/// @return Returns a handle with the grid of type NanoGrid<BuildT>
/// @throw std::invalid_argument if the voxel size is not positive and finite
template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
voxelsToGrid(const PtrT dGridVoxels,
             size_t voxelCount,
             double voxelSize = 1.0,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-------------------------------------------------------------------------------------------------------

/// @brief Generates one NanoGrid per voxel set, all sets built together in one pass (see the
///        batched overload below), into one buffer in set order.
/// @note An empty @c pointSet yields an empty handle, and a set without points an empty grid.
/// @throw std::invalid_argument if any set's voxel size is not positive and finite
template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
voxelsToGrid(const std::vector<std::tuple<const PtrT, size_t, double>> &pointSet,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-------------------------------------------------------------------------------------------------------

/// @brief Generates @c gridCount NanoGrids from one list of voxel coordinates on the device, in one pass:
///        grid g holds the voxels dGridVoxels[d_offsets[g]] up to dGridVoxels[d_offsets[g+1]-1]. The grids
///        share one buffer, in order, and the build blocks the host twice however many grids it makes.
/// @tparam BuildT Template type of the returned grids; Point grids are not supported
/// @tparam PtrT Template type to a raw or fancy-pointer of voxel coordinates in index space
/// @param dGridVoxels Raw or fancy pointer to the list of voxel coordinates on the device
/// @param voxelCount Total number of voxels in the list
/// @param d_offsets Device array of @c gridCount+1 non-decreasing offsets, from 0 to @c voxelCount
/// @param gridCount Number of grids to build
/// @param voxelSize Size of a voxel in world units, shared by every grid
/// @param buffer Instance of the device buffer used for memory allocation
/// @return Returns a handle with @c gridCount grids of type NanoGrid<BuildT>
/// @note A grid without voxels is a valid empty grid, and @c gridCount == 0 yields an empty handle.
/// @throw std::invalid_argument if the voxel size is not positive and finite, or the offsets are not as described
template<typename BuildT, typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
voxelsToGrid(const PtrT dGridVoxels,
             size_t voxelCount,
             const uint64_t *d_offsets,
             uint32_t gridCount,
             double voxelSize = 1.0,
             const BufferT &buffer = BufferT(),
             cudaStream_t stream = 0);

//-------------------------------------------------------------------------------------------------------

/// @brief Resource form of voxelsToGrid: the output and all internal scratch
///        allocate through @c resource, on @c stream. Passing a cudaStream_t, or a
///        non-buffer type that converts to one, selects this form (a literal 0 does not: write cudaStream_t(0)); the
///        prototype-buffer form above keeps its call shape.
/// @tparam ResourceT stream-ordered memory resource, held by value (a copy of
///         default_resource<ResourceT>() when omitted); pass a
///         cuda::ResourceRef<R> to allocate through a borrowed instance
/// @return a handle over cuda::Buffer<std::byte, ResourceT> on the device
/// @throw std::invalid_argument if the voxel size is not positive and finite
template<typename BuildT, typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const PtrT dGridVoxels,
             size_t voxelCount,
             double voxelSize,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

/// @brief Resource form of pointsToGrid for a fixed voxel size; see the
///        resource form of voxelsToGrid.
/// @throw std::invalid_argument if the point count is negative, or the voxel size is not positive and finite, or @c type is PointType::Disable
template<typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             double voxelSize,
             PointType type,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

/// @brief Resource form of pointsToGrid targeting a maximum point density;
///        see the resource form of voxelsToGrid.
/// @throw std::invalid_argument if the point count is negative, or the maximum points per voxel is not positive, or @c type is PointType::Disable
template<typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const PtrT dWorldPoints,
             int pointCount,
             int maxPointsPerVoxel,
             int tolerance,
             int maxIterations,
             PointType type,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

/// @brief Resource form of pointsToGrid over a point set: one grid per set,
///        merged into one handle; see the single-set resource form of
///        voxelsToGrid above.
/// @note An empty @c pointSet yields an empty handle, and a set without points an empty grid.
/// @throw std::invalid_argument if any set's voxel size is not positive and finite, or its point type is PointType::Disable
template<typename BuildT, typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const std::vector<std::tuple<const PtrT,size_t,double,PointType>> &pointSet,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

/// @brief Resource form of voxelsToGrid over a voxel set: one grid per set,
///        built together into one handle; see the single-set resource form of
///        voxelsToGrid above.
/// @note An empty @c pointSet yields an empty handle, and a set without points an empty grid.
/// @throw std::invalid_argument if any set's voxel size is not positive and finite
template<typename BuildT, typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const std::vector<std::tuple<const PtrT, size_t, double>> &pointSet,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

/// @brief Resource form of the batched voxelsToGrid; see the prototype form above.
/// @throw std::invalid_argument if the voxel size is not positive and finite, or the offsets are not as described
template<typename BuildT, typename PtrT, typename ResourceT = nanovdb::cuda::DeviceResource>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const PtrT dGridVoxels,
             size_t voxelCount,
             const uint64_t *d_offsets,
             uint32_t gridCount,
             double voxelSize,
             cudaStream_t stream,
             ResourceT resource = nanovdb::cuda::default_resource<ResourceT>());

}// namespace tools::cuda ========================================================================

/// @brief Example class of a fancy pointer that can optionally be used as a template for writing
///        a custom fancy pointer that allows for particle coordinates to be arrange non-linearly
///        in memory. For instance, when coordinates are interlaced with other data, e.g. an array
///        of structs, a custom implementation of fancy_ptr::operator[](size_t i) can account for
///        strides that skip other interlaces data.
/// @tparam T Template type that specifies the type use for the coordinates of the points
template<typename T>
class fancy_ptr
{
    const T* mPtr;
public:
    /// @brief Default constructor.
    /// @note  This method is actually not required by cuda::PointsToGrid
    /// @param ptr Pointer to array of elements
    __hostdev__ explicit fancy_ptr(const T* ptr = nullptr) : mPtr(ptr) {}
    /// @brief Index acces into the array pointed to by the stored pointer.
    /// @note  This method is required by cuda::PointsToGrid!
    /// @param i Unsigned index of the element to be returned
    /// @return Const reference to the element at the i'th position
    __hostdev__ inline const T& operator[](size_t i) const {return mPtr[i];}
    /// @brief Dummy implementation required by pointer_traits.
    /// @note  Note that only the return type matters!
    /// @details Unlike operator[] it is safe to assume that all pointer types have operator*,
    ///          which is why pointer_traits makes use of it to determine the element_type that
    ///          a pointer class is pointing to. E.g. operator[] is not always defined for std::shared_ptr!
    __hostdev__ inline const T& operator*() const {return *mPtr;}
};// fancy_ptr<T>

/// @brief Simple stand-alone function that can be used to conveniently construct a fancy_ptr
/// @tparam T Template type that specifies the type use for the coordinates of the points
/// @param ptr Raw pointer to data
/// @return a new instance of a fancy_ptr
template<typename T>
fancy_ptr<T> make_fancy(const T* ptr = nullptr) {return fancy_ptr<T>(ptr);}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief Trait of points, like type of pointer and size of the pointer type
template<typename>
struct pointer_traits;

template<typename T>
struct pointer_traits<T*> {
    using element_type = T;
    static constexpr size_t element_size = sizeof(T);
};

template<typename T>
struct pointer_traits {
    using element_type = typename util::remove_reference<decltype(*util::declval<T>())>::type;// assumes T::operator*() exists!
    static constexpr size_t element_size = sizeof(element_type);
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/// @brief computes the relative 8-bit voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel 8-bit output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u8 &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = Vec3d(indexToWorld.applyInverseMap(world));// world -> index
    static constexpr double encode = double((1<<8) - 1);
    voxel[0] = uint8_t( encode*(ijk[0] - math::Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint8_t( encode*(ijk[1] - math::Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint8_t( encode*(ijk[2] - math::Floor(ijk[2] + 0.5) + 0.5) );
}

/// @brief computes the relative 16-bit voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel 16-bit output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u16 &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = Vec3d(indexToWorld.applyInverseMap(world));// world -> index
    static constexpr double encode = double((1<<16) - 1);
    voxel[0] = uint16_t( encode*(ijk[0] - math::Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint16_t( encode*(ijk[1] - math::Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint16_t( encode*(ijk[2] - math::Floor(ijk[2] + 0.5) + 0.5) );
}

/// @brief computes the relative float voxel offsets from a world coordinate
/// @tparam Vec3T Type of the world coordinate
/// @param voxel float output coordinates that are relative to a voxel
/// @param world input world coordinates
/// @param indexToWorld Transform from index to world space
template<typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3f &voxel, const Vec3T &world, const Map &indexToWorld)
{
    const Vec3d ijk = Vec3d(indexToWorld.applyInverseMap(world));// world -> index
    voxel[0] = float( ijk[0] - math::Floor(ijk[0] + 0.5) );
    voxel[1] = float( ijk[1] - math::Floor(ijk[1] + 0.5) );
    voxel[2] = float( ijk[2] - math::Floor(ijk[2] + 0.5) );
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u8 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<8) - 1);
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u16 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<16) - 1);
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template<typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3f &voxel, const Coord &ijk, const Map &map)
{
    if constexpr(util::is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace tools::cuda {

template<typename BuildT>
struct PointsToGridData {
    Map map;
    void     *d_bufferPtr;
    uint64_t *d_keys, *d_tile_keys, *d_lower_keys, *d_leaf_keys;// device pointer to 64 bit keys
    uint64_t  grid, tree, root, upper, lower, leaf, meta, blind, size;// byte offsets to nodes in buffer
    uint32_t *d_indx;// device pointer to point indices (or IDs)
    uint32_t  nodeCount[3], *pointsPerLeafPrefix, *pointsPerLeaf;// 0=leaf,1=lower, 2=upper
    uint32_t  voxelCount,  *pointsPerVoxelPrefix, *pointsPerVoxel;
    __hostdev__ NanoGrid<BuildT>&  getGrid() const {return *util::PtrAdd<NanoGrid<BuildT>>(d_bufferPtr, grid);}
    __hostdev__ NanoTree<BuildT>&  getTree() const {return *util::PtrAdd<NanoTree<BuildT>>(d_bufferPtr, tree);}
    __hostdev__ NanoRoot<BuildT>&  getRoot() const {return *util::PtrAdd<NanoRoot<BuildT>>(d_bufferPtr, root);}
    __hostdev__ NanoUpper<BuildT>& getUpper(int i) const {return *(util::PtrAdd<NanoUpper<BuildT>>(d_bufferPtr, upper)+i);}
    __hostdev__ NanoLower<BuildT>& getLower(int i) const {return *(util::PtrAdd<NanoLower<BuildT>>(d_bufferPtr, lower)+i);}
    __hostdev__ NanoLeaf<BuildT>&  getLeaf(int i) const {return *(util::PtrAdd<NanoLeaf<BuildT>>(d_bufferPtr, leaf)+i);}
    __hostdev__ GridBlindMetaData& getMeta() const { return *util::PtrAdd<GridBlindMetaData>(d_bufferPtr, meta);};
     template<typename Vec3T>
    __hostdev__ Vec3T& getPoint(int i) const {return *(util::PtrAdd<Vec3T>(d_bufferPtr, blind)+i);}
};// PointsToGridData


namespace detail {

/// @brief Argument checks for the PointsToGrid constructors (map, voxel size,
///        density target) and the pointsToGrid free functions (point count).
inline double checkVoxelSize(double voxelSize, const char* tool)
{
    if (!(voxelSize > 0.0) || !std::isfinite(voxelSize))
        throw std::invalid_argument(std::string(tool) + ": voxel size must be positive and finite");
    return voxelSize;
}
inline void checkMap(const Map& map, const char* tool)
{
    const double* m = map.mMatD;
    const double det = m[0]*(m[4]*m[8] - m[5]*m[7]) - m[1]*(m[3]*m[8] - m[5]*m[6]) + m[2]*(m[3]*m[7] - m[4]*m[6]);
    if (!(det != 0.0) || !std::isfinite(det))
        throw std::invalid_argument(std::string(tool) + ": the map must be invertible and finite");
}
inline void checkPointCount(int pointCount, const char* tool)
{
    if (pointCount < 0) throw std::invalid_argument(std::string(tool) + ": negative point count");
}
inline void checkMaxPointsPerVoxel(int maxPointsPerVoxel, const char* tool)
{
    if (maxPointsPerVoxel <= 0) throw std::invalid_argument(std::string(tool) + ": maximum points per voxel must be positive");
}

}// namespace detail

template<typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class PointsToGrid
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "PointsToGrid allocates stream-ordered scratch and requires an AsyncResource");
public:
    /// @brief Map constructor, which other constructors might call
    /// @param map Map to be used for the output device grid
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @throw std::invalid_argument if @c map is singular or not finite
    PointsToGrid(const Map &map, cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream((detail::checkMap(map, "PointsToGrid"), stream))
        , mResource(&resource)
        , mTimer(stream)
        , mPointType(util::is_same<BuildT,Point>::value ? PointType::Default : PointType::Disable)
        , mDeviceDataBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 1, nanovdb::cuda::noInit)
        , mKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mTileKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mLeafKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mLowerKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mIndxBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerVoxelBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerVoxelPrefixBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerLeafBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerLeafPrefixBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mTempDevicePool(resource)
    {
        mData.map = map;
        mDeviceData = mDeviceDataBuf.data();
    }

    /// @brief Default constructor that calls the Map constructor defined above
    /// @param scale Voxel size in world units
    /// @param trans Translation of origin in world units
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @throw std::invalid_argument if @c scale is not positive
    PointsToGrid(const double scale = 1.0, const Vec3d &trans = Vec3d(0.0), cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : PointsToGrid(Map(detail::checkVoxelSize(scale, "PointsToGrid"), trans), stream, resource){}

    /// @brief Constructor from a target maximum number of particles per voxel. Calls the Map constructor defined above
    /// @param maxPointsPerVoxel Maximum number of points oer voxel
    /// @param stream optional CUDA stream (defaults to CUDA stream 0)
    /// @throw std::invalid_argument if @c maxPointsPerVoxel is not positive
    PointsToGrid(int maxPointsPerVoxel, int tolerance = 1, int maxIterations = 10, cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : PointsToGrid((detail::checkMaxPointsPerVoxel(maxPointsPerVoxel, "PointsToGrid"), Map(1.0)), stream, resource)
    {
        mMaxPointsPerVoxel = maxPointsPerVoxel;
        mTolerance = tolerance;
        mMaxIterations = maxIterations;
    }


    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) {mVerbose = level; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable){mChecksum = mode;}

    /// @brief Set the name of the output grid
    /// @param name name of the output grid
    void setGridName(const std::string &name) {mGridName = name;}

    /// @brief Set how the output grid represents the points, only available when BuildT == Point
    /// @throw std::invalid_argument if @c type is PointType::Disable, which would leave a point grid without points
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value>::type
    setPointType(PointType type)
    {
        if (type == PointType::Disable) throw std::invalid_argument("PointsToGrid<Point>: the point type must not be PointType::Disable");
        mPointType = type;
    }

    /// @brief Creates a handle to a grid with the specified build type from a list of points in index or world space
    /// @tparam BuildT Build type of the output grid, i.e NanoGrid<BuildT>
    /// @tparam PtrT Template type to a raw or fancy-pointer of point coordinates in world or index space.
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param points device point to an array of points in world space
    /// @param pointCount number of input points or voxels
    /// @param buffer prototype buffer whose resource (or create() pool) allocates the output
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    /// @note An empty input (@c pointCount == 0) yields a valid empty grid.
    template<typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer>
    GridHandle<BufferT> getHandle(const PtrT points,
                                  size_t pointCount,
                                  const BufferT &buffer = BufferT());

    /// @brief Resource form of getHandle: the grid allocates through this
    ///        builder's resource, the one its scratch uses, on its stream.
    /// @return a handle over cuda::Buffer<std::byte, ResourceRef<ResourceT>>,
    ///         which borrows the resource: it must outlive the handle
    /// @param points device point to an array of points in world space
    /// @param pointCount number of input points or voxels
    /// @note An empty input (@c pointCount == 0) yields a valid empty grid.
    template<typename PtrT>
    GridHandle<nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>> build(const PtrT points, size_t pointCount)
    {
        using BufferT = nanovdb::cuda::Buffer<std::byte, nanovdb::cuda::ResourceRef<ResourceT>>;
        return this->getHandle(points, pointCount, BufferT(mStream, this->ref(), 0, nanovdb::cuda::noInit));
    }

    template<typename PtrT>
    void countNodes(const PtrT points, size_t pointCount);

    template<typename PtrT>
    void processGridTreeRoot(const PtrT points, size_t pointCount);

    void processUpperNodes();

    void processLowerNodes();

    void processLeafNodes(size_t pointCount);

    template<typename PtrT>
    void processPoints(const PtrT points, size_t pointCount);

    // Only instantiated when BuildT == Point, from the corresponding branch of
    // processPoints. Holds the encode kernels because nvcc rejects an extended
    // lambda defined directly inside the block of an if constexpr statement on
    // some host compilers.
    template<typename PtrT>
    void encodePoints(const PtrT points, size_t pointCount);

    void processBBox();

    // the following methods are only defined when BuildT == Point
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value, uint32_t>::type
    maxPointsPerVoxel() const {return mMaxPointsPerVoxel;}
    template<typename T = BuildT> typename util::enable_if<util::is_same<T, Point>::value, uint32_t>::type
    maxPointsPerLeaf()  const {return mMaxPointsPerLeaf;}

private:
    static constexpr unsigned int mNumThreads = 128;// seems faster than the old value of 256!
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    template<typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::detail::borrowInternal(*mResource); }

    cudaStream_t             mStream{0};
    ResourceT*               mResource;// non-owning; all device allocations (mDeviceData + scratch) route through this resource instance
    util::cuda::Timer        mTimer;
    PointType                mPointType;
    std::string              mGridName;
    int                      mVerbose{0};
    PointsToGridData<BuildT> mData, *mDeviceData;// mDeviceData views mDeviceDataBuf
    // Owners of the device arrays that mData's raw pointers view. Raw views are
    // refreshed immediately after every owner event (assign, swap, destroy), so
    // the views -- including the device-visible fields inside mData -- are
    // never stale. Members rather than locals because the pipeline frees them
    // across member functions (countNodes allocates; processUpperNodes,
    // processLeafNodes, processPoints and processBBox release).
    BufT<PointsToGridData<BuildT>> mDeviceDataBuf;
    BufT<uint64_t>           mKeysBuf, mTileKeysBuf, mLeafKeysBuf, mLowerKeysBuf;
    BufT<uint32_t>           mIndxBuf, mPointsPerVoxelBuf, mPointsPerVoxelPrefixBuf, mPointsPerLeafBuf, mPointsPerLeafPrefixBuf;
    uint32_t                 mMaxPointsPerVoxel{0u}, mMaxPointsPerLeaf{0u};
    int                      mTolerance{1}, mMaxIterations{1};
    CheckMode                mChecksum{CheckMode::Disable};

    nanovdb::cuda::TempPool<ResourceT> mTempDevicePool;

    template<typename PtrT, typename BufferT>
    BufferT getBuffer(const PtrT points, size_t pointCount, const BufferT &buffer);
};// tools::cuda::PointsToGrid<BuildT, ResourceT>

namespace kernels {
/// @details Used by cuda::PointsToGrid<BuildT, ResourceT>::processLeafNodes before the computation
/// of prefix-sum for index grid.
/// Moving this away from an implementation using the lambdaKernel wrapper
/// to fix the following on Windows platform:
/// error : For this host platform/dialect, an extended lambda cannot be defined inside the 'if'
/// or 'else' block of a constexpr if statement.
/// function in a lambda through lambdaKernel wrapper defined in CudaUtils.h.
template<typename BuildT>
__global__ void fillValueIndexKernel(const size_t numItems, unsigned int offset, uint64_t* devValueIndex, PointsToGridData<BuildT>* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;
    devValueIndex[tid + offset] = static_cast<uint64_t>(d_data->getLeaf(tid + offset).mValueMask.countOn());
}

/// @details Used by PointsToGrid<BuildT, ResourceT>::processLeafNodes for the computation
/// of prefix-sum for index grid.
/// Moving this away from an implementation using the lambdaKernel wrapper
/// to fix the following on Windows platform:
/// error : For this host platform/dialect, an extended lambda cannot be defined inside the 'if'
/// or 'else' block of a constexpr if statement.
template<typename BuildT>
__global__ void leafPrefixSumKernel(const size_t numItems, unsigned int offset, uint64_t* devValueIndexPrefix, PointsToGridData<BuildT>* d_data) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numItems) return;

    auto &leaf = d_data->getLeaf(tid + offset);
    leaf.mOffset = 1u;// will be re-set below
    const uint64_t *w = leaf.mValueMask.words();
    uint64_t &prefixSum = leaf.mPrefixSum, sum = util::countOn(*w++);
    prefixSum = sum;
    for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
        sum += util::countOn(*w++);
        prefixSum |= sum << n;// each pre-fixed sum is encoded in 9 bits
    }
    if ((tid + offset) == 0) {
        d_data->getGrid().mData1 = 1u + devValueIndexPrefix[d_data->nodeCount[0]-1];// set total count
        d_data->getTree().mVoxelCount = devValueIndexPrefix[d_data->nodeCount[0]-1];
    } else {
        leaf.mOffset = 1u + devValueIndexPrefix[tid + offset -1];// background is index 0
    }
}

} // namespace kernels

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// fdef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename PtrT, typename BufferT>
inline GridHandle<BufferT>
PointsToGrid<BuildT, ResourceT>::getHandle(const PtrT points,
                                size_t pointCount,
                                const BufferT &pool)
{
    if (mVerbose==1) mTimer.start("\nCounting nodes");
    this->countNodes(points, pointCount);

    if (mVerbose==1) mTimer.restart("Initiate buffer");
    auto buffer = this->getBuffer(points, pointCount, pool);

    if (mVerbose==1) mTimer.restart("Process grid,tree,root");
    this->processGridTreeRoot(points, pointCount);

    if (pointCount > 0) {// no points: the headers alone form a valid empty grid
        if (mVerbose==1) mTimer.restart("Process upper nodes");
        this->processUpperNodes();

        if (mVerbose==1) mTimer.restart("Process lower nodes");
        this->processLowerNodes();

        if (mVerbose==1) mTimer.restart("Process leaf nodes");
        this->processLeafNodes(pointCount);

        if (mVerbose==1) mTimer.restart("Process points");
        this->processPoints(points, pointCount);

        if (mVerbose==1) mTimer.restart("Process bbox");
        this->processBBox();
    }
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.restart("Computation of checksum");
    tools::cuda::updateChecksum((GridData*)nanovdb::cuda::detail::deviceStorageData(buffer), mChecksum, mStream);
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return GridHandle<BufferT>(std::move(buffer));
}// PointsToGrid<BuildT, ResourceT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// --- CUB helpers ---
template<uint8_t BitCount, typename InT = uint64_t, typename OutT = uint64_t>
struct ShiftRight
{
    __hostdev__ inline OutT operator()(const InT& v) const {return static_cast<OutT>(v >> BitCount);}
};

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace detail {

/// @brief Key of the root tile containing @a ijk: 21 bits per axis, offset so negative coordinates sort first
__hostdev__ inline uint64_t tileKey(const Coord &ijk)
{
    // Note: int32_t has a range of -2^31 to 2^31 - 1 whereas uint32_t has a range of 0 to 2^32 - 1
    static constexpr int64_t kOffset = 1 << 31;
    return (uint64_t(uint32_t(int64_t(ijk[2]) + kOffset) >> 12)      ) | // z is the lower 21 bits
           (uint64_t(uint32_t(int64_t(ijk[1]) + kOffset) >> 12) << 21) | // y is the middle 21 bits
           (uint64_t(uint32_t(int64_t(ijk[0]) + kOffset) >> 12) << 42);  // x is the upper 21 bits
}

}// namespace detail

template<typename BuildT, typename PtrT>
struct TileKeyFunctor {
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, uint64_t* d_keys, uint32_t* d_indx) {
        d_indx[tid] = uint32_t(tid);
        uint64_t &key = d_keys[tid];
        if constexpr(util::is_same<BuildT, Point>::value) {// points are in world space
            if constexpr(util::is_same<Vec3T, Vec3f>::value) {
                key = detail::tileKey(d_data->map.applyInverseMapF(points[tid]).round());
            } else {// points are Vec3d
                key = detail::tileKey(d_data->map.applyInverseMap(points[tid]).round());
            }
        } else if constexpr(util::is_same<Vec3T, Coord>::value) {// points Coord are in index space
            key = detail::tileKey(points[tid]);
        } else {// points are Vec3f or Vec3d in index space
            key = detail::tileKey(points[tid].round());
        }
    }
};

template<typename BuildT, typename PtrT>
struct VoxelKeyFunctor {
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, uint64_t id, uint64_t *d_keys, const uint32_t *d_indx) {
        auto voxelKey = [] __device__ (uint64_t tileID, const Coord &ijk){
            return tileID << 36 |                                       // upper offset: 64-15-12-9=28, i.e. last 28 bits
                uint64_t(NanoUpper<BuildT>::CoordToOffset(ijk)) << 21 | // lower offset: 32^3 = 2^15,   i.e. next 15 bits
                uint64_t(NanoLower<BuildT>::CoordToOffset(ijk)) <<  9 | // leaf  offset: 16^3 = 2^12,   i.e. next 12 bits
                uint64_t(NanoLeaf< BuildT>::CoordToOffset(ijk));        // voxel offset:  8^3 =  2^9,   i.e. first 9 bits
        };// voxelKey lambda functor
        Vec3T p = points[d_indx[tid]];
        if constexpr(util::is_same<BuildT, Point>::value) p = util::is_same<Vec3T, Vec3f>::value ? d_data->map.applyInverseMapF(p) : d_data->map.applyInverseMap(p);
        d_keys[tid] = voxelKey(id, p.round());
    }
};

template<typename BuildT, typename PtrT>
struct BulkVoxelKeyFunctor {
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, const uint32_t *d_tile_offsets, uint32_t numTiles, uint64_t *d_keys, const uint32_t *d_indx, uint32_t tileIdOffset) {
        auto voxelKey = [] __device__ (uint64_t tileID, const Coord &ijk){
            return tileID << 36 |                                       // upper offset: 64-15-12-9=28, i.e. last 28 bits
                uint64_t(NanoUpper<BuildT>::CoordToOffset(ijk)) << 21 | // lower offset: 32^3 = 2^15,   i.e. next 15 bits
                uint64_t(NanoLower<BuildT>::CoordToOffset(ijk)) <<  9 | // leaf  offset: 16^3 = 2^12,   i.e. next 12 bits
                uint64_t(NanoLeaf< BuildT>::CoordToOffset(ijk));        // voxel offset:  8^3 =  2^9,   i.e. first 9 bits
        };// voxelKey lambda functor
        const uint64_t tileID = tileIdOffset + (thrust::upper_bound(thrust::seq, d_tile_offsets, d_tile_offsets + numTiles + 1, uint32_t(tid)) - d_tile_offsets - 1);
        Vec3T p = points[d_indx[tid]];
        if constexpr(util::is_same<BuildT, Point>::value) p = util::is_same<Vec3T, Vec3f>::value ? d_data->map.applyInverseMapF(p) : d_data->map.applyInverseMap(p);
        d_keys[tid] = voxelKey(tileID, p.round());
    }
};

namespace detail {

/// @brief Index-space voxel of point @a i, mapped from world space first when BuildT is Point
template<typename BuildT, typename PtrT>
__device__ inline Coord pointToVoxel(const PointsToGridData<BuildT> *d_data, const PtrT points, size_t i)
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;
    Vec3T p = points[i];
    if constexpr(util::is_same<BuildT, Point>::value) p = util::is_same<Vec3T, Vec3f>::value ? d_data->map.applyInverseMapF(p) : d_data->map.applyInverseMap(p);
    return p.round();
}

/// @brief Offset of @a ijk within its root tile: upper (15 bits), lower (12) and leaf (9) offsets
template<typename BuildT>
__hostdev__ inline uint64_t tileOffsetKey(const Coord &ijk)
{
    return uint64_t(NanoUpper<BuildT>::CoordToOffset(ijk)) << 21 |
           uint64_t(NanoLower<BuildT>::CoordToOffset(ijk)) <<  9 |
           uint64_t(NanoLeaf< BuildT>::CoordToOffset(ijk));
}

/// @brief Writes each point's in-tile offset key and the identity permutation
template<typename BuildT, typename PtrT>
struct TileOffsetKeyFunctor {
    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, uint64_t *d_keys, uint32_t *d_indx) {
        d_indx[tid] = uint32_t(tid);
        d_keys[tid] = tileOffsetKey<BuildT>(pointToVoxel(d_data, points, tid));
    }
};

/// @brief Writes the tile key of each point, taken in the order of @a d_indx
template<typename BuildT, typename PtrT>
struct PermutedTileKeyFunctor {
    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, const uint32_t *d_indx, uint64_t *d_keys) {
        d_keys[tid] = tileKey(pointToVoxel(d_data, points, d_indx[tid]));
    }
};

/// @brief 1 where a sorted key differs from its predecessor, else 0
struct TileStartFlag {
    const uint64_t *keys;
    __hostdev__ uint32_t operator()(uint32_t i) const { return i == 0 || keys[i] != keys[i - 1]; }
};

/// @brief Replaces each point's sorted tile key with its voxel key, the tile index (28 bits)
///        above the in-tile offset (36 bits), and gathers each tile's key at its index.
///        @a d_tileRank is the inclusive scan of TileStartFlag, so it is one past the tile index.
template<typename BuildT, typename PtrT>
struct RankedVoxelKeyFunctor {
    __device__
    void operator()(size_t tid, const PointsToGridData<BuildT> *d_data, const PtrT points, const uint32_t *d_indx,
                    const uint32_t *d_tileRank, uint64_t *d_keys, uint64_t *d_tileKeys) {
        const uint32_t tile = d_tileRank[tid] - 1u;
        if (tid == 0 || d_tileRank[tid - 1] != d_tileRank[tid]) d_tileKeys[tile] = d_keys[tid];
        d_keys[tid] = uint64_t(tile) << 36 | tileOffsetKey<BuildT>(pointToVoxel(d_data, points, d_indx[tid]));
    }
};

/// @brief Counts of leaf, lower and upper nodes and of voxels
struct NodeCounts { uint32_t leaf, lower, upper, voxel; };

/// @brief Nodes and voxels that begin at sorted voxel key @a i
struct NodeStarts {
    const uint64_t *keys;
    __hostdev__ NodeCounts operator()(uint32_t i) const {
        const uint64_t diff = i == 0 ? ~uint64_t(0) : keys[i] ^ keys[i - 1];// bits above a level's offset differ iff a node of that level starts here
        return {uint32_t(diff >> 9 != 0), uint32_t(diff >> 21 != 0), uint32_t(diff >> 36 != 0), uint32_t(diff != 0)};
    }
};

struct SumNodeCounts {
    __hostdev__ NodeCounts operator()(const NodeCounts &a, const NodeCounts &b) const {
        return {a.leaf + b.leaf, a.lower + b.lower, a.upper + b.upper, a.voxel + b.voxel};
    }
};

}// namespace detail

template<typename BuildT, typename ResourceT>
template<typename PtrT>
void PointsToGrid<BuildT, ResourceT>::countNodes(const PtrT points, size_t pointCount)
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;
    if constexpr(util::is_same<BuildT, Point>::value) {
        static_assert(util::is_same<Vec3T, Vec3f, Vec3d>::value, "Point (vs voxels) coordinates should be represented as Vec3f or Vec3d");
    } else {
        static_assert(util::is_same<Vec3T, Coord, Vec3f, Vec3d>::value, "Voxel coordinates should be represented as Coord, Vec3f or Vec3d");
    }
    if (pointCount == 0) {// no points: no nodes and no voxels
        mData.nodeCount[0] = mData.nodeCount[1] = mData.nodeCount[2] = 0;
        mData.voxelCount = 0;
        mMaxPointsPerLeaf = 0u;
        return;
    }

    mMaxPointsPerVoxel = math::Min(mMaxPointsPerVoxel, pointCount);
    int iterCounter = 0;
    struct Foo {// pairs current voxel size, dx, with the corresponding particle density, i.e. maximum number of points per voxel
        double   dx;
        uint32_t density;
        bool operator<(const Foo &rhs) const {return density < rhs.density || (density == rhs.density && dx < rhs.dx);}
    } min{0.0, 1}, max{0.0, 0};// min: as dx -> 0 density -> 1 point per voxel, max: density is 0 i.e. undefined

    // Points are sorted by their offset within a root tile, then stably by tile, which leaves
    // them in voxel order without the host knowing the tile count. Every node and voxel count
    // is then computed on the device and read back in one copy, which sizes the per-level
    // arrays exactly. Point grids also read back their densities, for the voxel-size search
    // and the points-per-leaf limit.
    BufT<uint64_t> keysScratch(mStream, this->ref(), 0, nanovdb::cuda::noInit);
    BufT<uint32_t> indxScratch(mStream, this->ref(), 0, nanovdb::cuda::noInit);
    BufT<detail::NodeCounts> countsScratch(mStream, this->ref(), 1, nanovdb::cuda::noInit);
    BufT<uint32_t> runCountScratch(mStream, this->ref(), 1, nanovdb::cuda::noInit);// run counts the host already has
    uint64_t *d_keys = nullptr;
    uint32_t *d_indx = nullptr;
    const auto pointIndex = thrust::make_counting_iterator<uint32_t>(0);

    // Bisection search for the voxel size dx that yields the target point
    // density: each iteration builds tile and voxel keys at the current dx,
    // then either converges or frees this iteration's buffers and retries.
    while (true) {

        mKeysBuf = BufT<uint64_t>(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        mIndxBuf = BufT<uint32_t>(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);// uint32_t can index 4.29 billion Coords, corresponding to 48 GB
        mData.d_keys = mKeysBuf.data();
        mData.d_indx = mIndxBuf.data();
        cudaCheck(cudaMemcpyAsync(mDeviceData, &mData, sizeof(PointsToGridData<BuildT>), cudaMemcpyHostToDevice, mStream));// copy mData from CPU -> GPU

        if (mVerbose==2) mTimer.start("\nAllocating arrays for keys and indices");
        keysScratch = BufT<uint64_t>(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        indxScratch = BufT<uint32_t>(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        d_keys = keysScratch.data();
        d_indx = indxScratch.data();

        if (mVerbose==2) mTimer.restart("Generate in-tile offset keys");
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::TileOffsetKeyFunctor<BuildT, PtrT>(), mDeviceData, points, d_keys, d_indx);
        cudaCheckError();
        if (mVerbose==2) mTimer.restart("DeviceRadixSort of "+std::to_string(pointCount)+" in-tile offset keys");
        CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, mData.d_keys, d_indx, mData.d_indx, pointCount, 0, 36);// 15+12+9 bits

        if (mVerbose==2) mTimer.restart("Generate tile keys");
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::PermutedTileKeyFunctor<BuildT, PtrT>(), mDeviceData, points, mData.d_indx, d_keys);
        cudaCheckError();
        if (mVerbose==2) mTimer.restart("DeviceRadixSort of "+std::to_string(pointCount)+" tile keys");
        CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, mData.d_keys, mData.d_indx, d_indx, pointCount, 0, 63);// 21 bits per coord; stable, so each tile stays in voxel order
        mIndxBuf.swap(indxScratch);// mIndxBuf now owns the voxel-sorted indices
        mData.d_indx = mIndxBuf.data();
        d_indx = indxScratch.data();// free: receives the tile ranks

        if (mVerbose==2) mTimer.restart("Generate voxel keys");
        CALL_CUBS(DeviceScan::InclusiveSum, thrust::make_transform_iterator(pointIndex, detail::TileStartFlag{mData.d_keys}), d_indx, pointCount);
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::RankedVoxelKeyFunctor<BuildT, PtrT>(), mDeviceData, points, mData.d_indx, d_indx, mData.d_keys, d_keys);
        cudaCheckError();
        indxScratch.destroy(); d_indx = nullptr;

        if (mVerbose==2) mTimer.restart("Count nodes and voxels");
        CALL_CUBS(DeviceReduce::Reduce, thrust::make_transform_iterator(pointIndex, detail::NodeStarts{mData.d_keys}), countsScratch.data(), pointCount, detail::SumNodeCounts(), detail::NodeCounts{0u, 0u, 0u, 0u});
        detail::NodeCounts counts;
        cudaCheck(cudaMemcpyAsync(&counts, countsScratch.data(), sizeof(counts), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        mData.nodeCount[0] = counts.leaf;
        mData.nodeCount[1] = counts.lower;
        mData.nodeCount[2] = counts.upper;
        mData.voxelCount   = counts.voxel;

        mTileKeysBuf = BufT<uint64_t>(mStream, this->ref(), mData.nodeCount[2], nanovdb::cuda::noInit);
        mData.d_tile_keys = mTileKeysBuf.data();
        cudaCheck(cudaMemcpyAsync(mData.d_tile_keys, d_keys, mData.nodeCount[2]*sizeof(uint64_t), cudaMemcpyDeviceToDevice, mStream));

        if (mVerbose==2) mTimer.restart("Count points per voxel");
        mPointsPerVoxelBuf = BufT<uint32_t>(mStream, this->ref(), mData.voxelCount, nanovdb::cuda::noInit);
        mData.pointsPerVoxel = mPointsPerVoxelBuf.data();
        CALL_CUBS(DeviceRunLengthEncode::Encode, mData.d_keys, d_keys, mData.pointsPerVoxel, runCountScratch.data(), pointCount);

        if (util::is_same<BuildT, Point>::value) {
            if (mVerbose==2) mTimer.restart("Count max points per voxel");
            BufT<uint32_t> maxPointsPerVoxelScratch(mStream, this->ref(), 1, nanovdb::cuda::noInit);
            uint32_t *d_maxPointsPerVoxel = maxPointsPerVoxelScratch.data(), maxPointsPerVoxel;
            CALL_CUBS(DeviceReduce::Max, mData.pointsPerVoxel, d_maxPointsPerVoxel, mData.voxelCount);
            cudaCheck(cudaMemcpyAsync(&maxPointsPerVoxel, d_maxPointsPerVoxel, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
            cudaCheck(cudaStreamSynchronize(mStream));
            maxPointsPerVoxelScratch.destroy(); d_maxPointsPerVoxel = nullptr;
            double dx = mData.map.getVoxelSize()[0];
            if (++iterCounter >= mMaxIterations || pointCount == 1u || math::Abs((int)maxPointsPerVoxel - (int)mMaxPointsPerVoxel) <= mTolerance) {
                mMaxPointsPerVoxel = maxPointsPerVoxel;
            } else {
                const Foo tmp{dx, maxPointsPerVoxel};
                if (maxPointsPerVoxel < mMaxPointsPerVoxel) {
                    if (min < tmp) min = tmp;
                } else if (max.density == 0 || tmp < max) {
                    max = tmp;
                }
                if (max.density) {
                    dx = (min.dx*(max.density - mMaxPointsPerVoxel) + max.dx*(mMaxPointsPerVoxel-min.density))/double(max.density-min.density);
                } else if (maxPointsPerVoxel > 1u) {
                    dx *= (mMaxPointsPerVoxel-1.0)/(maxPointsPerVoxel-1.0);
                } else {// maxPointsPerVoxel = 1 so increase dx significantly
                    dx *= 10.0;
                }
                if (mVerbose==2) printf("\ntarget density = %" PRIu32 ", current density = %" PRIu32 ", current dx = %f, next dx = %f\n", mMaxPointsPerVoxel, maxPointsPerVoxel, tmp.dx, dx);
                mData.map = Map(dx);
                // free before the next iteration reallocates, so peak device
                // memory matches the pre-loop behavior
                mKeysBuf.destroy();          mData.d_keys = nullptr;
                mIndxBuf.destroy();          mData.d_indx = nullptr;
                keysScratch.destroy();       d_keys = nullptr;
                mTileKeysBuf.destroy();      mData.d_tile_keys = nullptr;
                mPointsPerVoxelBuf.destroy();mData.pointsPerVoxel = nullptr;
                continue;
            }
        }
        break;
    }// while (true)
    if (iterCounter>1 && mVerbose) std::cerr << "Used " << iterCounter << " attempts to determine dx that produces a target dpoint denisty\n\n";

    if (mVerbose==2) mTimer.restart("Compute prefix sum of points per voxel");
    mPointsPerVoxelPrefixBuf = BufT<uint32_t>(mStream, this->ref(), mData.voxelCount, nanovdb::cuda::noInit);
    mData.pointsPerVoxelPrefix = mPointsPerVoxelPrefixBuf.data();
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerVoxel, mData.pointsPerVoxelPrefix, mData.voxelCount);

    mPointsPerLeafBuf = BufT<uint32_t>(mStream, this->ref(), mData.nodeCount[0], nanovdb::cuda::noInit);
    mData.pointsPerLeaf = mPointsPerLeafBuf.data();
    mLeafKeysBuf = BufT<uint64_t>(mStream, this->ref(), mData.nodeCount[0], nanovdb::cuda::noInit);
    mData.d_leaf_keys = mLeafKeysBuf.data();
    CALL_CUBS(DeviceRunLengthEncode::Encode, thrust::make_transform_iterator(mData.d_keys, ShiftRight<9>()), mData.d_leaf_keys, mData.pointsPerLeaf, runCountScratch.data(), pointCount);

    if constexpr(util::is_same<BuildT, Point>::value) {
        BufT<uint32_t> maxPointsPerLeafScratch(mStream, this->ref(), 1, nanovdb::cuda::noInit);
        uint32_t *d_maxPointsPerLeaf = maxPointsPerLeafScratch.data();
        CALL_CUBS(DeviceReduce::Max, mData.pointsPerLeaf, d_maxPointsPerLeaf, mData.nodeCount[0]);
        cudaCheck(cudaMemcpyAsync(&mMaxPointsPerLeaf, d_maxPointsPerLeaf, sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
        //printf("\n Leaf count = %u, max points per leaf = %u\n", mData.nodeCount[0], mMaxPointsPerLeaf);
        if (mMaxPointsPerLeaf > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Too many points per leaf: "+std::to_string(mMaxPointsPerLeaf));
        }
    }

    mPointsPerLeafPrefixBuf = BufT<uint32_t>(mStream, this->ref(), mData.nodeCount[0], nanovdb::cuda::noInit);
    mData.pointsPerLeafPrefix = mPointsPerLeafPrefixBuf.data();
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerLeaf, mData.pointsPerLeafPrefix, mData.nodeCount[0]);

    mLowerKeysBuf = BufT<uint64_t>(mStream, this->ref(), mData.nodeCount[1], nanovdb::cuda::noInit);
    mData.d_lower_keys = mLowerKeysBuf.data();
    CALL_CUBS(DeviceSelect::Unique, thrust::make_transform_iterator(mData.d_leaf_keys, ShiftRight<12>()), mData.d_lower_keys, runCountScratch.data(), mData.nodeCount[0]);// lower nodes

    if (mVerbose==2) mTimer.stop();

    //printf("Leaf count = %u, lower count = %u, upper count = %u\n", mData.nodeCount[0], mData.nodeCount[1], mData.nodeCount[2]);
}// PointsToGrid<BuildT, ResourceT>::countNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ResourceT>
template<typename PtrT, typename BufferT>
inline BufferT PointsToGrid<BuildT, ResourceT>::getBuffer(const PtrT, size_t pointCount, const BufferT &pool)
{
    auto sizeofPoint = [&]()->size_t{
        switch (mPointType){
        case PointType::PointID: return sizeof(uint32_t);
        case PointType::World64: return sizeof(Vec3d);
        case PointType::World32: return sizeof(Vec3f);
        case PointType::Grid64:  return sizeof(Vec3d);
        case PointType::Grid32:  return sizeof(Vec3f);
        case PointType::Voxel32: return sizeof(Vec3f);
        case PointType::Voxel16: return sizeof(Vec3u16);
        case PointType::Voxel8:  return sizeof(Vec3u8);
        case PointType::Default: return pointer_traits<PtrT>::element_size;
        default: return size_t(0);// PointType::Disable
        }
    };

    mData.grid  = 0;// grid is always stored at the start of the buffer!
    mData.tree  = NanoGrid<BuildT>::memUsage(); // grid ends and tree begins
    mData.root  = mData.tree  + NanoTree<BuildT>::memUsage(); // tree ends and root node begins
    mData.upper = mData.root  + NanoRoot<BuildT>::memUsage(mData.nodeCount[2]); // root node ends and upper internal nodes begin
    mData.lower = mData.upper + NanoUpper<BuildT>::memUsage()*mData.nodeCount[2]; // upper internal nodes ends and lower internal nodes begin
    mData.leaf  = mData.lower + NanoLower<BuildT>::memUsage()*mData.nodeCount[1]; // lower internal nodes ends and leaf nodes begin
    mData.meta  = mData.leaf  + NanoLeaf<BuildT>::DataType::memUsage()*mData.nodeCount[0];// leaf nodes end and blind meta data begins
    mData.blind = mData.meta  + sizeof(GridBlindMetaData)*int( mPointType!=PointType::Disable ); // meta data ends and blind data begins
    // padded like GridBlindMetaData::blindDataSize, so the grid size stays a multiple of
    // NANOVDB_DATA_ALIGNMENT and the next grid in a multi-grid buffer starts aligned
    mData.size  = mData.blind + math::AlignUp<NANOVDB_DATA_ALIGNMENT>(pointCount*sizeofPoint());// end of buffer

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = nanovdb::cuda::detail::createDeviceStorage<BufferT>(mData.size, &pool, device, mStream); // only allocate buffer on the device

    mData.d_bufferPtr = nanovdb::cuda::detail::deviceStorageData(buffer);
    if (mData.d_bufferPtr == nullptr) throw std::runtime_error("The grid buffer type produced no device-accessible memory");
    // Zero the whole grid buffer up front. This (a) makes the dense background
    // fills (upper/lower value tables, inactive leaf values - all zero in this
    // builder) redundant, so those kernels are skipped, and (b) makes the
    // output bit-deterministic: alignment padding and stats fields no longer
    // carry recycled pool bytes (which previously leaked into written files).
    cudaCheck(cudaMemsetAsync(mData.d_bufferPtr, 0, mData.size, mStream));
    cudaCheck(cudaMemcpyAsync(mDeviceData, &mData, sizeof(PointsToGridData<BuildT>), cudaMemcpyHostToDevice, mStream));// copy Data CPU -> GPU
    return buffer;
}// PointsToGrid<BuildT, ResourceT>::getBuffer

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT>
struct BuildGridTreeRootFunctor
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;

    __device__
    void operator()(size_t, PointsToGridData<BuildT> *d_data, PointType pointType, size_t pointCount) {
       // process Root
        auto &root = d_data->getRoot();
        root.mBBox = CoordBBox(); // init to empty
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = typename NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = typename NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = typename NanoRoot<BuildT>::FloatType(0);

        // process Tree
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        if (d_data->nodeCount[0]) {
            tree.setFirstNode(&d_data->getUpper(0));
            tree.setFirstNode(&d_data->getLower(0));
            tree.setFirstNode(&d_data->getLeaf(0));
        } else {// empty grid
            tree.template setFirstNode<NanoUpper<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLower<BuildT>>(nullptr);
            tree.template setFirstNode<NanoLeaf<BuildT>>(nullptr);
        }
        tree.mNodeCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = d_data->nodeCount[0];
        tree.mTileCount[2] = 0;
        tree.mTileCount[1] = 0;
        tree.mTileCount[0] = 0;
        tree.mVoxelCount = d_data->voxelCount;

        // process Grid
        auto &grid = d_data->getGrid();
        grid.init({GridFlags::HasBBox, GridFlags::IsBreadthFirst}, d_data->size, d_data->map, toGridType<BuildT>());
        if (d_data->nodeCount[0] == 0) grid.mFlags.setMaskOff(GridFlags::HasBBox);// an empty grid has no bbox
        grid.mChecksum = ~uint64_t(0);// set all bits on which means it's disabled
        grid.mBlindMetadataCount  = util::is_same<BuildT, Point>::value && pointType != PointType::Disable;
        grid.mBlindMetadataOffset = d_data->meta;
        if (pointType != PointType::Disable) {
            if (const uint32_t leafCount = tree.mNodeCount[0]) {
                grid.mData1 = d_data->pointsPerLeafPrefix[leafCount - 1] + d_data->pointsPerLeaf[leafCount - 1];
            }
            auto &meta = d_data->getMeta();
            meta.mDataOffset = sizeof(GridBlindMetaData);// blind data is placed right after this meta data
            meta.mValueCount = pointCount;
            // Blind meta data
            switch (pointType){
            case PointType::PointID:
                grid.mGridClass = GridClass::PointIndex;
                meta.mSemantic  = GridBlindDataSemantic::PointId;
                meta.mDataClass = GridBlindDataClass::IndexArray;
                meta.mDataType  = toGridType<uint32_t>();
                meta.mValueSize = sizeof(uint32_t);
                util::strcpy(meta.mName, "PointID: uint32_t indices to points");
                break;
            case PointType::World64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                util::strcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                break;
            case PointType::World32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                break;
            case PointType::Grid64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                util::strcpy(meta.mName, "Grid64: Vec3<double> point coordinates in grid space");
                break;
            case PointType::Grid32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "Grid32: Vec3<float> point coordinates in grid space");
                break;
            case PointType::Voxel32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                util::strcpy(meta.mName, "Voxel32: Vec3<float> point coordinates in voxel space");
                break;
            case PointType::Voxel16:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3u16>();
                meta.mValueSize = sizeof(Vec3u16);
                util::strcpy(meta.mName, "Voxel16: Vec3<uint16_t> point coordinates in voxel space");
                break;
            case PointType::Voxel8:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3u8>();
                meta.mValueSize = sizeof(Vec3u8);
                util::strcpy(meta.mName, "Voxel8: Vec3<uint8_t> point coordinates in voxel space");
                break;
            case PointType::Default:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = toGridType<Vec3T>();
                meta.mValueSize = sizeof(Vec3T);
                if constexpr(util::is_same<Vec3T, Vec3f>::value) {
                    util::strcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                } else if constexpr(util::is_same<Vec3T, Vec3d>::value){
                    util::strcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                } else {
                    printf("Error in PointsToGrid<BuildT, ResourceT>::processGridTreeRoot: expected Vec3T = Vec3f or Vec3d\n");
                }
                break;
            default:
                printf("Error in PointsToGrid<BuildT, ResourceT>::processGridTreeRoot: invalid pointType\n");
            }
        } else if constexpr(BuildTraits<BuildT>::is_onindex) {
            grid.mGridClass = GridClass::IndexGrid;
            if (d_data->nodeCount[0] == 0) grid.mData1 = 1u;// the background value only
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            grid.mData1 = 1u + 512u*d_data->nodeCount[0];
            grid.mGridClass = GridClass::IndexGrid;
        }
    }
};

template<typename BuildT, typename ResourceT>
template<typename PtrT>
inline void PointsToGrid<BuildT, ResourceT>::processGridTreeRoot(const PtrT points, size_t pointCount)
{
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, BuildGridTreeRootFunctor<BuildT, PtrT>(), mDeviceData, mPointType, pointCount);// lambdaKernel
    cudaCheckError();

    // Zero the name field, then copy only the actual string (if any). The
    // previous code copied MaxNameSize bytes from the std::string buffer
    // (whose .data() is never null, so the memset branch was dead), reading
    // up to 255 bytes past the allocation and leaking host heap contents
    // into the grid - nondeterministic output and a hygiene issue for files.
    char *dst = mData.getGrid().mGridName;
    cudaCheck(cudaMemsetAsync(dst, 0, GridData::MaxNameSize, mStream));
    if (!mGridName.empty()) {
        // Copy at most MaxNameSize-1 bytes so the memset's trailing '\0' always
        // survives; a name >= MaxNameSize is truncated, never left unterminated.
        const size_t nameSize = std::min<size_t>(mGridName.size(), GridData::MaxNameSize - 1);
        cudaCheck(cudaMemcpyAsync(dst, mGridName.c_str(), nameSize, cudaMemcpyHostToDevice, mStream));
    }
}// PointsToGrid<BuildT, ResourceT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct BuildUpperNodesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &root  = d_data->getRoot();
        auto &upper = d_data->getUpper(tid);
#if 1
        auto keyToCoord = [](uint64_t key)->nanovdb::Coord{
            static constexpr int64_t offset = 1 << 31;// max values of uint32_t is 2^31 - 1
            static constexpr uint64_t MASK = (1u << 21) - 1; // used to mask out 21 lower bits
            return nanovdb::Coord(int(int64_t(((key >> 42) & MASK) << 12) - offset),  // x are the upper 21 bits
                                  int(int64_t(((key >> 21) & MASK) << 12) - offset),  // y are the middle 21 bits
                                  int(int64_t(( key        & MASK) << 12) - offset)); // z are the lower 21 bits
        };
        const Coord ijk = keyToCoord(d_data->d_tile_keys[tid]);
#else
        const Coord ijk = typename NanoRoot<uint32_t>::KeyToCoord(d_data->d_tile_keys[tid]);
#endif
        root.tile(tid)->setChild(ijk, &upper, &root);
        upper.mBBox[0] = ijk;
        upper.mFlags = (uint64_t) GridFlags::HasBBox;
        upper.mValueMask.setOff();
        upper.mChildMask.setOff();
        upper.mMinimum = upper.mMaximum = typename NanoLower<BuildT>::ValueType(0);
        upper.mAverage = upper.mStdDevi = typename NanoLower<BuildT>::FloatType(0);
    }
};

template<typename BuildT>
struct SetUpperBackgroundValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &upper = d_data->getUpper(tid >> 15);
        upper.mTable[tid & 32767u].value = typename NanoUpper<BuildT>::ValueType(0);// background
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processUpperNodes()
{
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], BuildUpperNodesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    mTileKeysBuf.destroy(); mData.d_tile_keys = nullptr;

}// PointsToGrid<BuildT, ResourceT>::processUpperNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct BuildLowerNodesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &root  = d_data->getRoot();
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        const uint32_t upperOffset = lowerKey & 32767u;// (1 << 15) - 1 = 32767
        upper.mChildMask.setOnAtomic(upperOffset);
        auto &lower = d_data->getLower(tid);
        upper.setChild(upperOffset, &lower);
        lower.mBBox[0] = upper.offsetToGlobalCoord(upperOffset);
        lower.mFlags = (uint64_t) GridFlags::HasBBox;
        lower.mValueMask.setOff();
        lower.mChildMask.setOff();
        lower.mMinimum = lower.mMaximum = typename NanoLower<BuildT>::ValueType(0);// background;
        lower.mAverage = lower.mStdDevi = typename NanoLower<BuildT>::FloatType(0);
    }
};

template<typename BuildT>
struct SetLowerBackgroundValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &lower = d_data->getLower(tid >> 12);
        lower.mTable[tid & 4095u].value = typename NanoLower<BuildT>::ValueType(0);// background
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processLowerNodes()
{
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], BuildLowerNodesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

}// PointsToGrid<BuildT, ResourceT>::processLowerNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct ProcessLeafMetaDataFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data, uint8_t flags) {
        const uint64_t leafKey = d_data->d_leaf_keys[tid], tile_id = leafKey >> 27;
        auto &upper = d_data->getUpper(tile_id);
        const uint32_t lowerOffset = leafKey & 4095u, upperOffset = (leafKey >> 12) & 32767u;
        auto &lower = *upper.getChild(upperOffset);
        lower.mChildMask.setOnAtomic(lowerOffset);
        auto &leaf = d_data->getLeaf(tid);
        lower.setChild(lowerOffset, &leaf);
        leaf.mBBoxMin = lower.offsetToGlobalCoord(lowerOffset);
        leaf.mFlags = flags;
        auto &valueMask = leaf.mValueMask;
        valueMask.setOff();// initiate all bits to off

        if constexpr(util::is_same<Point, BuildT>::value) {
            leaf.mOffset = d_data->pointsPerLeafPrefix[tid];
            leaf.mPointCount = d_data->pointsPerLeaf[tid];
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            leaf.mOffset = tid*512u + 1u;// background is index 0
            leaf.mPrefixSum = 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mAverage = leaf.mStdDevi = typename NanoLeaf<BuildT>::FloatType(0);
            leaf.mMinimum = leaf.mMaximum = typename NanoLeaf<BuildT>::ValueType(0);
        }
    }
};

template<typename BuildT>
struct SetLeafActiveVoxelStateAndValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint32_t pointID  = d_data->pointsPerVoxelPrefix[tid];
        const uint64_t voxelKey = d_data->d_keys[pointID];
        auto &upper = d_data->getUpper(voxelKey >> 36);
        auto &lower = *upper.getChild((voxelKey >> 21) & 32767u);
        auto &leaf  = *lower.getChild((voxelKey >>  9) &  4095u);
        const uint32_t n = voxelKey & 511u;
        leaf.mValueMask.setOnAtomic(n);// <--- slow!
        if constexpr(util::is_same<Point, BuildT>::value) {
            leaf.mValues[n] = uint16_t(pointID + d_data->pointsPerVoxel[tid] - leaf.offset());
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = typename NanoLeaf<BuildT>::ValueType(1);// set value of active voxels that are not points (or index)
        }
    }
};

template<typename BuildT>
struct SetLeafInactiveVoxelValuesFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto &leaf = d_data->getLeaf(tid >> 9u);
        const uint32_t n = tid & 511u;
        if (leaf.mValueMask.isOn(n)) return;
        if constexpr(util::is_same<BuildT, Point>::value) {
            const uint32_t m = leaf.mValueMask.findPrev<true>(n - 1);
            leaf.mValues[n] = m < 512u ? leaf.mValues[m] : 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = typename NanoLeaf<BuildT>::ValueType(0);// value of inactive voxels
        }
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processLeafNodes(size_t pointCount)
{
    const uint8_t flags = (uint8_t) GridFlags::HasBBox;

    if (mVerbose==2) mTimer.start("process leaf meta data");
    // loop over leaf nodes and add it to its parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], ProcessLeafMetaDataFunctor<BuildT>(), mDeviceData, flags);
    cudaCheckError();

    if (mVerbose==2) mTimer.restart("set active voxel state and values");
    // loop over all active voxels and set LeafNode::mValueMask and LeafNode::mValues
    util::cuda::lambdaKernel<<<numBlocks(mData.voxelCount), mNumThreads, 0, mStream>>>(mData.voxelCount, SetLeafActiveVoxelStateAndValuesFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    mKeysBuf.destroy();                mData.d_keys = nullptr;
    mPointsPerVoxelBuf.destroy();      mData.pointsPerVoxel = nullptr;
    mPointsPerVoxelPrefixBuf.destroy();mData.pointsPerVoxelPrefix = nullptr;
    mPointsPerLeafPrefixBuf.destroy(); mData.pointsPerLeafPrefix = nullptr;
    mPointsPerLeafBuf.destroy();       mData.pointsPerLeaf = nullptr;

    // Inactive voxel values are zero for every build type except Point (which
    // copies the previous active value for rank queries); the zero cases are
    // covered by the buffer memset in getBuffer, and for index grids this
    // dense pass (nodeCount[0]<<9 threads) is a no-op.
    if constexpr(util::is_same<BuildT, Point>::value) {
        if (mVerbose==2) mTimer.restart("set inactive voxel values");
        const uint64_t denseVoxelCount = mData.nodeCount[0] << 9;
        util::cuda::lambdaKernel<<<numBlocks(denseVoxelCount), mNumThreads, 0, mStream>>>(denseVoxelCount, SetLeafInactiveVoxelValuesFunctor<BuildT>(), mDeviceData);
        cudaCheckError();
    }

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        if (mVerbose==2) mTimer.restart("prefix-sum for index grid");
        BufT<uint64_t> valueIndexScratch(mStream, this->ref(), mData.nodeCount[0], nanovdb::cuda::noInit);
        BufT<uint64_t> valueIndexPrefixScratch(mStream, this->ref(), mData.nodeCount[0], nanovdb::cuda::noInit);
        auto devValueIndex = valueIndexScratch.data();
        auto devValueIndexPrefix = valueIndexPrefixScratch.data();
        kernels::fillValueIndexKernel<BuildT><<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], 0, devValueIndex, mDeviceData);
        cudaCheckError();
        CALL_CUBS(DeviceScan::InclusiveSum, devValueIndex, devValueIndexPrefix, mData.nodeCount[0]);
        valueIndexScratch.destroy(); devValueIndex = nullptr;
        kernels::leafPrefixSumKernel<BuildT><<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], 0, devValueIndexPrefix, mDeviceData);
        cudaCheckError();
    }

    if (mVerbose==2) mTimer.stop();
}// PointsToGrid<BuildT, ResourceT>::processLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// The point-encode kernels operate on device data and are independent of the
// resource type; only the trailing d_indx deallocation routes through ResourceT.
// Dispatching the encode with if constexpr keeps it reachable for any ResourceT.
template<typename BuildT, typename ResourceT>
template<typename PtrT>
inline void PointsToGrid<BuildT, ResourceT>::processPoints(const PtrT points, size_t pointCount)
{
    if constexpr(util::is_same<BuildT, Point>::value) this->encodePoints(points, pointCount);
    mIndxBuf.destroy(); mData.d_indx = nullptr;
}// PointsToGrid<BuildT, ResourceT>::processPoints

template<typename BuildT, typename ResourceT>
template<typename PtrT>
inline void PointsToGrid<BuildT, ResourceT>::encodePoints(const PtrT points, size_t pointCount)
{
    switch (mPointType){
    case PointType::Disable:
        throw std::runtime_error("PointsToGrid<Point, ResourceT>::encodePoints: mPointType == PointType::Disable\n");
    case PointType::PointID:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<uint32_t>(tid) = d_data->d_indx[tid];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World64:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3d>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3f>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid64:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3d>(tid) = d_data->map.applyInverseMap(points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<Vec3f>(tid) = d_data->map.applyInverseMapF(points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel32:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3f>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel16:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u16>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel8:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u8>(tid), points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Default:
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, [=] __device__(size_t tid, PointsToGridData<Point> *d_data) {
            d_data->template getPoint<typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type>(tid) = points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    default:
        printf("Internal error in PointsToGrid<Point, ResourceT>::encodePoints\n");
    }
}// PointsToGrid<BuildT, ResourceT>::encodePoints

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
struct ResetLowerNodeBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getLower(tid).mBBox = CoordBBox();
    }
};

template<typename BuildT>
struct UpdateAndPropagateLeafBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint64_t leafKey = d_data->d_leaf_keys[tid];
        auto &upper = d_data->getUpper(leafKey >> 27);
        auto &lower = *upper.getChild((leafKey >> 12) & 32767u);
        auto &leaf = d_data->getLeaf(tid);
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }
};

template<typename BuildT>
struct ResetUpperNodeBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getUpper(tid).mBBox = CoordBBox();
    }
};

template<typename BuildT>
struct PropagateLowerBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox());
    }
};

template<typename BuildT>
struct PropagateUpperBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }
};

template<typename BuildT>
struct UpdateRootWorldBBoxFunctor
{
    __device__
    void operator()(size_t tid, PointsToGridData<BuildT> *d_data) {
        auto BBox = d_data->getRoot().mBBox;
        BBox.max() += 1;
        d_data->getGrid().mFlags.setMaskOn(GridFlags::HasBBox);
        d_data->getGrid().mWorldBBox = BBox.transform(d_data->map);
    }
};

template<typename BuildT, typename ResourceT>
inline void PointsToGrid<BuildT, ResourceT>::processBBox()
{
    // reset bbox in lower nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], ResetLowerNodeBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // update and propagate bbox from leaf -> lower/parent nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads, 0, mStream>>>(mData.nodeCount[0], UpdateAndPropagateLeafBBoxFunctor<BuildT>(), mDeviceData);
    mLeafKeysBuf.destroy(); mData.d_leaf_keys = nullptr;
    cudaCheckError();

    // reset bbox in upper nodes
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], ResetUpperNodeBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads, 0, mStream>>>(mData.nodeCount[1], PropagateLowerBBoxFunctor<BuildT>(), mDeviceData);
    mLowerKeysBuf.destroy(); mData.d_lower_keys = nullptr;
    cudaCheckError();

    // propagate bbox from upper -> root/parent node
    util::cuda::lambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads, 0, mStream>>>(mData.nodeCount[2], PropagateUpperBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();

    // update the world-bbox in the root node
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1, UpdateRootWorldBBoxFunctor<BuildT>(), mDeviceData);
    cudaCheckError();
}// PointsToGrid<BuildT, ResourceT>::processBBox

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace detail {

/// @brief Index of the grid that owns item @a i, given @a gridCount+1 non-decreasing @a starts:
///        the last g with starts[g] <= i, so empty grids are skipped
__hostdev__ inline uint32_t gridOf(const uint64_t *starts, uint32_t gridCount, uint64_t i)
{
    uint32_t lo = 0u, hi = gridCount;
    while (hi - lo > 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (starts[mid] <= i) lo = mid; else hi = mid;
    }
    return lo;
}

/// @brief Fancy pointer over several device arrays laid end to end: element i lives in the
///        array whose range of @a offsets holds it
template<typename PtrT>
struct SegmentedPtr {
    const PtrT     *ptrs;
    const uint64_t *offsets;
    uint32_t        count;
    __hostdev__ decltype(auto) operator[](size_t i) const {
        const uint32_t s = gridOf(offsets, count, i);
        return ptrs[s][i - offsets[s]];
    }
    __hostdev__ decltype(auto) operator*() const { return *ptrs[0]; }// only its type is used
};

/// @brief Start of every grid's points, voxels and nodes in a batched build: gridCount+1
///        entries per level, the last one the level's total
struct BatchStarts {
    const uint64_t *point, *voxel, *upper, *lower, *leaf;
    uint32_t gridCount;
};

/// @brief Runs a single-grid build functor on item @a tid of a batch, as item tid - starts[g]
///        of the grid g that owns it
template<typename FunctorT>
struct PerGridFunctor {
    template<typename BuildT, typename... ArgsT>
    __device__ void operator()(size_t tid, const uint64_t *starts, uint32_t gridCount, PointsToGridData<BuildT> *d_data, ArgsT... args) {
        const uint32_t g = gridOf(starts, gridCount, tid);
        FunctorT()(tid - starts[g], d_data + g, args...);
    }
};

/// @brief Writes the grid of each point, taken in the order of @a d_indx
struct PointGridFunctor {
    __device__ void operator()(size_t tid, const uint64_t *offsets, uint32_t gridCount, const uint32_t *d_indx, uint32_t *d_grids) {
        d_grids[tid] = gridOf(offsets, gridCount, d_indx[tid]);
    }
};

/// @brief TileStartFlag that also starts a tile at the first point of every grid
struct BatchTileStartFlag {
    const uint64_t *keys, *offsets;
    uint32_t gridCount;
    __hostdev__ uint32_t operator()(uint32_t i) const {
        if (i == 0 || keys[i] != keys[i - 1]) return 1u;
        return offsets[gridOf(offsets, gridCount, i)] == i;
    }
};

/// @brief Offset @a g clamped to the point count, so offsets the host has not yet validated
///        cannot send a reduction out of bounds
struct ClampedOffset {
    const uint64_t *offsets;
    uint64_t        count;
    __hostdev__ uint64_t operator()(uint32_t g) const { return offsets[g] < count ? offsets[g] : count; }
};

/// @brief Moves batch-wide tile indices to per-grid ones: key i, owned by the grid whose
///        range of @a starts holds it, drops that grid's first tile index shifted by @a shift
struct RebaseTileFunctor {
    __device__ void operator()(size_t tid, const uint64_t *starts, uint32_t gridCount, const uint64_t *tileStarts, int shift, uint64_t *keys) {
        keys[tid] -= tileStarts[gridOf(starts, gridCount, tid)] << shift;
    }
};

/// @brief Moves the batch-wide point offsets in pointsPerVoxelPrefix to per-grid ones
struct RebasePointPrefixFunctor {
    __device__ void operator()(size_t tid, BatchStarts starts, uint32_t *prefix) {
        prefix[tid] -= uint32_t(starts.point[gridOf(starts.voxel, starts.gridCount, tid)]);
    }
};

/// @brief Writes the grid, tree and root headers of grid @a g and numbers it in the batch
template<typename BuildT, typename PtrT>
struct BuildBatchGridTreeRootFunctor {
    __device__ void operator()(size_t g, PointsToGridData<BuildT> *d_data, const uint64_t *pointStarts, uint32_t gridCount) {
        BuildGridTreeRootFunctor<BuildT, PtrT>()(0, d_data + g, PointType::Disable, pointStarts[g + 1] - pointStarts[g]);
        auto &grid = d_data[g].getGrid();
        grid.mGridIndex = uint32_t(g);
        grid.mGridCount = gridCount;
    }
};

/// @brief Sets the world bbox of grid @a g, unless it has no voxels
template<typename BuildT>
struct UpdateBatchRootWorldBBoxFunctor {
    __device__ void operator()(size_t g, PointsToGridData<BuildT> *d_data) {
        if (d_data[g].nodeCount[0]) UpdateRootWorldBBoxFunctor<BuildT>()(0, d_data + g);
    }
};

/// @brief Writes the active voxel count of every leaf in the batch
template<typename BuildT>
struct BatchLeafValueCountFunctor {
    __device__ void operator()(size_t tid, const uint64_t *leafStarts, uint32_t gridCount, const PointsToGridData<BuildT> *d_data, uint64_t *counts) {
        const uint32_t g = gridOf(leafStarts, gridCount, tid);
        counts[tid] = d_data[g].getLeaf(tid - leafStarts[g]).mValueMask.countOn();
    }
};

/// @brief kernels::leafPrefixSumKernel for a batch: @a prefix is the inclusive sum of the
///        active voxel counts over all leaves, so each grid subtracts the sum before its first leaf
template<typename BuildT>
struct BatchLeafPrefixSumFunctor {
    __device__ void operator()(size_t tid, const uint64_t *leafStarts, uint32_t gridCount, PointsToGridData<BuildT> *d_data, const uint64_t *prefix) {
        const uint32_t g = gridOf(leafStarts, gridCount, tid);
        const uint64_t first = leafStarts[g], base = first ? prefix[first - 1] : 0u;
        auto &data = d_data[g];
        auto &leaf = data.getLeaf(tid - first);
        const uint64_t *w = leaf.mValueMask.words();
        uint64_t &prefixSum = leaf.mPrefixSum, sum = util::countOn(*w++);
        prefixSum = sum;
        for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
            sum += util::countOn(*w++);
            prefixSum |= sum << n;// each pre-fixed sum is encoded in 9 bits
        }
        if (tid == first) {
            const uint64_t voxelCount = prefix[first + data.nodeCount[0] - 1] - base;
            data.getGrid().mData1 = 1u + voxelCount;// set total count
            data.getTree().mVoxelCount = voxelCount;
            leaf.mOffset = 1u;// background is index 0
        } else {
            leaf.mOffset = 1u + prefix[tid - 1] - base;
        }
    }
};

}// namespace detail

/// @brief Builds one NanoGrid<BuildT> per segment of a voxel list, into one buffer, with the
///        single-grid build functors. Every array is sorted batch-wide with the grid as the
///        primary key, so each grid's points, voxels and nodes are contiguous; each grid's
///        PointsToGridData then views its own ranges, with tile indices that restart at 0.
template<typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class VoxelsToGridBatch
{
    static_assert(!util::is_same<BuildT, Point>::value, "VoxelsToGridBatch builds grids from voxels, not points");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "VoxelsToGridBatch allocates stream-ordered scratch and requires an AsyncResource");
public:
    /// @param maps One map per output grid
    VoxelsToGridBatch(std::vector<Map> maps, cudaStream_t stream = 0, ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mStream(stream)
        , mResource(&resource)
        , mGridCount(static_cast<uint32_t>(maps.size()))
        , mData(maps.size())
        , mDeviceDataBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), maps.size(), nanovdb::cuda::noInit)
        , mStartsBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 5 * (maps.size() + 1), nanovdb::cuda::noInit)
        , mKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mTileKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mLeafKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mLowerKeysBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerVoxelBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mPointsPerVoxelPrefixBuf(stream, nanovdb::cuda::detail::borrowInternal(resource), 0, nanovdb::cuda::noInit)
        , mTempDevicePool(resource)
    {
        if (maps.empty() || maps.size() > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("VoxelsToGridBatch: the grid count must be between 1 and 2^32-1");
        for (size_t g = 0; g < maps.size(); ++g) {
            detail::checkMap(maps[g], "VoxelsToGridBatch");
            mData[g] = PointsToGridData<BuildT>{};
            mData[g].map = maps[g];
        }
    }

    /// @brief Builds grid g from voxels[d_offsets[g]] up to voxels[d_offsets[g+1]-1]
    /// @param voxels device pointer to the voxel coordinates in index space
    /// @param voxelCount total number of voxels
    /// @param d_offsets device array of gridCount+1 non-decreasing offsets, from 0 to @c voxelCount
    /// @param pool prototype buffer whose resource (or create() pool) allocates the output
    /// @throw std::invalid_argument if the offsets are not as described
    template<typename PtrT, typename BufferT = nanovdb::cuda::DualDeviceBuffer>
    GridHandle<BufferT> getHandle(const PtrT voxels, size_t voxelCount, const uint64_t *d_offsets, const BufferT &pool = BufferT());

private:
    static constexpr unsigned int mNumThreads = 128;
    static unsigned int numBlocks(uint64_t n) {return static_cast<unsigned int>((n + mNumThreads - 1) / mNumThreads);}

    template<typename T>
    using BufT = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::detail::borrowInternal(*mResource); }
    template<typename T>
    static T* at(BufT<T> &buf, uint64_t i) { return buf.empty() ? nullptr : buf.data() + i; }

    /// @brief Sorts the voxels, counts every grid's nodes with one readback, and points each
    ///        grid's PointsToGridData at its ranges of the sorted arrays
    template<typename PtrT>
    void countNodes(const PtrT voxels, size_t voxelCount, const uint64_t *d_offsets);

    /// @brief Allocates and zeroes the output, lays the grids out back to back and uploads the per-grid data
    template<typename BufferT>
    BufferT getBuffer(const BufferT &pool, std::vector<GridHandleMetaData> &meta);

    detail::BatchStarts starts() const {
        const uint64_t *s = mStartsBuf.data(), n = mGridCount + 1u;
        return {s, s + n, s + 2*n, s + 3*n, s + 4*n, mGridCount};
    }

    cudaStream_t                          mStream{0};
    ResourceT*                            mResource;// non-owning
    uint32_t                              mGridCount;
    std::vector<PointsToGridData<BuildT>> mData;// host copy of the per-grid data
    std::vector<uint64_t>                 mStarts;// host copy of BatchStarts: point, voxel, upper, lower and leaf rows
    BufT<PointsToGridData<BuildT>>        mDeviceDataBuf;
    BufT<uint64_t>                        mStartsBuf, mKeysBuf, mTileKeysBuf, mLeafKeysBuf, mLowerKeysBuf;
    BufT<uint32_t>                        mPointsPerVoxelBuf, mPointsPerVoxelPrefixBuf;
    nanovdb::cuda::TempPool<ResourceT>    mTempDevicePool;
};// tools::cuda::VoxelsToGridBatch<BuildT, ResourceT>

template<typename BuildT, typename ResourceT>
template<typename PtrT>
void VoxelsToGridBatch<BuildT, ResourceT>::countNodes(const PtrT voxels, size_t voxelCount, const uint64_t *d_offsets)
{
    using Vec3T = typename util::remove_const<typename pointer_traits<PtrT>::element_type>::type;
    static_assert(util::is_same<Vec3T, Coord, Vec3f, Vec3d>::value, "Voxel coordinates should be represented as Coord, Vec3f or Vec3d");
    if (voxelCount > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("VoxelsToGridBatch: more than 2^32-1 voxels");
    const uint32_t gridCount = mGridCount, pointCount = static_cast<uint32_t>(voxelCount);
    PointsToGridData<BuildT> *d_data = mDeviceDataBuf.data();

    std::vector<detail::NodeCounts> counts(gridCount, detail::NodeCounts{0u, 0u, 0u, 0u});
    std::vector<uint64_t> offsets(gridCount + 1u, 0u);
    BufT<uint64_t> keysScratch(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
    if (pointCount > 0) {
        // The single-grid sorts, by in-tile offset and then stably by tile, followed by a stable
        // sort by grid when there is more than one, which leaves the points in (grid, voxel) order.
        BufT<uint32_t> indx(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        BufT<uint32_t> indxScratch(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        mKeysBuf = BufT<uint64_t>(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
        uint64_t *d_keys = mKeysBuf.data();
        const auto pointIndex = thrust::make_counting_iterator<uint32_t>(0);

        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::TileOffsetKeyFunctor<BuildT, PtrT>(), d_data, voxels, keysScratch.data(), indxScratch.data());
        cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortPairs, keysScratch.data(), d_keys, indxScratch.data(), indx.data(), pointCount, 0, 36);// 15+12+9 bits
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::PermutedTileKeyFunctor<BuildT, PtrT>(), d_data, voxels, indx.data(), keysScratch.data());
        cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortPairs, keysScratch.data(), d_keys, indx.data(), indxScratch.data(), pointCount, 0, 63);// stable, so each tile stays in voxel order
        indx.swap(indxScratch);

        if (gridCount > 1u) {
            int gridBits = 0;
            while ((uint64_t(1) << gridBits) < gridCount) ++gridBits;
            BufT<uint32_t> grids(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
            BufT<uint32_t> gridsScratch(mStream, this->ref(), pointCount, nanovdb::cuda::noInit);
            util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::PointGridFunctor(), d_offsets, gridCount, indx.data(), gridsScratch.data());
            cudaCheckError();
            CALL_CUBS(DeviceRadixSort::SortPairs, gridsScratch.data(), grids.data(), indx.data(), indxScratch.data(), pointCount, 0, gridBits);// stable, so each grid stays in voxel order
            indx.swap(indxScratch);
            util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::PermutedTileKeyFunctor<BuildT, PtrT>(), d_data, voxels, indx.data(), d_keys);
            cudaCheckError();
        }

        // Tile ranks run across the whole batch, and a grid's first point always starts a tile,
        // so voxel keys differ across grid boundaries and the run-length encodings below never
        // merge two grids.
        uint32_t *d_tileRank = indxScratch.data();
        if (gridCount > 1u) {
            CALL_CUBS(DeviceScan::InclusiveSum, thrust::make_transform_iterator(pointIndex, detail::BatchTileStartFlag{d_keys, d_offsets, gridCount}), d_tileRank, pointCount);
        } else {
            CALL_CUBS(DeviceScan::InclusiveSum, thrust::make_transform_iterator(pointIndex, detail::TileStartFlag{d_keys}), d_tileRank, pointCount);
        }
        util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::RankedVoxelKeyFunctor<BuildT, PtrT>(), d_data, voxels, indx.data(), d_tileRank, d_keys, keysScratch.data());
        cudaCheckError();
        indxScratch.destroy();
        indx.destroy();

        // One readback: every grid's node and voxel counts, and the offsets that delimit the grids
        BufT<detail::NodeCounts> countsScratch(mStream, this->ref(), gridCount, nanovdb::cuda::noInit);
        const auto segmentBegin = thrust::make_transform_iterator(thrust::make_counting_iterator<uint32_t>(0), detail::ClampedOffset{d_offsets, pointCount});
        CALL_CUBS(DeviceSegmentedReduce::Reduce, thrust::make_transform_iterator(pointIndex, detail::NodeStarts{d_keys}), countsScratch.data(),
                  gridCount, segmentBegin, segmentBegin + 1, detail::SumNodeCounts(), detail::NodeCounts{0u, 0u, 0u, 0u});
        cudaCheck(cudaMemcpyAsync(counts.data(), countsScratch.data(), gridCount * sizeof(detail::NodeCounts), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaMemcpyAsync(offsets.data(), d_offsets, (gridCount + 1u) * sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
        cudaCheck(cudaStreamSynchronize(mStream));
    }
    if (offsets[0] != 0u || offsets[gridCount] != voxelCount)
        throw std::invalid_argument("VoxelsToGridBatch: the offsets must run from 0 to the voxel count");
    for (uint32_t g = 0; g < gridCount; ++g) {
        if (offsets[g + 1] < offsets[g]) throw std::invalid_argument("VoxelsToGridBatch: the offsets must not decrease");
    }

    // Starts of every level, from which each grid's data views its ranges
    const uint64_t n = gridCount + 1u;
    mStarts.assign(5 * n, 0u);
    uint64_t *point = mStarts.data(), *voxel = point + n, *upper = voxel + n, *lower = upper + n, *leaf = lower + n;
    for (uint32_t g = 0; g < gridCount; ++g) {
        point[g + 1] = offsets[g + 1];
        voxel[g + 1] = voxel[g] + counts[g].voxel;
        upper[g + 1] = upper[g] + counts[g].upper;
        lower[g + 1] = lower[g] + counts[g].lower;
        leaf[g + 1]  = leaf[g]  + counts[g].leaf;
    }
    cudaCheck(cudaMemcpyAsync(mStartsBuf.data(), mStarts.data(), mStarts.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, mStream));

    if (pointCount > 0) {
        mTileKeysBuf = BufT<uint64_t>(mStream, this->ref(), upper[gridCount], nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(mTileKeysBuf.data(), keysScratch.data(), upper[gridCount] * sizeof(uint64_t), cudaMemcpyDeviceToDevice, mStream));

        BufT<uint32_t> runCountScratch(mStream, this->ref(), 1, nanovdb::cuda::noInit);
        mPointsPerVoxelBuf = BufT<uint32_t>(mStream, this->ref(), voxel[gridCount], nanovdb::cuda::noInit);
        CALL_CUBS(DeviceRunLengthEncode::Encode, mKeysBuf.data(), keysScratch.data(), mPointsPerVoxelBuf.data(), runCountScratch.data(), pointCount);
        mPointsPerVoxelPrefixBuf = BufT<uint32_t>(mStream, this->ref(), voxel[gridCount], nanovdb::cuda::noInit);
        CALL_CUBS(DeviceScan::ExclusiveSum, mPointsPerVoxelBuf.data(), mPointsPerVoxelPrefixBuf.data(), voxel[gridCount]);
        mLeafKeysBuf = BufT<uint64_t>(mStream, this->ref(), leaf[gridCount], nanovdb::cuda::noInit);
        CALL_CUBS(DeviceSelect::Unique, thrust::make_transform_iterator(mKeysBuf.data(), ShiftRight<9>()), mLeafKeysBuf.data(), runCountScratch.data(), pointCount);
        mLowerKeysBuf = BufT<uint64_t>(mStream, this->ref(), lower[gridCount], nanovdb::cuda::noInit);
        CALL_CUBS(DeviceSelect::Unique, thrust::make_transform_iterator(mLeafKeysBuf.data(), ShiftRight<12>()), mLowerKeysBuf.data(), runCountScratch.data(), leaf[gridCount]);
        keysScratch.destroy();

        if (gridCount > 1u) {// per-grid tile indices and point offsets, as a single-grid build has them
            const detail::BatchStarts s = this->starts();
            util::cuda::lambdaKernel<<<numBlocks(pointCount), mNumThreads, 0, mStream>>>(pointCount, detail::RebaseTileFunctor(), s.point, gridCount, s.upper, 36, mKeysBuf.data());
            util::cuda::lambdaKernel<<<numBlocks(leaf[gridCount]), mNumThreads, 0, mStream>>>(leaf[gridCount], detail::RebaseTileFunctor(), s.leaf, gridCount, s.upper, 27, mLeafKeysBuf.data());
            util::cuda::lambdaKernel<<<numBlocks(lower[gridCount]), mNumThreads, 0, mStream>>>(lower[gridCount], detail::RebaseTileFunctor(), s.lower, gridCount, s.upper, 15, mLowerKeysBuf.data());
            util::cuda::lambdaKernel<<<numBlocks(voxel[gridCount]), mNumThreads, 0, mStream>>>(voxel[gridCount], detail::RebasePointPrefixFunctor(), s, mPointsPerVoxelPrefixBuf.data());
            cudaCheckError();
        }
    }

    for (uint32_t g = 0; g < gridCount; ++g) {
        auto &data = mData[g];
        data.nodeCount[0] = counts[g].leaf;
        data.nodeCount[1] = counts[g].lower;
        data.nodeCount[2] = counts[g].upper;
        data.voxelCount   = counts[g].voxel;
        data.d_keys               = at(mKeysBuf, point[g]);
        data.d_tile_keys          = at(mTileKeysBuf, upper[g]);
        data.d_lower_keys         = at(mLowerKeysBuf, lower[g]);
        data.d_leaf_keys          = at(mLeafKeysBuf, leaf[g]);
        data.pointsPerVoxel       = at(mPointsPerVoxelBuf, voxel[g]);
        data.pointsPerVoxelPrefix = at(mPointsPerVoxelPrefixBuf, voxel[g]);
    }
}// VoxelsToGridBatch<BuildT, ResourceT>::countNodes

template<typename BuildT, typename ResourceT>
template<typename BufferT>
BufferT VoxelsToGridBatch<BuildT, ResourceT>::getBuffer(const BufferT &pool, std::vector<GridHandleMetaData> &meta)
{
    meta.resize(mGridCount);
    uint64_t size = 0u;
    for (uint32_t g = 0; g < mGridCount; ++g) {// each grid laid out as PointsToGrid::getBuffer lays out a grid without points
        auto &data = mData[g];
        data.grid  = 0;
        data.tree  = NanoGrid<BuildT>::memUsage();
        data.root  = data.tree  + NanoTree<BuildT>::memUsage();
        data.upper = data.root  + NanoRoot<BuildT>::memUsage(data.nodeCount[2]);
        data.lower = data.upper + NanoUpper<BuildT>::memUsage()*data.nodeCount[2];
        data.leaf  = data.lower + NanoLower<BuildT>::memUsage()*data.nodeCount[1];
        data.meta  = data.leaf  + NanoLeaf<BuildT>::DataType::memUsage()*data.nodeCount[0];
        data.blind = data.meta;
        data.size  = data.blind;
        meta[g] = GridHandleMetaData{size, data.size, toGridType<BuildT>()};
        size += data.size;
    }

    int device = 0;
    cudaGetDevice(&device);
    auto buffer = nanovdb::cuda::detail::createDeviceStorage<BufferT>(size, &pool, device, mStream);
    auto *d_buffer = static_cast<uint8_t*>(nanovdb::cuda::detail::deviceStorageData(buffer));
    if (d_buffer == nullptr) throw std::runtime_error("The grid buffer type produced no device-accessible memory");
    cudaCheck(cudaMemsetAsync(d_buffer, 0, size, mStream));// as PointsToGrid::getBuffer: background values and padding are zero
    for (uint32_t g = 0; g < mGridCount; ++g) mData[g].d_bufferPtr = d_buffer + meta[g].offset;
    cudaCheck(cudaMemcpyAsync(mDeviceDataBuf.data(), mData.data(), mGridCount * sizeof(PointsToGridData<BuildT>), cudaMemcpyHostToDevice, mStream));
    return buffer;
}// VoxelsToGridBatch<BuildT, ResourceT>::getBuffer

template<typename BuildT, typename ResourceT>
template<typename PtrT, typename BufferT>
GridHandle<BufferT> VoxelsToGridBatch<BuildT, ResourceT>::getHandle(const PtrT voxels, size_t voxelCount, const uint64_t *d_offsets, const BufferT &pool)
{
    this->countNodes(voxels, voxelCount, d_offsets);
    std::vector<GridHandleMetaData> meta;
    auto buffer = this->getBuffer(pool, meta);

    const uint32_t gridCount = mGridCount;
    const detail::BatchStarts s = this->starts();
    auto total = [&](int row) { return mStarts[row * (gridCount + 1u) + gridCount]; };// rows as in BatchStarts
    const uint64_t voxelTotal = total(1), upperCount = total(2), lowerCount = total(3), leafCount = total(4);
    PointsToGridData<BuildT> *d_data = mDeviceDataBuf.data();

    util::cuda::lambdaKernel<<<numBlocks(gridCount), mNumThreads, 0, mStream>>>(gridCount, detail::BuildBatchGridTreeRootFunctor<BuildT, PtrT>(), d_data, s.point, gridCount);
    cudaCheckError();

    if (leafCount > 0) {// no voxels in any grid: the headers alone form valid empty grids
        util::cuda::lambdaKernel<<<numBlocks(upperCount), mNumThreads, 0, mStream>>>(upperCount, detail::PerGridFunctor<BuildUpperNodesFunctor<BuildT>>(), s.upper, gridCount, d_data);
        cudaCheckError();
        mTileKeysBuf.destroy();
        util::cuda::lambdaKernel<<<numBlocks(lowerCount), mNumThreads, 0, mStream>>>(lowerCount, detail::PerGridFunctor<BuildLowerNodesFunctor<BuildT>>(), s.lower, gridCount, d_data);
        cudaCheckError();
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(leafCount, detail::PerGridFunctor<ProcessLeafMetaDataFunctor<BuildT>>(), s.leaf, gridCount, d_data, (uint8_t)GridFlags::HasBBox);
        cudaCheckError();
        util::cuda::lambdaKernel<<<numBlocks(voxelTotal), mNumThreads, 0, mStream>>>(voxelTotal, detail::PerGridFunctor<SetLeafActiveVoxelStateAndValuesFunctor<BuildT>>(), s.voxel, gridCount, d_data);
        cudaCheckError();
        mKeysBuf.destroy();
        mPointsPerVoxelBuf.destroy();
        mPointsPerVoxelPrefixBuf.destroy();

        if constexpr(BuildTraits<BuildT>::is_onindex) {
            BufT<uint64_t> valueCount(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
            BufT<uint64_t> valuePrefix(mStream, this->ref(), leafCount, nanovdb::cuda::noInit);
            util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(leafCount, detail::BatchLeafValueCountFunctor<BuildT>(), s.leaf, gridCount, d_data, valueCount.data());
            cudaCheckError();
            CALL_CUBS(DeviceScan::InclusiveSum, valueCount.data(), valuePrefix.data(), leafCount);
            util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(leafCount, detail::BatchLeafPrefixSumFunctor<BuildT>(), s.leaf, gridCount, d_data, valuePrefix.data());
            cudaCheckError();
        }

        util::cuda::lambdaKernel<<<numBlocks(lowerCount), mNumThreads, 0, mStream>>>(lowerCount, detail::PerGridFunctor<ResetLowerNodeBBoxFunctor<BuildT>>(), s.lower, gridCount, d_data);
        util::cuda::lambdaKernel<<<numBlocks(leafCount), mNumThreads, 0, mStream>>>(leafCount, detail::PerGridFunctor<UpdateAndPropagateLeafBBoxFunctor<BuildT>>(), s.leaf, gridCount, d_data);
        util::cuda::lambdaKernel<<<numBlocks(upperCount), mNumThreads, 0, mStream>>>(upperCount, detail::PerGridFunctor<ResetUpperNodeBBoxFunctor<BuildT>>(), s.upper, gridCount, d_data);
        util::cuda::lambdaKernel<<<numBlocks(lowerCount), mNumThreads, 0, mStream>>>(lowerCount, detail::PerGridFunctor<PropagateLowerBBoxFunctor<BuildT>>(), s.lower, gridCount, d_data);
        util::cuda::lambdaKernel<<<numBlocks(upperCount), mNumThreads, 0, mStream>>>(upperCount, detail::PerGridFunctor<PropagateUpperBBoxFunctor<BuildT>>(), s.upper, gridCount, d_data);
        util::cuda::lambdaKernel<<<numBlocks(gridCount), mNumThreads, 0, mStream>>>(gridCount, detail::UpdateBatchRootWorldBBoxFunctor<BuildT>(), d_data);
        cudaCheckError();
    }
    mLeafKeysBuf.destroy();
    mLowerKeysBuf.destroy();

    cudaCheck(cudaStreamSynchronize(mStream));
    return GridHandle<BufferT>::adopt(std::move(buffer), std::move(meta));
}// VoxelsToGridBatch<BuildT, ResourceT>::getHandle

// Undefine utility macro for cub functions
#ifdef CALL_CUBS
#undef CALL_CUBS
#endif

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace detail {

/// @brief Merges the per-set handles built by the point-set overloads,
///        allocating the result through @c buffer for either buffer family.
template<typename BufferT>
GridHandle<BufferT> mergeBuiltHandles(const std::vector<GridHandle<BufferT>> &handles, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(BufferHasDeviceSingle<BufferT>::value || BufferTraits<BufferT>::hasDeviceDual,
                  "the point-set overloads build into a device buffer");
    // explicit template arguments: MSVC cannot deduce the template-template VectorT
    if constexpr (BufferHasDeviceSingle<BufferT>::value) {// merging no handles gives an empty handle over the prototype's resource
        return nanovdb::cuda::mergeGridHandles<BufferT, std::vector>(handles, stream, &buffer);
    } else {
        if (handles.empty()) return GridHandle<BufferT>();// no sets: an empty handle
        return nanovdb::cuda::detail::mergeDualGridHandles<BufferT, std::vector>(handles, &buffer, stream);
    }
}

/// @brief Builds one grid per voxel set in one VoxelsToGridBatch pass, reading the sets in place
///        through a SegmentedPtr; Point grids are built one set at a time and merged. An empty
///        @a sets gives an empty handle over @a buffer's resource.
template<typename BuildT, typename PtrT, typename ResourceT, typename BufferT>
GridHandle<BufferT> voxelSetsToGrid(const std::vector<std::tuple<const PtrT, size_t, double>> &sets, const BufferT &buffer,
                                    cudaStream_t stream, ResourceT &resource)
{
    if constexpr(util::is_same<BuildT, Point>::value) {
        std::vector<GridHandle<BufferT>> handles;
        for (auto &s : sets) {
            PointsToGrid<BuildT, ResourceT> converter(std::get<2>(s), Vec3d(0.0), stream, resource);
            handles.push_back(converter.getHandle(std::get<0>(s), std::get<1>(s), buffer));
        }
        return mergeBuiltHandles(handles, buffer, stream);
    } else {
        if (sets.empty()) return mergeBuiltHandles(std::vector<GridHandle<BufferT>>(), buffer, stream);
        std::vector<Map> maps;
        std::vector<PtrT> ptrs;
        std::vector<uint64_t> offsets(1, 0u);
        for (auto &s : sets) {
            maps.emplace_back(checkVoxelSize(std::get<2>(s), "voxelsToGrid"), Vec3d(0.0));
            ptrs.push_back(std::get<0>(s));
            offsets.push_back(offsets.back() + std::get<1>(s));
        }
        auto ref = nanovdb::cuda::detail::borrowInternal(resource);
        nanovdb::cuda::Buffer<PtrT, decltype(ref)> d_ptrs(stream, ref, ptrs.size(), nanovdb::cuda::noInit);
        nanovdb::cuda::Buffer<uint64_t, decltype(ref)> d_offsets(stream, ref, offsets.size(), nanovdb::cuda::noInit);
        cudaCheck(cudaMemcpyAsync(d_ptrs.data(), ptrs.data(), ptrs.size() * sizeof(PtrT), cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaMemcpyAsync(d_offsets.data(), offsets.data(), offsets.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        const SegmentedPtr<PtrT> voxels{d_ptrs.data(), d_offsets.data(), static_cast<uint32_t>(sets.size())};
        VoxelsToGridBatch<BuildT, ResourceT> builder(std::move(maps), stream, resource);
        return builder.getHandle(voxels, offsets.back(), d_offsets.data(), buffer);
    }
}

}// namespace detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>// Grid<BuildT>
voxelsToGrid(const PtrT d_ijk, size_t voxelCount, double voxelSize, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "voxelsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    PointsToGrid<BuildT, ResourceT> converter(voxelSize, Vec3d(0.0), stream);
    return converter.getHandle(d_ijk, voxelCount, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>// Grid<Point> with PointType coordinates as blind data
pointsToGrid(const PtrT d_xyz, int pointCount, double voxelSize, PointType type, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "pointsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    detail::checkPointCount(pointCount, "PointsToGrid");
    PointsToGrid<Point, ResourceT> converter(voxelSize, Vec3d(0.0), stream);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>// Grid<Point> with PointType coordinates as blind data
pointsToGrid(const PtrT d_xyz, int pointCount, int maxPointsPerVoxel, int tolerance, int maxIterations, PointType type, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "pointsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    detail::checkPointCount(pointCount, "PointsToGrid");
    PointsToGrid<Point, ResourceT> converter(maxPointsPerVoxel, tolerance, maxIterations, stream);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
pointsToGrid(const std::vector<std::tuple<const PtrT,size_t,double,PointType>> &vec, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "pointsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    static_assert(util::is_same<BuildT, Point>::value, "pointsToGrid over a point set builds NanoGrid<Point>");
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) {
        PointsToGrid<Point, ResourceT> converter(std::get<2>(p), Vec3d(0.0), stream);
        converter.setPointType(std::get<3>(p));
        handles.push_back(converter.getHandle(std::get<0>(p), std::get<1>(p), buffer));
    }
    return detail::mergeBuiltHandles(handles, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
voxelsToGrid(const std::vector<std::tuple<const PtrT,size_t,double>> &vec, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "voxelsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    return detail::voxelSetsToGrid<BuildT, PtrT>(vec, buffer, stream, nanovdb::cuda::default_resource<ResourceT>());
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT, typename ResourceT>
nanovdb::cuda::detail::PrototypeFormResult<BufferT, GridHandle<BufferT>>
voxelsToGrid(const PtrT d_ijk, size_t voxelCount, const uint64_t *d_offsets, uint32_t gridCount, double voxelSize, const BufferT &buffer, cudaStream_t stream)
{
    static_assert(!std::is_arithmetic<BufferT>::value && !std::is_null_pointer<BufferT>::value,
                  "voxelsToGrid: pass a cudaStream_t, e.g. cudaStream_t(0), to select the resource form");
    const Map map(detail::checkVoxelSize(voxelSize, "voxelsToGrid"), Vec3d(0.0));
    if (gridCount == 0) return detail::mergeBuiltHandles(std::vector<GridHandle<BufferT>>(), buffer, stream);
    VoxelsToGridBatch<BuildT, ResourceT> builder(std::vector<Map>(gridCount, map), stream);
    return builder.getHandle(d_ijk, voxelCount, d_offsets, buffer);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const PtrT d_ijk, size_t voxelCount, double voxelSize, cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    PointsToGrid<BuildT, ResourceT> converter(voxelSize, Vec3d(0.0), stream, resource);
    return converter.getHandle(d_ijk, voxelCount, BufferT(stream, resource, 0, nanovdb::cuda::noInit));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const PtrT d_xyz, int pointCount, double voxelSize, PointType type, cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    detail::checkPointCount(pointCount, "PointsToGrid");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    PointsToGrid<Point, ResourceT> converter(voxelSize, Vec3d(0.0), stream, resource);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, BufferT(stream, resource, 0, nanovdb::cuda::noInit));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const PtrT d_xyz, int pointCount, int maxPointsPerVoxel, int tolerance, int maxIterations, PointType type,
             cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    detail::checkPointCount(pointCount, "PointsToGrid");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    PointsToGrid<Point, ResourceT> converter(maxPointsPerVoxel, tolerance, maxIterations, stream, resource);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, BufferT(stream, resource, 0, nanovdb::cuda::noInit));
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
pointsToGrid(const std::vector<std::tuple<const PtrT,size_t,double,PointType>> &vec, cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    static_assert(util::is_same<BuildT, Point>::value, "pointsToGrid over a point set builds NanoGrid<Point>");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    const BufferT proto(stream, resource, 0, nanovdb::cuda::noInit);
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) {
        PointsToGrid<Point, ResourceT> converter(std::get<2>(p), Vec3d(0.0), stream, resource);
        converter.setPointType(std::get<3>(p));
        handles.push_back(converter.getHandle(std::get<0>(p), std::get<1>(p), proto));
    }
    return detail::mergeBuiltHandles(handles, proto, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const std::vector<std::tuple<const PtrT,size_t,double>> &vec, cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    return detail::voxelSetsToGrid<BuildT, PtrT>(vec, BufferT(stream, resource, 0, nanovdb::cuda::noInit), stream, resource);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename ResourceT>
typename util::enable_if<(nanovdb::cuda::is_resource<ResourceT>::value || nanovdb::cuda::is_async_resource<ResourceT>::value), GridHandle<nanovdb::cuda::Buffer<std::byte, ResourceT>>>::type
voxelsToGrid(const PtrT d_ijk, size_t voxelCount, const uint64_t *d_offsets, uint32_t gridCount, double voxelSize, cudaStream_t stream, ResourceT resource)
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "the GPU tools allocate stream-ordered scratch: pass an AsyncResource, or wrap a synchronous one in cuda::AsyncFromSync");
    using BufferT = nanovdb::cuda::Buffer<std::byte, ResourceT>;
    const BufferT proto(stream, resource, 0, nanovdb::cuda::noInit);
    const Map map(detail::checkVoxelSize(voxelSize, "voxelsToGrid"), Vec3d(0.0));
    if (gridCount == 0) return detail::mergeBuiltHandles(std::vector<GridHandle<BufferT>>(), proto, stream);
    VoxelsToGridBatch<BuildT, ResourceT> builder(std::vector<Map>(gridCount, map), stream, resource);
    return builder.getHandle(d_ijk, voxelCount, d_offsets, proto);
}

}// namespace tools::cuda ======================================================================================================================================

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename PtrT, typename BufferT = cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::pointsToGrid instead")]]
GridHandle<BufferT>
cudaPointsToGrid(const PtrT dWorldPoints,
                 int pointCount,
                 double voxelSize = 1.0,
                 PointType type = PointType::Default,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::pointsToGrid<PtrT, BufferT, ResourceT>(dWorldPoints, pointCount, voxelSize, type, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::pointsToGrid instead")]]
GridHandle<BufferT>
cudaPointsToGrid(std::vector<std::tuple<const PtrT,size_t,double,PointType>> pointSet,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::pointsToGrid<BuildT, PtrT, BufferT, ResourceT>(pointSet, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::voxelsToGrid instead")]]
GridHandle<BufferT>
cudaVoxelsToGrid(const PtrT dGridVoxels,
                 size_t voxelCount,
                 double voxelSize = 1.0,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::voxelsToGrid<BuildT, PtrT, BufferT, ResourceT>(dGridVoxels, voxelCount, voxelSize, buffer, stream);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename PtrT, typename BufferT = cuda::DualDeviceBuffer, typename ResourceT = nanovdb::cuda::DeviceResource>
[[deprecated("Use cuda::voxelsToGrid instead")]]
GridHandle<BufferT>
cudaVoxelsToGrid(std::vector<std::tuple<const PtrT, size_t, double>> pointSet,
                 const BufferT &buffer = BufferT(),
                 cudaStream_t stream = 0)
{
    return tools::cuda::voxelsToGrid<BuildT, PtrT, BufferT, ResourceT>(pointSet, buffer, stream);
}

}// namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_POINTSTOGRID_CUH_HAS_BEEN_INCLUDED
