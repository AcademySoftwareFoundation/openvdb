// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file CudaPointsToGrid.cuh

    \brief Generates NanoVDB grids from a list of voxels or points on the device

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_CUDA_POINTS_TO_GRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_CUDA_POINTS_TO_GRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>
#include <cub/util_allocator.cuh>
#include <vector>
#include <tuple>

#include <nanovdb/NanoVDB.h>
#include "CudaDeviceBuffer.h"
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/cuda/GpuTimer.cuh>
#include <nanovdb/util/cuda/CudaUtils.h>

/*
Note: 4.29 billion (=2^32) coordinates of type Vec3f have a memory footprint of 48 GB!
*/

namespace nanovdb {

// Define the type used when the points are encoded as blind data in the output grid
enum class PointType : uint32_t { Disable = 0,// no point information e.g. when BuildT != Point
                                  PointID = 1,// linear index of type uint32_t to points
                                  World64 = 2,// Vec3d in world space
                                  World32 = 3,// Vec3f in world space
                                  Grid64  = 4,// Vec3d in grid space
                                  Grid32  = 5,// Vec3f in grid space
                                  Voxel32 = 6,// Vec3f in voxel space
                                  Voxel16 = 7,// Vec3u16 in voxel space
                                  Voxel8  = 8,// Vec3u8 in voxel space
                                  Default = 9,// output matches input, i.e. Vec3d or Vec3f in world space
                                  End     =10 };

//================================================================================================

/// @brief Generates a NanoGrid<Point> from a list of point coordinates on the device. This method is
///        mainly used as a means to build a BVH acceleration structure for points, e.g. for efficient rendering.
/// @tparam Vec3T Template type of the list of coordinates of points in world space. Should be Vec3f or Vec3d.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam AllocT  Template type of optional device allocator for internal tempoary memory
/// @param d_world List of coordinates of points in world space on the device
/// @param pointCount number of point in the list @c d_world
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param type Defined the way point information is represented in the output grid (see PointType enum above)
///             Should not be PointType::Disable!
/// @param buffer Instance of the device buffer used for memory allocation
/// @return Returns a handle with a grid of type NanoGrid<Point> where point information, e.g. coordinates,
///         are represented as blind data defined by @c type.
template<typename Vec3T, typename BufferT = CudaDeviceBuffer, typename AllocT = cub::CachingDeviceAllocator>
GridHandle<BufferT>
cudaPointsToGrid(const Vec3T* d_world,
                 int pointCount,
                 double voxelSize = 1.0,
                 PointType type = PointType::Default,
                 BufferT &buffer = BufferT());

//================================================================================================

template<typename BuildT, typename Vec3T = Coord, typename BufferT = CudaDeviceBuffer, typename AllocT = cub::CachingDeviceAllocator>
GridHandle<BufferT>
cudaPointsToGrid(std::vector<std::tuple<const Vec3T*,size_t,double,PointType>> pointSet, const BufferT &buffer = BufferT());

//================================================================================================

/// @brief Generates a NanoGrid of any type from a list of voxel coordinates on the device. Unlike @c cudaPointsToGrid
///        this method only builds the grid but does not encode the coordinates as blind data. It is mainly useful as a
///        means to generate a grid that is know to contain the voxels given in the list.
/// @tparam BuildT Template type of the return grid
/// @tparam Vec3T Template type of the list of coordinates of voxels in grid (or index) space. Should be Coord, Vec3f or Vec3f.
/// @tparam BufferT Template type of buffer used for memory allocation on the device
/// @tparam AllocT  Template type of optional device allocator for internal tempoary memory
/// @param d_voxels List of coordinates of voxels in grid (or index) space on the device
/// @param pointCount number of voxel in the list @c d_voxels
/// @param voxelSize Size of a voxel in world units used for the output grid
/// @param buffer Instance of the device buffer used for memory allocation
/// @return Returns a handle with the grid of type NanoGrid<BuildT>
template<typename BuildT, typename Vec3T = Coord, typename BufferT = CudaDeviceBuffer, typename AllocT = cub::CachingDeviceAllocator>
GridHandle<BufferT>
cudaVoxelsToGrid(const Vec3T* d_voxels,
                 int pointCount,
                 double voxelSize = 1.0,
                 const BufferT &buffer = BufferT());


//================================================================================================

template<typename BuildT, typename Vec3T = Coord, typename BufferT = CudaDeviceBuffer, typename AllocT = cub::CachingDeviceAllocator>
GridHandle<BufferT>
cudaVoxelsToGrid(std::vector<std::tuple<const Vec3T*,size_t,double>> pointSet, const BufferT &buffer = BufferT());

//================================================================================================

#if 0
// the following function is only included for backwards compatability
template<typename BuildT, typename Vec3T, typename BufferT = CudaDeviceBuffer>
typename enable_if<BuildTraits<BuildT>::is_index, GridHandle<BufferT>>::type
cudaCreateNanoGrid(const Vec3T* d_ijk, size_t pointCount, double voxelSize = 1.0, const BufferT &buffer = BufferT())
{
    return cudaVoxelsToGrid<BuildT, Vec3T, BufferT>(d_ijk, pointCount, voxelSize, buffer);
}
#endif

//================================================================================================

template <typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u8 &voxel, const Vec3T &world, const Map &map)
{
    const Vec3d ijk = map.applyInverseMap(world);// world -> index
    static constexpr double encode = double((1<<8) - 1);
    voxel[0] = uint8_t( encode*(ijk[0] - Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint8_t( encode*(ijk[1] - Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint8_t( encode*(ijk[2] - Floor(ijk[2] + 0.5) + 0.5) );
}

template <typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3u16 &voxel, const Vec3T &world, const Map &map)
{
    const Vec3d ijk = map.applyInverseMap(world);// world -> index
    static constexpr double encode = double((1<<16) - 1);
    voxel[0] = uint16_t( encode*(ijk[0] - Floor(ijk[0] + 0.5) + 0.5) );
    voxel[1] = uint16_t( encode*(ijk[1] - Floor(ijk[1] + 0.5) + 0.5) );
    voxel[2] = uint16_t( encode*(ijk[2] - Floor(ijk[2] + 0.5) + 0.5) );
}

template <typename Vec3T>
__hostdev__ inline static void worldToVoxel(Vec3f &voxel, const Vec3T &world, const Map &map)
{
    const Vec3d ijk = map.applyInverseMap(world);// world -> index
    voxel[0] = float( ijk[0] - Floor(ijk[0] + 0.5) );
    voxel[1] = float( ijk[1] - Floor(ijk[1] + 0.5) );
    voxel[2] = float( ijk[2] - Floor(ijk[2] + 0.5) );
}

//================================================================================================

template <typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u8 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<8) - 1);
    if constexpr(is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template <typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3u16 &voxel, const Coord &ijk, const Map &map)
{
    static constexpr double decode = 1.0/double((1<<16) - 1);
    if constexpr(is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + decode*voxel[0] - 0.5, ijk[1] + decode*voxel[1] - 0.5, ijk[2] + decode*voxel[2] - 0.5));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + decode*voxel[0] - 0.5f, ijk[1] + decode*voxel[1] - 0.5f, ijk[2] + decode*voxel[2] - 0.5f));
    }
}

template <typename Vec3T = Vec3d>
__hostdev__ inline static Vec3T voxelToWorld(const Vec3f &voxel, const Coord &ijk, const Map &map)
{
    if constexpr(is_same<Vec3T,Vec3d>::value) {
        return map.applyMap( Vec3d(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    } else {
        return map.applyMapF(Vec3f(ijk[0] + voxel[0], ijk[1] + voxel[1], ijk[2] + voxel[2]));
    }
}

//================================================================================================

namespace {// anonymous namespace

template <typename BuildT, typename AllocT = cub::CachingDeviceAllocator>
class CudaPointsToGrid
{
public:

    struct Data {
        Map map;
        void     *d_bufferPtr;
        uint64_t *d_keys, *d_tile_keys, *d_lower_keys, *d_leaf_keys;// device pointer to 64 bit keys
        uint64_t  grid, tree, root, upper, lower, leaf, meta, blind, size;// byte offsets to nodes in buffer
        uint32_t *d_indx;// device pointer to point indices (or IDs)
        uint32_t  nodeCount[3], *pointsPerLeafPrefix, *pointsPerLeaf;// 0=leaf,1=lower, 2=upper
        uint32_t  voxelCount,  *pointsPerVoxelPrefix, *pointsPerVoxel;
        BitFlags<16> flags;
        __hostdev__ NanoGrid<BuildT>&  getGrid() const {return *PtrAdd<NanoGrid<BuildT>>(d_bufferPtr, grid);}
        __hostdev__ NanoTree<BuildT>&  getTree() const {return *PtrAdd<NanoTree<BuildT>>(d_bufferPtr, tree);}
        __hostdev__ NanoRoot<BuildT>&  getRoot() const {return *PtrAdd<NanoRoot<BuildT>>(d_bufferPtr, root);}
        __hostdev__ NanoUpper<BuildT>& getUpper(int i) const {return *(PtrAdd<NanoUpper<BuildT>>(d_bufferPtr, upper)+i);}
        __hostdev__ NanoLower<BuildT>& getLower(int i) const {return *(PtrAdd<NanoLower<BuildT>>(d_bufferPtr, lower)+i);}
        __hostdev__ NanoLeaf<BuildT>&  getLeaf(int i) const {return *(PtrAdd<NanoLeaf<BuildT>>(d_bufferPtr, leaf)+i);}
        __hostdev__ GridBlindMetaData& getMeta() const { return *PtrAdd<GridBlindMetaData>(d_bufferPtr, meta);};
         template <typename Vec3T>
        __hostdev__ Vec3T& getPoint(int i) const {return *(PtrAdd<Vec3T>(d_bufferPtr, blind)+i);}
    };// Data

    /// @brief Constructor from a Map
    /// @param map Map to be used for the output device grid
    CudaPointsToGrid(const Map &map) : mPointType(is_same<BuildT,Point>::value ? PointType::Default : PointType::Disable){
        mData.map = map;
        mData.flags.initMask({GridFlags::HasBBox, GridFlags::IsBreadthFirst});
        cudaCheck(cudaMalloc((void**)&mDeviceData, sizeof(Data)));
    }

    /// @brief Default constructor
    /// @param scale Voxel size in world units
    /// @param trans Translation of origin in world units
    CudaPointsToGrid(const double scale = 1.0, const Vec3d &trans = Vec3d(0.0)) : CudaPointsToGrid(Map(scale, trans)) {}

    /// @brief Destructor
    ~CudaPointsToGrid() {cudaCheck(cudaFree(mDeviceData));}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) {mVerbose = level; mData.flags.setBit(7u, level); }

    /// @brief Toggle on and off the computation of a bounding-box
    /// @param on If true bbox will be computed
    void includeBBox(bool on = true) { mData.flags.setMask(GridFlags::HasBBox, on); }

    /// @brief Set the name of the output grid
    /// @param name name of the output grid
    void setGridName(const std::string &name) {mGridName = name;}

    // only available when BuildT == Point
    template <typename T = BuildT> typename enable_if<is_same<T, Point>::value>::type
    setPointType(PointType type) { mPointType = type; }

    /// @brief Creates a handle to a grid with the specified build type from a list of points in index or world space
    /// @tparam BuildT Build type of the output grid, i.e NanoGrid<BuildT>
    /// @tparam Vec3T Type of the input points. If Vec3<float|double> points are in world space and if Coord in Index space
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param d_xyz device point to an array of points in world space
    /// @param pointCount number of input points
    /// @param gridName optional name of the output grid
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    //template<typename BuildT, typename Vec3T, typename BufferT = CudaDeviceBuffer>
    template<typename Vec3T, typename BufferT = CudaDeviceBuffer>
    GridHandle<BufferT> getHandle(const Vec3T* d_xyz, size_t pointCount, const BufferT &buffer = BufferT());

    template <typename Vec3T>
    void countNodes(const Vec3T* d_points, size_t pointCount);

    template <typename Vec3T>
    void processGridTreeRoot(const Vec3T *d_points, size_t pointCount);

    void processUpperNodes();

    void processLowerNodes();

    template <typename Vec3T>
    void processLeafNodes(const Vec3T *d_points);

    template <typename Vec3T>
    void processPoints(const Vec3T *d_points, size_t pointCount);

    void processBBox();

    // the following methods are only defined when BuildT == Point
    template <typename T = BuildT> typename enable_if<is_same<T, Point>::value, uint32_t>::type
    maxPointsPerVoxel() const {return mMaxPointsPerVoxel;}
    template <typename T = BuildT> typename enable_if<is_same<T, Point>::value, uint32_t>::type
    maxPointsPerLeaf()  const {return mMaxPointsPerLeaf;}

private:
    static constexpr unsigned int mNumThreads = 128;// seems faster than the old value of 256!
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    GpuTimer mTimer;
    PointType mPointType;
    std::string mGridName;
    int mVerbose{0};
    Data mData, *mDeviceData;
    uint32_t mMaxPointsPerVoxel{0u}, mMaxPointsPerLeaf{0u};
    // wrapper of cub::CachingDeviceAllocator with a shared scratch space
    struct Allocator {
        AllocT mAllocator;
        void* d_scratch;
        size_t scratchSize, actualScratchSize;
        Allocator() : d_scratch(nullptr), scratchSize(0), actualScratchSize(0) {}
        ~Allocator() {
            if (scratchSize > 0) this->free(d_scratch);// a bug in cub makes this necessary
            mAllocator.FreeAllCached();
        }
        template <typename T>
        T* alloc(size_t count, cudaStream_t stream = 0) {
            T* d_ptr = nullptr;
            cudaCheck(mAllocator.DeviceAllocate((void**)&d_ptr, sizeof(T)*count, stream));
            return d_ptr;
        }
        void free(void *d_ptr) {if (d_ptr) cudaCheck(mAllocator.DeviceFree(d_ptr));}
        template<class... T>
        void free(void *d_ptr, T... other) {
            if (d_ptr) cudaCheck(mAllocator.DeviceFree(d_ptr));
            this->free(other...);
        }
        void adjustScratch(cudaStream_t stream = 0){
            if (scratchSize > actualScratchSize) {
                if (actualScratchSize>0) cudaCheck(mAllocator.DeviceFree(d_scratch));
                cudaCheck(mAllocator.DeviceAllocate((void**)&d_scratch, scratchSize, stream));
                actualScratchSize = scratchSize;
            }
        }
    } mMemPool;

    template<typename Vec3T, typename BufferT>
    BufferT getBuffer(const BufferT &buffer, size_t pointCount);
};// CudaPointsToGrid<BuildT>

//================================================================================================

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mMemPool.scratchSize, __VA_ARGS__)); \
    mMemPool.adjustScratch(); \
    cudaCheck(cub::func(mMemPool.d_scratch, mMemPool.scratchSize, __VA_ARGS__));
#else// fdef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mMemPool.scratchSize, args)); \
    mMemPool.adjustScratch(); \
    cudaCheck(cub::func(mMemPool.d_scratch, mMemPool.scratchSize, args));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

}// anonymous namespace

//================================================================================================

template<typename BuildT, typename AllocT>
template<typename Vec3T, typename BufferT>
inline GridHandle<BufferT>
CudaPointsToGrid<BuildT, AllocT>::getHandle(const Vec3T* d_xyz,
                                                size_t pointCount,
                                                const BufferT &pool)
{
    if (mVerbose==1) mTimer.start("\nCounting nodes");
    this->countNodes(d_xyz, pointCount);

    if (mVerbose==1) mTimer.restart("Initiate buffer");
    auto buffer = this->template getBuffer<Vec3T>(pool, pointCount);

    if (mVerbose==1) mTimer.restart("Process grid,tree,root");
    this->processGridTreeRoot(d_xyz, pointCount);

    if (mVerbose==1) mTimer.restart("Process upper nodes");
    this->processUpperNodes();

    if (mVerbose==1) mTimer.restart("Process lower nodes");
    this->processLowerNodes();

    if (mVerbose==1) mTimer.restart("Process leaf nodes");
    this->processLeafNodes(d_xyz);

    if (mVerbose==1) mTimer.restart("Process points");
    this->processPoints(d_xyz, pointCount);

    if (mVerbose==1) mTimer.restart("Process bbox");
    this->processBBox();

    cudaDeviceSynchronize();// finish all device tasks
    if (mVerbose==1) mTimer.stop();

    return GridHandle<BufferT>(std::move(buffer));
}// CudaPointsToGrid<BuildT>::getHandle

//================================================================================================

// --- CUB helpers ---
template<uint8_t BitCount, typename InT, typename OutT>
struct ShiftRight
{
    __hostdev__ inline OutT operator()(const InT& v) const {return static_cast<OutT>(v >> BitCount);}
};

template<uint8_t BitCount, typename InT = uint64_t, typename OutT = uint64_t>
struct ShiftRightIterator : public cub::TransformInputIterator<OutT, ShiftRight<BitCount, InT, OutT>, InT*>
{
    using BASE = cub::TransformInputIterator<OutT, ShiftRight<BitCount, InT, OutT>, InT*>;
    __hostdev__ inline ShiftRightIterator(uint64_t* input_itr) : BASE(input_itr, ShiftRight<BitCount, InT, OutT>()) {}
};

//================================================================================================

template <typename BuildT, typename AllocT>
template <typename Vec3T>
void CudaPointsToGrid<BuildT, AllocT>::countNodes(const Vec3T *d_points, size_t pointCount)
{
    mData.d_keys = mMemPool.template alloc<uint64_t>(pointCount);
    mData.d_indx = mMemPool.template alloc<uint32_t>(pointCount);// uint32_t can index 4.29 billion Coords, corresponding to 48 GB
    cudaCheck(cudaMemcpy(mDeviceData, &mData, sizeof(Data), cudaMemcpyHostToDevice));// copy mData from CPU -> GPU

    if (mVerbose==2) mTimer.start("\nAllocating arrays for keys and indices");
    auto *d_keys = mMemPool.template alloc<uint64_t>(pointCount);
    auto *d_indx = mMemPool.template alloc<uint32_t>(pointCount);

    if (mVerbose==2) mTimer.restart("Generate tile keys");
    if (is_same<BuildT, Point>::value) {// points in world space
        if (is_same<Vec3T, Vec3f>::value) {
            cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, const Data *d_data) {
                d_indx[tid] = uint32_t(tid);
                d_keys[tid] = NanoRoot<Point>::CoordToKey(d_data->map.applyInverseMapF(d_points[tid]).round());
            }, mDeviceData); cudaCheckError();
        } else if (is_same<Vec3T, Vec3d>::value) {
            cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, const Data *d_data) {
                d_indx[tid] = uint32_t(tid);
                d_keys[tid] = NanoRoot<Point>::CoordToKey(d_data->map.applyInverseMap(d_points[tid]).round());
            }, mDeviceData); cudaCheckError();
        } else {
            throw std::runtime_error("Point (vs voxels) coordinates should be represented as Vec3f or Vec3d");
        }
    } else if (is_same<Vec3T, Coord>::value) {
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, const Data *d_data) {
            d_indx[tid] = uint32_t(tid);
            d_keys[tid] = NanoRoot<BuildT>::CoordToKey(d_points[tid]);
        }, mDeviceData); cudaCheckError();
    } else if (is_same<Vec3T, Vec3f>::value || is_same<Vec3T, Vec3d>::value) {
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, const Data *d_data) {
            d_indx[tid] = uint32_t(tid);
            d_keys[tid] = NanoRoot<BuildT>::CoordToKey(d_points[tid].round());
        }, mDeviceData); cudaCheckError();
    } else {
        throw std::runtime_error("Voxel coordinates should be represented as Coord, Vec3f or Vec3d");
    }

    if (mVerbose==2) mTimer.restart("DeviceRadixSort of "+std::to_string(pointCount)+" tile keys");
    CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, mData.d_keys, d_indx, mData.d_indx, pointCount, 0, 62);// 21 bits per coord
    std::swap(d_indx, mData.d_indx);// sorted indices are now in d_indx

    if (mVerbose==2) mTimer.restart("Allocate runs");
    auto *d_points_per_tile = mMemPool.template alloc<uint32_t>(pointCount);
    uint32_t *d_node_count = mMemPool.template alloc<uint32_t>(3);

    if (mVerbose==2) mTimer.restart("DeviceRunLengthEncode tile keys");
    CALL_CUBS(DeviceRunLengthEncode::Encode, mData.d_keys, d_keys, d_points_per_tile, d_node_count+2, pointCount);
    cudaCheck(cudaMemcpy(mData.nodeCount+2, d_node_count+2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    mData.d_tile_keys = mMemPool.template alloc<uint64_t>(mData.nodeCount[2]);
    cudaCheck(cudaMemcpy(mData.d_tile_keys, d_keys, mData.nodeCount[2]*sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    if (mVerbose) mTimer.restart("DeviceRadixSort of " + std::to_string(pointCount) + " voxel keys in " + std::to_string(mData.nodeCount[2]) + " tiles");
    uint32_t *points_per_tile = new uint32_t[mData.nodeCount[2]];
    cudaCheck(cudaMemcpy(points_per_tile, d_points_per_tile, mData.nodeCount[2]*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    mMemPool.free(d_points_per_tile);

    auto voxelKey = [] __device__ (uint64_t tileID, const Coord &ijk){
        return tileID << 36 |                                          // upper offset: 64-15-12-9=28, i.e. last 28 bits
               uint64_t(NanoUpper<BuildT>::CoordToOffset(ijk)) << 21 | // lower offset: 32^3 = 2^15,   i.e. next 15 bits
               uint64_t(NanoLower<BuildT>::CoordToOffset(ijk)) <<  9 | // leaf  offset: 16^3 = 2^12,   i.e. next 12 bits
               uint64_t(NanoLeaf< BuildT>::CoordToOffset(ijk));        // voxel offset:  8^3 =  2^9,   i.e. first 9 bits
    };

    for (uint32_t id = 0, offset = 0; id < mData.nodeCount[2]; ++id) {
        const uint32_t count = points_per_tile[id];
        cudaLambdaKernel<<<numBlocks(count), mNumThreads>>>(count, [=] __device__(size_t tid, const Data *d_data) {
            tid += offset;
            Vec3T p = d_points[d_indx[tid]];
            if constexpr(is_same<BuildT, Point>::value) p = is_same<Vec3T, Vec3f>::value ? d_data->map.applyInverseMapF(p) : d_data->map.applyInverseMap(p);
            d_keys[tid] = voxelKey(id, p.round());
        }, mDeviceData); cudaCheckError();
        CALL_CUBS(DeviceRadixSort::SortPairs, d_keys + offset, mData.d_keys + offset, d_indx + offset, mData.d_indx + offset, count, 0, 36);// 9+12+15=36
        offset += count;
    }
    mMemPool.free(d_indx);
    delete [] points_per_tile;

    if (mVerbose==2) mTimer.restart("Count points per voxel");

    mData.pointsPerVoxel = mMemPool.template alloc<uint32_t>(pointCount);
    uint32_t *d_voxel_count = mMemPool.template alloc<uint32_t>(1);
    CALL_CUBS(DeviceRunLengthEncode::Encode, mData.d_keys, d_keys, mData.pointsPerVoxel, d_voxel_count, pointCount);
    cudaCheck(cudaMemcpy(&mData.voxelCount, d_voxel_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    mMemPool.free(d_voxel_count);

    if constexpr(is_same<BuildT, Point>::value) {
        if (mVerbose==2) mTimer.restart("Count max points per voxel");
        uint32_t *d_maxPointsPerVoxel = mMemPool.template alloc<uint32_t>(1);
        CALL_CUBS(DeviceReduce::Max, mData.pointsPerVoxel, d_maxPointsPerVoxel, mData.voxelCount);
        cudaCheck(cudaMemcpy(&mMaxPointsPerVoxel, d_maxPointsPerVoxel, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        mMemPool.free(d_maxPointsPerVoxel);
    }

    //printf("\n Active voxel count = %u, max points per voxel = %u\n", mData.voxelCount, mMaxPointsPerVoxel);
    if (mVerbose==2) mTimer.restart("Compute prefix sum of points per voxel");
    mData.pointsPerVoxelPrefix = mMemPool.template alloc<uint32_t>(mData.voxelCount);
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerVoxel, mData.pointsPerVoxelPrefix, mData.voxelCount);

    mData.pointsPerLeaf = mMemPool.template alloc<uint32_t>(pointCount);
    CALL_CUBS(DeviceRunLengthEncode::Encode, ShiftRightIterator<9>(mData.d_keys), d_keys, mData.pointsPerLeaf, d_node_count, pointCount);
    cudaCheck(cudaMemcpy(mData.nodeCount, d_node_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if constexpr(is_same<BuildT, Point>::value) {
        uint32_t *d_maxPointsPerLeaf = mMemPool.template alloc<uint32_t>(1);
        CALL_CUBS(DeviceReduce::Max, mData.pointsPerLeaf, d_maxPointsPerLeaf, mData.nodeCount[0]);
        cudaCheck(cudaMemcpy(&mMaxPointsPerLeaf, d_maxPointsPerLeaf, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        //printf("\n Leaf count = %u, max points per leaf = %u\n", mData.nodeCount[0], mMaxPointsPerLeaf);
        if (mMaxPointsPerLeaf > std::numeric_limits<uint16_t>::max()) {
            throw std::runtime_error("Too many points per leaf: "+std::to_string(mMaxPointsPerLeaf));
        }
        mMemPool.free(d_maxPointsPerLeaf);
    }

    mData.pointsPerLeafPrefix = mMemPool.template alloc<uint32_t>(mData.nodeCount[0]);
    CALL_CUBS(DeviceScan::ExclusiveSum, mData.pointsPerLeaf, mData.pointsPerLeafPrefix, mData.nodeCount[0]);

    mData.d_leaf_keys = mMemPool.template alloc<uint64_t>(mData.nodeCount[0]);
    cudaCheck(cudaMemcpy(mData.d_leaf_keys, d_keys, mData.nodeCount[0]*sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    CALL_CUBS(DeviceSelect::Unique, ShiftRightIterator<12>(mData.d_leaf_keys), d_keys, d_node_count+1, mData.nodeCount[0]);// count lower nodes
    cudaCheck(cudaMemcpy(mData.nodeCount+1, d_node_count+1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    mData.d_lower_keys = mMemPool.template alloc<uint64_t>(mData.nodeCount[1]);
    cudaCheck(cudaMemcpy(mData.d_lower_keys, d_keys, mData.nodeCount[1]*sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    mMemPool.free(d_keys, d_node_count);
    if (mVerbose==2) mTimer.stop();

    //printf("Leaf count = %u, lower count = %u, upper count = %u\n", mData.nodeCount[0], mData.nodeCount[1], mData.nodeCount[2]);
}// CudaPointsToGrid<BuildT>::countNodes

//================================================================================================

template <typename BuildT, typename AllocT>
template <typename Vec3T, typename BufferT>
inline BufferT CudaPointsToGrid<BuildT, AllocT>::getBuffer(const BufferT &pool, size_t pointCount)
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
        case PointType::Default: return sizeof(Vec3T);
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
    mData.size  = mData.blind + pointCount*sizeofPoint();// end of buffer

    auto buffer = BufferT::create(mData.size, &pool, false);
    mData.d_bufferPtr = buffer.deviceData();
    if (mData.d_bufferPtr == nullptr) throw std::runtime_error("Failed to allocate grid buffer on the device");
    cudaCheck(cudaMemcpy(mDeviceData, &mData, sizeof(Data), cudaMemcpyHostToDevice));// copy Data CPU -> GPU
    return buffer;
}// CudaPointsToGrid<BuildT>::getBuffer

//================================================================================================

template <typename BuildT, typename AllocT>
template <typename Vec3T>
inline void CudaPointsToGrid<BuildT, AllocT>::processGridTreeRoot(const Vec3T *d_points, size_t pointCount)
{
    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__(size_t, Data *d_data, PointType pointType) {
       // process Root
        auto &root = d_data->getRoot();
        root.mBBox = CoordBBox(); // init to empty
        root.mTableSize = d_data->nodeCount[2];
        root.mBackground = NanoRoot<BuildT>::ValueType(0);// background_value
        root.mMinimum = root.mMaximum = NanoRoot<BuildT>::ValueType(0);
        root.mAverage = root.mStdDevi = NanoRoot<BuildT>::FloatType(0);

        // process Tree
        auto &tree = d_data->getTree();
        tree.setRoot(&root);
        tree.setFirstNode(&d_data->getUpper(0));
        tree.setFirstNode(&d_data->getLower(0));
        tree.setFirstNode(&d_data->getLeaf(0));
        tree.mNodeCount[2] = tree.mTileCount[2] = d_data->nodeCount[2];
        tree.mNodeCount[1] = tree.mTileCount[1] = d_data->nodeCount[1];
        tree.mNodeCount[0] = tree.mTileCount[0] = d_data->nodeCount[0];
        tree.mVoxelCount = d_data->voxelCount;

        // process Grid
        auto &grid = d_data->getGrid();
        grid.init({GridFlags::HasBBox, GridFlags::IsBreadthFirst}, d_data->size, d_data->map, mapToGridType<BuildT>());
        grid.mBlindMetadataCount  = is_same<BuildT, Point>::value;// ? 1u : 0u;
        grid.mBlindMetadataOffset = d_data->meta;
        if (pointType != PointType::Disable) {
            const auto lastLeaf = tree.mNodeCount[0] - 1;
            grid.mData1 = d_data->pointsPerLeafPrefix[lastLeaf] + d_data->pointsPerLeaf[lastLeaf];
            auto &meta = d_data->getMeta();
            meta.mDataOffset = sizeof(GridBlindMetaData);// blind data is placed right after this meta data
            meta.mValueCount = pointCount;
            // Blind meta data
            switch (pointType){
            case PointType::PointID:
                grid.mGridClass = GridClass::PointIndex;
                meta.mSemantic  = GridBlindDataSemantic::PointId;
                meta.mDataClass = GridBlindDataClass::IndexArray;
                meta.mDataType  = mapToGridType<uint32_t>();
                meta.mValueSize = sizeof(uint32_t);
                cudaStrcpy(meta.mName, "PointID: uint32_t indices to points");
                break;
            case PointType::World64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                cudaStrcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                break;
            case PointType::World32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                cudaStrcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                break;
            case PointType::Grid64:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3d>();
                meta.mValueSize = sizeof(Vec3d);
                cudaStrcpy(meta.mName, "Grid64: Vec3<double> point coordinates in grid space");
                break;
            case PointType::Grid32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::GridCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                cudaStrcpy(meta.mName, "Grid32: Vec3<float> point coordinates in grid space");
                break;
            case PointType::Voxel32:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3f>();
                meta.mValueSize = sizeof(Vec3f);
                cudaStrcpy(meta.mName, "Voxel32: Vec3<float> point coordinates in voxel space");
                break;
            case PointType::Voxel16:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3u16>();
                meta.mValueSize = sizeof(Vec3u16);
                cudaStrcpy(meta.mName, "Voxel16: Vec3<uint16_t> point coordinates in voxel space");
                break;
            case PointType::Voxel8:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::VoxelCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3u8>();
                meta.mValueSize = sizeof(Vec3u8);
                cudaStrcpy(meta.mName, "Voxel8: Vec3<uint8_t> point coordinates in voxel space");
                break;
            case PointType::Default:
                grid.mGridClass = GridClass::PointData;
                meta.mSemantic  = GridBlindDataSemantic::WorldCoords;
                meta.mDataClass = GridBlindDataClass::AttributeArray;
                meta.mDataType  = mapToGridType<Vec3T>();
                meta.mValueSize = sizeof(Vec3T);
                if constexpr(is_same<Vec3T, Vec3f>::value) {
                    cudaStrcpy(meta.mName, "World32: Vec3<float> point coordinates in world space");
                } else if constexpr(is_same<Vec3T, Vec3d>::value){
                    cudaStrcpy(meta.mName, "World64: Vec3<double> point coordinates in world space");
                } else {
                    printf("Error in CudaPointsToGrid<BuildT>::processGridTreeRoot: expected Vec3T = Vec3f or Vec3d\n");
                }
                break;
            default:
                printf("Error in CudaPointsToGrid<BuildT>::processGridTreeRoot: invalid pointType\n");
            }
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            grid.mData1 = 1u + 512u*d_data->nodeCount[0];
            grid.mGridClass = GridClass::IndexGrid;
        }
    }, mDeviceData, mPointType);// cudaLambdaKernel
    cudaCheckError();

    char *dst = mData.getGrid().mGridName;
    if (const char *src = mGridName.data()) {
        cudaCheck(cudaMemcpy(dst, src, GridData::MaxNameSize, cudaMemcpyHostToDevice));
    } else {
        cudaCheck(cudaMemset(dst, 0, GridData::MaxNameSize));
    }
}// CudaPointsToGrid<BuildT>::processGridTreeRoot

//================================================================================================

template <typename BuildT, typename AllocT>
inline void CudaPointsToGrid<BuildT, AllocT>::processUpperNodes()
{
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads>>>(mData.nodeCount[2], [=] __device__(size_t tid, Data *d_data) {
        auto &root  = d_data->getRoot();
        auto &upper = d_data->getUpper(tid);
        const Coord ijk = NanoRoot<uint32_t>::KeyToCoord(d_data->d_tile_keys[tid]);
        root.tile(tid)->setChild(ijk, &upper, &root);
        upper.mBBox[0] = ijk;
        upper.mFlags = 0;
        upper.mValueMask.setOff();
        upper.mChildMask.setOff();
        upper.mMinimum = upper.mMaximum = NanoLower<BuildT>::ValueType(0);
        upper.mAverage = upper.mStdDevi = NanoLower<BuildT>::FloatType(0);
    }, mDeviceData);
    cudaCheckError();

    mMemPool.free(mData.d_tile_keys);

    const uint64_t valueCount = mData.nodeCount[2] << 15;
    cudaLambdaKernel<<<numBlocks(valueCount), mNumThreads>>>(valueCount, [=] __device__(size_t tid, Data *d_data) {
        auto &upper = d_data->getUpper(tid >> 15);
        upper.mTable[tid & 32767u].value = NanoUpper<BuildT>::ValueType(0);// background
    }, mDeviceData);
    cudaCheckError();
}// CudaPointsToGrid<BuildT>::processUpperNodes

//================================================================================================

template <typename BuildT, typename AllocT>
inline void CudaPointsToGrid<BuildT, AllocT>::processLowerNodes()
{
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads>>>(mData.nodeCount[1], [=] __device__(size_t tid, Data *d_data) {
        auto &root  = d_data->getRoot();
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        const uint32_t upperOffset = lowerKey & 32767u;// (1 << 15) - 1 = 32767
        upper.mChildMask.setOnAtomic(upperOffset);
        auto &lower = d_data->getLower(tid);
        upper.setChild(upperOffset, &lower);
        lower.mBBox[0] = upper.offsetToGlobalCoord(upperOffset);
        lower.mFlags = 0;
        lower.mValueMask.setOff();
        lower.mChildMask.setOff();
        lower.mMinimum = lower.mMaximum = NanoLower<BuildT>::ValueType(0);// background;
        lower.mAverage = lower.mStdDevi = NanoLower<BuildT>::FloatType(0);
    }, mDeviceData);
    cudaCheckError();

    const uint64_t valueCount = mData.nodeCount[1] << 12;
    cudaLambdaKernel<<<numBlocks(valueCount), mNumThreads>>>(valueCount, [=] __device__(size_t tid, Data *d_data) {
        auto &lower = d_data->getLower(tid >> 12);
        lower.mTable[tid & 4095u].value = NanoLower<BuildT>::ValueType(0);// background
    }, mDeviceData);
    cudaCheckError();
}// CudaPointsToGrid<BuildT>::processLowerNodes

//================================================================================================

template <typename BuildT, typename AllocT>
template <typename Vec3T>
inline void CudaPointsToGrid<BuildT, AllocT>::processLeafNodes(const Vec3T *d_points)
{
    const uint8_t flags = static_cast<uint8_t>(mData.flags.data());// mIncludeStats ? 16u : 0u;// 4th bit indicates stats

    if (mVerbose==2) mTimer.start("process leaf meta data");
    // loop over leaf nodes and add it to its parent node
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads>>>(mData.nodeCount[0], [=] __device__(size_t tid, Data *d_data) {
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

        if constexpr(is_same<Point, BuildT>::value) {
            leaf.mOffset = d_data->pointsPerLeafPrefix[tid];
            leaf.mPointCount = d_data->pointsPerLeaf[tid];
        } else if constexpr(BuildTraits<BuildT>::is_offindex) {
            leaf.mOffset = tid*512u + 1u;// background is index 0
            leaf.mPrefixSum = 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mAverage = leaf.mStdDevi = NanoLeaf<BuildT>::FloatType(0);
            leaf.mMinimum = leaf.mMaximum = NanoLeaf<BuildT>::ValueType(0);
        }
    }, mDeviceData); cudaCheckError();

    if (mVerbose==2) mTimer.restart("set active voxel state and values");
    // loop over all active voxels and set LeafNode::mValueMask and LeafNode::mValues
    cudaLambdaKernel<<<numBlocks(mData.voxelCount), mNumThreads>>>(mData.voxelCount, [=] __device__(size_t tid, Data *d_data) {
        const uint32_t pointID  = d_data->pointsPerVoxelPrefix[tid];
        const uint64_t voxelKey = d_data->d_keys[pointID];
        auto &upper = d_data->getUpper(voxelKey >> 36);
        auto &lower = *upper.getChild((voxelKey >> 21) & 32767u);
        auto &leaf  = *lower.getChild((voxelKey >>  9) &  4095u);
        const uint32_t n = voxelKey & 511u;
        leaf.mValueMask.setOnAtomic(n);// <--- slow!
        if constexpr(is_same<Point, BuildT>::value) {
            leaf.mValues[n] = uint16_t(pointID + d_data->pointsPerVoxel[tid] - leaf.offset());
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = NanoLeaf<BuildT>::ValueType(1);// set value of active voxels that are not points (or index)
        }
    }, mDeviceData); cudaCheckError();

    mMemPool.free(mData.d_keys, mData.pointsPerVoxel, mData.pointsPerVoxelPrefix, mData.pointsPerLeafPrefix, mData.pointsPerLeaf);

    if (mVerbose==2) mTimer.restart("set inactive voxel values");
    const uint64_t denseVoxelCount = mData.nodeCount[0] << 9;
    cudaLambdaKernel<<<numBlocks(denseVoxelCount), mNumThreads>>>(denseVoxelCount, [=] __device__(size_t tid, Data *d_data) {
        auto &leaf = d_data->getLeaf(tid >> 9u);
        const uint32_t n = tid & 511u;
        if (leaf.mValueMask.isOn(n)) return;
        if constexpr(is_same<BuildT, Point>::value) {
            const uint32_t m = leaf.mValueMask.findPrev<true>(n - 1);
            leaf.mValues[n] = m < 512u ? leaf.mValues[m] : 0u;
        } else if constexpr(!BuildTraits<BuildT>::is_special) {
            leaf.mValues[n] = NanoLeaf<BuildT>::ValueType(0);// value of inactive voxels
        }
    }, mDeviceData); cudaCheckError();

    if constexpr(BuildTraits<BuildT>::is_onindex) {
        if (mVerbose==2) mTimer.restart("prefix-sum for index grid");
        uint64_t *devValueIndex = mMemPool.template alloc<uint64_t>(mData.nodeCount[0]);
        auto devValueIndexPrefix = mMemPool.template alloc<uint64_t>(mData.nodeCount[0]);
        cudaLambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads>>>(mData.nodeCount[0], [=] __device__(size_t tid, Data *d_data) {
            devValueIndex[tid] = static_cast<uint64_t>(d_data->getLeaf(tid).mValueMask.countOn());
        }, mDeviceData); cudaCheckError();
        CALL_CUBS(DeviceScan::InclusiveSum, devValueIndex, devValueIndexPrefix, mData.nodeCount[0]);
        mMemPool.free(devValueIndex);
        cudaLambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads>>>(mData.nodeCount[0], [=] __device__(size_t tid, Data *d_data) {
            auto &leaf = d_data->getLeaf(tid);
            leaf.mOffset = 1u;// will be re-set below
            const uint64_t *w = leaf.mValueMask.words();
            uint64_t &prefixSum = leaf.mPrefixSum, sum = CountOn(*w++);
            prefixSum = sum;
            for (int n = 9; n < 55; n += 9) {// n=i*9 where i=1,2,..6
                sum += CountOn(*w++);
                prefixSum |= sum << n;// each pre-fixed sum is encoded in 9 bits
            }
            if (tid==0) {
                d_data->getGrid().mData1 = 1u + devValueIndexPrefix[d_data->nodeCount[0]-1];// set total count
                d_data->getTree().mVoxelCount = devValueIndexPrefix[d_data->nodeCount[0]-1];
            } else {
                leaf.mOffset = 1u + devValueIndexPrefix[tid-1];// background is index 0
            }
        }, mDeviceData); cudaCheckError();
        mMemPool.free(devValueIndexPrefix);
    }

    if constexpr(BuildTraits<BuildT>::is_indexmask) {
        if (mVerbose==2) mTimer.restart("leaf.mMask = leaf.mValueMask");
        cudaLambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads>>>(mData.nodeCount[0], [=] __device__(size_t tid, Data *d_data) {
            auto &leaf = d_data->getLeaf(tid);
            leaf.mMask = leaf.mValueMask;
        }, mDeviceData); cudaCheckError();
    }
    if (mVerbose==2) mTimer.stop();
}// CudaPointsToGrid<BuildT>::processLeafNodes

//================================================================================================

template <typename BuildT, typename AllocT>
template <typename Vec3T>
inline void CudaPointsToGrid<BuildT, AllocT>::processPoints(const Vec3T *d_points, size_t pointCount)
{
    mMemPool.free(mData.d_indx);
}

//================================================================================================

// Template specialization with BuildT = Point
template <>
template <typename Vec3T>
inline void CudaPointsToGrid<Point>::processPoints(const Vec3T *d_points, size_t pointCount)
{
    switch (mPointType){
    case PointType::Disable:
        throw std::runtime_error("CudaPointsToGrid<Point>::processPoints: mPointType == PointType::Disable\n");
    case PointType::PointID:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<uint32_t>(tid) = d_data->d_indx[tid];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World64:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<Vec3d>(tid) = d_points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::World32:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<Vec3f>(tid) = d_points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid64:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<Vec3d>(tid) = d_data->map.applyInverseMap(d_points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Grid32:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<Vec3f>(tid) = d_data->map.applyInverseMapF(d_points[d_data->d_indx[tid]]);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel32:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3f>(tid), d_points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel16:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u16>(tid), d_points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Voxel8:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            worldToVoxel(d_data->template getPoint<Vec3u8>(tid), d_points[d_data->d_indx[tid]], d_data->map);
        }, mDeviceData); cudaCheckError();
        break;
    case PointType::Default:
        cudaLambdaKernel<<<numBlocks(pointCount), mNumThreads>>>(pointCount, [=] __device__(size_t tid, Data *d_data) {
            d_data->template getPoint<Vec3T>(tid) = d_points[d_data->d_indx[tid]];
        }, mDeviceData); cudaCheckError();
        break;
    default:
        printf("Internal error in CudaPointsToGrid<Point>::processPoints\n");
    }
    mMemPool.free(mData.d_indx);
}// CudaPointsToGrid<Point>::processPoints

//================================================================================================

template <typename BuildT, typename AllocT>
inline void CudaPointsToGrid<BuildT, AllocT>::processBBox()
{
    if (mData.flags.isMaskOff(GridFlags::HasBBox)) {
        mMemPool.free(mData.d_leaf_keys, mData.d_lower_keys);
        return;
    }

    // reset bbox in lower nodes
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads>>>(mData.nodeCount[1], [=] __device__(size_t tid, Data *d_data) {
        d_data->getLower(tid).mBBox = CoordBBox();
    }, mDeviceData);
    cudaCheckError();

    // update and propagate bbox from leaf -> lower/parent nodes
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[0]), mNumThreads>>>(mData.nodeCount[0], [=] __device__(size_t tid, Data *d_data) {
        const uint64_t leafKey = d_data->d_leaf_keys[tid];
        auto &upper = d_data->getUpper(leafKey >> 27);
        auto &lower = *upper.getChild((leafKey >> 12) & 32767u);
        auto &leaf = d_data->getLeaf(tid);
        leaf.updateBBox();
        lower.mBBox.expandAtomic(leaf.bbox());
    }, mDeviceData);
    mMemPool.free(mData.d_leaf_keys);
    cudaCheckError();

    // reset bbox in upper nodes
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads>>>(mData.nodeCount[2], [=] __device__(size_t tid, Data *d_data) {
        d_data->getUpper(tid).mBBox = CoordBBox();
    }, mDeviceData);
    cudaCheckError();

    // propagate bbox from lower -> upper/parent node
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[1]), mNumThreads>>>(mData.nodeCount[1], [=] __device__(size_t tid, Data *d_data) {
        const uint64_t lowerKey = d_data->d_lower_keys[tid];
        auto &upper = d_data->getUpper(lowerKey >> 15);
        auto &lower = d_data->getLower(tid);
        upper.mBBox.expandAtomic(lower.bbox());
    }, mDeviceData);
    mMemPool.free(mData.d_lower_keys);
    cudaCheckError()

    // propagate bbox from upper -> root/parent node
    cudaLambdaKernel<<<numBlocks(mData.nodeCount[2]), mNumThreads>>>(mData.nodeCount[2], [=] __device__(size_t tid, Data *d_data) {
        d_data->getRoot().mBBox.expandAtomic(d_data->getUpper(tid).bbox());
    }, mDeviceData);
    cudaCheckError();

    // update the world-bbox in the root node
    cudaLambdaKernel<<<1, 1>>>(1, [=] __device__(size_t, Data *d_data) {
        d_data->getGrid().mWorldBBox = d_data->getRoot().mBBox.transform(d_data->map);
    }, mDeviceData);
    cudaCheckError();
}// CudaPointsToGrid<BuildT>::processBBox

//================================================================================================

template<typename Vec3T, typename BufferT, typename AllocT>
GridHandle<BufferT>// Grid<Point> with PointType coordinates as blind data
cudaPointsToGrid(const Vec3T* d_xyz, int pointCount, double voxelSize, PointType type, BufferT &buffer)
{
    CudaPointsToGrid<Point, AllocT> converter(voxelSize);
    converter.setPointType(type);
    return converter.getHandle(d_xyz, pointCount, buffer);
}

//================================================================================================

template<typename BuildT, typename Vec3T, typename BufferT, typename AllocT>
GridHandle<BufferT>// Grid<BuildT>
cudaVoxelsToGrid(const Vec3T* d_ijk, int pointCount, double voxelSize, const BufferT &buffer)
{
    CudaPointsToGrid<BuildT, AllocT> converter(voxelSize);
    return converter.getHandle(d_ijk, pointCount, buffer);
}

//================================================================================================

template<typename BuildT, typename Vec3T, typename BufferT, typename AllocT>
GridHandle<BufferT>
cudaPointsToGrid(std::vector<std::tuple<const Vec3T*,size_t,double,PointType>> vec, const BufferT &buffer)
{
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) handles.push_back(cudaPointsToGrid<BuildT, AllocT>(std::get<0>(p), std::get<1>(p), std::get<2>(p), std::get<3>(p), buffer));
    return mergeDeviceGrids(handles);
}

//================================================================================================

template<typename BuildT, typename Vec3T, typename BufferT, typename AllocT>
GridHandle<BufferT>
cudaVoxelsToGrid(std::vector<std::tuple<const Vec3T*,size_t,double>> vec, const BufferT &buffer)
{
    std::vector<GridHandle<BufferT>> handles;
    for (auto &p : vec) handles.push_back(cudaVoxelsToGrid<BuildT, Vec3T, BufferT, AllocT>(std::get<0>(p), std::get<1>(p), std::get<2>(p), buffer));
    return mergeDeviceGrids(handles);
}

}// nanovdb namespace

#endif // NVIDIA_CUDA_POINTS_TO_GRID_CUH_HAS_BEEN_INCLUDED
