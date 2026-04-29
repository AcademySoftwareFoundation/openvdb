// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToGrid.cuh

    \authors Efty Sifakis

    \brief Rasterization of triangle mesh into a sparse NanoVDB indexGrid on the device.
           Optionally an Unsigned Distance Field can be returned in a newly allocated sidecar.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED

#include <cub/cub.cuh>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/TempPool.h>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/tools/cuda/TopologyBuilder.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Rasterization.cuh>
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>

#include <utility>

namespace nanovdb {

namespace tools::cuda {

/// @brief POD triangle type with __hostdev__ accessors, safe for use in both host and device code.
///
///        Used as the element type of the transformed-triangle device buffer throughout MeshToGrid.
///        The only interface requirement imposed on this type by the rasterization functors in
///        Rasterization.cuh is a __hostdev__ const Vec3f& operator[](int) const returning the
///        i-th vertex (i = 0, 1, 2).
///
/// @todo  This type should ultimately migrate to a dedicated geometric-primitives header in
///        nanovdb/math/ (e.g. nanovdb/math/Primitives.h) once such a header is introduced.
struct Triangle {
    nanovdb::Vec3f v[3];
    __hostdev__ const nanovdb::Vec3f& operator[](int i) const { return v[i]; }
    __hostdev__       nanovdb::Vec3f& operator[](int i)       { return v[i]; }
};

template <typename BuildT>
class MeshToGrid
{
    using PointT = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using TriangleT = Triangle;
    using GridT  = NanoGrid<BuildT>;
    using TreeT  = NanoTree<BuildT>;
    using RootT  = NanoRoot<BuildT>;
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;

public:
    struct alignas(16) BoxTrianglePair { // sizeof(BoxTrianglePair) = 16B
        nanovdb::Coord origin; // 12B
        uint32_t triangleID;   // 4B
    };

    /// @brief Constructor
    /// @param devicePoints Vertex list for input triangle surface
    /// @param pointCount Vertex count for input triangle surface
    /// @param deviceTriangles Triangle index list
    /// @param triangleCount Triangle count
    /// @param map Affine map to be used in the conversion
    MeshToGrid(
        const nanovdb::Vec3f *devicePoints,
        const uint32_t pointCount,
        const nanovdb::Vec3i *deviceTriangles,
        const uint32_t triangleCount,
        const nanovdb::Map map = nanovdb::Map(),
        cudaStream_t stream = 0
    )
        : mStream(stream), mTimer(stream), mBuilder(stream), mDevicePoints(devicePoints), mPointCount(pointCount),
         mDeviceTriangles(deviceTriangles), mTriangleCount(triangleCount), mMap(map)
    {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing, 2=benchmarking
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of narrow band
    /// @param bandWidth Narrow band width in cell units
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief Set the AABB scale threshold below which the intersection test switches
    ///        from a conservative AABB-only test to a precise SAT test.
    ///        Default is 128 (the size of a NanoVDB lower node).
    /// @param threshold Index-space AABB scale threshold in voxel units
    void setSATThreshold(int threshold = 128) { mSATThreshold = threshold; }

    /// @brief Set the name of the output grid
    /// @param name Grid name string
    void setGridName(const std::string &name) { mGridName = name; }

    /// @brief Set the mode for checksum computation, which is disabled by default
    /// @param mode Mode of checksum computation
    void setChecksum(CheckMode mode = CheckMode::Disable) { mBuilder.mChecksum = mode; }

    /// @brief Creates a handle to the output grid
    /// @tparam BufferT Buffer type used for allocation of the grid handle
    /// @param buffer optional buffer (currently ignored)
    /// @return returns a handle with a grid of type NanoGrid<BuildT>
    template<typename BufferT = nanovdb::cuda::DeviceBuffer>
    GridHandle<BufferT>
    getHandle(const BufferT &buffer = BufferT());

    /// @brief Creates a grid handle and a device sidecar buffer containing the
    ///        per-active-voxel unsigned distance field values.
    ///
    ///        The sidecar buffer stores (activeVoxelCount+1) floats indexed by
    ///        leaf->getValue(voxelID) for ValueOnIndex grids:
    ///          - sidecar[0]   = mBandWidth (background slot)
    ///          - sidecar[k>0] = UDF at the k-th active voxel (in voxel units)
    ///
    /// @tparam GridBufferT    Buffer type for the output grid handle
    /// @tparam SidecarBufferT Buffer type for the UDF sidecar (defaults to DeviceBuffer)
    /// @param buffer          optional allocator for the grid handle (currently ignored)
    /// @param sidecarBuffer   optional allocator for the UDF sidecar (currently ignored)
    /// @return std::pair of grid handle and UDF sidecar buffer
    template<typename GridBufferT    = nanovdb::cuda::DeviceBuffer,
             typename SidecarBufferT = nanovdb::cuda::DeviceBuffer>
    std::pair<GridHandle<GridBufferT>, SidecarBufferT>
    getHandleAndUDF(const GridBufferT&    buffer        = GridBufferT(),
                    const SidecarBufferT& sidecarBuffer = SidecarBufferT());

private:
    void transformTriangles();

    void processRootTrianglePairs();

    void enumerateRootTiles();

    void processLeafTrianglePairs();

    void buildRasterizedRoot();

    void rasterizeInternalNodes();

    void processGridTreeRoot();

    void rasterizeLeafNodes();


    static constexpr unsigned int mNumThreads = 128;// for kernels spawned via lambdaKernel (others may specialize)
    static unsigned int numBlocks(unsigned int n) {return (n + mNumThreads - 1) / mNumThreads;}

    TopologyBuilder<BuildT>      mBuilder;
    cudaStream_t                 mStream{0};
    std::string                  mGridName;
    util::cuda::Timer            mTimer;
    int                          mVerbose{0};
    float                        mBandWidth{3.f};
    int                          mSATThreshold{128};
    const nanovdb::Vec3f         *mDevicePoints;
    const uint32_t               mPointCount;
    const nanovdb::Vec3i         *mDeviceTriangles;
    const uint32_t               mTriangleCount;
    const nanovdb::Map           mMap;

    nanovdb::cuda::DeviceBuffer  mXformedTriangles;
    nanovdb::cuda::DeviceBuffer  mBoxTrianglePairsBuffer;
    uint64_t                     mBoxTrianglePairCount{0};
    nanovdb::cuda::DeviceBuffer  mUniqueRootOriginsBuffer;
    uint64_t                     mUniqueRootTileCount{0};

    auto deviceXformedTriangles()  { return static_cast<TriangleT*>(mXformedTriangles.deviceData()); }
    auto deviceBoxTrianglePairs()  { return static_cast<BoxTrianglePair*>(mBoxTrianglePairsBuffer.deviceData()); }
    auto deviceUniqueRootOrigins() const { return static_cast<nanovdb::Coord*>(mUniqueRootOriginsBuffer.deviceData()); }

    nanovdb::cuda::TempDevicePool mTempDevicePool;
}; // tools::cuda::MeshToGrid<BuildT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Define utility macro used to call cub functions that use dynamic temporary storage
#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), __VA_ARGS__, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), __VA_ARGS__, mStream));
#else// ndef _WIN32
#define CALL_CUBS(func, args...) \
    cudaCheck(cub::func(nullptr, mTempDevicePool.requestedSize(), args, mStream)); \
    mTempDevicePool.reallocate(mStream); \
    cudaCheck(cub::func(mTempDevicePool.data(), mTempDevicePool.size(), args, mStream));
#endif// ifdef _WIN32
#endif// ifndef CALL_CUBS

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename BufferT>
GridHandle<BufferT>
MeshToGrid<BuildT>::getHandle(const BufferT &buffer)
{
    cudaStreamSynchronize(mStream);

    // Transform triangle data to (floating-point) index space
    if (mVerbose==1) mTimer.start("Transforming triangles to grid index space");
    transformTriangles();
    if (mVerbose==1) mTimer.stop();

    // Process Root-Triangle pairs
    if (mVerbose==1) mTimer.start("Computing candidate RootTile-Triangle intersection pairs");
    processRootTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    // Extract unique root tile origins while root-level pairs are still available
    if (mVerbose==1) mTimer.start("Enumerating unique root tiles");
    enumerateRootTiles();
    if (mVerbose==1) mTimer.stop();

    // Process Leaf-Triangle pairs
    if (mVerbose==1) mTimer.start("Computing candidate LeafNode-Triangle intersection pairs");
    processLeafTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    // Build rasterized root node (one tile per unique root origin)
    if (mVerbose==1) mTimer.start("Building rasterized root node");
    buildRasterizedRoot();
    if (mVerbose==1) mTimer.stop();

    // Allocate (zero-filled) upper/lower mask buffers sized from the root tile count
    if (mVerbose==1) mTimer.start("Allocating internal mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);
    if (mVerbose==1) mTimer.stop();

    // Scatter leaf/triangle pair origins into upper/lower topology masks
    if (mVerbose==1) mTimer.start("Rasterizing internal nodes");
    rasterizeInternalNodes();
    if (mVerbose==1) mTimer.stop();

    // Count nodes at all levels from the filled masks
    if (mVerbose==1) mTimer.start("Counting nodes");
    mBuilder.countNodes(mStream);
    cudaStreamSynchronize(mStream); // node counts written async; sync before reading or passing to getBuffer
    if (mVerbose==1) mTimer.stop();

    // Allocate output grid buffer
    if (mVerbose==1) mTimer.start("Allocating grid buffer");
    auto gridBuffer = mBuilder.getBuffer(buffer, mStream);
    if (mVerbose==1) mTimer.stop();

    // Initialize grid/tree/root metadata
    if (mVerbose==1) mTimer.start("Processing grid/tree/root");
    processGridTreeRoot();
    if (mVerbose==1) mTimer.stop();

    // Connect upper nodes to root tiles and fill upper node preambles
    if (mVerbose==1) mTimer.start("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);
    if (mVerbose==1) mTimer.stop();

    // Fill lower node data and leaf node preambles; releases intermediate mask buffers
    if (mVerbose==1) mTimer.start("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);
    if (mVerbose==1) mTimer.stop();

    // Fill leaf voxel value masks via exact point-to-triangle UDF
    if (mVerbose==1) mTimer.start("Rasterizing leaf nodes");
    rasterizeLeafNodes();
    if (mVerbose==1) mTimer.stop();
    if (mBoxTrianglePairCount) {
        mXformedTriangles.clear(mStream);
        mBoxTrianglePairsBuffer.clear(mStream);
    }

    // Update leaf value offsets (prefix sums of per-leaf active voxel counts)
    if (mVerbose==1) mTimer.start("Processing leaf offsets");
    mBuilder.processLeafOffsets(mStream);
    if (mVerbose==1) mTimer.stop();

    // Compute world-space bounding boxes
    if (mVerbose==1) mTimer.start("Processing bounding boxes");
    mBuilder.processBBox(mStream);
    if (mVerbose==1) mTimer.stop();

    // Finalize tree statistics (total active voxel count, etc.)
    if (mVerbose==1) mTimer.start("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    cudaStreamSynchronize(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Pruning empty leaves");
    int device = 0; cudaGetDevice(&device);
    const uint32_t leafCount = mBuilder.data()->nodeCount[0];
    auto handle = GridHandle<BufferT>(std::move(gridBuffer));
    if (leafCount) {
        nanovdb::cuda::DeviceBuffer retainMaskBuffer = nanovdb::cuda::DeviceBuffer::create(
            uint64_t(leafCount) * sizeof(nanovdb::Mask<3>), nullptr, device, mStream);
        cudaCheck(cudaMemsetAsync(retainMaskBuffer.deviceData(), 0xFF,
            uint64_t(leafCount) * sizeof(nanovdb::Mask<3>), mStream));
        tools::cuda::PruneGrid<BuildT> pruner(
            static_cast<const GridT*>(handle.deviceData()),
            static_cast<nanovdb::Mask<3>*>(retainMaskBuffer.deviceData()),
            mStream);
        handle = pruner.template getHandle<BufferT>(buffer);
    }
    if (mVerbose==1) mTimer.stop();
    return handle;
} // MeshToGrid<BuildT>::getHandle

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct TransformTrianglesFunctor
{
    const nanovdb::Vec3f* dPoints;
    const nanovdb::Vec3i* dTriangleIndices;
    Triangle* dXformedTriangles;
    nanovdb::Map map;

    __device__
    void operator()(size_t triangleID) const
    {
        for (int v = 0; v < 3; ++v) {
            dXformedTriangles[triangleID][v] = map.applyInverseMap(dPoints[dTriangleIndices[triangleID][v]]);
        }
    }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::transformTriangles()
{
    if (mTriangleCount == 0) return;

    int device = 0;
    cudaGetDevice(&device);

    mXformedTriangles = nanovdb::cuda::DeviceBuffer::create(mTriangleCount*sizeof(TriangleT), nullptr, device, mStream);
    if (mXformedTriangles.deviceData() == nullptr) throw std::runtime_error("Failed to allocate transofmed upper mask buffer on device");

    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount,
        topology::detail::TransformTrianglesFunctor<BuildT>{
            mDevicePoints, mDeviceTriangles, deviceXformedTriangles(), mMap
        }
    );

    cudaCheckError();

} // MeshToGrid<BuildT>::transformTriangles

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

template <typename BuildT>
struct CountRootBoxesFunctor
{
    const Triangle* dXformedTriangles;
    uint64_t* dCounts;
    float mBandWidth;

    __device__
    void operator()(size_t triangleID) const
    {
        const auto& tri = dXformedTriangles[triangleID];

        // Compute the strict AABB of the triangle in (cell-centered) index space
        float min_x = fminf(tri[0][0], fminf(tri[1][0], tri[2][0]));
        float min_y = fminf(tri[0][1], fminf(tri[1][1], tri[2][1]));
        float min_z = fminf(tri[0][2], fminf(tri[1][2], tri[2][2]));

        float max_x = fmaxf(tri[0][0], fmaxf(tri[1][0], tri[2][0]));
        float max_y = fmaxf(tri[0][1], fmaxf(tri[1][1], tri[2][1]));
        float max_z = fmaxf(tri[0][2], fmaxf(tri[1][2], tri[2][2]));

        // Find the exact integer Voxel/Cell indices we must cover
        // ceilf() gets the lowest integer >= min_valid
        // floorf() gets the highest integer <= max_valid
        float voxel_min_x = ceilf(min_x - mBandWidth);
        float voxel_min_y = ceilf(min_y - mBandWidth);
        float voxel_min_z = ceilf(min_z - mBandWidth);

        float voxel_max_x = floorf(max_x + mBandWidth);
        float voxel_max_y = floorf(max_y + mBandWidth);
        float voxel_max_z = floorf(max_z + mBandWidth);

        // Map the required voxels to Root Node index space
        const float invRootDim = 1.0f / 4096.0f;
        int min_i = static_cast<int>(floorf(voxel_min_x * invRootDim));
        int min_j = static_cast<int>(floorf(voxel_min_y * invRootDim));
        int min_k = static_cast<int>(floorf(voxel_min_z * invRootDim));

        int max_i = static_cast<int>(floorf(voxel_max_x * invRootDim));
        int max_j = static_cast<int>(floorf(voxel_max_y * invRootDim));
        int max_k = static_cast<int>(floorf(voxel_max_z * invRootDim));

        // Compute the 3D grid dimensions of overlapping root boxes
        uint64_t count_x = max_i - min_i + 1;
        uint64_t count_y = max_j - min_j + 1;
        uint64_t count_z = max_k - min_k + 1;

        // Write the total count of root boxes this triangle touches
        dCounts[triangleID] = count_x * count_y * count_z;
    }
};

template <typename BuildT>
struct ScatterRootTrianglePairsFunctor
{
    using PairT = typename MeshToGrid<BuildT>::BoxTrianglePair;

    const Triangle* dXformedTriangles;
    const uint64_t* dOffsets;
    PairT* dPairs;
    float mBandWidth;

    __device__
    void operator()(size_t triangleID) const
    {
        const auto& tri = dXformedTriangles[triangleID];

        // Recompute the strict AABB
        float min_x = fminf(tri[0][0], fminf(tri[1][0], tri[2][0]));
        float min_y = fminf(tri[0][1], fminf(tri[1][1], tri[2][1]));
        float min_z = fminf(tri[0][2], fminf(tri[1][2], tri[2][2]));

        float max_x = fmaxf(tri[0][0], fmaxf(tri[1][0], tri[2][0]));
        float max_y = fmaxf(tri[0][1], fmaxf(tri[1][1], tri[2][1]));
        float max_z = fmaxf(tri[0][2], fmaxf(tri[1][2], tri[2][2]));

        // Find the exact integer Voxel/Cell indices we must cover
        float voxel_min_x = ceilf(min_x - mBandWidth);
        float voxel_min_y = ceilf(min_y - mBandWidth);
        float voxel_min_z = ceilf(min_z - mBandWidth);

        float voxel_max_x = floorf(max_x + mBandWidth);
        float voxel_max_y = floorf(max_y + mBandWidth);
        float voxel_max_z = floorf(max_z + mBandWidth);

        // 3. Map the required voxels to Root Node index space
        const float invRootDim = 1.0f / 4096.0f;
        int min_i = static_cast<int>(floorf(voxel_min_x * invRootDim));
        int min_j = static_cast<int>(floorf(voxel_min_y * invRootDim));
        int min_k = static_cast<int>(floorf(voxel_min_z * invRootDim));

        int max_i = static_cast<int>(floorf(voxel_max_x * invRootDim));
        int max_j = static_cast<int>(floorf(voxel_max_y * invRootDim));
        int max_k = static_cast<int>(floorf(voxel_max_z * invRootDim));

        // Scatter the pairs into the global array
        uint64_t write_idx = dOffsets[triangleID];
        for (int k = min_k; k <= max_k; ++k)
        for (int j = min_j; j <= max_j; ++j)
        for (int i = min_i; i <= max_i; ++i) {
            // Multiply back by 4096 to get the true NanoVDB index-space origin
            dPairs[write_idx].origin = nanovdb::Coord(i * 4096, j * 4096, k * 4096);
            dPairs[write_idx].triangleID = static_cast<uint32_t>(triangleID);
            write_idx++;
        }
    }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::processRootTrianglePairs()
{
    if (mTriangleCount == 0) { mBoxTrianglePairCount = 0; return; }

    int device = 0;
    cudaGetDevice(&device);

    // Pass 1: Count intersecting root boxes per triangle

    nanovdb::cuda::DeviceBuffer
        rootBoxCounts = nanovdb::cuda::DeviceBuffer::create(mTriangleCount * sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxCounts.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box counts buffer");

    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount,
        topology::detail::CountRootBoxesFunctor<BuildT>{
            deviceXformedTriangles(),
            static_cast<uint64_t*>(rootBoxCounts.deviceData()),
            mBandWidth
        }
    );
    cudaCheckError();

    // Pass 2: InclusiveSum Scan to compute offsets and total allocations

    nanovdb::cuda::DeviceBuffer rootBoxOffsets =
        nanovdb::cuda::DeviceBuffer::create((mTriangleCount+1)*sizeof(uint64_t), nullptr, device, mStream);
    if (rootBoxOffsets.deviceData() == nullptr) throw std::runtime_error("Failed to allocate root box offsets buffer");

    cudaCheck(cudaMemsetAsync(rootBoxOffsets.deviceData(), 0, sizeof(uint64_t), mStream));
    CALL_CUBS(DeviceScan::InclusiveSum,
        static_cast<uint64_t*>(rootBoxCounts.deviceData()),
        static_cast<uint64_t*>(rootBoxOffsets.deviceData())+1,
        mTriangleCount);
    cudaCheck(cudaMemcpyAsync(&mBoxTrianglePairCount, static_cast<uint64_t*>(rootBoxOffsets.deviceData())+mTriangleCount, sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);

    // Pass 3: Re-enumerate intersections of (padded) root boxes and triangles, and scatter to allocated list

    mBoxTrianglePairsBuffer = nanovdb::cuda::DeviceBuffer::create(
        mBoxTrianglePairCount * sizeof(typename MeshToGrid<BuildT>::BoxTrianglePair), nullptr, device, mStream);
    if (mBoxTrianglePairsBuffer.deviceData() == nullptr) throw std::runtime_error("Failed to allocate pairs buffer");

    util::cuda::lambdaKernel<<<numBlocks(mTriangleCount), mNumThreads, 0, mStream>>>(
        mTriangleCount,
        topology::detail::ScatterRootTrianglePairsFunctor<BuildT>{
            deviceXformedTriangles(),
            static_cast<uint64_t*>(rootBoxOffsets.deviceData()),
            deviceBoxTrianglePairs(),
            mBandWidth
        }
    );

} // MeshToGrid<BuildT>::processRootTrianglePairs

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

/// @brief Scatters surviving child BoxTrianglePairs from a subdivision pass.
///        Launched with 1 thread per parent pair; each thread iterates over set bits
///        in the parent's Mask<3> and writes a new BoxTrianglePair for each surviving
///        child sub-box.
///
///        @note The 1-thread-per-parent approach is simple and efficient for sparse masks
///        (typical after SAT culling at finer scales). It does incur warp divergence since
///        different parents may have different child counts. An alternative would be a
///        1-CTA-per-parent launch (512 threads, one per child sub-box) with a warp-level
///        scan to compute local write offsets - better for dense masks but more complex.
///        Profile before switching.
template <typename BuildT>
struct ScatterChildPairsFunctor
{
    using PairT = typename MeshToGrid<BuildT>::BoxTrianglePair;

    const PairT*            dParents;
    const nanovdb::Mask<3>* dMasks;
    const uint64_t*         dOffsets;
    PairT*                  dNewPairs;
    int                     childScale;

    __device__
    void operator()(size_t parentID) const
    {
        const auto& parentPair = dParents[parentID];
        const auto& mask       = dMasks[parentID];
        uint64_t    writeIdx   = dOffsets[parentID];

        for (uint32_t n = 0; n < 512; ++n) {
            if (mask.isOn(n)) {
                int i =  n       & 0x7;
                int j = (n >> 3) & 0x7;
                int k = (n >> 6) & 0x7;
                dNewPairs[writeIdx].origin = nanovdb::Coord(
                    parentPair.origin[0] + i * childScale,
                    parentPair.origin[1] + j * childScale,
                    parentPair.origin[2] + k * childScale);
                dNewPairs[writeIdx].triangleID = parentPair.triangleID;
                ++writeIdx;
            }
        }
    }
};

/// @brief Tests if a Triangle intersects an Axis-Aligned Bounding Box.
template <bool OnlyUseAABB>
__device__ inline bool testTriangleAABB(
    const nanovdb::Vec3f& boxCenter,
    const nanovdb::Vec3f& boxHalfExtents,
    const nanovdb::Vec3f& V0,
    const nanovdb::Vec3f& V1,
    const nanovdb::Vec3f& V2)
{
    // Translate the triangle as if the AABB is centered at the origin
    nanovdb::Vec3f v0 = V0 - boxCenter;
    nanovdb::Vec3f v1 = V1 - boxCenter;
    nanovdb::Vec3f v2 = V2 - boxCenter;

    // --- PHASE 1: AABB OVERLAP (3 Axes) ---
    float minX = fminf(v0[0], fminf(v1[0], v2[0]));
    float maxX = fmaxf(v0[0], fmaxf(v1[0], v2[0]));
    if (minX > boxHalfExtents[0] || maxX < -boxHalfExtents[0]) return false;

    float minY = fminf(v0[1], fminf(v1[1], v2[1]));
    float maxY = fmaxf(v0[1], fmaxf(v1[1], v2[1]));
    if (minY > boxHalfExtents[1] || maxY < -boxHalfExtents[1]) return false;

    float minZ = fminf(v0[2], fminf(v1[2], v2[2]));
    float maxZ = fmaxf(v0[2], fmaxf(v1[2], v2[2]));
    if (minZ > boxHalfExtents[2] || maxZ < -boxHalfExtents[2]) return false;

    if constexpr (OnlyUseAABB) return true;

    // --- PHASE 2: SEPARATING AXIS THEOREM (SAT) (10 additional axes) ---
    nanovdb::Vec3f f0 = v1 - v0, f1 = v2 - v1, f2 = v0 - v2;
    float r, p0, p1, p2;

    // Axis 00, 01, 02 (X-axis cross products)
    p0 = v0[2]*f0[1] - v0[1]*f0[2]; p2 = v2[2]*f0[1] - v2[1]*f0[2];
    r = boxHalfExtents[1]*fabsf(f0[2]) + boxHalfExtents[2]*fabsf(f0[1]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[2]*f1[1] - v0[1]*f1[2]; p1 = v1[2]*f1[1] - v1[1]*f1[2];
    r = boxHalfExtents[1]*fabsf(f1[2]) + boxHalfExtents[2]*fabsf(f1[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[2]*f2[1] - v0[1]*f2[2]; p1 = v1[2]*f2[1] - v1[1]*f2[2];
    r = boxHalfExtents[1]*fabsf(f2[2]) + boxHalfExtents[2]*fabsf(f2[1]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Axis 10, 11, 12 (Y-axis cross products)
    p0 = v0[0]*f0[2] - v0[2]*f0[0]; p2 = v2[0]*f0[2] - v2[2]*f0[0];
    r = boxHalfExtents[0]*fabsf(f0[2]) + boxHalfExtents[2]*fabsf(f0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[0]*f1[2] - v0[2]*f1[0]; p1 = v1[0]*f1[2] - v1[2]*f1[0];
    r = boxHalfExtents[0]*fabsf(f1[2]) + boxHalfExtents[2]*fabsf(f1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[0]*f2[2] - v0[2]*f2[0]; p1 = v1[0]*f2[2] - v1[2]*f2[0];
    r = boxHalfExtents[0]*fabsf(f2[2]) + boxHalfExtents[2]*fabsf(f2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Axis 20, 21, 22 (Z-axis cross products)
    p0 = v0[1]*f0[0] - v0[0]*f0[1]; p2 = v2[1]*f0[0] - v2[0]*f0[1];
    r = boxHalfExtents[0]*fabsf(f0[1]) + boxHalfExtents[1]*fabsf(f0[0]);
    if (fminf(p0, p2) > r || fmaxf(p0, p2) < -r) return false;

    p0 = v0[1]*f1[0] - v0[0]*f1[1]; p1 = v1[1]*f1[0] - v1[0]*f1[1];
    r = boxHalfExtents[0]*fabsf(f1[1]) + boxHalfExtents[1]*fabsf(f1[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    p0 = v0[1]*f2[0] - v0[0]*f2[1]; p1 = v1[1]*f2[0] - v1[0]*f2[1];
    r = boxHalfExtents[0]*fabsf(f2[1]) + boxHalfExtents[1]*fabsf(f2[0]);
    if (fminf(p0, p1) > r || fmaxf(p0, p1) < -r) return false;

    // Face normal test
    nanovdb::Vec3f n(f0[1]*f1[2] - f0[2]*f1[1], f0[2]*f1[0] - f0[0]*f1[2], f0[0]*f1[1] - f0[1]*f1[0]);
    float d = n[0]*v0[0] + n[1]*v0[1] + n[2]*v0[2];
    r = boxHalfExtents[0]*fabsf(n[0]) + boxHalfExtents[1]*fabsf(n[1]) + boxHalfExtents[2]*fabsf(n[2]);
    if (fabsf(d) > r) return false;

    return true;
}

template <typename BuildT, bool OnlyUseAABB>
__global__ void evaluateAndCountSubBoxesKernel(
    const typename MeshToGrid<BuildT>::BoxTrianglePair* dParents,
    const Triangle* dXformedTriangles,
    nanovdb::Mask<3>* dMasks,
    uint64_t* dCounts,
    int parentScale,
    float padding)
{
    // 1 CTA exactly evaluates/refines 1 parent Pair
    uint64_t parentID = blockIdx.x;
    int threadID = threadIdx.x;

    const auto& parentPair = dParents[parentID];
    const auto& tri = dXformedTriangles[parentPair.triangleID];

    int childScale = parentScale / 8;

    // Thread to 3D sub-box index mapping
    int i =  threadID       & 0x7;
    int j = (threadID >> 3) & 0x7;
    int k = (threadID >> 6) & 0x7;

    // Compute voxel-encompassing bounding box for this subdomain
    // Voxel bounds are [origin - 0.5, origin + childScale - 0.5]
    float centerX = parentPair.origin[0] + i * childScale + (childScale * 0.5f) - 0.5f;
    float centerY = parentPair.origin[1] + j * childScale + (childScale * 0.5f) - 0.5f;
    float centerZ = parentPair.origin[2] + k * childScale + (childScale * 0.5f) - 0.5f;

    nanovdb::Vec3f boxCenter(centerX, centerY, centerZ);
    float halfExt = (childScale * 0.5f) + padding;
    nanovdb::Vec3f boxHalfExtents(halfExt, halfExt, halfExt);

    // Evaluate intersection
    bool hit = testTriangleAABB<OnlyUseAABB>(boxCenter, boxHalfExtents, tri[0], tri[1], tri[2]);

    // Mask Building without using atomics, via Warp Voting
    // threadID = i + j*8 + k*64 matches NanoVDB Mask<3> bit ordering exactly,
    // so warp ballots map directly into the mask with no stitching required.
    // We use reinterpret_cast rather than a union to avoid Mask<3>'s non-trivial
    // constructor deleting the union's default constructor.
    __shared__ alignas(nanovdb::Mask<3>) uint32_t s_words32[16]; // 16 warps * 32 bits = 512 bits total

    unsigned int ballot = __ballot_sync(0xFFFFFFFF, hit);
    if ((threadID & 31) == 0) {
        // The first thread of each warp writes its ballot directly into the mask
        s_words32[threadID >> 5] = ballot;
    }

    __syncthreads();

    // Thread 0 flushes the mask and its popcount to global memory
    if (threadID == 0) {
        const auto& outMask = reinterpret_cast<const nanovdb::Mask<3>&>(s_words32);
        dMasks[parentID] = outMask;
        dCounts[parentID] = outMask.countOn();
    }
}

} // namespace topology::detail

namespace topology::detail {

// coordToKey / keyToCoord: encode/decode a root-tile Coord origin as a
// sortable uint64_t key, following the convention from PointsToGrid.cuh.
//   key     = (field_x << 42) | (field_y << 21) | field_z
//   field_i = uint32_t(int64_t(ijk[i]) + kOffset) >> 12
// The >>12 groups all voxels sharing a 4096^3 root tile to the same key.
// kOffset = 1<<31 ensures the full int32_t coordinate range maps to [0, 2^32-1]
// before the uint32_t cast, making the encoding well-defined for all coordinates.

static constexpr int64_t kRootTileKeyOffset = int64_t(1) << 31;

__device__ inline uint64_t coordToKey(const nanovdb::Coord& ijk)
{
    return  (uint64_t(uint32_t(int64_t(ijk[0]) + kRootTileKeyOffset) >> 12) << 42) |
            (uint64_t(uint32_t(int64_t(ijk[1]) + kRootTileKeyOffset) >> 12) << 21) |
             uint64_t(uint32_t(int64_t(ijk[2]) + kRootTileKeyOffset) >> 12);
}

__device__ inline nanovdb::Coord keyToCoord(uint64_t key)
{
    const uint32_t field_x = uint32_t(key >> 42);
    const uint32_t field_y = uint32_t((key >> 21) & 0x1FFFFF);
    const uint32_t field_z = uint32_t(key & 0x1FFFFF);
    return nanovdb::Coord(
        int32_t(int64_t(uint64_t(field_x) << 12) - kRootTileKeyOffset),
        int32_t(int64_t(uint64_t(field_y) << 12) - kRootTileKeyOffset),
        int32_t(int64_t(uint64_t(field_z) << 12) - kRootTileKeyOffset));
}

template <typename BuildT>
struct EncodeRootOriginsFunctor
{
    const typename MeshToGrid<BuildT>::BoxTrianglePair* dPairs;
    uint64_t*                                           dKeys;

    __device__ void operator()(size_t i) const { dKeys[i] = coordToKey(dPairs[i].origin); }
};

template <typename BuildT>
struct DecodeRootOriginsFunctor
{
    const uint64_t*  dKeys;
    nanovdb::Coord*  dOrigins;

    __device__ void operator()(size_t i) const { dOrigins[i] = keyToCoord(dKeys[i]); }
};

} // namespace topology::detail

template<typename BuildT>
void MeshToGrid<BuildT>::enumerateRootTiles()
{
    if (mBoxTrianglePairCount == 0) return;

    int device = 0;
    cudaGetDevice(&device);

    // Step 1: Encode each pair's root origin as a sortable uint64_t key
    nanovdb::cuda::DeviceBuffer keysBuffer = nanovdb::cuda::DeviceBuffer::create(
        mBoxTrianglePairCount * sizeof(uint64_t), nullptr, device, mStream);
    auto *dKeys = static_cast<uint64_t*>(keysBuffer.deviceData());

    util::cuda::lambdaKernel<<<numBlocks(mBoxTrianglePairCount), mNumThreads, 0, mStream>>>(
        mBoxTrianglePairCount,
        topology::detail::EncodeRootOriginsFunctor<BuildT>{ deviceBoxTrianglePairs(), dKeys }
    );
    cudaCheckError();

    // Step 2: Sort keys (SortKeys requires separate in/out buffers)
    nanovdb::cuda::DeviceBuffer sortedKeysBuffer = nanovdb::cuda::DeviceBuffer::create(
        mBoxTrianglePairCount * sizeof(uint64_t), nullptr, device, mStream);
    auto *dSortedKeys = static_cast<uint64_t*>(sortedKeysBuffer.deviceData());

    CALL_CUBS(DeviceRadixSort::SortKeys, dKeys, dSortedKeys, (int)mBoxTrianglePairCount, 0, 64);

    // Step 3: Select unique keys
    nanovdb::cuda::DeviceBuffer uniqueKeysBuffer = nanovdb::cuda::DeviceBuffer::create(
        mBoxTrianglePairCount * sizeof(uint64_t), nullptr, device, mStream);
    auto *dUniqueKeys = static_cast<uint64_t*>(uniqueKeysBuffer.deviceData());

    nanovdb::cuda::DeviceBuffer numSelectedBuffer = nanovdb::cuda::DeviceBuffer::create(
        sizeof(int32_t), nullptr, device, mStream);
    auto *dNumSelected = static_cast<int32_t*>(numSelectedBuffer.deviceData());

    CALL_CUBS(DeviceSelect::Unique, dSortedKeys, dUniqueKeys, dNumSelected, (int)mBoxTrianglePairCount);

    int32_t uniqueCount = 0;
    cudaCheck(cudaMemcpyAsync(&uniqueCount, dNumSelected, sizeof(int32_t), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
    mUniqueRootTileCount = static_cast<uint64_t>(uniqueCount);

    // Step 4: Decode unique keys back to Coord origins
    mUniqueRootOriginsBuffer = nanovdb::cuda::DeviceBuffer::create(
        mUniqueRootTileCount * sizeof(nanovdb::Coord), nullptr, device, mStream);
    auto *dOrigins = deviceUniqueRootOrigins();

    util::cuda::lambdaKernel<<<numBlocks(mUniqueRootTileCount), mNumThreads, 0, mStream>>>(
        mUniqueRootTileCount,
        topology::detail::DecodeRootOriginsFunctor<BuildT>{ dUniqueKeys, dOrigins }
    );
    cudaCheckError();

} // MeshToGrid<BuildT>::enumerateRootTiles

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::buildRasterizedRoot()
{
    int device = 0;
    cudaGetDevice(&device);

    // Download unique root origins to host.
    // Origins are already in coordToKey-sorted order from enumerateRootTiles(),
    // so no further sorting or deduplication is required here.
    uint32_t tileCount = static_cast<uint32_t>(mUniqueRootTileCount);

    // Build the root node on CPU: one tile per unique root origin.
    // Only the NanoVDB tile key is set here; child pointers and values are
    // filled by TopologyBuilder's subsequent pipeline stages.
    uint64_t rootSize = RootT::memUsage(tileCount);
    mBuilder.mProcessedRoot = nanovdb::cuda::DeviceBuffer::create(rootSize);
    auto *rootPtr = static_cast<RootT*>(mBuilder.mProcessedRoot.data());
    rootPtr->mTableSize = tileCount;
    rootPtr->mBackground = typename RootT::ValueType{};

    if (tileCount) {
        std::vector<nanovdb::Coord> hostOrigins(tileCount);
        cudaCheck(cudaMemcpy(hostOrigins.data(),
            deviceUniqueRootOrigins(),
            tileCount * sizeof(nanovdb::Coord), cudaMemcpyDeviceToHost));
        for (uint32_t t = 0; t < tileCount; ++t)
            *rootPtr->tile(t) = typename RootT::DataType::Tile{RootT::CoordToKey(hostOrigins[t])};
        mBuilder.mProcessedRoot.deviceUpload(device, mStream, false);
        mUniqueRootOriginsBuffer.clear(mStream);
    }
} // MeshToGrid<BuildT>::buildRasterizedRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::rasterizeInternalNodes()
{
    if (mBoxTrianglePairCount == 0) return;

    using RasterizerT = util::rasterization::cuda::RasterizeInternalNodesFunctor<BuildT, BoxTrianglePair>;

    auto *dUpperMasks = static_cast<Mask<5>*>(mBuilder.deviceUpperMasks());
    auto *dLowerMasks = static_cast<Mask<4>(*)[Mask<5>::SIZE]>(mBuilder.deviceLowerMasks());

    util::cuda::lambdaKernel<<<numBlocks(mBoxTrianglePairCount), mNumThreads, 0, mStream>>>(
        mBoxTrianglePairCount,
        RasterizerT{ deviceBoxTrianglePairs(), mBuilder.deviceProcessedRoot(), dUpperMasks, dLowerMasks }
    );
    cudaCheckError();

} // MeshToGrid<BuildT>::rasterizeInternalNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::processGridTreeRoot()
{
    // Initialize grid/tree/root metadata from scratch using the provided map.
    // InitGridTreeRootFunctor sets all GridData fields explicitly (magic, version,
    // flags, map, gridClass=IndexGrid, etc.) since there is no source grid to copy from.
    util::cuda::lambdaKernel<<<1, 1, 0, mStream>>>(1,
        topology::detail::InitGridTreeRootFunctor<BuildT>{mMap}, mBuilder.deviceData());
    cudaCheckError();

    // Copy grid name into the output grid's name field
    char *dst = mBuilder.data()->getGrid().mGridName;
    if (!mGridName.empty()) {
        cudaCheck(cudaMemcpyAsync(dst, mGridName.data(), GridData::MaxNameSize, cudaMemcpyHostToDevice, mStream));
    } else {
        cudaCheck(cudaMemsetAsync(dst, 0, GridData::MaxNameSize, mStream));
    }

} // MeshToGrid<BuildT>::processGridTreeRoot

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::rasterizeLeafNodes()
{
    if (mBoxTrianglePairCount == 0) return;

    using FunctorT = util::rasterization::cuda::RasterizeLeafNodesFunctor<BuildT, BoxTrianglePair, Triangle>;
    util::cuda::operatorKernelInstance<FunctorT>
        <<<mBoxTrianglePairCount, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            FunctorT{ deviceBoxTrianglePairs(), deviceXformedTriangles(),
                      &mBuilder.data()->getGrid(), mBandWidth * mBandWidth });
    cudaCheckError();

} // MeshToGrid<BuildT>::rasterizeLeafNodes

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
void MeshToGrid<BuildT>::processLeafTrianglePairs()
{
    if (mBoxTrianglePairCount == 0) return;

    int device = 0;
    cudaGetDevice(&device);

    int scale = 4096; // Start at Root node scale

    for (int pass = 0; pass < 3; ++pass) {
        // Allocate Mask<3> buffer for the CTA hit results
        // Size: mBoxTrianglePairCount * sizeof(nanovdb::Mask<3>)
        nanovdb::cuda::DeviceBuffer maskBuffer = nanovdb::cuda::DeviceBuffer::create(
            mBoxTrianglePairCount * sizeof(nanovdb::Mask<3>), nullptr, device, mStream);
        if (maskBuffer.deviceData() == nullptr) {
            throw std::runtime_error("Failed to allocate mask buffer for subdivision pass");
        }
        auto* dMasks = static_cast<nanovdb::Mask<3>*>(maskBuffer.deviceData());

        // Allocate Counts buffer for Prefix Sum
        // Size: mBoxTrianglePairCount * sizeof(uint64_t)
        nanovdb::cuda::DeviceBuffer countsBuffer = nanovdb::cuda::DeviceBuffer::create(
            mBoxTrianglePairCount * sizeof(uint64_t), nullptr, device, mStream);
        if (countsBuffer.deviceData() == nullptr) {
            throw std::runtime_error("Failed to allocate counts buffer for subdivision pass");
        }
        auto* dCounts = static_cast<uint64_t*>(countsBuffer.deviceData());

        // Evaluate & Count: 1 CTA per parent pair, 512 threads per CTA.
        // Uses AABB-only test for large child scales (>= mSATThreshold), full SAT below.
        // padding = mBandWidth: tight bound would be mBandWidth-0.5 (geometric box already
        // extends 0.5 beyond outermost cell centers), but we add 0.5 extra for safety.
        int childScale = scale / 8;
        const float padding = mBandWidth;
        if (childScale >= mSATThreshold)
            topology::detail::evaluateAndCountSubBoxesKernel<BuildT, true>
                <<<mBoxTrianglePairCount, 512, 0, mStream>>>(
                    deviceBoxTrianglePairs(), deviceXformedTriangles(), dMasks, dCounts, scale, padding);
        else
            topology::detail::evaluateAndCountSubBoxesKernel<BuildT, false>
                <<<mBoxTrianglePairCount, 512, 0, mStream>>>(
                    deviceBoxTrianglePairs(), deviceXformedTriangles(), dMasks, dCounts, scale, padding);
        cudaCheckError();

        // Prefix Sum: element [i+1] = exclusive write offset for parent i's children,
        // element [0] = 0, element [mBoxTrianglePairCount] = total child pair count.
        nanovdb::cuda::DeviceBuffer offsetsBuffer = nanovdb::cuda::DeviceBuffer::create(
            (mBoxTrianglePairCount + 1) * sizeof(uint64_t), nullptr, device, mStream);
        if (offsetsBuffer.deviceData() == nullptr)
            throw std::runtime_error("Failed to allocate offsets buffer for subdivision pass");
        auto* dOffsets = static_cast<uint64_t*>(offsetsBuffer.deviceData());

        cudaCheck(cudaMemsetAsync(dOffsets, 0, sizeof(uint64_t), mStream));
        CALL_CUBS(DeviceScan::InclusiveSum,
            dCounts,
            dOffsets + 1,
            mBoxTrianglePairCount);

        uint64_t newPairCount = 0;
        cudaCheck(cudaMemcpyAsync(&newPairCount, dOffsets + mBoxTrianglePairCount,
            sizeof(uint64_t), cudaMemcpyDeviceToHost, mStream));
        cudaStreamSynchronize(mStream);

        // Allocate new child pair buffer
        nanovdb::cuda::DeviceBuffer newPairsBuffer = nanovdb::cuda::DeviceBuffer::create(
            newPairCount * sizeof(BoxTrianglePair), nullptr, device, mStream);
        if (newPairsBuffer.deviceData() == nullptr)
            throw std::runtime_error("Failed to allocate child pairs buffer for subdivision pass");
        auto* dNewPairs = static_cast<BoxTrianglePair*>(newPairsBuffer.deviceData());

        // Scatter surviving child pairs into the new buffer
        util::cuda::lambdaKernel<<<numBlocks(mBoxTrianglePairCount), mNumThreads, 0, mStream>>>(
            mBoxTrianglePairCount,
            topology::detail::ScatterChildPairsFunctor<BuildT>{
                deviceBoxTrianglePairs(), dMasks, dOffsets, dNewPairs, childScale
            }
        );
        cudaCheckError();

        // Ping-pong: replace the parent pair buffer with the new child pair buffer
        mBoxTrianglePairsBuffer = std::move(newPairsBuffer);
        mBoxTrianglePairCount   = newPairCount;

        scale /= 8;
    }

} // MeshToGrid<BuildT>::processLeafTrianglePairs

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace topology::detail {

/// @brief Fills each sidecar slot with the FLT_MAX bit pattern (0x7F7FFFFFu),
///        serving as a "not yet written" sentinel for the UDF atomic-min pass.
struct InitSidecarFunctor
{
    uint32_t *dSidecar;
    __device__ void operator()(size_t i) const { dSidecar[i] = 0x7F7FFFFFu; }
};

/// @brief Finalizes the sidecar after the atomic-min UDF pass, emitting world-space distances:
///          - slot 0 (background) <- mBandWidth * voxelSize
///          - slot k (active)     <- sqrtf(distSqr) * voxelSize  if a triangle contributed
///                                <- mBandWidth * voxelSize       otherwise (false-positive voxels)
struct FinalizeSidecarFunctor
{
    float *dSidecar;
    float  bandWidth; // world-space = mBandWidth * voxelSize
    float  voxelSize;

    __device__ void operator()(size_t i) const
    {
        if (i == 0) { dSidecar[0] = bandWidth; return; }
        const uint32_t bits = __float_as_uint(dSidecar[i]);
        dSidecar[i] = (bits == 0x7F7FFFFFu) ? bandWidth : sqrtf(dSidecar[i]) * voxelSize;
    }
};

} // namespace topology::detail

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT>
template<typename GridBufferT, typename SidecarBufferT>
std::pair<GridHandle<GridBufferT>, SidecarBufferT>
MeshToGrid<BuildT>::getHandleAndUDF(const GridBufferT& buffer, const SidecarBufferT&)
{
    cudaStreamSynchronize(mStream);

    // ---- Topology pipeline (mirrors getHandle) ----

    if (mVerbose==1) mTimer.start("Transforming triangles to grid index space");
    transformTriangles();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Computing candidate RootTile-Triangle intersection pairs");
    processRootTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Enumerating unique root tiles");
    enumerateRootTiles();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Computing candidate LeafNode-Triangle intersection pairs");
    processLeafTrianglePairs();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Building rasterized root node");
    buildRasterizedRoot();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Allocating internal mask buffers");
    mBuilder.allocateInternalMaskBuffers(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Rasterizing internal nodes");
    rasterizeInternalNodes();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Counting nodes");
    mBuilder.countNodes(mStream);
    cudaStreamSynchronize(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Allocating grid buffer");
    auto gridBuffer = mBuilder.getBuffer(buffer, mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Processing grid/tree/root");
    processGridTreeRoot();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Processing upper nodes");
    mBuilder.processUpperNodes(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Processing lower nodes");
    mBuilder.processLowerNodes(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Rasterizing leaf nodes");
    rasterizeLeafNodes();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Processing leaf offsets");
    mBuilder.processLeafOffsets(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Processing bounding boxes");
    mBuilder.processBBox(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Post-processing grid/tree data");
    mBuilder.postProcessGridTree(mStream);
    cudaStreamSynchronize(mStream);
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Pruning empty leaves");
    int device = 0; cudaGetDevice(&device);
    const uint32_t leafCount = mBuilder.data()->nodeCount[0];
    auto handle = GridHandle<GridBufferT>(std::move(gridBuffer));
    if (leafCount) {
        nanovdb::cuda::DeviceBuffer retainMaskBuffer = nanovdb::cuda::DeviceBuffer::create(
            uint64_t(leafCount) * sizeof(nanovdb::Mask<3>), nullptr, device, mStream);
        cudaCheck(cudaMemsetAsync(retainMaskBuffer.deviceData(), 0xFF,
            uint64_t(leafCount) * sizeof(nanovdb::Mask<3>), mStream));
        tools::cuda::PruneGrid<BuildT> pruner(
            static_cast<const GridT*>(handle.deviceData()),
            static_cast<nanovdb::Mask<3>*>(retainMaskBuffer.deviceData()),
            mStream);
        handle = pruner.template getHandle<GridBufferT>(buffer);
    }
    if (mVerbose==1) mTimer.stop();

    // ---- UDF sidecar ----

    const float voxelSize = (float)mMap.getVoxelSize()[0];

    const uint64_t activeVoxelCount = util::cuda::DeviceGridTraits<BuildT>::getActiveVoxelCount(
        handle.template deviceGrid<BuildT>());

    auto sidecarBuffer = nanovdb::cuda::DeviceBuffer::create(
        (activeVoxelCount + 1) * sizeof(float), nullptr, device, mStream);
    auto *dSidecar = static_cast<float*>(sidecarBuffer.deviceData());

    if (mVerbose==1) mTimer.start("Initializing UDF sidecar");
    util::cuda::lambdaKernel<<<numBlocks(activeVoxelCount + 1), mNumThreads, 0, mStream>>>(
        activeVoxelCount + 1,
        topology::detail::InitSidecarFunctor{ reinterpret_cast<uint32_t*>(dSidecar) });
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Computing UDF via leaf/triangle pairs");
    if (mBoxTrianglePairCount) {
        using UDFFunctorT = util::rasterization::cuda::ComputeUDFFunctor<BuildT, BoxTrianglePair, Triangle>;
        util::cuda::operatorKernelInstance<UDFFunctorT>
            <<<mBoxTrianglePairCount, UDFFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
                UDFFunctorT{ deviceBoxTrianglePairs(), deviceXformedTriangles(),
                             handle.template deviceGrid<BuildT>(), dSidecar,
                             mBandWidth * mBandWidth });
        cudaCheckError();
        mXformedTriangles.clear(mStream);
        mBoxTrianglePairsBuffer.clear(mStream);
    }
    if (mVerbose==1) mTimer.stop();

    if (mVerbose==1) mTimer.start("Finalizing UDF sidecar (sqrt + clamp)");
    util::cuda::lambdaKernel<<<numBlocks(activeVoxelCount + 1), mNumThreads, 0, mStream>>>(
        activeVoxelCount + 1,
        topology::detail::FinalizeSidecarFunctor{ dSidecar, mBandWidth * voxelSize, voxelSize });
    cudaCheckError();
    if (mVerbose==1) mTimer.stop();

    cudaStreamSynchronize(mStream);

    return { std::move(handle), std::move(sidecarBuffer) };
} // MeshToGrid<BuildT>::getHandleAndUDF

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_MESHTOGRID_CUH_HAS_BEEN_INCLUDED
