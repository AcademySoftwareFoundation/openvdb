// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh
    \authors Efty Sifakis and JaeHyun Lee
    \brief GPU conversion of triangle meshes to narrow-band signed distance fields on NanoVDB
           index grids.

    \details The pipeline rasterizes a UDF, partitions its band into surfaces, signs each surface,
             and composes nested surfaces. It returns the grid and the signed distances of its active
             voxels; tools::cuda::PropagateSign extends the sign beyond the active voxels.

    \warning This header contains CUDA device code and must be included from a .cu or .cuh file.
*/

#ifndef NANOVDB_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/math/Proximity.h>                    // closestPointOnTriangleToPoint
#include <nanovdb/tools/cuda/ConnectedComponents.cuh>
#include <nanovdb/tools/cuda/MeshToGrid.cuh>
#include <nanovdb/tools/cuda/PartitionGrid.cuh>
#include <nanovdb/tools/cuda/PropagateSign.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>             // InjectGridDataFunctor
#include <nanovdb/util/cuda/DeviceGridTraits.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h>                    // operatorKernel, cudaCheck

#include <cuda/std/algorithm> // cuda::std::binary_search

#include <cstddef>   // std::byte
#include <cstdint>   // uint8_t, uint32_t, uint64_t
#include <stdexcept> // std::runtime_error
#include <tuple>     // std::tie
#include <utility>   // std::move, std::pair
#include <vector>    // std::vector

namespace nanovdb {

namespace tools::cuda {

/// @brief Convert closed triangle surfaces to a narrow-band signed distance field.
/// @warning The map must scale uniformly: one voxel size is used for all three axes.
/// @note A surface is a connected component of the rasterized narrow band, so meshes whose bands touch are signed
///       as one surface, and the nesting rule applies only between surfaces whose bands are apart.
/// @tparam BuildT Build type of the index grid (e.g. nanovdb::ValueOnIndex).
/// @tparam ResourceT Stream-ordered resource used for the output and all scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class MeshToSDF
{
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "MeshToSDF allocates stream-ordered storage and requires an AsyncResource");

    using PointT         = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using GridT          = NanoGrid<BuildT>;
    using TreeT          = NanoTree<BuildT>;
    using TraitsT        = util::cuda::DeviceGridTraits<BuildT>;
    using SurfaceLabelT  = ConnectedComponentsBase::ComponentLabelT;

    /// @brief Stream-ordered, device-only storage borrowed from the injected resource.
    template <typename T>
    using BufferT        = nanovdb::cuda::Buffer<T, nanovdb::cuda::ResourceRef<ResourceT>>;
public:
    using HandleT        = GridHandle<BufferT<std::byte>>;

    /// @brief Construct from device-resident mesh data. Processing starts in getHandleAndSDF().
    /// @param devicePoints   device vertex list in world space
    /// @param pointCount     vertex count
    /// @param deviceTriangles device triangle vertex-index list
    /// @param triangleCount  triangle count
    /// @param map            world-to-index transform
    /// @param stream         CUDA stream
    /// @param resource       resource instance the output and all device scratch are allocated from; must outlive
    ///                       the returned grid and distances
    MeshToSDF(const PointT* devicePoints, uint32_t pointCount,
              const TriangleIndexT* deviceTriangles, uint32_t triangleCount,
              const nanovdb::Map& map = nanovdb::Map(), cudaStream_t stream = 0,
              ResourceT& resource = nanovdb::cuda::default_resource<ResourceT>())
        : mDevicePoints(devicePoints), mPointCount(pointCount)
        , mDeviceTriangles(deviceTriangles), mTriangleCount(triangleCount)
        , mMap(map), mStream(stream), mTimer(stream), mResource(&resource)
        , mGridHandle(BufferT<std::byte>(stream, resource, 0, nanovdb::cuda::noInit))
        , mSDF(stream, resource, 0, nanovdb::cuda::noInit)
        , mTriangleIndex(stream, resource, 0, nanovdb::cuda::noInit)
        , mSurfaceLabels(stream, resource, 0, nanovdb::cuda::noInit)
        , mDeviceSurfaceGrids(stream, resource, 0, nanovdb::cuda::noInit)
        , mNestingDepth(stream, resource, 0, nanovdb::cuda::noInit) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of the narrow band
    /// @param bandWidth Narrow band width in cell units
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief Policy for signing voxels in the surface barrier.
    enum class BarrierSigning {
        Interior,   ///< classify all barrier voxels as interior
        Heuristic   ///< use the OpenVDB intersecting-voxel heuristic
    };

    /// @brief Choose the barrier signing method (default Heuristic).
    void setBarrierSigning(BarrierSigning m) { mBarrierSigning = m; }

    /// @brief How a point enclosed by several of the input's closed surfaces is signed.
    enum class NestingRule {
        EvenOdd,  ///< inside when enclosed by an odd number of surfaces
        Solid     ///< inside when enclosed by one or more surfaces
    };

    /// @brief Choose the nesting rule (default EvenOdd).
    void setNestingRule(NestingRule r) { mNestingRule = r; }

    /// @brief Sign the isosurface where the mesh UDF equals @a isoValue.
    /// @param isoValue Non-negative offset in world units; zero signs the input surface.
    /// @note Rasterization includes the offset, so time and memory increase with @a isoValue.
    void setIsoValue(float isoValue = 0.f) { mIsoValue = isoValue; }

    /// @brief Run the pipeline and return the index grid and its signed distances.
    /// @return { grid, sdf }: the narrow-band index grid, and a sidecar of (activeVoxelCount+1) floats indexed
    ///         like the grid's values: sdf[0] is the positive background, and sdf[k>0] is the signed distance of
    ///         the k-th active voxel to the signed surface, negative inside. Pass both to PropagateSign for the
    ///         sign of the inactive voxels and tiles.
    /// @throw std::runtime_error if the iso value is negative.
    /// @note Call once: the grid and the distances are moved out of this object.
    std::pair<HandleT, BufferT<std::byte>> getHandleAndSDF()
    {
        if (mIsoValue < 0.f)
            throw std::runtime_error("MeshToSDF: setIsoValue() must be >= 0");

        if (mVerbose==1) mTimer.start("\nRasterizing mesh");
        rasterize();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Partitioning surfaces");
        partitionSurfaces();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Processing surfaces");
        for (SurfaceLabelT surfaceID = 0; surfaceID < mSurfaceCount; ++surfaceID)
            processSurface(surfaceID);
        if (mVerbose==1) mTimer.stop();

        // Each intermediate is released right after its last use, to cut peak memory. The frees are ordered on
        // mStream behind the kernels that read them.
        mSurfaceGrids.clear();
        mDeviceSurfaceGrids.destroy();
        mTriangleIndex.destroy();

        if (mVerbose==1) mTimer.start("Resolving signs on the original grid");
        resolveSigns();
        if (mVerbose==1) mTimer.stop();

        mSurfaceLabels.destroy();
        mNestingDepth.destroy();

        if (mVerbose==1) mTimer.start("Finalizing distances");
        finalizeDistances();
        if (mVerbose==1) mTimer.stop();

        cudaCheck(cudaStreamSynchronize(mStream));
        return {std::move(mGridHandle), std::move(mSDF)};
    }

private:
    // --- Pipeline stages (run in order by getHandleAndSDF). ---

    // Stage 1: rasterize the mesh -> the grid, and the unsigned distance and nearest triangle of each voxel.
    void rasterize();

    // Stage 2: split the grid into its surfaces, the connected components of its active voxels.
    void partitionSurfaces();

    // Stage 3, once per surface: sign the distances of the surface (steps 3a-3c), then add one to the nesting
    // depth of every other surface it encloses (3d).
    void processSurface(uint32_t surfaceID);

    // Stage 3a: the grid of the surface without its barrier, the voxels nearest the signed surface.
    HandleT pruneBarrier(const GridT* dGrid, const float* dUDF);

    // Stage 3b: negate the distances outside the barrier that are not in the exterior component.
    BufferT<unsigned long long> findExteriorLabel(const GridT* dDerivedGrid, const uint32_t* dVoxelLabel);
    void signNonBarrierComponents(const GridT* dGrid, const GridT* dDerivedGrid, const uint32_t* dVoxelLabel,
                                  float* dSDF);

    // Stage 3c: sign the distances in the barrier by the chosen BarrierSigning.
    void signBarrier(const GridT* dGrid, float* dSDF, const uint32_t* dTriangleIndex);

    // Stage 4: with several surfaces, compose their signs by the nesting rule.
    void resolveSigns();

    // Stage 5: with a nonzero iso value, convert the distances from the mesh to distances from the signed surface.
    void finalizeDistances();

    /// @brief Borrowed handle to the injected resource, for the storage allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    const PointT*         mDevicePoints{nullptr};
    uint32_t              mPointCount{0};
    const TriangleIndexT* mDeviceTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    cudaStream_t          mStream{0};
    util::cuda::Timer     mTimer;
    ResourceT*            mResource{nullptr};
    int                   mVerbose{0};
    float                 mBandWidth{3.f};
    BarrierSigning        mBarrierSigning{BarrierSigning::Heuristic};
    float                 mIsoValue{0.f};  // world units, >= 0; see setIsoValue()
    // Per-voxel sidecars include slot 0 for the background.
    HandleT                     mGridHandle;
    typename TreeT::DataType    mTreeData{};    // host copy of the rasterized grid's tree header, read by rasterize()
    BufferT<std::byte>          mSDF;           // float per voxel slot: the UDF from MeshToGrid, signed in place
    BufferT<std::byte>          mTriangleIndex; // uint32_t per voxel slot, as returned by MeshToGrid
    BufferT<SurfaceLabelT>      mSurfaceLabels;
    SurfaceLabelT               mSurfaceCount{0};
    std::vector<HandleT>        mSurfaceGrids;  // [s] the voxels of surface s, when there are several surfaces
    BufferT<const GridT*>       mDeviceSurfaceGrids; // [s] device pointer of mSurfaceGrids[s]
    BufferT<uint32_t>           mNestingDepth;

    NestingRule                 mNestingRule{NestingRule::EvenOdd};

}; // tools::cuda::MeshToSDF<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 1: rasterization.

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::rasterize()
{
    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // Preserve the requested band width beyond the offset surface.
    const float extra = (voxelSize > 0.f) ? mIsoValue / voxelSize : 0.f;

    MeshToGrid<BuildT, ResourceT> converter(mDevicePoints, mPointCount, mDeviceTriangles, mTriangleCount, mMap, mStream,
                                            *mResource);
    converter.setNarrowBandWidth(mBandWidth + extra);
    std::tie(mGridHandle, mSDF, mTriangleIndex) =
        converter.template getHandleAndUDFAndIndex<BufferT<std::byte>, BufferT<std::byte>>(mGridHandle.buffer(), mSDF);
    mTreeData = TraitsT::getTreeData(mGridHandle.template deviceGrid<BuildT>(), mStream);
} // MeshToSDF<BuildT, ResourceT>::rasterize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 2: surface partition.

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::partitionSurfaces()
{
    const auto* dGrid = mGridHandle.template deviceGrid<BuildT>();
    ConnectedComponents<BuildT, ResourceT> components(dGrid, mStream, *mResource);
    auto result = components.getVoxelLabelsAndCount();
    mSurfaceCount = result.second;
    // The labels are read only when there are several surfaces; otherwise they are freed here, with result.
    if (mSurfaceCount <= 1) return;
    mSurfaceLabels = std::move(result.first);

    mSurfaceGrids = PartitionGrid<BuildT, ResourceT>(dGrid, mSurfaceLabels.data(), mSurfaceCount, mStream, *mResource)
                        .getHandles(mGridHandle.buffer());
    std::vector<const GridT*> grids;
    for (const auto& handle : mSurfaceGrids) grids.push_back(handle.template deviceGrid<BuildT>());
    mDeviceSurfaceGrids = BufferT<const GridT*>(mStream, this->ref(), mSurfaceCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mDeviceSurfaceGrids.data(), grids.data(), mDeviceSurfaceGrids.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));

    mNestingDepth = BufferT<uint32_t>(mStream, this->ref(), mSurfaceCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mNestingDepth.data(), 0, mNestingDepth.size_bytes(), mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
} // MeshToSDF<BuildT, ResourceT>::partitionSurfaces

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3a: barrier pruning.

/// @brief Half-width, in voxels, of the barrier around the signed surface {udf == isoValue}. Two face neighbours on
///        opposite sides of it are one voxel apart, so the distances of both from it sum to at most one voxel: at
///        least one lies within half a voxel. A barrier slightly wider than half a voxel therefore leaves no face
///        neighbours on opposite sides outside it; the 0.001 margin covers float error in the distances.
static constexpr float BarrierHalfWidth = 0.501f;

/// @brief Whether a voxel at unsigned distance @a udf from the mesh lies in the barrier: within BarrierHalfWidth
///        voxels of the signed surface {udf == isoValue}. All values are in world units.
/// @note Pruning and barrier signing must classify a voxel alike, so both use this test. Signing negates the
///       distance exactly, so the test gives the same answer on the magnitude of a signed distance.
__hostdev__ inline bool isBarrier(float udf, float isoValue, float voxelSize)
{
    return fabsf(udf - isoValue) < BarrierHalfWidth * voxelSize;
}

/// @brief Retain the voxels outside the barrier.
template <typename BuildT>
struct UDFBarrierPruneMaskFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE / 2;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* dGrid,
        const float*            dUDF,           // unsigned distance of each voxel slot, world
        float                   isoValue,       // signed surface = { udf == isoValue }, world
        float                   voxelSize,      // world
        nanovdb::Mask<3>*       dDstLeafMasks)  // one per leaf: the voxels to retain
    {
        const int leafID = blockIdx.x, warpID = threadIdx.x >> 5, laneID = threadIdx.x & 31;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        const uint32_t n0 = 64 * warpID + laneID, n1 = n0 + 32;

        const bool keep0 = leaf.isActive(n0) && !isBarrier(dUDF[leaf.getValue(n0)], isoValue, voxelSize);
        const bool keep1 = leaf.isActive(n1) && !isBarrier(dUDF[leaf.getValue(n1)], isoValue, voxelSize);

        // Every lane participates, including inactive voxels. Each warp writes one complete mask word.
        const uint32_t low  = __ballot_sync(0xFFFFFFFFu, keep0);
        const uint32_t high = __ballot_sync(0xFFFFFFFFu, keep1);
        if (laneID == 0)
            dDstLeafMasks[leafID].words()[warpID] = uint64_t(low) | (uint64_t(high) << 32);
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
typename MeshToSDF<BuildT, ResourceT>::HandleT
MeshToSDF<BuildT, ResourceT>::pruneBarrier(const GridT* dGrid, const float* dUDF)
{
    using FunctorT = sdf::detail::UDFBarrierPruneMaskFunctor<BuildT>;

    const float    voxelSize = float(mMap.getVoxelSize()[0]);
    const uint32_t leaves    = TraitsT::getTreeData(dGrid, mStream).mNodeCount[0];

    BufferT<nanovdb::Mask<3>> retainMask(mStream, this->ref(), leaves, nanovdb::cuda::noInit);

    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dUDF, mIsoValue, voxelSize, retainMask.data());
    cudaCheckError();

    PruneGrid<BuildT, ResourceT> pruner(dGrid, retainMask.data(), mStream, *mResource);
    return pruner.getHandle(BufferT<std::byte>(mStream, this->ref(), 0, nanovdb::cuda::noInit));
} // MeshToSDF<BuildT, ResourceT>::pruneBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3b: non-barrier signing.

/// @brief Find the component containing the minimum-x voxel, which is exterior.
/// @note Each warp reduces its keys first, so only one thread per warp updates @a dMinKey.
template <typename BuildT>
struct FindExteriorLabelFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const uint32_t* dVoxelLabel,
                               unsigned long long* dMinKey)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        unsigned long long key = ~0ull;  // inactive voxels take part in the reduction with the largest key
        if (leaf.isActive(uint32_t(n))) {
            const uint64_t v = leaf.getValue(uint32_t(n));
            const nanovdb::Coord ijk = leaf.origin() + NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
            const uint32_t label = dVoxelLabel[v];                            // component label
            const uint32_t ux  = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x, shifted to unsigned-comparable
            key = (static_cast<unsigned long long>(ux) << 32) | label;
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            key = min(key, __shfl_down_sync(0xFFFFFFFFu, key, offset));
        if ((n & 31) == 0 && key != ~0ull) atomicMin(dMinKey, key);
    }
};

/// @brief Negate the distance of each voxel of the barrier-free grid @a dDerivedGrid that is not in the exterior
///        component, the component of the minimum-x voxel found by FindExteriorLabelFunctor in @a dMinKey. The
///        distances are indexed by @a dGrid, which holds every voxel of @a dDerivedGrid.
template <typename BuildT>
struct SignNonBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dDerivedGrid, const uint32_t* dVoxelLabel,
                               const unsigned long long* dMinKey, const NanoGrid<BuildT>* dGrid, float* dSDF)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dDerivedGrid->tree().template getFirstNode<0>()[leafID];
        const uint32_t exteriorLabel = uint32_t(*dMinKey);  // low 32 bits of the key
        if (!leaf.isActive(uint32_t(n)) || dVoxelLabel[leaf.getValue(uint32_t(n))] == exteriorLabel) return;
        const uint64_t v = dGrid->tree().root().probeLeaf(leaf.origin())->getValue(uint32_t(n));
        dSDF[v] = -dSDF[v];
    }
};

} // namespace sdf::detail

/// @brief Find the exterior component of the barrier-free grid @a dDerivedGrid, which must have a leaf. Returns
///        the key of FindExteriorLabelFunctor on the device; its low 32 bits are the component label.
template <typename BuildT, typename ResourceT>
typename MeshToSDF<BuildT, ResourceT>::template BufferT<unsigned long long>
MeshToSDF<BuildT, ResourceT>::findExteriorLabel(const GridT* dDerivedGrid, const uint32_t* dVoxelLabel)
{
    const uint32_t leaves = TraitsT::getTreeData(dDerivedGrid, mStream).mNodeCount[0];
    BufferT<unsigned long long> minKey(mStream, this->ref(), 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(minKey.data(), 0xFF, sizeof(unsigned long long), mStream));
    using FunctorT = sdf::detail::FindExteriorLabelFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dDerivedGrid, dVoxelLabel, minKey.data());
    cudaCheckError();
    return minKey;
} // MeshToSDF<BuildT, ResourceT>::findExteriorLabel

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signNonBarrierComponents(const GridT* dGrid, const GridT* dDerivedGrid,
                                                            const uint32_t* dVoxelLabel, float* dSDF)
{
    const uint32_t leaves = TraitsT::getTreeData(dDerivedGrid, mStream).mNodeCount[0];
    if (leaves == 0) return;

    const auto minKey = findExteriorLabel(dDerivedGrid, dVoxelLabel);
    using SignFunctorT = sdf::detail::SignNonBarrierFunctor<BuildT>;
    util::cuda::operatorKernel<SignFunctorT><<<leaves, SignFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dDerivedGrid, dVoxelLabel, minKey.data(), dGrid, dSDF);
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::signNonBarrierComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3c: barrier signing.

static constexpr uint32_t InvalidTriangle = 0xFFFFFFFFu;

/// @brief Least distance, in voxels, of an anchor from the signed surface: half a voxel diagonal, as in OpenVDB's
///        meshToLevelSet. A single anchor can place a barrier voxel outside, and anchors nearer the surface, down
///        to BarrierHalfWidth, place interior voxels next to it outside far more often.
static constexpr float AnchorMinDistance = 0.8660254f;

/// @brief Whether voxel @a nv at @a nijk is an anchor: an exterior voxel at least AnchorMinDistance voxels from the
///        signed surface, with a nearest triangle. If so, @a dir is the unit direction from the triangle towards it
///        and @a base the point of the signed surface below it.
__hostdev__ inline bool
anchorGeometry(uint64_t nv, const nanovdb::Coord& nijk, const float* dSDF, float isoValue, float voxelSize,
               const uint32_t* dTriangleIndex, const nanovdb::Vec3f* dPoints, const nanovdb::Vec3i* dTriangles,
               const nanovdb::Map& map, float isoValueIndex, nanovdb::Vec3f& base, nanovdb::Vec3f& dir)
{
    if (signbit(dSDF[nv]) || fabsf(fabsf(dSDF[nv]) - isoValue) < AnchorMinDistance * voxelSize) return false;
    const uint32_t tid = dTriangleIndex[nv];
    if (tid == InvalidTriangle) return false;

    const nanovdb::Vec3i& T  = dTriangles[tid];
    const nanovdb::Vec3f  v0 = map.applyInverseMapF(dPoints[T[0]]);  // world -> index
    const nanovdb::Vec3f  v1 = map.applyInverseMapF(dPoints[T[1]]);
    const nanovdb::Vec3f  v2 = map.applyInverseMapF(dPoints[T[2]]);
    const nanovdb::Vec3f  n_xyz(static_cast<float>(nijk[0]), static_cast<float>(nijk[1]), static_cast<float>(nijk[2]));

    float t0, t1;
    const nanovdb::Vec3f cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, n_xyz, t0, t1);
    dir = n_xyz - cp; dir.normalize();   // surface -> anchor (its confident side)
    base = cp + dir * isoValueIndex;
    return true;
}

/// @brief Whether the anchor with @a base and @a dir places @a q outside: on the same side of the surface as itself.
__hostdev__ inline bool isExteriorBy(const nanovdb::Vec3f& base, const nanovdb::Vec3f& dir, const nanovdb::Vec3f& q)
{
    return dir.dot(q - base) > 0.f;
}

/// @brief Negate the distance of every barrier voxel; slot 0, the background, is left as is.
struct BarrierToInteriorFunctor
{
    __device__ void operator()(const uint64_t slot, float isoValue, float voxelSize, float* dSDF) const
    {
        if (slot != 0 && isBarrier(fabsf(dSDF[slot]), isoValue, voxelSize)) dSDF[slot] = -dSDF[slot];
    }
};

/// @brief Mark in @a dNegate the barrier voxels that are interior, using exterior neighbours and their nearest
///        triangles; the other voxels are signed already. Unresolved voxels default to interior.
/// @note The distances are only read here: NegateMarkedFunctor applies the marks once every voxel is decided, so no
///       thread reads a distance that another thread writes.
///       The anchor geometry of the leaf and of a one-voxel halo around it is computed first into shared memory, so
///       each anchor is computed once for all the barrier voxels around it.
template <typename BuildT>
struct SignBarrierFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(
        const NanoGrid<BuildT>* dGrid,
        const float*            dSDF,           // signed outside the barrier, unsigned in it
        float                   isoValue,       // signed surface = { udf == isoValue }, world
        float                   voxelSize,      // world
        const uint32_t*         dTriangleIndex, // nearest triangle of each voxel slot of dGrid
        const nanovdb::Vec3f*   dPoints,        // mesh vertices, world
        const nanovdb::Vec3i*   dTriangles,     // triangle vertex indices
        nanovdb::Map            map,            // world <-> index transform
        float                   isoValueIndex,  // isoValue in voxels
        uint8_t*                dNegate)        // set to 1 for each interior barrier voxel
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        const bool     active  = leaf.isActive(uint32_t(n));
        const uint64_t qv      = active ? leaf.getValue(uint32_t(n)) : 0;
        const bool     barrier = active && isBarrier(fabsf(dSDF[qv]), isoValue, voxelSize);
        if (!__syncthreads_or(barrier)) return;  // no barrier voxel in this leaf

        // The leaf and its 26 neighbours, at (dx+1)*9 + (dy+1)*3 + (dz+1) for leaf offsets dx, dy, dz in [-1, 1].
        using LeafT = NanoLeaf<BuildT>;
        const nanovdb::Coord origin = leaf.origin();
        __shared__ const LeafT* sLeaves[27];
        if (n < 27)
            sLeaves[n] = dGrid->tree().root().probeLeaf(
                origin.offsetBy(LeafT::DIM * (n / 9 - 1), LeafT::DIM * (n / 3 % 3 - 1), LeafT::DIM * (n % 3 - 1)));
        __syncthreads();

        // The anchor geometry of the leaf and a one-voxel halo around it: local coordinates x, y, z in [-1, 8] at
        // cell (x+1)*HaloDim^2 + (y+1)*HaloDim + (z+1).
        static constexpr int HaloDim = LeafT::DIM + 2, HaloSize = HaloDim * HaloDim * HaloDim;
        __shared__ nanovdb::Vec3f sBase[HaloSize], sDir[HaloSize];
        __shared__ bool           sAnchor[HaloSize];
        for (int c = n; c < HaloSize; c += MaxThreadsPerBlock) {
            const int x = c / (HaloDim * HaloDim) - 1, y = c / HaloDim % HaloDim - 1, z = c % HaloDim - 1;
            const LeafT*   nLeaf = sLeaves[((x >> 3) + 1) * 9 + ((y >> 3) + 1) * 3 + ((z >> 3) + 1)];
            const uint32_t nOff  = (uint32_t(x & 7) << 6) | (uint32_t(y & 7) << 3) | uint32_t(z & 7);
            sAnchor[c] = nLeaf && nLeaf->isActive(nOff) &&
                         anchorGeometry(nLeaf->getValue(nOff), origin.offsetBy(x, y, z), dSDF, isoValue, voxelSize,
                                        dTriangleIndex, dPoints, dTriangles, map, isoValueIndex, sBase[c], sDir[c]);
        }
        __syncthreads();
        if (!barrier) return;  // non-barrier: signed already

        const nanovdb::Coord local = LeafT::OffsetToLocalCoord(uint32_t(n));
        const nanovdb::Vec3f q_xyz(float(origin[0] + local[0]), float(origin[1] + local[1]),
                                   float(origin[2] + local[2]));

        // q itself is a barrier voxel, never an anchor, so all 27 cells around it can be tested.
        bool exterior = false;
        for (int dx = 0; dx < 3 && !exterior; ++dx)
            for (int dy = 0; dy < 3 && !exterior; ++dy)
                for (int dz = 0; dz < 3 && !exterior; ++dz) {
                    const int c = (local[0] + dx) * HaloDim * HaloDim + (local[1] + dy) * HaloDim + (local[2] + dz);
                    exterior = sAnchor[c] && isExteriorBy(sBase[c], sDir[c], q_xyz);
                }

        if (!exterior) dNegate[qv] = 1;
    }
};

/// @brief Negate the distance of every voxel slot marked by SignBarrierFunctor.
struct NegateMarkedFunctor
{
    __device__ void operator()(const uint64_t slot, const uint8_t* dNegate, float* dSDF) const
    {
        if (dNegate[slot]) dSDF[slot] = -dSDF[slot];
    }
};

} // namespace sdf::detail

/// @brief Sign the barrier voxels of the distances @a dSDF by the chosen BarrierSigning. Interior runs one thread
///        per voxel slot, which is several times faster than the per-leaf launch the heuristic needs.
template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signBarrier(const GridT* dGrid, float* dSDF, const uint32_t* dTriangleIndex)
{
    const auto tree = TraitsT::getTreeData(dGrid, mStream);
    if (tree.mNodeCount[0] == 0) return;

    static constexpr unsigned int threadsPerBlock = 256;
    const uint64_t slots     = tree.mVoxelCount + 1;
    const float    voxelSize = float(mMap.getVoxelSize()[0]);
    if (mBarrierSigning == BarrierSigning::Interior) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(slots, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            slots, sdf::detail::BarrierToInteriorFunctor{}, mIsoValue, voxelSize, dSDF);
        cudaCheckError();
        return;
    }

    // The heuristic reads the distances of the neighbours, so it only marks the voxels to negate, and a second
    // kernel negates them once every voxel is decided.
    BufferT<uint8_t> negate(mStream, this->ref(), slots, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(negate.data(), 0, negate.size_bytes(), mStream));
    using FunctorT = sdf::detail::SignBarrierFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<tree.mNodeCount[0], FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dSDF, mIsoValue, voxelSize,
        dTriangleIndex, mDevicePoints, mDeviceTriangles, mMap,
        (voxelSize > 0.f) ? mIsoValue / voxelSize : 0.f, negate.data());
    cudaCheckError();
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(slots, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        slots, sdf::detail::NegateMarkedFunctor{}, negate.data(), dSDF);
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::signBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3d: nesting depth.

/// @brief Whether the invert bit that PropagateSign computed for index coordinate @a ijk of @a grid is on, i.e.
///        whether the level set takes the background value with its sign inverted there.
/// @details Descends from the root to the first element without a child that holds @a ijk: a region without a
///          root-table entry (its bit is on if its origin is among @a rootOrigins), a childless upper or lower slot,
///          or a leaf voxel. The bit of an active voxel is off.
/// @param leafMasks, lowerMasks, upperMasks The invert masks PropagateSign::propagate() filled.
/// @param rootOrigins, rootOriginCount The root origins propagate() returned, in increasing (x, y, z) order.
template <typename BuildT>
__hostdev__ inline bool isInverted(const NanoGrid<BuildT>& grid, const nanovdb::Coord& ijk,
                                   const nanovdb::Mask<3>* leafMasks, const nanovdb::Mask<4>* lowerMasks,
                                   const nanovdb::Mask<5>* upperMasks, const nanovdb::Coord* rootOrigins,
                                   uint64_t rootOriginCount)
{
    using UpperT = NanoUpper<BuildT>;
    using LowerT = NanoLower<BuildT>;
    using LeafT  = NanoLeaf<BuildT>;
    const auto& tree = grid.tree();

    const auto* tile = tree.root().probeTile(ijk);
    if (!tile) return ::cuda::std::binary_search(rootOrigins, rootOrigins + rootOriginCount,
                                                 ijk & ~int(UpperT::DIM - 1));
    NANOVDB_ASSERT(tile->isChild()); // PropagateSign rejects root value tiles

    const auto*    upper = tree.root().getChild(tile);
    const uint32_t u     = UpperT::CoordToOffset(ijk);
    if (!upper->isChild(u)) return upperMasks[upper - tree.getFirstUpper()].isOn(u);

    const auto*    lower = upper->getChild(u);
    const uint32_t l     = LowerT::CoordToOffset(ijk);
    if (!lower->isChild(l)) return lowerMasks[lower - tree.getFirstLower()].isOn(l);

    return leafMasks[lower->getChild(l) - tree.getFirstLeaf()].isOn(LeafT::CoordToOffset(ijk));
} // isInverted

/// @brief Count surface j as enclosed by the surface of @a dGrid if a voxel of surface j is inside it.
/// @details Any voxel of surface j gives the same answer: its voxels are connected and none is active in @a dGrid,
///          so they all lie in one region of the inactive space of @a dGrid. The first active voxel of the first leaf
///          of surface j's grid is used. That leaf exists and holds an active voxel: every label has a voxel, and
///          PartitionGrid builds an output leaf only where a source leaf keeps a voxel of that label.
template <typename BuildT>
struct AccumulateNestingFunctor
{
    __device__ void operator()(size_t j, const NanoGrid<BuildT>* dGrid,
                               const NanoGrid<BuildT>* const* dSurfaceGrids,
                               const nanovdb::Mask<3>* dLeaf, const nanovdb::Mask<4>* dLower,
                               const nanovdb::Mask<5>* dUpper, const nanovdb::Coord* dRootTiles,
                               uint64_t rootTileCount, uint32_t surfaceID,
                               uint32_t* dDepth) const
    {
        if (j == surfaceID) return;
        const auto* leaf = dSurfaceGrids[j]->tree().getFirstLeaf();
        const nanovdb::Coord ijk = leaf->offsetToGlobalCoord(leaf->valueMask().template findFirst<true>());
        if (isInverted(*dGrid, ijk, dLeaf, dLower, dUpper, dRootTiles, rootTileCount))
            ++dDepth[j];
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::processSurface(uint32_t surfaceID)
{
    const auto* dOriginalGrid = mGridHandle.template deviceGrid<BuildT>();

    // The distances of this surface's voxels, unsigned until the signing below: the grid's own when there is
    // one surface, else a copy indexed by the surface's grid. Likewise the triangle indices.
    BufferT<float>    surfaceSDF(mStream, this->ref(), 0, nanovdb::cuda::noInit);
    BufferT<uint32_t> surfaceTriangleIndex(mStream, this->ref(), 0, nanovdb::cuda::noInit);
    const GridT*    dGrid          = dOriginalGrid;
    auto            tree           = mTreeData;   // the tree header of dGrid
    float*          dSDF           = reinterpret_cast<float*>(mSDF.data());
    const uint32_t* dTriangleIndex = reinterpret_cast<const uint32_t*>(mTriangleIndex.data());

    if (mSurfaceCount > 1) {
        dGrid = mSurfaceGrids[surfaceID].template deviceGrid<BuildT>();
        tree = TraitsT::getTreeData(dGrid, mStream);

        surfaceSDF = BufferT<float>(mStream, this->ref(), tree.mVoxelCount + 1, nanovdb::cuda::noInit);
        surfaceTriangleIndex = BufferT<uint32_t>(mStream, this->ref(), tree.mVoxelCount + 1, nanovdb::cuda::noInit);
        cudaCheck(cudaMemsetAsync(surfaceSDF.data(), 0, surfaceSDF.size_bytes(), mStream));
        cudaCheck(cudaMemsetAsync(surfaceTriangleIndex.data(), 0xFF, surfaceTriangleIndex.size_bytes(), mStream));

        using InjectSDFFunctorT = util::cuda::InjectGridDataFunctor<BuildT, float>;
        using InjectIndexFunctorT = util::cuda::InjectGridDataFunctor<BuildT, uint32_t>;
        static constexpr int threads = InjectSDFFunctorT::MaxThreadsPerBlock; // the same for both
        const uint32_t leaves = mTreeData.mNodeCount[0];
        util::cuda::operatorKernel<InjectSDFFunctorT><<<leaves, threads, 0, mStream>>>(
            dOriginalGrid, dGrid, reinterpret_cast<const float*>(mSDF.data()), surfaceSDF.data());
        cudaCheckError();
        util::cuda::operatorKernel<InjectIndexFunctorT><<<leaves, threads, 0, mStream>>>(
            dOriginalGrid, dGrid, reinterpret_cast<const uint32_t*>(mTriangleIndex.data()),
            surfaceTriangleIndex.data());
        cudaCheckError();

        dSDF = surfaceSDF.data();
        dTriangleIndex = surfaceTriangleIndex.data();
    }

    auto derived = pruneBarrier(dGrid, dSDF);
    const auto* dDerivedGrid = derived.template deviceGrid<BuildT>();

    ConnectedComponents<BuildT, ResourceT> components(dDerivedGrid, mStream, *mResource);
    auto componentLabels = components.getVoxelLabelsAndCount();

    signNonBarrierComponents(dGrid, dDerivedGrid, componentLabels.first.data(), dSDF);
    componentLabels.first.destroy();
    derived.reset();

    signBarrier(dGrid, dSDF, dTriangleIndex);

    if (mSurfaceCount == 1) return;

    // The invert masks of this surface, for the nesting test below.
    BufferT<nanovdb::Mask<3>> leafMasks(mStream, this->ref(), tree.mNodeCount[0], nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<4>> lowerMasks(mStream, this->ref(), tree.mNodeCount[1], nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<5>> upperMasks(mStream, this->ref(), tree.mNodeCount[2], nanovdb::cuda::noInit);
    const auto rootTileOrigins = PropagateSign<BuildT, float, ResourceT>(dGrid, dSDF, mStream, *mResource)
                                     .propagate(leafMasks.data(), lowerMasks.data(), upperMasks.data()).first;
    uint32_t* dNestingDepth = mNestingDepth.data();
    using NestingFunctorT = sdf::detail::AccumulateNestingFunctor<BuildT>;
    static constexpr unsigned int threadsPerBlock = 256;
    const auto blocks = util::cuda::blocksPerGrid(mSurfaceCount, threadsPerBlock);
    util::cuda::lambdaKernel<<<blocks, threadsPerBlock, 0, mStream>>>(
        mSurfaceCount, NestingFunctorT{}, dGrid, mDeviceSurfaceGrids.data(),
        leafMasks.data(), lowerMasks.data(), upperMasks.data(),
        rootTileOrigins.data(), uint64_t(rootTileOrigins.size()), surfaceID, dNestingDepth);
    cudaCheckError();

    using InjectSDFFunctorT = util::cuda::InjectGridDataFunctor<BuildT, float>;
    const uint32_t leaves = tree.mNodeCount[0];
    util::cuda::operatorKernel<InjectSDFFunctorT><<<leaves, InjectSDFFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dOriginalGrid, dSDF, reinterpret_cast<float*>(mSDF.data()));
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::processSurface

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 4: nesting rule.

/// @brief Compose per-surface signs under the selected nesting rule.
/// @note The enclosing count is the surface depth plus its local inside state.
struct ResolveNestingFunctor
{
    __device__ void operator()(size_t v, const uint32_t* dSurfaceLabel, const uint32_t* dDepth, bool evenOdd,
                               float* dSDF) const
    {
        if (v == 0) return;                                  // slot 0 is the background
        const uint32_t enclosing = dDepth[dSurfaceLabel[v]] + (signbit(dSDF[v]) ? 1u : 0u);
        const bool interior = evenOdd ? ((enclosing & 1u) != 0u) : (enclosing != 0u);
        const float m = fabsf(dSDF[v]);
        dSDF[v] = interior ? -m : m;
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::resolveSigns()
{
    if (mSurfaceCount <= 1) return;

    const uint64_t activeCount = mTreeData.mVoxelCount;
    static constexpr unsigned int threadsPerBlock = 256;
    const auto blocks = util::cuda::blocksPerGrid(activeCount + 1, threadsPerBlock);
    util::cuda::lambdaKernel<<<blocks, threadsPerBlock, 0, mStream>>>(
        activeCount + 1, sdf::detail::ResolveNestingFunctor{}, mSurfaceLabels.data(), mNestingDepth.data(),
        mNestingRule == NestingRule::EvenOdd, reinterpret_cast<float*>(mSDF.data()));
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::resolveSigns

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 5: iso-value offset.

/// @brief Mark in @a dFloor the interior voxels whose distance from the requested isosurface is below
///        @a interiorFloor and that have no exterior face neighbour: IsoMagnitudeFunctor floors them.
/// @note The distances are only read here, so no thread reads a distance that another thread writes.
template <typename BuildT>
struct IsoFloorFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dGrid, const float* dSDF, float isoValue, float interiorFloor,
                               uint8_t* dFloor)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        if (!leaf.isActive(uint32_t(n))) return;

        const uint64_t v      = leaf.getValue(uint32_t(n));
        const bool     inside = signbit(dSDF[v]);
        const float    m      = fabsf(fabsf(dSDF[v]) - isoValue);
        if (!inside || m >= interiorFloor) return;   // nothing to floor

        // Preserve voxels that can carry a marching-cubes crossing.
        const nanovdb::Coord local = NanoLeaf<BuildT>::OffsetToLocalCoord(uint32_t(n));
        const int lx = local[0], ly = local[1], lz = local[2];
        const int off[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

        bool touchesExterior = false, needsAccessor = false;
        for (int k = 0; k < 6 && !touchesExterior; ++k) {
            const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
            if (nx < 0 || nx > 7 || ny < 0 || ny > 7 || nz < 0 || nz > 7) { needsAccessor = true; continue; }
            const uint32_t nOff = (uint32_t(nx) << 6) | (uint32_t(ny) << 3) | uint32_t(nz);
            if (leaf.isActive(nOff) && !signbit(dSDF[leaf.getValue(nOff)])) touchesExterior = true;
        }
        if (!touchesExterior && needsAccessor) {
            const nanovdb::Coord origin = leaf.origin();
            auto acc = dGrid->getAccessor();
            for (int k = 0; k < 6 && !touchesExterior; ++k) {
                const int nx = lx + off[k][0], ny = ly + off[k][1], nz = lz + off[k][2];
                if (nx >= 0 && nx <= 7 && ny >= 0 && ny <= 7 && nz >= 0 && nz <= 7) continue;
                const nanovdb::Coord nijk(origin[0] + nx, origin[1] + ny, origin[2] + nz);
                if (acc.isActive(nijk) && !signbit(dSDF[acc.getValue(nijk)])) touchesExterior = true;
            }
        }
        if (!touchesExterior) dFloor[v] = 1;
    }
};

/// @brief Convert the signed mesh UDF to signed distance from the requested isosurface, in place, and floor the
///        voxels marked by IsoFloorFunctor at -interiorFloor. Slot 0, the background, is left as is.
struct IsoMagnitudeFunctor
{
    __device__ void operator()(const uint64_t v, float isoValue, float interiorFloor, const uint8_t* dFloor,
                               float* dSDF) const
    {
        if (v == 0) return;
        const bool  inside = signbit(dSDF[v]);
        const float m      = fabsf(fabsf(dSDF[v]) - isoValue);
        dSDF[v] = dFloor[v] ? -interiorFloor : (inside ? -m : m);
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::finalizeDistances()
{
    if (mIsoValue == 0.f) return;

    const float voxelSize = float(mMap.getVoxelSize()[0]);

    // Match the barrier used during signing.
    const float interiorFloor = sdf::detail::BarrierHalfWidth * voxelSize;

    // The floor test reads the signs of the neighbours, so the floored voxels are marked first, and a second kernel
    // writes every distance.
    static constexpr unsigned int threadsPerBlock = 256;
    const uint64_t slots = mTreeData.mVoxelCount + 1;
    float*         dSDF  = reinterpret_cast<float*>(mSDF.data());
    BufferT<uint8_t> floored(mStream, this->ref(), slots, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(floored.data(), 0, floored.size_bytes(), mStream));
    using FunctorT = sdf::detail::IsoFloorFunctor<BuildT>;
    if (const uint32_t leaves = mTreeData.mNodeCount[0])
        util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
            mGridHandle.template deviceGrid<BuildT>(), dSDF, mIsoValue, interiorFloor, floored.data());
    cudaCheckError();
    util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(slots, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
        slots, sdf::detail::IsoMagnitudeFunctor{}, mIsoValue, interiorFloor, floored.data(), dSDF);
    cudaCheckError();

    // The voxel kernels do not visit the background slot.
    const float bg = mBandWidth * voxelSize;
    cudaCheck(cudaMemcpyAsync(mSDF.data(), &bg, sizeof(float), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
} // MeshToSDF<BuildT, ResourceT>::finalizeDistances

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
