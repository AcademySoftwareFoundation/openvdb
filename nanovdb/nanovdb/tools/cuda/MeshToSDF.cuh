// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/MeshToSDF.cuh
    \authors Efty Sifakis and JaeHyun Lee
    \brief GPU conversion of triangle meshes to narrow-band signed distance fields on NanoVDB
           ValueOnIndex grids.

    \details The pipeline rasterizes a UDF, partitions its band into components, signs each component,
             and composes nested components. It returns the grid and the signed distances of its active
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
/// @warning The map must scale uniformly and must not rotate: the voxel size is read from the x axis of the map
///          and used for all three axes.
/// @note A component is a connected component of the rasterized narrow band, so meshes whose bands touch are signed
///       as one component, and the nesting rule applies only between components whose bands are apart.
/// @tparam BuildT NanoVDB build type; only ValueOnIndex is supported.
/// @tparam ResourceT Stream-ordered resource used for the output and all scratch allocations.
template <typename BuildT, typename ResourceT = nanovdb::cuda::DeviceResource>
class MeshToSDF
{
    static_assert(nanovdb::BuildTraits<BuildT>::is_onindex, "MeshToSDF requires a ValueOnIndex grid");
    static_assert(nanovdb::cuda::is_async_resource<ResourceT>::value,
                  "MeshToSDF allocates stream-ordered storage and requires an AsyncResource");

    using PointT         = nanovdb::Vec3f;
    using TriangleIndexT = nanovdb::Vec3i;
    using GridT          = NanoGrid<BuildT>;
    using TreeT          = NanoTree<BuildT>;
    using TraitsT        = util::cuda::DeviceGridTraits<BuildT>;
    using ComponentLabelT = ConnectedComponentsBase::ComponentLabelT;

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
        , mMap(map), mVoxelSize(float(map.getVoxelSize()[0])), mStream(stream), mTimer(stream), mResource(&resource)
        , mGridHandle(BufferT<std::byte>(stream, resource, 0, nanovdb::cuda::noInit))
        , mSDF(stream, resource, 0, nanovdb::cuda::noInit)
        , mNearestTriangle(stream, resource, 0, nanovdb::cuda::noInit)
        , mComponentLabels(stream, resource, 0, nanovdb::cuda::noInit)
        , mDeviceComponentGrids(stream, resource, 0, nanovdb::cuda::noInit)
        , mNestingDepth(stream, resource, 0, nanovdb::cuda::noInit) {}

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Set desired width of the narrow band
    /// @param bandWidth Narrow band width in voxels
    void setNarrowBandWidth(float bandWidth = 3.f) { mBandWidth = bandWidth; }

    /// @brief Policy for signing voxels in the surface barrier.
    enum class BarrierSigning {
        Interior,   ///< classify all barrier voxels as interior
        Heuristic   ///< use the OpenVDB intersecting-voxel heuristic
    };

    /// @brief Choose the barrier signing method (default Heuristic).
    void setBarrierSigning(BarrierSigning signing) { mBarrierSigning = signing; }

    /// @brief How a point enclosed by several of the input's closed surfaces is signed.
    enum class NestingRule {
        EvenOdd,  ///< interior when enclosed by an odd number of surfaces
        Solid     ///< interior when enclosed by one or more surfaces
    };

    /// @brief Choose the nesting rule (default EvenOdd).
    void setNestingRule(NestingRule rule) { mNestingRule = rule; }

    /// @brief Sign the isosurface where the mesh UDF equals @a isoValue.
    /// @param isoValue Non-negative offset in world units; zero signs the input surface.
    /// @note Rasterization includes the offset, so time and memory increase with @a isoValue.
    void setIsoValue(float isoValue = 0.f) { mIsoValue = isoValue; }

    /// @brief Run the pipeline and return the grid and its signed distances.
    /// @return { grid, sdf }: the narrow-band ValueOnIndex grid, and a sidecar of (activeVoxelCount+1) floats indexed
    ///         like the grid's values: sdf[0] is the positive background, and sdf[k>0] is the signed distance of
    ///         the k-th active voxel to the signed surface, negative in the interior. Pass both to PropagateSign for
    ///         the sign of the inactive voxels and tiles.
    /// @throw std::runtime_error if the iso value is negative.
    /// @note Call once: the grid and the distances are moved out of this object.
    std::pair<HandleT, BufferT<std::byte>> getHandleAndSDF()
    {
        if (mIsoValue < 0.f)
            throw std::runtime_error("MeshToSDF: setIsoValue() must be >= 0");

        if (mVerbose==1) mTimer.start("\nRasterizing mesh");
        rasterize();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Partitioning components");
        partitionComponents();
        if (mVerbose==1) mTimer.stop();

        if (mVerbose==1) mTimer.start("Signing components");
        signComponents();
        if (mVerbose==1) mTimer.stop();

        // Each intermediate is released right after its last use, to cut peak memory. The frees are ordered on
        // mStream behind the kernels that read them.
        mComponentGrids.clear();
        mDeviceComponentGrids.destroy();
        mNearestTriangle.destroy();

        if (mVerbose==1) mTimer.start("Applying the nesting rule");
        applyNestingRule();
        if (mVerbose==1) mTimer.stop();

        mComponentLabels.destroy();
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

    // Stage 2: split the grid into its components, the connected components of its active voxels.
    void partitionComponents();

    // Stage 3: sign the distances of every component, and with several components, accumulate their nesting depths.
    void signComponents();

    // Copy the values that the voxels of dSrcGrid have in dSrc to the slots the same voxels have in dDstGrid.
    template <typename T>
    void copySidecar(const GridT* dSrcGrid, const GridT* dDstGrid, const T* dSrc, T* dDst);

    // Stages 3a-3c: sign the distances of one component, the voxels of dGrid.
    void signComponent(const GridT* dGrid, float* dSDF, const uint32_t* dNearestTriangle);

    // Stage 3a: the grid of the component without its barrier, the voxels nearest the signed surface.
    HandleT pruneBarrier(const GridT* dGrid, const float* dUDF);

    // Stage 3b: negate the distances outside the barrier that are not in the exterior barrier-free component.
    BufferT<unsigned long long> findExteriorLabel(const GridT* dBarrierFreeGrid, uint32_t leaves,
                                                  const uint32_t* dVoxelLabel);
    void signBarrierFreeComponents(const GridT* dGrid, const GridT* dBarrierFreeGrid, const uint32_t* dVoxelLabel,
                                   float* dSDF);

    // Stage 3c: sign the distances in the barrier by the chosen BarrierSigning.
    void signBarrier(const GridT* dGrid, float* dSDF, const uint32_t* dNearestTriangle);

    // Stage 3d: add one to the nesting depth of every other component this component encloses.
    void accumulateNesting(const GridT* dGrid, const float* dSDF, ComponentLabelT componentID);

    // Stage 4: with several components, compose their signs by the nesting rule.
    void applyNestingRule();

    // Stage 5: with a nonzero iso value, convert the distances from the mesh to distances from the signed surface.
    void finalizeDistances();

    /// @brief Borrowed handle to the injected resource, for the storage allocated after construction.
    nanovdb::cuda::ResourceRef<ResourceT> ref() { return nanovdb::cuda::ResourceRef<ResourceT>(*mResource); }

    const PointT*         mDevicePoints{nullptr};
    uint32_t              mPointCount{0};
    const TriangleIndexT* mDeviceTriangles{nullptr};
    uint32_t              mTriangleCount{0};
    nanovdb::Map          mMap{};
    float                 mVoxelSize{0.f}; // world units; see the class @warning
    cudaStream_t          mStream{0};
    util::cuda::Timer     mTimer;
    ResourceT*            mResource{nullptr};
    int                   mVerbose{0};
    float                 mBandWidth{3.f};
    BarrierSigning        mBarrierSigning{BarrierSigning::Heuristic};
    NestingRule           mNestingRule{NestingRule::EvenOdd};
    float                 mIsoValue{0.f};  // world units, >= 0; see setIsoValue()

    // "Per voxel slot" below: one entry per active voxel, indexed like the grid's values, plus slot 0 for the
    // background.
    HandleT                     mGridHandle;
    typename TreeT::DataType    mTreeData{};    // host copy of the rasterized grid's tree header, set by rasterize()
    BufferT<std::byte>          mSDF;           // float per voxel slot: the UDF from MeshToGrid, signed in place
    BufferT<std::byte>          mNearestTriangle; // uint32_t per voxel slot, as returned by MeshToGrid
    BufferT<ComponentLabelT>    mComponentLabels; // component of each voxel slot, when there are several components
    ComponentLabelT             mComponentCount{0};
    std::vector<HandleT>        mComponentGrids;  // [c] the voxels of component c, when there are several components
    BufferT<const GridT*>       mDeviceComponentGrids; // [c] device pointer of mComponentGrids[c]
    BufferT<uint32_t>           mNestingDepth;  // [c] the number of other components that enclose component c

}; // tools::cuda::MeshToSDF<BuildT, ResourceT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 1: rasterization.

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::rasterize()
{
    // Preserve the requested band width beyond the offset surface.
    const float extra = mIsoValue / mVoxelSize;

    MeshToGrid<BuildT, ResourceT> converter(mDevicePoints, mPointCount, mDeviceTriangles, mTriangleCount, mMap, mStream,
                                            *mResource);
    converter.setNarrowBandWidth(mBandWidth + extra);
    std::tie(mGridHandle, mSDF, mNearestTriangle) =
        converter.template getHandleAndUDFAndIndex<BufferT<std::byte>, BufferT<std::byte>>(mGridHandle.buffer(), mSDF);
    mTreeData = TraitsT::getTreeData(mGridHandle.template deviceGrid<BuildT>(), mStream);
} // MeshToSDF<BuildT, ResourceT>::rasterize

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 2: component partition.

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::partitionComponents()
{
    const auto* dGrid = mGridHandle.template deviceGrid<BuildT>();
    ConnectedComponents<BuildT, ResourceT> components(dGrid, mStream, *mResource);
    auto [labels, count] = components.getVoxelLabelsAndCount();
    mComponentCount = count;
    // The labels are read only when there are several components; otherwise they are freed on return.
    if (mComponentCount <= 1) return;
    mComponentLabels = std::move(labels);

    mComponentGrids = PartitionGrid<BuildT, ResourceT>(dGrid, mStream, *mResource)
                          .getHandles(mComponentLabels.data(), mComponentCount, mGridHandle.buffer());
    std::vector<const GridT*> grids;
    for (const auto& handle : mComponentGrids) grids.push_back(handle.template deviceGrid<BuildT>());
    mDeviceComponentGrids = BufferT<const GridT*>(mStream, this->ref(), mComponentCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemcpyAsync(mDeviceComponentGrids.data(), grids.data(), mDeviceComponentGrids.size_bytes(),
                              cudaMemcpyHostToDevice, mStream));

    mNestingDepth = BufferT<uint32_t>(mStream, this->ref(), mComponentCount, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(mNestingDepth.data(), 0, mNestingDepth.size_bytes(), mStream));
    cudaCheck(cudaStreamSynchronize(mStream)); // the copy above reads grids, which goes out of scope here
} // MeshToSDF<BuildT, ResourceT>::partitionComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// Stage 3: component signing.

/// @brief Copy the values that the voxels of @a dSrcGrid have in @a dSrc to the slots the same voxels have in
///        @a dDstGrid, in @a dDst. Voxels of @a dSrcGrid that @a dDstGrid lacks are skipped, and the other slots of
///        @a dDst, slot 0 included, are left as they are.
template <typename BuildT, typename ResourceT>
template <typename T>
void MeshToSDF<BuildT, ResourceT>::copySidecar(const GridT* dSrcGrid, const GridT* dDstGrid, const T* dSrc, T* dDst)
{
    using FunctorT = util::cuda::InjectGridDataFunctor<BuildT, T>;
    const uint32_t leaves = TraitsT::getTreeData(dSrcGrid, mStream).mNodeCount[0];
    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dSrcGrid, dDstGrid, dSrc, dDst);
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::copySidecar

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signComponents()
{
    const auto*     dGrid            = mGridHandle.template deviceGrid<BuildT>();
    float*          dSDF             = reinterpret_cast<float*>(mSDF.data());
    const uint32_t* dNearestTriangle = reinterpret_cast<const uint32_t*>(mNearestTriangle.data());

    if (mComponentCount == 1) { // the grid is the component: sign its own distances
        signComponent(dGrid, dSDF, dNearestTriangle);
        return;
    }

    for (ComponentLabelT componentID = 0; componentID < mComponentCount; ++componentID) {
        // The distances and nearest triangles of the component, indexed by its own grid. The copies write every
        // slot but slot 0, the background, which the memsets define.
        const auto*    dComponentGrid = mComponentGrids[componentID].template deviceGrid<BuildT>();
        const uint64_t slots          = TraitsT::getTreeData(dComponentGrid, mStream).mVoxelCount + 1;
        BufferT<float>    componentSDF(mStream, this->ref(), slots, nanovdb::cuda::noInit);
        BufferT<uint32_t> componentNearestTriangle(mStream, this->ref(), slots, nanovdb::cuda::noInit);
        cudaCheck(cudaMemsetAsync(componentSDF.data(), 0, sizeof(float), mStream));
        cudaCheck(cudaMemsetAsync(componentNearestTriangle.data(), 0xFF, sizeof(uint32_t), mStream));
        copySidecar(dGrid, dComponentGrid, dSDF, componentSDF.data());
        copySidecar(dGrid, dComponentGrid, dNearestTriangle, componentNearestTriangle.data());

        signComponent(dComponentGrid, componentSDF.data(), componentNearestTriangle.data());
        accumulateNesting(dComponentGrid, componentSDF.data(), componentID);

        copySidecar(dComponentGrid, dGrid, componentSDF.data(), dSDF);
    }
} // MeshToSDF<BuildT, ResourceT>::signComponents

/// @brief Sign the distances @a dSDF of one component, the voxels of @a dGrid: first the barrier-free components,
///        then the barrier voxels.
template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signComponent(const GridT* dGrid, float* dSDF, const uint32_t* dNearestTriangle)
{
    auto barrierFree = pruneBarrier(dGrid, dSDF);
    const auto* dBarrierFreeGrid = barrierFree.template deviceGrid<BuildT>();

    ConnectedComponents<BuildT, ResourceT> barrierFreeComponents(dBarrierFreeGrid, mStream, *mResource);
    auto barrierFreeLabels = barrierFreeComponents.getVoxelLabelsAndCount().first;

    signBarrierFreeComponents(dGrid, dBarrierFreeGrid, barrierFreeLabels.data(), dSDF);
    barrierFreeLabels.destroy();
    barrierFree.reset();

    signBarrier(dGrid, dSDF, dNearestTriangle);
} // MeshToSDF<BuildT, ResourceT>::signComponent

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
struct BarrierPruneMaskFunctor
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
    using FunctorT = sdf::detail::BarrierPruneMaskFunctor<BuildT>;

    const uint32_t leaves = TraitsT::getTreeData(dGrid, mStream).mNodeCount[0];

    BufferT<nanovdb::Mask<3>> retainMask(mStream, this->ref(), leaves, nanovdb::cuda::noInit);

    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dUDF, mIsoValue, mVoxelSize, retainMask.data());
    cudaCheckError();

    PruneGrid<BuildT, ResourceT> pruner(dGrid, retainMask.data(), mStream, *mResource);
    return pruner.getHandle(BufferT<std::byte>(mStream, this->ref(), 0, nanovdb::cuda::noInit));
} // MeshToSDF<BuildT, ResourceT>::pruneBarrier

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3b: signing of the barrier-free components.

/// @brief Find the barrier-free component containing the minimum-x voxel, which is exterior: the band reaches
///        bandWidth voxels beyond the signed surface and the barrier only BarrierHalfWidth, so the voxel of least x
///        lies at a smaller x than any point of the signed surface, where that closed surface cannot enclose it.
/// @details Each active voxel packs its x and its barrier-free component label into one 64-bit key, x in the high
///          32 bits and the label in the low 32 bits, and @a dMinKey becomes the minimum of all the keys. With x in
///          the high bits, the minimum key belongs to a voxel of least x, so its low 32 bits are the exterior label:
///          one atomicMin keeps the x and the label of the same voxel together. Voxels tied on the least x are all
///          exterior, so the smaller label the tie leaves is the exterior label too. x is shifted by 2^31 so that
///          the unsigned order of the keys is the signed order of x.
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
            const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(uint32_t(n));
            const uint32_t label = dVoxelLabel[v];
            const uint32_t ux    = uint32_t(int64_t(ijk[0]) + (int64_t(1) << 31));  // x + 2^31
            key = (static_cast<unsigned long long>(ux) << 32) | label;           // (x, label) packed
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            key = min(key, __shfl_down_sync(0xFFFFFFFFu, key, offset));
        if ((n & 31) == 0 && key != ~0ull) atomicMin(dMinKey, key);
    }
};

/// @brief Negate the distance of each voxel of @a dBarrierFreeGrid that is not in the exterior
///        barrier-free component, the one of the minimum-x voxel found by FindExteriorLabelFunctor in @a dMinKey. The
///        distances are indexed by @a dGrid, which holds every voxel of @a dBarrierFreeGrid.
template <typename BuildT>
struct SignBarrierFreeFunctor
{
    static constexpr int MaxThreadsPerBlock         = NanoLeaf<BuildT>::SIZE;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__ void operator()(const NanoGrid<BuildT>* dBarrierFreeGrid, const uint32_t* dVoxelLabel,
                               const unsigned long long* dMinKey, const NanoGrid<BuildT>* dGrid, float* dSDF)
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dBarrierFreeGrid->tree().template getFirstNode<0>()[leafID];
        const uint32_t exteriorLabel = uint32_t(*dMinKey);  // low 32 bits of the key
        if (!leaf.isActive(uint32_t(n)) || dVoxelLabel[leaf.getValue(uint32_t(n))] == exteriorLabel) return;
        const uint64_t v = dGrid->tree().root().probeLeaf(leaf.origin())->getValue(uint32_t(n));
        dSDF[v] = -dSDF[v];
    }
};

} // namespace sdf::detail

/// @brief Find the exterior barrier-free component of @a dBarrierFreeGrid, which has @a leaves > 0 leaves. Returns
///        the key of FindExteriorLabelFunctor on the device; its low 32 bits are the label of that component.
template <typename BuildT, typename ResourceT>
typename MeshToSDF<BuildT, ResourceT>::template BufferT<unsigned long long>
MeshToSDF<BuildT, ResourceT>::findExteriorLabel(const GridT* dBarrierFreeGrid, uint32_t leaves,
                                                const uint32_t* dVoxelLabel)
{
    BufferT<unsigned long long> minKey(mStream, this->ref(), 1, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(minKey.data(), 0xFF, sizeof(unsigned long long), mStream));
    using FunctorT = sdf::detail::FindExteriorLabelFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<leaves, FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dBarrierFreeGrid, dVoxelLabel, minKey.data());
    cudaCheckError();
    return minKey;
} // MeshToSDF<BuildT, ResourceT>::findExteriorLabel

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signBarrierFreeComponents(const GridT* dGrid, const GridT* dBarrierFreeGrid,
                                                             const uint32_t* dVoxelLabel, float* dSDF)
{
    const uint32_t leaves = TraitsT::getTreeData(dBarrierFreeGrid, mStream).mNodeCount[0];
    if (leaves == 0) return;

    const auto minKey = findExteriorLabel(dBarrierFreeGrid, leaves, dVoxelLabel);
    using SignFunctorT = sdf::detail::SignBarrierFreeFunctor<BuildT>;
    util::cuda::operatorKernel<SignFunctorT><<<leaves, SignFunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dBarrierFreeGrid, dVoxelLabel, minKey.data(), dGrid, dSDF);
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::signBarrierFreeComponents

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 3c: barrier signing.

static constexpr uint32_t InvalidTriangle = 0xFFFFFFFFu;

/// @brief Least distance, in voxels, of an anchor from the signed surface: half a voxel diagonal, as in OpenVDB's
///        meshToLevelSet. A single anchor can place a barrier voxel in the exterior, and anchors nearer the surface,
///        down to BarrierHalfWidth, place interior voxels next to it in the exterior far more often.
static constexpr float AnchorMinDistance = 0.8660254f;

/// @brief Whether voxel slot @a v at @a ijk is an anchor: an exterior voxel at least AnchorMinDistance voxels from the
///        signed surface, with a nearest triangle. If so, @a dir is the unit direction from the triangle towards it
///        and @a base the point of the signed surface below it.
__hostdev__ inline bool
anchorGeometry(uint64_t v, const nanovdb::Coord& ijk, const float* dSDF, float isoValue, float voxelSize,
               const uint32_t* dNearestTriangle, const nanovdb::Vec3f* dPoints, const nanovdb::Vec3i* dTriangles,
               const nanovdb::Map& map, float isoValueIndex, nanovdb::Vec3f& base, nanovdb::Vec3f& dir)
{
    if (signbit(dSDF[v]) || fabsf(fabsf(dSDF[v]) - isoValue) < AnchorMinDistance * voxelSize) return false;
    const uint32_t triangleID = dNearestTriangle[v];
    if (triangleID == InvalidTriangle) return false;

    const nanovdb::Vec3i& triangle = dTriangles[triangleID];
    const nanovdb::Vec3f  v0 = map.applyInverseMapF(dPoints[triangle[0]]);  // world -> index
    const nanovdb::Vec3f  v1 = map.applyInverseMapF(dPoints[triangle[1]]);
    const nanovdb::Vec3f  v2 = map.applyInverseMapF(dPoints[triangle[2]]);
    const nanovdb::Vec3f  xyz = ijk.asVec3s();

    float t0, t1;
    const nanovdb::Vec3f cp = nanovdb::math::closestPointOnTriangleToPoint(v0, v1, v2, xyz, t0, t1);
    dir = xyz - cp; dir.normalize();   // surface -> anchor (its confident side)
    base = cp + dir * isoValueIndex;
    return true;
}

/// @brief Whether the anchor with @a base and @a dir places @a q in the exterior: on its own side of the surface.
__hostdev__ inline bool isExteriorBy(const nanovdb::Vec3f& base, const nanovdb::Vec3f& dir, const nanovdb::Vec3f& q)
{
    return dir.dot(q - base) > 0.f;
}

/// @brief Negate the distance of every barrier voxel; slot 0, the background, is left as is.
struct BarrierToInteriorFunctor
{
    __device__ void operator()(size_t v, float isoValue, float voxelSize, float* dSDF) const
    {
        if (v != 0 && isBarrier(fabsf(dSDF[v]), isoValue, voxelSize)) dSDF[v] = -dSDF[v];
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
        const uint32_t*         dNearestTriangle, // nearest triangle of each voxel slot of dGrid
        const nanovdb::Vec3f*   dPoints,        // mesh vertices, world
        const nanovdb::Vec3i*   dTriangles,     // triangle vertex indices
        nanovdb::Map            map,            // world <-> index transform
        float                   isoValueIndex,  // isoValue in voxels
        uint8_t*                dNegate)        // set to 1 for each interior barrier voxel
    {
        const int leafID = blockIdx.x, n = threadIdx.x;
        const auto& leaf = dGrid->tree().template getFirstNode<0>()[leafID];
        const bool     active  = leaf.isActive(uint32_t(n));
        const uint64_t v       = active ? leaf.getValue(uint32_t(n)) : 0;
        const bool     barrier = active && isBarrier(fabsf(dSDF[v]), isoValue, voxelSize);
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
        // index (x+1)*HaloDim^2 + (y+1)*HaloDim + (z+1).
        static constexpr int HaloDim = LeafT::DIM + 2, HaloSize = HaloDim * HaloDim * HaloDim;
        __shared__ nanovdb::Vec3f sBase[HaloSize], sDir[HaloSize];
        __shared__ bool           sAnchor[HaloSize];
        for (int c = n; c < HaloSize; c += MaxThreadsPerBlock) {
            const int x = c / (HaloDim * HaloDim) - 1, y = c / HaloDim % HaloDim - 1, z = c % HaloDim - 1;
            // x >> 3 = floor(x / 8): -1 in the leaf below, 0 in this leaf, 1 in the leaf above; likewise y and z.
            const LeafT*   haloLeaf   = sLeaves[((x >> 3) + 1) * 9 + ((y >> 3) + 1) * 3 + ((z >> 3) + 1)];
            const uint32_t haloOffset = LeafT::CoordToOffset(nanovdb::Coord(x, y, z));
            sAnchor[c] = haloLeaf && haloLeaf->isActive(haloOffset) &&
                         anchorGeometry(haloLeaf->getValue(haloOffset), origin.offsetBy(x, y, z), dSDF, isoValue,
                                        voxelSize, dNearestTriangle, dPoints, dTriangles, map, isoValueIndex,
                                        sBase[c], sDir[c]);
        }
        __syncthreads();
        if (!barrier) return;  // non-barrier: signed already

        const nanovdb::Coord local = LeafT::OffsetToLocalCoord(uint32_t(n));
        const nanovdb::Vec3f q     = leaf.offsetToGlobalCoord(uint32_t(n)).asVec3s();

        // This voxel, at q, is a barrier voxel, never an anchor, so all 27 voxels around it can be tested. The
        // halo index of local + (dx-1, dy-1, dz-1) is that of local + (dx, dy, dz) minus one per axis.
        bool exterior = false;
        for (int dx = 0; dx < 3 && !exterior; ++dx)
            for (int dy = 0; dy < 3 && !exterior; ++dy)
                for (int dz = 0; dz < 3 && !exterior; ++dz) {
                    const int c = (local[0] + dx) * HaloDim * HaloDim + (local[1] + dy) * HaloDim + (local[2] + dz);
                    exterior = sAnchor[c] && isExteriorBy(sBase[c], sDir[c], q);
                }

        if (!exterior) dNegate[v] = 1;
    }
};

/// @brief Negate the distance of every voxel slot marked by SignBarrierFunctor.
struct NegateMarkedFunctor
{
    __device__ void operator()(size_t v, const uint8_t* dNegate, float* dSDF) const
    {
        if (dNegate[v]) dSDF[v] = -dSDF[v];
    }
};

} // namespace sdf::detail

/// @brief Sign the barrier voxels of the distances @a dSDF by the chosen BarrierSigning. Interior runs one thread
///        per voxel slot, which is several times faster than the per-leaf launch the heuristic needs.
template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::signBarrier(const GridT* dGrid, float* dSDF, const uint32_t* dNearestTriangle)
{
    const auto tree = TraitsT::getTreeData(dGrid, mStream);
    if (tree.mNodeCount[0] == 0) return;

    static constexpr unsigned int threadsPerBlock = 256;
    const uint64_t slots = tree.mVoxelCount + 1;
    if (mBarrierSigning == BarrierSigning::Interior) {
        util::cuda::lambdaKernel<<<util::cuda::blocksPerGrid(slots, threadsPerBlock), threadsPerBlock, 0, mStream>>>(
            slots, sdf::detail::BarrierToInteriorFunctor{}, mIsoValue, mVoxelSize, dSDF);
        cudaCheckError();
        return;
    }

    // The heuristic reads the distances of the neighbours, so it only marks the voxels to negate, and a second
    // kernel negates them once every voxel is decided.
    BufferT<uint8_t> negate(mStream, this->ref(), slots, nanovdb::cuda::noInit);
    cudaCheck(cudaMemsetAsync(negate.data(), 0, negate.size_bytes(), mStream));
    using FunctorT = sdf::detail::SignBarrierFunctor<BuildT>;
    util::cuda::operatorKernel<FunctorT><<<tree.mNodeCount[0], FunctorT::MaxThreadsPerBlock, 0, mStream>>>(
        dGrid, dSDF, mIsoValue, mVoxelSize,
        dNearestTriangle, mDevicePoints, mDeviceTriangles, mMap, mIsoValue / mVoxelSize, negate.data());
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

/// @brief Count component j as enclosed by the component of @a dGrid if a voxel of component j is in its interior.
/// @details Any voxel of component j gives the same answer: its voxels are connected and none is active in @a dGrid,
///          so they all lie in one region of the inactive space of @a dGrid. The first active voxel of the first leaf
///          of component j's grid is used. That leaf exists and holds an active voxel: every label has a voxel, and
///          PartitionGrid builds an output leaf only where a source leaf keeps a voxel of that label.
template <typename BuildT>
struct AccumulateNestingFunctor
{
    __device__ void operator()(size_t j, const NanoGrid<BuildT>* dGrid,
                               const NanoGrid<BuildT>* const* dComponentGrids,
                               const nanovdb::Mask<3>* dLeafMasks, const nanovdb::Mask<4>* dLowerMasks,
                               const nanovdb::Mask<5>* dUpperMasks, const nanovdb::Coord* dRootOrigins,
                               uint64_t rootOriginCount, uint32_t componentID, uint32_t* dDepth) const
    {
        if (j == componentID) return;
        const auto* leaf = dComponentGrids[j]->tree().getFirstLeaf();
        const nanovdb::Coord ijk = leaf->offsetToGlobalCoord(leaf->valueMask().template findFirst<true>());
        if (isInverted(*dGrid, ijk, dLeafMasks, dLowerMasks, dUpperMasks, dRootOrigins, rootOriginCount))
            ++dDepth[j];
    }
};

} // namespace sdf::detail

/// @brief Add one to the nesting depth of every other component that the component of @a dGrid, signed by @a dSDF,
///        encloses. The test reads the invert masks PropagateSign computes for this component.
template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::accumulateNesting(const GridT* dGrid, const float* dSDF, ComponentLabelT componentID)
{
    const auto tree = TraitsT::getTreeData(dGrid, mStream);
    BufferT<nanovdb::Mask<3>> leafMasks(mStream, this->ref(), tree.mNodeCount[0], nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<4>> lowerMasks(mStream, this->ref(), tree.mNodeCount[1], nanovdb::cuda::noInit);
    BufferT<nanovdb::Mask<5>> upperMasks(mStream, this->ref(), tree.mNodeCount[2], nanovdb::cuda::noInit);
    const auto rootOrigins = PropagateSign<BuildT, float, ResourceT>(dGrid, dSDF, mStream, *mResource)
                                 .propagate(leafMasks.data(), lowerMasks.data(), upperMasks.data()).first;

    using FunctorT = sdf::detail::AccumulateNestingFunctor<BuildT>;
    static constexpr unsigned int threadsPerBlock = 256;
    const auto blocks = util::cuda::blocksPerGrid(mComponentCount, threadsPerBlock);
    util::cuda::lambdaKernel<<<blocks, threadsPerBlock, 0, mStream>>>(
        mComponentCount, FunctorT{}, dGrid, mDeviceComponentGrids.data(),
        leafMasks.data(), lowerMasks.data(), upperMasks.data(),
        rootOrigins.data(), uint64_t(rootOrigins.size()), componentID, mNestingDepth.data());
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::accumulateNesting

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 4: nesting rule.

/// @brief Sign each voxel by the chosen nesting rule, from the sign it has for its own component.
/// @note A voxel is enclosed by the other components that enclose its own component, dDepth[c], plus its own
///       component if the voxel is in that component's interior, i.e. its distance is negative.
struct ApplyNestingRuleFunctor
{
    __device__ void operator()(size_t v, const uint32_t* dComponentLabel, const uint32_t* dDepth, bool evenOdd,
                               float* dSDF) const
    {
        if (v == 0) return;                                  // slot 0 is the background
        const uint32_t enclosing = dDepth[dComponentLabel[v]] + uint32_t(signbit(dSDF[v]));
        const bool     interior  = evenOdd ? enclosing % 2 == 1 : enclosing > 0;
        dSDF[v] = interior ? -fabsf(dSDF[v]) : fabsf(dSDF[v]);
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::applyNestingRule()
{
    if (mComponentCount <= 1) return;

    const uint64_t activeCount = mTreeData.mVoxelCount;
    static constexpr unsigned int threadsPerBlock = 256;
    const auto blocks = util::cuda::blocksPerGrid(activeCount + 1, threadsPerBlock);
    util::cuda::lambdaKernel<<<blocks, threadsPerBlock, 0, mStream>>>(
        activeCount + 1, sdf::detail::ApplyNestingRuleFunctor{}, mComponentLabels.data(), mNestingDepth.data(),
        mNestingRule == NestingRule::EvenOdd, reinterpret_cast<float*>(mSDF.data()));
    cudaCheckError();
} // MeshToSDF<BuildT, ResourceT>::applyNestingRule

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace sdf::detail {

// Stage 5: iso-value offset.

/// @brief Mark in @a dFloor the interior voxels whose distance from the requested isosurface is below
///        @a interiorFloor and that have no exterior face neighbour: IsoMagnitudeFunctor floors them.
/// @details With isoValue > 0, {udf == isoValue} also passes through the interior, e.g. at depth isoValue within a
///          closed mesh. Both sides of such a sheet are interior, so no surface lies there, yet the distance
///          |udf - isoValue| falls to zero across it and would let a contouring or redistancing pass put a surface
///          back. Flooring the interior voxels near it removes these near-zero values. A voxel with an exterior face
///          neighbour is left as is: marching cubes places a crossing between face neighbours of opposite sign, so
///          its small distance is that of the signed surface. The floor removes only the near-zero values; the other
///          interior distances within the mesh remain distances to the interior sheet, not to the signed surface.
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

        const uint64_t v        = leaf.getValue(uint32_t(n));
        const bool     negative = signbit(dSDF[v]);
        const float    distance = fabsf(fabsf(dSDF[v]) - isoValue);  // from the signed surface
        if (!negative || distance >= interiorFloor) return;   // nothing to floor

        // Preserve voxels that can carry a marching-cubes crossing: those with an exterior face neighbour.
        const nanovdb::Coord ijk = leaf.offsetToGlobalCoord(uint32_t(n));
        auto acc = dGrid->getAccessor();
        for (int axis = 0; axis < 3; ++axis)
            for (const int step : {-1, 1}) {
                nanovdb::Coord neighbour = ijk;
                neighbour[axis] += step;
                if (acc.isActive(neighbour) && !signbit(dSDF[acc.getValue(neighbour)])) return;
            }
        dFloor[v] = 1;
    }
};

/// @brief Convert the signed mesh UDF to signed distance from the requested isosurface, in place, and floor the
///        voxels marked by IsoFloorFunctor at -interiorFloor. Slot 0, the background, is left as is.
struct IsoMagnitudeFunctor
{
    __device__ void operator()(size_t v, float isoValue, float interiorFloor, const uint8_t* dFloor,
                               float* dSDF) const
    {
        if (v == 0) return;
        const bool  negative = signbit(dSDF[v]);
        const float distance = fabsf(fabsf(dSDF[v]) - isoValue);  // from the signed surface
        dSDF[v] = dFloor[v] ? -interiorFloor : (negative ? -distance : distance);
    }
};

} // namespace sdf::detail

template <typename BuildT, typename ResourceT>
void MeshToSDF<BuildT, ResourceT>::finalizeDistances()
{
    if (mIsoValue == 0.f) return;

    // The barrier half-width. TODO: state why the floor must be this value.
    const float interiorFloor = sdf::detail::BarrierHalfWidth * mVoxelSize;

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
    const float bg = mBandWidth * mVoxelSize;
    cudaCheck(cudaMemcpyAsync(mSDF.data(), &bg, sizeof(float), cudaMemcpyHostToDevice, mStream));
    cudaCheck(cudaStreamSynchronize(mStream));
} // MeshToSDF<BuildT, ResourceT>::finalizeDistances

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

} // namespace tools::cuda

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_MESHTOSDF_CUH_HAS_BEEN_INCLUDED
