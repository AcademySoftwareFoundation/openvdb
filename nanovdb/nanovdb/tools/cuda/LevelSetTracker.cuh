// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

/*!
    \file nanovdb/tools/cuda/LevelSetTracker.cuh

    \authors JaeHyun Lee and Efty Sifakis

    \brief Renormalization of a narrow-band level set stored as a NanoVDB indexGrid plus a value sidecar.

    \note Comments marked "Provisional" record choices made without a final decision; IDs such as C1
          refer to the open questions in nanovdb/examples/ex_renormalize_levelset_cuda/RENORMALIZATION_DESIGN.md,
          and "the reference" to the CUDA implementation imported under that directory's Benchmark/.

    \warning The header file contains cuda device code so be sure
             to only include it in .cu files (or other .cuh files)
*/

#ifndef NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED
#define NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/Buffer.h>
#include <nanovdb/cuda/DeviceResource.h>
#include <nanovdb/math/FiniteDifference.h>
#include <nanovdb/math/Stencils.h>
#include <nanovdb/tools/cuda/VoxelBlockManager.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/util/cuda/Util.h> // for operatorKernel

#include <cstddef>      // std::byte
#include <stdexcept>    // std::runtime_error
#include <type_traits>  // std::is_same, std::is_floating_point
#include <utility>      // std::pair, std::move

namespace nanovdb {

namespace tools::cuda {

/// @brief Restores |grad phi| = 1 to a narrow-band level set without moving its zero crossing,
///        by integrating d(phi)/d(tau) = S(phi) (1 - |grad phi|) in pseudo-time.
/// @details The level set is a ValueOnIndex grid and a sidecar phi holding one value per grid index.
///          Slot 0 of the sidecar holds the background (the band's half-width in world units) and stands
///          in for every voxel outside the band, whose sign is taken from the nearest in-band voxel along
///          each stencil axis.
///          The tracker owns the grid, the sidecar, the time-stepping scratch, and the VoxelBlockManager,
///          which all depend on the grid's active-voxel numbering.
/// @tparam BuildT Grid build type; must be ValueOnIndex.
/// @tparam ValueT Sidecar element type.
/// @tparam BufferT Single-space device buffer of the grid, e.g. cuda::Buffer<std::byte, R>; the sidecar and
///         the scratch are the same buffer over ValueT.
template<typename BuildT, typename ValueT, typename BufferT = nanovdb::cuda::Buffer<std::byte>>
class LevelSetTracker
{
    static_assert(std::is_same<BuildT, ValueOnIndex>::value, "LevelSetTracker requires a ValueOnIndex grid");
    static_assert(std::is_floating_point<ValueT>::value, "LevelSetTracker requires floating-point values");
    static_assert(BufferHasDeviceSingle<BufferT>::value,
                  "LevelSetTracker requires a single-space device buffer, e.g. cuda::Buffer<std::byte, R>");
    static_assert(nanovdb::cuda::is_async_resource<typename BufferT::ResourceType>::value,
                  "LevelSetTracker allocates stream-ordered scratch and requires an AsyncResource");

public:
    using GridHandleT    = GridHandle<BufferT>;
    using SidecarBufferT = typename BufferT::template rebind<ValueT>;

    /// @param grid Device handle of the index grid.
    /// @param phi Device sidecar with one value per grid index; slot 0 is overwritten with @a background.
    /// @param background Magnitude assigned to voxels outside the band.
    /// @param stream Stream all work of this tracker is ordered on.
    /// @throw std::runtime_error if the voxels are not isotropic or the leaf nodes are not sequential.
    LevelSetTracker(GridHandleT&& grid, SidecarBufferT&& phi, ValueT background, cudaStream_t stream = 0)
        : mStream(stream)
        , mGrid(std::move(grid))
        , mPhi(std::move(phi))
        , mScratch(stream, mPhi.resource(), 0, nanovdb::cuda::noInit)
        , mBackground(background)
        , mDx(validate(mGrid, stream))
        , mVBM(buildVoxelBlockManager<Log2BlockWidth, BufferT>(
              mGrid.template deviceGrid<BuildT>(), 0, 0, 0, stream, &mGrid.buffer()))
    {
        cudaCheck(cudaMemcpyAsync(mPhi.data(), &mBackground, sizeof(ValueT), cudaMemcpyHostToDevice, mStream));
    }

    LevelSetTracker(const LevelSetTracker&) = delete;
    LevelSetTracker& operator=(const LevelSetTracker&) = delete;
    LevelSetTracker(LevelSetTracker&&) = default;
    LevelSetTracker& operator=(LevelSetTracker&&) = default;

    /// @brief Toggle on and off verbose mode
    /// @param level Verbose level: 0=quiet, 1=timing
    void setVerbose(int level = 1) { mVerbose = level; }

    /// @brief Moves the grid and the sidecar out; the tracker cannot be used afterwards.
    std::pair<GridHandleT, SidecarBufferT> release()
    {
        mVBM.reset();
        return {std::move(mGrid), std::move(mPhi)};
    }

    // Provisional (A2): schemes are runtime enums checked in normalize(), rather than compile-time tags.
    math::BiasedGradientScheme getSpatialScheme() const { return mSpatialScheme; }
    void setSpatialScheme(math::BiasedGradientScheme scheme) { mSpatialScheme = scheme; }

    math::TemporalIntegrationScheme getTemporalScheme() const { return mTemporalScheme; }
    void setTemporalScheme(math::TemporalIntegrationScheme scheme) { mTemporalScheme = scheme; }

    /// @brief Number of pseudo-time steps per normalize(); each step moves information about one voxel.
    int getNormCount() const { return mNormCount; }
    void setNormCount(int n) { mNormCount = n; }

    /// @brief Renormalizes the active values in place with normCount TVD-RK2 steps. The topology is unchanged.
    /// @note The stream is synchronized before returning.
    void normalize()
    {
        if (mSpatialScheme != math::HJWENO5_BIAS || mTemporalScheme != math::TVD_RK2)
            throw std::runtime_error("LevelSetTracker::normalize: only HJWENO5_BIAS with TVD_RK2 is implemented");
        if (mPhi.empty()) throw std::runtime_error("LevelSetTracker::normalize: the grid has been released");
        if (!mVBM.blockCount()) return;

        allocateScratch(scratchCount(math::TVD_RK2));
        const ValueT dt = ValueT(0.9) * mDx; // TVD-RK2 CFL number, as in openvdb::tools::LevelSetTracker
        for (int n = 0; n < mNormCount; ++n) {
            eulerStep<0, 1, Phi, Phi, Scratch>(dt); // scratch = phi - dt * L(phi)
            eulerStep<1, 2, Scratch, Phi, Phi>(dt); // phi = (phi + scratch - dt * L(scratch)) / 2
        }
        cudaCheck(cudaStreamSynchronize(mStream));
    }

private:

    // --- Stages (validate runs in the constructor, the others in normalize). ---

    // Throw unless the voxels are isotropic and the leaf nodes are sequential; return the voxel size.
    static ValueT validate(const GridHandleT& grid, cudaStream_t stream);

    // Allocate count RK temporaries in mScratch, each with the background in slot 0, since a stage
    // may gather its stencil from scratch. Skipped when the size already matches.
    void allocateScratch(int count);

    // One Euler stage over all active voxels:
    // buffer(ResultID) = alpha * buffer(BlendID) + (1 - alpha) * (s - dt * L(s)), s = buffer(StencilID),
    // alpha = Numerator / Denominator.
    template<int Numerator, int Denominator, int StencilID, int BlendID, int ResultID>
    void eulerStep(ValueT dt);

    // Buffer IDs bound by eulerStep: phi, then the RK temporaries in mScratch.
    enum BufferID : int { Phi = 0, Scratch = 1 };
    ValueT* buffer(int id) { return id ? mScratch.data() + (id - 1) * mPhi.size() : mPhi.data(); }

    // Number of RK temporaries a temporal scheme needs.
    static constexpr int scratchCount(math::TemporalIntegrationScheme scheme) { return scheme == math::TVD_RK2 ? 1 : 0; }

    static constexpr int Log2BlockWidth = 7; // 128 active voxels per CUDA block. Provisional (B2): fixed here

    cudaStream_t                     mStream{0};
    int                              mVerbose{0};
    GridHandleT                      mGrid;
    SidecarBufferT                   mPhi;        // slot 0 = background
    SidecarBufferT                   mScratch;    // scratchCount() x mPhi.size(); contents undefined between calls
    ValueT                           mBackground; // Provisional: no separate half-width member (design note 3.5)
    ValueT                           mDx;         // declared before mVBM: validate() must run before the VBM is built
    VoxelBlockManagerHandle<BufferT> mVBM;

    math::BiasedGradientScheme      mSpatialScheme{math::HJWENO5_BIAS};
    math::TemporalIntegrationScheme mTemporalScheme{math::TVD_RK2};
    int                             mNormCount{3};
}; // tools::cuda::LevelSetTracker<BuildT, ValueT, BufferT>

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename BufferT>
ValueT LevelSetTracker<BuildT, ValueT, BufferT>::validate(const GridHandleT& grid, cudaStream_t stream)
{
    GridData data;
    cudaCheck(cudaMemcpyAsync(&data, grid.template deviceGrid<BuildT>(), sizeof(GridData),
                              cudaMemcpyDeviceToHost, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    const auto* header = reinterpret_cast<const NanoGrid<BuildT>*>(&data); // reads GridData members only

    const Vec3d dx = header->voxelSize();
    if (dx[0] != dx[1] || dx[0] != dx[2]) // Provisional: exact; OpenVDB's uniform-scale test uses a tolerance
        throw std::runtime_error("LevelSetTracker: voxels must be isotropic");
    if (!header->template isSequential<0>())
        throw std::runtime_error("LevelSetTracker: leaf nodes must be sequential");
    return ValueT(dx[0]);
}// LevelSetTracker<BuildT, ValueT, BufferT>::validate

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<typename BuildT, typename ValueT, typename BufferT>
void LevelSetTracker<BuildT, ValueT, BufferT>::allocateScratch(int count)
{
    const size_t size = size_t(count) * mPhi.size();
    if (mScratch.size() == size) return;

    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is
    if (mVerbose==1) timer.start("Allocating RK scratch");
    mScratch = SidecarBufferT(mStream, mPhi.resource(), size, nanovdb::cuda::noInit);
    for (int k = 0; k < count; ++k)
        cudaCheck(cudaMemcpyAsync(mScratch.data() + k * mPhi.size(), &mBackground, sizeof(ValueT),
                                  cudaMemcpyHostToDevice, mStream));
    if (mVerbose==1) timer.stop();
}// LevelSetTracker<BuildT, ValueT, BufferT>::allocateScratch

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace levelset::detail {

// One thread per active voxel (decoded through the VBM): gather the 19-point WENO stencil, extend
// the band's sign into the taps that fall outside it, and apply one Euler step of
// d(phi)/d(tau) = S(phi) (1 - |grad phi|), blended with phi by alpha = Numerator / Denominator.
template<typename BuildT, typename ValueT, int Log2BlockWidth, int Numerator, int Denominator>
struct NormalizeEulerFunctor
{
    using VBM = VoxelBlockManager<Log2BlockWidth>;
    // Provisional (B4): WenoStencil serves only for SIZE, WenoPt and normSqGrad; the gather is written here.
    using StencilT = math::WenoStencil<NanoGrid<ValueT>>;

    static constexpr int MaxThreadsPerBlock = VBM::BlockWidth;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    /// @brief Position in the WenoPt layout of the tap at offset @a d (|d| in 1..3) along @a axis.
    __hostdev__ static constexpr int tap(int axis, int d) { return 1 + 6 * axis + (d < 0 ? d + 3 : d + 2); }

    __device__ void operator()(const NanoGrid<BuildT>* grid,
                               const uint32_t* firstLeafID,
                               const uint64_t* jumpMap,
                               uint64_t firstOffset,
                               const ValueT* stencilBuffer,
                               const ValueT* blendBuffer,
                               ValueT* resultBuffer,
                               ValueT dt,
                               ValueT invDx) const
    {
        static_assert(tap(0, -3) == math::WenoPt<-3, 0, 0>::idx && tap(1, 1) == math::WenoPt<0, 1, 0>::idx &&
                      tap(2, 3) == math::WenoPt<0, 0, 3>::idx, "tap() must match the WenoPt layout");

        uint32_t leafIndex;
        uint16_t voxelOffset;
        VBM::decodeInverseMap(grid, firstLeafID[blockIdx.x], jumpMap + VBM::JumpMapLength * blockIdx.x,
                              firstOffset + uint64_t(blockIdx.x) * VBM::BlockWidth, threadIdx.x,
                              leafIndex, voxelOffset);
        if (leafIndex == VBM::UnusedLeafIndex) return;

        const auto& tree = grid->tree();
        const auto& leaf = tree.template getFirstNode<0>()[leafIndex];
        const Coord ijk      = leaf.offsetToGlobalCoord(voxelOffset);
        const Coord localIjk = NanoLeaf<BuildT>::OffsetToLocalCoord(voxelOffset);

        // Gather the sidecar index of every tap; index 0 means "no value": the tap lies outside the band.
        uint64_t idx[StencilT::SIZE] = {};
        idx[0] = leaf.getValue(voxelOffset);
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            // Taps reach at most 3 voxels, so only the neighbor leaf on the voxel's side of its own leaf is needed.
            const bool upper = localIjk[axis] & 4;
            const NanoLeaf<BuildT>* leaves[3] = {nullptr, &leaf, nullptr};
            Coord neighborIjk = ijk;
            neighborIjk[axis] += upper ? 4 : -4;
            leaves[upper ? 2 : 0] = tree.root().probeLeaf(neighborIjk);
            const int stride = 1 << (3 * (2 - axis));
            #pragma unroll
            for (int d = -3; d <= 3; ++d) {
                if (d == 0) continue;
                const int tapLocal = localIjk[axis] + d; // in [-3, 10]: leaves[0] below 0, leaves[2] above 7
                if (const NanoLeaf<BuildT>* tapLeaf = leaves[(tapLocal + 8) >> 3])
                    idx[tap(axis, d)] = tapLeaf->getValue(uint32_t(voxelOffset + ((tapLocal & 7) - localIjk[axis]) * stride));
            }
        }

        ValueT v[StencilT::SIZE];
        #pragma unroll
        for (int i = 0; i < StencilT::SIZE; ++i) v[i] = stencilBuffer[idx[i]];

        // A missing tap read the background from slot 0; give it the sign of the next tap inward.
        // Rings are resolved outward (1, 2, 3), each from the already-resolved ring inside it.
        // Provisional (B1): an inner tap of exactly 0 gives Sign() == 0, so the missing tap becomes 0.
        #pragma unroll
        for (int axis = 0; axis < 3; ++axis) {
            #pragma unroll
            for (int side = -1; side <= 1; side += 2) {
                #pragma unroll
                for (int r = 1; r <= 3; ++r) {
                    const int i = tap(axis, side * r);
                    if (!idx[i]) v[i] *= math::Sign(v[r == 1 ? 0 : tap(axis, side * (r - 1))]);
                }
            }
        }

        // Index-space gradient (invDx2 = 1); the physical scale enters through invDx below.
        // Provisional (C1): the third argument is WENO5's scale2 (epsilon = 1e-6 * scale2) and RealT is
        // ValueT. scale2 = 1 as in the reference; OpenVDB uses 0.01, and dx^2 is the other candidate.
        const ValueT normSqGradPhi = StencilT::normSqGrad(v, ValueT(1), ValueT(1));
        const ValueT phi0 = v[0];
        ValueT s = phi0 / (math::Sqrt(math::Pow2(phi0) + normSqGradPhi) + math::Tolerance<ValueT>::value());
        s = phi0 - dt * s * (math::Sqrt(normSqGradPhi) * invDx - ValueT(1));

        constexpr ValueT alpha = ValueT(Numerator) / ValueT(Denominator);
        resultBuffer[idx[0]] = Numerator ? alpha * blendBuffer[idx[0]] + (ValueT(1) - alpha) * s : s;
    }
}; // NormalizeEulerFunctor

} // namespace levelset::detail

template<typename BuildT, typename ValueT, typename BufferT>
template<int Numerator, int Denominator, int StencilID, int BlendID, int ResultID>
void LevelSetTracker<BuildT, ValueT, BufferT>::eulerStep(ValueT dt)
{
    static_assert(ResultID != StencilID, "a stage must not overwrite the buffer its stencil gathers from");

    using Op = levelset::detail::NormalizeEulerFunctor<BuildT, ValueT, Log2BlockWidth, Numerator, Denominator>;
    util::cuda::Timer timer(mStream); // local: Timer is not safely movable, and the tracker is
    if (mVerbose==1) timer.start("Euler stage of renormalization");
    util::cuda::operatorKernel<Op>
        <<<unsigned(mVBM.blockCount()), Op::MaxThreadsPerBlock, 0, mStream>>>(
            mGrid.template deviceGrid<BuildT>(), mVBM.deviceFirstLeafID(), mVBM.deviceJumpMap(), mVBM.firstOffset(),
            buffer(StencilID), buffer(BlendID), buffer(ResultID), dt, ValueT(1) / mDx);
    cudaCheckError();
    if (mVerbose==1) timer.stop();
}// LevelSetTracker<BuildT, ValueT, BufferT>::eulerStep

} // namespace tools::cuda

} // namespace nanovdb

#endif // NVIDIA_TOOLS_CUDA_LEVELSETTRACKER_CUH_HAS_BEEN_INCLUDED
