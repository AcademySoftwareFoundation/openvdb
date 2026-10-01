// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Efty Sifakis
///
/// @file Benchmark.h
///
/// @brief Full declaration of the Benchmark singleton class.
///
/// @details The Benchmark class is split across two translation units:
///   - BenchmarkIO.cpp  : OpenVDB/NanoVDB bridge, I/O, and comparison routines
///   - Benchmark.cu     : NanoVDB CUDA simulation kernels

#pragma once

// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/math/Stencils.h>

// NanoVDB (host-side only; CUDA kernels include additional .cuh headers)
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/tools/VoxelBlockManager.h>

/// @brief NanoVDB level-set simulation singleton: proof-of-concept for a
///        GPU-accelerated level-set propagation pipeline alongside OpenVDB.
///
/// @details
/// This class is a **singleton** so that NanoVDB simulation calls can be
/// injected at arbitrary synchronization points during an existing
/// OpenVDB-driven simulation loop (LevelSetPropagate / LevelSetTrackerNew)
/// without restructuring the host control flow.  Each call to getInstance()
/// returns the single shared state (grid handle, sidecar buffers,
/// VoxelBlockManager) that persists across frames.
///
/// The class is organized into two logical sections:
///
/// **1. OpenVDB / NanoVDB bridge** (implemented in BenchmarkIO.cpp)
///   Host-only I/O helpers and correctness-comparison routines that transfer
///   data between an OpenVDB grid on the CPU and a NanoVDB IndexGrid + sidecar
///   buffer on the GPU.  No CUDA device code.
///
/// **2. NanoVDB simulation modules** (implemented in Benchmark.cu)
///   CUDA implementations of the level-set update pipeline:
///     - VoxelBlockManager initialization
///     - Narrow-band topological dilation + sidecar injection
///     - WENO5 level-set normalization (TVD-RK2)
///     - Narrow-band topological pruning + sidecar injection
///     - Speed-grid update (lateral-ratio speed function)
///     - Level-set propagation (TVD-RK2)
///
/// @note Simulation state (mHandle, mPhi, mSpeed, mVBMHandle) uses
///   UnifiedBuffer (cudaMallocManaged) so that future CPU implementations can
///   access the same allocations and be validated side-by-side against the
///   CUDA path without explicit host/device transfers.
///
/// @note The VoxelBlockManager (VBM) is the acceleration structure that maps
///   contiguous blocks of active voxels onto CUDA thread blocks.  BlockWidth
///   active voxels are processed per CUDA block, indexed via the JumpMap +
///   FirstLeafID arrays maintained by the VBMHandle.
/// @brief Selects the execution path for dual-templated simulation methods.
enum class ExecutionPolicy { CPU, CUDA };

class Benchmark
{
    // -----------------------------------------------------------------------
    // Singleton enforcement
    // -----------------------------------------------------------------------
    Benchmark();
    Benchmark(Benchmark const&) = delete;
    Benchmark& operator=(Benchmark const&) = delete;

    // -----------------------------------------------------------------------
    // Type aliases
    // -----------------------------------------------------------------------
    using OpenVDBGridT    = openvdb::FloatGrid;
    using ValueType       = OpenVDBGridT::ValueType;
    using BuildT          = nanovdb::ValueOnIndex;
    using GridT           = nanovdb::NanoGrid<BuildT>;
    using BufferT         = nanovdb::cuda::UnifiedBuffer;
    using GridHandleT     = nanovdb::GridHandle<BufferT>;
    using VBMHandleT      = nanovdb::tools::VoxelBlockManagerHandle<BufferT>;
    using OpenVDBStencilT = openvdb::math::NineteenPointStencil<OpenVDBGridT>;
    using OpenCoordT      = openvdb::Coord;

public:

    // -----------------------------------------------------------------------
    /// @name OpenVDB / NanoVDB bridge  (BenchmarkIO.cpp)
    /// @{

    /// @brief Convert an OpenVDB FloatGrid to a NanoVDB ValueOnIndex grid and
    ///        upload it to device memory.
    /// @param grid    Source OpenVDB level-set grid (host).
    /// @param handle  Output GridHandle backed by UnifiedBuffer.
    /// @param verbose Print timing if true.
    static void copyHostOpenVDBToDeviceNanoVDB(OpenVDBGridT& grid, GridHandleT& handle,
                                               const bool verbose = false);

    /// @brief Print topology and memory statistics for a NanoVDB IndexGrid
    ///        to stdout.
    static void printGridDiagnostics(GridHandleT& handle);

    /// @brief Return true if the NanoVDB IndexGrid in @p handle has
    ///        bit-identical topology to a freshly converted copy of @p grid.
    static bool compareHostOpenVDBToDeviceNanoVDB(OpenVDBGridT& grid, GridHandleT& handle);

    /// @brief Allocate a sidecar buffer sized to the active-voxel count
    ///        of @p handle and initialize slot 0 to @p background.
    ///
    /// @details The sidecar uses index 0 as a sentinel/background slot;
    ///   active voxels are addressed by the sequential indices stored in the
    ///   IndexGrid leaves. The CUDA path sizes the buffer via a D2H read and
    ///   writes slot 0 with a one-element cudaMemcpy; the CPU path reads the
    ///   host grid's valueCount() and writes slot 0 directly. Both produce the
    ///   same UnifiedBuffer result; the split avoids a spurious page migration.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void initializeGPUSidecarAndBackgroundValue(GridHandleT& handle, BufferT& buffer,
                                                       const ValueType background,
                                                       bool verbose = false);

    /// @brief Copy level-set values from an OpenVDB grid to a NanoVDB sidecar
    ///        buffer via the IndexGrid mapping.
    static void copyOpenVDBDataToNanoVDBSidecar(OpenVDBGridT& grid, GridHandleT& handle,
                                                BufferT& buffer, bool verbose = false);

    /// @brief Copy level-set values from a NanoVDB sidecar buffer back to an
    ///        OpenVDB grid via the IndexGrid mapping.
    static void copyNanoVDBSidecarToOpenVDBData(OpenVDBGridT& grid, GridHandleT& handle,
                                                BufferT& buffer, bool verbose = false);

    /// @brief Return true if the maximum per-voxel difference between the
    ///        OpenVDB grid values and the NanoVDB sidecar is within @p tolerance.
    static bool compareOpenVDBDataToNanoVDBSidecar(const OpenVDBGridT& grid,
                                                   GridHandleT& handle,
                                                   BufferT& buffer,
                                                   ValueType tolerance = 0.);

    /// @brief Validate that an OpenVDB NineteenPointStencil at @p coord matches
    ///        the corresponding WENO5 stencil values in the NanoVDB sidecar,
    ///        including boundary sign-extrapolation for missing neighbors.
    ///        Also cross-checks normSqGrad between OpenVDB and NanoVDB.
    static bool compareStencil(OpenVDBStencilT& stencil, const OpenCoordT& coord,
                                BufferT& buffer);

    /// @brief Return the singleton Benchmark instance.
    static Benchmark& getInstance();

    /// @}

    // -----------------------------------------------------------------------
    /// @name NanoVDB simulation modules  (Benchmark.cu / Benchmark.cpp)
    /// @{

    /// @brief Build or rebuild the VoxelBlockManager for the IndexGrid in
    ///        @p gridHandle, storing the result in @p vbmHandle.
    ///
    /// @details The CUDA specialization calls
    ///   nanovdb::tools::cuda::buildVoxelBlockManager, reading grid dimensions
    ///   from device memory.  The CPU specialization calls
    ///   nanovdb::tools::buildVoxelBlockManager using the host-accessible grid
    ///   pointer.  Both produce a VBMHandleT backed by BufferT (UnifiedBuffer),
    ///   so the result is immediately accessible from both host and device.
    ///   Must be called after any topological change (dilation, pruning).
    ///
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void initializeVoxelBlockManager(GridHandleT& gridHandle,
                                            VBMHandleT&  vbmHandle,
                                            const bool   verbose = false);

    /// @brief Topologically dilate the narrow band by one face-neighbor shell,
    ///        rebuild the VBM, and extrapolate the sidecar into new voxels.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void dilateActiveValues(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                   BufferT& phiBuffer, const ValueType background,
                                   bool verbose = false);

    /// @brief Re-initialize the level set to a signed-distance function using
    ///        @p normCount TVD-RK2 steps of the WENO5 Godunov normalization PDE.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void normalizeLevelSet(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                  typename GridHandleT::BufferType& phiBuffer,
                                  const ValueType background, const int normCount,
                                  const ValueType voxelSize, const bool verbose = false);

    /// @brief Topologically prune voxels whose |phi| >= background, rebuild
    ///        the VBM, and inject surviving sidecar values into the pruned grid.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void pruneNarrowBand(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                BufferT& phiBuffer, const ValueType background,
                                bool verbose = false);

    /// @brief Compute the per-voxel speed field from the current phi gradient
    ///        using the lateral-ratio speed function and store in @p speedBuffer.
    ///
    /// @details Speed function:
    ///   @code
    ///   s = (1 - lateralRatio) * max(0, (grad_z - alpha) * beta) - lateralRatio
    ///   @endcode
    ///   where grad_z is the z-component of the normalized WENO5 gradient,
    ///   alpha = cos(pi*(0.5 - 30/180)), beta = 1/(alpha - 1),
    ///   lateralRatio = 0.2.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void updateSpeedGrid(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                typename GridHandleT::BufferType& phiBuffer,
                                typename GridHandleT::BufferType& speedBuffer,
                                const ValueType voxelSize, const bool verbose = false);

    /// @brief Advect the level set by @p dt using TVD-RK2 and the precomputed
    ///        speed field in @p speedBuffer.
    /// @tparam Policy  ExecutionPolicy::CUDA or ExecutionPolicy::CPU
    template<ExecutionPolicy Policy>
    static void propagateLevelSet(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                  typename GridHandleT::BufferType& phiBuffer,
                                  typename GridHandleT::BufferType& speedBuffer,
                                  const ValueType background, const ValueType dt,
                                  const ValueType voxelSize, const bool verbose = false);

    /// @}

    // -----------------------------------------------------------------------
    /// @name Configuration constants
    /// @{

    /// Log2 of the number of active voxels per VoxelBlockManager block.
    /// BlockWidth = 2^BlockWidthLog2 = 128 active voxels per CUDA block.
    static constexpr int BlockWidthLog2 = 7;
    static constexpr int BlockWidth     = 1 << BlockWidthLog2;

    /// Default thread count for lambdaKernel-launched CUDA kernels.
    /// Kernels with specialized occupancy requirements may override this.
    static constexpr unsigned int mNumThreads = 128;
    static unsigned int numBlocks(unsigned int n) { return (n + mNumThreads - 1) / mNumThreads; }

    /// @}

    // -----------------------------------------------------------------------
    /// @name Simulation state  (owned by the singleton)
    /// @{

    int            mVerbose{0};
    OpenVDBGridT*  mGrid{nullptr};   ///< Non-owning pointer to the driving OpenVDB grid. @todo Remove.
    GridHandleT    mHandle;          ///< NanoVDB IndexGrid (UnifiedBuffer).
    ValueType      mDx;              ///< Voxel size (isotropic).
    ValueType      mBackground;      ///< Narrow-band background / truncation value.
    BufferT        mPhi;             ///< Sidecar: level-set values, indexed by the IndexGrid (UnifiedBuffer).
    BufferT        mSpeed;           ///< Sidecar: per-voxel speed values (UnifiedBuffer).
    VBMHandleT     mVBMHandle;       ///< VoxelBlockManager: maps active-voxel blocks to CUDA blocks.
    ExecutionPolicy mPlatform{ExecutionPolicy::CUDA}; ///< Active execution path for dual-templated methods.

    /// @brief Return true if the active execution path is CPU.
    bool onCPU() const { return mPlatform == ExecutionPolicy::CPU; }

    /// @}
};
