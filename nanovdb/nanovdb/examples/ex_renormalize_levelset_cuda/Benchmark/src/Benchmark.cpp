// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Efty Sifakis
///
/// @file Benchmark.cpp
///
/// @brief CPU implementations of the NanoVDB simulation modules declared in
///        Benchmark.h.  Each function is an explicit specialization of a
///        template declared with an ExecutionPolicy parameter; the CUDA
///        counterparts live in Benchmark.cu.
///
///        No CUDA device code is present in this file; it compiles as a
///        standard C++ translation unit.

#include "Benchmark.h"

#include <openvdb/util/CpuTimer.h>
#include <nanovdb/tools/VoxelBlockManager.h>
#include <nanovdb/tools/DilateGrid.h>   // host-side topological dilation (CPU dilateActiveValues)
#include <nanovdb/tools/PruneGrid.h>    // host-side topological prune (CPU pruneNarrowBand)
#include <nanovdb/util/Injection.h>     // host sidecar value injection (CPU pruneNarrowBand)
#include <nanovdb/util/Timer.h>         // host timer (fractional-ms, matches util::cuda::Timer)
#include <nanovdb/util/WenoStencil.h>
#include <nanovdb/util/ForEach.h>
#include <nanovdb/math/Math.h>          // Tolerance, Sqrt

#include <cmath>        // std::cos
#include <cstdint>
#include <cstring>      // std::memcpy
#include <stdexcept>

// ---------------------------------------------------------------------------

template<>
void
Benchmark::
initializeVoxelBlockManager<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, const bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Initializing VoxelBlockManager [CPU]");

    auto grid = gridHandle.grid<BuildT>();
    if (!grid) throw std::logic_error("No host-accessible IndexGrid found in initializeVoxelBlockManager<CPU>");

    vbmHandle = nanovdb::tools::buildVoxelBlockManager<BlockWidthLog2, BufferT>(grid);

    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CPU path: size the sidecar from the host grid's valueCount() and seed slot 0 (background) with a
// plain host store -- no D2H/H2D copies -- keeping the buffer host-coherent for the CPU consumers.
template<>
void
Benchmark::
initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(GridHandleT& handle, BufferT& buffer, const ValueType background, bool verbose)
{
    nanovdb::util::Timer cpuTimer;
    if (verbose) cpuTimer.start("Initializing sidecar buffer (including background value) [CPU]");

    auto hostGrid = handle.grid<BuildT>();
    if (!hostGrid) throw std::logic_error("No host-accessible IndexGrid found in initializeGPUSidecarAndBackgroundValue<CPU>");
    auto valueCount = hostGrid->valueCount();
    buffer.clear();
    buffer = BufferT::create(valueCount*sizeof(ValueType));
    auto hostData = static_cast<ValueType*>(buffer.data());
    if (!hostData) throw std::logic_error("Buffer allocation unsuccessful in initializeGPUSidecarAndBackgroundValue<CPU>");
    hostData[0] = background;

    if (verbose) cpuTimer.stop();
}

// ---------------------------------------------------------------------------
// Per-VBM-block CPU kernel for one TVD-RK2 stage.  Mirrors Benchmark.cu's
// NormalizationEulerStepFunctor<BuildT, Numerator, Denominator> but uses the
// nanovdb::WenoStencil<Simd<float,W>> + LegacyStencilAccessor pipeline.
//
// Stage parameters:
//   stencilSrc -- buffer feeding the 19-point WENO5 stencil
//   blendSrc   -- buffer providing phi_orig for the RK2 blend (only used when
//                 Numerator != 0; otherwise its values are dead-coded out)
//   dest       -- output buffer
//
// Layout convention (mirrors ex_weno_nanovdb_cpu): for each VBM voxel slot p
// in block bID, the global sidecar index is blockBase + p where blockBase =
// firstOffset + bID * BlockWidth.

namespace {

template<int Numerator, int Denominator, typename GridT, typename VBMHandleT>
void
normalizationEulerStepCPU(const GridT&   indexGrid,
                          VBMHandleT&    vbmHandle,
                          const float*   stencilSrc,
                          const float*   blendSrc,
                          float*         dest,
                          const float    dt,
                          const float    invDx)
{
    using BuildT     = typename GridT::BuildType;
    using LeafT      = nanovdb::NanoLeaf<BuildT>;
    using CPUVBM     = nanovdb::tools::VoxelBlockManager<Benchmark::BlockWidthLog2>;
    constexpr int BlockWidth = Benchmark::BlockWidth;
    constexpr int SIMDw      = 16;
    using FloatV     = nanovdb::util::experimental::Simd    <float, SIMDw>;
    using MaskV      = nanovdb::util::experimental::SimdMask<float, SIMDw>;
    using StencilT   = nanovdb::WenoStencil<FloatV>;
    constexpr int SIZE = StencilT::size();

    const LeafT*    firstLeaf   = indexGrid.tree().template getFirstNode<0>();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();
    const uint64_t  firstOffset = vbmHandle.firstOffset();

    constexpr float alpha = float(Numerator) / float(Denominator);
    constexpr float beta  = 1.f - alpha;

    nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
        [&](const nanovdb::util::Range1D& range) {
            alignas(64) uint32_t leafIndex[BlockWidth];
            alignas(64) uint16_t voxelOffset[BlockWidth];

            alignas(64) float raw_values[SIZE][SIMDw];
            alignas(64) bool  raw_active[SIZE][SIMDw];
            alignas(64) float raw_phi0orig[SIMDw];

            // dx=1 in the stencil -- normSqGrad returns the raw Godunov sum
            // (no inverse-dx scaling), matching CUDA's StencilT::normSqGrad(stencil, 1.f, 1.f).
            StencilT  stencil(1.f);
            // One leaf-only ReadAccessor per TBB task; cache stays warm
            // across the SIZE getValue calls in WenoStencil::gatherIndices().
            nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(indexGrid.tree().root());

            for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                CPUVBM::decodeInverseMaps(
                    &indexGrid, firstLeafID[bID],
                    &jumpMap[bID * CPUVBM::JumpMapLength],
                    firstOffset + bID * BlockWidth,
                    leafIndex, voxelOffset);

                const uint64_t blockBase = firstOffset + (uint64_t)bID * BlockWidth;

                for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                    // -------- Fill phase (scalar scatter into raw_*) --------
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;

                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) {
                            for (int k = 0; k < SIZE; ++k) {
                                raw_values[k][i] = 0.f;
                                raw_active[k][i] = false;
                            }
                            raw_phi0orig[i] = 0.f;
                            continue;
                        }

                        const uint16_t vo = voxelOffset[p];
                        const uint32_t li = leafIndex[p];
                        const auto cOrigin = firstLeaf[li].origin();
                        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                        const nanovdb::Coord center = cOrigin + nanovdb::Coord(lx, ly, lz);

                        uint64_t indices[SIZE];
                        nanovdb::WenoStencil<float>::gatherIndices(acc, center, indices);
                        for (int k = 0; k < SIZE; ++k) {
                            raw_values[k][i] = stencilSrc[indices[k]];
                            raw_active[k][i] = (indices[k] != 0);
                        }
                        raw_phi0orig[i] = blendSrc[blockBase + p];
                    }

                    // -------- Load (per-tap SIMD load into stencil view) --------
                    for (int k = 0; k < SIZE; ++k) {
                        stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::experimental::element_aligned);
                        stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::experimental::element_aligned);
                    }
                    const FloatV phi0_orig(raw_phi0orig, nanovdb::util::experimental::element_aligned);

                    // -------- Sign-extrapolate missing taps (CUDA-equivalent: stencil[k] *= Sign(parent)) --------
                    stencil.extrapolate();

                    // -------- Raw Godunov norm-square gradient (dx=1 in stencil) --------
                    const FloatV normSqGradPhi = stencil.normSqGrad(0.f);

                    // -------- Euler step (matches NormalizationEulerStepFunctor in Benchmark.cu) --------
                    //   v = phi0 / sqrt(phi0^2 + normSqGradPhi + tol);
                    //   v = phi0 - dt * v * (sqrt(normSqGradPhi) * invDx - 1);
                    //   result = (Numerator==0) ? v : alpha * phi0_orig + beta * v;
                    using nanovdb::math::Sqrt;
                    const FloatV phi0   = stencil.values[0];
                    const FloatV phi0sq = phi0 * phi0;
                    const FloatV tol(nanovdb::math::Tolerance<float>::value());

                    FloatV smoothedSign = phi0 / Sqrt(phi0sq + normSqGradPhi + tol);
                    FloatV v = phi0 - dt * smoothedSign * (Sqrt(normSqGradPhi) * invDx - FloatV(1.f));
                    FloatV result;
                    if constexpr (Numerator == 0) {
                        result = v;
                    } else {
                        result = alpha * phi0_orig + beta * v;
                    }

                    // -------- Per-lane scalar store --------
                    alignas(64) float result_lanes[SIMDw];
                    result.copy_to(result_lanes, nanovdb::util::experimental::element_aligned);
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;
                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                        dest[blockBase + p] = result_lanes[i];
                    }
                }
            }
        });
}

} // anonymous namespace

template<>
void
Benchmark::
normalizeLevelSet<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                         typename GridHandleT::BufferType& phiBuffer,
                                         const ValueType background, const int normCount,
                                         const ValueType voxelSize, const bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Normalizing Level Set [CPU]");

    auto* indexGrid = gridHandle.template grid<BuildT>();
    if (!indexGrid) throw std::logic_error("normalizeLevelSet<CPU>: no host-accessible IndexGrid");

    auto* phiData = static_cast<float*>(phiBuffer.data());
    if (!phiData) throw std::logic_error("normalizeLevelSet<CPU>: phi buffer not host-accessible");

    // Allocate temp sidecar with the same byte size as phiBuffer.  Slot 0 holds
    // the background value (the convention required by WenoStencil::extrapolate:
    // missing taps load from sidecar slot 0 and inherit the background magnitude).
    BufferT tempBuffer = BufferT::create(phiBuffer.size());
    auto* tempData = static_cast<float*>(tempBuffer.data());
    if (!tempData) throw std::logic_error("normalizeLevelSet<CPU>: temp buffer allocation failed");
    tempData[0] = background;

    const float dt    = float(voxelSize) * 0.9f;
    const float invDx = 1.f / float(voxelSize);

    for (int n = 0; n < normCount; ++n) {
        // Stage 0/1 (Numerator=0): result = v.  stencilSrc=phi, blendSrc=phi (unused), dest=temp.
        normalizationEulerStepCPU<0, 1>(*indexGrid, vbmHandle, phiData, phiData, tempData, dt, invDx);
        // Stage 1/2 (Numerator=1): result = alpha*phi_orig + beta*v.  stencilSrc=temp, blendSrc=phi, dest=phi.
        normalizationEulerStepCPU<1, 2>(*indexGrid, vbmHandle, tempData, phiData, phiData, dt, invDx);
    }

    if (verbose) cpuTimer.stop();
}

// ---------------------------------------------------------------------------
// Per-VBM-block CPU kernel for one TVD-RK2 propagation stage.  Mirrors
// Benchmark.cu's PropagationEulerStepFunctor<BuildT, Numerator, Denominator>:
// gather + extrapolate + raw-Godunov-norm-square + advection Euler step
//
//   v = phi0 - dt * speed * invdxdx * normSqGradPhi
//   result = (Numerator==0) ? v : alpha * phi_orig + beta * v
//
// Per-lane store gated on (active && speed != 0); voxels with zero speed
// retain their dest value (the temp buffer is pre-filled with phi to make
// this a no-op for stage 0/1, and for stage 1/2 it's a write-back of phi
// which equals the existing dest).

namespace {

template<int Numerator, int Denominator, typename GridT, typename VBMHandleT>
void
propagationEulerStepCPU(const GridT&   indexGrid,
                        VBMHandleT&    vbmHandle,
                        const float*   stencilSrc,
                        const float*   blendSrc,
                        float*         dest,
                        const float*   speedSrc,
                        const float    dt,
                        const float    invdxdx)
{
    using BuildT     = typename GridT::BuildType;
    using LeafT      = nanovdb::NanoLeaf<BuildT>;
    using CPUVBM     = nanovdb::tools::VoxelBlockManager<Benchmark::BlockWidthLog2>;
    constexpr int BlockWidth = Benchmark::BlockWidth;
    constexpr int SIMDw      = 16;
    using FloatV     = nanovdb::util::experimental::Simd    <float, SIMDw>;
    using MaskV      = nanovdb::util::experimental::SimdMask<float, SIMDw>;
    using StencilT   = nanovdb::WenoStencil<FloatV>;
    constexpr int SIZE = StencilT::size();

    const LeafT*    firstLeaf   = indexGrid.tree().template getFirstNode<0>();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();
    const uint64_t  firstOffset = vbmHandle.firstOffset();

    constexpr float alpha = float(Numerator) / float(Denominator);
    constexpr float beta  = 1.f - alpha;

    nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
        [&](const nanovdb::util::Range1D& range) {
            alignas(64) uint32_t leafIndex[BlockWidth];
            alignas(64) uint16_t voxelOffset[BlockWidth];

            alignas(64) float raw_values[SIZE][SIMDw];
            alignas(64) bool  raw_active[SIZE][SIMDw];
            alignas(64) float raw_phi0orig[SIMDw];
            alignas(64) float raw_speed[SIMDw];

            // dx=1 in the stencil -- normSqGrad returns the raw Godunov sum
            // (no inverse-dx scaling).  invdxdx applied at the Euler step.
            StencilT  stencil(1.f);
            nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(indexGrid.tree().root());

            for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                CPUVBM::decodeInverseMaps(
                    &indexGrid, firstLeafID[bID],
                    &jumpMap[bID * CPUVBM::JumpMapLength],
                    firstOffset + bID * BlockWidth,
                    leafIndex, voxelOffset);

                const uint64_t blockBase = firstOffset + (uint64_t)bID * BlockWidth;

                for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                    // -------- Fill phase --------
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;

                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) {
                            for (int k = 0; k < SIZE; ++k) {
                                raw_values[k][i] = 0.f;
                                raw_active[k][i] = false;
                            }
                            raw_phi0orig[i] = 0.f;
                            raw_speed[i]    = 0.f;
                            continue;
                        }

                        const uint16_t vo = voxelOffset[p];
                        const uint32_t li = leafIndex[p];
                        const auto cOrigin = firstLeaf[li].origin();
                        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                        const nanovdb::Coord center = cOrigin + nanovdb::Coord(lx, ly, lz);

                        uint64_t indices[SIZE];
                        nanovdb::WenoStencil<float>::gatherIndices(acc, center, indices);
                        for (int k = 0; k < SIZE; ++k) {
                            raw_values[k][i] = stencilSrc[indices[k]];
                            raw_active[k][i] = (indices[k] != 0);
                        }
                        raw_phi0orig[i] = blendSrc[blockBase + p];
                        raw_speed[i]    = speedSrc[blockBase + p];
                    }

                    // -------- Load --------
                    for (int k = 0; k < SIZE; ++k) {
                        stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::experimental::element_aligned);
                        stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::experimental::element_aligned);
                    }
                    const FloatV phi0_orig(raw_phi0orig, nanovdb::util::experimental::element_aligned);
                    const FloatV speed    (raw_speed,    nanovdb::util::experimental::element_aligned);

                    // -------- Sign-extrapolate missing taps --------
                    stencil.extrapolate();

                    // -------- Raw Godunov norm-square gradient (dx=1 in stencil) --------
                    const FloatV normSqGradPhi = stencil.normSqGrad(0.f);

                    // -------- Advection Euler step --------
                    const FloatV phi0 = stencil.values[0];
                    FloatV v = phi0 - dt * speed * invdxdx * normSqGradPhi;
                    FloatV result;
                    if constexpr (Numerator == 0) {
                        result = v;
                    } else {
                        result = alpha * phi0_orig + beta * v;
                    }

                    // -------- Per-lane scalar store, gated on speed != 0 --------
                    alignas(64) float result_lanes[SIMDw];
                    result.copy_to(result_lanes, nanovdb::util::experimental::element_aligned);
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;
                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                        if (nanovdb::math::isApproxZero(raw_speed[i])) continue;
                        dest[blockBase + p] = result_lanes[i];
                    }
                }
            }
        });
}

} // anonymous namespace

template<>
void
Benchmark::
propagateLevelSet<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                         typename GridHandleT::BufferType& phiBuffer,
                                         typename GridHandleT::BufferType& speedBuffer,
                                         const ValueType background, const ValueType dt,
                                         const ValueType voxelSize, const bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Propagating Level Set [CPU]");

    auto* indexGrid = gridHandle.template grid<BuildT>();
    if (!indexGrid) throw std::logic_error("propagateLevelSet<CPU>: no host-accessible IndexGrid");

    auto* phiData   = static_cast<float*>(phiBuffer.data());
    auto* speedData = static_cast<float*>(speedBuffer.data());
    if (!phiData || !speedData)
        throw std::logic_error("propagateLevelSet<CPU>: phi or speed buffer not host-accessible");

    // Allocate temp sidecar.  Pre-fill with phi (mirrors the CUDA path's
    // cudaMemcpy DeviceToDevice) so that voxels skipped due to zero speed in
    // stage 0/1 retain their phi value at the corresponding temp slot.
    BufferT tempBuffer = BufferT::create(phiBuffer.size());
    auto* tempData = static_cast<float*>(tempBuffer.data());
    if (!tempData) throw std::logic_error("propagateLevelSet<CPU>: temp buffer allocation failed");
    std::memcpy(tempData, phiData, phiBuffer.size());
    tempData[0] = background;

    const float dtScalar  = float(dt);
    const float invdxdx   = 1.f / float(voxelSize * voxelSize);

    // Stage 0/1 (Numerator=0): result = v.  stencilSrc=phi, blendSrc=phi (unused), dest=temp.
    propagationEulerStepCPU<0, 1>(*indexGrid, vbmHandle, phiData, phiData, tempData, speedData, dtScalar, invdxdx);
    // Stage 1/2 (Numerator=1): result = alpha*phi_orig + beta*v.  stencilSrc=temp, blendSrc=phi, dest=phi.
    propagationEulerStepCPU<1, 2>(*indexGrid, vbmHandle, tempData, phiData, phiData, speedData, dtScalar, invdxdx);

    if (verbose) cpuTimer.stop();
}

// ---------------------------------------------------------------------------
// Per-VBM-block CPU kernel for the speed-grid update.  Mirrors Benchmark.cu's
// UpdateSpeedGridFunctor<BuildT>: gather + extrapolate + central-difference
// gradient + lateral-ratio speed function.  Single pass; no RK2.
//
// Per-voxel:
//   grad = (1/(2*dx)) * (v[+1,0,0]-v[-1,0,0], v[0,+1,0]-v[0,-1,0], v[0,0,+1]-v[0,0,-1])
//   gradZ_normalized = grad.z / |grad|
//   inner_speed = (gradZ_normalized < alpha) ? 0 : (gradZ_normalized - alpha) * beta
//   speed = (1 - lateralRatio) * inner_speed - lateralRatio
// where alpha = cos(pi*(0.5 - 30/180)), beta = 1/(alpha-1), lateralRatio = 0.2.

namespace {

template<typename GridT, typename VBMHandleT>
void
updateSpeedGridStepCPU(const GridT&   indexGrid,
                       VBMHandleT&    vbmHandle,
                       const float*   phiSrc,
                       float*         speedDest,
                       const float    inv2Dx)
{
    using BuildT     = typename GridT::BuildType;
    using LeafT      = nanovdb::NanoLeaf<BuildT>;
    using CPUVBM     = nanovdb::tools::VoxelBlockManager<Benchmark::BlockWidthLog2>;
    constexpr int BlockWidth = Benchmark::BlockWidth;
    constexpr int SIMDw      = 16;
    using FloatV     = nanovdb::util::experimental::Simd    <float, SIMDw>;
    using MaskV      = nanovdb::util::experimental::SimdMask<float, SIMDw>;
    using StencilT   = nanovdb::WenoStencil<FloatV>;
    constexpr int SIZE = StencilT::size();

    const LeafT*    firstLeaf   = indexGrid.tree().template getFirstNode<0>();
    const uint32_t  nBlocks     = (uint32_t)vbmHandle.blockCount();
    const uint32_t* firstLeafID = vbmHandle.hostFirstLeafID();
    const uint64_t* jumpMap     = vbmHandle.hostJumpMap();
    const uint64_t  firstOffset = vbmHandle.firstOffset();

    // Speed-function constants (match Benchmark.cu's UpdateSpeedGridFunctor).
    const float sAlpha        = std::cos(nanovdb::math::pi<float>() * (0.5f - 30.0f / 180.f));
    const float sBeta         = 1.f / (sAlpha - 1.f);
    const float sLateralRatio = 0.2f;
    const float sOneMinusLat  = 1.f - sLateralRatio;

    nanovdb::util::forEach(size_t(0), size_t(nBlocks), size_t(1),
        [&](const nanovdb::util::Range1D& range) {
            alignas(64) uint32_t leafIndex[BlockWidth];
            alignas(64) uint16_t voxelOffset[BlockWidth];

            alignas(64) float raw_values[SIZE][SIMDw];
            alignas(64) bool  raw_active[SIZE][SIMDw];

            StencilT  stencil(1.f);
            nanovdb::ReadAccessor<BuildT, 0, -1, -1> acc(indexGrid.tree().root());

            for (size_t bID = range.begin(); bID != range.end(); ++bID) {
                CPUVBM::decodeInverseMaps(
                    &indexGrid, firstLeafID[bID],
                    &jumpMap[bID * CPUVBM::JumpMapLength],
                    firstOffset + bID * BlockWidth,
                    leafIndex, voxelOffset);

                const uint64_t blockBase = firstOffset + (uint64_t)bID * BlockWidth;

                for (int batchStart = 0; batchStart < BlockWidth; batchStart += SIMDw) {
                    // -------- Fill phase --------
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;

                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) {
                            for (int k = 0; k < SIZE; ++k) {
                                raw_values[k][i] = 0.f;
                                raw_active[k][i] = false;
                            }
                            continue;
                        }

                        const uint16_t vo = voxelOffset[p];
                        const uint32_t li = leafIndex[p];
                        const auto cOrigin = firstLeaf[li].origin();
                        const int lx = (vo >> 6) & 7, ly = (vo >> 3) & 7, lz = vo & 7;
                        const nanovdb::Coord center = cOrigin + nanovdb::Coord(lx, ly, lz);

                        uint64_t indices[SIZE];
                        nanovdb::WenoStencil<float>::gatherIndices(acc, center, indices);
                        for (int k = 0; k < SIZE; ++k) {
                            raw_values[k][i] = phiSrc[indices[k]];
                            raw_active[k][i] = (indices[k] != 0);
                        }
                    }

                    // -------- Load --------
                    for (int k = 0; k < SIZE; ++k) {
                        stencil.values  [k] = FloatV(raw_values[k], nanovdb::util::experimental::element_aligned);
                        stencil.isActive[k] = MaskV (raw_active[k], nanovdb::util::experimental::element_aligned);
                    }

                    // -------- Sign-extrapolate missing taps --------
                    stencil.extrapolate();

                    // -------- Central-difference gradient (uses |d|=1 taps only) --------
                    // Indices: pointIndex<+1,0,0>=4, <-1,0,0>=3, <0,+1,0>=10, <0,-1,0>=9, <0,0,+1>=16, <0,0,-1>=15.
                    const FloatV gx = (stencil.values[ 4] - stencil.values[ 3]) * inv2Dx;
                    const FloatV gy = (stencil.values[10] - stencil.values[ 9]) * inv2Dx;
                    const FloatV gz = (stencil.values[16] - stencil.values[15]) * inv2Dx;

                    // -------- Normalize gradient (only z component matters for speed) --------
                    using nanovdb::math::Sqrt;
                    const FloatV gradMag = Sqrt(gx*gx + gy*gy + gz*gz);
                    const FloatV gzNorm  = gz / gradMag;

                    // -------- Speed function --------
                    //   inner_speed = (gzNorm < alpha) ? 0 : (gzNorm - alpha) * beta
                    //   speed = (1 - lateralRatio) * inner_speed - lateralRatio
                    const MaskV  belowAlpha   = gzNorm < FloatV(sAlpha);
                    const FloatV innerSpeed   = nanovdb::math::Select(belowAlpha,
                                                                       FloatV(0.f),
                                                                       (gzNorm - FloatV(sAlpha)) * FloatV(sBeta));
                    const FloatV result       = FloatV(sOneMinusLat) * innerSpeed - FloatV(sLateralRatio);

                    // -------- Per-lane scalar store --------
                    alignas(64) float result_lanes[SIMDw];
                    result.copy_to(result_lanes, nanovdb::util::experimental::element_aligned);
                    for (int i = 0; i < SIMDw; ++i) {
                        const int p = batchStart + i;
                        if (leafIndex[p] == CPUVBM::UnusedLeafIndex) continue;
                        speedDest[blockBase + p] = result_lanes[i];
                    }
                }
            }
        });
}

} // anonymous namespace

template<>
void
Benchmark::
updateSpeedGrid<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
                                       typename GridHandleT::BufferType& phiBuffer,
                                       typename GridHandleT::BufferType& speedBuffer,
                                       const ValueType voxelSize, const bool verbose)
{
    openvdb::util::CpuTimer cpuTimer;
    if (verbose) cpuTimer.start("Updating speed grid [CPU]");

    auto* indexGrid = gridHandle.template grid<BuildT>();
    if (!indexGrid) throw std::logic_error("updateSpeedGrid<CPU>: no host-accessible IndexGrid");

    auto* phiData   = static_cast<float*>(phiBuffer.data());
    auto* speedData = static_cast<float*>(speedBuffer.data());
    if (!phiData || !speedData)
        throw std::logic_error("updateSpeedGrid<CPU>: phi or speed buffer not host-accessible");

    const float inv2Dx = 0.5f / float(voxelSize);

    updateSpeedGridStepCPU(*indexGrid, vbmHandle, phiData, speedData, inv2Dx);

    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CPU execution path for pruneNarrowBand -- fully host-side (no CUDA calls): build the prune mask,
// run the host PruneGrid, rebuild the VBM, and inject surviving sidecar values, all over UnifiedBuffer
// memory accessed through host pointers.
template<>
void
Benchmark::
pruneNarrowBand<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, BufferT& phiBuffer, const ValueType background, bool verbose)
{
    nanovdb::util::Timer cpuTimer;

    auto srcGrid = gridHandle.grid<BuildT>();
    if (!srcGrid) throw std::logic_error("No host-accessible IndexGrid found in pruneNarrowBand<CPU>");
    auto srcLeafCount = srcGrid->tree().nodeCount(0);

    // [Building prune mask] One task per leaf; each sequentially scans its 512 voxels and marks the
    // narrow-band survivors (|phi| < background). Per-leaf-exclusive writes, so plain setOn (the device
    // functor's setOnAtomic is unnecessary). The mask is a BufferT (host-accessible) so the host-side
    // PruneGrid below reads it directly.
    if (verbose) cpuTimer.start("Pruning narrow band [Building prune mask]");
    auto pruneMaskBuffer = BufferT::create(srcLeafCount*sizeof(nanovdb::Mask<3>));
    if (!pruneMaskBuffer.data()) throw std::logic_error("Allocation failure for prune mask in pruneNarrowBand<CPU>");
    {
        auto maskPtr = static_cast<nanovdb::Mask<3>*>(pruneMaskBuffer.data());
        auto hostPhi = static_cast<const ValueType*>(phiBuffer.data());
        if (!hostPhi) throw std::logic_error("No sidecar buffer found in pruneNarrowBand<CPU>");
        const auto threshold = hostPhi[0];
        nanovdb::util::forEach(0, srcLeafCount, 8, [=](const nanovdb::util::Range1D& r) {
            const auto& tree = srcGrid->tree();
            for (auto leafID = r.begin(); leafID != r.end(); ++leafID) {
                const auto& leaf = tree.template getFirstNode<0>()[leafID];
                auto& resultMask = maskPtr[leafID];
                resultMask.setOff();
                for (int offset = 0; offset < 512; ++offset)
                    if (auto n = leaf.data()->getValue(offset))
                        if (nanovdb::math::Abs(hostPhi[n]) < threshold)
                            resultMask.setOn(offset);
            }
        });
    }
    if (verbose) cpuTimer.stop();

    // [Topological pruning] host nanovdb::tools::PruneGrid on the UnifiedBuffer grid + mask.
    if (verbose) cpuTimer.start("Pruning narrow band [Topological pruning]");
    auto hostPruneMask = static_cast<nanovdb::Mask<3>*>(pruneMaskBuffer.data());
    nanovdb::tools::PruneGrid<BuildT> pruner( srcGrid, hostPruneMask );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(0);
    auto prunedHandle = pruner.getHandle<BufferT>();
    if (verbose) cpuTimer.stop();

    // [Updating VoxelBlockManager]
    if (verbose) cpuTimer.start("Pruning narrow band [Updating VoxelBlockManager]");
    initializeVoxelBlockManager<ExecutionPolicy::CPU>(prunedHandle, vbmHandle);
    if (verbose) cpuTimer.stop();

    // [Updating pruned sidecar data] allocate + seed the sidecar host-side, then copy surviving values
    // old->pruned via injectGridData (drives the source/old grid; pruned voxels are a subset, so every
    // destination voxel is written).
    if (verbose) cpuTimer.start("Pruning narrow band [Updating pruned sidecar data]");
    BufferT prunedPhi;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(prunedHandle, prunedPhi, background);
    nanovdb::util::injectGridData<BuildT, ValueType>(
        srcGrid,                          // source (old) grid, host-resident
        prunedHandle.grid<BuildT>(),      // destination (pruned) grid, host-resident
        static_cast<const ValueType*>(phiBuffer.data()),
        static_cast<ValueType*>(prunedPhi.data()),
        srcLeafCount);
    gridHandle.reset(); // TODO: Remove after memory leak in move constructor is fixed
    gridHandle = std::move(prunedHandle);
    phiBuffer.clear(); // TODO: Remove after memory leak in move constructor is fixed
    phiBuffer = std::move(prunedPhi);
    if (verbose) cpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CPU execution path for dilateActiveValues -- fully host-side (no CUDA calls): host topological
// dilation (nanovdb::tools::DilateGrid), CPU VBM rebuild + sidecar allocation, then a host forEach that
// copies pre-existing sidecar values and sign-extrapolates new narrow-band voxels from their nearest
// old-grid face-neighbor.
template<>
void
Benchmark::
dilateActiveValues<ExecutionPolicy::CPU>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, BufferT& phiBuffer, const ValueType background, bool verbose)
{
    nanovdb::util::Timer cpuTimer;

    auto srcGrid = gridHandle.grid<BuildT>();
    if (!srcGrid) throw std::logic_error("No host-accessible IndexGrid found in dilateActiveValues<CPU>");

    // [Topological dilation] host nanovdb::tools::DilateGrid on the UnifiedBuffer grid.
    if (verbose) cpuTimer.start("Dilating active values [Topological dilation]");
    nanovdb::tools::DilateGrid<BuildT> dilator( srcGrid );
    dilator.setOperation(nanovdb::tools::morphology::NN_FACE);
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(0);
    auto dilatedHandle = dilator.getHandle<BufferT>();
    if (verbose) cpuTimer.stop();

    // [Updating VoxelBlockManager and allocating new sidecar]
    if (verbose) cpuTimer.start("Dilating active values [Updating VoxelBlockManager and allocating new sidecar]");
    initializeVoxelBlockManager<ExecutionPolicy::CPU>(dilatedHandle, vbmHandle);
    BufferT dilatedPhi;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(dilatedHandle, dilatedPhi, background);
    if (verbose) cpuTimer.stop();

    // [Injecting & extrapolating sidecar data] one task per new leaf; for each active voxel copy its
    // pre-existing old-grid value, else sign-extrapolate from the nearest old-grid face-neighbor
    // (phi = background * sign(neighbor)). No atomics (each new voxel written once). A per-task
    // ValueAccessor caches the old-grid traversal across the spatially-local probeLeaf + getValue calls.
    if (verbose) cpuTimer.start("Dilating active values [Injecting & extrapolating sidecar data]");
    const std::size_t leafCount = dilatedHandle.grid<BuildT>()->tree().nodeCount(0);
    {
        auto oldGrid = gridHandle.grid<BuildT>();
        auto newGrid = dilatedHandle.grid<BuildT>();
        auto oldData = static_cast<const ValueType*>(phiBuffer.data());
        auto newData = static_cast<ValueType*>(dilatedPhi.data());
        nanovdb::util::forEach(0, leafCount, 8, [=](const nanovdb::util::Range1D& r) {
            auto oldAcc = oldGrid->getAccessor();   // per-task: caches the old-grid traversal path
            const auto& newTree = newGrid->tree();
            for (auto leafID = r.begin(); leafID != r.end(); ++leafID) {
                const auto& newLeaf = newTree.template getFirstNode<0>()[leafID];
                auto oldLeafPtr = oldAcc.probeLeaf(newLeaf.origin());
                for (int n = 0; n < 512; ++n) {
                    if (!newLeaf.isActive(n)) continue;
                    const auto newIdx = newLeaf.data()->getValue(n);
                    if (oldLeafPtr && oldLeafPtr->isActive(n)) { // copy pre-existing value
                        newData[newIdx] = oldData[oldLeafPtr->data()->getValue(n)];
                    } else { // extrapolate sign of nearest face-neighbor (for "faraway" values)
                        const auto coord = newLeaf.offsetToGlobalCoord(n);
                        if      (auto i = oldAcc.getValue(coord.offsetBy( 0, 0, 1))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                        else if (auto i = oldAcc.getValue(coord.offsetBy( 0, 0,-1))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                        else if (auto i = oldAcc.getValue(coord.offsetBy( 0, 1, 0))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                        else if (auto i = oldAcc.getValue(coord.offsetBy( 0,-1, 0))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                        else if (auto i = oldAcc.getValue(coord.offsetBy( 1, 0, 0))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                        else if (auto i = oldAcc.getValue(coord.offsetBy(-1, 0, 0))) newData[newIdx] = newData[0] * nanovdb::math::Sign(oldData[i]);
                    }
                }
            }
        });
    }
    gridHandle.reset(); // TODO: Remove after memory leak in move constructor is fixed
    gridHandle = std::move(dilatedHandle);
    phiBuffer.clear(); // TODO: Remove after memory leak in move constructor is fixed
    phiBuffer = std::move(dilatedPhi);
    if (verbose) cpuTimer.stop();
}
