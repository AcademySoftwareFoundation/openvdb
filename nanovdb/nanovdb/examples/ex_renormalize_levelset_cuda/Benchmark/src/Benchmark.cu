// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
/// @author Efty Sifakis
///
/// @file Benchmark.cu
///
/// @brief CUDA kernel implementations of the NanoVDB level-set simulation
///        modules declared in Benchmark.h.
///
/// @details
/// Each simulation step follows the same three-phase pattern inside its
/// CUDA kernel functor:
///
///   Phase 1 -- Decode inverse maps
///     VoxelBlockManager::decodeInverseMaps() converts the block-local voxel
///     index (blockIdx.x * BlockWidth + threadIdx.x) back to a (leafIndex,
///     voxelOffset) pair stored in shared memory.
///
///   Phase 2 -- Gather WENO5 stencil indices
///     For each active voxel, the 19 sequential indices needed by the WENO5
///     stencil are resolved from the IndexGrid by probing the up-to-two
///     neighboring leaf nodes along each axis.  Missing (boundary) neighbors
///     are handled by sign-extrapolation from the nearest interior value.
///
///   Phase 3 -- Compute and store result
///     The per-voxel scalar kernel (normalization or propagation) is evaluated
///     from the gathered stencil values and written to the output sidecar.
///
/// The normalization and propagation steps both use TVD-RK2 (Heun's method),
/// split into two half-steps (euler01 / euler12) via the Numerator/Denominator
/// template parameters of their respective functor templates.

#include <cuda.h>  // must come before other includes
#include <cuda_runtime.h>

// Local extension of WenoStencil with static normSqGrad/gradient overloads.
// Must precede Benchmark.h / any <nanovdb/...> include, because upstream
// <nanovdb/math/Stencils.h> shares the same include guard; whichever is
// seen first wins. We want the local (superset) version to win.
#include "Stencils.h"

#include "Benchmark.h"

// NanoVDB CUDA utilities
#include <nanovdb/tools/cuda/DilateGrid.cuh>
#include <nanovdb/tools/cuda/PruneGrid.cuh>
#include <nanovdb/util/cuda/Injection.cuh>
#include <nanovdb/util/cuda/Timer.h>
#include <nanovdb/cuda/DeviceBuffer.h>   // for GPU-only transient allocations

// Forward-declare explicit specializations so that call sites earlier in this
// file (dilateActiveValues, pruneNarrowBand) see them before their definitions.
// The CUDA specialization is defined later in this file; the CPU specialization
// is defined in Benchmark.cpp and linked in.
template<>
void Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CUDA>(
    Benchmark::GridHandleT&, Benchmark::VBMHandleT&, const bool);
template<>
void Benchmark::initializeVoxelBlockManager<ExecutionPolicy::CPU>(
    Benchmark::GridHandleT&, Benchmark::VBMHandleT&, const bool);
template<>
void Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(
    Benchmark::GridHandleT&, Benchmark::BufferT&, const Benchmark::ValueType, bool);
template<>
void Benchmark::initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CPU>(
    Benchmark::GridHandleT&, Benchmark::BufferT&, const Benchmark::ValueType, bool);

// ---------------------------------------------------------------------------
/// @brief CUDA functor: copy sidecar values from an old narrow band into a
///        topologically dilated grid, extrapolating sign for new voxels.
///
/// @details Launched via nanovdb::util::cuda::operatorKernel, warp-per-leaf
///   (4 warps / block, 32 threads / warp).
///   For each active voxel in the dilated grid:
///     - If the voxel existed in the old grid, its value is copied directly.
///     - Otherwise the sign of the nearest face-neighbor from the old grid is
///       used to extrapolate a consistent boundary value (mBackground * sign).
template <typename BuildT, typename ValueType>
struct DilateNarrowBandFunctor
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;
    static constexpr int WarpsPerBlock = MaxThreadsPerBlock >> 5;

    __device__
    void operator()(size_t leafCount,
        typename nanovdb::NanoGrid<BuildT> *d_oldGrid,
        typename nanovdb::NanoGrid<BuildT> *d_newGrid,
        ValueType *d_oldData,
        ValueType *d_newData)
    {
        int warpID = threadIdx.x >> 5;
        int threadInWarpID = threadIdx.x & 0x1f;
        int leafID = blockIdx.x * WarpsPerBlock + warpID;
        // Guard BEFORE indexing the leaf array: the last block has padding warps with
        // leafID >= leafCount, and newTree.getFirstNode<0>()[leafID] / newLeaf.origin() would read
        // out of bounds (illegal access once that address is unmapped).
        if (leafID < leafCount) {
            const auto& oldTree = d_oldGrid->tree();
            const auto& newTree = d_newGrid->tree();
            const auto& newLeaf = newTree.template getFirstNode<0>()[leafID];
            auto oldLeafPtr = oldTree.root().probeLeaf(newLeaf.origin());
            for (int n = threadInWarpID; n < 512; n += 32)
                if (newLeaf.isActive(n))
                    if (oldLeafPtr && oldLeafPtr->isActive(n)) // Copy pre-existing values
                        d_newData[newLeaf.data()->getValue(n)] = d_oldData[oldLeafPtr->data()->getValue(n)];
                    else { // Extrapolate sign of neighbors (for "faraway" values)
                        const auto& coord = newLeaf.offsetToGlobalCoord(n);
                        const auto newIdx = newLeaf.data()->getValue(n);
                        if      (auto oldIdx_pX = oldTree.getValue(coord.offsetBy( 0, 0, 1))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_pX]);
                        else if (auto oldIdx_mX = oldTree.getValue(coord.offsetBy( 0, 0,-1))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_mX]);
                        else if (auto oldIdx_pY = oldTree.getValue(coord.offsetBy( 0, 1, 0))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_pY]);
                        else if (auto oldIdx_mY = oldTree.getValue(coord.offsetBy( 0,-1, 0))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_mY]);
                        else if (auto oldIdx_pZ = oldTree.getValue(coord.offsetBy( 1, 0, 0))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_pZ]);
                        else if (auto oldIdx_mZ = oldTree.getValue(coord.offsetBy(-1, 0, 0))) d_newData[newIdx] = d_newData[0] * nanovdb::math::Sign(d_oldData[oldIdx_mZ]);
                    }
        }
        __syncthreads();
    }
};


template<>
void
Benchmark::
dilateActiveValues<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, BufferT& phiBuffer, const ValueType background, bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;

    auto deviceGrid = gridHandle.deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU IndexGrid found in dilateActiveValues()");

    if (verbose) gpuTimer.start("Dilating active values [Topological dilation]");
    nanovdb::tools::cuda::DilateGrid<BuildT> dilator( deviceGrid );
    dilator.setOperation(nanovdb::tools::morphology::NN_FACE);
    dilator.setChecksum(nanovdb::CheckMode::Default);
    dilator.setVerbose(0);
    auto dilatedHandle = dilator.getHandle<BufferT>();
    if (verbose) gpuTimer.stop();

    if (verbose) gpuTimer.start("Dilating active values [Updating VoxelBlockManager and allocating new sidecar]");

    if (getInstance().onCPU())
        initializeVoxelBlockManager<ExecutionPolicy::CPU>(dilatedHandle, vbmHandle);
    else
        initializeVoxelBlockManager<ExecutionPolicy::CUDA>(dilatedHandle, vbmHandle);

    auto oldDeviceGrid = gridHandle.deviceGrid<BuildT>();
    auto oldDeviceData = static_cast<ValueType*>(phiBuffer.deviceData());
    if (!oldDeviceData) throw std::logic_error("No GPU sidecar data not found in dilateActiveValues()");
    auto newDeviceGrid = dilatedHandle.deviceGrid<BuildT>();
    BufferT dilatedPhi;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(dilatedHandle, dilatedPhi, background);
    auto newDeviceData = static_cast<ValueType*>(dilatedPhi.deviceData());
    if (verbose) gpuTimer.stop();

    if (verbose) gpuTimer.start("Dilating active values [Injecting & extrapolating sidecar data]");
    const std::size_t leafCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(newDeviceGrid).mNodeCount[0];
    using Op = DilateNarrowBandFunctor<BuildT,ValueType>;
    nanovdb::util::cuda::operatorKernel<Op>
        <<<(leafCount+Op::WarpsPerBlock-1)/Op::WarpsPerBlock, Op::MaxThreadsPerBlock>>>
        (leafCount, oldDeviceGrid, newDeviceGrid, oldDeviceData, newDeviceData);    
    gridHandle.reset(); // TODO: Remove after memory leak in move constructor is fixed
    gridHandle = std::move(dilatedHandle);
    phiBuffer.clear(); // TODO: Remove after memory leak in move constructor is fixed
    phiBuffer = std::move(dilatedPhi);
    if (verbose)
        gpuTimer.stop(); // Already inludes synchronization of default stream
    else
        cudaStreamSynchronize(0);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// dilateActiveValues<ExecutionPolicy::CPU> is now fully host-side (no CUDA calls); its definition
// lives in Benchmark.cpp alongside the other CPU specializations.

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<>
void
Benchmark::
initializeVoxelBlockManager<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("Initializing VoxelBlockManager");

    auto deviceGrid = gridHandle.deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU IndexGrid found in initializeVoxelBlockManager");

    vbmHandle = nanovdb::tools::cuda::buildVoxelBlockManager<BlockWidthLog2, BufferT>(deviceGrid);

    if (verbose) gpuTimer.stop();
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// CUDA path: size the sidecar from a D2H valueCount read and seed slot 0 (background) with a
// one-element H2D copy, keeping the buffer device-resident for the GPU consumers.
template<>
void
Benchmark::
initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(GridHandleT& handle, BufferT& buffer, const ValueType background, bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("Initializing GPU sidecar buffer (including background value)");

    auto deviceGrid = handle.deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU IndexGrid found in initializeGPUSidecarAndBackgroundValue()");
    auto valueCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getValueCount(deviceGrid);
    buffer.clear();
    buffer = BufferT::create(valueCount*sizeof(ValueType));
    auto deviceData = static_cast<ValueType*>(buffer.deviceData());
    if (!deviceData) throw std::logic_error("GPU buffer allocation unsuccessful in initializeGPUSidecarAndBackgroundValue()");
    cudaCheck(cudaMemcpy(deviceData, &background, sizeof(ValueType), cudaMemcpyHostToDevice));

    if (verbose) gpuTimer.stop();
}

// ---------------------------------------------------------------------------
/// @brief CUDA functor: one TVD-RK2 half-step of the WENO5 level-set
///        normalization (re-initialization) PDE.
///
/// @details The normalization PDE drives phi towards a signed-distance
///   function:  d(phi)/dt + sign(phi0) * (|grad phi| - 1) = 0
///
///   The Godunov upwind scheme (GodunovsNormSqrd) selects one-sided WENO5
///   finite differences based on the sign of phi at each voxel.
///
///   Launched via nanovdb::util::cuda::dynamicSharedMemoryLauncher (requires
///   shared memory for the decoded leafIndex/voxelOffset arrays).
///
///   TVD-RK2 is split into two kernel launches via the Numerator/Denominator
///   template parameters:
///     - euler01<0,1>: result = phi - dt * F(phi)           (pure Euler step)
///     - euler12<1,2>: result = 0.5*phi + 0.5*(euler01 - dt * F(euler01))
///
/// @tparam Numerator    RK2 blending numerator   (0 for first half, 1 for second)
/// @tparam Denominator  RK2 blending denominator (1 for first half, 2 for second)

namespace{

template<class BuildT, int Numerator, int Denominator>
struct NormalizationEulerStepFunctor
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    static constexpr int BlockWidthLog2 = Benchmark::BlockWidthLog2;
    static constexpr int BlockWidth = Benchmark::BlockWidth;
    static constexpr int JumpMapLength = BlockWidth/64;

    struct SharedStorage {
        uint32_t leafIndex[BlockWidth];
        uint16_t voxelOffset[BlockWidth];
    };

    void __device__
    operator()(
        nanovdb::NanoGrid<BuildT> *grid,
        uint32_t *firstLeafIDArray,        
        uint64_t *jumpMapArray,
        uint64_t firstOffset,
        float *stencilBuffer,
        float *phiBuffer,
        float *resultBuffer,
        const float dt,
        const float invDx,
        char *smem_buf)
    {
        int bID = blockIdx.x;
        int tID = threadIdx.x;
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // Phase 1: Build leafIndex and voxelOffset arrays
        // Supports up to 512 threads/block (designed for 128)

        uint32_t &firstLeafID = firstLeafIDArray[bID];
        uint64_t *jumpMap = jumpMapArray + JumpMapLength * bID;
        int blockFirstOffset = firstOffset + bID * BlockWidth;

        nanovdb::tools::cuda::VoxelBlockManager<BlockWidthLog2>::decodeInverseMaps(
            grid, firstLeafID, jumpMap, blockFirstOffset, &storage.leafIndex[0], &storage.voxelOffset[0]);

        // Phase 2: Compute stencil neighbor lists

        uint64_t data[19] = {};
        const auto& tree = grid->tree();
        using LeafPtrT = const nanovdb::NanoLeaf<BuildT>*;

        if (storage.leafIndex[tID] != 0xffffffff) {
            const auto& leaf = tree.template getFirstNode<0>()[ storage.leafIndex[tID] ];
            const auto coord = leaf.offsetToGlobalCoord( storage.voxelOffset[tID] );
            const auto leafOrigin = leaf.origin();
            const nanovdb::Coord localCoord = leaf.OffsetToLocalCoord( storage.voxelOffset[tID] );
            const auto index = leaf.getValue( storage.voxelOffset[tID] );
            data[0] = index;
            LeafPtrT leafPtrs[3][3] = { { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr } };

            for (int axis = 0; axis < 3; ++axis) {
                auto axialNeighborLeafCoord = coord;
                axialNeighborLeafCoord[axis] += (localCoord[axis] & 0x4) ? 4 : -4;
                leafPtrs[axis][(localCoord[axis] & 0x4) >> 1] = tree.root().probeLeaf(axialNeighborLeafCoord);}

            if (leafPtrs[0][(localCoord.x()+ 5)>>3]) data[nanovdb::math::WenoPt<-3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 3) ? 0500 : -0300) );
            if (leafPtrs[0][(localCoord.x()+ 6)>>3]) data[nanovdb::math::WenoPt<-2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 2) ? 0600 : -0200) );
            if (leafPtrs[0][(localCoord.x()+ 7)>>3]) data[nanovdb::math::WenoPt<-1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 1) ? 0700 : -0100) );
            if (leafPtrs[0][(localCoord.x()+ 9)>>3]) data[nanovdb::math::WenoPt< 1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 7) ? 0100 : -0700) );
            if (leafPtrs[0][(localCoord.x()+10)>>3]) data[nanovdb::math::WenoPt< 2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 6) ? 0200 : -0600) );
            if (leafPtrs[0][(localCoord.x()+11)>>3]) data[nanovdb::math::WenoPt< 3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 5) ? 0300 : -0500) );

            if (leafPtrs[1][(localCoord.y()+ 5)>>3]) data[nanovdb::math::WenoPt< 0,-3, 0>::idx] = leafPtrs[1][(localCoord.y()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 3) ? 0050 : -0030) );
            if (leafPtrs[1][(localCoord.y()+ 6)>>3]) data[nanovdb::math::WenoPt< 0,-2, 0>::idx] = leafPtrs[1][(localCoord.y()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 2) ? 0060 : -0020) );
            if (leafPtrs[1][(localCoord.y()+ 7)>>3]) data[nanovdb::math::WenoPt< 0,-1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 1) ? 0070 : -0010) );
            if (leafPtrs[1][(localCoord.y()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 7) ? 0010 : -0070) );
            if (leafPtrs[1][(localCoord.y()+10)>>3]) data[nanovdb::math::WenoPt< 0, 2, 0>::idx] = leafPtrs[1][(localCoord.y()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 6) ? 0020 : -0060) );
            if (leafPtrs[1][(localCoord.y()+11)>>3]) data[nanovdb::math::WenoPt< 0, 3, 0>::idx] = leafPtrs[1][(localCoord.y()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 5) ? 0030 : -0050) );

            if (leafPtrs[2][(localCoord.z()+ 5)>>3]) data[nanovdb::math::WenoPt< 0, 0,-3>::idx] = leafPtrs[2][(localCoord.z()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 3) ? 0005 : -0003) );
            if (leafPtrs[2][(localCoord.z()+ 6)>>3]) data[nanovdb::math::WenoPt< 0, 0,-2>::idx] = leafPtrs[2][(localCoord.z()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 2) ? 0006 : -0002) );
            if (leafPtrs[2][(localCoord.z()+ 7)>>3]) data[nanovdb::math::WenoPt< 0, 0,-1>::idx] = leafPtrs[2][(localCoord.z()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 1) ? 0007 : -0001) );
            if (leafPtrs[2][(localCoord.z()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 0, 1>::idx] = leafPtrs[2][(localCoord.z()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 7) ? 0001 : -0007) );
            if (leafPtrs[2][(localCoord.z()+10)>>3]) data[nanovdb::math::WenoPt< 0, 0, 2>::idx] = leafPtrs[2][(localCoord.z()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 6) ? 0002 : -0006) );
            if (leafPtrs[2][(localCoord.z()+11)>>3]) data[nanovdb::math::WenoPt< 0, 0, 3>::idx] = leafPtrs[2][(localCoord.z()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 5) ? 0003 : -0005) );
        }

        // Phase 3: Fetch input data and compute Euler step

        if (data[0]) {
            using StencilT = nanovdb::math::WenoStencil<nanovdb::FloatGrid>;
            float stencil[StencilT::SIZE];
            for (int i = 0; i < StencilT::SIZE; i++)
                stencil[i] = stencilBuffer[data[i]];
            if (!data[nanovdb::math::WenoPt< 1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt<-1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);

            if (!data[nanovdb::math::WenoPt< 2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx]);
            if (!data[nanovdb::math::WenoPt<-2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx]);

            if (!data[nanovdb::math::WenoPt< 3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx]);
            if (!data[nanovdb::math::WenoPt<-3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx]);

            static const float alpha = float(Numerator)/float(Denominator);
            static const float beta  = 1.f - alpha;
            const float normSqGradPhi = StencilT::normSqGrad(stencil, 1.f, 1.f);
            const float phi0 = stencil[0];
            float v = phi0 / ( nanovdb::math::Sqrt(nanovdb::math::Pow2(phi0) + normSqGradPhi) +
                nanovdb::math::Tolerance<float>::value() );
            v = phi0 - dt * v * (nanovdb::math::Sqrt(normSqGradPhi) * invDx - 1.f);
            resultBuffer[data[0]] = Numerator ? alpha * phiBuffer[data[0]] + beta * v : v;
        }

    }
}; // NormalizationEulerStepFunctor

template<class BuildT, class GridHandleT, class VBMHandleT, class ValueType>
void normalizationEuler01(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
    typename GridHandleT::BufferType& phiBuffer, typename GridHandleT::BufferType& tempBuffer,
    const ValueType dt, const ValueType invDx, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("NormalizeLevelSet: First half of RK2 timestep (euler01)");

    auto deviceGrid = gridHandle.template deviceGrid<BuildT>();
    auto deviceBuffer0 = static_cast<ValueType*>(phiBuffer.deviceData());
    auto deviceBuffer1 = static_cast<ValueType*>(tempBuffer.deviceData());

    using Op = NormalizationEulerStepFunctor<BuildT, 0, 1>;
    nanovdb::util::cuda::dynamicSharedMemoryLauncher<Op>(
        vbmHandle.blockCount(),             // nBlocks
        sizeof(typename Op::SharedStorage), // smem_size
        cudaStream_t(0),                    // stream
        deviceGrid,                         // grid
        vbmHandle.deviceFirstLeafID(),      // firstLeafID
        vbmHandle.deviceJumpMap(),          // jumpMap
        vbmHandle.firstOffset(),            // firstOffset
        deviceBuffer0,
        deviceBuffer0,
        deviceBuffer1,
        dt,
        invDx
    );

    if (verbose) gpuTimer.stop();
}

template<class BuildT, class GridHandleT, class VBMHandleT, class ValueType>
void normalizationEuler12(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
    typename GridHandleT::BufferType& phiBuffer, typename GridHandleT::BufferType& tempBuffer,
    const ValueType dt, const ValueType invDx, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("NormalizeLevelSet: Second half of RK2 timestep (euler12)");

    auto deviceGrid = gridHandle.template deviceGrid<BuildT>();
    auto deviceBuffer0 = static_cast<ValueType*>(phiBuffer.deviceData());
    auto deviceBuffer1 = static_cast<ValueType*>(tempBuffer.deviceData());

    using Op = NormalizationEulerStepFunctor<BuildT, 1, 2>;
    nanovdb::util::cuda::dynamicSharedMemoryLauncher<Op>(
        vbmHandle.blockCount(),             // nBlocks
        sizeof(typename Op::SharedStorage), // smem_size
        cudaStream_t(0),                    // stream
        deviceGrid,                         // grid
        vbmHandle.deviceFirstLeafID(),      // firstLeafID
        vbmHandle.deviceJumpMap(),          // jumpMap
        vbmHandle.firstOffset(),            // firstOffset
        deviceBuffer1,
        deviceBuffer0,
        deviceBuffer0,
        dt,
        invDx
    );

    if (verbose) gpuTimer.stop();
}

}

template<>
void
Benchmark::
normalizeLevelSet<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, typename GridHandleT::BufferType& phiBuffer, const ValueType background,
    const int normCount, const ValueType voxelSize, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;

    if (verbose) gpuTimer.start("NormalizeLevelSet: Allocating temporary buffer for RK2");
    // TODO(cpu-port): tempBuffer is currently UnifiedBuffer (via BufferT) rather than
    // DeviceBuffer, even though the CUDA path never accesses it from the host.  This is
    // intentional for now: keeping it UnifiedBuffer lets the future CPU SIMD prototype
    // accept the same buffer via data() without an adaptation layer.  The right long-term
    // fix -- DeviceBuffer for the CUDA path, a separate allocation for the CPU path -- requires
    // templating initializeGPUSidecarAndBackgroundValue on buffer type, which is deferred to
    // Phase 2c when the CPU normalize/propagate kernel is written and the CPU allocation
    // pattern is known.  See CPU_PORT_PLAN.md Phase 2c notes.
    BufferT tempBuffer;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(gridHandle, tempBuffer, background);
    if (verbose) gpuTimer.stop();

    for (int n = 0; n < normCount; ++n) {
        float dt=voxelSize*0.9f;
        float invDx=1.0f/voxelSize;
        normalizationEuler01<BuildT>(gridHandle, vbmHandle, phiBuffer, tempBuffer, dt, invDx, verbose);
        normalizationEuler12<BuildT>(gridHandle, vbmHandle, phiBuffer, tempBuffer, dt, invDx, verbose);
    }

    cudaStreamSynchronize(0);
}

// ---------------------------------------------------------------------------
/// @brief CUDA functor: build per-leaf active-voxel bitmasks for narrow-band
///        pruning.
///
/// @details For each leaf, marks voxels whose |phi| < background as active in
///   the output mask.  The mask is consumed by PruneGrid to produce a
///   topologically pruned IndexGrid.  Launched via
///   nanovdb::util::cuda::operatorKernel, one block per leaf, up to 512
///   threads per block (one thread per voxel in the 8^3 = 512 leaf).
template <typename BuildT, typename ValueType>
struct PruneNarrowBandFunctor
{

    static constexpr int MaxThreadsPerBlock = 512;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    __device__
    void operator()(
        const typename nanovdb::NanoGrid<BuildT> *d_grid,
        const ValueType *d_value,
        typename nanovdb::Mask<3> *d_dstLeafMasks)
    {
        int leafID = blockIdx.x;
        int threadID = threadIdx.x;

        const auto &tree = d_grid->tree();
        const auto &leaf = tree.template getFirstNode<0>()[leafID];
        const auto threshold = d_value[0];
        auto &resultMask = d_dstLeafMasks[leafID];
        if (threadID < nanovdb::Mask<3>::WORD_COUNT)
            resultMask.words()[threadID] = 0UL;
        __syncthreads();
        if (auto n = leaf.data()->getValue(threadID))
            if (nanovdb::math::Abs(d_value[n]) < threshold)
                resultMask.setOnAtomic(threadID);
    }
};


template<>
void
Benchmark::
pruneNarrowBand<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, BufferT& phiBuffer, const ValueType background, bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;

    auto deviceGrid = gridHandle.deviceGrid<BuildT>();
    if (!deviceGrid) throw std::logic_error("No GPU IndexGrid found in pruneNarrowBand()");
    auto deviceData = static_cast<ValueType*>(phiBuffer.deviceData());
    if (!deviceData) throw std::logic_error("No GPU sidecar buffer found in pruneNarrowBand()");

    if (verbose) gpuTimer.start("Pruning narrow band [Building prune mask]");
    auto srcLeafCount = nanovdb::util::cuda::DeviceGridTraits<BuildT>::getTreeData(deviceGrid).mNodeCount[0];
    auto pruneMaskBuffer = nanovdb::cuda::DeviceBuffer::create(srcLeafCount*sizeof(nanovdb::Mask<3>), nullptr, false, 0);
    auto devicePruneMask = static_cast<nanovdb::Mask<3>*>(pruneMaskBuffer.deviceData());
    if (!devicePruneMask) throw std::logic_error("Allocation failure for GPU prune mask in pruneNarrowBand()");
    using PruneOp = PruneNarrowBandFunctor<BuildT,ValueType>;
    nanovdb::util::cuda::operatorKernel<PruneOp>
        <<<srcLeafCount, PruneOp::MaxThreadsPerBlock>>>
        (deviceGrid, deviceData, devicePruneMask);
    if (verbose) gpuTimer.stop();

    if (verbose) gpuTimer.start("Pruning narrow band [Topological pruning]");
    nanovdb::tools::cuda::PruneGrid<BuildT> pruner( deviceGrid, devicePruneMask );
    pruner.setChecksum(nanovdb::CheckMode::Default);
    pruner.setVerbose(0);
    auto prunedHandle = pruner.getHandle<BufferT>();
    if (verbose) gpuTimer.stop();

    if (verbose) gpuTimer.start("Pruning narrow band [Updating VoxelBlockManager]");
    if (getInstance().onCPU())
        initializeVoxelBlockManager<ExecutionPolicy::CPU>(prunedHandle, vbmHandle);
    else
        initializeVoxelBlockManager<ExecutionPolicy::CUDA>(prunedHandle, vbmHandle);
    if (verbose) gpuTimer.stop();

    if (verbose) gpuTimer.start("Pruning narrow band [Updating pruned sidecar data]");
    BufferT prunedPhi;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(prunedHandle, prunedPhi, background);
    using InjectionOp = nanovdb::util::cuda::InjectGridDataFunctor<BuildT,ValueType>;
    nanovdb::util::cuda::operatorKernel<InjectionOp>
        <<<srcLeafCount, InjectionOp::MaxThreadsPerBlock>>>(
            gridHandle.deviceGrid<BuildT>(),
            prunedHandle.deviceGrid<BuildT>(),
            static_cast<ValueType*>(phiBuffer.deviceData()),
            static_cast<ValueType*>(prunedPhi.deviceData())
        );
    gridHandle.reset(); // TODO: Remove after memory leak in move constructor is fixed
    gridHandle = std::move(prunedHandle);
    phiBuffer.clear(); // TODO: Remove after memory leak in move constructor is fixed
    phiBuffer = std::move(prunedPhi);
    if (verbose)
        gpuTimer.stop(); // Already inludes synchronization of default stream
    else
        cudaStreamSynchronize(0);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

// pruneNarrowBand<ExecutionPolicy::CPU> is now fully host-side (no CUDA calls); its definition
// lives in Benchmark.cpp alongside the other CPU specializations.

// ---------------------------------------------------------------------------
/// @brief CUDA functor: compute the per-voxel speed field from the WENO5
///        phi gradient using the lateral-ratio speed function.
///
/// @details The speed function models directional etch selectivity:
///   @code
///   grad = WENO5_gradient(phi) / dx   (normalized to unit length)
///   speed = (1 - lateralRatio) * max(0, (grad_z - alpha) * beta) - lateralRatio
///   @endcode
///   Constants: alpha = cos(pi*(0.5 - 30/180)), beta = 1/(alpha-1),
///              lateralRatio = 0.2.
///
///   Launched via nanovdb::util::cuda::dynamicSharedMemoryLauncher (requires
///   shared memory for the decoded leafIndex/voxelOffset arrays).
///   One thread per active voxel.

namespace{

template<class BuildT>
struct UpdateSpeedGridFunctor
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    static constexpr int BlockWidthLog2 = Benchmark::BlockWidthLog2;
    static constexpr int BlockWidth = Benchmark::BlockWidth;
    static constexpr int JumpMapLength = BlockWidth/64;

    __device__ const float sAlpha = std::cos(nanovdb::math::pi<float>()*(0.5f - 30.0f/180.f));
    __device__ const float sBeta = 1.0f/(sAlpha - 1.0f);
    __device__ const float sLateralRatio = 0.2f;

    struct SharedStorage {
        uint32_t leafIndex[BlockWidth];
        uint16_t voxelOffset[BlockWidth];
    };

    void __device__
    operator()(
        nanovdb::NanoGrid<BuildT> *grid,
        uint32_t *firstLeafIDArray,        
        uint64_t *jumpMapArray,
        uint64_t firstOffset,
        float *phiBuffer,
        float *speedBuffer,
        const float inv2Dx,
        char *smem_buf)
    {
        int bID = blockIdx.x;
        int tID = threadIdx.x;
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // Phase 1: Build leafIndex and voxelOffset arrays
        // Supports up to 512 threads/block (designed for 128)

        uint32_t &firstLeafID = firstLeafIDArray[bID];
        uint64_t *jumpMap = jumpMapArray + JumpMapLength * bID;
        int blockFirstOffset = firstOffset + bID * BlockWidth;

        nanovdb::tools::cuda::VoxelBlockManager<BlockWidthLog2>::decodeInverseMaps(
            grid, firstLeafID, jumpMap, blockFirstOffset, &storage.leafIndex[0], &storage.voxelOffset[0]);

        // Phase 2: Compute stencil neighbor lists

        uint64_t data[19] = {};
        const auto& tree = grid->tree();
        using LeafPtrT = const nanovdb::NanoLeaf<BuildT>*;

        if (storage.leafIndex[tID] != 0xffffffff) {
            const auto& leaf = tree.template getFirstNode<0>()[ storage.leafIndex[tID] ];
            const auto coord = leaf.offsetToGlobalCoord( storage.voxelOffset[tID] );
            const auto leafOrigin = leaf.origin();
            const nanovdb::Coord localCoord = leaf.OffsetToLocalCoord( storage.voxelOffset[tID] );
            const auto index = leaf.getValue( storage.voxelOffset[tID] );
            data[0] = index;
            LeafPtrT leafPtrs[3][3] = { { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr } };

            for (int axis = 0; axis < 3; ++axis) {
                auto axialNeighborLeafCoord = coord;
                axialNeighborLeafCoord[axis] += (localCoord[axis] & 0x4) ? 4 : -4;
                leafPtrs[axis][(localCoord[axis] & 0x4) >> 1] = tree.root().probeLeaf(axialNeighborLeafCoord);}

            if (leafPtrs[0][(localCoord.x()+ 5)>>3]) data[nanovdb::math::WenoPt<-3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 3) ? 0500 : -0300) );
            if (leafPtrs[0][(localCoord.x()+ 6)>>3]) data[nanovdb::math::WenoPt<-2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 2) ? 0600 : -0200) );
            if (leafPtrs[0][(localCoord.x()+ 7)>>3]) data[nanovdb::math::WenoPt<-1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 1) ? 0700 : -0100) );
            if (leafPtrs[0][(localCoord.x()+ 9)>>3]) data[nanovdb::math::WenoPt< 1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 7) ? 0100 : -0700) );
            if (leafPtrs[0][(localCoord.x()+10)>>3]) data[nanovdb::math::WenoPt< 2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 6) ? 0200 : -0600) );
            if (leafPtrs[0][(localCoord.x()+11)>>3]) data[nanovdb::math::WenoPt< 3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 5) ? 0300 : -0500) );

            if (leafPtrs[1][(localCoord.y()+ 5)>>3]) data[nanovdb::math::WenoPt< 0,-3, 0>::idx] = leafPtrs[1][(localCoord.y()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 3) ? 0050 : -0030) );
            if (leafPtrs[1][(localCoord.y()+ 6)>>3]) data[nanovdb::math::WenoPt< 0,-2, 0>::idx] = leafPtrs[1][(localCoord.y()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 2) ? 0060 : -0020) );
            if (leafPtrs[1][(localCoord.y()+ 7)>>3]) data[nanovdb::math::WenoPt< 0,-1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 1) ? 0070 : -0010) );
            if (leafPtrs[1][(localCoord.y()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 7) ? 0010 : -0070) );
            if (leafPtrs[1][(localCoord.y()+10)>>3]) data[nanovdb::math::WenoPt< 0, 2, 0>::idx] = leafPtrs[1][(localCoord.y()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 6) ? 0020 : -0060) );
            if (leafPtrs[1][(localCoord.y()+11)>>3]) data[nanovdb::math::WenoPt< 0, 3, 0>::idx] = leafPtrs[1][(localCoord.y()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 5) ? 0030 : -0050) );

            if (leafPtrs[2][(localCoord.z()+ 5)>>3]) data[nanovdb::math::WenoPt< 0, 0,-3>::idx] = leafPtrs[2][(localCoord.z()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 3) ? 0005 : -0003) );
            if (leafPtrs[2][(localCoord.z()+ 6)>>3]) data[nanovdb::math::WenoPt< 0, 0,-2>::idx] = leafPtrs[2][(localCoord.z()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 2) ? 0006 : -0002) );
            if (leafPtrs[2][(localCoord.z()+ 7)>>3]) data[nanovdb::math::WenoPt< 0, 0,-1>::idx] = leafPtrs[2][(localCoord.z()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 1) ? 0007 : -0001) );
            if (leafPtrs[2][(localCoord.z()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 0, 1>::idx] = leafPtrs[2][(localCoord.z()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 7) ? 0001 : -0007) );
            if (leafPtrs[2][(localCoord.z()+10)>>3]) data[nanovdb::math::WenoPt< 0, 0, 2>::idx] = leafPtrs[2][(localCoord.z()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 6) ? 0002 : -0006) );
            if (leafPtrs[2][(localCoord.z()+11)>>3]) data[nanovdb::math::WenoPt< 0, 0, 3>::idx] = leafPtrs[2][(localCoord.z()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 5) ? 0003 : -0005) );
        }

        // Phase 3: Fetch input data and compute Euler step

        if (data[0]) {
            using StencilT = nanovdb::math::WenoStencil<nanovdb::FloatGrid>;
            float stencil[StencilT::SIZE];
            for (int i = 0; i < StencilT::SIZE; i++)
                stencil[i] = phiBuffer[data[i]];
            if (!data[nanovdb::math::WenoPt< 1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt<-1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);

            if (!data[nanovdb::math::WenoPt< 2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx]);
            if (!data[nanovdb::math::WenoPt<-2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx]);

            if (!data[nanovdb::math::WenoPt< 3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx]);
            if (!data[nanovdb::math::WenoPt<-3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx]);

            auto grad = StencilT::gradient(stencil, inv2Dx);
            grad.normalize();
            auto speed = grad[2] < sAlpha ? 0.0f : (grad[2] - sAlpha)*sBeta;
            speedBuffer[data[0]] = (1.0f - sLateralRatio) * speed - sLateralRatio;
        }

    }
}; // UpdateSpeedGridFunctor

}

template<>
void
Benchmark::
updateSpeedGrid<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, typename GridHandleT::BufferType& phiBuffer,
    typename GridHandleT::BufferType& speedBuffer, const ValueType voxelSize, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;


    float inv2Dx=0.5f/voxelSize;

    auto deviceGrid = gridHandle.template deviceGrid<BuildT>();
    auto devicePhiBuffer = static_cast<ValueType*>(phiBuffer.deviceData());
    auto deviceSpeedBuffer = static_cast<ValueType*>(speedBuffer.deviceData());

    using Op = UpdateSpeedGridFunctor<BuildT>;
    nanovdb::util::cuda::dynamicSharedMemoryLauncher<Op>(
        vbmHandle.blockCount(),             // nBlocks
        sizeof(typename Op::SharedStorage), // smem_size
        cudaStream_t(0),                    // stream
        deviceGrid,                         // grid
        vbmHandle.deviceFirstLeafID(),      // firstLeafID
        vbmHandle.deviceJumpMap(),          // jumpMap
        vbmHandle.firstOffset(),            // firstOffset
        devicePhiBuffer,
        deviceSpeedBuffer,
        inv2Dx
    );
 
   cudaStreamSynchronize(0);
}

// ---------------------------------------------------------------------------
/// @brief CUDA functor: one TVD-RK2 half-step of the level-set propagation
///        (advection) PDE using the precomputed speed field.
///
/// @details The propagation PDE is:
///   @code
///   d(phi)/dt + speed * |grad phi| / dx = 0
///   @endcode
///   where |grad phi| is approximated via the WENO5 Godunov scheme and
///   speed is read from the precomputed speedBuffer (one value per voxel).
///   Voxels where speed == 0 are skipped.
///
///   Launched via nanovdb::util::cuda::dynamicSharedMemoryLauncher (requires
///   shared memory for the decoded leafIndex/voxelOffset arrays).
///
///   TVD-RK2 blending follows the same Numerator/Denominator convention as
///   NormalizationEulerStepFunctor.
///
/// @tparam Numerator    RK2 blending numerator   (0 for first half, 1 for second)
/// @tparam Denominator  RK2 blending denominator (1 for first half, 2 for second)

namespace{

template<class BuildT, int Numerator, int Denominator>
struct PropagationEulerStepFunctor
{
    static constexpr int MaxThreadsPerBlock = 128;
    static constexpr int MinBlocksPerMultiprocessor = 1;

    static constexpr int BlockWidthLog2 = Benchmark::BlockWidthLog2;
    static constexpr int BlockWidth = Benchmark::BlockWidth;
    static constexpr int JumpMapLength = BlockWidth/64;

    struct SharedStorage {
        uint32_t leafIndex[BlockWidth];
        uint16_t voxelOffset[BlockWidth];
    };

    void __device__
    operator()(
        nanovdb::NanoGrid<BuildT> *grid,
        uint32_t *firstLeafIDArray,        
        uint64_t *jumpMapArray,
        uint64_t firstOffset,
        float *stencilBuffer,
        float *phiBuffer,
        float *resultBuffer,
        float *speedBuffer,
        const float dt,
        const float invdxdx,
        char *smem_buf)
    {
        int bID = blockIdx.x;
        int tID = threadIdx.x;
        SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        // Phase 1: Build leafIndex and voxelOffset arrays
        // Supports up to 512 threads/block (designed for 128)

        uint32_t &firstLeafID = firstLeafIDArray[bID];
        uint64_t *jumpMap = jumpMapArray + JumpMapLength * bID;
        int blockFirstOffset = firstOffset + bID * BlockWidth;

        nanovdb::tools::cuda::VoxelBlockManager<BlockWidthLog2>::decodeInverseMaps(
            grid, firstLeafID, jumpMap, blockFirstOffset, &storage.leafIndex[0], &storage.voxelOffset[0]);

        // Phase 2: Compute stencil neighbor lists

        uint64_t data[19] = {};
        const auto& tree = grid->tree();
        using LeafPtrT = const nanovdb::NanoLeaf<BuildT>*;

        if (storage.leafIndex[tID] != 0xffffffff) {
            const auto& leaf = tree.template getFirstNode<0>()[ storage.leafIndex[tID] ];
            const auto coord = leaf.offsetToGlobalCoord( storage.voxelOffset[tID] );
            const auto leafOrigin = leaf.origin();
            const nanovdb::Coord localCoord = leaf.OffsetToLocalCoord( storage.voxelOffset[tID] );
            const auto index = leaf.getValue( storage.voxelOffset[tID] );
            data[0] = index;
            LeafPtrT leafPtrs[3][3] = { { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr }, { nullptr, &leaf, nullptr } };

            for (int axis = 0; axis < 3; ++axis) {
                auto axialNeighborLeafCoord = coord;
                axialNeighborLeafCoord[axis] += (localCoord[axis] & 0x4) ? 4 : -4;
                leafPtrs[axis][(localCoord[axis] & 0x4) >> 1] = tree.root().probeLeaf(axialNeighborLeafCoord);}

            if (leafPtrs[0][(localCoord.x()+ 5)>>3]) data[nanovdb::math::WenoPt<-3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 3) ? 0500 : -0300) );
            if (leafPtrs[0][(localCoord.x()+ 6)>>3]) data[nanovdb::math::WenoPt<-2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 2) ? 0600 : -0200) );
            if (leafPtrs[0][(localCoord.x()+ 7)>>3]) data[nanovdb::math::WenoPt<-1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 1) ? 0700 : -0100) );
            if (leafPtrs[0][(localCoord.x()+ 9)>>3]) data[nanovdb::math::WenoPt< 1, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 7) ? 0100 : -0700) );
            if (leafPtrs[0][(localCoord.x()+10)>>3]) data[nanovdb::math::WenoPt< 2, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 6) ? 0200 : -0600) );
            if (leafPtrs[0][(localCoord.x()+11)>>3]) data[nanovdb::math::WenoPt< 3, 0, 0>::idx] = leafPtrs[0][(localCoord.x()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[0] < 5) ? 0300 : -0500) );

            if (leafPtrs[1][(localCoord.y()+ 5)>>3]) data[nanovdb::math::WenoPt< 0,-3, 0>::idx] = leafPtrs[1][(localCoord.y()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 3) ? 0050 : -0030) );
            if (leafPtrs[1][(localCoord.y()+ 6)>>3]) data[nanovdb::math::WenoPt< 0,-2, 0>::idx] = leafPtrs[1][(localCoord.y()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 2) ? 0060 : -0020) );
            if (leafPtrs[1][(localCoord.y()+ 7)>>3]) data[nanovdb::math::WenoPt< 0,-1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 1) ? 0070 : -0010) );
            if (leafPtrs[1][(localCoord.y()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 1, 0>::idx] = leafPtrs[1][(localCoord.y()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 7) ? 0010 : -0070) );
            if (leafPtrs[1][(localCoord.y()+10)>>3]) data[nanovdb::math::WenoPt< 0, 2, 0>::idx] = leafPtrs[1][(localCoord.y()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 6) ? 0020 : -0060) );
            if (leafPtrs[1][(localCoord.y()+11)>>3]) data[nanovdb::math::WenoPt< 0, 3, 0>::idx] = leafPtrs[1][(localCoord.y()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[1] < 5) ? 0030 : -0050) );

            if (leafPtrs[2][(localCoord.z()+ 5)>>3]) data[nanovdb::math::WenoPt< 0, 0,-3>::idx] = leafPtrs[2][(localCoord.z()+ 5)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 3) ? 0005 : -0003) );
            if (leafPtrs[2][(localCoord.z()+ 6)>>3]) data[nanovdb::math::WenoPt< 0, 0,-2>::idx] = leafPtrs[2][(localCoord.z()+ 6)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 2) ? 0006 : -0002) );
            if (leafPtrs[2][(localCoord.z()+ 7)>>3]) data[nanovdb::math::WenoPt< 0, 0,-1>::idx] = leafPtrs[2][(localCoord.z()+ 7)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 1) ? 0007 : -0001) );
            if (leafPtrs[2][(localCoord.z()+ 9)>>3]) data[nanovdb::math::WenoPt< 0, 0, 1>::idx] = leafPtrs[2][(localCoord.z()+ 9)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 7) ? 0001 : -0007) );
            if (leafPtrs[2][(localCoord.z()+10)>>3]) data[nanovdb::math::WenoPt< 0, 0, 2>::idx] = leafPtrs[2][(localCoord.z()+10)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 6) ? 0002 : -0006) );
            if (leafPtrs[2][(localCoord.z()+11)>>3]) data[nanovdb::math::WenoPt< 0, 0, 3>::idx] = leafPtrs[2][(localCoord.z()+11)>>3]->getValue( storage.voxelOffset[tID] + ((localCoord[2] < 5) ? 0003 : -0005) );
        }

        // Phase 3: Fetch input data and compute Euler step

        if (data[0]) {
            using StencilT = nanovdb::math::WenoStencil<nanovdb::FloatGrid>;
            float stencil[StencilT::SIZE];
            for (int i = 0; i < StencilT::SIZE; i++)
                stencil[i] = stencilBuffer[data[i]];
            if (!data[nanovdb::math::WenoPt< 1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt<-1, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-1, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-1>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 0>::idx]);

            if (!data[nanovdb::math::WenoPt< 2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 1>::idx]);
            if (!data[nanovdb::math::WenoPt<-2, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-1, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-2, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-1, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-2>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-1>::idx]);

            if (!data[nanovdb::math::WenoPt< 3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt< 3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0, 3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0, 3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0, 3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0, 2>::idx]);
            if (!data[nanovdb::math::WenoPt<-3, 0, 0>::idx]) stencil[nanovdb::math::WenoPt<-3, 0, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt<-2, 0, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0,-3, 0>::idx]) stencil[nanovdb::math::WenoPt< 0,-3, 0>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0,-2, 0>::idx]);
            if (!data[nanovdb::math::WenoPt< 0, 0,-3>::idx]) stencil[nanovdb::math::WenoPt< 0, 0,-3>::idx] *= nanovdb::math::Sign(stencil[nanovdb::math::WenoPt< 0, 0,-2>::idx]);

            if (!nanovdb::math::isApproxZero(speedBuffer[data[0]])) {
                static const float alpha = float(Numerator)/float(Denominator);
                static const float beta  = 1.f - alpha;
                const float normSqGradPhi = StencilT::normSqGrad(stencil, 1.f, 1.f);
                float v = stencil[0] - dt * speedBuffer[data[0]] * invdxdx * normSqGradPhi;
                resultBuffer[data[0]] = Numerator ? alpha * phiBuffer[data[0]] + beta * v : v;
            }
        }
    }
}; // PropagationEulerStepFunctor

template<class BuildT, class GridHandleT, class VBMHandleT, class ValueType>
void propagationEuler01(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
    typename GridHandleT::BufferType& phiBuffer, typename GridHandleT::BufferType& tempBuffer, typename GridHandleT::BufferType& speedBuffer,
    const ValueType dt, const ValueType invdxdx, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("PropagateLevelSet: First half of RK2 timestep (euler01)");

    auto deviceGrid = gridHandle.template deviceGrid<BuildT>();
    auto devicePhiBuffer0 = static_cast<ValueType*>(phiBuffer.deviceData());
    auto devicePhiBuffer1 = static_cast<ValueType*>(tempBuffer.deviceData());
    auto deviceSpeedBuffer = static_cast<ValueType*>(speedBuffer.deviceData());

    using Op = PropagationEulerStepFunctor<BuildT, 0, 1>;
    nanovdb::util::cuda::dynamicSharedMemoryLauncher<Op>(
        vbmHandle.blockCount(),             // nBlocks
        sizeof(typename Op::SharedStorage), // smem_size
        cudaStream_t(0),                    // stream
        deviceGrid,                         // grid
        vbmHandle.deviceFirstLeafID(),      // firstLeafID
        vbmHandle.deviceJumpMap(),          // jumpMap
        vbmHandle.firstOffset(),            // firstOffset
        devicePhiBuffer0,
        devicePhiBuffer0,
        devicePhiBuffer1,
        deviceSpeedBuffer,
        dt,
        invdxdx
    );

    if (verbose) gpuTimer.stop();
}

template<class BuildT, class GridHandleT, class VBMHandleT, class ValueType>
void propagationEuler12(GridHandleT& gridHandle, VBMHandleT& vbmHandle,
    typename GridHandleT::BufferType& phiBuffer, typename GridHandleT::BufferType& tempBuffer, typename GridHandleT::BufferType& speedBuffer,
    const ValueType dt, const ValueType invdxdx, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;
    if (verbose) gpuTimer.start("PropagateLevelSet: Second half of RK2 timestep (euler12)");

    auto deviceGrid = gridHandle.template deviceGrid<BuildT>();
    auto devicePhiBuffer0 = static_cast<ValueType*>(phiBuffer.deviceData());
    auto devicePhiBuffer1 = static_cast<ValueType*>(tempBuffer.deviceData());
    auto deviceSpeedBuffer = static_cast<ValueType*>(speedBuffer.deviceData());

    using Op = PropagationEulerStepFunctor<BuildT, 1, 2>;
    nanovdb::util::cuda::dynamicSharedMemoryLauncher<Op>(
        vbmHandle.blockCount(),             // nBlocks
        sizeof(typename Op::SharedStorage), // smem_size
        cudaStream_t(0),                    // stream
        deviceGrid,                         // grid
        vbmHandle.deviceFirstLeafID(),      // firstLeafID
        vbmHandle.deviceJumpMap(),          // jumpMap
        vbmHandle.firstOffset(),            // firstOffset
        devicePhiBuffer1,
        devicePhiBuffer0,
        devicePhiBuffer0,
        deviceSpeedBuffer,
        dt,
        invdxdx
    );

    if (verbose) gpuTimer.stop();
}

}

template<>
void
Benchmark::
propagateLevelSet<ExecutionPolicy::CUDA>(GridHandleT& gridHandle, VBMHandleT& vbmHandle, typename GridHandleT::BufferType& phiBuffer, typename GridHandleT::BufferType& speedBuffer,
        const ValueType background, const ValueType dt, const ValueType voxelSize, const bool verbose)
{
    nanovdb::util::cuda::Timer gpuTimer;

    if (verbose) gpuTimer.start("PropagateLevelSet: Allocating temporary buffer for RK2");
    // TODO(cpu-port): see equivalent note in normalizeLevelSet -- same deferred decision.
    BufferT tempBuffer;
    initializeGPUSidecarAndBackgroundValue<ExecutionPolicy::CUDA>(gridHandle, tempBuffer, background);
    cudaCheck(cudaMemcpy(tempBuffer.deviceData(), phiBuffer.deviceData(), phiBuffer.size(), cudaMemcpyDeviceToDevice));
    
    if (verbose) gpuTimer.stop();

    ValueType invdxdx = 1.0f/(voxelSize*voxelSize);

    propagationEuler01<BuildT>(gridHandle, vbmHandle, phiBuffer, tempBuffer, speedBuffer, dt, invdxdx, verbose);
    propagationEuler12<BuildT>(gridHandle, vbmHandle, phiBuffer, tempBuffer, speedBuffer, dt, invdxdx, verbose);

    cudaStreamSynchronize(0);
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

namespace{
    void compareBuffers(void *cpuData_, void *gpuData, std::size_t size)
    {
        auto tempData = new char[size];
        cudaCheck(cudaMemcpy(tempData, gpuData, size, cudaMemcpyHostToDevice));
        auto cpuData = reinterpret_cast<char*>(cpuData_);
        bool equal = true;
        std::size_t firstByte;
        for (std::size_t i = 0; i < size; i++)
            if (cpuData[i] != tempData[i]) { equal = false; firstByte = i; break; }
        if (equal)
            std::cout << "Comparison EQUAL" << std::endl;
        else
            std::cout << "Comparison UNEQUAL at byte " << firstByte << ", cpuData = " << cpuData[firstByte] << ", gpuData = " << tempData[firstByte] << std::endl;
    }
}
