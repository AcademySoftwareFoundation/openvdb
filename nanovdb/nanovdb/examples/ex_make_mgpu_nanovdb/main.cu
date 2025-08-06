// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/cuda/UnifiedBuffer.h>
#include <nanovdb/cuda/DeviceMesh.h>
#include <nanovdb/util/cuda/Timer.h>

#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <curand.h>
#include <mma.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <thread>
#include <type_traits>
#include <map>

namespace {// anonymous namespace

template <typename frag_t>
inline __device__
void to_TF32(frag_t& frag) {
#pragma unroll
    for (int t = 0; t < frag.num_elements; t++)
        frag.x[t] =  nvcuda::wmma::__float_to_tf32(frag.x[t]);
}

__global__
__launch_bounds__(256)
void stencilConvolve_v7(nanovdb::NanoGrid<nanovdb::ValueOnIndex> *deviceGrid, int leafOffset,
    float *inputBuffer, uint32_t *haloIndices, float *haloBuffer,
    float *stencil, float *outputBuffer, float *denseOutputBuffer)
{
    static constexpr int Di = 64;
    static constexpr int Do = 128;
    static constexpr int M = 16;
    static constexpr int N = 16;
    static constexpr int K = 8;

    using InputBufferType = float (&)[][Di];
    using HaloBufferType = float (&)[4][4][6][Di];
    using StencilType = float (&)[3][3][3][Di][Do];
    using OutputBufferType = float (&)[][Do];
    using DenseOutputBufferType = float (&)[2][2][4][Do];
    using SpokeStencilType = float (&)[2][2][4][Di];

    InputBufferType mInputBuffer = reinterpret_cast<InputBufferType>(*inputBuffer);
    OutputBufferType mOutputBuffer = reinterpret_cast<OutputBufferType>(*outputBuffer);
    StencilType mStencil = reinterpret_cast<StencilType>(*stencil);

    const int Bk = (blockIdx.x & 1) * 4;
    const int Bj = ((blockIdx.x >> 1) & 3) * 2;
    const int Bi = ((blockIdx.x >> 3) & 3) * 2;
    const int leafId = (blockIdx.x >> 5) + leafOffset;
    const int tid = threadIdx.x;

    using LeafNodeType = nanovdb::NanoGrid<nanovdb::ValueOnIndex>::TreeType::LeafNodeType;
    const auto& tree = deviceGrid->tree();
    auto acc = tree.getAccessor();
    const LeafNodeType& leaf = tree.template getFirstNode<LeafNodeType>()[leafId];
    const nanovdb::Coord leafOrigin = leaf.origin();
    const auto& valueMask = leaf.valueMask();

    uint64_t activeMask = valueMask.words()[Bi] | valueMask.words()[Bi + 1];
    activeMask &= (0xffffUL << (Bj << 3));
    activeMask &= (0xf0f0f0f0f0f0f0fUL << Bk);
    if (!activeMask) return;

    int II = (tid >> 6) & 0x3;
    int E  =  tid       & 0x3f;

    __shared__ float sBufferRaw[6144]; // 4x4x6 array of elements of size Di=64
    HaloBufferType sHaloBuffer = reinterpret_cast<HaloBufferType>(sBufferRaw[0]);
    __shared__ float sSpokeStencil[2][2][4][Di];
    __shared__ float sOutputBuffer[2][2][4][Do];

    auto origin = leafOrigin.offsetBy(Bi, Bj, Bk);

    // -----------------------------
    for (int jj = 0; jj < 4; jj++)
        for (int kk = 0; kk < 6; kk++) {
            const auto& offset = acc.getValue(origin.offsetBy(II - 1, jj - 1, kk - 1));
            sHaloBuffer[II][jj][kk][E] = mInputBuffer[offset][E];
        }
    __syncthreads();

// -----------------------------

    using a_frag_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major>;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;
    __shared__ float sFragBuffer[8][32][4];
    const int warpID = tid >> 5;
    const int laneID = tid & 0x1f;
    nvcuda::wmma::fill_fragment(c_frag, 0.f);

    for (int di = 0; di <= 2; di++)
    for (int dj = 0; dj <= 2; dj++)
    for (int dk = 0; dk <= 2; dk++) {
        // Create a contiguous copy of the 2x2x4 Spoke-Stencil
        // No sychthreads here, because we take care of that at the end of the loop
        for (int b = 0; b < 2 * 2 * 4 * Di; b += blockDim.x) {
            auto eid = tid + b;
            int ii = ( eid >> 9 ) & 0x1;  // 1-wide in X
            int jj = ( eid >> 8 ) & 0x1;  // 2-wide in Y
            int kk = ( eid >> 6 ) & 0x3;  // 4-wide in Z
            int ee = ( eid >> 0 ) & 0x3f; // 64 entries
            sSpokeStencil[ii][jj][kk][ee] = sHaloBuffer[ii + di][jj + dj][kk + dk][ee];
        }
        __syncthreads();

        {
            int inBase = warpID << 3;

            a_frag_t& a_frag = *reinterpret_cast<a_frag_t*>(sFragBuffer[warpID][laneID]);
            nvcuda::wmma::load_matrix_sync(a_frag, &sSpokeStencil[0][0][0][inBase], 64);
            to_TF32(a_frag);

            __syncthreads();
        }

#pragma unroll
        for (int sweep = 0; sweep < 8; ++sweep) {
            const int inBlock = sweep;
            const int outBlock = warpID;
            const int inBase = inBlock << 3;
            const int outBase = outBlock << 4;

            a_frag_t& a_frag = *reinterpret_cast<a_frag_t*>(sFragBuffer[inBlock][laneID]);

            nvcuda::wmma::load_matrix_sync(b_frag, &mStencil[di][dj][dk][inBase][outBase], 128);
            to_TF32(b_frag);
            nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        }
    }
    nvcuda::wmma::store_matrix_sync(&sOutputBuffer[0][0][0][warpID << 4], c_frag, 128, nvcuda::wmma::mem_row_major);

    // Sparse commit phase
    __syncthreads();
    const int warpI     = (tid >> 8) & 0x1;
    const int warpJ     = (tid >> 7) & 0x1;
    const int warpK     = (tid >> 5) & 0x3;
    const int elementID = laneID | ((tid & 0x200) >> 4);

    // NB: this if fixed for 256 threads/block
    const int elementsPerSM = 32;
    const int xSpanPerSM = 1;
    const int ySpanPerSM = 2;

#pragma unroll
    for (int xOffset = 0; xOffset < 2; xOffset += xSpanPerSM)
    for (int yOffset = 0; yOffset < 2; yOffset += ySpanPerSM) {
        const auto coord = origin.offsetBy(warpI + xOffset, warpJ + yOffset, warpK);
        const auto& offset = acc.getValue(coord);
#pragma unroll
        for (int elementOffset = 0; elementOffset < Do; elementOffset += elementsPerSM)
            mOutputBuffer[offset][elementID + elementOffset] = sOutputBuffer[warpI + xOffset][warpJ + yOffset][warpK][elementID + elementOffset];
    }
}// stencilConvolve_v7

}// anonymous namespace

void testConvolution()
{
    static constexpr int Di = 64, Do = 128;
    using InputBufferType  = std::array<float, Di>;
    using OutputBufferType = std::array<float, Do>;
    using StencilType = float (&)[3][3][3][Di][Do];

    float* stencilHostPtr = new float[Do * 27 * Di];

    std::mt19937 gen(42u);
    std::uniform_real_distribution<float> uniform_dist(-1., 1.);

    for (int i = 0; i < Do * 27 * Di; i++) stencilHostPtr[i] = uniform_dist(gen);

    nanovdb::cuda::DeviceMesh deviceMesh;
    const size_t deviceCount = deviceMesh.deviceCount();
    std::cout << "Number of devices that supports unified memory: " << deviceCount << std::endl;

    // Calculate minimum page size which corresponds to minimum physical allocation granularity
    size_t minGranularity = nanovdb::cuda::minDevicePageSize(deviceMesh);

    // Ensure that we don't split an input and/or output feature across page boundaries
    minGranularity = std::lcm(std::lcm(minGranularity, sizeof(InputBufferType)), sizeof(OutputBufferType));
    size_t valueCountGranularity = minGranularity / min(sizeof(InputBufferType), sizeof(OutputBufferType));

    // Initialize and replicate an IndexGrid on each device
    auto floatHandle = nanovdb::tools::createLevelSetSphere<float>(100, nanovdb::Vec3d(0), 1, 3, nanovdb::Vec3d(0), "test");
    nanovdb::FloatGrid* floatGrid = floatHandle.grid<float>();

    using BufferT = nanovdb::cuda::DeviceBuffer;
    auto indexHandle = nanovdb::tools::createNanoGrid<nanovdb::FloatGrid, nanovdb::ValueOnIndex, BufferT>(*floatGrid, 0u, false, false, 1);
    std::for_each(deviceMesh.begin(), deviceMesh.end(), [&](const nanovdb::cuda::DeviceNode& node) {// copy host buffer to all the device buffers
        cudaCheck(cudaSetDevice(node.id));
        indexHandle.deviceUpload(node.id, node.stream, true);
    });
    auto* indexGrid = indexHandle.grid<nanovdb::ValueOnIndex>();

    auto ceil = [](size_t x, size_t y)->size_t{return ((x + y - 1) / y) * y;};
    const size_t valueCount = ceil(indexGrid->valueCount(), deviceCount * valueCountGranularity);

    const size_t inputAllocationSize  = valueCount * sizeof(InputBufferType);
    const size_t outputAllocationSize = valueCount * sizeof(OutputBufferType);

    nanovdb::cuda::UnifiedBuffer inputBuffer(inputAllocationSize, 2*inputAllocationSize);// over-allocate
    nanovdb::cuda::UnifiedBuffer outputBuffer(outputAllocationSize);

    const size_t deviceValueCount = valueCount / deviceCount;

    // Randomly initialize input features
    std::vector<std::thread> threads;
    std::for_each(deviceMesh.begin(), deviceMesh.end(), [&](const nanovdb::cuda::DeviceNode& node) {
        threads.emplace_back([&](int device, cudaStream_t stream) {
            cudaCheck(cudaSetDevice(device));
            float* inputStripePtr  =  inputBuffer.data<float>(deviceValueCount * Di * device);
            float* outputStripePtr = outputBuffer.data<float>(deviceValueCount * Do * device);

            unsigned long long seed = 42u;
            curandGenerator_t rng;
            curandStatus_t curandStatus = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
            curandStatus = curandSetStream(rng, stream);
            curandStatus = curandSetPseudoRandomGeneratorSeed(rng, seed);
            curandStatus = curandGenerateUniform(rng, inputStripePtr,  deviceValueCount * Di);
            curandStatus = curandGenerateUniform(rng, outputStripePtr, deviceValueCount * Do);
            curandStatus = curandDestroyGenerator(rng);
            NANOVDB_ASSERT(curandStatus == CURAND_STATUS_SUCCESS);

            cudaCheck(cudaStreamSynchronize(stream));

            // If we use managed memory, we need to advise about the usage of the memory range in order to obtain an
            // equivalently optimal paging strategy. cudaMemAdviseSetReadMostly instructs the "paging policy" that data
            // is far more likely to be read than written, cudaMemAdviseSetPreferredLocation suggests the preferred device to place the data on, and cudaMemAdviseSetAccessedBy is a hint about the which devices are accessing the data.
            const size_t inPageSize  = deviceValueCount * Di * sizeof(float), inPageOffset  =  inPageSize * device;// in bytes
            const size_t outPageSize = deviceValueCount * Do * sizeof(float), outPageOffset = outPageSize * device;// in bytes
            inputBuffer.advise(inPageOffset, inPageSize, device, {cudaMemAdviseSetReadMostly, cudaMemAdviseSetPreferredLocation});
            inputBuffer.prefetch(inPageOffset, inPageSize, device, stream);
            outputBuffer.advise(outPageOffset, outPageSize, device, cudaMemAdviseSetPreferredLocation);
            std::for_each(deviceMesh.begin(), deviceMesh.end(), [&](const nanovdb::cuda::DeviceNode& otherNode) {
                inputBuffer.advise(  inPageOffset,  inPageSize, otherNode.id, cudaMemAdviseSetAccessedBy);
                outputBuffer.advise(outPageOffset, outPageSize, otherNode.id, cudaMemAdviseSetAccessedBy);
            });
        }, node.id, node.stream);
    });
    std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
    threads.clear();

    // Spawn a convolution kernel on each device that operates on a disjoint subset of the leaves. The leaves operated on by
    // each device corresponds approximately to the features in the virtual address range that are physically allocated on
    // the same device.
    auto **timers = new nanovdb::util::cuda::Timer*[deviceCount];
    const size_t leafNodeCount = indexGrid->tree().nodeCount(0);
    std::for_each(deviceMesh.begin(), deviceMesh.end(), [&](const nanovdb::cuda::DeviceNode& node) {
        threads.emplace_back([&](int device, cudaStream_t stream) {
            cudaCheck(cudaSetDevice(device));
            timers[device] = new nanovdb::util::cuda::Timer(stream);

            void* stencilDevicePtr;
            cudaCheck(nanovdb::util::cuda::mallocAsync(&stencilDevicePtr, sizeof(float) * Do * 27 * Di, stream));
            cudaCheck(cudaMemcpyAsync(stencilDevicePtr, stencilHostPtr, sizeof(float) * Do * 27 * Di, cudaMemcpyHostToDevice, stream));

            size_t deviceLeafNodeCount = (leafNodeCount + deviceCount - 1) / deviceCount;
            const size_t deviceLeafNodeOffset = deviceLeafNodeCount * device;
            deviceLeafNodeCount = std::min(deviceLeafNodeCount, leafNodeCount - deviceLeafNodeOffset);
            auto deviceIndexGrid =  reinterpret_cast<nanovdb::OnIndexGrid*>(indexHandle.deviceData(device));

            // Run 10 warmup iterations
            float* stencilPtr = (float*)stencilDevicePtr;
            dim3 blockDim(256);
            for (int k = 0; k < 10; ++k) {
                stencilConvolve_v7<<<deviceLeafNodeCount * 2 * 4 * 4, blockDim, 0, stream>>>(
                    deviceIndexGrid, deviceLeafNodeOffset, inputBuffer.data<float>(), nullptr, nullptr, stencilPtr, outputBuffer.data<float>(), nullptr);
            }

            timers[device]->start();
            stencilConvolve_v7<<<deviceLeafNodeCount * 2 * 4 * 4, blockDim, 0, stream>>>(
                deviceIndexGrid, deviceLeafNodeOffset, inputBuffer.data<float>(), nullptr, nullptr, stencilPtr, outputBuffer.data<float>(), nullptr);
            timers[device]->record();
            cudaCheck(nanovdb::util::cuda::freeAsync(stencilDevicePtr, stream));
            cudaCheck(cudaStreamSynchronize(stream));
        }, node.id, node.stream);
    });
    std::for_each(threads.begin(), threads.end(), [](std::thread& t) { t.join(); });
    threads.clear();
    delete[] stencilHostPtr;

    std::for_each(deviceMesh.begin(), deviceMesh.end(), [&](const nanovdb::cuda::DeviceNode& node) {
        timers[node.id]->print("Device " + std::to_string(node.id) + " GPU convolution ", std::cout);
        delete timers[node.id];
    });
    delete[] timers;
    std::cout << "Multi-GPU sparse convolution test complete!" << std::endl;
}// testConvolution

int main(int argc, char** argv)
{
    testConvolution();
    return 0;
}
