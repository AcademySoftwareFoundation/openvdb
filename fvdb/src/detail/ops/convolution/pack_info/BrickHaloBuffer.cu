#include <c10/cuda/CUDAException.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/OpMathType.h>
#include <cute/tensor.hpp>

#include "detail/utils/cuda/Utils.cuh"
#include "detail/ops/convolution/pack_info/PackInfoOps.h"
#include "Types.h"


namespace fvdb {
namespace detail {
namespace ops {

namespace {
    using halo_brick_shape = decltype(cute::make_shape(cute::_6{}, cute::_4{}, cute::_4{}));
    using brick_shape = decltype(cute::make_shape(cute::_4{}, cute::_2{}, cute::_2{}));
    using halo_index_buffer_layout = decltype(cute::make_layout(cute::insert<0>(halo_brick_shape{}, 0), cute::GenRowMajor{}));
    using output_index_buffer_layout = decltype(cute::make_layout(cute::insert<0>(brick_shape{}, 0), cute::GenRowMajor{}));
}

// There are 32 4x2x2 bricks in a leaf node
// This returns the base i,j,k based on laying out the bricks lexicographically,
// in a k-major ordering.
__host__ __device__
auto base_ijk_for_brick_in_leaf(int b_id)
{
    return cute::make_arithmetic_tuple((b_id & 16) >> 2, (b_id & 12) >> 1, (b_id & 3) << 1);
}

template<typename ValueMask>
__host__ __device__
inline uint64_t get_active_mask(const ValueMask& valueMask, const cute::ArithmeticTuple<int, int, int>& B)
{
    const auto [Bi, Bj, Bk] = B;
    uint64_t active_mask =
        valueMask.words()[Bi] |
        valueMask.words()[Bi + 1] |
        valueMask.words()[Bi + 2] |
        valueMask.words()[Bi + 3];
    active_mask &= (0x303UL << Bk) << (Bj * 8);
    return active_mask;
}

template <typename GridType>
__global__ __launch_bounds__(32) void mark_bricks(
        BatchGridAccessor<GridType> gridAcc,
        TorchRAcc32<uint8_t, 1> brick_usage_flags) {
    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    const auto brickId = blockDim.x * blockIdx.x + threadIdx.x;
    const auto brickInLeafId = threadIdx.x;
    const auto leafIdx = blockIdx.x;

    const int64_t batchIdx = gridAcc.leafBatchIndex(leafIdx);
    const int64_t localLeafIdx = leafIdx - gridAcc.leafOffset(batchIdx);

    const nanovdb::NanoGrid<GridType>* deviceGrid = gridAcc.grid(batchIdx);
    const LeafNodeType& leaf = deviceGrid->tree().template getFirstNode<0>()[localLeafIdx];

    const auto& valueMask = leaf.valueMask();
    const uint64_t active_mask =
        get_active_mask(valueMask, base_ijk_for_brick_in_leaf(brickInLeafId));

    brick_usage_flags[brickId] = bool(active_mask);
}

template <int BATCH>
__device__
auto offset_from_tiwid(int tiwid)
{
    static_assert(BATCH >=0 && BATCH < 3);
    const int n = BATCH * 32 + tiwid;
    const int k = n & 3;
    const int j = (n >> 2) & 3;
    const int i = n >> 4;
    return cute::make_arithmetic_tuple(i, j, k);
}

template <typename GridType>
__global__ __launch_bounds__(1024) void populate_halo_index_buffer(
        BatchGridAccessor<GridType> gridAcc,
        TorchRAcc32<uint8_t, 1> brick_usage_flags,
        TorchRAcc32<int, 1> brick_offsets,
        int *halo_index_buffer,
        int *output_index_buffer,
        bool benchmark) {         // Use raw pointer and templated cute to accelerate pointer arithmetic

    using LeafNodeType = typename nanovdb::NanoTree<GridType>::LeafNodeType;

    const int leafIdx = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = threadIdx.x / 32;      // == brick ID in leaf
    const int tiwid = threadIdx.x % 32;    // thread in warp
    const int brickId = 32 * leafIdx + wid; // This is the global brick ID

    const int64_t batchIdx = gridAcc.leafBatchIndex(leafIdx);
    const int64_t localLeafIdx = leafIdx - gridAcc.leafOffset(batchIdx);
    const int64_t baseOffset = gridAcc.voxelOffset(batchIdx);

    const nanovdb::NanoGrid<GridType>* deviceGrid = gridAcc.grid(batchIdx);
    const LeafNodeType& leaf = deviceGrid->tree().template getFirstNode<0>()[localLeafIdx];
    const nanovdb::Coord origin = leaf.origin();
    auto deviceGridAcc = deviceGrid->getAccessor();

    __shared__ uint32_t sHaloBuffer[10][10][10];

    if (tid < 1000) {
        const int di = ((tid / 100) % 10) - 1;
        const int dj = ((tid / 10) % 10) - 1;
        const int dk = (tid % 10) - 1;

        auto coord = origin.offsetBy(di, dj, dk);

        // NOTE: Put 0 for inactive indices and shift feature by 1.
        if (deviceGridAcc.template get<ActiveOrUnmasked<GridType>>(coord)) {
            const int offset = benchmark ? -1 : 0;
            sHaloBuffer[0][0][tid] = deviceGridAcc.getValue(coord) + baseOffset + offset;
        }
        else {
            sHaloBuffer[0][0][tid] = 0;
        }
    }
    __syncthreads();

    using cute::_;

    if (!brick_usage_flags[brickId]) return;
    static const auto full_halo_layout = cute::make_layout(cute::Shape<cute::_10, cute::_10, cute::_10>{}, cute::GenRowMajor{});

    cute::Tensor s_full_halo = cute::make_tensor(cute::make_smem_ptr(&sHaloBuffer[0][0][0]), full_halo_layout);

    cute::Tensor g_halo_index_buffer = cute::make_tensor(
        cute::make_gmem_ptr(halo_index_buffer), halo_index_buffer_layout{}
    );
    cute::Tensor g_brick_halo = g_halo_index_buffer(brick_offsets[brickId], _, _, _);

    cute::Tensor g_output_index_buffer = cute::make_tensor(
        cute::make_gmem_ptr(output_index_buffer), output_index_buffer_layout{}
    );
    cute::Tensor g_output_brick = g_output_index_buffer(brick_offsets[brickId], _, _, _);

    // Note the full aritmethic is this:
    // base_ijk_in_halo_buffer = base_ijk + 1
    // real_halo_offset = offset_ijk - 1
    // halo(real_halo_offset + 1) = sHaloBuffer(base_ijk_in_halo_buffer + real_halo_offset)
    // i.e the +1s and -1s cancel out
    const auto ones = cute::make_arithmetic_tuple(1, 1, 1);
    const auto base_ijk = base_ijk_for_brick_in_leaf(wid);

    const auto offset_ijk0 = offset_from_tiwid<0>(tiwid);
    g_brick_halo(offset_ijk0) = s_full_halo(base_ijk + offset_ijk0);

    const auto offset_ijk1 = offset_from_tiwid<1>(tiwid);
    g_brick_halo(offset_ijk1) = s_full_halo(base_ijk + offset_ijk1);

    const auto offset_ijk2 = offset_from_tiwid<2>(tiwid);
    g_brick_halo(offset_ijk2) = s_full_halo(base_ijk + offset_ijk2);

    if (tiwid < 16) {
        const auto output_crd = cute::idx2crd(tiwid, brick_shape{});
        g_output_brick(output_crd) = s_full_halo(base_ijk + ones + output_crd);
    }
}

template <>
std::vector<torch::Tensor> dispatchBrickHaloBuffer<torch::kCUDA>(const GridBatchImpl& batchHdl, bool benchmark) {

    const size_t num_leaf_nodes = batchHdl.totalLeaves();
    const size_t num_bricks = num_leaf_nodes * 32;
    const size_t TileM = 4;

    // (num_total_bricks,) either 0 or 1 indicating if the brick is occupied
    torch::Tensor brick_usage_flags = torch::empty(
        {(int64_t) num_bricks}, torch::dtype(torch::kUInt8).device(batchHdl.device()));

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        auto batchAcc = gridBatchAccessor<torch::kCUDA, GridType>(batchHdl);
        mark_bricks<GridType><<<num_leaf_nodes, 32>>>(
            batchAcc,
            brick_usage_flags.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>()
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    // (num_total_bricks,) the index of the brick in the halo buffer
    // values in non-occupied bricks are undefined!
    torch::Tensor brick_offsets = torch::cumsum(brick_usage_flags, 0, torch::kInt32) - 1;
    const int32_t num_active_bricks = brick_offsets[-1].item<int32_t>() + 1;
    const size_t num_active_bricks_with_padding = ((num_active_bricks + TileM - 1) / TileM) * TileM;

    // (#active_brick, 6, 4, 4) the index of the brick in the halo buffer
    // (#active_brick, 4, 2, 2) the index of the brick in the output (sub-portion of the above)
    torch::Tensor halo_index_buffer = torch::zeros(
        {(int64_t) num_active_bricks_with_padding * size(halo_brick_shape{})},
        torch::dtype(torch::kInt32).device(batchHdl.device()));
    torch::Tensor output_index_buffer = torch::zeros(
        {(int64_t) num_active_bricks_with_padding * size(brick_shape{})},
        torch::dtype(torch::kInt32).device(batchHdl.device()));

    FVDB_DISPATCH_GRID_TYPES(batchHdl, [&]() {
        auto batchAcc = gridBatchAccessor<torch::kCUDA, GridType>(batchHdl);
        populate_halo_index_buffer<GridType><<<num_leaf_nodes, 1024>>>(
            batchAcc,
            brick_usage_flags.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
            brick_offsets.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            halo_index_buffer.data_ptr<int>(),
            output_index_buffer.data_ptr<int>(),
            benchmark
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });

    return {brick_offsets, halo_index_buffer, output_index_buffer};
}


template <>
std::vector<torch::Tensor> dispatchBrickHaloBuffer<torch::kCPU>(const GridBatchImpl& batchHdl, bool benchmark) {
    TORCH_CHECK(false, "CPU not supported");
}


} // namespace ops
} // namespace detail
} // namespace fvdb
