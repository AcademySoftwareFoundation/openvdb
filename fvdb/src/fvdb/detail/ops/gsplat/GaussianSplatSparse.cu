// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/Ops.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>

#include <ATen/Dispatch_v2.h>

#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <thrust/sort.h>

namespace fvdb::detail::ops {

#define CUB_WRAPPER(func, ...)                                                    \
    do {                                                                          \
        size_t temp_storage_bytes = 0;                                            \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                           \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();      \
        auto temp_storage       = caching_allocator.allocate(temp_storage_bytes); \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);                \
    } while (false)

// Definitions:
// -----------------
//
// Let:
//     C denote the number of views to render in a minibatch
//     W the pixel width (columns) of the output image
//     H the pixel height (rows) of the output image
//     T the width (in pixels of each tile). i.e. tiles are TxT pixels
//     TH = ceil(H / T) denote the number of tiles along the height dimension in the output image
//     TW = ceil(W / T) denote the number of tiles along the width dimension in the output image
//
// Pixels are defined by three coordinates (c, i, j)
// where:
//    c \in [0, ..., C-1] denotes which image a pixel belongs to in the batch
//    i \in [0, ..., H-1] denotes the row of the pixel (starting from the top of the image)
//    j \in [0, ..., W-1] denotes the column of the pixel (starting from the left of the image)
//
// Tiles are patches of TxT pixels and also defined by three coordinates (c, ti, tj)
// Each tile has a tile_id which is the lexographical order of the tile. i.e.
//     tile_id = c * (TH * TW) + ti * TW + tj
// where:
//     c \in [0, ..., C-1]
//     ti = floor(i / T)
//     tj = floor(j / T)
//
//
// Sparse Gaussian Splatting:
// -------------------------
//
// We receive an additional JaggedTensor to the original Gaussian Splatting algorithm which
// specifies a set of pixel coordinates to render in each image. i.e.
//
//      pixels_to_render = [uv_0, ..., uv_{C-1}]
//
//  where uv_n is a tensor of shape [P_n, 2] of pixel coordinates in the n^th image in the batch.
//  If a pixel (c, i, j) is in in pixels_to_render we call it *active*, otherwise it is *inactive*.
//  NOTE: we're going to assume for now that pixels_to_render does not contain duplicates.
//
// Let:
//     AP denote the number of active pixels.
//     AT denote the number of tiles containing active pixels.
//
// Instead of rendering full images, Sparse Gaussian Splatting returns a JaggedTensor of rendered
// pixels matching the input. i.e.
//
//     output = [pixels_0, ..., pixels_{C-1}]
//
// where pixels_n is a tensor of shape [P_n, D] rendered pixels (D is the channel dimension)
//
// To do this, we only render tiles which pixels in pixels_to_render and only write out active
// pixels within those tiles. Thus, we need to compute:
//     1. active_tiles: An integer tensor with shape [AT] indicating the tile_ids
//        corresponding to tiles which contain active pixels.
//     2. tile_pixel_mask: An int64 tensor of bitmasks with shape [AT, words_per_tile] where
//        words_per_tile is the number of int64_t words needed to make a bitmask for a PxP tile.
//        We asume bits are in raster order (top left to bottom right)
//     3. tile_pixel_cumsum: An int64 tensor with shape [AT] encoding the cumuluative sum of
//        active pixels in each active tile. i.e. tile_pixel_cumsum[i-1] is the number of active
//        pixels in all tiles before the i^th active tile
//     4. pixel_map: An integer tensor with shape [AP] specifying the order that pixels should be
//        written within each tile.
//        i.e. Suppose we're rendering the k^th active pixel in tile_id = active_tiles[t],
//             we write its rendered value to index pixel_map[tile_pixel_cumsum[tile_id-1] + k]
//             in the output
//
// To compute these quantities, we
//     1. Compute the tile_id for each pixel in pixels_to_render
//     2. Compute the order of each pixel within a tile
//     2. Sort the pixels by tile_id and pixel order, storing the inverse mapping from
//        sorted pixels to the original ordering
//     3. Run length encode the sorted tile_ids to count the number of pixels per tile and
//        compute the set of unique tile_ids
//     4. Cumsum the number of pixels in each tile
//     5. Compute the bitmask for each unique tile_id
//

// Compute tile mask and tile ids for each pixel coordinate
template <typename CoordType>
__global__ void __launch_bounds__(256)
computeTileMask(const fvdb::JaggedRAcc32<CoordType, 2> pixelCoords,
                const int32_t tileSideLength,
                const int32_t numTilesW,
                const int32_t numTilesH,
                fvdb::TorchRAcc32<bool, 1> outTileMask,
                fvdb::TorchRAcc32<int64_t, 1> outTileIds) {
    for (auto pixelId = blockIdx.x * blockDim.x + threadIdx.x; pixelId < pixelCoords.elementCount();
         pixelId += blockDim.x * gridDim.x) {
        auto const batchId = pixelCoords.batchIdx(pixelId);

        // Can't guarantee contiguity so can't do vectorized loads in general here
        const CoordType pixelCoordU = pixelCoords.data()[pixelId][0];
        const CoordType pixelCoordV = pixelCoords.data()[pixelId][1];

        const int32_t tileCoordU = pixelCoordU / tileSideLength;
        const int32_t tileCoordV = pixelCoordV / tileSideLength;

        // Note this assumes all images are the same size
        const int64_t tileId =
            tileCoordU + tileCoordV * numTilesW + batchId * numTilesW * numTilesH;

        // Assume contiguous because we're allocating this tensor
        outTileMask[tileId] = true;

        // Note we could fit this into a 32-bit int if we wanted to save memory
        const int32_t pixelIdInTile =
            (pixelCoordU % tileSideLength) + (pixelCoordV % tileSideLength) * tileSideLength;
        outTileIds[pixelId] = tileId << 32 | pixelIdInTile;
    }
}

// This kernel computes a bitmask for each unique tile in an image.
// Each bit in the bitmask represents whether a pixel within the tile is set or not.
// The bitmask is stored in shared memory and then written to the output array.

// NOTE (mharris): it looks like this is doing one tile per thread, which means
// threads in a block are not actually sharing anything in shared memory. I guess
// shared is just scratchpad here?
template <typename CoordType>
__global__ void __launch_bounds__(256)
computePerTileBitMask(
    const int64_t numUniqueTiles,
    const int32_t numWordsPerTile,
    const int32_t numTilesW,
    const int32_t numTilesH,
    const int32_t tileSideLength,
    const fvdb::TorchRAcc32<int32_t, 1> uniqueTileIds,    // [numUniqueTiles]
    const fvdb::TorchRAcc32<int64_t, 1> cumPixelsPerTile, // [numUniqueTiles] (inclusive cumsum)
    const fvdb::TorchRAcc32<int64_t, 1> unsortPerPixelTileIds, // [numPixels]
    const fvdb::JaggedRAcc32<CoordType, 2> pixelCoords,
    fvdb::TorchRAcc32<uint64_t, 2> outBitmask) {
    // Contains threadsPerBlock * numWordsPerTile elements
    extern __shared__ uint64_t sharedMemBitmask[];

    for (int tileOrdinal = blockIdx.x * blockDim.x + threadIdx.x; tileOrdinal < numUniqueTiles;
         tileOrdinal += blockDim.x * gridDim.x) {
        auto tileId = uniqueTileIds[tileOrdinal];

        // Zero out bits for this tile
        for (int i = 0; i < numWordsPerTile; i++) {
            sharedMemBitmask[threadIdx.x * numWordsPerTile + i] = 0;
        }

        const int tileStart = tileOrdinal > 0 ? cumPixelsPerTile[tileOrdinal - 1] : 0;
        const int tileEnd   = cumPixelsPerTile[tileOrdinal];

        // tileId = batch * numTilesW * numTilesH + tileU * numTilesW + tileV
        const int32_t batchId         = tileId / (numTilesW * numTilesH);
        const int32_t tileU           = (tileId - batchId * numTilesW * numTilesH) % numTilesW;
        const int32_t tileV           = (tileId - batchId * numTilesW * numTilesH) / numTilesW;
        const int32_t tileStartPixelU = tileU * tileSideLength;
        const int32_t tileStartPixelV = tileV * tileSideLength;

        // Note, we could use a warp per tile and do more in parallel.
        for (int i = tileStart; i < tileEnd; ++i) {
            const int32_t pixelId  = unsortPerPixelTileIds[i];
            const CoordType pixelU = pixelCoords.data()[pixelId][0];
            const CoordType pixelV = pixelCoords.data()[pixelId][1];

            const int32_t pixelUInTile = pixelU - tileStartPixelU;
            const int32_t pixelVInTile = pixelV - tileStartPixelV;
            const int32_t bitId        = pixelUInTile + pixelVInTile * tileSideLength;
            const int32_t wordId       = bitId / 64;
            const int32_t bitInWord    = bitId % 64;

            sharedMemBitmask[threadIdx.x * numWordsPerTile + wordId] |= 1ull << bitInWord;
        }

        // Write out the bitmask
        for (int i = 0; i < numWordsPerTile; i++) {
            outBitmask[tileOrdinal][i] = sharedMemBitmask[threadIdx.x * numWordsPerTile + i];
        }
    }
}

// TODO these should be centralized somewhere
torch::Tensor
empty(at::IntArrayRef dims, c10::ScalarType dtype, torch::Device device) {
    return torch::empty(dims, torch::TensorOptions().device(device).dtype(dtype));
}

torch::Tensor
zeros(at::IntArrayRef dims, c10::ScalarType dtype, torch::Device device) {
    return torch::zeros(dims, torch::TensorOptions().device(device).dtype(dtype));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
computeSparseInfo(const int32_t tileSideLength,
                  const int32_t numTilesW,
                  const int32_t numTilesH,
                  const fvdb::JaggedTensor &pixelsToRender) {
    TORCH_CHECK_NOT_IMPLEMENTED(pixelsToRender.device().is_cuda(),
                                "computeSparseInfo only implemented on the device");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(pixelsToRender.jdata()));
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(pixelsToRender.device().index());

    TORCH_CHECK_TYPE(pixelsToRender.scalar_type() == torch::kInt32 ||
                         pixelsToRender.scalar_type() == torch::kInt64,
                     "pixelsToRender must be of type int32 or int64");

    auto const numImages                 = pixelsToRender.num_outer_lists();
    auto const numPixels                 = pixelsToRender.rsize(0);
    const int32_t numWordsPerTileBitmask = (tileSideLength * tileSideLength + 63) / 64;

    const torch::Device device = pixelsToRender.device();

    const torch::TensorOptions optionsInt64 =
        torch::TensorOptions().device(device).dtype(torch::kInt64);
    const torch::TensorOptions optionsInt32 =
        torch::TensorOptions().device(device).dtype(torch::kInt32);
    const torch::TensorOptions optionsBool =
        torch::TensorOptions().device(device).dtype(torch::kBool);

    if (numImages == 0 || numPixels == 0) {
        return {empty({0}, torch::kInt, device),
                zeros({numImages, numTilesW, numTilesH}, torch::kBool, device),
                empty({0, numWordsPerTileBitmask}, torch::kLong, device),
                zeros({1}, torch::kLong, device),
                empty({0}, torch::kLong, device)};
    }

    // Compute a boolean tile
    torch::Tensor tileMask        = torch::zeros({numTilesW * numTilesH * numImages}, optionsBool);
    torch::Tensor perPixelTileIds = torch::empty({numPixels}, optionsInt64);

    auto outMaskAccessor   = fvdb::tensorAccessor<torch::kCUDA, bool, 1>(tileMask);
    auto outTileIdAccessor = fvdb::tensorAccessor<torch::kCUDA, int64_t, 1>(perPixelTileIds);

    // TODO we do not output tileMask currently, but we should
    AT_DISPATCH_INDEX_TYPES(pixelsToRender.scalar_type(), "computeTileMask", [&]() {
        const int32_t NUM_THREADS = 256;
        const int32_t NUM_BLOCKS  = (numPixels + NUM_THREADS - 1) / NUM_THREADS;
        computeTileMask<index_t><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(
            pixelsToRender.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>(),
            tileSideLength,
            numTilesW,
            numTilesH,
            outMaskAccessor,
            outTileIdAccessor);
        C10_CUDA_KERNEL_LAUNCH_CHECK(); // TODO use our own error management
    });

    tileMask = tileMask.view({numImages, numTilesH, numTilesW});

    // Sort pixels by their tile id and store the inverse mapping from sorted to original
    // TODO can we reduce the data sizes here?
    torch::Tensor pixelIds              = torch::arange(0, numPixels, optionsInt64);
    torch::Tensor sortedPerPixelTileIds = torch::empty_like(perPixelTileIds);
    torch::Tensor unsortPerPixelTileIds = torch::empty_like(perPixelTileIds);

    // Sort by tile ID (out-of-place)
    CUB_WRAPPER(cub::DeviceRadixSort::SortPairs,
                perPixelTileIds.data_ptr<int64_t>(),
                sortedPerPixelTileIds.data_ptr<int64_t>(),
                pixelIds.data_ptr<int64_t>(),
                unsortPerPixelTileIds.data_ptr<int64_t>(),
                numPixels,
                0 /* begin_bit */,
                sizeof(int64_t) * 8 /* end_bit */,
                stream);

    // Make a transform_iterator to extract the tile id from the sorted pixel ids
    auto tileIdIter =
        thrust::make_transform_iterator(sortedPerPixelTileIds.data_ptr<int64_t>(),
                                        [] __host__ __device__(int64_t x) { return x >> 32; });

    // Run length encode to get the unique tile ids, and the number of pixels per tile
    torch::Tensor numPixelsPerTile = torch::empty(
        {std::min(numPixels, int64_t(numImages * numTilesW * numTilesH))}, optionsInt64);
    torch::Tensor uniqueTileIds = torch::empty({numPixels}, optionsInt32);
    torch::Tensor uniqueCounts  = torch::empty({1}, optionsInt32);
    CUB_WRAPPER(cub::DeviceRunLengthEncode::Encode,
                tileIdIter,
                uniqueTileIds.data_ptr<int32_t>(),
                numPixelsPerTile.data_ptr<int64_t>(),
                uniqueCounts.data_ptr<int32_t>(),
                numPixels,
                stream);
    cudaStreamSynchronize(stream);
    auto const numUniqueTiles = uniqueCounts.item<int32_t>();
    uniqueTileIds             = uniqueTileIds.index({at::indexing::Slice(0, numUniqueTiles)});
    numPixelsPerTile          = numPixelsPerTile.index({at::indexing::Slice(0, numUniqueTiles)});

    // Cumsum so we know where each tile starts in the sorted array
    // TODO: should we use CUB?
    torch::cumsum_out(numPixelsPerTile, numPixelsPerTile, 0);

    // Compute a bitmask for each tile indicating which pixels are active in that tile
    torch::Tensor tileBitMask =
        torch::zeros({numUniqueTiles, numWordsPerTileBitmask},
                     torch::TensorOptions().device(device).dtype(torch::kUInt64));

    // TODO: Check available shared memory and adjust block size accordingly
    AT_DISPATCH_INDEX_TYPES(pixelsToRender.scalar_type(), "computePerTileBitMask", [&]() {
        const uint32_t NUM_THREADS2 = 256;
        const uint32_t NUM_BLOCKS2  = (numUniqueTiles + NUM_THREADS2 - 1) / NUM_THREADS2;
        computePerTileBitMask<index_t><<<NUM_BLOCKS2,
                                         NUM_THREADS2,
                                         NUM_THREADS2 * numWordsPerTileBitmask * sizeof(uint64_t),
                                         stream>>>(
            numUniqueTiles,
            numWordsPerTileBitmask,
            numTilesW,
            numTilesH,
            tileSideLength,
            uniqueTileIds.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            numPixelsPerTile.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            unsortPerPixelTileIds.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            pixelsToRender.packed_accessor32<index_t, 2, torch::RestrictPtrTraits>(),
            tileBitMask.packed_accessor32<uint64_t, 2, torch::RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK(); // TODO use our own error management
    });

    //     1. active_tiles: An integer tensor with shape [AT] indicating the tile_ids
    //        corresponding to tiles which contain active pixels.
    //     2. tile_pixel_mask: An int64 tensor of bitmasks with shape [AT, words_per_tile] where
    //        words_per_tile is the number of int64_t words needed to make a bitmask for a PxP
    //        tile. We asume bits are in raster order (top left to bottom right)
    //     3. tile_pixel_cumsum: An int64 tensor with shape [AT] encoding the cumuluative sum of
    //        active pixels in each active tile. i.e. tile_pixel_cumsum[i-1] is the number of
    //        active pixels in all tiles before the i^th active tile
    //     4. pixel_map: An int64 tensor with shape [AP] specifying the order that pixels should
    //        be written within each tile.
    //        i.e. Suppose we're rendering the k^th active pixel in tile_id = active_tiles[t],
    //             we write its rendered value to index pixel_map[tile_pixel_cumsum[tile_id-1] +
    //             k] in the output
    return std::make_tuple(
        uniqueTileIds, tileMask, tileBitMask, numPixelsPerTile, unsortPerPixelTileIds);
}

} // namespace fvdb::detail::ops
