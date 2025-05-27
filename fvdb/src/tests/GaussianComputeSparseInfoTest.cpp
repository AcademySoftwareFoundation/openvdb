// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"
#include "utils/TestUtilities.h"
#include "utils/TileBitMask.h"

#include <detail/ops/gsplat/GaussianSplatSparse.h>

#include <thrust/sort.h>

#include <gtest/gtest.h>

#include <cstdint>

using fvdb::test::tensorOpts;
using fvdb::test::TileBitMask;

// Helper function to calculate the expected tensors for computeSparseInfo:
// 1. activeTiles: A 1D tensor of tile ids that have at least one active pixel
// 2. tileBitMasks: A 2D tensor of tile bitmasks of shape {numActiveTiles, numWordsPerTile}
// 3. pixelsPerTile: A 1D tensor of the inclusive cumulative sum of the number of active pixels in
// each tile
// 4. pixelMap: A 1D tensor of pixel ordinals sorted by tile and pixel order within the tile
// given the tensor of UV pixel coordinates and the tile and image sizes
template <typename CoordType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
expectedTensors(fvdb::JaggedTensor const &uvsCPU,
                std::int64_t tileSize,
                std::int64_t numTilesPerAxis,
                std::int64_t numPixelsPerAxis) {
    std::int64_t numImages = uvsCPU.num_outer_lists();
    std::int64_t numTiles  = numTilesPerAxis * numTilesPerAxis;
    std::int64_t numPixels = numPixelsPerAxis * numPixelsPerAxis;
    std::vector<std::int64_t> expectedPixelMap{};
    std::vector<std::vector<std::int64_t>> tilePixelIds(numTiles * numImages);
    std::vector<std::vector<std::int64_t>> tilePixelKeys(numTiles * numImages);
    std::vector<std::uint64_t> tileBitMasks;
    std::set<std::uint64_t> activeTiles;

    torch::Tensor activeTileMask = torch::zeros({numImages, numTilesPerAxis, numTilesPerAxis},
                                                tensorOpts<bool>(uvsCPU.device()));
    auto activeTileMaskAccessor  = activeTileMask.accessor<bool, 3>();

    auto uvsCPUAccessor = uvsCPU.jdata().accessor<CoordType, 2>();

    // Bucket active pixel ordinals into per-tile buckets
    for (int image = 0; image < numImages; image++) {
        for (int pixelOrdinal = uvsCPU.joffsets()[image].item<std::int64_t>();
             pixelOrdinal < uvsCPU.joffsets()[image + 1].item<std::int64_t>();
             pixelOrdinal++) {
            auto pixelU = uvsCPUAccessor[pixelOrdinal][0];
            auto pixelV = uvsCPUAccessor[pixelOrdinal][1];

            auto tileU = pixelU / tileSize;
            auto tileV = pixelV / tileSize;

            auto tileId  = image * numTiles + tileV * numTilesPerAxis + tileU;
            auto pixelId = image * numPixels + pixelV * numPixelsPerAxis + pixelU;

            activeTiles.insert(tileId);
            activeTileMaskAccessor[image][tileV][tileU] = true;
            tilePixelIds[tileId].push_back(pixelOrdinal);
            tilePixelKeys[tileId].push_back(pixelId);
        }
    }

    // Sort each bucket by the linearized pixel order (called pixelId above)
    for (std::size_t tileId = 0; tileId < tilePixelKeys.size(); tileId += 1) {
        std::sort(tilePixelIds[tileId].begin(), tilePixelIds[tileId].end());
        thrust::sort_by_key(tilePixelKeys[tileId].begin(),
                            tilePixelKeys[tileId].end(),
                            tilePixelIds[tileId].begin());
    }

    // Concatenate all the sorted pixel ids into one big vector
    std::vector<std::int64_t> pixelsPerTile{};
    for (auto tileId: activeTiles) {
        pixelsPerTile.push_back(tilePixelIds[tileId].size());
        TileBitMask bitMask{static_cast<std::size_t>(tileSize)};

        for (std::size_t i = 0; i < tilePixelKeys[tileId].size(); i++) {
            expectedPixelMap.push_back(tilePixelIds[tileId][i]);
            bitMask.setBit(tilePixelKeys[tileId][i], numTilesPerAxis);
        }
        if (tilePixelKeys[tileId].size() > 0) {
            tileBitMasks.insert(tileBitMasks.end(), bitMask.begin(), bitMask.end());
        }
    }

    // compute cumulative sum of pixels per tile
    std::inclusive_scan(pixelsPerTile.begin(), pixelsPerTile.end(), pixelsPerTile.begin());

    // activeTiles is a set, so we need to convert it to a vector in order to convert to tensor
    auto activeTileIds = std::vector<std::int64_t>(activeTiles.begin(), activeTiles.end());

    // Construct and return output torch tensors
    auto activeTilesT = torch::from_blob(activeTileIds.data(),
                                         {static_cast<std::int64_t>(activeTileIds.size())},
                                         tensorOpts<std::int64_t>(uvsCPU.device()));

    std::int64_t numWordsPerTile = TileBitMask<std::uint64_t>::numWordsPerTile(tileSize);
    auto tileBitMasksT           = torch::from_blob(
        tileBitMasks.data(),
        {static_cast<std::int64_t>(tileBitMasks.size() / numWordsPerTile), numWordsPerTile},
        tensorOpts<std::uint64_t>(uvsCPU.device()));

    auto pixelsPerTileT = torch::from_blob(pixelsPerTile.data(),
                                           {static_cast<std::int64_t>(pixelsPerTile.size())},
                                           tensorOpts<std::int64_t>(uvsCPU.device()));

    auto pixelMapT = torch::from_blob(expectedPixelMap.data(),
                                      {static_cast<std::int64_t>(expectedPixelMap.size())},
                                      tensorOpts<std::int64_t>(uvsCPU.device()));

    return {activeTilesT.clone(),
            activeTileMask.to(activeTilesT.device()),
            tileBitMasksT.clone(),
            pixelsPerTileT.clone(),
            pixelMapT.clone()};
}

// Randomize the order of the UVs within each image in a JaggedTensor
fvdb::JaggedTensor
randomPermuteUVs(fvdb::JaggedTensor const &uvs) {
    auto tensors    = uvs.unbind1();
    auto outTensors = std::vector<torch::Tensor>{};
    outTensors.reserve(tensors.size());

    for (auto &tensor: tensors) {
        auto randIndices = torch::randperm(tensor.size(0), tensorOpts<std::int64_t>(torch::kCPU));
        outTensors.emplace_back(tensor.index_select(0, randIndices));
    }

    return fvdb::JaggedTensor(outTensors).to(uvs.device());
}

// Test fixtures
template <typename CoordType> struct ComputeSparseInfo : public ::testing::Test {
    ComputeSparseInfo()
        : mNumImages(0), mTileSize(0), mNumTilesPerAxis(0), mNumTiles(0), mNumPixelsPerAxis(0),
          mNumActivePixels(0), mNumWordsPerTile(0) {
        // seed torch random number generator so we get consistent results
        //
        torch::manual_seed(0);
    }

    void
    setTiling(std::int64_t numImages, std::int64_t tileSize, std::int64_t numTilesPerAxis) {
        mNumImages        = numImages;
        mTileSize         = tileSize;
        mNumTilesPerAxis  = numTilesPerAxis;
        mNumTiles         = numTilesPerAxis * numTilesPerAxis;
        mNumPixelsPerAxis = tileSize * numTilesPerAxis;
        mNumWordsPerTile  = TileBitMask<std::uint64_t>::numWordsPerTile(tileSize);
    }

    // Helper function to generate UVs for a grid of tiles, with everyNth pixel in each row
    // active. Returns a torch CPU tensor of shape {numActivePixels, 2}
    fvdb::JaggedTensor
    makeStridedUVs(std::int64_t numImages,
                   std::int64_t const tileSize,
                   std::int64_t const numTilesPerAxis,
                   std::int64_t const pixelStride) {
        setTiling(numImages, tileSize, numTilesPerAxis);
        std::int64_t const numPixels = mNumTiles * mTileSize * mTileSize;

        auto opts = tensorOpts<CoordType>(torch::kCPU);

        auto numActivePixels = (numPixels + pixelStride - 1) / pixelStride;

        torch::Tensor uvsCPU = torch::empty({numActivePixels, 2}, opts);
        auto uvsCPUAccessor  = uvsCPU.accessor<CoordType, 2>();

        for (std::int64_t i = 0; i < numPixels; i += pixelStride) {
            uvsCPUAccessor[i / pixelStride][0] = i % mNumPixelsPerAxis;
            uvsCPUAccessor[i / pixelStride][1] = i / mNumPixelsPerAxis;
        }

        std::vector<torch::Tensor> uvs(mNumImages, uvsCPU.clone());
        return fvdb::JaggedTensor(uvs);
    }

    void
    testStridedUVs(std::int64_t numImages,
                   std::int64_t const tileSize,
                   std::int64_t const numTilesPerAxis,
                   std::int64_t const pixelStride,
                   bool randomPermute = false) {
        auto uvsCPU = [&]() {
            if (randomPermute) {
                return randomPermuteUVs(
                    this->makeStridedUVs(numImages, tileSize, numTilesPerAxis, pixelStride));
            } else {
                return this->makeStridedUVs(numImages, tileSize, numTilesPerAxis, pixelStride);
            }
        }();
        auto uvs = uvsCPU.to(torch::kCUDA);

        auto [activeTiles, activeTileMask, tileBitMasks, tilePixelOffsets, pixelMap] =
            fvdb::detail::ops::computeSparseInfo(
                this->mTileSize, this->mNumTilesPerAxis, this->mNumTilesPerAxis, uvs);

        auto [expectedActiveTiles,
              expectedActiveTileMask,
              expectedBitMasks,
              expectedPixelOffsets,
              expectedPixelMap] =
            expectedTensors<CoordType>(
                uvsCPU, this->mTileSize, this->mNumTilesPerAxis, this->mNumPixelsPerAxis);

        EXPECT_TRUE(torch::equal(activeTiles, expectedActiveTiles.to(activeTiles.device())));
        EXPECT_TRUE(
            torch::equal(activeTileMask, expectedActiveTileMask.to(activeTileMask.device())));
        EXPECT_TRUE(torch::equal(tileBitMasks, expectedBitMasks.to(tileBitMasks.device())));
        EXPECT_TRUE(
            torch::equal(tilePixelOffsets, expectedPixelOffsets.to(tilePixelOffsets.device())));
        EXPECT_TRUE(torch::equal(pixelMap, expectedPixelMap.to(pixelMap.device())));
    }

    std::int64_t mNumImages;
    std::int64_t mTileSize;
    std::int64_t mNumTilesPerAxis;
    std::int64_t mNumTiles;
    std::int64_t mNumPixelsPerAxis;
    std::int64_t mNumActivePixels;
    std::int64_t mNumWordsPerTile;
};
template <typename CoordType> struct BadTypeTest : public ::testing::Test {};

// type lists
using CoordTypes    = ::testing::Types<std::int32_t, std::int64_t>;
using BadCoordTypes = ::testing::Types<std::int16_t, std::uint32_t, std::uint64_t>;

TYPED_TEST_SUITE(ComputeSparseInfo, CoordTypes);
TYPED_TEST_SUITE(BadTypeTest, BadCoordTypes);

TYPED_TEST(BadTypeTest, GPUThrows) {
    auto const emptyPixels = fvdb::JaggedTensor{torch::empty({0, 0}, tensorOpts<TypeParam>())};
    EXPECT_THROW(fvdb::detail::ops::computeSparseInfo(16, 4, 4, emptyPixels), c10::TypeError);
}

TEST(BadTypeTest, CPUThrows) {
    auto const emptyPixels =
        fvdb::JaggedTensor{torch::empty({0, 0}, tensorOpts<std::int32_t>(torch::kCPU))};
    EXPECT_THROW(fvdb::detail::ops::computeSparseInfo(16, 4, 4, emptyPixels),
                 c10::NotImplementedError);
}

TYPED_TEST(ComputeSparseInfo, Empty) {
    this->setTiling(1, 16, 4);
    auto opts = tensorOpts<TypeParam>();

    auto const emptyPixels = fvdb::JaggedTensor(torch::empty({0, 0}, opts));
    auto [activeTiles, activeTileMask, tileBitMask, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, 4, 4, emptyPixels);

    EXPECT_TRUE(
        torch::equal(activeTiles, torch::empty({0}, tensorOpts<std::int32_t>(torch::kCUDA))));
    EXPECT_TRUE(
        torch::equal(activeTileMask, torch::zeros({1, 4, 4}, tensorOpts<bool>(torch::kCUDA))));
    EXPECT_TRUE(torch::equal(
        tileBitMask,
        torch::empty({0, this->mNumWordsPerTile}, tensorOpts<std::uint64_t>(torch::kCUDA))));
    EXPECT_TRUE(
        torch::equal(tilePixelOffsets, torch::zeros({1}, tensorOpts<std::int64_t>(torch::kCUDA))));
    EXPECT_TRUE(torch::equal(pixelMap, torch::empty({0}, tensorOpts<std::int64_t>(torch::kCUDA))));
}

// x = active pixel, o = inactive pixel
static const std::vector<std::tuple<std::string, std::array<std::int64_t, 5>>> configs{
    {"Single Pixel top left tile", {1, 16, 4, 4097, 0}},
    {"Single Pixel top left tile, 10 images", {10, 16, 4, 4097, 0}},

    //   u 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 || 1 1 1 1 ... || 2 2 2 2 ... || 3 3 3 3 ...
    // v   0 1 2 3 4 5 6 7 8 9 a b c d e f || 0 1 2 3 ... || 0 1 2 3 ... || 0 1 2 3 ...
    // ==================================  || ==========  || ==========  || ==========
    // 00: x o o o x o o o x o o o x o o o || x o o o ... || x o o o ... || x o o o ...
    // 01: x o o o x o o o x o o o x o o o || x o o o ... || x o o o ... || x o o o ...
    // 02: x o o o x o o o x o o o x o o o || x o o o ... || x o o o ... || x o o o ...
    // 03: x o o o x o o o x o o o x o o o || x o o o ... || x o o o ... || x o o o ...
    //     ...                             || ...         || ...         || ...
    // 0f: x o o o x o o o x o o o x o o o || x o o o ... || x o o o ... || x o o o ...
    // ----------------------------------  || ...
    // ... 3 more rows of tiles vertically ...

    {"Every 4th pixel", {1, 16, 4, 4, 0}},
    {"Every 4th pixel, 10 images", {10, 16, 4, 4, 0}},

    // 0 1 | 2 3 | 4 5 | 6 7   <-u/v
    // ========================    |
    // x o | o o | x o | o o || 0  v
    // x o | o o | x o | o o || 1
    // ----------------------||
    // x o | o o | x o | o o || 2
    // x o | o o | x o | o o || 3
    // ----------------------||
    // x o | o o | x o | o o || 4
    // x o | o o | x o | o o || 5
    // ----------------------||
    // x o | o o | x o | o o || 6
    // x o | o o | x o | o o || 7

    {"Every other column of tiles", {1, 2, 4, 4, 0}},
    {"Every other column of tiles, 10 images", {10, 2, 4, 4, 0}},

    // 0 1 2 3 | 4 5 6 7 | 8 9 a b | c d e f   <-u/v
    // ========================================
    // x o o o | o o o o | o o o o | o o o o || 0
    // o x o o | o o o o | o o o o | o o o o || 1
    // o o x o | o o o o | o o o o | o o o o || 2
    // o o o x | o o o o | o o o o | o o o o || 3
    // ----------------------------------------
    // o o o o | x o o o | o o o o | o o o o || 4
    // o o o o | o x o o | o o o o | o o o o || 5
    // o o o o | o o x o | o o o o | o o o o || 6
    // o o o o | o o o x | o o o o | o o o o || 7
    // ----------------------------------------
    // o o o o | o o o o | x o o o | o o o o || 8
    // o o o o | o o o o | o x o o | o o o o || 9
    // o o o o | o o o o | o o x o | o o o o || a
    // o o o o | o o o o | o o o x | o o o o || b
    // ----------------------------------------
    // o o o o | o o o o | o o o o | x o o o || c
    // o o o o | o o o o | o o o o | o x o o || d
    // o o o o | o o o o | o o o o | o o x o || e
    // o o o o | o o o o | o o o o | o o o x || f

    {"Prime stride", {1, 4, 4, 17, 0}},
    {"Prime stride, 10 images", {10, 4, 4, 17, 0}},
    {"Single Pixel top left tile, permuted", {1, 16, 4, 4097, 1}},
    {"Single Pixel top left tile, 10 images, permuted", {10, 16, 4, 4097, 1}},
    {"Every 4th pixel, permuted", {1, 16, 4, 4, 1}},
    {"Every 4th pixel, 10 images, permuted", {10, 16, 4, 4, 1}},
    {"Every other column of tiles, permuted", {1, 2, 4, 4, 1}},
    {"Every other column of tiles, 10 images, permuted", {10, 2, 4, 4, 1}},
    {"Prime stride, permuted", {1, 4, 4, 17, 1}},
    {"Prime stride, 10 images, permuted", {10, 4, 4, 17, 1}},
};

TYPED_TEST(ComputeSparseInfo, StridedUVs) {
    for (auto const &[name, config]: configs) {
        fvdb::test::printSubtestPrefix(name);
        this->testStridedUVs(config[0], config[1], config[2], config[3], config[4]);
        if (not testing::Test::HasFailure()) {
            fvdb::test::printGreenOK();
        }
    }
}
