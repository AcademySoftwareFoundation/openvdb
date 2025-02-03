// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include "utils/Tensor.h"
#include "utils/TileBitMask.h"
#include <detail/ops/gsplat/GaussianSplatSparse.h>

#include <gtest/gtest.h>
#include <thrust/sort.h>

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
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
expectedTensors(torch::Tensor const &uvsCPU, std::int64_t tileSize, std::int64_t numTilesPerAxis,
                std::int64_t numPixelsPerAxis) {
    std::int64_t                           numTiles = numTilesPerAxis * numTilesPerAxis;
    std::vector<std::int64_t>              expectedPixelMap{};
    std::vector<std::vector<std::int64_t>> tilePixelIds(numTiles);
    std::vector<std::vector<std::int64_t>> tilePixelKeys(numTiles);
    std::vector<std::uint64_t>             tileBitMasks;
    std::set<std::uint64_t>                activeTiles;
    auto                                   uvsCPUAccessor = uvsCPU.accessor<CoordType, 2>();
    // Bucket active pixel ordinals into per-tile buckets
    for (int pixelOrdinal = 0; pixelOrdinal < uvsCPU.size(0); pixelOrdinal += 1) {
        auto pixelU = uvsCPUAccessor[pixelOrdinal][0];
        auto pixelV = uvsCPUAccessor[pixelOrdinal][1];

        auto tileU = pixelU / tileSize;
        auto tileV = pixelV / tileSize;

        auto tileId  = tileV * numTilesPerAxis + tileU;
        auto pixelId = pixelV * numPixelsPerAxis + pixelU;
        activeTiles.insert(tileId);
        tilePixelIds[tileId].push_back(pixelOrdinal);
        tilePixelKeys[tileId].push_back(pixelId);
    }

    // Sort each bucket by the linearized pixel order (called pixelId above)
    for (int tileId = 0; tileId < tilePixelKeys.size(); tileId += 1) {
        std::sort(tilePixelIds[tileId].begin(), tilePixelIds[tileId].end());
        thrust::sort_by_key(tilePixelKeys[tileId].begin(), tilePixelKeys[tileId].end(),
                            tilePixelIds[tileId].begin());
    }
    // Concatenate all the sorted pixel ids into one big vector
    std::vector<std::int64_t> pixelsPerTile{};
    for (auto tileId: activeTiles) {
        pixelsPerTile.push_back(tilePixelIds[tileId].size());
        TileBitMask bitMask{ static_cast<std::size_t>(tileSize) };

        for (int i = 0; i < tilePixelKeys[tileId].size(); i++) {
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
    auto activeTilesT =
        torch::from_blob(activeTileIds.data(), { static_cast<std::int64_t>(activeTileIds.size()) },
                         tensorOpts<std::int64_t>(uvsCPU.device()));

    std::int64_t numWordsPerTile = TileBitMask<std::uint64_t>::numWordsPerTile(tileSize);
    auto         tileBitMasksT   = torch::from_blob(
        tileBitMasks.data(),
        { static_cast<std::int64_t>(tileBitMasks.size() / numWordsPerTile), numWordsPerTile },
        tensorOpts<std::uint64_t>(uvsCPU.device()));

    auto pixelsPerTileT =
        torch::from_blob(pixelsPerTile.data(), { static_cast<std::int64_t>(pixelsPerTile.size()) },
                         tensorOpts<std::int64_t>(uvsCPU.device()));

    auto pixelMapT = torch::from_blob(expectedPixelMap.data(),
                                      { static_cast<std::int64_t>(expectedPixelMap.size()) },
                                      tensorOpts<std::int64_t>(uvsCPU.device()));

    return { activeTilesT.clone(), tileBitMasksT.clone(), pixelsPerTileT.clone(),
             pixelMapT.clone() };
}

// Test fixtures
template <typename CoordType> struct ComputeSparseInfo : public ::testing::Test {
    ComputeSparseInfo()
        : mTileSize(0), mNumTilesPerAxis(0), mNumTiles(0), mNumPixelsPerAxis(0),
          mNumActivePixels(0), mNumWordsPerTile(0) {
        // seed torch random number generator so we get consistent results
        //
        torch::manual_seed(0);
    }

    void
    setTiling(std::int64_t tileSize, std::int64_t numTilesPerAxis) {
        mTileSize         = tileSize;
        mNumTilesPerAxis  = numTilesPerAxis;
        mNumTiles         = numTilesPerAxis * numTilesPerAxis;
        mNumPixelsPerAxis = tileSize * numTilesPerAxis;
        mNumWordsPerTile  = TileBitMask<std::uint64_t>::numWordsPerTile(tileSize);
    }

    // Helper function to generate UVs for a grid of tiles, with everyNth pixel in each row active
    // Returns a torch CPU tensor of shape {numActivePixels, 2}
    torch::Tensor
    makeStridedUVs(std::int64_t const tileSize, std::int64_t const numTilesPerAxis,
                   std::int64_t const pixelStride) {
        setTiling(tileSize, numTilesPerAxis);
        std::int64_t const numActivePixelsU = mNumPixelsPerAxis / pixelStride;
        mNumActivePixels                    = numActivePixelsU * mNumPixelsPerAxis;

        auto opts = tensorOpts<CoordType>(torch::kCPU);

        torch::Tensor uvsCPU         = torch::empty({ mNumActivePixels, 2 }, opts);
        auto          uvsCPUAccessor = uvsCPU.accessor<CoordType, 2>();
        for (std::int64_t i = 0; i < numActivePixelsU; ++i) {
            for (std::int64_t j = 0; j < mNumPixelsPerAxis; ++j) {
                uvsCPUAccessor[j * numActivePixelsU + i][0] = i * pixelStride;
                uvsCPUAccessor[j * numActivePixelsU + i][1] = j;
            }
        }
        return uvsCPU.clone();
    }

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
    auto const emptyPixels = fvdb::JaggedTensor{ torch::empty({ 0, 0 }, tensorOpts<TypeParam>()) };
    EXPECT_THROW(fvdb::detail::ops::computeSparseInfo(16, 4, 4, emptyPixels), c10::TypeError);
}

TEST(BadTypeTest, CPUThrows) {
    auto const emptyPixels =
        fvdb::JaggedTensor{ torch::empty({ 0, 0 }, tensorOpts<std::int32_t>(torch::kCPU)) };
    EXPECT_THROW(fvdb::detail::ops::computeSparseInfo(16, 4, 4, emptyPixels),
                 c10::NotImplementedError);
}

TYPED_TEST(ComputeSparseInfo, Empty) {
    this->setTiling(16, 4);
    auto opts = tensorOpts<TypeParam>();

    auto const emptyPixels = fvdb::JaggedTensor(torch::empty({ 0, 0 }, opts));
    auto [activeTiles, tileBitMask, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, 4, 4, emptyPixels);

    EXPECT_TRUE(
        torch::equal(activeTiles, torch::empty({ 0 }, tensorOpts<std::int32_t>(torch::kCUDA))));
    EXPECT_TRUE(torch::equal(tileBitMask, torch::empty({ 0, this->mNumWordsPerTile },
                                                       tensorOpts<std::uint64_t>(torch::kCUDA))));
    EXPECT_TRUE(torch::equal(tilePixelOffsets,
                             torch::zeros({ 1 }, tensorOpts<std::int64_t>(torch::kCUDA))));
    EXPECT_TRUE(
        torch::equal(pixelMap, torch::empty({ 0 }, tensorOpts<std::int64_t>(torch::kCUDA))));
}

TYPED_TEST(ComputeSparseInfo, SinglePixelInTopLeftTile) {
    this->setTiling(16, 4);
    auto const singlePixelCPU = torch::ones({ 1, 2 }, tensorOpts<TypeParam>(torch::kCPU));

    auto const singlePixel = singlePixelCPU.to(torch::kCUDA);

    auto [activeTiles, tileBitMasks, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, this->mNumTilesPerAxis,
                                             this->mNumTilesPerAxis, singlePixel);

    auto [expectedActiveTiles, expectedBitMasks, expectedPixelOffsets, expectedPixelMap] =
        expectedTensors<TypeParam>(singlePixelCPU, this->mTileSize, this->mNumTilesPerAxis,
                                   this->mNumPixelsPerAxis);

    EXPECT_TRUE(torch::equal(activeTiles, expectedActiveTiles.to(activeTiles.device())));
    EXPECT_TRUE(torch::equal(tileBitMasks, expectedBitMasks.to(tileBitMasks.device())));
    EXPECT_TRUE(torch::equal(tilePixelOffsets, expectedPixelOffsets.to(tilePixelOffsets.device())));
    EXPECT_TRUE(torch::equal(pixelMap, expectedPixelMap.to(pixelMap.device())));
}

TYPED_TEST(ComputeSparseInfo, Every4thPixel) {
    // x = active pixel, o = inactive pixel

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

    torch::Tensor uvsCPU = this->makeStridedUVs(16, 4, 4);
    auto          uvs    = uvsCPU.to(torch::kCUDA);

    auto [activeTiles, tileBitMasks, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, this->mNumTilesPerAxis,
                                             this->mNumTilesPerAxis, uvs);

    auto [expectedActiveTiles, expectedBitMasks, expectedPixelOffsets, expectedPixelMap] =
        expectedTensors<TypeParam>(uvsCPU, this->mTileSize, this->mNumTilesPerAxis,
                                   this->mNumPixelsPerAxis);

    EXPECT_TRUE(torch::equal(activeTiles, expectedActiveTiles.to(activeTiles.device())));
    EXPECT_TRUE(torch::equal(tileBitMasks, expectedBitMasks.to(tileBitMasks.device())));
    EXPECT_TRUE(torch::equal(tilePixelOffsets, expectedPixelOffsets.to(tilePixelOffsets.device())));
    EXPECT_TRUE(torch::equal(pixelMap, expectedPixelMap.to(pixelMap.device())));
}

TYPED_TEST(ComputeSparseInfo, EveryOtherColumnOfTiles) {
    // x = active pixel, o = inactive pixel
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

    torch::Tensor uvsCPU = this->makeStridedUVs(2, 4, 4);
    auto          uvs    = uvsCPU.to(torch::kCUDA);

    auto [activeTiles, tileBitMasks, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, this->mNumTilesPerAxis,
                                             this->mNumTilesPerAxis, uvs);

    auto [expectedActiveTiles, expectedBitMasks, expectedPixelOffsets, expectedPixelMap] =
        expectedTensors<TypeParam>(uvsCPU, this->mTileSize, this->mNumTilesPerAxis,
                                   this->mNumPixelsPerAxis);

    EXPECT_TRUE(torch::equal(activeTiles, expectedActiveTiles.to(activeTiles.device())));
    EXPECT_TRUE(torch::equal(tileBitMasks, expectedBitMasks.to(tileBitMasks.device())));
    EXPECT_TRUE(torch::equal(tilePixelOffsets, expectedPixelOffsets.to(tilePixelOffsets.device())));
    EXPECT_TRUE(torch::equal(pixelMap, expectedPixelMap.to(pixelMap.device())));
}

TYPED_TEST(ComputeSparseInfo, EveryOtherColumnOfTiles_Permuted) {
    // Start with this, but randomly permute the order of the input

    // x = active pixel, o = inactive pixel
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

    torch::Tensor uvsCPU = this->makeStridedUVs(2, 4, 4);
    torch::Tensor randIndices =
        torch::randperm(uvsCPU.size(0), tensorOpts<std::int64_t>(torch::kCPU));
    uvsCPU   = uvsCPU.index_select(0, randIndices);
    auto uvs = uvsCPU.to(torch::kCUDA);

    auto [activeTiles, tileBitMasks, tilePixelOffsets, pixelMap] =
        fvdb::detail::ops::computeSparseInfo(this->mTileSize, this->mNumTilesPerAxis,
                                             this->mNumTilesPerAxis, uvs);

    auto [expectedActiveTiles, expectedBitMasks, expectedPixelOffsets, expectedPixelMap] =
        expectedTensors<TypeParam>(uvsCPU, this->mTileSize, this->mNumTilesPerAxis,
                                   this->mNumPixelsPerAxis);

    EXPECT_TRUE(torch::equal(activeTiles, expectedActiveTiles.to(activeTiles.device())));
    EXPECT_TRUE(torch::equal(tileBitMasks, expectedBitMasks.to(tileBitMasks.device())));
    EXPECT_TRUE(torch::equal(tilePixelOffsets, expectedPixelOffsets.to(tilePixelOffsets.device())));
    EXPECT_TRUE(torch::equal(pixelMap, expectedPixelMap.to(pixelMap.device())));
}
