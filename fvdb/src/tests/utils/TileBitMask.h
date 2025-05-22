// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_TESTS_UTILS_TILEBITMASK_H
#define FVDB_TESTS_UTILS_TILEBITMASK_H

#include <climits>
#include <cstdint>
#include <vector>

namespace fvdb::test {

/// @brief A class to represent a bitmask for a square tile of a given size.
/// @tparam T The storage type for the bitmask (default: std::uint64_t).
template <typename T = std::uint64_t> class TileBitMask {
  public:
    /// @brief The number of bits in a word of type T
    static constexpr std::size_t wordSize = sizeof(T) * CHAR_BIT;

    /// @brief Get the number of words required to store the bitmask for a tile of the given size.
    /// @param tileSize The size of each dimension of the square tile.
    /// @return The number of words required to store the bitmask for the tile.
    static constexpr std::size_t
    numWordsPerTile(std::size_t tileSize) {
        return (tileSize * tileSize + wordSize - 1) / wordSize;
    }

    /// @brief Construct a TileBitMask for a square tile of the given size.
    TileBitMask(std::size_t tileSize)
        : mTileSize{tileSize}, mBitMask(numWordsPerTile(tileSize), 0) {}

    /// @brief Return whether the bit at the given index is set.
    /// @param bit The index of the bit to check.
    /// @return true if the bit is set, false otherwise.
    bool
    isBitSet(std::size_t bit) const {
        return mBitMask[word(bit)] & (T{1} << bitInWord(bit));
    }

    /// @brief Whether the bit at the given row and column is set.
    /// @param row The row of the bitmask to check.
    /// @param col The column of the bitmask to check.
    /// @return true if the bit is set, false otherwise.
    bool
    isBitSet(std::size_t row, std::size_t col) const {
        return isBitSet(row * mTileSize + col);
    }

    /// @brief Set the bit at the given index. (nonatomic)
    /// @param bit The index of the bit to set.
    void
    setBit(std::size_t bit) {
        // important that we use T{1} here to ensure the shifted value is as large as the shift
        mBitMask[word(bit)] |= (T{1} << bitInWord(bit));
    }

    /// @brief Set the bit of the tile containig the given linear pixel index within an image
    /// numTilesU tiles wide. (nonatomic)
    /// @param linearPixel The linear index of the pixel within the image.
    /// @param numTilesU The number of tiles in the U direction.
    void
    setBit(std::size_t linearPixel, std::size_t numTilesU) {
        auto pixelU     = linearPixel % (mTileSize * numTilesU);
        auto pixelV     = linearPixel / (mTileSize * numTilesU);
        auto tilePixelU = pixelU % mTileSize;
        auto tilePixelV = pixelV % mTileSize;
        auto tilePixel  = tilePixelV * mTileSize + tilePixelU;
        setBit(tilePixel);
    }

    /// @brief Get a vector of the UV coordinates of the set bits in the bitmask.
    /// @return A vector of pairs of UV coordinates of the set bits.
    std::vector<std::pair<std::size_t, std::size_t>>
    getUVs() const {
        std::vector<std::pair<std::size_t, std::size_t>> uvs;
        for (int i = 0; i < mTileSize * mTileSize; ++i) {
            if (isBitSet(i)) {
                uvs.emplace_back(
                    std::pair<std::size_t, std::size_t>({i % mTileSize, i / mTileSize}));
            }
        }
        return uvs;
    }

    /// @brief Const iterator to the beginning of the bitmask.
    auto
    begin() const {
        return mBitMask.begin();
    }

    /// @brief Const iterator to the end of the bitmask.
    auto
    end() const {
        return mBitMask.end();
    }

    /// @brief Get the number of words in the bitmask
    std::size_t
    size() const {
        return mBitMask.size();
    }

  private:
    /// @brief Get the word index for the given bit index.
    std::size_t
    word(std::size_t bit) const {
        return bit / wordSize;
    }

    /// @brief Get the bit index within the word for the given bit index.
    T
    bitInWord(std::size_t bit) const {
        return bit % wordSize;
    }

    std::size_t mTileSize;   ///< The size of the square tile.
    std::vector<T> mBitMask; ///< The bitmask storage for the tile.
};
} // namespace fvdb::test

#endif // FVDB_TESTS_UTILS_TILEBITMASK_H
