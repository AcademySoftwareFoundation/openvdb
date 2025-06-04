// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
#define FVDB_DETAIL_UTILS_CUDA_UTILS_CUH

#include <c10/util/Exception.h>

#include <limits>

namespace fvdb {

/// @brief Get the number of blocks for a CUDA kernel launch given the number of elements and the
/// maximum number of threads per block
/// @param N The number of elements to parallelize over
/// @param maxThreadsPer dBlock The maximum number of threads per block
/// @return The number of blocks for a CUDA kernel launch
static int
GET_BLOCKS(const int64_t N, const int64_t maxThreadsPerBlock) {
    if (N <= 0) {
        return 0;
    }

    constexpr int64_t max_int = std::numeric_limits<int>::max();

    // Round up division for positive number that cannot cause integer overflow
    auto block_num = (N - 1) / maxThreadsPerBlock + 1;
    TORCH_INTERNAL_ASSERT(block_num <= max_int, "Can't schedule too many blocks on CUDA device");

    return static_cast<int>(block_num);
}

} // namespace fvdb

#endif // FVDB_DETAIL_UTILS_CUDA_UTILS_CUH
