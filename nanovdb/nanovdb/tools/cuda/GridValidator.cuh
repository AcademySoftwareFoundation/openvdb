// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/cuda/GridValidator.cuh

    \author Ken Museth

    \date November 3, 2023

    \brief Checks the validity of an existing NanoVDB device grid.
*/

#ifndef NANOVDB_TOOLS_CUDA_GRIDVALIDATOR_CUH_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_CUDA_GRIDVALIDATOR_CUH_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridValidator.h>
#include <nanovdb/tools/cuda/GridChecksum.cuh>
#include <nanovdb/util/cuda/Util.h>

namespace nanovdb {

namespace tools::cuda {

/// @brief Return true if the specified grid passes several validation tests.
///
/// @param grid Grid to validate
/// @param detailed If true the validation test is detailed and relatively slow.
/// @param verbose If true information about the first failed test is printed to std::cerr
template <typename ValueT>
bool isValid(const NanoGrid<ValueT> *d_grid, CheckMode mode, bool verbose = false, cudaStream_t stream = 0)
{
    static const int size = 100;
    std::unique_ptr<char[]> strUP(new char[size]);
    util::cuda::unique_ptr<char> d_strUP(size);
    char *str = strUP.get(), *d_str = d_strUP.get();

    util::cuda::lambdaKernel<<<1, 1, 0, stream>>>(1, [=] __device__(size_t) {nanovdb::tools::checkGrid(d_grid, d_str, mode);});
    cudaMemcpyAsync(str, d_str, size, cudaMemcpyDeviceToHost, stream);

    if (util::empty(str) && !cuda::validateChecksum(d_grid, mode)) util::strcpy(str, "Mis-matching checksum");
    if (verbose && !util::empty(str)) std::cerr << "Validation failed: " << str << std::endl;

    return util::empty(str);
}// tools::cuda::isValid

}// namespace tools::cuda

template <typename ValueT>
[[deprecated("Use cuda::isValid() instead.")]]
bool cudaIsValid(const NanoGrid<ValueT> *d_grid, CheckMode mode, bool verbose = false, cudaStream_t stream = 0)
{
    return tools::cuda::isValid(d_grid, mode, verbose, stream);
}

} // namespace nanovdb

#endif // NANOVDB_TOOLS_CUDA_GRIDVALIDATOR_CUH_HAS_BEEN_INCLUDED
