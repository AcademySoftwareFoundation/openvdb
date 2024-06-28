// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/util/PrefixSum.h

    \author Ken Museth

    \date March 12, 2023

    \brief Multi-threaded implementations of inclusive prefix sum

    \note An exclusive prefix sum is simply an array starting with zero
          followed by the elements in the inclusive prefix sum, minus its
          last entry which is the sum of all the input elements.
*/

#ifndef NANOVDB_UTIL_PREFIX_SUM_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_PREFIX_SUM_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Range.h>// for Range1D
#include <vector>
#include <functional>// for std::plus

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_scan.h>
#endif

namespace nanovdb {

namespace util {

/// @brief Computes inclusive prefix sum of a vector
/// @tparam T Type of the elements in the input/out vector
/// @tparam OpT Type of operation performed on each element (defaults to sum)
/// @param vec input and output vector
/// @param threaded if true multi-threading is used
/// @note Inclusive prefix sum: for (i=1; i<N; ++i) vec[i] += vec[i-1]
/// @return sum of all input elements, which is also the last element of the inclusive prefix sum
template<typename T, typename OpT = std::plus<T>>
T prefixSum(std::vector<T> &vec, bool threaded = true, OpT op = OpT());

/// @brief An inclusive scan includes in[i] when computing out[i]
/// @note Inclusive prefix operation: for (i=1; i<N; ++i) vec[i] = Op(vec[i],vec[i-1])
template<typename T, typename Op>
void inclusiveScan(T *array, size_t size, const T &identity, bool threaded, Op op)
{
#ifndef NANOVDB_USE_TBB
    threaded = false;
    (void)identity;// avoids compiler warning
#endif

    if (threaded) {
#ifdef NANOVDB_USE_TBB
        using RangeT = tbb::blocked_range<size_t>;
        tbb::parallel_scan(RangeT(0, size), identity,
            [&](const RangeT &r, T sum, bool is_final_scan)->T {
                T tmp = sum;
                for (size_t i = r.begin(); i < r.end(); ++i) {
                    tmp = op(tmp, array[i]);
                    if (is_final_scan) array[i] = tmp;
                }
                return tmp;
            },[&](const T &a, const T &b) {return op(a, b);}
        );
#endif
    } else { // serial inclusive prefix operation
        for (size_t i=1; i<size; ++i) array[i] = op(array[i], array[i-1]);
    }
}// inclusiveScan

template<typename T, typename OpT>
T prefixSum(std::vector<T> &vec, bool threaded, OpT op)
{
    inclusiveScan(vec.data(), vec.size(), T(0), threaded, op);
    return vec.back();// sum of all input elements
}// prefixSum

}// namespace util

template<typename T, typename OpT = std::plus<T>>
[[deprecated("Use nanovdb::util::prefixSum instead")]]
T prefixSum(std::vector<T> &vec, bool threaded = true, OpT op = OpT())
{
    return util::prefixSum<T, OpT>(vec, threaded, op);
}// prefixSum

}// namespace nanovdb

#endif // NANOVDB_UTIL_PREFIX_SUM_H_HAS_BEEN_INCLUDED
