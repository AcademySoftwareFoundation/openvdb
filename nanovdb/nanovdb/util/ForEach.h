// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file ForEach.h

    \author Ken Museth

    \date August 24, 2020

    \brief A unified wrapper for tbb::parallel_for and a naive std::thread analog
*/

#ifndef NANOVDB_FOREACH_H_HAS_BEEN_INCLUDED
#define NANOVDB_FOREACH_H_HAS_BEEN_INCLUDED

#include "Range.h"// for Range1D 

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_for.h>
#endif

#include <thread>
#include <mutex>
#include <vector>

namespace nanovdb {

/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
/// func = [](const RangeT&){...}, 
/// RangeT = Range, CoordBBox, tbb::blocked_range, blocked_range2D, or blocked_range3D.
template <typename RangeT, typename Func>
int forEach(RangeT taskRange, const Func &taskFunc)
{
    if (taskRange.empty()) return 0;
#ifdef NANOVDB_USE_TBB
    tbb::parallel_for(taskRange, taskFunc);
    return 2;/// tbb multi-threading
#else
    if (const size_t threadCount = std::thread::hardware_concurrency()>>1) {
        std::vector<RangeT> rangePool{ taskRange };
        while(rangePool.size() < threadCount) {
            const size_t oldSize = rangePool.size();
            for (size_t i = 0; i < oldSize && rangePool.size() < threadCount; ++i) {
                auto &r = rangePool[i];
                if (r.is_divisible()) rangePool.push_back(RangeT(r, Split()));
            }
            if (rangePool.size() == oldSize) break;// none of the ranges were divided so stop
        }
        std::vector<std::thread> threadPool;
        for (auto &r : rangePool) threadPool.emplace_back(taskFunc, r);// launch threads
        for (auto &t : threadPool) t.join();// syncronize threads
        return 3;// std multi-threading
    } else {
        taskFunc(taskRange);
        return 1;// serial
    }
#endif
    return -1;// should never happen
}

/// @brief Simple wrapper to the method defined above func = [](const Range1D&){...}
///
/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template <typename Func>
int forEach(size_t begin, size_t end, size_t grainSize, const Func& func)
{
    return forEach(Range1D(begin, end, grainSize), func);
}

}// namespace nanovdb

#endif // NANOVDB_FOREACH_H_HAS_BEEN_INCLUDED
