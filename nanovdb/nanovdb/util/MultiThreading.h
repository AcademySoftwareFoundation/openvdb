// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file MultiThreading.h

    \author Ken Museth

    \date August 24, 2020

    \brief A unified wrapper for tbb and std multi-threading
*/

#ifndef NANOVDB_THREADING_H_HAS_BEEN_INCLUDED
#define NANOVDB_THREADING_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"// for nanovdb::CoordBBox

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#endif

#include <thread>
#include <mutex>
#include <vector>

namespace nanovdb {

class Split {};// Dummy class used by split constructors

template <typename IndexT>
class BlockedRange
{
    static_assert(std::is_integral<IndexT>::value, "Expected an integer type");
    IndexT mBegin, mEnd, mGrainSize;
    
public:
    BlockedRange() : mBegin(0), mEnd(0), mGrainSize(1) {}
    BlockedRange(IndexT begin, IndexT end, IndexT grainSize = IndexT(1)) 
        : mBegin(begin), mEnd(end), mGrainSize(grainSize) 
    { 
        assert(grainSize > IndexT(0)); 
    }
    BlockedRange(BlockedRange &r, Split) : mBegin(r.mBegin), mEnd(r.mEnd), mGrainSize(r.mGrainSize) {
        assert(r.is_divisible());
        r.mBegin = mEnd = this->middle();
    }
#ifdef NANOVDB_USE_TBB
    BlockedRange(BlockedRange &r, tbb::split) : BlockedRange(r, Split()) {}
#endif
    IndexT middle() const {return mBegin + ((mEnd - mBegin) >> 1);}
    IndexT size()  const { return mEnd - mBegin; }
    bool empty()   const { return !(mBegin < mEnd); }
    IndexT grainsize() const {return mGrainSize;}
    bool is_divisible() const {return mGrainSize < this->size();}
    IndexT begin() const { return mBegin; }
    IndexT end()   const { return mEnd; }
};// BlockedRange

/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template <typename RangeT, typename Func>
int parallel_for(RangeT taskRange, const Func &taskFunc)
{
    if (taskRange.empty()) return 0;
#ifdef NANOVDB_USE_TBB
    tbb::parallel_for(taskRange, taskFunc);
    return 2;/// tbb multi-threading
#else
    if (const size_t threadCount = std::thread::hardware_concurrency()>>1) {
        std::vector<RangeT> vec{ taskRange };
        while(vec.size() < threadCount) {
            const size_t m = vec.size();
            for (size_t n = 0; n < m && vec.size() < threadCount; ++n) {
                if (vec[n].is_divisible()) vec.push_back(RangeT(vec[n], Split()));
            }
            if (vec.size() == m) break;// none of the ranges were devisible
        }
        std::vector<std::thread> threadPool;
        for (auto &r : vec) threadPool.emplace_back(taskFunc, r);// launch threads
        for (auto &t : threadPool) t.join();// syncronize threads
        return 3;// std multi-threading
    } else {
        taskFunc(taskRange);
        return 1;// serial
    }
#endif
    return -1;// should never happen
}

/// @brief Simple wrapper to the method defined above
///
/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template <typename Func>
int parallel_for(size_t begin, size_t end, size_t grainSize, const Func& func)
{
    return parallel_for(BlockedRange<size_t>(begin, end, grainSize), func);
}

#ifndef NANOVDB_USE_TBB
// Base case
template<typename Func>
void parallel_invoke(std::vector<std::thread> &threadPool, const Func &taskFunc) {
    threadPool.emplace_back(taskFunc);
}

// Iterative call
template<typename Func, typename... Rest>
void parallel_invoke(std::vector<std::thread> &threadPool, const Func &taskFunc, Rest... rest) {
    threadPool.emplace_back(taskFunc);
    parallel_invoke(threadPool, rest...);
}

// Base case
template<typename Func>
void serial_invoke(const Func &taskFunc) {taskFunc();}

// Iterative call
template<typename Func, typename... Rest>
void serial_invoke(const Func &taskFunc, Rest... rest) {
    taskFunc();
    serial_invoke(rest...);
}
#endif

/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template<typename Func, typename... Rest>
int parallel_invoke(const Func &taskFunc, Rest... rest) {
#ifdef NANOVDB_USE_TBB
    tbb::parallel_invoke(taskFunc, rest...);
    return 2;
#else
    const auto threadCount = std::thread::hardware_concurrency()>>1;
    if (1 + sizeof...(Rest) <= threadCount) {
        std::vector<std::thread> threadPool;
        threadPool.emplace_back(taskFunc);
        parallel_invoke(threadPool, rest...);
        for (auto &t : threadPool) t.join();
        return 3;// std multi-threading
    } else {
        taskFunc();
        serial_invoke(rest...);
        return 1;// serial
    }
#endif
    return -1;// should never happen
}

}// namespace nanovdb

#endif // NANOVDB_THREADING_H_HAS_BEEN_INCLUDED
