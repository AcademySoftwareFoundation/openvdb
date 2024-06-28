// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/util/Invoke.h

    \author Ken Museth

    \date August 24, 2020

    \brief A unified wrapper for tbb::parallel_invoke and a naive std::thread analog

    @code
    template<typename Func0, typename Func1, ..., typename FuncN>
    int invoke(const Func0& f0, const Func1& f1, ..., const FuncN& fN);
    @endcode
*/

#ifndef NANOVDB_UTIL_INVOKE_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_INVOKE_H_HAS_BEEN_INCLUDED

#include <nanovdb/NanoVDB.h>// for nanovdb::CoordBBox

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_invoke.h>
#endif

#include <thread>
#include <mutex>
#include <vector>

namespace nanovdb {

namespace util {

namespace {
#ifndef NANOVDB_USE_TBB
// Base case
template<typename Func>
void parallel_invoke(std::vector<std::thread> &threadPool, const Func &taskFunc) {
    threadPool.emplace_back(taskFunc);
}

// Iterative call
template<typename Func, typename... Rest>
void parallel_invoke(std::vector<std::thread> &threadPool, const Func &taskFunc1, Rest... taskFuncN) {
    threadPool.emplace_back(taskFunc1);
    parallel_invoke(threadPool, taskFuncN...);
}

// Base case
template<typename Func>
void serial_invoke(const Func &taskFunc) {taskFunc();}

// Iterative call
template<typename Func, typename... Rest>
void serial_invoke(const Func &taskFunc1, Rest... taskFuncN) {
    taskFunc1();
    serial_invoke(taskFuncN...);
}
#endif
}// unnamed namespace

/// @return 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template<typename Func, typename... Rest>
int invoke(const Func &taskFunc1, Rest... taskFuncN) {
#ifdef NANOVDB_USE_TBB
    tbb::parallel_invoke(taskFunc1, taskFuncN...);
    return 2;
#else
    const auto threadCount = std::thread::hardware_concurrency()>>1;
    if (1 + sizeof...(Rest) <= threadCount) {
        std::vector<std::thread> threadPool;
        threadPool.emplace_back(taskFunc1);
        parallel_invoke(threadPool, taskFuncN...);
        for (auto &t : threadPool) t.join();
        return 3;// std multi-threading
    } else {
        taskFunc1();
        serial_invoke(taskFuncN...);
        return 1;// serial
    }
#endif
    return -1;// should never happen
}

}// namespace util

template<typename Func, typename... Rest>
[[deprecated("Use nanovdb::util::invoke instead")]]
int invoke(const Func &taskFunc1, Rest... taskFuncN) {
    return util::invoke<Func, Rest...>(taskFunc1, taskFuncN...);
}

}// namespace nanovdb

#endif // NANOVDB_UTIL_INVOKE_H_HAS_BEEN_INCLUDED
