// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file Invoke.h

    \author Ken Museth

    \date August 24, 2020

    \brief A unified wrapper for tbb::parallel_invoke and a naive std::thread analog
*/

#ifndef NANOVDB_INVOKE_H_HAS_BEEN_INCLUDED
#define NANOVDB_INVOKE_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"// for nanovdb::CoordBBox

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_invoke.h>
#endif

#include <thread>
#include <mutex>
#include <vector>

namespace nanovdb {

namespace {
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
}// unnamed namespace

/// @return 0 for no work, 1 for serial, 2 for tbb multi-threading, and 3 for std multi-threading
template<typename Func, typename... Rest>
int invoke(const Func &taskFunc, Rest... rest) {
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

#endif // NANOVDB_INVOKE_H_HAS_BEEN_INCLUDED
