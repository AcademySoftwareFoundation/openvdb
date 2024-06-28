// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/util/Reduce.h

    \author Ken Museth

    \date March 4, 2021

    \brief A unified wrapper for tbb::parallel_reduce and a naive std::future analog
*/

#ifndef NANOVDB_UTIL_REDUCE_H_HAS_BEEN_INCLUDED
#define NANOVDB_UTIL_REDUCE_H_HAS_BEEN_INCLUDED

#include <nanovdb/util/Range.h>// for util::Range1D

#ifdef NANOVDB_USE_TBB
#include <tbb/parallel_reduce.h>
#else
#include <thread>
#include <future>
#include <vector>
#endif

namespace nanovdb {

namespace util {

/// @return reduction
///
/// @param range  RangeT can be Range<dim,T>, CoordBBox, tbb::blocked_range, blocked_range2D, or blocked_range3D.
/// @param identity  initial value
/// @param func   functor with signature T FuncT::operator()(const RangeT& range, const T& a) const
/// @param join   functor with the signature T JoinT::operator()(const T& a, const T& b) const
/// @code
///     std::vector<int> array(100, 1);
///     auto func = [&array](auto &r, int a){for (auto i=r.begin(); i!=r.end(); ++i) a+=array[i]; return a;};
///     int sum = reduce(array, 0, func, [](int a, int b){return a + b;});
/// @endcode
template <typename RangeT, typename T, typename FuncT, typename JoinT>
inline T reduce(RangeT range, const T& identity, const FuncT &func, const JoinT &join)
{
    if (range.empty()) return identity;
#ifdef NANOVDB_USE_TBB
    return tbb::parallel_reduce(range, identity, func, join);
#else// naive and likely slow alternative based on std::future
    if (const size_t threadCount = std::thread::hardware_concurrency()>>1) {
        std::vector<RangeT> rangePool{ range };
        while(rangePool.size() < threadCount) {
            const size_t oldSize = rangePool.size();
            for (size_t i = 0; i < oldSize && rangePool.size() < threadCount; ++i) {
                auto &r = rangePool[i];
                if (r.is_divisible()) rangePool.push_back(RangeT(r, Split()));
            }
            if (rangePool.size() == oldSize) break;// none of the ranges were divided so stop
        }
        std::vector< std::future<T> > futurePool;
        for (auto &r : rangePool) {
            auto task = std::async(std::launch::async, [&](){return func(r, identity);});
            futurePool.push_back( std::move(task) );// launch tasks
        }
        T result = identity;
        for (auto &f : futurePool) {
            result = join(result, f.get());// join results
        }
        return result;
    } else {// serial
        return static_cast<T>(func(range, identity));
    }
#endif
    return identity;// should never happen
}

/// @brief Simple wrapper to the function defined above
template <typename T, typename FuncT, typename JoinT>
inline T reduce(size_t begin, size_t end, size_t grainSize, const T& identity, const FuncT& func, const JoinT& join)
{
    Range1D range(begin, end, grainSize);
    return reduce( range, identity, func, join );
}

/// @brief Simple wrapper that works with std::containers
template <template<typename...> class ContainerT, typename... ArgT, typename T, typename FuncT, typename JoinT >
inline T reduce(const ContainerT<ArgT...> &c, const T& identity, const FuncT& func, const JoinT& join)
{
    Range1D range(0, c.size(), 1);
    return reduce( range, identity, func, join );

}

/// @brief Simple wrapper that works with std::containers
template <template<typename...> class ContainerT, typename... ArgT, typename T, typename FuncT, typename JoinT >
inline T reduce(const ContainerT<ArgT...> &c, size_t grainSize, const T& identity, const FuncT& func, const JoinT& join)
{
    Range1D range(0, c.size(), grainSize);
    return reduce( range, identity, func, join );
}

}// namespace util

/// @brief Simple wrapper to the function defined above
template <typename T, typename FuncT, typename JoinT>
[[deprecated("Use nanovdb::util::reduce instead")]]
inline T reduce(size_t begin, size_t end, size_t grainSize, const T& identity, const FuncT& func, const JoinT& join)
{
    util::Range1D range(begin, end, grainSize);
    return util::reduce( range, identity, func, join );
}

/// @brief Simple wrapper that works with std::containers
template <template<typename...> class ContainerT, typename... ArgT, typename T, typename FuncT, typename JoinT >
[[deprecated("Use nanovdb::util::reduce instead")]]
inline T reduce(const ContainerT<ArgT...> &c, const T& identity, const FuncT& func, const JoinT& join)
{
    util::Range1D range(0, c.size(), 1);
    return util::reduce( range, identity, func, join );

}

/// @brief Simple wrapper that works with std::containers
template <template<typename...> class ContainerT, typename... ArgT, typename T, typename FuncT, typename JoinT >
[[deprecated("Use nanovdb::util::reduce instead")]]
T reduce(const ContainerT<ArgT...> &c, size_t grainSize, const T& identity, const FuncT& func, const JoinT& join)
{
    util::Range1D range(0, c.size(), grainSize);
    return util::reduce( range, identity, func, join );
}

}// namespace nanovdb

#endif // NANOVDB_UTIL_REDUCE_H_HAS_BEEN_INCLUDED
