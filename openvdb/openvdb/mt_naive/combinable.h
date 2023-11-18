
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file combinable.h

#ifndef OPENVDB_COMBINABLE_HAS_BEEN_INCLUDED
#define OPENVDB_COMBINABLE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <unordered_map>
#include <shared_mutex>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

#if 1

template <typename T>
using combinable = ::tbb::combinable<T>;

#else

/** \name combinable
**/
//@{
//! Thread-local storage with optional reduction
/** @ingroup containers */
template <typename T>
class combinable {

private:
    my_ets_type;
    my_ets_type my_ets;

public:

    combinable() { }

    template <typename finit>
    explicit combinable( finit _finit) : my_ets(_finit) { }

    //! destructor
    ~combinable() { }

    combinable( const combinable& other) : my_ets(other.my_ets) { }

#if __TBB_ETS_USE_CPP11
    combinable( combinable&& other) : my_ets( std::move(other.my_ets)) { }
#endif

    combinable & operator=( const combinable & other) {
        my_ets = other.my_ets;
        return *this;
    }

#if __TBB_ETS_USE_CPP11
    combinable & operator=( combinable && other) {
        my_ets=std::move(other.my_ets);
        return *this;
    }
#endif

    void clear() { my_ets.clear(); }

    T& local() { return my_ets.local(); }

    T& local(bool & exists) { return my_ets.local(exists); }

    // combine_func_t has signature T(T,T) or T(const T&, const T&)
    template <typename combine_func_t>
    T combine(combine_func_t f_combine) { return my_ets.combine(f_combine); }

    // combine_func_t has signature void(T) or void(const T&)
    template <typename combine_func_t>
    void combine_each(combine_func_t f_combine) { my_ets.combine_each(f_combine); }

};

#endif

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_COMBINABLE_HAS_BEEN_INCLUDED
