
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
///
/// @file blocked_range.h

#ifndef OPENVDB_BLOCKED_RANGE_HAS_BEEN_INCLUDED
#define OPENVDB_BLOCKED_RANGE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>

#include <openvdb/mt/split.h>

#include <cassert>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace mt {

template<typename Value>
class blocked_range {
public:
    typedef Value const_iterator;
    typedef std::size_t size_type;

    //! Construct range over half-open interval [begin,end), with the given grainsize.
    blocked_range( Value begin_, Value end_, size_type grainsize_=1 ) :
        my_end(end_), my_begin(begin_), my_grainsize(grainsize_)
    {
        assert( my_grainsize>0 && "grainsize must be positive" );
    }

    //! Beginning of range.
    const_iterator begin() const {return my_begin;}

    //! One past last value in range.
    const_iterator end() const {return my_end;}

    //! Size of the range
    /** Unspecified if end()<begin(). */
    size_type size() const {
        assert( !(end()<begin()) && "size() unspecified if end()<begin()" );
        return size_type(my_end-my_begin);
    }

    //! The grain size for this range.
    size_type grainsize() const {return my_grainsize;}

    //------------------------------------------------------------------------
    // Methods that implement Range concept
    //------------------------------------------------------------------------

    //! True if range is empty.
    bool empty() const {return !(my_begin<my_end);}

    //! True if range is divisible.
    /** Unspecified if end()<begin(). */
    bool is_divisible() const {return my_grainsize<size();}

    //! Split range.
    /** The new Range *this has the second part, the old range r has the first part.
        Unspecified if end()<begin() or !is_divisible(). */
    blocked_range( blocked_range& r, split ) :
        my_end(r.my_end),
        my_begin(do_split(r, split())),
        my_grainsize(r.my_grainsize)
    {
        // only comparison 'less than' is required from values of blocked_range objects
        assert( !(my_begin < r.my_end) && !(r.my_end < my_begin) && "blocked_range has been split incorrectly" );
    }

private:
    /** NOTE: my_end MUST be declared before my_begin, otherwise the splitting constructor will break. */
    Value my_end;
    Value my_begin;
    size_type my_grainsize;

    //! Auxiliary function used by the splitting constructor.
    static Value do_split( blocked_range& r, split )
    {
        assert( r.is_divisible() && "cannot split blocked_range that is not divisible" );
        Value middle = r.my_begin + (r.my_end - r.my_begin) / 2u;
        r.my_end = middle;
        return middle;
    }

    template<typename RowValue, typename ColValue>
    friend class blocked_range2d;

    template<typename RowValue, typename ColValue, typename PageValue>
    friend class blocked_range3d;

    //template<typename DimValue, unsigned int N, typename>
    //friend class internal::blocked_rangeNd_impl;
};

} // mt
} // OPENVDB_VERSION_NAME
} // openvdb

#endif // OPENVDB_BLOCKED_RANGE_HAS_BEEN_INCLUDED
