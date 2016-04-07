///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015-2016 Double Negative Visual Effects
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of Double Negative Visual Effects nor the names
// of its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// IN NO EVENT SHALL THE COPYRIGHT HOLDERS' AND CONTRIBUTORS' AGGREGATE
// LIABILITY FOR ALL CLAIMS REGARDLESS OF THEIR BASIS EXCEED US$250.00.
//
///////////////////////////////////////////////////////////////////////////
//
/// @author Dan Bailey
///
/// @file PointCount.h
///
/// @brief  Various point counting methods using a VDB Point Grid.
///


#ifndef OPENVDB_TOOLS_POINT_COUNT_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_COUNT_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>

#include <openvdb_points/tools/AttributeSet.h>
#include <openvdb_points/tools/PointDataGrid.h>
#include <openvdb_points/tools/PointAttribute.h>

#include <boost/ptr_container/ptr_vector.hpp>

#include <tbb/parallel_reduce.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Total points in the PointDataTree
/// @param tree PointDataTree.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 pointCount(const PointDataTreeT& tree, const bool inCoreOnly = false);


/// @brief Total active points in the PointDataTree
/// @param tree PointDataTree.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 activePointCount(const PointDataTreeT& tree, const bool inCoreOnly = false);


/// @brief Total inactive points in the PointDataTree
/// @param tree PointDataTree.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 inactivePointCount(const PointDataTreeT& tree, const bool inCoreOnly = false);


/// @brief Total points in the group in the PointDataTree
/// @param tree PointDataTree.
/// @param name group name.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 groupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly = false);


/// @brief Total active points in the group in the PointDataTree
/// @param tree PointDataTree.
/// @param name group name.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 activeGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly = false);


/// @brief Total inactive points in the group in the PointDataTree
/// @param tree PointDataTree.
/// @param name group name.
/// @param inCoreOnly if true, points in out-of-core leaf nodes are not counted
template <typename PointDataTreeT>
Index64 inactiveGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly = false);


////////////////////////////////////////


namespace point_count_internal {

template <  typename PointDataTreeT,
            typename ValueIterT,
            typename FilterT>
struct PointCountOp
{
    typedef typename tree::LeafManager<const PointDataTreeT>    LeafManagerT;
    typedef IndexIterTraits<PointDataTreeT, ValueIterT>         IndexIteratorFromLeafT;
    typedef typename IndexIteratorFromLeafT::Iterator           IndexIterator;
    typedef typename FilterT::Data                              FilterDataT;
    typedef FilterIndexIter<IndexIterator, FilterT>             Iterator;

    PointCountOp(const FilterDataT& filterData,
                 const bool inCoreOnly = false)
        : mFilterData(filterData)
        , mInCoreOnly(inCoreOnly) { }

    Index64 operator()(const typename LeafManagerT::LeafRange& range, Index64 size) const {

        for (typename LeafManagerT::LeafRange::Iterator leaf = range.begin(); leaf; ++leaf) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
            if (mInCoreOnly && leaf->buffer().isOutOfCore())     continue;
#endif
            IndexIterator indexIterator(IndexIteratorFromLeafT::begin(*leaf));
            FilterT filter(FilterT::create(*leaf, mFilterData));
            Iterator iter(indexIterator, filter);
            size += iterCount(iter);
        }

        return size;
    }

    static Index64 join(Index64 size1, Index64 size2) {
        return size1 + size2;
    }

private:
    const FilterDataT& mFilterData;
    const bool mInCoreOnly;
}; // struct PointCountOp


template <typename PointDataTreeT, typename FilterT, typename ValueIterT>
Index64 threadedFilterPointCount(   const PointDataTreeT& tree,
                                    const typename FilterT::Data& filter,
                                    const bool inCoreOnly = false)
{
    typedef point_count_internal::PointCountOp< PointDataTreeT, ValueIterT, FilterT> PointCountOp;

    typename tree::LeafManager<const PointDataTreeT> leafManager(tree);
    const PointCountOp pointCountOp(filter, inCoreOnly);
    return tbb::parallel_reduce(leafManager.leafRange(), Index64(0), pointCountOp, PointCountOp::join);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterPointCount(const PointDataTreeT& tree,
                         const typename FilterT::Data& filter,
                         const bool inCoreOnly = false)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueAllCIter ValueIterT;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterActivePointCount( const PointDataTreeT& tree,
                                const typename FilterT::Data& filter,
                                const bool inCoreOnly = false)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueOnCIter ValueIterT;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterInactivePointCount(   const PointDataTreeT& tree,
                                    const typename FilterT::Data& filter,
                                    const bool inCoreOnly = false)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueOffCIter ValueIterT;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


} // namespace point_count_internal


template <typename PointDataTreeT>
Index64 pointCount(const PointDataTreeT& tree, const bool inCoreOnly)
{
    (void) inCoreOnly;
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#endif
        size += iter->pointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 activePointCount(const PointDataTreeT& tree, const bool inCoreOnly)
{
    (void) inCoreOnly;
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#endif
        size += iter->onPointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 inactivePointCount(const PointDataTreeT& tree, const bool inCoreOnly)
{
    (void) inCoreOnly;
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#endif
        size += iter->offPointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 groupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter::Data groupFilterData(name);
    return point_count_internal::filterPointCount<PointDataTreeT, GroupFilter>(tree, groupFilterData, inCoreOnly);
}


template <typename PointDataTreeT>
Index64 activeGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter::Data groupFilterData(name);
    return point_count_internal::filterActivePointCount<PointDataTreeT, GroupFilter>(tree, groupFilterData, inCoreOnly);
}


template <typename PointDataTreeT>
Index64 inactiveGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter::Data groupFilterData(name);
    return point_count_internal::filterInactivePointCount<PointDataTreeT, GroupFilter>(tree, groupFilterData, inCoreOnly);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_COUNT_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
