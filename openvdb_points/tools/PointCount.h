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
#include <openvdb_points/tools/IndexFilter.h>

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


/// @brief Populate an array of cumulative point offsets per leaf node.
/// @param pointOffsets     array of offsets to be populated.
/// @param tree             PointDataTree from which to populate the offsets.
/// @param includeGroups    the group of names to include.
/// @param excludeGroups    the group of names to exclude.
/// @param inCoreOnly       if true, points in out-of-core leaf nodes are ignored
/// @note returns the final cumulative point offset.
template <typename PointDataTreeT>
Index64 getPointOffsets(std::vector<Index64>& pointOffsets, const PointDataTreeT& tree,
                     const std::vector<Name>& includeGroups = std::vector<Name>(),
                     const std::vector<Name>& excludeGroups = std::vector<Name>(),
                     const bool inCoreOnly = true);


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
    using LeafManagerT = typename tree::LeafManager<const PointDataTreeT>;

    PointCountOp(const FilterT& filter,
                 const bool inCoreOnly = false)
        : mFilter(filter)
        , mInCoreOnly(inCoreOnly) { }

    Index64 operator()(const typename LeafManagerT::LeafRange& range, Index64 size) const {

        for (auto leaf = range.begin(); leaf; ++leaf) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
            if (mInCoreOnly && leaf->buffer().isOutOfCore())     continue;
#endif

            auto iter = leaf->template beginIndex<ValueIterT, FilterT>(mFilter);
            size += iterCount(iter);
        }

        return size;
    }

    static Index64 join(Index64 size1, Index64 size2) {
        return size1 + size2;
    }

private:
    const FilterT& mFilter;
    const bool mInCoreOnly;
}; // struct PointCountOp


template <typename PointDataTreeT, typename FilterT, typename ValueIterT>
Index64 threadedFilterPointCount(   const PointDataTreeT& tree,
                                    const FilterT& filter,
                                    const bool inCoreOnly = false)
{
    using PointCountOp = point_count_internal::PointCountOp< PointDataTreeT, ValueIterT, FilterT>;

    typename tree::LeafManager<const PointDataTreeT> leafManager(tree);
    const PointCountOp pointCountOp(filter, inCoreOnly);
    return tbb::parallel_reduce(leafManager.leafRange(), Index64(0), pointCountOp, PointCountOp::join);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterPointCount(const PointDataTreeT& tree,
                         const FilterT& filter,
                         const bool inCoreOnly = false)
{
    using ValueIterT = typename PointDataTreeT::LeafNodeType::ValueAllCIter;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterActivePointCount( const PointDataTreeT& tree,
                                const FilterT& filter,
                                const bool inCoreOnly = false)
{
    using ValueIterT = typename PointDataTreeT::LeafNodeType::ValueOnCIter;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


template <typename PointDataTreeT, typename FilterT>
Index64 filterInactivePointCount(   const PointDataTreeT& tree,
                                    const FilterT& filter,
                                    const bool inCoreOnly = false)
{
    using ValueIterT = typename PointDataTreeT::LeafNodeType::ValueOffCIter;
    return threadedFilterPointCount<  PointDataTreeT, FilterT, ValueIterT>(tree, filter, inCoreOnly);
}


} // namespace point_count_internal


template <typename PointDataTreeT>
Index64 pointCount(const PointDataTreeT& tree, const bool inCoreOnly)
{
    (void) inCoreOnly;
    Index64 size = 0;
    for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#else
        (void) inCoreOnly; // unused variable
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
    for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#else
        (void) inCoreOnly; // unused variable
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
    for (auto iter = tree.cbeginLeaf(); iter; ++iter) {
#ifndef OPENVDB_2_ABI_COMPATIBLE
        if (inCoreOnly && iter->buffer().isOutOfCore())     continue;
#else
        (void) inCoreOnly; // unused variable
#endif
        size += iter->offPointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 groupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter groupFilter(name);
    return point_count_internal::filterPointCount<PointDataTreeT, GroupFilter>(tree, groupFilter, inCoreOnly);
}


template <typename PointDataTreeT>
Index64 activeGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter groupFilter(name);
    return point_count_internal::filterActivePointCount<PointDataTreeT, GroupFilter>(tree, groupFilter, inCoreOnly);
}


template <typename PointDataTreeT>
Index64 inactiveGroupPointCount(const PointDataTreeT& tree, const Name& name, const bool inCoreOnly)
{
    GroupFilter groupFilter(name);
    return point_count_internal::filterInactivePointCount<PointDataTreeT, GroupFilter>(tree, groupFilter, inCoreOnly);
}


template <typename PointDataTreeT>
Index64 getPointOffsets(std::vector<Index64>& pointOffsets, const PointDataTreeT& tree,
                     const std::vector<Name>& includeGroups, const std::vector<Name>& excludeGroups,
                     const bool inCoreOnly)
{
    using LeafNode = typename PointDataTreeT::LeafNodeType;

    const bool useGroup = includeGroups.size() > 0 || excludeGroups.size() > 0;

    tree::LeafManager<const PointDataTreeT> leafManager(tree);
    const size_t leafCount = leafManager.leafCount();

    pointOffsets.reserve(leafCount);

    Index64 pointOffset = 0;
    for (size_t n = 0; n < leafCount; n++)
    {
        const LeafNode& leaf = leafManager.leaf(n);

#ifndef OPENVDB_2_ABI_COMPATIBLE
        // skip out-of-core leafs
        if (inCoreOnly && leaf.buffer().isOutOfCore()) {
            pointOffsets.push_back(pointOffset);
            continue;
        }
#else
        (void) inCoreOnly; // unused variable
#endif

        if (useGroup) {
            auto iter = leaf.beginValueOn();
            MultiGroupFilter filter(includeGroups, excludeGroups);
            filter.reset(leaf);
            IndexIter<typename LeafNode::ValueOnCIter, MultiGroupFilter> filterIndexIter(iter, filter);
            pointOffset += iterCount(filterIndexIter);
        }
        else {
            pointOffset += leaf.onPointCount();
        }
        pointOffsets.push_back(pointOffset);
    }
    return pointOffset;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_COUNT_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
