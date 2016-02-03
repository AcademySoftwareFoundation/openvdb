///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2015 Double Negative Visual Effects
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

template <typename IterT>
inline Index64 iterCount(const IterT& iter)
{
    Index64 size = 0;
    for (IterT newIter(iter); newIter; ++newIter, ++size) { }
    return size;
}


template <typename PointDataTreeT>
Index64 pointCount(const PointDataTreeT& tree)
{
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
        size += iter->pointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 activePointCount(const PointDataTreeT& tree)
{
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
        size += iter->onPointCount();
    }
    return size;
}


template <typename PointDataTreeT>
Index64 inactivePointCount(const PointDataTreeT& tree)
{
    Index64 size = 0;
    for (typename PointDataTreeT::LeafCIter iter = tree.cbeginLeaf(); iter; ++iter) {
        size += iter->offPointCount();
    }
    return size;
}


namespace point_count_internal {

template <  typename PointDataTreeT,
            typename ValueIterT,
            typename FilterFromLeafT>
struct PointCountOp
{
    typedef typename tree::LeafManager<PointDataTreeT>      LeafManagerT;
    typedef typename PointDataTreeT::LeafNodeType           LeafT;
    typedef IndexIterTraits<PointDataTreeT, ValueIterT>     IndexIteratorFromLeafT;
    typedef typename IndexIteratorFromLeafT::Iterator       IndexIterator;
    typedef typename FilterFromLeafT::Filter                Filter;
    typedef FilterIndexIter<IndexIterator, Filter>          Iterator;

    PointCountOp(const PointDataTreeT& tree,
                 const LeafManagerT& leafs,
                 const FilterFromLeafT& filterFromLeaf)
        : mTree(tree)
        , mLeafs(leafs)
        , mFilterFromLeaf(filterFromLeaf) { }

    Index64 operator()(const tbb::blocked_range<size_t>& range, Index64 size) const {

        for (size_t n = range.begin(); n != range.end(); ++n) {
            const LeafT& leaf = mLeafs.leaf(n);

            IndexIterator indexIterator(IndexIteratorFromLeafT::begin(leaf));
            Filter filter(mFilterFromLeaf.fromLeaf(leaf));
            Iterator iter(indexIterator, filter);
            size += iterCount<Iterator>(iter);
        }

        return size;
    }

    static Index64 join(Index64 size1, Index64 size2) {
        return size1 + size2;
    }

private:
    const PointDataTreeT& mTree;
    const LeafManagerT& mLeafs;
    const FilterFromLeafT& mFilterFromLeaf;
}; // struct PointCountOp


template <typename PointDataTreeT, typename FilterFromLeafT, typename ValueIterT>
Index64 threadedFilterPointCount(   const PointDataTreeT& tree,
                                    const FilterFromLeafT& filterFromLeaf)
{
    typedef point_count_internal::PointCountOp< PointDataTreeT, ValueIterT, FilterFromLeafT> PointCountOp;

    typename tree::LeafManager<PointDataTreeT> leafs(const_cast<PointDataTreeT&>(tree));
    const PointCountOp pointCountOp(tree, leafs, filterFromLeaf);
    return tbb::parallel_reduce(leafs.getRange(), Index64(0), pointCountOp, PointCountOp::join);
}

} // namespace point_count_internal


template <typename PointDataTreeT, typename FilterFromLeafT>
Index64 filterPointCount(const PointDataTreeT& tree,
                         const FilterFromLeafT& filterFromLeaf)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueAllCIter ValueIterT;
    return point_count_internal::threadedFilterPointCount<  PointDataTreeT, FilterFromLeafT,
                                                            ValueIterT>(tree, filterFromLeaf);
}

template <typename PointDataTreeT, typename FilterFromLeafT>
Index64 filterActivePointCount( const PointDataTreeT& tree,
                                const FilterFromLeafT& filterFromLeaf)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueOnCIter ValueIterT;
    return point_count_internal::threadedFilterPointCount<  PointDataTreeT, FilterFromLeafT,
                                                            ValueIterT>(tree, filterFromLeaf);
}

template <typename PointDataTreeT, typename FilterFromLeafT>
Index64 filterInactivePointCount(   const PointDataTreeT& tree,
                                    const FilterFromLeafT& filterFromLeaf)
{
    typedef typename PointDataTreeT::LeafNodeType::ValueOffCIter ValueIterT;
    return point_count_internal::threadedFilterPointCount<  PointDataTreeT, FilterFromLeafT,
                                                            ValueIterT>(tree, filterFromLeaf);
}

template <typename PointDataTreeT>
Index64 groupPointCount(const PointDataTreeT& tree, const Name& name)
{
    GroupFilterFromLeaf filterFromLeaf(name);
    return filterPointCount(tree, filterFromLeaf);
}


template <typename PointDataTreeT>
Index64 activeGroupPointCount(const PointDataTreeT& tree, const Name& name)
{
    GroupFilterFromLeaf filterFromLeaf(name);
    return filterActivePointCount(tree, filterFromLeaf);
}


template <typename PointDataTreeT>
Index64 inactiveGroupPointCount(const PointDataTreeT& tree, const Name& name)
{
    GroupFilterFromLeaf filterFromLeaf(name);
    return filterInactivePointCount(tree, filterFromLeaf);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_COUNT_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
