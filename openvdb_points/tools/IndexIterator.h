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
/// @file IndexIterator.h
///
/// @authors Dan Bailey
///
/// @brief  Index Iterators.
///


#ifndef OPENVDB_TOOLS_INDEX_ITERATOR_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_INDEX_ITERATOR_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


/// @brief Count up the number of times the iterator can iterate
///
/// @param iter the iterator.
///
/// @note counting by iteration only performed where a dynamic filter is in use,
template <typename IterT>
inline Index64 iterCount(const IterT& iter);


////////////////////////////////////////


/// @brief A forward iterator over array indices
class IndexIter
{
public:
    IndexIter()
        : mEnd(0), mItem(0) {}
    IndexIter(Index32 item, Index32 end)
        : mEnd(end), mItem(item) {}
    IndexIter(const IndexIter& other)
        : mEnd(other.mEnd), mItem(other.mItem) { }

    inline Index32 end() const { return mEnd; }

    /// @brief Reset the begining and end of the iterator.
    inline void reset(Index32 item, Index32 end) {
        mItem = item;
        mEnd = end;
    }

    /// @brief  Returns the item to which this iterator is currently pointing.
    inline Index32 operator*() { return mItem; }
    inline Index32 operator*() const { return mItem; }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    inline operator bool() const { return mItem < mEnd; }
    inline bool test() const { return mItem < mEnd; }

    /// @brief  Advance to the next (valid) item (prefix).
    inline IndexIter& operator++() {
        ++mItem;
        return *this;
    }

    /// @brief  Advance to the next (valid) item (postfix).
    inline IndexIter operator++(int /*dummy*/) {
        IndexIter newIterator(*this);
        this->operator++();
        return newIterator;
    }

    /// @brief  Advance to the next (valid) item.
    inline bool next() { this->operator++(); return this->test(); }
    inline bool increment() { this->next(); return this->test(); }

    /// Throw an error as Coord methods are not available on this iterator
    inline Coord getCoord() const { OPENVDB_THROW(RuntimeError, "IndexIter does not provide a valid Coord, use a ValueIndexIter instead."); }
    /// Throw an error as Coord methods are not available on this iterator
    inline void getCoord(Coord&) const { OPENVDB_THROW(RuntimeError, "IndexIter does not provide a valid Coord, use a ValueIndexIter instead."); }

    /// @brief Equality operators
    inline bool operator==(const IndexIter& other) const { return mItem == other.mItem; }
    inline bool operator!=(const IndexIter& other) const { return !this->operator==(other); }

private:
    Index32 mEnd, mItem;
}; // class IndexIter


/// @brief A forward iterator over array indices from a value iterator (such as ValueOnCIter)
template <typename ValueIterT>
class ValueIndexIter
{
public:
    ValueIndexIter(ValueIterT& iter)
        : mIndexIter(), mIter(iter), mParent(mIter.parent())
    {
        if (mIter) {
            Index32 start = mIter.offset() > 0 ? Index32(mParent.getValue(mIter.offset() - 1)) : Index32(0);
            mIndexIter.reset(start, *mIter);
            if (!mIndexIter.test())   this->operator++();
        }
    }
    ValueIndexIter(const ValueIndexIter& other)
        : mIndexIter(other.mIndexIter), mIter(other.mIter), mParent(other.mParent) { }

    inline Index32 end() const { return mIndexIter.end(); }

    inline void reset(Index32 item, Index32 end) {
        mIndexIter.reset(item, end);
    }

    /// @brief  Returns the item to which this iterator is currently pointing.
    inline Index32 operator*() { return *mIndexIter; }
    inline Index32 operator*() const { return *mIndexIter; }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    inline operator bool() const { return mIter; }
    inline bool test() const { return mIter; }

    /// @brief  Advance to the next (valid) item (prefix).
    inline ValueIndexIter& operator++() {
        mIndexIter.next();
        while (!mIndexIter.test() && mIter.next()) {
            mIndexIter.reset(mParent.getValue(mIter.offset() - 1), *mIter);
        }
        return *this;
    }

    /// @brief  Advance to the next (valid) item (postfix).
    inline ValueIndexIter operator++(int /*dummy*/) {
        IndexIter newIterator(*this);
        this->operator++();
        return newIterator;
    }

    /// @brief  Advance to the next (valid) item.
    inline bool next() { this->operator++(); return this->test(); }
    inline bool increment() { this->next(); return this->test(); }

    /// Return the coordinates of the item to which the value iterator is pointing.
    inline Coord getCoord() const { return mIter.getCoord(); }
    /// Return in @a xyz the coordinates of the item to which the value iterator is pointing.
    inline void getCoord(Coord& xyz) const { xyz = mIter.getCoord(); }

    /// Return the const index iterator
    inline const IndexIter& indexIter() const { return mIndexIter; }
    /// Return the const value iterator
    inline const ValueIterT& valueIter() const { return mIter; }

    /// @brief Equality operators
    bool operator==(const ValueIndexIter& other) const { return *mIndexIter == *other.mIndexIter; }
    bool operator!=(const ValueIndexIter& other) const { return !this->operator==(other); }

private:
    IndexIter mIndexIter;
    ValueIterT mIter;
    const typename ValueIterT::NodeType& mParent;
}; // ValueIndexIter


/// IndexIterTraits provides the following for iterators of the three value
/// types, i.e., for {Value}{On,Off,All}{CIter}:
/// - a begin(leaf) function that returns an index iterator or an index value
///   iterator for the leaf provided,
///   eg IndexIterTraits<Tree, Tree::LeafNodeType::ValueOn>::begin(leaf) returns
///   leaf.beginIndexOn()
/// - an Iterator typedef that aliases to the index iterator for this value type
template<typename TreeT, typename ValueT> struct IndexIterTraits;

template<typename TreeT>
struct IndexIterTraits<TreeT, typename TreeT::LeafNodeType::ValueAllCIter> {
    typedef IndexIter Iterator;
    static Iterator begin(const typename TreeT::LeafNodeType& leaf) {
        return Iterator(leaf.beginIndexAll());
    }
};

template<typename TreeT>
struct IndexIterTraits<TreeT, typename TreeT::LeafNodeType::ValueOnCIter> {
    typedef typename TreeT::LeafNodeType::IndexOnIter Iterator;
    static Iterator begin(const typename TreeT::LeafNodeType& leaf) {
        return Iterator(leaf.beginIndexOn());
    }
};

template<typename TreeT>
struct IndexIterTraits<TreeT, typename TreeT::LeafNodeType::ValueOffCIter> {
    typedef typename TreeT::LeafNodeType::IndexOffIter Iterator;
    static Iterator begin(const typename TreeT::LeafNodeType& leaf) {
        return Iterator(leaf.beginIndexOff());
    }
};


/// @brief A forward iterator over array indices with filtering
/// IteratorT can be either IndexIter or ValueIndexIter (or some custom index iterator)
/// FilterT should be a struct or class with a valid() method than can be evaluated per index
/// Here's a simple filter example that only accepts even indices:
///
/// struct EvenIndexFilter
/// {
///     bool valid(const Index32 offset) const {
///         return (offset % 2) == 0;
///     }
/// };
///
template <typename IteratorT, typename FilterT>
class FilterIndexIter
{
public:
    FilterIndexIter(const IteratorT& iterator, const FilterT& filter)
        : mIterator(iterator), mFilter(filter) { if (mIterator) { this->reset(*mIterator, mIterator.end()); } }
    FilterIndexIter(const FilterIndexIter& other)
        : mIterator(other.mIterator), mFilter(other.mFilter) { }

    Index32 end() const { return mIterator.end(); }

    /// @brief Reset the begining and end of the iterator.
    void reset(Index32 begin, Index32 end) {
        mIterator.reset(begin, end);
        while (mIterator.test() && !mFilter.template valid<IteratorT>(mIterator)) {
            ++mIterator;
        }
    }

    /// @brief  Returns the item to which this iterator is currently pointing.
    Index32 operator*() { return *mIterator; }
    Index32 operator*() const { return *mIterator; }

    /// @brief  Return @c true if this iterator is not yet exhausted.
    operator bool() const { return mIterator.test(); }
    bool test() const { return mIterator.test(); }

    /// @brief  Advance to the next (valid) item (prefix).
    FilterIndexIter& operator++() {
        while (true) {
            ++mIterator;
            if (!mIterator.test() || mFilter.template valid<IteratorT>(mIterator)) {
                break;
            }
        }
        return *this;
    }

    /// @brief  Advance to the next (valid) item (postfix).
    FilterIndexIter operator++(int /*dummy*/) {
        FilterIndexIter newIterator(*this);
        this->operator++();
        return newIterator;
    }

    /// @brief  Advance to the next (valid) item.
    bool next() { this->operator++(); return this->test(); }
    bool increment() { this->next(); return this->test(); }

    /// Return the const index iterator
    inline const IteratorT& indexIter() const { return mIterator; }
    /// Return the const filter
    inline const FilterT& filter() const { return mFilter; }

    /// @brief Equality operators
    bool operator==(const FilterIndexIter& other) const { return mIterator == other.mIterator; }
    bool operator!=(const FilterIndexIter& other) const { return !this->operator==(other); }

private:
    IteratorT mIterator;
    const FilterT mFilter;
}; // class FilterIndexIter


////////////////////////////////////////


template <typename IterT>
inline Index64 iterCount(const IterT& iter)
{
    Index64 size = 0;
    for (IterT newIter(iter); newIter; ++newIter, ++size) { }
    return size;
}


template <>
inline Index64 iterCount(const IndexIter& iter)
{
    return iter ? iter.end() - *iter : 0;
}


template <typename T>
inline Index64 iterCount(const ValueIndexIter<T>& iter)
{
    T newIter(iter.valueIter());
    Index64 size = 0;
    for ( ; newIter; ++newIter) {
        size += *newIter - (newIter.offset() == 0 ? Index32(0) : Index32(newIter.parent().getValue(newIter.offset() - 1)));
    }
    return size;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_INDEX_ITERATOR_HAS_BEEN_INCLUDED


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
