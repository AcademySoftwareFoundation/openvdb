// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/GridBuilder.h

    \author Ken Museth

    \date June 26, 2020

    \brief This file defines a minimum set of tree nodes and tools that
           can be used (instead of OpenVDB) to build nanovdb grids on the CPU.
*/

#ifndef NANOVDB_TOOLS_BUILD_GRIDBUILDER_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_BUILD_GRIDBUILDER_H_HAS_BEEN_INCLUDED

#include <iostream>

#include <map>
#include <limits>
#include <sstream> // for stringstream
#include <vector>
#include <cstring> // for memcpy
#include <mutex>
#include <array>
#include <atomic>

#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/Range.h>
#include <nanovdb/util/ForEach.h>

namespace nanovdb {

namespace tools::build {

// ----------------------------> Froward decelerations of random access methods <--------------------------------------

template <typename T> struct GetValue;
template <typename T> struct SetValue;
template <typename T> struct TouchLeaf;
template <typename T> struct GetState;
template <typename T> struct ProbeValue;

// ----------------------------> RootNode <--------------------------------------

template<typename ChildT>
struct RootNode
{
    using ValueType = typename ChildT::ValueType;
    using BuildType = typename ChildT::BuildType;
    using ChildNodeType = ChildT;
    using LeafNodeType = typename ChildT::LeafNodeType;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    struct Tile {
        Tile(ChildT* c = nullptr) : child(c) {}
        Tile(const ValueType& v, bool s) : child(nullptr), value(v), state(s) {}
        bool isChild() const { return child!=nullptr; }
        bool isValue() const { return child==nullptr; }
        bool isActive() const { return child==nullptr && state; }
        ChildT*   child;
        ValueType value;
        bool      state;
    };
    using MapT = std::map<Coord, Tile>;
    MapT      mTable;
    ValueType mBackground;

    Tile* probeTile(const Coord &ijk) {
        auto iter = mTable.find(CoordToKey(ijk));
        return iter == mTable.end() ? nullptr : &(iter->second);
    }

    const Tile* probeTile(const Coord &ijk) const {
        auto iter = mTable.find(CoordToKey(ijk));
        return iter == mTable.end() ? nullptr : &(iter->second);
    }

    class ChildIterator
    {
        const RootNode *mParent;
        typename MapT::const_iterator mIter;
    public:
        ChildIterator() : mParent(nullptr), mIter() {}
        ChildIterator(const RootNode *parent) : mParent(parent), mIter(parent->mTable.begin()) {
            while (mIter!=parent->mTable.end() && mIter->second.child==nullptr) ++mIter;
        }
        ChildIterator& operator=(const ChildIterator&) = default;
        ChildT& operator*() const {NANOVDB_ASSERT(*this); return *mIter->second.child;}
        ChildT* operator->() const {NANOVDB_ASSERT(*this); return mIter->second.child;}
        Coord getOrigin() const { NANOVDB_ASSERT(*this); return mIter->first;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mIter->first;}
        operator bool() const {return mParent && mIter!=mParent->mTable.end();}
        ChildIterator& operator++() {
            NANOVDB_ASSERT(mParent);
            ++mIter;
            while (mIter!=mParent->mTable.end() && mIter->second.child==nullptr) ++mIter;
            return *this;
        }
        ChildIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        uint32_t pos() const {
            NANOVDB_ASSERT(mParent);
            return uint32_t(std::distance(mParent->mTable.begin(), mIter));
        }
    }; // Member class ChildIterator

    ChildIterator  cbeginChild()  const {return ChildIterator(this);}
    ChildIterator cbeginChildOn() const {return ChildIterator(this);}// match openvdb

    class ValueIterator
    {
        const RootNode *mParent;
        typename MapT::const_iterator mIter;
    public:
        ValueIterator() : mParent(nullptr), mIter() {}
        ValueIterator(const RootNode *parent) : mParent(parent), mIter(parent->mTable.begin()) {
            while (mIter!=parent->mTable.end() && mIter->second.child!=nullptr) ++mIter;
        }
        ValueIterator& operator=(const ValueIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mIter->second.value;}
        bool isActive() const {NANOVDB_ASSERT(*this); return mIter->second.state;}
        Coord getOrigin() const { NANOVDB_ASSERT(*this); return mIter->first;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mIter->first;}
        operator bool() const {return mParent && mIter!=mParent->mTable.end();}
        ValueIterator& operator++() {
            NANOVDB_ASSERT(mParent);
            ++mIter;
            while (mIter!=mParent->mTable.end() && mIter->second.child!=nullptr) ++mIter;
            return *this;;
        }
        ValueIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        uint32_t pos() const {
            NANOVDB_ASSERT(mParent);
            return uint32_t(std::distance(mParent->mTable.begin(), mIter));
        }
    }; // Member class ValueIterator

    ValueIterator  beginValue()          {return ValueIterator(this);}
    ValueIterator cbeginValueAll() const {return ValueIterator(this);}

    class ValueOnIterator
    {
        const RootNode *mParent;
        typename MapT::const_iterator mIter;
    public:
        ValueOnIterator() : mParent(nullptr), mIter() {}
        ValueOnIterator(const RootNode *parent) : mParent(parent), mIter(parent->mTable.begin()) {
            while (mIter!=parent->mTable.end() && (mIter->second.child!=nullptr || !mIter->second.state)) ++mIter;
        }
        ValueOnIterator& operator=(const ValueOnIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mIter->second.value;}
        Coord getOrigin() const { NANOVDB_ASSERT(*this); return mIter->first;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mIter->first;}
        operator bool() const {return mParent && mIter!=mParent->mTable.end();}
        ValueOnIterator& operator++() {
            NANOVDB_ASSERT(mParent);
            ++mIter;
            while (mIter!=mParent->mTable.end() && (mIter->second.child!=nullptr || !mIter->second.state)) ++mIter;
            return *this;;
        }
        ValueOnIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        uint32_t pos() const {
            NANOVDB_ASSERT(mParent);
            return uint32_t(std::distance(mParent->mTable.begin(), mIter));
        }
    }; // Member class ValueOnIterator

    ValueOnIterator  beginValueOn()       {return ValueOnIterator(this);}
    ValueOnIterator cbeginValueOn() const {return ValueOnIterator(this);}

    class TileIterator
    {
        const RootNode *mParent;
        typename MapT::const_iterator mIter;
    public:
        TileIterator() : mParent(nullptr), mIter() {}
        TileIterator(const RootNode *parent) : mParent(parent), mIter(parent->mTable.begin()) {
            NANOVDB_ASSERT(mParent);
        }
        TileIterator& operator=(const TileIterator&) = default;
        const Tile& operator*() const {NANOVDB_ASSERT(*this); return mIter->second;}
        const Tile* operator->() const {NANOVDB_ASSERT(*this); return &(mIter->second);}
        Coord getOrigin() const { NANOVDB_ASSERT(*this); return mIter->first;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mIter->first;}
        operator bool() const {return mParent && mIter!=mParent->mTable.end();}
        const ChildT* probeChild(ValueType &value) {
            NANOVDB_ASSERT(*this);
            const ChildT *child = mIter->second.child;
            if (child==nullptr) value = mIter->second.value;
            return child;
        }
        bool isValueOn() const {return mIter->second.child==nullptr && mIter->second.state;}
        TileIterator& operator++() {
            NANOVDB_ASSERT(mParent);
            ++mIter;
            return *this;
        }
        TileIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
        uint32_t pos() const {
            NANOVDB_ASSERT(mParent);
            return uint32_t(std::distance(mParent->mTable.begin(), mIter));
        }
    }; // Member class TileIterator

    TileIterator  beginTile()           {return TileIterator(this);}
    TileIterator cbeginChildAll() const {return TileIterator(this);}

    //class DenseIterator : public TileIterator

    RootNode(const ValueType& background) : mBackground(background) {}
    RootNode(const RootNode&) = delete; // disallow copy-construction
    RootNode(RootNode&&) = default; // allow move construction
    RootNode& operator=(const RootNode&) = delete; // disallow copy assignment
    RootNode& operator=(RootNode&&) = default; // allow move assignment

    ~RootNode() { this->clear(); }

    uint32_t tileCount()    const { return uint32_t(mTable.size()); }
    uint32_t getTableSize() const { return uint32_t(mTable.size()); }// match openvdb
    const ValueType& background() const {return mBackground;}

    void nodeCount(std::array<size_t,3> &count) const
    {
        for (auto it = this->cbeginChild(); it; ++it) {
            count[ChildT::LEVEL] += 1;
            it->nodeCount(count);
        }
    }

    bool empty() const { return mTable.empty(); }

    void clear()
    {
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) delete iter->second.child;
        mTable.clear();
    }

    static Coord CoordToKey(const Coord& ijk) { return ijk & ~ChildT::MASK; }

#ifdef NANOVDB_NEW_ACCESSOR_METHODS
    template<typename OpT, typename... ArgsT>
    auto get(const Coord& ijk, ArgsT&&... args) const
    {
        if (const Tile *tile = this->probeTile(ijk)) {
            if (auto *child = tile->child) return child->template get<OpT>(ijk, args...);
            return OpT::get(*tile, args...);
        }
        return OpT::get(*this, args...);
    }
    template<typename OpT, typename... ArgsT>
    auto set(const Coord& ijk, ArgsT&&... args)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        NANOVDB_ASSERT(child);
        return child->template set<OpT>(ijk, args...);
    }
    template<typename OpT, typename AccT, typename... ArgsT>
    auto getAndCache(const Coord& ijk, const AccT& acc, ArgsT&&... args) const
    {
        if (const Tile *tile = this->probeTile(ijk)) {
            if (auto *child = tile->child) {
                acc.insert(ijk, child);
                return child->template get<OpT>(ijk, args...);
            }
            return OpT::get(*tile, args...);
        }
        return OpT::get(*this, args...);
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    auto setAndCache(const Coord& ijk, const AccT& acc, ArgsT&&... args)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        NANOVDB_ASSERT(child);
        acc.insert(ijk, child);
        return child->template setAndCache<OpT>(ijk, acc, args...);
    }
    ValueType getValue(const Coord& ijk) const {return this->template get<GetValue<BuildType>>(ijk);}
    ValueType getValue(int i, int j, int k) const {return this->template get<GetValue<BuildType>>(Coord(i,j,k));}
    ValueType operator()(const Coord& ijk) const {return this->template get<GetValue<BuildType>>(ijk);}
    ValueType operator()(int i, int j, int k) const {return this->template get<GetValue<BuildType>>(Coord(i,j,k));}
    void setValue(const Coord& ijk, const ValueType& value) {this->template set<SetValue<BuildType>>(ijk, value);}
    bool probeValue(const Coord& ijk, ValueType& value) const {return this->template get<ProbeValue<BuildType>>(ijk, value);}
    bool isActive(const Coord& ijk) const {return this->template get<GetState<BuildType>>(ijk);}
#else
    ValueType getValue(const Coord& ijk) const
    {
#if 1
        if (auto *tile = this->probeTile(ijk)) return tile->child ? tile->child->getValue(ijk) : tile->value;
        return mBackground;
#else
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end()) {
            return mBackground;
        } else if (iter->second.child) {
            return iter->second.child->getValue(ijk);
        } else {
            return iter->second.value;
        }
#endif
    }
    ValueType getValue(int i, int j, int k) const {return this->getValue(Coord(i,j,k));}

    void setValue(const Coord& ijk, const ValueType& value)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        NANOVDB_ASSERT(child);
        child->setValue(ijk, value);
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end())
            return false;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->isActiveAndCache(ijk, acc);
        }
        return iter->second.state;
    }

    template<typename AccT>
    ValueType getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        auto iter = mTable.find(CoordToKey(ijk));
        if (iter == mTable.end())
            return mBackground;
        if (iter->second.child) {
            acc.insert(ijk, iter->second.child);
            return iter->second.child->getValueAndCache(ijk, acc);
        }
        return iter->second.value;
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueType& value, AccT& acc)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        NANOVDB_ASSERT(child);
        acc.insert(ijk, child);
        child->setValueAndCache(ijk, value, acc);
    }
    template<typename AccT>
    void setValueOnAndCache(const Coord& ijk, AccT& acc)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        NANOVDB_ASSERT(child);
        acc.insert(ijk, child);
        child->setValueOnAndCache(ijk, acc);
    }
    template<typename AccT>
    void touchLeafAndCache(const Coord &ijk, AccT& acc)
    {
        ChildT* child = nullptr;
        const Coord key = CoordToKey(ijk);
        auto iter = mTable.find(key);
        if (iter == mTable.end()) {
            child = new ChildT(ijk, mBackground, false);
            mTable[key] = Tile(child);
        } else if (iter->second.child != nullptr) {
            child = iter->second.child;
        } else {
            child = new ChildT(ijk, iter->second.value, iter->second.state);
            iter->second.child = child;
        }
        acc.insert(ijk, child);
        child->touchLeafAndCache(ijk, acc);
    }
#endif// NANOVDB_NEW_ACCESSOR_METHODS

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(util::is_same<ValueType, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        uint32_t sum = 0;
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr) continue; // skip tiles
            if constexpr(util::is_same<NodeT, ChildT>::value) { //resolved at compile-time
                ++sum;
            } else {
                sum += iter->second.child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template<typename NodeT>
    void getNodes(std::vector<NodeT*>& array)
    {
        static_assert(util::is_same<ValueType, typename NodeT::ValueType>::value, "Root::getNodes: Invalid type");
        static_assert(NodeT::LEVEL < LEVEL, "Root::getNodes: LEVEL error");
        for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
            if (iter->second.child == nullptr)
                continue;
            if constexpr(util::is_same<NodeT, ChildT>::value) { //resolved at compile-time
                array.push_back(reinterpret_cast<NodeT*>(iter->second.child));
            } else {
                iter->second.child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        NANOVDB_ASSERT(child);
        const Coord key = CoordToKey(child->mOrigin);
        auto iter = mTable.find(key);
        if (iter != mTable.end() && iter->second.child != nullptr) { // existing child node
            delete iter->second.child;
            iter->second.child = child;
        } else {
            mTable[key] = Tile(child);
        }
        child = nullptr;
    }

    /// @brief Add a tile containing voxel (i, j, k) at the specified tree level,
    /// creating a new branch if necessary.  Delete any existing lower-level nodes
    /// that contain (x, y, z).
    /// @tparam level tree level at which the tile is inserted. Must be 1, 2 or 3.
    /// @param ijk Index coordinate that map to the tile being inserted
    /// @param value Value of the tile
    /// @param state Binary state of the tile
    template <uint32_t level>
    void addTile(const Coord& ijk, const ValueType& value, bool state)
    {
        static_assert(level > 0 && level <= LEVEL, "invalid template value of level");
        const Coord key = CoordToKey(ijk);
        auto        iter = mTable.find(key);
        if constexpr(level == LEVEL) {
            if (iter == mTable.end()) {
                mTable[key] = Tile(value, state);
            } else if (iter->second.child == nullptr) {
                iter->second.value = value;
                iter->second.state = state;
            } else {
                delete iter->second.child;
                iter->second.child = nullptr;
                iter->second.value = value;
                iter->second.state = state;
            }
        } else if constexpr(level < LEVEL) {
            ChildT* child = nullptr;
            if (iter == mTable.end()) {
                child = new ChildT(ijk, mBackground, false);
                mTable[key] = Tile(child);
            } else if (iter->second.child != nullptr) {
                child = iter->second.child;
            } else {
                child = new ChildT(ijk, iter->second.value, iter->second.state);
                iter->second.child = child;
            }
            child->template addTile<level>(ijk, value, state);
        }
    }

    template<typename NodeT>
    void addNode(NodeT*& node)
    {
        if constexpr(util::is_same<NodeT, ChildT>::value) { //resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else {
            ChildT*     child = nullptr;
            const Coord key = CoordToKey(node->mOrigin);
            auto        iter = mTable.find(key);
            if (iter == mTable.end()) {
                child = new ChildT(node->mOrigin, mBackground, false);
                mTable[key] = Tile(child);
            } else if (iter->second.child != nullptr) {
                child = iter->second.child;
            } else {
                child = new ChildT(node->mOrigin, iter->second.value, iter->second.state);
                iter->second.child = child;
            }
            child->addNode(node);
        }
    }

    void merge(RootNode &other)
    {
        for (auto iter1 = other.mTable.begin(); iter1 != other.mTable.end(); ++iter1) {
            if (iter1->second.child == nullptr) continue;// ignore input tiles
            auto iter2 = mTable.find(iter1->first);
            if (iter2 == mTable.end() || iter2->second.child == nullptr) {
                mTable[iter1->first] = Tile(iter1->second.child);
                iter1->second.child = nullptr;
            } else {
                iter2->second.child->merge(*iter1->second.child);
            }
        }
        other.clear();
    }

    template<typename T>
    typename util::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);

}; // tools::build::RootNode

//================================================================================================

template<typename ChildT>
template<typename T>
inline typename util::enable_if<std::is_floating_point<T>::value>::type
RootNode<ChildT>::signedFloodFill(T outside)
{
    std::map<Coord, ChildT*> nodeKeys;
    for (auto iter = mTable.begin(); iter != mTable.end(); ++iter) {
        if (iter->second.child == nullptr)
            continue;
        nodeKeys.insert(std::pair<Coord, ChildT*>(iter->first, iter->second.child));
    }

    // We employ a simple z-scanline algorithm that inserts inactive tiles with
    // the inside value if they are sandwiched between inside child nodes only!
    auto b = nodeKeys.begin(), e = nodeKeys.end();
    if (b == e)
        return;
    for (auto a = b++; b != e; ++a, ++b) {
        Coord d = b->first - a->first; // delta of neighboring coordinates
        if (d[0] != 0 || d[1] != 0 || d[2] == int(ChildT::DIM))
            continue; // not same z-scanline or neighbors
        const ValueType fill[] = {a->second->getLastValue(), b->second->getFirstValue()};
        if (!(fill[0] < 0) || !(fill[1] < 0))
            continue; // scanline isn't inside
        Coord c = a->first + Coord(0u, 0u, ChildT::DIM);
        for (; c[2] != b->first[2]; c[2] += ChildT::DIM) {
            const Coord key = RootNode<ChildT>::CoordToKey(c);
            mTable[key] = typename RootNode<ChildT>::Tile(-outside, false); // inactive tile
        }
    }
} // tools::build::RootNode::signedFloodFill

// ----------------------------> InternalNode <--------------------------------------

template<typename ChildT>
struct InternalNode
{
    using ValueType = typename ChildT::ValueType;
    using BuildType = typename ChildT::BuildType;
    using ChildNodeType = ChildT;
    using LeafNodeType = typename ChildT::LeafNodeType;
    static constexpr uint32_t LOG2DIM = ChildT::LOG2DIM + 1;
    static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
    static constexpr uint32_t MASK = DIM - 1;
    static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using MaskT = Mask<LOG2DIM>;
    template<bool On>
    using MaskIterT = typename MaskT::template Iterator<On>;
    using NanoNodeT = typename NanoNode<BuildType, LEVEL>::Type;

    struct Tile {
        Tile(ChildT* c = nullptr) : child(c) {}
        Tile(const ValueType& v) : value(v) {}
        union{
            ChildT*   child;
            ValueType value;
        };
    };
    Coord      mOrigin;
    MaskT      mValueMask;
    MaskT      mChildMask;
    Tile       mTable[SIZE];

    union {
        NanoNodeT *mDstNode;
        uint64_t   mDstOffset;
    };

    /// @brief Visits child nodes of this node only
    class ChildIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const InternalNode *mParent;
    public:
        ChildIterator() : BaseT(), mParent(nullptr) {}
        ChildIterator(const InternalNode* parent) : BaseT(parent->mChildMask.beginOn()), mParent(parent) {}
        ChildIterator& operator=(const ChildIterator&) = default;
        const ChildT& operator*() const {NANOVDB_ASSERT(*this); return *mParent->mTable[BaseT::pos()].child;}
        const ChildT* operator->() const {NANOVDB_ASSERT(*this); return mParent->mTable[BaseT::pos()].child;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return (*this)->origin();}
    }; // Member class ChildIterator

    ChildIterator  beginChild()         {return ChildIterator(this);}
    ChildIterator cbeginChildOn() const {return ChildIterator(this);}// match openvdb

     /// @brief Visits all tile values in this node, i.e. both inactive and active tiles
    class ValueIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const InternalNode *mParent;
    public:
        ValueIterator() : BaseT(), mParent(nullptr) {}
        ValueIterator(const InternalNode* parent) :  BaseT(parent->mChildMask.beginOff()), mParent(parent) {}
        ValueIterator& operator=(const ValueIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mParent->mTable[BaseT::pos()].value;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
        bool isActive() const { NANOVDB_ASSERT(*this); return mParent->mValueMask.isOn(BaseT::pos());}
    }; // Member class ValueIterator

    ValueIterator  beginValue()          {return ValueIterator(this);}
    ValueIterator cbeginValueAll() const {return ValueIterator(this);}

    /// @brief Visits active tile values of this node only
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const InternalNode *mParent;
    public:
        ValueOnIterator() : BaseT(), mParent(nullptr) {}
        ValueOnIterator(const InternalNode* parent) :  BaseT(parent->mValueMask.beginOn()), mParent(parent) {}
        ValueOnIterator& operator=(const ValueOnIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mParent->mTable[BaseT::pos()].value;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOnIterator

    ValueOnIterator  beginValueOn()       {return ValueOnIterator(this);}
    ValueOnIterator cbeginValueOn() const {return ValueOnIterator(this);}

    /// @brief Visits all tile values and child nodes of this node
    class DenseIterator : public MaskT::DenseIterator
    {
        using BaseT = typename MaskT::DenseIterator;
        const InternalNode *mParent;
    public:
        DenseIterator() : BaseT(), mParent(nullptr) {}
        DenseIterator(const InternalNode* parent) :  BaseT(0), mParent(parent) {}
        DenseIterator& operator=(const DenseIterator&) = default;
        ChildT* probeChild(ValueType& value) const
        {
            NANOVDB_ASSERT(mParent && bool(*this));
            ChildT *child = nullptr;
            if (mParent->mChildMask.isOn(BaseT::pos())) {
                child = mParent->mTable[BaseT::pos()].child;
            } else {
                value = mParent->mTable[BaseT::pos()].value;
            }
            return child;
        }
        Coord getCoord() const { NANOVDB_ASSERT(mParent && bool(*this)); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class DenseIterator

    DenseIterator     beginDense()       {return DenseIterator(this);}
    DenseIterator cbeginChildAll() const {return DenseIterator(this);}// matches openvdb

    InternalNode(const Coord& origin, const ValueType& value, bool state)
        : mOrigin(origin & ~MASK)
        , mValueMask(state)
        , mChildMask()
        , mDstOffset(0)
    {
        for (uint32_t i = 0; i < SIZE; ++i) mTable[i].value = value;
    }
    InternalNode(const InternalNode&) = delete; // disallow copy-construction
    InternalNode(InternalNode&&) = delete; // disallow move construction
    InternalNode& operator=(const InternalNode&) = delete; // disallow copy assignment
    InternalNode& operator=(InternalNode&&) = delete; // disallow move assignment
    ~InternalNode()
    {
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            delete mTable[*iter].child;
        }
    }
    const MaskT& getValueMask() const {return mValueMask;}
    const MaskT& valueMask() const {return mValueMask;}
    const MaskT& getChildMask() const {return mChildMask;}
    const MaskT& childMask() const {return mChildMask;}
    const Coord& origin() const {return mOrigin;}

    void nodeCount(std::array<size_t,3> &count) const
    {
        count[ChildT::LEVEL] += mChildMask.countOn();
        if constexpr(ChildT::LEVEL>0) {
            for (auto it = const_cast<InternalNode*>(this)->beginChild(); it; ++it) it->nodeCount(count);
        }
    }

    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return (((ijk[0] & int32_t(MASK)) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
               (((ijk[1] & int32_t(MASK)) >> ChildT::TOTAL) << (LOG2DIM)) +
                ((ijk[2] & int32_t(MASK)) >> ChildT::TOTAL);
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & ((1 << LOG2DIM) - 1));
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk <<= ChildT::TOTAL;
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = InternalNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    ValueType getFirstValue() const { return mChildMask.isOn(0) ? mTable[0].child->getFirstValue() : mTable[0].value; }
    ValueType getLastValue() const { return mChildMask.isOn(SIZE - 1) ? mTable[SIZE - 1].child->getLastValue() : mTable[SIZE - 1].value; }

    template<typename OpT, typename... ArgsT>
    auto get(const Coord& ijk, ArgsT&&... args) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) return mTable[n].child->template get<OpT>(ijk, args...);
        return OpT::get(*this, n, args...);
    }

    template<typename OpT, typename... ArgsT>
    auto set(const Coord& ijk, ArgsT&&... args)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT* child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        NANOVDB_ASSERT(child);
        return child->template set<OpT>(ijk, args...);
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    auto getAndCache(const Coord& ijk, const AccT& acc, ArgsT&&... args) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOff(n)) return OpT::get(*this, n, args...);
        ChildT* child = mTable[n].child;
        acc.insert(ijk, child);
        if constexpr(ChildT::LEVEL == 0) {
            return child->template get<OpT>(ijk, args...);
        } else {
            return child->template getAndCache<OpT>(ijk, acc, args...);
        }
    }

    template<typename OpT, typename AccT, typename... ArgsT>
    auto setAndCache(const Coord& ijk, const AccT& acc, ArgsT&&... args)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT* child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        NANOVDB_ASSERT(child);
        acc.insert(ijk, child);
        if constexpr(ChildT::LEVEL == 0) {
            return child->template set<OpT>(ijk, args...);
        } else {
            return child->template setAndCache<OpT>(ijk, acc, args...);
        }
    }

#ifdef NANOVDB_NEW_ACCESSOR_METHODS
    ValueType getValue(const Coord& ijk) const {return this->template get<GetValue<BuildType>>(ijk);}
    LeafNodeType& setValue(const Coord& ijk, const ValueType& value){return this->template set<SetValue<BuildType>>(ijk, value);}
#else
    ValueType getValue(const Coord& ijk) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            return mTable[n].child->getValue(ijk);
        }
        return mTable[n].value;
    }
    void setValue(const Coord& ijk, const ValueType& value)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT*        child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        child->setValue(ijk, value);
    }

    template<typename AccT>
    ValueType getValueAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->getValueAndCache(ijk, acc);
        }
        return mTable[n].value;
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueType& value, AccT& acc)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT*        child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        acc.insert(ijk, child);
        child->setValueAndCache(ijk, value, acc);
    }

    template<typename AccT>
    void setValueOnAndCache(const Coord& ijk, AccT& acc)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT*        child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        acc.insert(ijk, child);
        child->setValueOnAndCache(ijk, acc);
    }

    template<typename AccT>
    void touchLeafAndCache(const Coord &ijk, AccT& acc)
    {
        const uint32_t n = CoordToOffset(ijk);
        ChildT* child = nullptr;
        if (mChildMask.isOn(n)) {
            child = mTable[n].child;
        } else {
            child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
            mTable[n].child = child;
            mChildMask.setOn(n);
        }
        acc.insert(ijk, child);
        if constexpr(LEVEL>1) child->touchLeafAndCache(ijk, acc);
    }
    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, AccT& acc) const
    {
        const uint32_t n = CoordToOffset(ijk);
        if (mChildMask.isOn(n)) {
            acc.insert(ijk, const_cast<ChildT*>(mTable[n].child));
            return mTable[n].child->isActiveAndCache(ijk, acc);
        }
        return mValueMask.isOn(n);
    }
#endif

    template<typename NodeT>
    uint32_t nodeCount() const
    {
        static_assert(util::is_same<ValueType, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        NANOVDB_ASSERT(NodeT::LEVEL < LEVEL);
        uint32_t sum = 0;
        if constexpr(util::is_same<NodeT, ChildT>::value) { // resolved at compile-time
            sum += mChildMask.countOn();
        } else if constexpr(LEVEL>1) {
            for (auto iter = mChildMask.beginOn(); iter; ++iter) {
                sum += mTable[*iter].child->template nodeCount<NodeT>();
            }
        }
        return sum;
    }

    template<typename NodeT>
    void getNodes(std::vector<NodeT*>& array)
    {
        static_assert(util::is_same<ValueType, typename NodeT::ValueType>::value, "Node::getNodes: Invalid type");
        NANOVDB_ASSERT(NodeT::LEVEL < LEVEL);
        for (auto iter = mChildMask.beginOn(); iter; ++iter) {
            if constexpr(util::is_same<NodeT, ChildT>::value) { // resolved at compile-time
                array.push_back(reinterpret_cast<NodeT*>(mTable[*iter].child));
            } else if constexpr(LEVEL>1) {
                mTable[*iter].child->getNodes(array);
            }
        }
    }

    void addChild(ChildT*& child)
    {
        NANOVDB_ASSERT(child && (child->mOrigin & ~MASK) == this->mOrigin);
        const uint32_t n = CoordToOffset(child->mOrigin);
        if (mChildMask.isOn(n)) {
            delete mTable[n].child;
        } else {
            mChildMask.setOn(n);
        }
        mTable[n].child = child;
        child = nullptr;
    }

    /// @brief Add a tile containing voxel (i, j, k) at the specified tree level,
    /// creating a new branch if necessary.  Delete any existing lower-level nodes
    /// that contain (x, y, z).
    /// @tparam level tree level at which the tile is inserted. Must be 1 or 2.
    /// @param ijk Index coordinate that map to the tile being inserted
    /// @param value Value of the tile
    /// @param state Binary state of the tile
    template <uint32_t level>
    void addTile(const Coord& ijk, const ValueType& value, bool state)
    {
        static_assert(level > 0 && level <= LEVEL, "invalid template value of level");
        const uint32_t n = CoordToOffset(ijk);
        if constexpr(level == LEVEL) {
            if (mChildMask.isOn(n)) {
                delete mTable[n].child;
                mTable[n] = Tile(value);
            } else {
                mValueMask.set(n, state);
                mTable[n].value = value;
            }
        } else if constexpr(level < LEVEL) {
            ChildT* child = nullptr;
            if (mChildMask.isOn(n)) {
                child = mTable[n].child;
            } else {
                child = new ChildT(ijk, value, state);
                mTable[n].child = child;
                mChildMask.setOn(n);
            }
            child->template addTile<level>(ijk, value, state);
        }
    }

    template<typename NodeT>
    void addNode(NodeT*& node)
    {
        if constexpr(util::is_same<NodeT, ChildT>::value) { //resolved at compile-time
            this->addChild(reinterpret_cast<ChildT*&>(node));
        } else if constexpr(LEVEL>1) {
            const uint32_t n = CoordToOffset(node->mOrigin);
            ChildT*        child = nullptr;
            if (mChildMask.isOn(n)) {
                child = mTable[n].child;
            } else {
                child = new ChildT(node->mOrigin, mTable[n].value, mValueMask.isOn(n));
                mTable[n].child = child;
                mChildMask.setOn(n);
            }
            child->addNode(node);
        }
    }

    void merge(InternalNode &other)
    {
        for (auto iter = other.mChildMask.beginOn(); iter; ++iter) {
            const uint32_t n = *iter;
            if (mChildMask.isOn(n)) {
                mTable[n].child->merge(*other.mTable[n].child);
            } else {
                mTable[n].child = other.mTable[n].child;
                other.mChildMask.setOff(n);
                mChildMask.setOn(n);
            }
        }
    }

    template<typename T>
    typename util::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);

}; // tools::build::InternalNode

//================================================================================================

template<typename ChildT>
template<typename T>
inline typename util::enable_if<std::is_floating_point<T>::value>::type
InternalNode<ChildT>::signedFloodFill(T outside)
{
    const uint32_t first = *mChildMask.beginOn();
    if (first < NUM_VALUES) {
        bool xInside = mTable[first].child->getFirstValue() < 0;
        bool yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != (1 << LOG2DIM); ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM); // offset for block(x, 0, 0)
            if (mChildMask.isOn(x00)) {
                xInside = mTable[x00].child->getLastValue() < 0;
            }
            yInside = xInside;
            for (uint32_t y = 0; y != (1u << LOG2DIM); ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM); // offset for block(x, y, 0)
                if (mChildMask.isOn(xy0))
                    yInside = mTable[xy0].child->getLastValue() < 0;
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // offset for block(x, y, z)
                    if (mChildMask.isOn(xyz)) {
                        zInside = mTable[xyz].child->getLastValue() < 0;
                    } else {
                        mTable[xyz].value = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
} // tools::build::InternalNode::signedFloodFill

// ----------------------------> LeafNode <--------------------------------------

template<typename BuildT>
struct LeafNode
{
    using BuildType = BuildT;
    using ValueType = typename BuildToValueMap<BuildT>::type;
    using LeafNodeType = LeafNode<BuildT>;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = DIM - 1; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    template<bool ON>
    using MaskIterT = typename Mask<LOG2DIM>::template Iterator<ON>;
    using NanoLeafT = typename NanoNode<BuildT, 0>::Type;

    Coord         mOrigin;
    Mask<LOG2DIM> mValueMask;
    ValueType     mValues[SIZE];
    union {
        NanoLeafT *mDstNode;
        uint64_t   mDstOffset;
    };

    /// @brief Visits all active values in a leaf node
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const LeafNode *mParent;
    public:
        ValueOnIterator() : BaseT(), mParent(nullptr) {}
        ValueOnIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOn()), mParent(parent) {}
        ValueOnIterator& operator=(const ValueOnIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mParent->mValues[BaseT::pos()];}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOnIterator

    ValueOnIterator  beginValueOn()       {return ValueOnIterator(this);}
    ValueOnIterator cbeginValueOn() const {return ValueOnIterator(this);}

    /// @brief Visits all inactive values in a leaf node
    class ValueOffIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const LeafNode *mParent;
    public:
        ValueOffIterator() : BaseT(), mParent(nullptr) {}
        ValueOffIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOff()), mParent(parent) {}
        ValueOffIterator& operator=(const ValueOffIterator&) = default;
        ValueType operator*() const {NANOVDB_ASSERT(*this); return mParent->mValues[BaseT::pos()];}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOffIterator

    ValueOffIterator  beginValueOff()       {return ValueOffIterator(this);}
    ValueOffIterator cbeginValueOff() const {return ValueOffIterator(this);}

    /// @brief Visits all values in a leaf node, i.e. both active and inactive values
    class ValueIterator
    {
        const LeafNode *mParent;
        uint32_t mPos;
    public:
        ValueIterator() : mParent(nullptr), mPos(1u << 3 * LOG2DIM) {}
        ValueIterator(const LeafNode* parent) :  mParent(parent), mPos(0) {NANOVDB_ASSERT(parent);}
        ValueIterator& operator=(const ValueIterator&) = default;
        ValueType operator*() const { NANOVDB_ASSERT(*this); return mParent->mValues[mPos];}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(mPos);}
        bool isActive() const { NANOVDB_ASSERT(*this); return mParent->isActive(mPos);}
        operator bool() const {return mPos < SIZE;}
        ValueIterator& operator++() {++mPos; return *this;}
        ValueIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
    }; // Member class ValueIterator

    ValueIterator  beginValue()          {return ValueIterator(this);}
    ValueIterator cbeginValueAll() const {return ValueIterator(this);}

    LeafNode(const Coord& ijk, const ValueType& value, bool state)
        : mOrigin(ijk & ~MASK)
        , mValueMask(state) //invalid
        , mDstOffset(0)
    {
        ValueType*  target = mValues;
        uint32_t n = SIZE;
        while (n--) {
            *target++ = value;
        }
    }
    LeafNode(const LeafNode&) = delete; // disallow copy-construction
    LeafNode(LeafNode&&) = delete; // disallow move construction
    LeafNode& operator=(const LeafNode&) = delete; // disallow copy assignment
    LeafNode& operator=(LeafNode&&) = delete; // disallow move assignment
    ~LeafNode() = default;

    const Mask<LOG2DIM>& getValueMask() const {return mValueMask;}
    const Mask<LOG2DIM>& valueMask() const {return mValueMask;}
    const Coord& origin() const {return mOrigin;}

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & int32_t(MASK)) << (2 * LOG2DIM)) +
               ((ijk[1] & int32_t(MASK)) << LOG2DIM) +
                (ijk[2] & int32_t(MASK));
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const int32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & int32_t(MASK));
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = LeafNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    ValueType getFirstValue() const { return mValues[0]; }
    ValueType getLastValue() const { return mValues[SIZE - 1]; }
    const ValueType& getValue(uint32_t i) const {return mValues[i];}
    const ValueType& getValue(const Coord& ijk) const {return mValues[CoordToOffset(ijk)];}

    template<typename OpT, typename... ArgsT>
    auto get(const Coord& ijk, ArgsT&&... args) const {return OpT::get(*this, CoordToOffset(ijk), args...);}

    template<typename OpT, typename... ArgsT>
    auto set(const Coord& ijk, ArgsT&&... args) {return OpT::set(*this, CoordToOffset(ijk), args...);}

#ifndef NANOVDB_NEW_ACCESSOR_METHODS
    template<typename AccT>
    const ValueType& getValueAndCache(const Coord& ijk, const AccT&) const
    {
        return mValues[CoordToOffset(ijk)];
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, const ValueType& value, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues[n] = value;
    }

    template<typename AccT>
    void setValueOnAndCache(const Coord& ijk, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, const AccT&) const
    {
        return mValueMask.isOn(CoordToOffset(ijk));
    }
#endif

    void setValue(uint32_t n, const ValueType& value)
    {
        mValueMask.setOn(n);
        mValues[n] = value;
    }
    void setValue(const Coord& ijk, const ValueType& value){this->setValue(CoordToOffset(ijk), value);}

    void merge(LeafNode &other)
    {
        other.mValueMask -= mValueMask;
        for (auto iter = other.mValueMask.beginOn(); iter; ++iter) {
            const uint32_t n = *iter;
            mValues[n] = other.mValues[n];
        }
        mValueMask |= other.mValueMask;
    }

    template<typename T>
    typename util::enable_if<std::is_floating_point<T>::value>::type
    signedFloodFill(T outside);

}; // tools::build::LeafNode<T>

//================================================================================================

template <>
struct LeafNode<ValueMask>
{
    using ValueType = bool;
    using BuildType = ValueMask;
    using LeafNodeType = LeafNode<BuildType>;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = DIM - 1; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    template<bool ON>
    using MaskIterT = typename Mask<LOG2DIM>::template Iterator<ON>;
    using NanoLeafT = typename NanoNode<BuildType, 0>::Type;

    Coord         mOrigin;
    Mask<LOG2DIM> mValueMask;
    union {
        NanoLeafT *mDstNode;
        uint64_t   mDstOffset;
    };

    /// @brief Visits all active values in a leaf node
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const LeafNode *mParent;
    public:
        ValueOnIterator() : BaseT(), mParent(nullptr) {}
        ValueOnIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOn()), mParent(parent) {}
        ValueOnIterator& operator=(const ValueOnIterator&) = default;
        bool operator*() const {NANOVDB_ASSERT(*this); return true;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOnIterator

    ValueOnIterator  beginValueOn()       {return ValueOnIterator(this);}
    ValueOnIterator cbeginValueOn() const {return ValueOnIterator(this);}

    /// @brief Visits all inactive values in a leaf node
    class ValueOffIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const LeafNode *mParent;
    public:
        ValueOffIterator() : BaseT(), mParent(nullptr) {}
        ValueOffIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOff()), mParent(parent) {}
        ValueOffIterator& operator=(const ValueOffIterator&) = default;
        bool operator*() const {NANOVDB_ASSERT(*this); return false;}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOffIterator

    ValueOffIterator  beginValueOff()       {return ValueOffIterator(this);}
    ValueOffIterator cbeginValueOff() const {return ValueOffIterator(this);}

    /// @brief Visits all values in a leaf node, i.e. both active and inactive values
    class ValueIterator
    {
        const LeafNode *mParent;
        uint32_t mPos;
    public:
        ValueIterator() : mParent(nullptr), mPos(1u << 3 * LOG2DIM) {}
        ValueIterator(const LeafNode* parent) :  mParent(parent), mPos(0) {NANOVDB_ASSERT(parent);}
        ValueIterator& operator=(const ValueIterator&) = default;
        bool operator*() const { NANOVDB_ASSERT(*this); return mParent->mValueMask.isOn(mPos);}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(mPos);}
        bool isActive() const { NANOVDB_ASSERT(*this); return mParent->mValueMask.isOn(mPos);}
        operator bool() const {return mPos < SIZE;}
        ValueIterator& operator++() {++mPos; return *this;}
        ValueIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
    }; // Member class ValueIterator

    ValueIterator  beginValue()          {return ValueIterator(this);}
    ValueIterator cbeginValueAll() const {return ValueIterator(this);}

    LeafNode(const Coord& ijk, const ValueType&, bool state)
        : mOrigin(ijk & ~MASK)
        , mValueMask(state) //invalid
        , mDstOffset(0)
    {
    }
    LeafNode(const LeafNode&) = delete; // disallow copy-construction
    LeafNode(LeafNode&&) = delete; // disallow move construction
    LeafNode& operator=(const LeafNode&) = delete; // disallow copy assignment
    LeafNode& operator=(LeafNode&&) = delete; // disallow move assignment
    ~LeafNode() = default;

    const Mask<LOG2DIM>& valueMask() const {return mValueMask;}
    const Mask<LOG2DIM>& getValueMask() const {return mValueMask;}
    const Coord& origin() const {return mOrigin;}

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & int32_t(MASK)) << (2 * LOG2DIM)) +
               ((ijk[1] & int32_t(MASK)) <<       LOG2DIM) +
                (ijk[2] & int32_t(MASK));
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const int32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & int32_t(MASK));
    }

    void localToGlobalCoord(Coord& ijk) const {ijk += mOrigin;}

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = LeafNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }

    bool getFirstValue() const { return mValueMask.isOn(0); }
    bool getLastValue() const { return mValueMask.isOn(SIZE - 1); }
    bool getValue(uint32_t i) const {return mValueMask.isOn(i);}
    bool getValue(const Coord& ijk) const {return mValueMask.isOn(CoordToOffset(ijk));}

    template<typename OpT, typename... ArgsT>
    auto get(const Coord& ijk, ArgsT&&... args) const {return OpT::get(*this, CoordToOffset(ijk), args...);}

    template<typename OpT, typename... ArgsT>
    auto set(const Coord& ijk, ArgsT&&... args) {return OpT::set(*this, CoordToOffset(ijk), args...);}

#ifndef NANOVDB_NEW_ACCESSOR_METHODS
    template<typename AccT>
    bool getValueAndCache(const Coord& ijk, const AccT&) const
    {
        return mValueMask.isOn(CoordToOffset(ijk));
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, bool, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
    }

    template<typename AccT>
    void setValueOnAndCache(const Coord& ijk, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
    }

    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, const AccT&) const
    {
        return mValueMask.isOn(CoordToOffset(ijk));
    }
#endif

    void setValue(uint32_t n, bool) {mValueMask.setOn(n);}
    void setValue(const Coord& ijk) {mValueMask.setOn(CoordToOffset(ijk));}

    void merge(LeafNode &other)
    {
        mValueMask |= other.mValueMask;
    }

}; // tools::build::LeafNode<ValueMask>

//================================================================================================

template <>
struct LeafNode<bool>
{
    using ValueType = bool;
    using BuildType = ValueMask;
    using LeafNodeType = LeafNode<BuildType>;
    static constexpr uint32_t LOG2DIM = 3;
    static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
    static constexpr uint32_t DIM = 1u << TOTAL;
    static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
    static constexpr uint32_t MASK = DIM - 1; // mask for bit operations
    static constexpr uint32_t LEVEL = 0; // level 0 = leaf
    static constexpr uint64_t NUM_VALUES = uint64_t(1) << (3 * TOTAL); // total voxel count represented by this node
    using NodeMaskType = Mask<LOG2DIM>;
    template<bool ON>
    using MaskIterT = typename Mask<LOG2DIM>::template Iterator<ON>;
    using NanoLeafT = typename NanoNode<BuildType, 0>::Type;

    Coord         mOrigin;
    Mask<LOG2DIM> mValueMask, mValues;
    union {
        NanoLeafT *mDstNode;
        uint64_t   mDstOffset;
    };

    /// @brief Visits all active values in a leaf node
    class ValueOnIterator : public MaskIterT<true>
    {
        using BaseT = MaskIterT<true>;
        const LeafNode *mParent;
    public:
        ValueOnIterator() : BaseT(), mParent(nullptr) {}
        ValueOnIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOn()), mParent(parent) {}
        ValueOnIterator& operator=(const ValueOnIterator&) = default;
        bool operator*() const {NANOVDB_ASSERT(*this); return mParent->mValues.isOn(BaseT::pos());}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOnIterator

    ValueOnIterator  beginValueOn()       {return ValueOnIterator(this);}
    ValueOnIterator cbeginValueOn() const {return ValueOnIterator(this);}

    /// @brief Visits all inactive values in a leaf node
    class ValueOffIterator : public MaskIterT<false>
    {
        using BaseT = MaskIterT<false>;
        const LeafNode *mParent;
    public:
        ValueOffIterator() : BaseT(), mParent(nullptr) {}
        ValueOffIterator(const LeafNode* parent) :  BaseT(parent->mValueMask.beginOff()), mParent(parent) {}
        ValueOffIterator& operator=(const ValueOffIterator&) = default;
        bool operator*() const {NANOVDB_ASSERT(*this); return mParent->mValues.isOn(BaseT::pos());}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(BaseT::pos());}
    }; // Member class ValueOffIterator

    ValueOffIterator  beginValueOff()       {return ValueOffIterator(this);}
    ValueOffIterator cbeginValueOff() const {return ValueOffIterator(this);}

    /// @brief Visits all values in a leaf node, i.e. both active and inactive values
    class ValueIterator
    {
        const LeafNode *mParent;
        uint32_t mPos;
    public:
        ValueIterator() : mParent(nullptr), mPos(1u << 3 * LOG2DIM) {}
        ValueIterator(const LeafNode* parent) :  mParent(parent), mPos(0) {NANOVDB_ASSERT(parent);}
        ValueIterator& operator=(const ValueIterator&) = default;
        bool operator*() const { NANOVDB_ASSERT(*this); return mParent->mValues.isOn(mPos);}
        Coord getCoord() const { NANOVDB_ASSERT(*this); return mParent->offsetToGlobalCoord(mPos);}
        bool isActive() const { NANOVDB_ASSERT(*this); return mParent->mValueMask.isOn(mPos);}
        operator bool() const {return mPos < SIZE;}
        ValueIterator& operator++() {++mPos; return *this;}
        ValueIterator operator++(int) {
            auto tmp = *this;
            ++(*this);
            return tmp;
        }
    }; // Member class ValueIterator

    ValueIterator beginValue()           {return ValueIterator(this);}
    ValueIterator cbeginValueAll() const {return ValueIterator(this);}

    LeafNode(const Coord& ijk, bool value, bool state)
        : mOrigin(ijk & ~MASK)
        , mValueMask(state)
        , mValues(value)
        , mDstOffset(0)
    {
    }
    LeafNode(const LeafNode&) = delete; // disallow copy-construction
    LeafNode(LeafNode&&) = delete; // disallow move construction
    LeafNode& operator=(const LeafNode&) = delete; // disallow copy assignment
    LeafNode& operator=(LeafNode&&) = delete; // disallow move assignment
    ~LeafNode() = default;

    const Mask<LOG2DIM>& valueMask() const {return mValueMask;}
    const Mask<LOG2DIM>& getValueMask() const {return mValueMask;}
    const Coord& origin() const {return mOrigin;}

    /// @brief Return the linear offset corresponding to the given coordinate
    static uint32_t CoordToOffset(const Coord& ijk)
    {
        return ((ijk[0] & int32_t(MASK)) << (2 * LOG2DIM)) +
               ((ijk[1] & int32_t(MASK)) << LOG2DIM) +
                (ijk[2] & int32_t(MASK));
    }

    static Coord OffsetToLocalCoord(uint32_t n)
    {
        NANOVDB_ASSERT(n < SIZE);
        const int32_t m = n & ((1 << 2 * LOG2DIM) - 1);
        return Coord(n >> 2 * LOG2DIM, m >> LOG2DIM, m & int32_t(MASK));
    }

    void localToGlobalCoord(Coord& ijk) const
    {
        ijk += mOrigin;
    }

    Coord offsetToGlobalCoord(uint32_t n) const
    {
        Coord ijk = LeafNode::OffsetToLocalCoord(n);
        this->localToGlobalCoord(ijk);
        return ijk;
    }
    bool getFirstValue() const { return mValues.isOn(0); }
    bool getLastValue() const { return mValues.isOn(SIZE - 1); }

    bool getValue(uint32_t i) const {return mValues.isOn(i);}
    bool getValue(const Coord& ijk) const
    {
        return mValues.isOn(CoordToOffset(ijk));
    }
#ifndef NANOVDB_NEW_ACCESSOR_METHODS
    template<typename AccT>
    bool isActiveAndCache(const Coord& ijk, const AccT&) const
    {
        return mValueMask.isOn(CoordToOffset(ijk));
    }

    template<typename AccT>
    bool getValueAndCache(const Coord& ijk, const AccT&) const
    {
        return mValues.isOn(CoordToOffset(ijk));
    }

    template<typename AccT>
    void setValueAndCache(const Coord& ijk, bool value, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
        mValues.setOn(n);
    }

    template<typename AccT>
    void setValueOnAndCache(const Coord& ijk, const AccT&)
    {
        const uint32_t n = CoordToOffset(ijk);
        mValueMask.setOn(n);
    }
#endif

    void setValue(uint32_t n, bool value)
    {
        mValueMask.setOn(n);
        mValues.set(n, value);
    }
    void setValue(const Coord& ijk, bool value) {return this->setValue(CoordToOffset(ijk), value);}

    void merge(LeafNode &other)
    {
        mValues |= other.mValues;
        mValueMask |= other.mValueMask;
    }

}; // tools::build::LeafNode<bool>

//================================================================================================

template<typename BuildT>
template<typename T>
inline typename util::enable_if<std::is_floating_point<T>::value>::type
LeafNode<BuildT>::signedFloodFill(T outside)
{
    const uint32_t first = *mValueMask.beginOn();
    if (first < SIZE) {
        bool xInside = mValues[first] < 0, yInside = xInside, zInside = xInside;
        for (uint32_t x = 0; x != DIM; ++x) {
            const uint32_t x00 = x << (2 * LOG2DIM);
            if (mValueMask.isOn(x00))
                xInside = mValues[x00] < 0; // element(x, 0, 0)
            yInside = xInside;
            for (uint32_t y = 0; y != DIM; ++y) {
                const uint32_t xy0 = x00 + (y << LOG2DIM);
                if (mValueMask.isOn(xy0))
                    yInside = mValues[xy0] < 0; // element(x, y, 0)
                zInside = yInside;
                for (uint32_t z = 0; z != (1 << LOG2DIM); ++z) {
                    const uint32_t xyz = xy0 + z; // element(x, y, z)
                    if (mValueMask.isOn(xyz)) {
                        zInside = mValues[xyz] < 0;
                    } else {
                        mValues[xyz] = zInside ? -outside : outside;
                    }
                }
            }
        }
    }
} // tools::build::LeafNode<T>::signedFloodFill

// ----------------------------> ValueAccessor <--------------------------------------

template<typename BuildT>
struct ValueAccessor
{
    using ValueType = typename BuildToValueMap<BuildT>::type;
    using LeafT = LeafNode<BuildT>;
    using Node1 = InternalNode<LeafT>;
    using Node2 = InternalNode<Node1>;
    using RootNodeType = RootNode<Node2>;
    using LeafNodeType = typename RootNodeType::LeafNodeType;

    ValueAccessor(RootNodeType& root)
        : mRoot(root)
        , mKeys{Coord(math::Maximum<int>::value()), Coord(math::Maximum<int>::value()), Coord(math::Maximum<int>::value())}
        , mNode{nullptr, nullptr, nullptr}
    {
    }
    ValueAccessor(ValueAccessor&&) = default; // allow move construction
    ValueAccessor(const ValueAccessor&) = delete; // disallow copy construction
    ValueType getValue(int i, int j, int k) const {return this->getValue(Coord(i,j,k));}
    template<typename NodeT>
    bool isCached(const Coord& ijk) const
    {
        return (ijk[0] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][0] &&
               (ijk[1] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][1] &&
               (ijk[2] & int32_t(~NodeT::MASK)) == mKeys[NodeT::LEVEL][2];
    }

    template <typename OpT, typename... ArgsT>
    auto get(const Coord& ijk, ArgsT&&... args) const
    {
        if (this->template isCached<LeafT>(ijk)) {
            return ((const LeafT*)mNode[0])->template get<OpT>(ijk, args...);
        } else if (this->template isCached<Node1>(ijk)) {
            return ((const Node1*)mNode[1])->template getAndCache<OpT>(ijk, *this, args...);
        } else if (this->template isCached<Node2>(ijk)) {
            return ((const Node2*)mNode[2])->template getAndCache<OpT>(ijk, *this, args...);
        }
        return mRoot.template getAndCache<OpT>(ijk, *this, args...);
    }

    template <typename OpT, typename... ArgsT>
    auto set(const Coord& ijk, ArgsT&&... args) const
    {
        if (this->template isCached<LeafT>(ijk)) {
            return ((LeafT*)mNode[0])->template set<OpT>(ijk, args...);
        } else if (this->template isCached<Node1>(ijk)) {
            return ((Node1*)mNode[1])->template setAndCache<OpT>(ijk, *this, args...);
        } else if (this->template isCached<Node2>(ijk)) {
            return ((Node2*)mNode[2])->template setAndCache<OpT>(ijk, *this, args...);
        }
        return mRoot.template setAndCache<OpT>(ijk, *this, args...);
    }

#ifdef NANOVDB_NEW_ACCESSOR_METHODS
    ValueType getValue(const Coord& ijk) const {return this->template get<GetValue<BuildT>>(ijk);}
    LeafT* setValue(const Coord& ijk, const ValueType& value) {return this->template set<SetValue<BuildT>>(ijk, value);}
    LeafT* setValueOn(const Coord& ijk) {return this->template set<SetValue<BuildT>>(ijk);}
    LeafT& touchLeaf(const Coord& ijk) {return this->template set<TouchLeaf<BuildT>>(ijk);}
    bool isActive(const Coord& ijk) const {return this->template get<GetState<BuildT>>(ijk);}
#else
    ValueType getValue(const Coord& ijk) const
    {
        if (this->template isCached<LeafT>(ijk)) {
            return ((LeafT*)mNode[0])->getValueAndCache(ijk, *this);
        } else if (this->template isCached<Node1>(ijk)) {
            return ((Node1*)mNode[1])->getValueAndCache(ijk, *this);
        } else if (this->template isCached<Node2>(ijk)) {
            return ((Node2*)mNode[2])->getValueAndCache(ijk, *this);
        }
        return mRoot.getValueAndCache(ijk, *this);
    }

    /// @brief Sets value in a leaf node and returns it.
    LeafT* setValue(const Coord& ijk, const ValueType& value)
    {
        if (this->template isCached<LeafT>(ijk)) {
            ((LeafT*)mNode[0])->setValueAndCache(ijk, value, *this);
        } else if (this->template isCached<Node1>(ijk)) {
            ((Node1*)mNode[1])->setValueAndCache(ijk, value, *this);
        } else if (this->template isCached<Node2>(ijk)) {
            ((Node2*)mNode[2])->setValueAndCache(ijk, value, *this);
        } else {
            mRoot.setValueAndCache(ijk, value, *this);
        }
        NANOVDB_ASSERT(this->isCached<LeafT>(ijk));
        return (LeafT*)mNode[0];
    }
    void setValueOn(const Coord& ijk)
    {
        if (this->template isCached<LeafT>(ijk)) {
            ((LeafT*)mNode[0])->setValueOnAndCache(ijk, *this);
        } else if (this->template isCached<Node1>(ijk)) {
            ((Node1*)mNode[1])->setValueOnAndCache(ijk, *this);
        } else if (this->template isCached<Node2>(ijk)) {
            ((Node2*)mNode[2])->setValueOnAndCache(ijk, *this);
        } else {
            mRoot.setValueOnAndCache(ijk, *this);
        }
    }
    void touchLeaf(const Coord& ijk) const
    {
        if (this->template isCached<LeafT>(ijk)) {
            return;
        } else if (this->template isCached<Node1>(ijk)) {
            ((Node1*)mNode[1])->touchLeafAndCache(ijk, *this);
        } else if (this->template isCached<Node2>(ijk)) {
            ((Node2*)mNode[2])->touchLeafAndCache(ijk, *this);
        } else {
            mRoot.touchLeafAndCache(ijk, *this);
        }
    }
    bool isActive(const Coord& ijk) const
    {
        if (this->template isCached<LeafT>(ijk)) {
            return ((LeafT*)mNode[0])->isActiveAndCache(ijk, *this);
        } else if (this->template isCached<Node1>(ijk)) {
            return ((Node1*)mNode[1])->isActiveAndCache(ijk, *this);
        } else if (this->template isCached<Node2>(ijk)) {
            return ((Node2*)mNode[2])->isActiveAndCache(ijk, *this);
        }
        return mRoot.isActiveAndCache(ijk, *this);
    }
#endif

    bool isValueOn(const Coord& ijk) const { return this->isActive(ijk); }
    template<typename NodeT>
    void insert(const Coord& ijk, NodeT* node) const
    {
        mKeys[NodeT::LEVEL] = ijk & ~NodeT::MASK;
        mNode[NodeT::LEVEL] = node;
    }
    RootNodeType& mRoot;
    mutable Coord mKeys[3];
    mutable void* mNode[3];
}; // tools::build::ValueAccessor<BuildT>

// ----------------------------> Tree <--------------------------------------

template<typename BuildT>
struct Tree
{
    using ValueType = typename BuildToValueMap<BuildT>::type;
    using Node0 = LeafNode<BuildT>;
    using Node1 = InternalNode<Node0>;
    using Node2 = InternalNode<Node1>;
    using RootNodeType = RootNode<Node2>;
    using LeafNodeType = typename RootNodeType::LeafNodeType;
    struct WriteAccessor;

    RootNodeType  mRoot;
    std::mutex    mMutex;

    Tree(const ValueType &background) : mRoot(background) {}
    Tree(const Tree&) = delete; // disallow copy construction
    Tree(Tree&&) = delete; // disallow move construction
    Tree& tree() {return *this;}
    RootNodeType& root() {return mRoot;}
    ValueType getValue(const Coord& ijk) const {return mRoot.getValue(ijk);}
    ValueType getValue(int i, int j, int k) const {return this->getValue(Coord(i,j,k));}
    void setValue(const Coord& ijk, const ValueType &value) {mRoot.setValue(ijk, value);}
    std::array<size_t,3> nodeCount() const
    {
        std::array<size_t, 3> count{0,0,0};
        mRoot.nodeCount(count);
        return count;
    }
    /// @brief regular accessor for thread-safe reading and non-thread-safe writing
    ValueAccessor<BuildT> getAccessor() { return ValueAccessor<BuildT>(mRoot); }
    /// @brief special accessor for thread-safe writing only
    WriteAccessor getWriteAccessor() { return WriteAccessor(mRoot, mMutex); }
};// tools::build::Tree<BuildT>

// ----------------------------> Tree::WriteAccessor <--------------------------------------

template<typename BuildT>
struct Tree<BuildT>::WriteAccessor
{
    using AccT   = ValueAccessor<BuildT>;
    using ValueType = typename AccT::ValueType;
    using LeafT  = typename AccT::LeafT;
    using Node1  = typename AccT::Node1;
    using Node2  = typename AccT::Node2;
    using RootNodeType  = typename AccT::RootNodeType;

    WriteAccessor(RootNodeType& parent, std::mutex &mx)
        : mParent(parent)
        , mRoot(parent.mBackground)
        , mAcc(mRoot)
        , mMutex(mx)
    {
    }
    WriteAccessor(const WriteAccessor&) = delete; // disallow copy construction
    WriteAccessor(WriteAccessor&&) = default; // allow move construction
    ~WriteAccessor() { this->merge(); }
    void merge()
    {
        mMutex.lock();
        mParent.merge(mRoot);
        mMutex.unlock();
    }
    inline void setValueOn(const Coord& ijk) {mAcc.setValueOn(ijk);}
    inline void setValue(const Coord& ijk, const ValueType &value) {mAcc.setValue(ijk, value);}

    RootNodeType &mParent, mRoot;
    AccT          mAcc;
    std::mutex   &mMutex;
}; // tools::build::Tree<BuildT>::WriteAccessor

// ----------------------------> Grid <--------------------------------------

template<typename BuildT>
struct Grid : public Tree<BuildT>
{
    using BuildType = BuildT;
    using ValueType = typename BuildToValueMap<BuildT>::type;
    using TreeType = Tree<BuildT>;
    using Node0 = LeafNode<BuildT>;
    using Node1 = InternalNode<Node0>;
    using Node2 = InternalNode<Node1>;
    using RootNodeType = RootNode<Node2>;

    GridClass   mGridClass;
    GridType    mGridType;
    Map         mMap;
    std::string mName;

    Grid(const ValueType &background, const std::string &name = "", GridClass gClass = GridClass::Unknown)
      : TreeType(background)
      , mGridClass(gClass)
      , mGridType(toGridType<BuildT>())
      , mName(name)
    {
        mMap.set(1.0, Vec3d(0.0), 1.0);
    }
    TreeType& tree() {return *this;}
    const GridType&  gridType() const { return mGridType; }
    const GridClass& gridClass() const { return mGridClass; }
    const Map& map() const { return mMap; }
    void setTransform(double scale=1.0, const Vec3d &translation = Vec3d(0.0)) {mMap.set(scale, translation, 1.0);}
    const std::string& gridName() const { return mName; }
    const std::string& getName() const { return mName; }
    void setName(const std::string &name) { mName = name; }
    /// @brief Sets grids values in domain of the @a bbox to those returned by the specified @a func with the
    ///        expected signature [](const Coord&)->ValueType.
    ///
    /// @note If @a func returns a value equal to the background value of the input grid at a
    ///       specific voxel coordinate, then the active state of that coordinate is off! Else the value
    ///       value is set and the active state is on. This is done to allow for sparse grids to be generated.
    ///
    /// @param func  Functor used to evaluate the grid values in the @a bbox
    /// @param bbox  Coordinate bounding-box over which the grid values will be set.
    /// @param delta Specifies a lower threshold value for rendering (optional). Typically equals the voxel size
    ///              for level sets and otherwise it's zero.
    template <typename Func>
    void operator()(const Func& func, const CoordBBox& bbox, ValueType delta = ValueType(0));
};// tools::build::Grid

template <typename BuildT>
template <typename Func>
void Grid<BuildT>::operator()(const Func& func, const CoordBBox& bbox, ValueType delta)
{
    auto &root = this->tree().root();
#if __cplusplus >= 201703L
    static_assert(util::is_same<ValueType, typename std::invoke_result<Func,const Coord&>::type>::value, "GridBuilder: mismatched ValueType");
#else// invoke_result was introduced in C++17 and result_of was removed in C++20
    static_assert(util::is_same<ValueType, typename std::result_of<Func(const Coord&)>::type>::value, "GridBuilder: mismatched ValueType");
#endif
    const CoordBBox leafBBox(bbox[0] >> Node0::TOTAL, bbox[1] >> Node0::TOTAL);
    std::mutex mutex;
    util::forEach(leafBBox, [&](const CoordBBox& b) {
        Node0* leaf = nullptr;
        for (auto it = b.begin(); it; ++it) {
            Coord min(*it << Node0::TOTAL), max(min + Coord(Node0::DIM - 1));
            const CoordBBox b(min.maxComponent(bbox.min()),
                              max.minComponent(bbox.max()));// crop
            if (leaf == nullptr) {
                leaf = new Node0(b[0], root.mBackground, false);
            } else {
                leaf->mOrigin = b[0] & ~Node0::MASK;
                NANOVDB_ASSERT(leaf->mValueMask.isOff());
            }
            leaf->mDstOffset = 0;// no prune
            for (auto ijk = b.begin(); ijk; ++ijk) {
                const auto v = func(*ijk);// call functor
                if (v != root.mBackground) leaf->setValue(*ijk, v);// don't insert background values
            }
            if (!leaf->mValueMask.isOff()) {// has active values
                if (leaf->mValueMask.isOn()) {// only active values
                    const auto first = leaf->getFirstValue();
                    int n=1;
                    while (n<512) {// 8^3 = 512
                        if (leaf->mValues[n++] != first) break;
                    }
                    if (n == 512) leaf->mDstOffset = 1;// prune below
                }
                std::lock_guard<std::mutex> guard(mutex);
                NANOVDB_ASSERT(leaf != nullptr);
                root.addNode(leaf);
                NANOVDB_ASSERT(leaf == nullptr);
            }
        }// loop over sub-part of leafBBox
        if (leaf) delete leaf;
    });

    // Prune leaf and tile nodes
    for (auto it2 = root.mTable.begin(); it2 != root.mTable.end(); ++it2) {
        if (auto *upper = it2->second.child) {//upper level internal node
            for (auto it1 = upper->mChildMask.beginOn(); it1; ++it1) {
                auto *lower = upper->mTable[*it1].child;// lower level internal node
                for (auto it0 = lower->mChildMask.beginOn(); it0; ++it0) {
                    auto *leaf = lower->mTable[*it0].child;// leaf nodes
                    if (leaf->mDstOffset) {
                        lower->mTable[*it0].value = leaf->getFirstValue();
                        lower->mChildMask.setOff(*it0);
                        lower->mValueMask.setOn(*it0);
                        delete leaf;
                    }
                }// loop over leaf nodes
                if (lower->mChildMask.isOff()) {//only tiles
                    const auto first = lower->getFirstValue();
                    int n=1;
                    while (n < 4096) {// 16^3 = 4096
                        if (lower->mTable[n++].value != first) break;
                    }
                    if (n == 4096) {// identical tile values so prune
                        upper->mTable[*it1].value = first;
                        upper->mChildMask.setOff(*it1);
                        upper->mValueMask.setOn(*it1);
                        delete lower;
                    }
                }
            }// loop over lower internal nodes
            if (upper->mChildMask.isOff()) {//only tiles
                const auto first = upper->getFirstValue();
                int n=1;
                while (n < 32768) {// 32^3 = 32768
                    if (upper->mTable[n++].value != first) break;
                }
                if (n == 32768) {// identical tile values so prune
                    it2->second.value = first;
                    it2->second.state = upper->mValueMask.isOn();
                    it2->second.child = nullptr;
                    delete upper;
                }
            }
        }// is child node of the root
    }// loop over root table
}// tools::build::Grid::operator()

//================================================================================================

template <typename T>
using BuildLeaf = LeafNode<T>;
template <typename T>
using BuildLower = InternalNode<BuildLeaf<T>>;
template <typename T>
using BuildUpper = InternalNode<BuildLower<T>>;
template <typename T>
using BuildRoot  = RootNode<BuildUpper<T>>;
template <typename T>
using BuildTile  = typename BuildRoot<T>::Tile;

using FloatGrid  = Grid<float>;
using Fp4Grid    = Grid<Fp4>;
using Fp8Grid    = Grid<Fp8>;
using Fp16Grid   = Grid<Fp16>;
using FpNGrid    = Grid<FpN>;
using DoubleGrid = Grid<double>;
using Int32Grid  = Grid<int32_t>;
using UInt32Grid = Grid<uint32_t>;
using Int64Grid  = Grid<int64_t>;
using Vec3fGrid  = Grid<Vec3f>;
using Vec3dGrid  = Grid<Vec3d>;
using Vec4fGrid  = Grid<Vec4f>;
using Vec4dGrid  = Grid<Vec4d>;
using MaskGrid   = Grid<ValueMask>;
using IndexGrid  = Grid<ValueIndex>;
using OnIndexGrid = Grid<ValueOnIndex>;
using BoolGrid   = Grid<bool>;

// ----------------------------> NodeManager <--------------------------------------

// GridT can be openvdb::Grid and nanovdb::tools::build::Grid
template <typename GridT>
class NodeManager
{
public:

    using ValueType = typename GridT::ValueType;
    using BuildType = typename GridT::BuildType;
    using GridType = GridT;
    using TreeType = typename GridT::TreeType;
    using RootNodeType = typename TreeType::RootNodeType;
    static_assert(RootNodeType::LEVEL == 3, "NodeManager expected LEVEL=3");
    using Node2 = typename RootNodeType::ChildNodeType;
    using Node1 = typename Node2::ChildNodeType;
    using Node0 = typename Node1::ChildNodeType;

    NodeManager(GridT &grid) : mGrid(grid) {this->init();}
    void init()
    {
        mArray0.clear();
        mArray1.clear();
        mArray2.clear();
        auto counts = mGrid.tree().nodeCount();
        mArray0.reserve(counts[0]);
        mArray1.reserve(counts[1]);
        mArray2.reserve(counts[2]);

        for (auto it2 = mGrid.tree().root().cbeginChildOn(); it2; ++it2) {
            Node2 &upper = const_cast<Node2&>(*it2);
            mArray2.emplace_back(&upper);
            for (auto it1 = upper.cbeginChildOn(); it1; ++it1) {
                Node1 &lower = const_cast<Node1&>(*it1);
                mArray1.emplace_back(&lower);
                for (auto it0 = lower.cbeginChildOn(); it0; ++it0) {
                    Node0 &leaf = const_cast<Node0&>(*it0);
                    mArray0.emplace_back(&leaf);
                }// loop over leaf nodes
            }// loop over lower internal nodes
        }// loop over root node
    }

    /// @brief Return the number of tree nodes at the specified level
    /// @details 0 is leaf, 1 is lower internal, and 2 is upper internal level
    uint64_t nodeCount(int level) const
    {
        NANOVDB_ASSERT(level==0 || level==1 || level==2);
        return level==0 ? mArray0.size() : level==1 ? mArray1.size() : mArray2.size();
    }

    template <int LEVEL>
    typename util::enable_if<LEVEL==0, Node0&>::type node(int i) {return *mArray0[i];}
    template <int LEVEL>
    typename util::enable_if<LEVEL==0, const Node0&>::type node(int i) const {return *mArray0[i];}
    template <int LEVEL>
    typename util::enable_if<LEVEL==1, Node1&>::type node(int i) {return *mArray1[i];}
    template <int LEVEL>
    typename util::enable_if<LEVEL==1, const Node1&>::type node(int i) const {return *mArray1[i];}
    template <int LEVEL>
    typename util::enable_if<LEVEL==2, Node2&>::type node(int i) {return *mArray2[i];}
    template <int LEVEL>
    typename util::enable_if<LEVEL==2, const Node2&>::type node(int i) const {return *mArray2[i];}

    /// @brief Return the i'th leaf node with respect to breadth-first ordering
    const Node0& leaf(uint32_t i) const { return *mArray0[i]; }
    Node0& leaf(uint32_t i) { return *mArray0[i]; }
    uint64_t leafCount() const {return mArray0.size();}

    /// @brief Return the i'th lower internal node with respect to breadth-first ordering
    const Node1& lower(uint32_t i) const { return *mArray1[i]; }
    Node1& lower(uint32_t i) { return *mArray1[i]; }
    uint64_t lowerCount() const {return mArray1.size();}

    /// @brief Return the i'th upper internal node with respect to breadth-first ordering
    const Node2& upper(uint32_t i) const { return *mArray2[i]; }
    Node2& upper(uint32_t i) { return *mArray2[i]; }
    uint64_t upperCount() const {return mArray2.size();}

    RootNodeType& root() {return mGrid.tree().root();}
    const RootNodeType& root() const {return mGrid.tree().root();}

    TreeType& tree() {return mGrid.tree();}
    const TreeType& tree() const {return mGrid.tree();}

    GridType& grid() {return mGrid;}
    const GridType& grid() const {return mGrid;}

protected:

    GridT                &mGrid;
    std::vector<Node0*>   mArray0; // leaf nodes
    std::vector<Node1*>   mArray1; // lower internal nodes
    std::vector<Node2*>   mArray2; // upper internal nodes

};// NodeManager

template <typename NodeManagerT>
typename util::enable_if<util::is_floating_point<typename NodeManagerT::ValueType>::value>::type
sdfToLevelSet(NodeManagerT &mgr)
{
    mgr.grid().mGridClass = GridClass::LevelSet;
    // Note that the bottom-up flood filling is essential
    const auto outside = mgr.root().mBackground;
    util::forEach(0, mgr.leafCount(), 8, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) mgr.leaf(i).signedFloodFill(outside);
    });
    util::forEach(0, mgr.lowerCount(), 1, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) mgr.lower(i).signedFloodFill(outside);
    });
    util::forEach(0, mgr.upperCount(), 1, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) mgr.upper(i).signedFloodFill(outside);
    });
    mgr.root().signedFloodFill(outside);
}// sdfToLevelSet

template <typename NodeManagerT>
void levelSetToFog(NodeManagerT &mgr, bool rebuild = true)
{
    using ValueType = typename NodeManagerT::ValueType;
    mgr.grid().mGridClass = GridClass::FogVolume;
    const ValueType d = -mgr.root().mBackground, w = 1.0f / d;
    //std::atomic_bool prune{false};
    std::atomic<bool> prune{false};
    auto op = [&](ValueType& v) -> bool {
        if (v > ValueType(0)) {
            v = ValueType(0);
            return false;
        }
        v = v > d ? v * w : ValueType(1);
        return true;
    };
    util::forEach(0, mgr.leafCount(), 8, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto& leaf = mgr.leaf(i);
            for (uint32_t i = 0; i < 512u; ++i) leaf.mValueMask.set(i, op(leaf.mValues[i]));
        }
    });
    util::forEach(0, mgr.lowerCount(), 1, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto& node = mgr.lower(i);
            for (uint32_t i = 0; i < 4096u; ++i) {
                if (node.mChildMask.isOn(i)) {
                    auto* leaf = node.mTable[i].child;
                    if (leaf->mValueMask.isOff()) {// prune leaf node
                        node.mTable[i].value = leaf->getFirstValue();
                        node.mChildMask.setOff(i);
                        delete leaf;
                        prune = true;
                    }
                } else {
                    node.mValueMask.set(i, op(node.mTable[i].value));
                }
            }
        }
    });
    util::forEach(0, mgr.upperCount(), 1, [&](const util::Range1D& r) {
        for (auto i = r.begin(); i != r.end(); ++i) {
            auto& node = mgr.upper(i);
            for (uint32_t i = 0; i < 32768u; ++i) {
                if (node.mChildMask.isOn(i)) {// prune lower internal node
                    auto* child = node.mTable[i].child;
                    if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
                        node.mTable[i].value = child->getFirstValue();
                        node.mChildMask.setOff(i);
                        delete child;
                        prune = true;
                    }
                } else {
                    node.mValueMask.set(i, op(node.mTable[i].value));
                }
            }
        }
    });

    for (auto it = mgr.root().mTable.begin(); it != mgr.root().mTable.end(); ++it) {
        auto* child = it->second.child;
        if (child == nullptr) {
            it->second.state = op(it->second.value);
        } else if (child->mChildMask.isOff() && child->mValueMask.isOff()) {
            it->second.value = child->getFirstValue();
            it->second.state = false;
            it->second.child = nullptr;
            delete child;
            prune = true;
        }
    }
    if (rebuild && prune) mgr.init();
}// levelSetToFog

// ----------------------------> Implementations of random access methods <--------------------------------------

template <typename T>
struct TouchLeaf {
    static BuildLeaf<T>& set(BuildLeaf<T> &leaf, uint32_t)  {return leaf;}
};// TouchLeaf<BuildT>

/// @brief Implements Tree::getValue(Coord), i.e. return the value associated with a specific coordinate @c ijk.
/// @tparam BuildT Build type of the grid being called
/// @details The value at a coordinate maps to the background, a tile value or a leaf value.
template <typename T>
struct GetValue {
    static auto get(const BuildRoot<T>  &root) {return root.mBackground;}
    static auto get(const BuildTile<T>  &tile) {return tile.value;}
    static auto get(const BuildUpper<T> &node, uint32_t n) {return node.mTable[n].value;}
    static auto get(const BuildLower<T> &node, uint32_t n) {return node.mTable[n].value;}
    static auto get(const BuildLeaf<T>  &leaf, uint32_t n) {return leaf.getValue(n);}
};// GetValue<T>

/// @brief Implements Tree::isActive(Coord)
/// @tparam T Build type of the grid being called
template <typename T>
struct GetState {
    static bool get(const BuildRoot<T>&) {return false;}
    static bool get(const BuildTile<T>  &tile) {return tile.state;}
    static bool get(const BuildUpper<T> &node, uint32_t n) {return node.mValueMask.isOn(n);}
    static bool get(const BuildLower<T> &node, uint32_t n) {return node.mValueMask.isOn(n);}
    static bool get(const BuildLeaf<T>  &leaf, uint32_t n) {return leaf.mValueMask.isOn(n);}
};// GetState<T>

/// @brief Set the value and its state at the leaf level mapped to by ijk, and create the leaf node and branch if needed.
/// @tparam T BuildType of the corresponding tree
template <typename T>
struct SetValue {
    static BuildLeaf<T>* set(BuildLeaf<T> &leaf, uint32_t n) {
        leaf.mValueMask.setOn(n);// always set the active bit
        return &leaf;
    }
    static BuildLeaf<T>* set(BuildLeaf<T> &leaf, uint32_t n, const typename BuildLeaf<T>::ValueType &v) {
        leaf.setValue(n, v);
        return &leaf;
    }
};// SetValue<T>

/// @brief Implements Tree::probeLeaf(Coord)
/// @tparam T Build type of the grid being called
template <typename T>
struct ProbeValue {
    using ValueT = typename BuildLeaf<T>::ValueType;
    static bool get(const BuildRoot<T>  &root, ValueT &v) {
        v = root.mBackground;
        return false;
    }
    static bool get(const BuildTile<T> &tile, ValueT &v) {
        v = tile.value;
        return tile.state;
    }
    static bool get(const BuildUpper<T> &node, uint32_t n, ValueT &v) {
        v = node.mTable[n].value;
        return node.mValueMask.isOn(n);
    }
    static bool get(const BuildLower<T> &node, uint32_t n, ValueT &v) {
        v = node.mTable[n].value;
        return node.mValueMask.isOn(n);
    }
    static bool get(const BuildLeaf<T>  &leaf, uint32_t n, ValueT &v) {
        v = leaf.getValue(n);
        return leaf.isActive(n);
    }
};// ProbeValue<T>

} // namespace tools::build

} // namespace nanovdb

#endif // NANOVDB_TOOLS_BUILD_GRIDBUILDER_H_HAS_BEEN_INCLUDED
