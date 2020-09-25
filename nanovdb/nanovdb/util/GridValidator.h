// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file GridValidator.h

    \author Ken Museth

    \date August 30, 2020

    \brief Checks the validity of an existing NanoVDB grid.
*/

#ifndef NANOVDB_GRIDVALIDATOR_H_HAS_BEEN_INCLUDED
#define NANOVDB_GRIDVALIDATOR_H_HAS_BEEN_INCLUDED

#include "../NanoVDB.h"
#include "GridChecksum.h"
#include "Range.h"
#include "ForEach.h"

#include <atomic>
namespace nanovdb {

/// @brief Return true if the specified grid passes several validation tests.
///
/// @param grid Grid to validate
/// @param detailed If true the validation test is detailed and relatively slow.
/// @param verbose If true information about the first failed test is printed to std::cerr
template <typename ValueT>
bool isValid(const NanoGrid<ValueT> &grid, bool detailed = true, bool verbose = false);

/// @brief Allows for the construction of NanoVDB grids without any dependecy
template <typename ValueT>
class GridValidator
{
    using Node0 = LeafNode<ValueT>; // leaf
    using Node1 = InternalNode<Node0>; // lower
    using Node2 = InternalNode<Node1>; // upper
    using RootT = RootNode<Node2>;
    using TreeT = Tree<RootT>;
    using GridT = Grid<TreeT>;

    const NanoGrid<ValueT>* mGrid;
    std::string             mErrorStr;
    
    void checkGrid(bool detailed);
    void checkTree();
    void checkRoot();
    template <typename NodeT>
    void checkNodes();

public:
    GridValidator() : mGrid(nullptr) {}

    /// @brief Returns an error message (an empty string means no error)
    const std::string& operator()(const NanoGrid<ValueT> &grid, bool detailed = true);

};// GridValidator

//================================================================================================

template <typename ValueT>
const std::string& GridValidator<ValueT>::operator()(const NanoGrid<ValueT> &grid, bool detailed)
{
    mGrid = &grid;
    mErrorStr.clear();
    
    this->checkGrid(detailed);
    this->checkTree();
    this->checkRoot();
    if (detailed) {
        this->template checkNodes<Node2>();
        this->template checkNodes<Node2>();
    }
    return mErrorStr;
}

//================================================================================================

template<typename ValueT>
void GridValidator<ValueT>::checkGrid(bool detailed)
{
    if (!mErrorStr.empty()) return;
    auto *data = reinterpret_cast<const typename GridT::DataType*>(mGrid);
    std::stringstream ss;
    if (data->mMagic != NANOVDB_MAGIC_NUMBER) {
        ss << "Incorrect magic number: Expected " << NANOVDB_MAGIC_NUMBER << ", but read " << data->mMagic;
        mErrorStr = ss.str();
    } else if (!validateChecksum(*mGrid, detailed ? ChecksumMode::Full : ChecksumMode::Partial)) {
        mErrorStr.assign("Mis-matching checksum");
    } else if (data->mMajor != NANOVDB_MAJOR_VERSION_NUMBER) {
        ss << "Invalid major version number: Expected " << NANOVDB_MAJOR_VERSION_NUMBER << ", but read " << data->mMajor;
        mErrorStr = ss.str();
    } else if (data->mGridClass >= GridClass::End) {
        mErrorStr.assign("Invalid Grid Class");
    } else if (data->mGridType != mapToGridType<ValueT>()) {
        mErrorStr.assign("Invalid Grid Type");
    } else if ( (const void*)(&(mGrid->tree())) != (const void*)(mGrid+1) ) {
        mErrorStr.assign("Invalid Tree pointer");
    }
}// GridValidator::validateGrid

//================================================================================================

template<typename ValueT>
void GridValidator<ValueT>::checkTree()
{
    if (!mErrorStr.empty()) return;
    auto *data = mGrid->tree().data();
    if (data->mCount[3] != 1u) {
        mErrorStr.assign("Invalide number of root nodes in the tree");
    }  else if ( (const uint8_t*)(&mGrid->tree().root()) < (const uint8_t*)(&mGrid->tree()+1)) {
       mErrorStr.assign("Invalide root pointer (should be located after the Grid and Tree)");
    } else if ( mGrid->isEmpty() && (const uint8_t*)(&mGrid->tree().root()) > (const uint8_t*)(mGrid) + mGrid->totalMemUsage()) {
       mErrorStr.assign("Invalide root pointer (appears to be located after the end of the buffer)");
    } else if (!mGrid->isEmpty() && (const uint8_t*)(&mGrid->tree().root()) > (const uint8_t*)(mGrid->tree().template getNode<2>(0)) ) {
       mErrorStr.assign("Invalide root pointer (appears to be after the first upper internal node)");
    }
}// GridValidator::checkTree

//================================================================================================

template<typename ValueT>
void GridValidator<ValueT>::checkRoot()
{
    if (!mErrorStr.empty()) return;
    auto *data = mGrid->tree().root().data();
    const int32_t maxID = static_cast<int32_t>(data->mTileCount);
    const auto nodeCount = mGrid->tree().nodeCount(2);
    const uint8_t *minPtr = nodeCount>0 ? (const uint8_t*)(mGrid->tree().template getNode<2>(0)) : nullptr;
    const uint8_t *maxPtr = nodeCount>0 ? (const uint8_t*)(mGrid->tree().template getNode<2>(nodeCount-1)) : nullptr;
    for (uint32_t i = 0; i<data->mTileCount; ++i) {
        auto& tile = data->tile(i);
        if (tile.childID == -1) continue;
        if (tile.childID < -1 || tile.childID >= maxID) {
            mErrorStr.assign("Invalid Tile.childID");
            return;
        } else if ( (const uint8_t*)(&data->child(tile)) < minPtr ) {
            mErrorStr.assign("Invalid root child pointer (below lower bound");
            return;
        } else if ( (const uint8_t*)(&data->child(tile)) > maxPtr ) {
            mErrorStr.assign("Invalid root child pointer (above higher bound");
            return;
        }
    }
}// GridValidator::processRoot

//================================================================================================
template<typename ValueT>
template<typename NodeT>
void GridValidator<ValueT>::
checkNodes()
{// this is just a place holder for now
    using ChildT = typename NodeT::ChildNodeType;
    const auto &tree = mGrid->tree();
    const auto nodeCount = mGrid->tree().template nodeCount<ChildT>();
    if (!mErrorStr.empty() || nodeCount == 0) return;
    const uint8_t *minPtr = (const uint8_t*)(mGrid->tree().template getNode<ChildT>(0));
    const uint8_t *maxPtr = (const uint8_t*)(mGrid->tree().template getNode<ChildT>(nodeCount-1));
    std::atomic_bool error{false};
    auto kernel = [&](const Range1D &r) 
    {
        for (auto i = r.begin(); !error && i != r.end(); ++i) {
            auto *node = const_cast<NodeT*>(tree.template getNode<NodeT>(i));
            auto *data = node->data();
            for (auto childIter = data->mChildMask.beginOn(); !error && childIter; ++childIter) {
                auto* p = (const uint8_t*)(data->child(*childIter));
                if (p < minPtr ||  p > maxPtr) error = true;
            }
        }
    };
    forEach(0, tree.template nodeCount<NodeT>(), 4, kernel);
    if (error) mErrorStr.assign("Invalid internal child node pointer");
} // GridValidator::processNodes


//================================================================================================

template <typename ValueT>
bool isValid(const NanoGrid<ValueT> &grid, bool detailed, bool verbose)
{
    GridValidator<ValueT> validator;
    const std::string &str = validator( grid, detailed );
    if (verbose && !str.empty()) std::cerr << "Validation failed: " << str << std::endl;
    return str.empty();
}

} // namespace nanovdb

#endif // NANOVDB_GRIDVALIDATOR_H_HAS_BEEN_INCLUDED
