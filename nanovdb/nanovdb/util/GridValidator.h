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
    using GridT = NanoGrid<ValueT>;
    inline static void checkTree( const GridT&, std::string&, bool);
    inline static void checkRoot( const GridT&, std::string&, bool);
    inline static void checkNodes(const GridT&, std::string&);

public:
    /// @brief Returns an error message (an empty string means no error)
    ///
    /// @param grid NanoVDB grid to be tested
    /// @param detailed If true the checksum is computed and validated as well as all the node pointers
    ///
    /// @note The validation is much slower if @c detailed == true!
    static std::string check(const GridT &grid, bool detailed = true);

};// GridValidator

//================================================================================================

template <typename ValueT>
std::string GridValidator<ValueT>::check(const GridT &grid, bool detailed)
{
    std::string errorStr;

    // First check the Grid
    auto *data = reinterpret_cast<const typename GridT::DataType*>(&grid);
    std::stringstream ss;
    if (!isValid(data)) {
        errorStr.assign("Grid is not 32B aligned");
    } else if (data->mMagic != NANOVDB_MAGIC_NUMBER) {
        ss << "Incorrect magic number: Expected " << NANOVDB_MAGIC_NUMBER << ", but read " << data->mMagic;
        errorStr = ss.str();
    } else if (!validateChecksum(grid, detailed ? ChecksumMode::Full : ChecksumMode::Partial)) {
        errorStr.assign("Mis-matching checksum");
    } else if (data->mVersion >= Version(29,0,0) && data->mVersion.getMajor() != NANOVDB_MAJOR_VERSION_NUMBER) {
        ss << "Invalid major version number: Expected " << NANOVDB_MAJOR_VERSION_NUMBER << ", but read " << data->mVersion.c_str();
        errorStr = ss.str();
    } else if (data->mVersion < Version(29,0,0) && data->mVersion.id() != 28u) {
        ss << "Invalid old major version number: Expected 28 or newer, but read " << data->mVersion.id();
        errorStr = ss.str();
    } else if (data->mGridClass >= GridClass::End) {
        errorStr.assign("Invalid GridClass");
     } else if (data->mGridType >= GridType::End) {
        errorStr.assign("Invalid GridType");
    } else if (data->mGridType != mapToGridType<ValueT>()) {
        errorStr.assign("Invalid combination of ValueType and GridType");
    } else if (!isValid(data->mGridType, data->mGridClass)) {
        errorStr.assign("Invalid combination of GridType and GridClass");
    } else if ( (const uint8_t*)(&(grid.tree())) != (const uint8_t*)(&grid+1) ) {
        errorStr.assign("Invalid Tree pointer");
    } else {
        checkTree(grid, errorStr, detailed);
    }
    return errorStr;
}

//================================================================================================

template<typename ValueT>
void GridValidator<ValueT>::checkTree(const GridT &grid, std::string &errorStr, bool detailed)
{
    if (!isValid(&grid.tree())) {
        errorStr.assign("Tree is not 32B aligned");
    } else if ( (const uint8_t*)(&grid.tree().root()) < (const uint8_t*)(&grid.tree()+1)) {
       errorStr.assign("Invalid root pointer (should be located after the Grid and Tree)");
    } else if ( (const uint8_t*)(&grid.tree().root()) > (const uint8_t*)(&grid) + grid.gridSize() - sizeof(grid.tree().root()) ) {
       errorStr.assign("Invalid root pointer (appears to be located after the end of the buffer)");
    } else {
       checkRoot(grid, errorStr, detailed);
    }
}// GridValidator::checkTree

//================================================================================================

template<typename ValueT>
void GridValidator<ValueT>::checkRoot(const GridT &grid, std::string &errorStr, bool detailed)
{
    auto &root = grid.tree().root();
    auto *data = root.data();
    if (!isValid(data)) {
        errorStr.assign("Root is not 32B aligned");
    }
    const uint8_t *minPtr = (const uint8_t*)(&root + 1);
    const uint8_t *maxPtr = (const uint8_t*)(&root) + root.memUsage();
    for (uint32_t i = 0; errorStr.empty() && i<data->mTableSize; ++i) {
        const auto *tile = data->tile(i);
        if ( (const uint8_t *) tile < minPtr ) {
            errorStr.assign("Invalid root tile pointer (below lower bound");
        } else if ( (const uint8_t *) tile > maxPtr - sizeof(*tile) ) {
            errorStr.assign("Invalid root tile pointer (above higher bound");
        }
    }
    if (detailed && errorStr.empty()) {
        checkNodes(grid, errorStr);
    }
}// GridValidator::processRoot

//================================================================================================
template<typename ValueT>
void GridValidator<ValueT>::checkNodes(const GridT &grid, std::string &errorStr)
{
    auto &root = grid.tree().root();// note, the root node was already checked
    const uint8_t *minPtr = (const uint8_t*)(&root) + root.memUsage();
    const uint8_t *maxPtr = (const uint8_t*)(&grid) + grid.gridSize();

    auto check = [&](const void * ptr, size_t ptrSize) -> bool {
        if (!isValid(ptr)) {
            errorStr.assign("Invalid node pointer: not 32B aligned");
        } else if ( (const uint8_t *) ptr < minPtr ) {
            errorStr.assign("Invalid node pointer: below lower bound");
        } else if ( (const uint8_t *) ptr > maxPtr - ptrSize ) {
            errorStr.assign("Invalid node pointer: above higher bound");
        }
        return errorStr.empty();
    };

    for (auto it2 = grid.tree().root().beginChild(); it2; ++it2) {
        auto &node2 = *it2;
        if (!check(&node2, sizeof(node2))) return;
        for (auto it1 = node2.beginChild(); it1; ++it1) {
            auto &node1 = *it1;
            if (!check(&node1, sizeof(node1))) return;
            for (auto it0 = node1.beginChild(); it0; ++it0) {
                auto &node0 = *it0;
                if (!check(&node2, sizeof(node2))) return;
            }// loop over child nodes of the lower internal node
        }// loop over child nodes of the upper internal node
    }// loop over child nodes of the root node

} // GridValidator::processNodes


//================================================================================================

template <typename ValueT>
bool isValid(const NanoGrid<ValueT> &grid, bool detailed, bool verbose)
{
    const std::string str = GridValidator<ValueT>::check( grid, detailed );
    if (verbose && !str.empty()) std::cerr << "Validation failed: " << str << std::endl;
    return str.empty();
}

} // namespace nanovdb

#endif // NANOVDB_GRIDVALIDATOR_H_HAS_BEEN_INCLUDED
