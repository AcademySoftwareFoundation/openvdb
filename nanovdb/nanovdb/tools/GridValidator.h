// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*!
    \file nanovdb/tools/GridValidator.h

    \author Ken Museth

    \date August 30, 2020

    \brief Checks the validity of an existing NanoVDB grid.

    \note before v32.6.0: checksum[0] = Grid+Tree+Root, checksum[1] = nodes
          after  v32.6.0: checksum[0] = Grid+Tree,      checksum[1] = nodes + blind data in 4K blocks

    When serialized:
    [Grid,Tree][Root][ROOT TILES...][Node<5>...][Node<4>...][Leaf<3>...][BlindMeta...][BlindData...]
*/

#ifndef NANOVDB_TOOLS_GRID_VALIDATOR_H_HAS_BEEN_INCLUDED
#define NANOVDB_TOOLS_GRID_VALIDATOR_H_HAS_BEEN_INCLUDED

#include <iostream> // for std::cerr

#include <nanovdb/NanoVDB.h>
#include <nanovdb/tools/GridChecksum.h>

namespace nanovdb {

namespace tools {

/// @brief Performs several validation tests on a grid pointer.
/// @tparam ValueT Build type of the input grid
/// @param grid const point to the grid that needs validation
/// @param mode Mode of the validation check (defined in GridChecksum.h)
/// @param verbose If true information about the first failed test is printed to std::cerr
/// @return Return true if the specified grid passes several validation tests.
template <typename ValueT>
bool isValid(const NanoGrid<ValueT> *grid, CheckMode mode, bool verbose = false);

/// @brief Return true if the specified grid passes several validation tests.
/// @tparam ValueT Build type of the input grid
/// @param grid Grid to validate
/// @param detailed If true the validation test is detailed and relatively slow.
/// @param verbose If true information about the first failed test is printed to std::cerr
/// @note This method has been deprecated by the one defined above
template <typename ValueT>
[[deprecated("Use isValue(const NanoGrid<ValueT>*, CheckMode, bool) instead.")]]
bool isValid(const NanoGrid<ValueT> &grid, bool detailed = true, bool verbose = false)
{
    return isValid(&grid, detailed ? CheckMode::Full : CheckMode::Half, verbose);
}

//================================================================================================

/// @brief validate grid
template<typename ValueT>
__hostdev__ char* checkGrid(const NanoGrid<ValueT> *grid, char *error, CheckMode mode = CheckMode::Full)
{
    *error = '\0';// reset error string
    char str[32];// temporary buffer for toStr

    // check Grid
    if (grid == nullptr) {
        return util::sprint(error, "Invalid pointer: Grid is NULL");
    } else if (!isAligned(grid)) {
        return util::sprint(error, "Invalid pointer: Grid is misaligned");
    } else if (grid->mMagic != NANOVDB_MAGIC_NUMB && grid->mMagic != NANOVDB_MAGIC_GRID) {
        return util::sprint(error, "Invalid magic number: ", toStr(str, toMagic(grid->mMagic)));
    } else if (!grid->mVersion.isCompatible()) {
        return util::sprint(error, "Incompatible version number: ", toStr(str, grid->mVersion));
    } else if (grid->mGridCount == 0) {
        return util::sprint(error, "Zero grid count");
    } else if (grid->mGridIndex >= grid->mGridCount) {
        return util::sprint(error, "grid index(", int(grid->mGridIndex), ") >= grid count(", int(grid->mGridCount), ")");
    } else if (grid->mGridClass >= GridClass::End) {
        return util::sprint(error, "Invalid GridClass(", toStr(str, grid->mGridClass), ")");
     } else if (grid->mGridType >= GridType::End) {
        return util::sprint(error, "Invalid GridType(", toStr(str, grid->mGridType), ")");
     } else if (grid->mGridType != toGridType<ValueT>()) {
        return util::sprint(error, "Invalid combination of BuildType(", toStr(str, toGridType<ValueT>()), ") and GridType(", toStr(str+16, grid->mGridType), ")");
    } else if (!isValid(grid->mGridType, grid->mGridClass)) {
        return util::sprint(error, "Invalid combination of GridType(", toStr(str, grid->mGridType), ") and GridClass(", toStr(str+16,grid->mGridClass), ")");
    }

    // check Tree
    auto &tree = grid->tree();
    if (auto *p = tree.getRoot()) {
        if (!isAligned(p)) return util::strcpy(error, "Invalid pointer: Root is misaligned");
    } else {
        return util::strcpy(error, "Invalid pointer: Root is NULL");
    }

    // check Root
    auto &root = tree.root();
    auto *rootData = root.data();
    if (rootData == nullptr) {
        return util::strcpy(error, "Invalid pointer: Root is NULL");
    } else if (!isAligned((const void*)rootData)) {
        return util::strcpy(error, "Invalid pointer: Root is misaligned");
    } else if ( (const uint8_t*)(rootData) < (const uint8_t*)(&tree+1)) {
       return util::strcpy(error, "Invalid root pointer (should be located after the Grid and Tree)");
    } else if ( (const void*)(rootData) > util::PtrAdd(rootData, root.memUsage())) {
       return util::strcpy(error, "Invalid root pointer (appears to be located after the end of the buffer)");
    } else {// check root tiles
        const void *bounds[2] = {rootData + 1, util::PtrAdd(rootData, root.memUsage())};
        for (uint32_t i = 0; i<rootData->mTableSize; ++i) {
            const void *tile = rootData->tile(i);
            if ( tile < bounds[0] ) {
                return util::strcpy(error, "Invalid root tile pointer (below lower bound");
            } else if (tile >= bounds[1]) {
                return util::strcpy(error, "Invalid root tile pointer (above higher bound");
            }
        }
    }
    if (mode == CheckMode::Half) return error;

    // check nodes
    const bool test = grid->isBreadthFirst();
    auto *n0 = tree.template getFirstNode<0>();
    auto *n1 = tree.template getFirstNode<1>();
    auto *n2 = tree.template getFirstNode<2>();
    const void *bounds[3][2] = {{n0, util::PtrAdd(n0, grid->gridSize())}, {n1, n0}, {n2, n1}};

    auto check = [&](const void *ptr, int level) -> bool {
        if (ptr==nullptr) {
            util::strcpy(error, "Invalid node pointer: node is NULL");
        } else if (!isAligned(ptr)) {
            util::strcpy(error, "Invalid node pointer: node is misaligned");
        } else if (test && level == 0 && (const void*)(n0++) != ptr) {
            util::strcpy(error, "Leaf node is not stored breadth-first");
        } else if (test && level == 1 && (const void*)(n1++) != ptr) {
            util::strcpy(error, "Lower node is not stored breadth-first");
        } else if (test && level == 2 && (const void*)(n2++) != ptr) {
            util::strcpy(error, "Upper node is not stored breadth-first");
        } else if ( ptr < bounds[level][0] ) {
            util::strcpy(error, "Invalid node pointer: below lower bound");
        } else if ( ptr >= bounds[level][1] ) {
            util::strcpy(error, "Invalid node pointer: above higher bound");
        }
        return !util::empty(error);
    };

    for (auto it2 = root.cbeginChild(); it2; ++it2) {
        if (check(&*it2, 2)) return error;
        for (auto it1 = it2->cbeginChild(); it1; ++it1) {
            if (check(&*it1, 1)) return error;
            for (auto it0 = it1->cbeginChild(); it0; ++it0) if (check(&*it0, 0)) return error;
        }// loop over child nodes of the upper internal node
    }// loop over child nodes of the root node

    return error;
} // checkGrid

//================================================================================================

template <typename ValueT>
bool isValid(const NanoGrid<ValueT> *grid, CheckMode mode, bool verbose)
{
    std::unique_ptr<char[]> strUP(new char[100]);
    char *str = strUP.get();

    tools::checkGrid(grid, str, mode);

    if (util::empty(str) && !validateChecksum(grid, mode)) util::strcpy(str, "Mis-matching checksum");
    if (verbose && !util::empty(str)) std::cerr << "Validation failed: " << str << std::endl;

    return util::empty(str);
}// isValid

//================================================================================================

struct IsNanoGridValid {
    template <typename BuildT>
    static bool   known(const GridData *gridData, CheckMode mode, bool verbose)
    {
        return tools::isValid((const NanoGrid<BuildT>*)gridData, mode, verbose);
    }
    static bool unknown(const GridData *gridData, CheckMode, bool verbose)
    {
        if (verbose) {
            char str[16];
            std::cerr << "Unsupported GridType: \"" << toStr(str,  gridData->mGridType) << "\"\n" << std::endl;
        }
        return false;
    }
};// IsNanoGridValid

/// @brief Validate a specific grid in a GridHandle
/// @tparam GridHandleT Type of GridHandle
/// @param handle GridHandle containing host grids
/// @param gridID linear index of the grid to be validated
/// @param mode node of validation tests
/// @param verbose if true information is printed if the grid fails a validation test
/// @return true if grid @c gridID passes all the validation tests
template <typename GridHandleT>
bool validateGrid(const GridHandleT &handle, uint32_t gridID, CheckMode mode, bool verbose)
{
    if (mode == CheckMode::Disable) {
        return true;
    } else if (gridID >= handle.gridCount()) {
        if (verbose) std::cerr << "grid index " << gridID << " exceeds available grid count " << handle.gridCount() << std::endl;
        return false;
    }
    return callNanoGrid<IsNanoGridValid>(handle.gridData(gridID), mode, verbose);
}// validateGrid

//================================================================================================

/// @brief Validate all the grids in a GridHandle
/// @tparam GridHandleT Type of GridHandle
/// @param handle GridHandle containing host grids (0,1...,N)
/// @param mode node of validation tests
/// @param verbose if true information is printed if a grid fails a validation test
/// @return true if all grids pass alle the validation tests
template <typename GridHandleT>
bool validateGrids(const GridHandleT &handle, CheckMode mode, bool verbose)
{
    if (mode == CheckMode::Disable) return true;
    for (uint32_t gridID=0; gridID<handle.gridCount(); ++gridID) {
        if (!validateGrid(handle, gridID, mode, verbose)) return false;
    }
    return true;
}// validateGrids

}// namespace tools

template<typename ValueT>
[[deprecated("Use nanovdb:tools::checkGrid instead.")]]
__hostdev__ char* checkGrid(const NanoGrid<ValueT> *grid, char *error, CheckMode mode = CheckMode::Full)
{
    return tools::checkGrid<ValueT>(grid, error, mode);
}

template <typename ValueT>
[[deprecated("Use nanovdb:tools::isValid instead.")]]
bool isValid(const NanoGrid<ValueT> *grid, CheckMode mode, bool verbose = false)
{
    return tools::isValid<ValueT>(grid, mode, verbose);
}

}// namespace nanovdb

#endif // NANOVDB_TOOLS_GRID_VALIDATOR_H_HAS_BEEN_INCLUDED
