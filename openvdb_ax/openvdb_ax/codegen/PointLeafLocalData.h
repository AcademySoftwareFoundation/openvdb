// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/// @file codegen/PointLeafLocalData.h
///
/// @authors Nick Avramoussis
///
/// @brief  Thread/Leaf local data used during execution over OpenVDB Points
///

#ifndef OPENVDB_AX_COMPILER_LEAF_LOCAL_DATA_HAS_BEEN_INCLUDED
#define OPENVDB_AX_COMPILER_LEAF_LOCAL_DATA_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb/version.h>
#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointAttribute.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointGroup.h>
#include <openvdb/util/Assert.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {

namespace ax {
namespace codegen {

namespace codegen_internal {


/// @brief  Various functions can request the use and initialization of point data from within
///         the kernel that does not use the standard attribute handle methods. This data can
///         then be accessed after execution to perform post-processes such as adding new groups,
///         adding new string attributes or updating positions.
///
/// @note  Due to the way string handles work, string write attribute handles cannot
///        be constructed in parallel, nor can read handles retrieve values in parallel
///        if there is a chance the shared metadata is being written to (with set()).
///        As the compiler allows for any arbitrary string setting/getting, leaf local
///        maps are used for temporary storage per point. The maps use the string array
///        pointers as a key for later synchronization.
///
struct PointLeafLocalData
{
    using UniquePtr = std::unique_ptr<PointLeafLocalData>;
    using GroupArrayT = openvdb::points::GroupAttributeArray;
    using GroupHandleT = openvdb::points::GroupWriteHandle;

    using PointStringMap = std::map<uint64_t, std::string>;
    using StringArrayMap = std::map<points::AttributeArray*, PointStringMap>;

    using LeafNode = openvdb::points::PointDataTree::LeafNodeType;

    /// @brief  Construct a new data object to keep track of various data objects
    ///         created per leaf by the point compute generator.
    ///
    /// @param  count  The number of points within the current leaf, used to initialize
    ///                the size of new arrays
    ///
    PointLeafLocalData(const size_t count)
        : mPointCount(count)
        , mArrays()
        , mOffset(0)
        , mHandles()
        , mStringMap() {}

    ////////////////////////////////////////////////////////////////////////

    /// Group methods

    /// @brief  Return a group write handle to a specific group name, creating the
    ///         group array if it doesn't exist. This includes either registering a
    ///         new offset or allocating an entire array. The returned handle is
    ///         guaranteed to be valid.
    ///
    /// @param  name  The group name
    ///
    inline GroupHandleT* getOrInsert(const std::string& name)
    {
        GroupHandleT* ptr = get(name);
        if (ptr) return ptr;

        static const size_t maxGroupsInArray =
#if (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER > 7 ||  \
    (OPENVDB_LIBRARY_MAJOR_VERSION_NUMBER >= 7 && \
     OPENVDB_LIBRARY_MINOR_VERSION_NUMBER >= 1))
            points::AttributeSet::Descriptor::groupBits();
#else
            // old removed method
            points::point_group_internal::GroupInfo::groupBits();
#endif

        if (mArrays.empty() || mOffset == maxGroupsInArray) {
            OPENVDB_ASSERT(mPointCount < static_cast<size_t>(std::numeric_limits<openvdb::Index>::max()));
            mArrays.emplace_back(new GroupArrayT(static_cast<openvdb::Index>(mPointCount)));
            mOffset = 0;
        }

        GroupArrayT* array = mArrays.back().get();
        OPENVDB_ASSERT(array);

        std::unique_ptr<GroupHandleT>& handle = mHandles[name];
        handle.reset(new GroupHandleT(*array, mOffset++));
        return handle.get();
    }

    /// @brief  Return a group write handle to a specific group name if it exists.
    ///         Returns a nullptr if no group exists of the given name
    ///
    /// @param  name  The group name
    ///
    inline GroupHandleT* get(const std::string& name) const
    {
        const auto iter = mHandles.find(name);
        if (iter == mHandles.end()) return nullptr;
        return iter->second.get();
    }

    /// @brief  Return true if a valid group handle exists
    ///
    /// @param  name  The group name
    ///
    inline bool hasGroup(const std::string& name) const {
        return mHandles.find(name) != mHandles.end();
    }

    /// @brief  Populate a set with all the groups which have been inserted into
    ///         this object. Used to compute a final set of all new groups which
    ///         have been created across all leaf nodes
    ///
    /// @param  groups  The set to populate
    ///
    inline void getGroups(std::set<std::string>& groups) const {
        for (const auto& iter : mHandles) {
            groups.insert(iter.first);
        }
    }

    /// @brief  Compact all arrays stored on this object. This does not invalidate
    ///         any active write handles.
    ///
    inline void compact() {
        for (auto& array : mArrays) array->compact();
    }


    ////////////////////////////////////////////////////////////////////////

    /// String methods

    /// @brief  Get any new string data associated with a particular point on a
    ///         particular string attribute array. Returns true if data was set,
    ///         false if no data was found.
    ///
    /// @param  array  The array pointer to use as a key lookup
    /// @param  idx    The point index
    /// @param  data   The string to set if data is stored
    ///
    inline bool
    getNewStringData(const points::AttributeArray* array, const uint64_t idx, std::string& data) const {
        const auto arrayMapIter = mStringMap.find(const_cast<points::AttributeArray*>(array));
        if (arrayMapIter == mStringMap.end()) return false;
        const auto iter = arrayMapIter->second.find(idx);
        if (iter == arrayMapIter->second.end()) return false;
        data = iter->second;
        return true;
    }

    /// @brief  Set new string data associated with a particular point on a
    ///         particular string attribute array.
    ///
    /// @param  array  The array pointer to use as a key lookup
    /// @param  idx    The point index
    /// @param  data   The string to set
    ///
    inline void
    setNewStringData(points::AttributeArray* array, const uint64_t idx, const std::string& data) {
        mStringMap[array][idx] = data;
    }

    /// @brief  Remove any new string data associated with a particular point on a
    ///         particular string attribute array. Does nothing if no data exists
    ///
    /// @param  array  The array pointer to use as a key lookup
    /// @param  idx    The point index
    ///
    inline void
    removeNewStringData(points::AttributeArray* array, const uint64_t idx) {
        const auto arrayMapIter = mStringMap.find(array);
        if (arrayMapIter == mStringMap.end()) return;
        arrayMapIter->second.erase(idx);
        if (arrayMapIter->second.empty()) mStringMap.erase(arrayMapIter);
    }

    /// @brief  Insert all new point strings stored across all collected string
    ///         attribute arrays into a StringMetaInserter. Returns false if the
    ///         inserter was not accessed and true if it was potentially modified.
    ///
    /// @param  inserter  The string meta inserter to update
    ///
    inline bool
    insertNewStrings(points::StringMetaInserter& inserter) const {
        for (const auto& arrayIter : mStringMap) {
            for (const auto& iter : arrayIter.second) {
                inserter.insert(iter.second);
            }
        }
        return !mStringMap.empty();
    }

    /// @brief  Returns a const reference to the string array map
    ///
    inline const StringArrayMap& getStringArrayMap() const {
        return mStringMap;
    }

private:

    const size_t mPointCount;
    std::vector<std::unique_ptr<GroupArrayT>> mArrays;
    points::GroupType mOffset;
    std::map<std::string, std::unique_ptr<GroupHandleT>> mHandles;
    StringArrayMap mStringMap;
};

} // codegen_internal

} // namespace compiler
} // namespace ax
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_AX_COMPILER_LEAF_LOCAL_DATA_HAS_BEEN_INCLUDED

