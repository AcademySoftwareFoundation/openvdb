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
/// @file AttributeGroup.h
///
/// @authors Dan Bailey
///
/// @brief  Attribute Group access and filtering for iteration.
///


#ifndef OPENVDB_TOOLS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED

#include <openvdb/openvdb.h>
#include <openvdb_points/tools/AttributeArray.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////

class GroupHandle
{
public:
    // Dummy class that distinguishes an offset from a bitmask on construction
    struct BitMask { };

    typedef std::pair<size_t, uint8_t> GroupIndex;

    GroupHandle(const GroupAttributeArray& array,
                const GroupType& offset)
        : mArray(array)
        , mBitMask(GroupType(1) << offset) { assert(mArray.isGroup()); }

    GroupHandle(const GroupAttributeArray& array,
                const GroupType& bitMask,
                BitMask)
        : mArray(array)
        , mBitMask(bitMask) { assert(mArray.isGroup()); }

    size_t size() const { return mArray.size(); }

    bool get(Index n) const {
        return (mArray.get(n) & mBitMask) == mBitMask;
    }

private:
    const GroupAttributeArray& mArray;
    const GroupType mBitMask;
}; // class GroupHandle


class GroupWriteHandle
{
public:
    typedef std::pair<size_t, uint8_t> GroupIndex;

    GroupWriteHandle(   GroupAttributeArray& array,
                        const GroupType& offset)
        : mArray(array)
        , mBitMask(GroupType(1) << offset) { assert(mArray.isGroup()); }

    size_t size() const { return mArray.size(); }

    bool get(Index n) const {
        return (mArray.get(n) & mBitMask) == mBitMask;
    }

    void set(Index n, bool on) {
        const GroupType& value = mArray.get(n);

        if (on)     mArray.set(n, value | mBitMask);
        else        mArray.set(n, value & ~mBitMask);
    }

private:
    GroupAttributeArray& mArray;
    const GroupType mBitMask;
}; // class GroupWriteHandle


/// Index filtering on group membership
class GroupFilter
{
public:
    GroupFilter(const GroupHandle& handle)
        : mHandle(handle) { }

    bool valid(const Index32 offset) const {
        return mHandle.get(offset);
    }

private:
    const GroupHandle mHandle;
}; // class GroupFilter


class GroupFilterFromLeaf
{
public:
    typedef GroupFilter Filter;

    GroupFilterFromLeaf(const Name& name)
        : mName(name) { }

    template <typename LeafT>
    Filter fromLeaf(const LeafT& leaf) const {
        return Filter(leaf.groupHandle(mName));
    }

private:
    const Name mName;
}; // GroupFilterFromLeaf


////////////////////////////////////////


} // namespace tools

} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_ATTRIBUTE_GROUP_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
