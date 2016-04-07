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
/// @file AttributeGroup.cc
///
/// @authors Dan Bailey


#include <openvdb_points/tools/AttributeGroup.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {


////////////////////////////////////////

// GroupAttributeArray implementation


GroupAttributeArray::GroupAttributeArray(size_t n, const ValueType& uniformValue)
    : TypedAttributeArray<GroupType, NullAttributeCodec<GroupType> >(n, uniformValue)
{
    this->setGroup(true);
}


GroupAttributeArray::GroupAttributeArray(const GroupAttributeArray& array, const bool decompress)
        : TypedAttributeArray<GroupType, NullAttributeCodec<GroupType> >(array, decompress)
{
    this->setGroup(true);
}


void GroupAttributeArray::setGroup(bool state)
{
    if (state) mFlags |= Int16(GROUP);
    else mFlags &= ~Int16(GROUP);
}


////////////////////////////////////////

// GroupHandle implementation


GroupHandle::GroupHandle(const GroupAttributeArray& array, const GroupType& offset)
        : mArray(array)
        , mBitMask(GroupType(1) << offset)
{
    assert(mArray.isGroup());
}


GroupHandle::GroupHandle(const GroupAttributeArray& array, const GroupType& bitMask,
            BitMask)
    : mArray(array)
    , mBitMask(bitMask)
{
    assert(mArray.isGroup());
}


bool GroupHandle::get(Index n) const
{
    return (mArray.get(n) & mBitMask) == mBitMask;
}


////////////////////////////////////////

// GroupWriteHandle implementation


GroupWriteHandle::GroupWriteHandle(GroupAttributeArray& array, const GroupType& offset)
    : GroupHandle(array, offset)
{
}


void GroupWriteHandle::set(Index n, bool on)
{
    const GroupType& value = mArray.get(n);

    GroupAttributeArray& array(const_cast<GroupAttributeArray&>(mArray));

    if (on)     array.set(n, value | mBitMask);
    else        array.set(n, value & ~mBitMask);
}


bool GroupWriteHandle::collapse(bool on)
{
    GroupAttributeArray& array(const_cast<GroupAttributeArray&>(mArray));

    array.compact();

    if (this->isUniform()) {
        if (on)     array.collapse(array.get(0) | mBitMask);
        else        array.collapse(array.get(0) & ~mBitMask);
        return true;
    }

    for (size_t i = 0; i < array.size(); i++) {
        if (on)     array.set(i, array.get(i) | mBitMask);
        else        array.set(i, array.get(i) & ~mBitMask);
    }

    return false;
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


// Copyright (c) 2015-2016 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
