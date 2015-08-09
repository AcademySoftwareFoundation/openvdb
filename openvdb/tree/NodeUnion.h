///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2012-2015 DreamWorks Animation LLC
//
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
//
// Redistributions of source code must retain the above copyright
// and license notice and the following restrictions and disclaimer.
//
// *     Neither the name of DreamWorks Animation nor the names of
// its contributors may be used to endorse or promote products derived
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
/// @file NodeUnion.h
///
/// @author Peter Cucka
///
/// NodeUnion is a templated helper class that controls access to either
/// the child node pointer or the value for a particular element of a root
/// or internal node.  For space efficiency, the child pointer and the value
/// are unioned, since the two are never in use simultaneously.
/// Template specializations of NodeUnion allow for values of either POD
/// (int, float, pointer, etc.) or class (std::string, math::Vec, etc.) types.
/// (The latter cannot be stored directly in a union.)

#ifndef OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED
#define OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED

#include <boost/type_traits/is_class.hpp>
#include <openvdb/version.h>

namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tree {

// Internal implementation of a union of a child node pointer and a value
template<bool ValueIsClass, class ValueT, class ChildT> class NodeUnionImpl;


// Partial specialization for values of non-class types
// (int, float, pointer, etc.) that stores elements by value
template<typename ValueT, typename ChildT>
class NodeUnionImpl</*ValueIsClass=*/false, ValueT, ChildT>
{
private:
    union { ChildT* child; ValueT value; } mUnion;

public:
    NodeUnionImpl() { setChild(NULL); }

    ChildT* getChild() const { return mUnion.child; }
    const ValueT& getValue() const { return mUnion.value; }
    ValueT& getValue() { return mUnion.value; }
    void setChild(ChildT* child) { mUnion.child = child; }
    void setValue(const ValueT& val) { mUnion.value = val; }
};


// Partial specialization for values of class types (std::string,
// math::Vec, etc.) that stores elements by pointer
template<typename ValueT, typename ChildT>
class NodeUnionImpl</*ValueIsClass=*/true, ValueT, ChildT>
{
private:
    union { ChildT* child; ValueT* value; } mUnion;
    bool mHasChild;

public:
    NodeUnionImpl(): mHasChild(true) { setChild(NULL); }
    NodeUnionImpl(const NodeUnionImpl& other)
    {
        if (other.mHasChild) setChild(other.getChild());
        else setValue(other.getValue());
    }
    NodeUnionImpl& operator=(const NodeUnionImpl& other)
    {
        if (other.mHasChild) setChild(other.getChild());
        else setValue(other.getValue());
    }
    ~NodeUnionImpl() { setChild(NULL); }

    ChildT* getChild() const
        { return mHasChild ? mUnion.child : NULL; }
    void setChild(ChildT* child)
    {
        if (!mHasChild) delete mUnion.value;
        mUnion.child = child;
        mHasChild = true;
    }

    const ValueT& getValue() const { return *mUnion.value; }
    ValueT& getValue() { return *mUnion.value; }
    void setValue(const ValueT& val)
    {
        /// @todo To minimize storage across nodes, intern and reuse
        /// common values, using, e.g., boost::flyweight.
        if (!mHasChild) delete mUnion.value;
        mUnion.value = new ValueT(val);
        mHasChild = false;
    }
};


template<typename ValueT, typename ChildT>
struct NodeUnion: public NodeUnionImpl<
    boost::is_class<ValueT>::value, ValueT, ChildT>
{
    NodeUnion() {}
};

} // namespace tree
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb

#endif // OPENVDB_TREE_NODEUNION_HAS_BEEN_INCLUDED

// Copyright (c) 2012-2015 DreamWorks Animation LLC
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
