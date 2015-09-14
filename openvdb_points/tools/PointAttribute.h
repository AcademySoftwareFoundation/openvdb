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
/// @file PointAttribute.h
///
/// @brief  Add attributes to a VDB Point Grid.
///


#ifndef OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED
#define OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED

#include <openvdb/version.h>
#include <openvdb/Types.h>
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/math/Transform.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/tree/LeafNode.h>
#include <openvdb/tree/LeafNodeBool.h>
#include <openvdb/util/Name.h>

#include <openvdb/tools/PointIndexGrid.h>


namespace openvdb {
OPENVDB_USE_VERSION_NAMESPACE
namespace OPENVDB_VERSION_NAME {
namespace tools {

/// @brief Appends a new attribute to the VDB tree.
///
/// @param tree          the PointDataTree to be appended to.
/// @param newAttribute  name and type for the new attribute.

template <typename PointDataTree>
inline void appendAttribute(PointDataTree& tree,
                            const AttributeSet::Util::NameAndType& newAttribute);


////////////////////////////////////////


namespace point_attribute_internal {

template<typename PointDataTreeType>
struct AppendAttributeOp {

    typedef typename tree::LeafManager<PointDataTreeType>       LeafManagerT;
    typedef typename LeafManagerT::LeafRange                    LeafRangeT;
    typedef typename PointDataTreeType::LeafNodeType            PointDataLeafNode;
    typedef AttributeSet::Descriptor::NameAndType               NameAndType;

    AppendAttributeOp(  PointDataTreeType& tree,
                        const NameAndType& newAttribute,
                        AttributeSet::DescriptorPtr& descriptor)
        : mTree(tree)
        , mNewAttribute(newAttribute)
        , mDescriptor(descriptor) { }

    void operator()(const typename LeafManagerT::LeafRange& range) const {

        for (typename LeafManagerT::LeafRange::Iterator leaf=range.begin(); leaf; ++leaf) {

            const AttributeSet::Descriptor& expected = leaf->attributeSet().descriptor();

            leaf->appendAttribute(mNewAttribute, expected, mDescriptor);
        }
    }

    //////////

    PointDataTreeType&              mTree;
    const NameAndType&              mNewAttribute;
    AttributeSet::DescriptorPtr&    mDescriptor;
};

} // namespace point_attribute_internal


////////////////////////////////////////


template <typename PointDataTree>
inline void appendAttribute(PointDataTree& tree,
                            const AttributeSet::Util::NameAndType& newAttribute)
{
    typedef AttributeSet::Util::NameAndTypeVec                    NameAndTypeVec;
    typedef AttributeSet::Descriptor                              Descriptor;

    typedef typename tree::LeafManager<PointDataTree>             LeafManagerT;
    typedef typename LeafManagerT::LeafRange                      LeafRangeT;

    using point_attribute_internal::AppendAttributeOp;

    typename PointDataTree::LeafCIter iter = tree.cbeginLeaf();

    if (!iter)  return;

    // create a new attribute descriptor
    NameAndTypeVec vec;
    vec.push_back(newAttribute);

    Descriptor::Ptr descriptor = iter->attributeSet().descriptor().duplicateAppend(vec);

    // insert attributes using the new descriptor

    AppendAttributeOp<PointDataTree> append(tree, newAttribute, descriptor);
    tbb::parallel_for(typename tree::template LeafManager<PointDataTree>(tree).leafRange(), append);
}


////////////////////////////////////////


} // namespace tools
} // namespace OPENVDB_VERSION_NAME
} // namespace openvdb


#endif // OPENVDB_TOOLS_POINT_ATTRIBUTE_HAS_BEEN_INCLUDED


// Copyright (c) 2015 Double Negative Visual Effects
// All rights reserved. This software is distributed under the
// Mozilla Public License 2.0 ( http://www.mozilla.org/MPL/2.0/ )
